#include "pt/BvhVariants.h"

#include <algorithm>
#include <immintrin.h>
#include <stdexcept>

namespace pt {

PtHit traceWideBvhAvx2(const WideBvh &bvh, const Material *materials, const uint *indices,
                       const float *vertices, const HostTextures &textures, float3 origin,
                       float3 direction, float tMax, uint mask, uint seed, float2 cone, uint anyHit) {
  PtHit hit{}; hit.t = tMax;
  PtRayPrep ray{}; float3 prepOrigin = origin, prepDirection = direction;
  bool prepare = true, done = false; uint stack[128]{};
  uint depth = bvh.nodes.empty() ? 0u : 1u, bottomBase = 0u, inBottom = 0u, instance = 0u;
  while (depth > 0u && !done) {
    if (inBottom != 0u && depth <= bottomBase) {
      inBottom = 0u; prepOrigin = origin; prepDirection = direction; prepare = true;
    }
    if (prepare) { ray = ptPrepareRay(prepOrigin, prepDirection); prepare = false; }
    const uint entry = stack[--depth];
    if ((entry & kBvhLeafTag) != 0u) {
      if (inBottom == 0u) {
        instance = entry & ~kBvhLeafTag;
        const TraceInstance &row = bvh.instances[instance];
        if ((row.mask & mask) != 0u) {
          prepOrigin = float3(dot(row.worldToObject0, float4(origin, 1.0f)),
                              dot(row.worldToObject1, float4(origin, 1.0f)),
                              dot(row.worldToObject2, float4(origin, 1.0f)));
          prepDirection = float3(dot(xyz(row.worldToObject0), direction),
                                 dot(xyz(row.worldToObject1), direction),
                                 dot(xyz(row.worldToObject2), direction));
          prepare = true; inBottom = 1u; bottomBase = depth; stack[depth++] = row.blasRoot;
        }
      } else {
        const uint first = entry & kBvhFirstMask;
        const uint count = (entry & ~kBvhLeafTag) >> kBvhCountShift;
        for (uint i = 0; i < count && !done; ++i) {
          const uint triangle = (first + i) * 3u; const float4 v0 = bvh.triangles[triangle];
          float distance = 0.0f; float2 barycentric(0.0f);
          if (ptIntersectTriangle(ray, xyz(v0), xyz(bvh.triangles[triangle + 1u]),
              xyz(bvh.triangles[triangle + 2u]), hit.t, distance, barycentric, hit.ambiguous)) {
            const uint primitive = as_type<uint>(v0.w);
            if ((bvh.instances[instance].flags & kInstanceBlended) != 0u) hit.ambiguous |= 1u;
            if (ptCandidateSolid(bvh.instances.data(), materials, indices, vertices, textures,
                instance, primitive, barycentric, (hit.ambiguous >> 1u) & 1u, seed, direction,
                ptConeWidthOrLevelZero(cone, distance))) {
              hit.ambiguous &= 1u; hit.t = distance; hit.barycentric = barycentric;
              hit.instance = instance; hit.primitive = primitive; hit.found = 1u;
              if (anyHit != 0u) done = true;
            }
          }
        }
      }
      continue;
    }

    const QuantizedWideNode &node = bvh.nodes[entry];
    const uint childCount = as_type<uint>(node.origin.w);
    alignas(32) uint px[8]{}, py[8]{}, pz[8]{};
    for (uint i = 0; i < childCount; ++i) {
      px[i] = node.children[i].x; py[i] = node.children[i].y; pz[i] = node.children[i].z;
    }
    const __m256i mask16 = _mm256_set1_epi32(0xFFFF);
    auto bounds = [&](const uint *packed, float base, float step, __m256 &low, __m256 &high) {
      const __m256i words = _mm256_load_si256(reinterpret_cast<const __m256i *>(packed));
      const __m256 ql = _mm256_cvtepi32_ps(_mm256_and_si256(words, mask16));
      const __m256 qh = _mm256_cvtepi32_ps(_mm256_srli_epi32(words, 16));
      const __m256 b = _mm256_set1_ps(base), s = _mm256_set1_ps(step);
      low = _mm256_add_ps(b, _mm256_mul_ps(s, ql));
      high = _mm256_add_ps(b, _mm256_mul_ps(s, qh));
    };
    __m256 lx, hx, ly, hy, lz, hz;
    bounds(px, node.origin.x, node.scale.x, lx, hx);
    bounds(py, node.origin.y, node.scale.y, ly, hy);
    bounds(pz, node.origin.z, node.scale.z, lz, hz);
    auto slab = [](__m256 low, __m256 high, float o, float inverse, __m256 &nearV, __m256 &farV) {
      const __m256 originV = _mm256_set1_ps(o), inverseV = _mm256_set1_ps(inverse);
      const __m256 a = _mm256_mul_ps(_mm256_sub_ps(low, originV), inverseV);
      const __m256 z = _mm256_mul_ps(_mm256_sub_ps(high, originV), inverseV);
      nearV = _mm256_min_ps(a, z); farV = _mm256_max_ps(a, z);
    };
    __m256 nx, fx, ny, fy, nz, fz;
    slab(lx, hx, ray.origin.x, ray.inverse.x, nx, fx);
    slab(ly, hy, ray.origin.y, ray.inverse.y, ny, fy);
    slab(lz, hz, ray.origin.z, ray.inverse.z, nz, fz);
    const __m256 nearV = _mm256_max_ps(_mm256_setzero_ps(), _mm256_max_ps(nx, _mm256_max_ps(ny, nz)));
    const __m256 farV = _mm256_mul_ps(_mm256_min_ps(_mm256_set1_ps(hit.t),
        _mm256_min_ps(fx, _mm256_min_ps(fy, fz))), _mm256_set1_ps(1.0000004f));
    uint hitMask = static_cast<uint>(_mm256_movemask_ps(_mm256_cmp_ps(nearV, farV, _CMP_LE_OQ)));
    hitMask &= childCount == 8u ? 0xFFu : ((1u << childCount) - 1u);
    alignas(32) float nearDistances[8];
    _mm256_store_ps(nearDistances, nearV);
    float distances[8]{}; uint entries[8]{}; uint found = 0u;
    for (uint child = 0; child < childCount; ++child)
      if ((hitMask & (1u << child)) != 0u) {
        distances[found] = nearDistances[child]; entries[found++] = node.children[child].w;
      }
    for (uint i = 1u; i < found; ++i) {
      const float d = distances[i]; const uint e = entries[i]; uint j = i;
      while (j > 0u && distances[j - 1u] < d) {
        distances[j] = distances[j - 1u]; entries[j] = entries[j - 1u]; --j;
      }
      distances[j] = d; entries[j] = e;
    }
    for (uint i = 0u; i < found; ++i) {
      if (depth >= 128u) throw std::runtime_error("AVX2 wide BVH traversal stack overflow");
      stack[depth++] = entries[i];
    }
  }
  hit.ambiguous &= 1u;  // drop the per-candidate facing scratch bit
  return hit;
}

} // namespace pt
