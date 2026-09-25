// The software BVH, laid out and traversed the way the hardware structure behaves: a
// top level over instances, one bottom level per instance in object space, masks and
// non-opaque candidates. Shared by the CPU backend and the software GPU backend. The node
// layouts are in bvh_layout.h.
#pragma once
#include "path.h"
#include "bvh_layout.h"


// The ray, prepared for boxes (a safe inverse) and for Woop, Benthin and Wald's
// watertight triangle test. The test permutes the axes so the dominant one is z and shears
// x and y along it; here that permutation and shear are three rows, so a vertex's sheared
// coordinates are dot products and nothing indexes a vector by a variable.
struct PtRayPrep {
  float3 origin;
  float3 inverse;
  float3 rowX;     // e_kx - (d_kx / d_kz) e_kz
  float3 rowY;     // e_ky - (d_ky / d_kz) e_kz
  float3 rowZ;     // e_kz
  float shearZ;    // 1 / d_kz
};

inline PtRayPrep ptPrepareRay(float3 origin, float3 direction) {
  PtRayPrep ray;
  ray.origin = origin;
  // A zero component would make an infinite inverse, and zero times infinity a NaN.
  float3 safe = direction;
  if (abs(safe.x) < 1e-20f) safe.x = 1e-20f;
  if (abs(safe.y) < 1e-20f) safe.y = 1e-20f;
  if (abs(safe.z) < 1e-20f) safe.z = 1e-20f;
  ray.inverse = float3(1.0f) / safe;
  const float3 magnitude = abs(direction);
  // Unit vectors for kx, ky, kz, the dominant axis last; swapping kx and ky when the
  // dominant component is negative keeps the triangles' winding.
  float3 ex = float3(0.0f, 1.0f, 0.0f);
  float3 ey = float3(0.0f, 0.0f, 1.0f);
  float3 ez = float3(1.0f, 0.0f, 0.0f);
  float3 d = float3(direction.y, direction.z, direction.x);
  if (magnitude.y >= magnitude.x && magnitude.y >= magnitude.z) {
    ex = float3(0.0f, 0.0f, 1.0f);
    ey = float3(1.0f, 0.0f, 0.0f);
    ez = float3(0.0f, 1.0f, 0.0f);
    d = float3(direction.z, direction.x, direction.y);
  } else if (magnitude.z >= magnitude.x) {
    ex = float3(1.0f, 0.0f, 0.0f);
    ey = float3(0.0f, 1.0f, 0.0f);
    ez = float3(0.0f, 0.0f, 1.0f);
    d = direction;
  }
  if (d.z < 0.0f) {
    const float3 swap = ex;
    ex = ey;
    ey = swap;
    d = float3(d.y, d.x, d.z);
  }
  ray.rowX = ex - ez * (d.x / d.z);
  ray.rowY = ey - ez * (d.y / d.z);
  ray.rowZ = ez;
  ray.shearZ = 1.0f / d.z;
  return ray;
}

// Entry distances into a node's two child boxes, kPtInfinity for a miss or a missing child.
// The far distance is widened by a few ulps (Ize 2013) so a ray grazing a face is not lost.
inline float2 ptChildEntries(PtRayPrep ray, float4 n0, float4 n1, float4 n2, float4 n3, float tMax) {
  const float3 l0 = (xyz(n0) - ray.origin) * ray.inverse;
  const float3 l1 = (xyz(n1) - ray.origin) * ray.inverse;
  const float3 r0 = (xyz(n2) - ray.origin) * ray.inverse;
  const float3 r1 = (xyz(n3) - ray.origin) * ray.inverse;
  const float2 entry = float2(max(max(min(l0.x, l1.x), min(l0.y, l1.y)), max(min(l0.z, l1.z), 0.0f)),
                              max(max(min(r0.x, r1.x), min(r0.y, r1.y)), max(min(r0.z, r1.z), 0.0f)));
  const float2 exit = float2(min(min(max(l0.x, l1.x), max(l0.y, l1.y)), min(max(l0.z, l1.z), tMax)),
                             min(min(max(r0.x, r1.x), max(r0.y, r1.y)), min(max(r0.z, r1.z), tMax))) * 1.0000004f;
  float2 result = float2(kPtInfinity);
  if (entry.x <= exit.x && as_type<uint>(n0.w) != kBvhEmpty) result.x = entry.x;
  if (entry.y <= exit.y && as_type<uint>(n2.w) != kBvhEmpty) result.y = entry.y;
  return result;
}

inline bool ptIntersectTriangle(PtRayPrep ray, float3 v0, float3 v1, float3 v2, float tMax,
                                thread float &t, thread float2 &barycentric, thread uint &candidateFlags) {
  const float3 sheared = float3(dot(ray.rowX, v0 - ray.origin), dot(ray.rowX, v1 - ray.origin),
                                dot(ray.rowX, v2 - ray.origin));
  const float3 shearedY = float3(dot(ray.rowY, v0 - ray.origin), dot(ray.rowY, v1 - ray.origin),
                                 dot(ray.rowY, v2 - ray.origin));
  // Scaled barycentrics: the 2D edge functions of the sheared triangle.
  const float3 edges = float3(sheared.z * shearedY.y - shearedY.z * sheared.y,
                              sheared.x * shearedY.z - shearedY.x * sheared.z,
                              sheared.y * shearedY.x - shearedY.y * sheared.x);
  const float determinant = edges.x + edges.y + edges.z;
  if (!((min(min(edges.x, edges.y), edges.z) >= 0.0f || max(max(edges.x, edges.y), edges.z) <= 0.0f)) ||
      determinant == 0.0f) return false;
  const float3 heights = float3(dot(ray.rowZ, v0 - ray.origin), dot(ray.rowZ, v1 - ray.origin),
                                 dot(ray.rowZ, v2 - ray.origin));
  const float distance = dot(edges, heights) * ray.shearZ / determinant;
  if (!(distance > 0.0f) || !(distance < tMax)) return false;
  t = distance;
  barycentric = float2(edges.y, edges.z) / determinant;
  candidateFlags = (candidateFlags & 1u) | (determinant > 0.0f ? 2u : 0u);
  return true;
}

// Whether a candidate on a non-opaque instance stops the ray: masked materials test their
// alpha against the cutoff; blended ones pass through with probability 1 - alpha. The alpha
// texture is read at the ray cone's level of detail at the candidate: the world
// ray direction and the cone width there (ptConeWidthOrLevelZero; -1 reads level zero), so
// every intersector sees the same coverage. A filtered alpha averages the footprint, and a
// fixed cutoff on an average turns a partly opaque footprint wholly solid or clear, so the
// cutoff is dithered over the footprint's width in texels f (ptTextureFootprint, capped at 1):
// cutoff + (xi - 0.5) f. From one texel up, a binary mask with cutoff 0.5 stops the ray with
// probability its opaque fraction, at every distance; magnified edges stay within a
// footprint; level-zero mode (f = 0) tests the cutoff exactly.
inline bool ptCandidateSolid(PT_SCENE_PARAMS, PT_TEXTURE_PARAMS, uint instance, uint primitive,
                             float2 barycentric, uint frontFacing, uint seed, float3 direction, float coneWidth) {
  if ((traceInstances[instance].flags & kInstanceDoubleSided) == 0u && frontFacing == 0u) return false;
  if ((traceInstances[instance].flags & (kInstanceMasked | kInstanceBlended)) == 0u) return true;
  const uint v0 = (indices[traceInstances[instance].firstIndex + primitive * 3u] +
                   traceInstances[instance].vertexOffset) * kVertexFloats;
  const uint v1 = (indices[traceInstances[instance].firstIndex + primitive * 3u + 1u] +
                   traceInstances[instance].vertexOffset) * kVertexFloats;
  const uint v2 = (indices[traceInstances[instance].firstIndex + primitive * 3u + 2u] +
                   traceInstances[instance].vertexOffset) * kVertexFloats;
  const float3 weights = float3(1.0f - barycentric.x - barycentric.y, barycentric.x, barycentric.y);
  const uint uvOffset = (materials[traceInstances[instance].material].texture.x & 0xFFu) == 1u ? 12u : 10u;
  float lodBase = kPtLevelZero;  // level-zero mode skips the loads
  if (coneWidth >= 0.0f) {
    lodBase = ptConeLodBase(
        ptUvCross(float2(vertices[v0 + uvOffset], vertices[v0 + uvOffset + 1u]),
                  float2(vertices[v1 + uvOffset], vertices[v1 + uvOffset + 1u]),
                  float2(vertices[v2 + uvOffset], vertices[v2 + uvOffset + 1u])),
        ptWorldCross(traceInstances[instance].objectToWorld0, traceInstances[instance].objectToWorld1,
                     traceInstances[instance].objectToWorld2, float3(vertices[v0], vertices[v0 + 1u], vertices[v0 + 2u]),
                     float3(vertices[v1], vertices[v1 + 1u], vertices[v1 + 2u]),
                     float3(vertices[v2], vertices[v2 + 1u], vertices[v2 + 2u])),
        direction, coneWidth);
  }
#define PT_CANDIDATE_ALPHA                                                                       \
  (ptSampleTexture(PT_TEXTURE_ARGS, traceInstances[instance].slots & 0xFFu,                      \
      float2(dot(weights, float3(vertices[v0 + ((materials[traceInstances[instance].material].texture.x & 0xFFu) == 1u ? 12u : 10u)], \
                                  vertices[v1 + ((materials[traceInstances[instance].material].texture.x & 0xFFu) == 1u ? 12u : 10u)], \
                                  vertices[v2 + ((materials[traceInstances[instance].material].texture.x & 0xFFu) == 1u ? 12u : 10u)])), \
             dot(weights, float3(vertices[v0 + ((materials[traceInstances[instance].material].texture.x & 0xFFu) == 1u ? 13u : 11u)], \
                                  vertices[v1 + ((materials[traceInstances[instance].material].texture.x & 0xFFu) == 1u ? 13u : 11u)], \
                                  vertices[v2 + ((materials[traceInstances[instance].material].texture.x & 0xFFu) == 1u ? 13u : 11u)]))), \
      materials[traceInstances[instance].material].texture.y & 0xFFu, lodBase).w *              \
   materials[traceInstances[instance].material].baseColorFactor.w *                               \
   dot(weights, float3(vertices[v0 + 17u], vertices[v1 + 17u], vertices[v2 + 17u])))
  if ((traceInstances[instance].flags & kInstanceBlended) != 0u)
    return hashFloat(seed ^ pcgHash(instance * 0x9E3779B9u + primitive)) < PT_CANDIDATE_ALPHA;
  if ((traceInstances[instance].flags & kInstanceMasked) != 0u)
    return PT_CANDIDATE_ALPHA >= materials[traceInstances[instance].material].alpha.x +
               (lodBase > kPtLevelZero
                    ? (hashFloat(seed ^ pcgHash(instance * 0x9E3779B9u + primitive)) - 0.5f) *
                          ptTextureFootprintOf(PT_TEXTURE_ARGS, traceInstances[instance].slots & 0xFFu, lodBase)
                    : 0.0f);
#undef PT_CANDIDATE_ALPHA
  return true;
}

// Closest hit, or with anyHit the first solid one. One stack serves both levels: an
// instance's bottom level is traversed above the entries its top level left, and the ray
// returns to world space when the stack falls back to them.
// cone: the ray cone where the ray leaves (width, spread; ptCameraCone), for alpha tests.
inline PtHit ptTraceBvh(device const float4 *nodes, device const float4 *triangles, PT_SCENE_PARAMS,
                        PT_TEXTURE_PARAMS, float3 origin, float3 direction, float tMax, uint mask,
                        uint seed, float2 cone, uint anyHit) {
  PtHit hit;
  hit.ambiguous = 0u;
  hit.t = tMax;
  hit.barycentric = float2(0.0f);
  hit.instance = 0u;
  hit.primitive = 0u;
  hit.found = 0u;
  hit.nodeVisits = 0u;
  hit.triangleTests = 0u;

  PtRayPrep ray;
  ray.origin = origin;
  ray.inverse = float3(1.0f);
  ray.rowX = float3(1.0f, 0.0f, 0.0f);
  ray.rowY = float3(0.0f, 1.0f, 0.0f);
  ray.rowZ = float3(0.0f, 0.0f, 1.0f);
  ray.shearZ = 1.0f;
  float3 prepareOrigin = origin;
  float3 prepareDirection = direction;
  bool prepare = true;
  // The entry being visited is held in a register; the stack keeps only the entries still to
  // visit (a node's far child), so the near child is never stored and reloaded. The visiting
  // order is the one of pushing both children and popping the near one.
  uint stack[PT_BVH_STACK] = {};
  uint depth = 0u;
  uint entry = 0u;  // the top level's root
  bool visiting = true;
  uint bottomBase = 0u;
  uint inBottom = 0u;
  uint instance = 0u;
  bool done = false;

  while (visiting && !done) {
    if (prepare) {
      ray = ptPrepareRay(prepareOrigin, prepareDirection);
      prepare = false;
    }
    // Set by an entry that leaves nothing to visit next in its place: take the next from the stack.
    bool pop = true;

    if ((entry & kBvhLeafTag) != 0u) {
      if (inBottom == 0u) {
        // An instance: move the ray into its object space and descend into its bottom level.
        instance = entry & ~kBvhLeafTag;
        if ((traceInstances[instance].mask & mask) != 0u) {
          prepareOrigin = float3(dot(traceInstances[instance].worldToObject0, float4(origin, 1.0f)),
                                 dot(traceInstances[instance].worldToObject1, float4(origin, 1.0f)),
                                 dot(traceInstances[instance].worldToObject2, float4(origin, 1.0f)));
          prepareDirection = float3(dot(xyz(traceInstances[instance].worldToObject0), direction),
                                    dot(xyz(traceInstances[instance].worldToObject1), direction),
                                    dot(xyz(traceInstances[instance].worldToObject2), direction));
          prepare = true;
          inBottom = 1u;
          bottomBase = depth;
          entry = traceInstances[instance].blasRoot;
          pop = false;
        }
      } else {
        // A leaf of triangles. The object-space ray is an affine image of the world ray, so
        // its distances are world distances and compare with hit.t directly.
        const uint first = entry & kBvhFirstMask;
        const uint count = (entry & ~kBvhLeafTag) >> kBvhCountShift;
        for (uint i = 0u; i < count && !done; ++i) {
          hit.triangleTests = hit.triangleTests + 1u;
          const uint triangle = (first + i) * 3u;
          const float4 v0 = triangles[triangle];
          float distance = 0.0f;
          float2 barycentric = float2(0.0f);
          if (ptIntersectTriangle(ray, xyz(v0), xyz(triangles[triangle + 1u]), xyz(triangles[triangle + 2u]), hit.t,
                                  distance, barycentric, hit.ambiguous)) {
            const uint local = as_type<uint>(v0.w);
            if ((traceInstances[instance].flags & kInstanceBlended) != 0u) hit.ambiguous = hit.ambiguous | 1u;
            if (ptCandidateSolid(PT_SCENE_ARGS, PT_TEXTURE_ARGS, instance, local, barycentric,
                                 (hit.ambiguous >> 1u) & 1u, seed, direction, ptConeWidthOrLevelZero(cone, distance))) {
              hit.ambiguous = hit.ambiguous & 1u;
              hit.t = distance;
              hit.barycentric = barycentric;
              hit.instance = instance;
              hit.primitive = local;
              hit.found = 1u;
              if (anyHit != 0u) done = true;
            }
          }
        }
      }
    } else {
      // An interior node: test both children, push the far one first so the near one pops first.
      hit.nodeVisits = hit.nodeVisits + 1u;
      const uint base = entry * 4u;
      const float4 n0 = nodes[base];
      const float4 n1 = nodes[base + 1u];
      const float4 n2 = nodes[base + 2u];
      const float4 n3 = nodes[base + 3u];
      const float2 entries = ptChildEntries(ray, n0, n1, n2, n3, hit.t);
      // A child's stack entry: its node index, or a tagged leaf (the instance at the top
      // level; the count and first triangle at the bottom).
      uint2 push = uint2(as_type<uint>(n0.w), as_type<uint>(n2.w));
      const uint2 counts = uint2(as_type<uint>(n1.w), as_type<uint>(n3.w));
      if (counts.x > 0u) push.x = kBvhLeafTag | (counts.x << kBvhCountShift) * inBottom | push.x;
      if (counts.y > 0u) push.y = kBvhLeafTag | (counts.y << kBvhCountShift) * inBottom | push.y;
      if (entries.y < entries.x) push = uint2(push.y, push.x);
      // CPU-built structures are rejected when their combined TLAS/BLAS depth can
      // exceed this stack. GPU builders must enforce the same layout contract before
      // publishing a root. The checks remain as defence against malformed buffers.
      if (max(entries.x, entries.y) < hit.t && depth < uint(PT_BVH_STACK)) {
        stack[depth] = push.y;
        depth = depth + 1u;
      }
      if (min(entries.x, entries.y) < hit.t && depth < uint(PT_BVH_STACK)) {
        entry = push.x;
        pop = false;
      }
    }
    if (pop) {
      // The next entry from the stack; back in world space when it is the top level's.
      if (depth == 0u) {
        visiting = false;
      } else {
        if (inBottom != 0u && depth <= bottomBase) {
          inBottom = 0u;
          prepareOrigin = origin;
          prepareDirection = direction;
          prepare = true;
        }
        depth = depth - 1u;
        entry = stack[depth];
      }
    }
  }
  // Bit 1 is per-candidate facing scratch; only bit 0 (a passed-through blend) is returned.
  hit.ambiguous = hit.ambiguous & 1u;
  return hit;
}
