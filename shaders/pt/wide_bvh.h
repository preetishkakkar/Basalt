#pragma once
#include "bvh.h"

struct PtWideNode {
  float4 origin;
  float4 scale;
  uint4 children[8];
};

inline float ptWideComponent(float3 v, uint axis) {
  return axis == 0u ? v.x : axis == 1u ? v.y : v.z;
}

inline PtHit ptTraceWideBvh(device const PtWideNode *nodes, device const float4 *triangles,
                            PT_SCENE_PARAMS, PT_TEXTURE_PARAMS, float3 origin, float3 direction,
                            float tMax, uint mask, uint seed, float2 cone, uint anyHit) {
  PtHit hit;
  hit.ambiguous = 0u; hit.t = tMax; hit.barycentric = float2(0.0f);
  hit.instance = 0u; hit.primitive = 0u; hit.found = 0u;
  hit.nodeVisits = 0u; hit.triangleTests = 0u;
  PtRayPrep ray;
  ray.origin = origin; ray.inverse = float3(1.0f); ray.rowX = float3(1.0f, 0.0f, 0.0f);
  ray.rowY = float3(0.0f, 1.0f, 0.0f); ray.rowZ = float3(0.0f, 0.0f, 1.0f); ray.shearZ = 1.0f;
  float3 prepareOrigin = origin, prepareDirection = direction;
  bool prepare = true, done = false;
  uint stack[64] = {};
  uint depth = 1u, bottomBase = 0u, inBottom = 0u, instance = 0u;
  while (depth > 0u && !done) {
    if (inBottom != 0u && depth <= bottomBase) {
      inBottom = 0u; prepareOrigin = origin; prepareDirection = direction; prepare = true;
    }
    if (prepare) { ray = ptPrepareRay(prepareOrigin, prepareDirection); prepare = false; }
    const uint entry = stack[--depth];
    if ((entry & kBvhLeafTag) != 0u) {
      if (inBottom == 0u) {
        instance = entry & ~kBvhLeafTag;
        if ((traceInstances[instance].mask & mask) != 0u) {
          prepareOrigin = float3(dot(traceInstances[instance].worldToObject0, float4(origin, 1.0f)),
                                 dot(traceInstances[instance].worldToObject1, float4(origin, 1.0f)),
                                 dot(traceInstances[instance].worldToObject2, float4(origin, 1.0f)));
          prepareDirection = float3(dot(xyz(traceInstances[instance].worldToObject0), direction),
                                    dot(xyz(traceInstances[instance].worldToObject1), direction),
                                    dot(xyz(traceInstances[instance].worldToObject2), direction));
          prepare = true; inBottom = 1u; bottomBase = depth;
          stack[depth++] = traceInstances[instance].blasRoot;
        }
      } else {
        const uint first = entry & kBvhFirstMask;
        const uint count = (entry & ~kBvhLeafTag) >> kBvhCountShift;
        for (uint i = 0u; i < count && !done; ++i) {
          hit.triangleTests = hit.triangleTests + 1u;
          const uint triangle = (first + i) * 3u;
          const float4 v0 = triangles[triangle];
          float distance = 0.0f; float2 barycentric = float2(0.0f);
          if (ptIntersectTriangle(ray, xyz(v0), xyz(triangles[triangle + 1u]), xyz(triangles[triangle + 2u]),
                                  hit.t, distance, barycentric, hit.ambiguous)) {
            const uint primitive = as_type<uint>(v0.w);
            if ((traceInstances[instance].flags & kInstanceBlended) != 0u) hit.ambiguous |= 1u;
            if (ptCandidateSolid(PT_SCENE_ARGS, PT_TEXTURE_ARGS, instance, primitive, barycentric,
                                 (hit.ambiguous >> 1u) & 1u, seed, direction, ptConeWidthOrLevelZero(cone, distance))) {
              hit.ambiguous &= 1u; hit.t = distance; hit.barycentric = barycentric;
              hit.instance = instance; hit.primitive = primitive; hit.found = 1u;
              if (anyHit != 0u) done = true;
            }
          }
        }
      }
    } else {
      hit.nodeVisits = hit.nodeVisits + 1u;
      const PtWideNode node = nodes[entry];
      const uint childCount = as_type<uint>(node.origin.w);
      float distances[8] = {};
      uint entries[8] = {};
      uint found = 0u;
      for (uint child = 0u; child < childCount; ++child) {
        const uint4 packed = node.children[child];
        const uint3 words = uint3(packed.x, packed.y, packed.z);
        const float3 low = node.origin.xyz + node.scale.xyz * float3(words & uint3(0xFFFFu));
        const float3 high = node.origin.xyz + node.scale.xyz * float3(words >> uint3(16u));
        const float3 a = (low - ray.origin) * ray.inverse;
        const float3 z = (high - ray.origin) * ray.inverse;
        const float3 nearValue = min(a, z), farValue = max(a, z);
        const float nearDistance = max(max(nearValue.x, nearValue.y), max(nearValue.z, 0.0f));
        const float farDistance = min(min(farValue.x, farValue.y), min(farValue.z, hit.t)) * 1.0000004f;
        if (nearDistance <= farDistance) {
          distances[found] = nearDistance; entries[found] = packed.w; found = found + 1u;
        }
      }
      for (uint i = 1u; i < found; ++i) {
        const float d = distances[i]; const uint e = entries[i]; uint j = i;
        while (j > 0u && distances[j - 1u] < d) {
          distances[j] = distances[j - 1u]; entries[j] = entries[j - 1u]; j = j - 1u;
        }
        distances[j] = d; entries[j] = e;
      }
      // The host converter proves that the complete hierarchy needs at most 64 entries.
      for (uint i = 0u; i < found; ++i) stack[depth++] = entries[i];
    }
  }
  hit.ambiguous &= 1u;  // drop the per-candidate facing scratch bit
  return hit;
}
