// Quantized BVH4/BVH8 traversal. Children are ordered by octant with the nearest pushed last
// (lean: what the register-bound megakernel wants), or, with PT_WIDE_ORDER_SORT, by a full
// far-to-near sort (what the wavefront stages want, whose order sensitivity outweighs the sort).
#pragma once
#include "bvh.h"

inline float ptWideComponent(float3 v, uint axis) {
  return axis == 0u ? v.x : axis == 1u ? v.y : v.z;
}

// Entry distance into child slot `slot` of a quantized wide node, or -1 when the ray misses
// it or the node has fewer children. base and step fold the dequantization into the slab test.
inline float ptWideChildEntry(PtWideNode node, uint slot, uint slots, float3 base, float3 step, float tMax) {
  if (slot >= slots) return -1.0f;
  const uint4 packed = node.children[slot];
  if (packed.w == kBvhEmpty) return -1.0f;
  const uint3 words = uint3(packed.x, packed.y, packed.z);
  // 16-bit integers to float exactly without the quarter-rate conversion: placed in the
  // mantissa of 2^23 and 2^23 subtracted.
  const float3 lowQ = as_type<float3>((words & uint3(0xFFFFu)) | uint3(0x4B000000u)) - float3(8388608.0f);
  const float3 highQ = as_type<float3>((words >> uint3(16u)) | uint3(0x4B000000u)) - float3(8388608.0f);
  const float3 a = fma(lowQ, step, base);
  const float3 z = fma(highQ, step, base);
  const float3 nearValue = min(a, z), farValue = max(a, z);
  const float nearDistance = max(max(nearValue.x, nearValue.y), max(nearValue.z, 0.0f));
  const float farDistance = min(min(farValue.x, farValue.y), min(farValue.z, tMax)) * 1.0000004f;
  if (nearDistance <= farDistance) return nearDistance;
  return -1.0f;
}

// Records a slot the ray enters (distance >= 0) and keeps the nearest one.
inline void ptWideEnter(float distance, uint slot, thread uint &hitMask, thread uint &nearest,
                        thread float &nearestDistance) {
  if (distance < 0.0f) return;
  hitMask |= 1u << slot;
  if (nearest == 8u || distance < nearestDistance) {
    nearest = slot;
    nearestDistance = distance;
  }
}

// One comparator of the far-to-near sorting network (PT_WIDE_ORDER_SORT): the farther entry
// (the lower slot on a tie) moves to the first position.
inline void ptWideOrder(thread float &da, thread uint &sa, thread float &db, thread uint &sb) {
  if (db > da || (db == da && sb < sa)) {
    const float d = da; da = db; db = d;
    const uint slot = sa; sa = sb; sb = slot;
  }
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
  uint stack[PT_WIDE_BVH_STACK] = {};
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
      const uint slots = as_type<uint>(node.origin.w) & kWideSlotMask;
      // A plane's distance, (origin + scale * q - rayOrigin) * inverse, as one fma per plane:
      // q * (scale * inverse) + (origin - rayOrigin) * inverse. The rounding differs from the
      // decoded box by far less than the one-bin margin every child box carries.
      const float3 base = (xyz(node.origin) - ray.origin) * ray.inverse;
      const float3 step = xyz(node.scale) * ray.inverse;
#ifdef PT_WIDE_ORDER_SORT
      // Every slot's entry distance in its own register (a miss sorts last as -1), ordered far to
      // near by a fixed network: no dynamically indexed arrays (which the driver keeps in
      // per-thread memory) and no data-dependent sort loop. Ties keep slot order.
      float d0 = ptWideChildEntry(node, 0u, slots, base, step, hit.t), d1 = ptWideChildEntry(node, 1u, slots, base, step, hit.t);
      float d2 = ptWideChildEntry(node, 2u, slots, base, step, hit.t), d3 = ptWideChildEntry(node, 3u, slots, base, step, hit.t);
      float d4 = -1.0f, d5 = -1.0f, d6 = -1.0f, d7 = -1.0f;
      uint s0 = 0u, s1 = 1u, s2 = 2u, s3 = 3u, s4 = 4u, s5 = 5u, s6 = 6u, s7 = 7u;
      ptWideOrder(d0, s0, d1, s1); ptWideOrder(d2, s2, d3, s3);
      ptWideOrder(d0, s0, d2, s2); ptWideOrder(d1, s1, d3, s3);
      ptWideOrder(d1, s1, d2, s2);
      if (slots > 4u) {
        // Slots 4-7 only when the node has them: sort them alike, then merge the two sorted
        // halves (Batcher's odd-even merge).
        d4 = ptWideChildEntry(node, 4u, slots, base, step, hit.t); d5 = ptWideChildEntry(node, 5u, slots, base, step, hit.t);
        d6 = ptWideChildEntry(node, 6u, slots, base, step, hit.t); d7 = ptWideChildEntry(node, 7u, slots, base, step, hit.t);
        ptWideOrder(d4, s4, d5, s5); ptWideOrder(d6, s6, d7, s7);
        ptWideOrder(d4, s4, d6, s6); ptWideOrder(d5, s5, d7, s7);
        ptWideOrder(d5, s5, d6, s6);
        ptWideOrder(d0, s0, d4, s4); ptWideOrder(d1, s1, d5, s5); ptWideOrder(d2, s2, d6, s6); ptWideOrder(d3, s3, d7, s7);
        ptWideOrder(d2, s2, d4, s4); ptWideOrder(d3, s3, d5, s5);
        ptWideOrder(d1, s1, d2, s2); ptWideOrder(d3, s3, d4, s4); ptWideOrder(d5, s5, d6, s6);
      }
      // The far child first, so the near one pops first. The host converter proves that the
      // complete hierarchy needs at most 64 entries.
      if (d0 >= 0.0f) stack[depth++] = nodes[entry].children[s0].w;
      if (d1 >= 0.0f) stack[depth++] = nodes[entry].children[s1].w;
      if (d2 >= 0.0f) stack[depth++] = nodes[entry].children[s2].w;
      if (d3 >= 0.0f) stack[depth++] = nodes[entry].children[s3].w;
      if (d4 >= 0.0f) stack[depth++] = nodes[entry].children[s4].w;
      if (d5 >= 0.0f) stack[depth++] = nodes[entry].children[s5].w;
      if (d6 >= 0.0f) stack[depth++] = nodes[entry].children[s6].w;
      if (d7 >= 0.0f) stack[depth++] = nodes[entry].children[s7].w;
#else
      // Which slots the ray enters, each tested at a constant index (the node stays in registers),
      // and the nearest of them.
      uint hitMask = 0u, nearest = 8u;
      float nearestDistance = 0.0f;
      ptWideEnter(ptWideChildEntry(node, 0u, slots, base, step, hit.t), 0u, hitMask, nearest, nearestDistance);
      ptWideEnter(ptWideChildEntry(node, 1u, slots, base, step, hit.t), 1u, hitMask, nearest, nearestDistance);
      ptWideEnter(ptWideChildEntry(node, 2u, slots, base, step, hit.t), 2u, hitMask, nearest, nearestDistance);
      ptWideEnter(ptWideChildEntry(node, 3u, slots, base, step, hit.t), 3u, hitMask, nearest, nearestDistance);
      ptWideEnter(ptWideChildEntry(node, 4u, slots, base, step, hit.t), 4u, hitMask, nearest, nearestDistance);
      ptWideEnter(ptWideChildEntry(node, 5u, slots, base, step, hit.t), 5u, hitMask, nearest, nearestDistance);
      ptWideEnter(ptWideChildEntry(node, 6u, slots, base, step, hit.t), 6u, hitMask, nearest, nearestDistance);
      ptWideEnter(ptWideChildEntry(node, 7u, slots, base, step, hit.t), 7u, hitMask, nearest, nearestDistance);
      // The rest in octant order (bvh_layout.h: roughly near to far as k ^ octant), far first,
      // and the nearest last so it pops next. The host converter proves that the complete
      // hierarchy needs at most 64 entries.
      uint octant = 0u;
      if ((as_type<uint>(node.origin.w) & kWideOctantOrdered) != 0u) {
        if (ray.inverse.x < 0.0f) octant |= 1u;
        if (ray.inverse.y < 0.0f) octant |= 2u;
        if (ray.inverse.z < 0.0f) octant |= 4u;
      }
      for (uint k = 8u; k > 0u; --k) {
        const uint slot = (k - 1u) ^ octant;
        if (slot != nearest && ((hitMask >> slot) & 1u) != 0u) stack[depth++] = nodes[entry].children[slot].w;
      }
      if (nearest < 8u) stack[depth++] = nodes[entry].children[nearest].w;
#endif
    }
  }
  hit.ambiguous &= 1u;  // drop the per-candidate facing scratch bit
  return hit;
}
