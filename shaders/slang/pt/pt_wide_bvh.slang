// Quantized BVH4/BVH8 traversal. Children are ordered by octant with the nearest pushed last
// (lean: what the register-bound megakernel wants), or with SORTED by a full far-to-near sort
// (what the wavefront stages want, whose order sensitivity outweighs the sort).
module pt_wide_bvh;

import pt_path;
import pt_raycone;
import pt_bvh;

public static const uint kWideSlotMask = 0xFFu;       // origin.w: slots in use
public static const uint kWideOctantOrdered = 0x100u; // origin.w: slots assigned by octant

// The quantized wide traversal pushes every hit child, so a wide tree's worst-case stack
// (every child hit) grows by up to width - 1 per level; LBVH trees reach the 60s as BVH8
// (Sponza, animated). Kernels come with a 64-entry stack (larger local arrays leave the NVIDIA
// compiler's faster placement, 20% slower on Sponza) or kWideStackDeep, for trees whose bound
// exceeds 64.
public static const int kWideStack = 64;
public static const int kWideStackDeep = 96;

// Quantized wide node (BVH4/8): origin.w = slot count (low byte) and flags; child low/high =
// origin + scale * (word & 0xFFFF) / (word >> 16) per axis; children[i].w = node index, leaf
// entry, or kBvhEmpty for an empty slot. With kWideOctantOrdered, slot s holds the child a ray
// whose direction is negative on the axes of s's bits (x 1, y 2, z 4) meets first, so visiting
// slots k ^ octant for k = 0, 1, ... goes roughly near to far without sorting (Ylitie, Karras
// and Laine 2017); such a node has 8 slots and may leave some empty.
public __extern_cpp struct PtWideNode {
  __extern_cpp float4 origin;
  __extern_cpp float4 scale;
  __extern_cpp uint4 children[8];
};

// Entry distance into child slot `slot` of a quantized wide node, or -1 when the ray misses
// it or the node has fewer children. base and step fold the dequantization into the slab test.
float wideChildEntry(PtWideNode node, uint slot, uint slots, float3 base, float3 step, float tMax) {
  if (slot >= slots) return -1.0f;
  const uint4 packed = node.children[slot];
  if (packed.w == kBvhEmpty) return -1.0f;
  const uint3 words = packed.xyz;
  // 16-bit integers to float exactly without the quarter-rate conversion: placed in the
  // mantissa of 2^23 and 2^23 subtracted.
  const float3 lowQ = asfloat((words & uint3(0xFFFFu)) | uint3(0x4B000000u)) - float3(8388608.0f);
  const float3 highQ = asfloat((words >> uint3(16u)) | uint3(0x4B000000u)) - float3(8388608.0f);
  const float3 a = fma(lowQ, step, base);
  const float3 z = fma(highQ, step, base);
  const float3 nearValue = min(a, z), farValue = max(a, z);
  const float nearDistance = max(max(nearValue.x, nearValue.y), max(nearValue.z, 0.0f));
  const float farDistance = min(min(farValue.x, farValue.y), min(farValue.z, tMax)) * 1.0000004f;
  if (nearDistance <= farDistance) return nearDistance;
  return -1.0f;
}

// Records a slot the ray enters (distance >= 0) and keeps the nearest one.
void wideEnter(float distance, uint slot, inout uint hitMask, inout uint nearest, inout float nearestDistance) {
  if (distance < 0.0f) return;
  hitMask |= 1u << slot;
  if (nearest == 8u || distance < nearestDistance) {
    nearest = slot;
    nearestDistance = distance;
  }
}

// One comparator of the far-to-near sorting network: the farther entry (the lower slot on a
// tie) moves to the first position.
void wideOrder(inout float da, inout uint sa, inout float db, inout uint sb) {
  if (db > da || (db == da && sb < sa)) {
    const float d = da;
    da = db;
    db = d;
    const uint slot = sa;
    sa = sb;
    sb = slot;
  }
}

// The quantized wide software BVH, with a STACK-entry stack; SORTED orders children by a full
// far-to-near sort instead of by octant.
public struct WideBvhTracer<M : IPtMaps, let STACK : int, let SORTED : bool> : IPtTracer {
  StructuredBuffer<PtWideNode> nodes;
  StructuredBuffer<float4> triangles;
  PtGeometry geometry;
  M maps;

  PtHit trace(float3 origin, float3 direction, float tMax, uint mask, uint seed, float2 cone, bool anyHit) {
    PtHit hit = PtHit(tMax);
    PtRayPrep ray = PtRayPrep(origin, direction);
    float3 prepareOrigin = origin, prepareDirection = direction;
    bool prepare = false, done = false;
    uint stack[STACK]; // not filled per ray: read only below depth
    stack[0] = 0u;     // the top level's root
    uint depth = 1u, bottomBase = 0u, inBottom = 0u, instance = 0u;
    while (depth > 0u && !done) {
      if (inBottom != 0u && depth <= bottomBase) {
        inBottom = 0u;
        prepareOrigin = origin;
        prepareDirection = direction;
        prepare = true;
      }
      if (prepare) {
        ray = PtRayPrep(prepareOrigin, prepareDirection);
        prepare = false;
      }
      depth = depth - 1u;
      const uint entry = stack[depth];
      if ((entry & kBvhLeafTag) != 0u) {
        if (inBottom == 0u) {
          instance = entry & ~kBvhLeafTag;
          const TraceInstance info = geometry.traceInstances[instance];
          if ((info.mask & mask) != 0u) {
            prepareOrigin = float3(dot(info.worldToObject0, float4(origin, 1.0f)), dot(info.worldToObject1, float4(origin, 1.0f)),
                                   dot(info.worldToObject2, float4(origin, 1.0f)));
            prepareDirection = float3(dot(info.worldToObject0.xyz, direction), dot(info.worldToObject1.xyz, direction),
                                      dot(info.worldToObject2.xyz, direction));
            prepare = true;
            inBottom = 1u;
            bottomBase = depth;
            stack[depth] = info.blasRoot;
            depth = depth + 1u;
          }
        } else {
          const uint first = entry & kBvhFirstMask;
          const uint count = (entry & ~kBvhLeafTag) >> kBvhCountShift;
          for (uint i = 0u; i < count && !done; ++i) {
            hit.triangleTests = hit.triangleTests + 1u;
            const uint triangle = (first + i) * 3u;
            const float4 v0 = triangles[triangle];
            float distance;
            float2 barycentric;
            if (ray.intersectTriangle(v0.xyz, triangles[triangle + 1u].xyz, triangles[triangle + 2u].xyz, hit.t, distance,
                                      barycentric, hit.ambiguous)) {
              const uint primitive = asuint(v0.w);
              if ((geometry.traceInstances[instance].flags & kInstanceBlended) != 0u) hit.ambiguous |= 1u;
              if (ptCandidateSolid(geometry, maps, instance, primitive, barycentric, (hit.ambiguous >> 1u) & 1u, seed, direction,
                                   ptConeWidthOrLevelZero(cone, distance))) {
                hit.ambiguous &= 1u;
                hit.t = distance;
                hit.barycentric = barycentric;
                hit.instance = instance;
                hit.primitive = primitive;
                hit.found = 1u;
                if (anyHit) done = true;
              }
            }
          }
        }
      } else {
        hit.nodeVisits = hit.nodeVisits + 1u;
        const PtWideNode node = nodes[entry];
        const uint slots = asuint(node.origin.w) & kWideSlotMask;
        // A plane's distance, (origin + scale * q - rayOrigin) * inverse, as one fma per plane:
        // q * (scale * inverse) + (origin - rayOrigin) * inverse. The rounding differs from the
        // decoded box by far less than the one-bin margin every child box carries.
        const float3 base = (node.origin.xyz - ray.origin) * ray.inverse;
        const float3 step = node.scale.xyz * ray.inverse;
        // Children are pushed unchecked: the host converter proves that the complete hierarchy
        // fits the stack.
        if (SORTED) {
          // Every slot's entry distance in its own register (a miss sorts last as -1), ordered far
          // to near by a fixed network: no dynamically indexed arrays (which the driver keeps in
          // per-thread memory) and no data-dependent sort loop. Ties keep slot order.
          float d0 = wideChildEntry(node, 0u, slots, base, step, hit.t), d1 = wideChildEntry(node, 1u, slots, base, step, hit.t);
          float d2 = wideChildEntry(node, 2u, slots, base, step, hit.t), d3 = wideChildEntry(node, 3u, slots, base, step, hit.t);
          float d4 = -1.0f, d5 = -1.0f, d6 = -1.0f, d7 = -1.0f;
          uint s0 = 0u, s1 = 1u, s2 = 2u, s3 = 3u, s4 = 4u, s5 = 5u, s6 = 6u, s7 = 7u;
          wideOrder(d0, s0, d1, s1);
          wideOrder(d2, s2, d3, s3);
          wideOrder(d0, s0, d2, s2);
          wideOrder(d1, s1, d3, s3);
          wideOrder(d1, s1, d2, s2);
          if (slots > 4u) {
            // Slots 4-7 only when the node has them: sort them alike, then merge the two sorted
            // halves (Batcher's odd-even merge).
            d4 = wideChildEntry(node, 4u, slots, base, step, hit.t);
            d5 = wideChildEntry(node, 5u, slots, base, step, hit.t);
            d6 = wideChildEntry(node, 6u, slots, base, step, hit.t);
            d7 = wideChildEntry(node, 7u, slots, base, step, hit.t);
            wideOrder(d4, s4, d5, s5);
            wideOrder(d6, s6, d7, s7);
            wideOrder(d4, s4, d6, s6);
            wideOrder(d5, s5, d7, s7);
            wideOrder(d5, s5, d6, s6);
            wideOrder(d0, s0, d4, s4);
            wideOrder(d1, s1, d5, s5);
            wideOrder(d2, s2, d6, s6);
            wideOrder(d3, s3, d7, s7);
            wideOrder(d2, s2, d4, s4);
            wideOrder(d3, s3, d5, s5);
            wideOrder(d1, s1, d2, s2);
            wideOrder(d3, s3, d4, s4);
            wideOrder(d5, s5, d6, s6);
          }
          // The far child first, so the near one pops first.
          if (d0 >= 0.0f) stack[depth++] = nodes[entry].children[s0].w;
          if (d1 >= 0.0f) stack[depth++] = nodes[entry].children[s1].w;
          if (d2 >= 0.0f) stack[depth++] = nodes[entry].children[s2].w;
          if (d3 >= 0.0f) stack[depth++] = nodes[entry].children[s3].w;
          if (d4 >= 0.0f) stack[depth++] = nodes[entry].children[s4].w;
          if (d5 >= 0.0f) stack[depth++] = nodes[entry].children[s5].w;
          if (d6 >= 0.0f) stack[depth++] = nodes[entry].children[s6].w;
          if (d7 >= 0.0f) stack[depth++] = nodes[entry].children[s7].w;
        } else {
          // Which slots the ray enters, each tested at a constant index (the node stays in
          // registers), and the nearest of them.
          uint hitMask = 0u, nearest = 8u;
          float nearestDistance = 0.0f;
          wideEnter(wideChildEntry(node, 0u, slots, base, step, hit.t), 0u, hitMask, nearest, nearestDistance);
          wideEnter(wideChildEntry(node, 1u, slots, base, step, hit.t), 1u, hitMask, nearest, nearestDistance);
          wideEnter(wideChildEntry(node, 2u, slots, base, step, hit.t), 2u, hitMask, nearest, nearestDistance);
          wideEnter(wideChildEntry(node, 3u, slots, base, step, hit.t), 3u, hitMask, nearest, nearestDistance);
          wideEnter(wideChildEntry(node, 4u, slots, base, step, hit.t), 4u, hitMask, nearest, nearestDistance);
          wideEnter(wideChildEntry(node, 5u, slots, base, step, hit.t), 5u, hitMask, nearest, nearestDistance);
          wideEnter(wideChildEntry(node, 6u, slots, base, step, hit.t), 6u, hitMask, nearest, nearestDistance);
          wideEnter(wideChildEntry(node, 7u, slots, base, step, hit.t), 7u, hitMask, nearest, nearestDistance);
          // The rest in octant order (roughly near to far as k ^ octant), far first, and the
          // nearest last so it pops next.
          uint octant = 0u;
          if ((asuint(node.origin.w) & kWideOctantOrdered) != 0u) {
            if (ray.inverse.x < 0.0f) octant |= 1u;
            if (ray.inverse.y < 0.0f) octant |= 2u;
            if (ray.inverse.z < 0.0f) octant |= 4u;
          }
          for (uint k = 8u; k > 0u; --k) {
            const uint slot = (k - 1u) ^ octant;
            if (slot != nearest && ((hitMask >> slot) & 1u) != 0u) stack[depth++] = nodes[entry].children[slot].w;
          }
          if (nearest < 8u) stack[depth++] = nodes[entry].children[nearest].w;
        }
      }
    }
    hit.ambiguous &= 1u; // drop the per-candidate facing scratch bit
    return hit;
  }
};
