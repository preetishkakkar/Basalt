// Tracing a ray: the interface every intersector implements, the alpha test they share, and
// the software BVH, laid out and traversed the way the hardware structure behaves: a top
// level over instances, one bottom level per instance in object space, masks and non-opaque
// candidates.
//
// Binary node: four float4, the bounds of both children, so one fetch tests two boxes.
//   n0 = left.min,  w = left data      n1 = left.max,  w = left count
//   n2 = right.min, w = right data     n3 = right.max, w = right count
// A count of zero is an interior child whose data is its node index; kBvhEmpty data marks
// a missing child. A positive count is a leaf: at the top level its data is an instance
// (count 1); at the bottom level its data is the first of `count` triangles.
// Triangle: three float4, the object-space vertices; the first w holds the triangle's index
// within its primitive, as an int's bits.
module pt_bvh;

import pt_path;
import pt_raycone;

public static const uint kBvhEmpty = 0xFFFFFFFFu;
public static const uint kBvhLeafTag = 0x80000000u; // stack entries: a leaf, not a node
public static const uint kBvhCountShift = 27u;      // bottom-level leaf entries: count above the first triangle
public static const uint kBvhFirstMask = 0x07FFFFFFu;
public static const int kBvhStack = 64;
// The binary traversal's stack for trees deeper than kBvhStack (top depth plus bottom depth): the
// GPU builders' trees may reach it (PLOC's over large scenes); its kernels are the *_deep entries.
public static const int kBvhStackDeep = 96;

// A ray through the scene: the closest solid hit, or with anyHit an occlusion query, whose only
// defined field is found (a tracer may stop at any solid hit and describe it or not). mask
// selects instances; seed drives the stochastic decisions on blended surfaces; cone is the ray
// cone where the ray leaves (width, spread; ptCameraCone), for the alpha tests.
public interface IPtTracer {
  PtHit trace(float3 origin, float3 direction, float tMax, uint mask, uint seed, float2 cone, bool anyHit);
};

// Whether a candidate on a non-opaque instance stops the ray: masked materials test their
// alpha against the cutoff; blended ones pass through with probability 1 - alpha. The alpha
// texture is read at the ray cone's level of detail at the candidate: the world ray direction
// and the cone width there (ptConeWidthOrLevelZero; -1 reads level zero), so every intersector
// sees the same coverage. A filtered alpha averages the footprint, and a fixed cutoff on an
// average turns a partly opaque footprint wholly solid or clear, so the cutoff is dithered over
// the footprint's width in texels f (ptTextureFootprint, capped at 1): cutoff + (xi - 0.5) f.
// From one texel up, a binary mask with cutoff 0.5 stops the ray with probability its opaque
// fraction, at every distance; magnified edges stay within a footprint; level-zero mode (f = 0)
// tests the cutoff exactly.
public bool ptCandidateSolid<M : IPtMaps>(PtGeometry geometry, M maps, uint instance, uint primitive, float2 barycentric,
                                          uint frontFacing, uint seed, float3 direction, float coneWidth) {
  const TraceInstance info = geometry.traceInstances[instance];
  if ((info.flags & kInstanceDoubleSided) == 0u && frontFacing == 0u) return false;
  if ((info.flags & (kInstanceMasked | kInstanceBlended)) == 0u) return true;
  const uint3 v = geometry.triangleVertices(instance, primitive);
  const float3 weights = float3(1.0f - barycentric.x - barycentric.y, barycentric.x, barycentric.y);
  const Material material = geometry.materials[info.material];
  const uint uvOffset = (material.texture.x & 0xFFu) == 1u ? 12u : 10u;
  float lodBase = kPtLevelZero; // level-zero mode skips the loads
  if (coneWidth >= 0.0f) {
    lodBase = ptConeLodBase(ptUvCross(geometry.uv(v.x, uvOffset), geometry.uv(v.y, uvOffset), geometry.uv(v.z, uvOffset)),
                            ptWorldCross(info.objectToWorld0, info.objectToWorld1, info.objectToWorld2,
                                         geometry.position(v.x), geometry.position(v.y), geometry.position(v.z)),
                            direction, coneWidth);
  }
  const float alpha = maps.sampleTexture(info.slots & 0xFFu, geometry.interpolate2(v, uvOffset, weights),
                                         material.texture.y & 0xFFu, lodBase).w *
                      material.baseColorFactor.w * geometry.interpolate1(v, 17u, weights);
  const float xi = hashFloat(seed ^ pcgHash(instance * 0x9E3779B9u + primitive));
  if ((info.flags & kInstanceBlended) != 0u) return xi < alpha;
  const float dither = lodBase > kPtLevelZero ? (xi - 0.5f) * maps.textureFootprint(info.slots & 0xFFu, lodBase) : 0.0f;
  return alpha >= material.alpha.x + dither;
}

// The ray, prepared for boxes (a safe inverse) and for Woop, Benthin and Wald's watertight
// triangle test. The test permutes the axes so the dominant one is z and shears x and y along
// it; here that permutation and shear are three rows, so a vertex's sheared coordinates are dot
// products and nothing indexes a vector by a variable.
public __extern_cpp struct PtRayPrep {
  __extern_cpp float3 origin;
  __extern_cpp float3 inverse;
  __extern_cpp float3 rowX;  // e_kx - (d_kx / d_kz) e_kz
  __extern_cpp float3 rowY;  // e_ky - (d_ky / d_kz) e_kz
  __extern_cpp float3 rowZ;  // e_kz
  __extern_cpp float shearZ; // 1 / d_kz

  // The constructor and the two tests are force-inlined: they run per node and per triangle, and
  // the generated C++ would otherwise call them.
  [ForceInline]
  __init(float3 origin, float3 direction) {
    this.origin = origin;
    // A zero component would make an infinite inverse, and zero times infinity a NaN.
    float3 safe = direction;
    if (abs(safe.x) < 1e-20f) safe.x = 1e-20f;
    if (abs(safe.y) < 1e-20f) safe.y = 1e-20f;
    if (abs(safe.z) < 1e-20f) safe.z = 1e-20f;
    inverse = float3(1.0f) / safe;
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
    rowX = ex - ez * (d.x / d.z);
    rowY = ey - ez * (d.y / d.z);
    rowZ = ez;
    shearZ = 1.0f / d.z;
  }

  // Entry distances into a node's two child boxes, kPtInfinity for a miss or a missing child.
  // The far distance is widened by a few ulps (Ize 2013) so a ray grazing a face is not lost.
  [ForceInline]
  float2 childEntries(float4 n0, float4 n1, float4 n2, float4 n3, float tMax) {
    const float3 l0 = (n0.xyz - origin) * inverse;
    const float3 l1 = (n1.xyz - origin) * inverse;
    const float3 r0 = (n2.xyz - origin) * inverse;
    const float3 r1 = (n3.xyz - origin) * inverse;
    const float2 entry = float2(max(max(min(l0.x, l1.x), min(l0.y, l1.y)), max(min(l0.z, l1.z), 0.0f)),
                                max(max(min(r0.x, r1.x), min(r0.y, r1.y)), max(min(r0.z, r1.z), 0.0f)));
    const float2 exit = float2(min(min(max(l0.x, l1.x), max(l0.y, l1.y)), min(max(l0.z, l1.z), tMax)),
                               min(min(max(r0.x, r1.x), max(r0.y, r1.y)), min(max(r0.z, r1.z), tMax))) *
                        1.0000004f;
    float2 result = float2(kPtInfinity);
    if (entry.x <= exit.x && asuint(n0.w) != kBvhEmpty) result.x = entry.x;
    if (entry.y <= exit.y && asuint(n2.w) != kBvhEmpty) result.y = entry.y;
    return result;
  }

  // Entry distances into a node's two child boxes, before any miss test (kPtInfinity for a
  // missing child): the restart trail's child order, which must not change as the ray's hit
  // shortens it.
  [ForceInline]
  float2 childOrder(float4 n0, float4 n1, float4 n2, float4 n3) {
    const float3 l0 = (n0.xyz - origin) * inverse;
    const float3 l1 = (n1.xyz - origin) * inverse;
    const float3 r0 = (n2.xyz - origin) * inverse;
    const float3 r1 = (n3.xyz - origin) * inverse;
    float2 entry = float2(max(max(min(l0.x, l1.x), min(l0.y, l1.y)), max(min(l0.z, l1.z), 0.0f)),
                          max(max(min(r0.x, r1.x), min(r0.y, r1.y)), max(min(r0.z, r1.z), 0.0f)));
    if (asuint(n0.w) == kBvhEmpty) entry.x = kPtInfinity;
    if (asuint(n2.w) == kBvhEmpty) entry.y = kPtInfinity;
    return entry;
  }

  // The watertight test; on a hit, the distance, the barycentrics of the second and third
  // vertices, and bit 1 of candidateFlags set when the triangle faces the ray.
  [ForceInline]
  bool intersectTriangle(float3 v0, float3 v1, float3 v2, float tMax, out float t, out float2 barycentric,
                                inout uint candidateFlags) {
    t = 0.0f;
    barycentric = float2(0.0f);
    const float3 sheared = float3(dot(rowX, v0 - origin), dot(rowX, v1 - origin), dot(rowX, v2 - origin));
    const float3 shearedY = float3(dot(rowY, v0 - origin), dot(rowY, v1 - origin), dot(rowY, v2 - origin));
    // Scaled barycentrics: the 2D edge functions of the sheared triangle.
    const float3 edges = float3(sheared.z * shearedY.y - shearedY.z * sheared.y, sheared.x * shearedY.z - shearedY.x * sheared.z,
                                sheared.y * shearedY.x - shearedY.y * sheared.x);
    const float determinant = edges.x + edges.y + edges.z;
    if (!((min(min(edges.x, edges.y), edges.z) >= 0.0f || max(max(edges.x, edges.y), edges.z) <= 0.0f)) ||
        determinant == 0.0f)
      return false;
    const float3 heights = float3(dot(rowZ, v0 - origin), dot(rowZ, v1 - origin), dot(rowZ, v2 - origin));
    const float distance = dot(edges, heights) * shearZ / determinant;
    if (!(distance > 0.0f) || !(distance < tMax)) return false;
    t = distance;
    barycentric = float2(edges.y, edges.z) / determinant;
    candidateFlags = (candidateFlags & 1u) | (determinant > 0.0f ? 2u : 0u);
    return true;
  }
};

// The binary software BVH's traversals (--bvh-traversal). kBinaryStack: a stack, the near child
// in a register (the default). kBinaryWhileWhile: the same order, as Aila and Laine's while-while
// loops (interior nodes until a leaf, then leaves until a node). kBinarySpeculative: while-while
// where a lane that finds a triangle leaf postpones it and keeps traversing nodes until every
// lane has one, so the leaf loop runs on full warps (Aila and Laine 2009). kBinaryRestartTrail:
// Laine's restart trail (2010) with a short stack: a bit per tree level records which child is
// being visited, so an empty stack restarts from the root instead of storing every pending node.
// All find the same closest hit.
public static const int kBinaryStack = 0;
public static const int kBinaryWhileWhile = 1;
public static const int kBinarySpeculative = 2;
public static const int kBinaryRestartTrail = 3;
public static const uint kRestartShortStack = 4u;
public static const uint kTrailLevels = 128u;  // the restart trail's tree levels

// A traversal's state between its steps (the while-while variants'): the entry to visit (none
// when hasEntry is false: the next comes from the stack) and the level it is in.
public struct BinaryWalk<let STACK : int> {
  PtRayPrep ray;
  uint stack[STACK];
  uint depth;
  uint entry;
  bool hasEntry;
  bool visiting;
  bool done;
  uint inBottom;
  uint bottomBase;
  uint instance;
};

// The binary software BVH. One stack serves both levels: an instance's bottom level is
// traversed above the entries its top level left, and the ray returns to world space when
// the stack falls back to them.
public struct BinaryBvhTracer<M : IPtMaps, let TRAVERSAL : int = kBinaryStack, let STACK : int = kBvhStack> : IPtTracer {
  StructuredBuffer<float4> nodes;
  StructuredBuffer<float4> triangles;
  PtGeometry geometry;
  M maps;

  // A leaf's triangles. The object-space ray is an affine image of the world ray, so its
  // distances are world distances and compare with hit.t directly.
  [ForceInline]
  void testTriangles(PtRayPrep ray, uint first, uint count, uint instance, float3 direction, uint seed, float2 cone,
                     bool anyHit, inout PtHit hit, inout bool done) {
    for (uint i = 0u; i < count && !done; ++i) {
      hit.triangleTests = hit.triangleTests + 1u;
      const uint triangle = (first + i) * 3u;
      const float4 v0 = triangles[triangle];
      float distance;
      float2 barycentric;
      if (ray.intersectTriangle(v0.xyz, triangles[triangle + 1u].xyz, triangles[triangle + 2u].xyz, hit.t, distance,
                                barycentric, hit.ambiguous)) {
        const uint local = asuint(v0.w);
        if ((geometry.traceInstances[instance].flags & kInstanceBlended) != 0u) hit.ambiguous = hit.ambiguous | 1u;
        if (ptCandidateSolid(geometry, maps, instance, local, barycentric, (hit.ambiguous >> 1u) & 1u, seed, direction,
                             ptConeWidthOrLevelZero(cone, distance))) {
          hit.ambiguous = hit.ambiguous & 1u;
          hit.t = distance;
          hit.barycentric = barycentric;
          hit.instance = instance;
          hit.primitive = local;
          hit.found = 1u;
          if (anyHit) done = true;
        }
      }
    }
  }

  // The ray in an instance's object space.
  [ForceInline]
  PtRayPrep objectRay(TraceInstance info, float3 origin, float3 direction) {
    return PtRayPrep(float3(dot(info.worldToObject0, float4(origin, 1.0f)), dot(info.worldToObject1, float4(origin, 1.0f)),
                            dot(info.worldToObject2, float4(origin, 1.0f))),
                     float3(dot(info.worldToObject0.xyz, direction), dot(info.worldToObject1.xyz, direction),
                            dot(info.worldToObject2.xyz, direction)));
  }

  PtHit trace(float3 origin, float3 direction, float tMax, uint mask, uint seed, float2 cone, bool anyHit) {
    if (TRAVERSAL == kBinaryWhileWhile || TRAVERSAL == kBinarySpeculative)
      return traceWhileWhile(origin, direction, tMax, mask, seed, cone, anyHit);
    if (TRAVERSAL == kBinaryRestartTrail) return traceRestartTrail(origin, direction, tMax, mask, seed, cone, anyHit);
    PtHit hit = PtHit(tMax);
    PtRayPrep ray = PtRayPrep(origin, direction);
    float3 prepareOrigin = origin;
    float3 prepareDirection = direction;
    bool prepare = false;
    // The entry being visited is held in a register; the stack keeps only the entries still to
    // visit (a node's far child), so the near child is never stored and reloaded. The visiting
    // order is the one of pushing both children and popping the near one.
    // Not filled per ray: only entries below depth are read, and a push wrote each of them. The
    // one store lets slangc's uninitialised-use analysis, which cannot follow depth, see the array
    // written; it costs one write where a fill cost STACK.
    uint stack[STACK];
    stack[0] = 0u;
    uint depth = 0u;
    uint entry = 0u; // the top level's root
    bool visiting = true;
    uint bottomBase = 0u;
    uint inBottom = 0u;
    uint instance = 0u;
    bool done = false;

    while (visiting && !done) {
      if (prepare) {
        ray = PtRayPrep(prepareOrigin, prepareDirection);
        prepare = false;
      }
      // Set by an entry that leaves nothing to visit next in its place: take the next from the stack.
      bool pop = true;

      if ((entry & kBvhLeafTag) != 0u) {
        if (inBottom == 0u) {
          // An instance: move the ray into its object space and descend into its bottom level.
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
            entry = info.blasRoot;
            pop = false;
          }
        } else {
          testTriangles(ray, entry & kBvhFirstMask, (entry & ~kBvhLeafTag) >> kBvhCountShift, instance, direction, seed,
                        cone, anyHit, hit, done);
        }
      } else {
        // An interior node: test both children; the far one is pushed, the near one visited next.
        hit.nodeVisits = hit.nodeVisits + 1u;
        const uint base = entry * 4u;
        const float4 n0 = nodes[base];
        const float4 n1 = nodes[base + 1u];
        const float4 n2 = nodes[base + 2u];
        const float4 n3 = nodes[base + 3u];
        const float2 entries = ray.childEntries(n0, n1, n2, n3, hit.t);
        // A child's stack entry: its node index, or a tagged leaf (the instance at the top
        // level; the count and first triangle at the bottom).
        uint2 push = uint2(asuint(n0.w), asuint(n2.w));
        const uint2 counts = uint2(asuint(n1.w), asuint(n3.w));
        if (counts.x > 0u) push.x = kBvhLeafTag | (counts.x << kBvhCountShift) * inBottom | push.x;
        if (counts.y > 0u) push.y = kBvhLeafTag | (counts.y << kBvhCountShift) * inBottom | push.y;
        if (entries.y < entries.x) push = push.yx;
        // Builders guarantee the depth fits; the checks guard against malformed buffers.
        if (max(entries.x, entries.y) < hit.t && depth < uint(STACK)) {
          stack[depth] = push.y;
          depth = depth + 1u;
        }
        if (min(entries.x, entries.y) < hit.t && depth < uint(STACK)) {
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

  // The while-while traversals' steps, each the default traversal's for one entry.

  // An interior node: the far child pushed, the near one next (or the stack's next).
  [ForceInline]
  void visitNode(inout BinaryWalk<STACK> w, inout PtHit hit) {
    hit.nodeVisits = hit.nodeVisits + 1u;
    const uint base = w.entry * 4u;
    const float4 n0 = nodes[base];
    const float4 n1 = nodes[base + 1u];
    const float4 n2 = nodes[base + 2u];
    const float4 n3 = nodes[base + 3u];
    const float2 entries = w.ray.childEntries(n0, n1, n2, n3, hit.t);
    uint2 push = uint2(asuint(n0.w), asuint(n2.w));
    const uint2 counts = uint2(asuint(n1.w), asuint(n3.w));
    if (counts.x > 0u) push.x = kBvhLeafTag | (counts.x << kBvhCountShift) * w.inBottom | push.x;
    if (counts.y > 0u) push.y = kBvhLeafTag | (counts.y << kBvhCountShift) * w.inBottom | push.y;
    if (entries.y < entries.x) push = push.yx;
    if (max(entries.x, entries.y) < hit.t && w.depth < uint(STACK)) {
      w.stack[w.depth] = push.y;
      w.depth = w.depth + 1u;
    }
    w.hasEntry = min(entries.x, entries.y) < hit.t && w.depth < uint(STACK);
    if (w.hasEntry) w.entry = push.x;
  }

  // The next entry from the stack, back in world space when it is the top level's; false when
  // none is left. With `stayInLevel`, false too rather than leave the bottom level.
  [ForceInline]
  bool popEntry(inout BinaryWalk<STACK> w, float3 origin, float3 direction, bool stayInLevel) {
    if (w.depth == 0u) {
      w.visiting = false;
      return false;
    }
    if (w.inBottom != 0u && w.depth <= w.bottomBase) {
      if (stayInLevel) return false;
      w.inBottom = 0u;
      w.ray = PtRayPrep(origin, direction);
    }
    w.depth = w.depth - 1u;
    w.entry = w.stack[w.depth];
    w.hasEntry = true;
    return true;
  }

  // A leaf entry: an instance (the ray moves into its bottom level) or triangles.
  [ForceInline]
  void visitLeaf(inout BinaryWalk<STACK> w, inout PtHit hit, float3 origin, float3 direction, uint mask, uint seed, float2 cone,
                 bool anyHit) {
    w.hasEntry = false;
    if (w.inBottom == 0u) {
      w.instance = w.entry & ~kBvhLeafTag;
      const TraceInstance info = geometry.traceInstances[w.instance];
      if ((info.mask & mask) != 0u) {
        w.ray = objectRay(info, origin, direction);
        w.inBottom = 1u;
        w.bottomBase = w.depth;
        w.entry = info.blasRoot;
        w.hasEntry = true;
      }
    } else {
      testTriangles(w.ray, w.entry & kBvhFirstMask, (w.entry & ~kBvhLeafTag) >> kBvhCountShift, w.instance, direction, seed,
                    cone, anyHit, hit, w.done);
    }
  }

  PtHit traceWhileWhile(float3 origin, float3 direction, float tMax, uint mask, uint seed, float2 cone, bool anyHit) {
    PtHit hit = PtHit(tMax);
    BinaryWalk<STACK> w;
    w.ray = PtRayPrep(origin, direction);
    w.stack[0] = 0u;
    w.depth = 0u;
    w.entry = 0u;
    w.hasEntry = true;
    w.visiting = true;
    w.done = false;
    w.inBottom = 0u;
    w.bottomBase = 0u;
    w.instance = 0u;
    // Speculative: a triangle leaf this lane found and postponed; it needs this bottom level's ray,
    // so the traversal does not leave the level while one waits.
    uint postponed = 0u;
    bool waiting = false;
    while (w.visiting && !w.done) {
      // Interior nodes, until a leaf.
      for (;;) {
        if (!w.hasEntry && !popEntry(w, origin, direction, waiting)) break;
        if ((w.entry & kBvhLeafTag) != 0u) {
          if (TRAVERSAL != kBinarySpeculative || w.inBottom == 0u || waiting) break;
          postponed = w.entry;
          waiting = true;
          w.hasEntry = false;
        } else {
          visitNode(w, hit);
        }
        if (TRAVERSAL == kBinarySpeculative && WaveActiveAllTrue(waiting)) break;
      }
      if (waiting) {
        testTriangles(w.ray, postponed & kBvhFirstMask, (postponed & ~kBvhLeafTag) >> kBvhCountShift, w.instance, direction,
                      seed, cone, anyHit, hit, w.done);
        waiting = false;
      }
      // Leaves, until an interior node.
      while (w.visiting && !w.done) {
        if (!w.hasEntry && !popEntry(w, origin, direction, false)) break;
        if ((w.entry & kBvhLeafTag) == 0u) break;
        visitLeaf(w, hit, origin, direction, mask, seed, cone, anyHit);
      }
    }
    hit.ambiguous = hit.ambiguous & 1u;
    return hit;
  }

  // The restart trail's bit for tree level `level` (root 0), kTrailLevels in four words (it keeps
  // no stack, so it takes trees as deep as the deep kernels' and deeper).
  [ForceInline]
  bool trailBit(uint4 trail, uint level) { return ((trail[level >> 5u] >> (level & 31u)) & 1u) != 0u; }

  // The trail with `level`'s bit set.
  [ForceInline]
  uint4 trailOr(uint4 trail, uint level) {
    trail[level >> 5u] = trail[level >> 5u] | (1u << (level & 31u));
    return trail;
  }

  // The trail with `level`'s bit set and every deeper one cleared.
  [ForceInline]
  uint4 trailSet(uint4 trail, uint level) {
    const uint word = level >> 5u, bit = level & 31u;
    for (uint w = word + 1u; w < 4u; ++w) trail[w] = 0u;
    trail[word] = (trail[word] & ((1u << bit) - 1u)) | (1u << bit);
    return trail;
  }

  // Laine's restart trail. A node at tree level d visits its first (near) child while bit d is
  // clear and its second once the first is done: its bit set. A node with one child hit sets the
  // bit on the way down, so that child finishes the node. A finished subtree at level d resumes
  // at the deepest shallower level whose bit is clear, setting it: from the short stack's top
  // (pushed when that level's node went to its first child) or, the stack empty, by descending
  // again from the root along the trail. The child order is the boxes' entry distances before
  // the miss test (childOrder), which a shorter hit does not change; a descent that finds a first
  // child now missed takes the second and clears the deeper bits, which described the first's.
  PtHit traceRestartTrail(float3 origin, float3 direction, float tMax, uint mask, uint seed, float2 cone, bool anyHit) {
    PtHit hit = PtHit(tMax);
    PtRayPrep ray = PtRayPrep(origin, direction);
    uint4 trail = uint4(0u);
    uint entry = 0u;       // the top level's root
    uint level = 0u;       // the entry's tree level
    uint inBottom = 0u;
    uint bottomBase = 0u;  // the level of the bottom level's root (its instance's)
    uint instance = 0u;
    uint2 shortStack[kRestartShortStack];  // (entry, level), a ring: overflow drops the oldest
    shortStack[0] = uint2(0u);
    uint top = 0u, held = 0u;
    bool done = false;
    for (;;) {
      bool finished = false;  // the subtree at `level` is done
      if ((entry & kBvhLeafTag) == 0u) {
        hit.nodeVisits = hit.nodeVisits + 1u;
        const uint base = entry * 4u;
        const float4 n0 = nodes[base];
        const float4 n1 = nodes[base + 1u];
        const float4 n2 = nodes[base + 2u];
        const float4 n3 = nodes[base + 3u];
        const float2 entries = ray.childEntries(n0, n1, n2, n3, hit.t);
        const float2 order = ray.childOrder(n0, n1, n2, n3);
        uint2 child = uint2(asuint(n0.w), asuint(n2.w));
        const uint2 counts = uint2(asuint(n1.w), asuint(n3.w));
        if (counts.x > 0u) child.x = kBvhLeafTag | (counts.x << kBvhCountShift) * inBottom | child.x;
        if (counts.y > 0u) child.y = kBvhLeafTag | (counts.y << kBvhCountShift) * inBottom | child.y;
        bool2 hits = bool2(entries.x < hit.t, entries.y < hit.t);
        if (order.y < order.x) {
          child = child.yx;
          hits = hits.yx;
        }
        const bool second = trailBit(trail, level);
        if (level + 1u >= kTrailLevels) {
          finished = true;  // deeper than the trail holds: a malformed tree
        } else if (hits.x && hits.y) {
          if (second) {
            entry = child.y;
          } else {
            entry = child.x;
            shortStack[top] = uint2(child.y, level + 1u);
            top = (top + 1u) % kRestartShortStack;
            held = min(held + 1u, kRestartShortStack);
          }
          level = level + 1u;
        } else if (hits.x || hits.y) {
          // One child: the node's only one, as if its second (the bit set). A first child keeps
          // the deeper bits, which describe it on a restart; a second child the first of which
          // was pending (bit clear) but now misses clears them. A set bit with only the first
          // child hit may mean that child is done and the second now missed: it is walked again,
          // which finds nothing new.
          if (!second) trail = hits.x ? trailOr(trail, level) : trailSet(trail, level);
          entry = hits.x ? child.x : child.y;
          level = level + 1u;
        } else {
          finished = true;
        }
      } else if (inBottom == 0u) {
        // An instance: its bottom level's root takes its place, at its level.
        instance = entry & ~kBvhLeafTag;
        const TraceInstance info = geometry.traceInstances[instance];
        if ((info.mask & mask) != 0u) {
          ray = objectRay(info, origin, direction);
          inBottom = 1u;
          bottomBase = level;
          entry = info.blasRoot;
        } else {
          finished = true;
        }
      } else {
        testTriangles(ray, entry & kBvhFirstMask, (entry & ~kBvhLeafTag) >> kBvhCountShift, instance, direction, seed, cone,
                      anyHit, hit, done);
        finished = true;
      }
      if (done) break;
      if (!finished) continue;
      // Resume at the deepest shallower level with its bit clear.
      uint resume = kTrailLevels;
      for (int w = int(level >> 5u); w >= 0 && resume == kTrailLevels; --w) {
        uint clear = ~trail[w];
        if (uint(w) == (level >> 5u)) clear &= (1u << (level & 31u)) - 1u;
        if (clear != 0u) resume = uint(w) * 32u + firstbithigh(clear);
      }
      if (resume == kTrailLevels) break;
      trail = trailSet(trail, resume);
      if (held > 0u) {
        // The short stack's top is that level's second child.
        top = (top + kRestartShortStack - 1u) % kRestartShortStack;
        held = held - 1u;
        entry = shortStack[top].x;
        level = shortStack[top].y;
        if (inBottom != 0u && level <= bottomBase) {
          inBottom = 0u;
          ray = PtRayPrep(origin, direction);
        }
      } else {
        entry = 0u;
        level = 0u;
        if (inBottom != 0u) {
          inBottom = 0u;
          ray = PtRayPrep(origin, direction);
        }
      }
    }
    hit.ambiguous = hit.ambiguous & 1u;
    return hit;
  }
};
