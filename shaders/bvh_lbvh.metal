// Parallel LBVH (Karras 2012) for every bottom level and the TLAS in one set of dispatches.
// It publishes the serial builder's output (bvh_build.metal) byte for byte: the same records,
// Morton codes, stable order, splits and bounds, and the serial numbering, whose top levels
// sit in the first nodes where every ray reads them. Segments are the instances' bottom
// levels, then the TLAS; see BvhSegment. Fence-free: a kernel reads only what earlier
// dispatches wrote, and talks across workgroups only through atomics on uints. Sequence:
// extents, instances, morton, the radix sort (bvh_sort.metal), topology, fit (bottom-up, once
// per tree level, indirect), number (top-down, once per level), emit, finish.
#include "shared/prelude.h"
#include "pt/bvh_build.h"

constant uint kLbvhEmpty = 0xFFFFFFFFu;
constant uint kLbvhLeaf = 0x80000000u;  // child encoding: a leaf at this sorted position
constant float kLbvhHuge = 3.0e38f;

// The segment holding record r among the bottom levels: the last whose first record is <= r
// (empty segments share their successor's first record and are skipped that way).
inline uint lbvhSegmentOfRecord(const device BvhSegment *segments, uint bottomSegments, uint record) {
  uint low = 0u, high = bottomSegments;
  while (high - low > 1u) {
    const uint middle = (low + high) >> 1u;
    if (segments[middle].range.x <= record) low = middle;
    else high = middle;
  }
  return low;
}

// The segment holding global internal node n: the last whose first internal node is <= n.
inline uint lbvhSegmentOfNode(const device BvhSegment *segments, uint segmentCount, uint node) {
  uint low = 0u, high = segmentCount;
  while (high - low > 1u) {
    const uint middle = (low + high) >> 1u;
    if (segments[middle].flags.w <= node) low = middle;
    else high = middle;
  }
  return low;
}

// Grows the six ordered-bit words at boxes[base..base+5] by a box.
inline void lbvhGrow(device atomic_uint *boxes, uint base, float3 low, float3 high) {
  atomic_fetch_min_explicit(&boxes[base], ptOrderedBits(low.x), memory_order_relaxed);
  atomic_fetch_min_explicit(&boxes[base + 1u], ptOrderedBits(low.y), memory_order_relaxed);
  atomic_fetch_min_explicit(&boxes[base + 2u], ptOrderedBits(low.z), memory_order_relaxed);
  atomic_fetch_max_explicit(&boxes[base + 3u], ptOrderedBits(high.x), memory_order_relaxed);
  atomic_fetch_max_explicit(&boxes[base + 4u], ptOrderedBits(high.y), memory_order_relaxed);
  atomic_fetch_max_explicit(&boxes[base + 5u], ptOrderedBits(high.z), memory_order_relaxed);
}

inline void lbvhFail(device atomic_uint *error, uint code) {
  atomic_fetch_max_explicit(&error[0], code, memory_order_relaxed);
}

// 1. Per triangle: its record and its bottom level's object box.
kernel void bvh_lbvh_extents(const device BvhSegment *segments [[buffer(0)]],
                             const device TraceInstance *instances [[buffer(1)]],
                             const device uint *indices [[buffer(2)]],
                             const device float *vertices [[buffer(3)]],
                             device BvhBuildRecord *records [[buffer(4)]],
                             device atomic_uint *segmentBox [[buffer(5)]],
                             device atomic_uint *error [[buffer(6)]],
                             constant BvhLbvhControl &control [[buffer(7)]],
                             uint id [[thread_position_in_grid]]) {
  if (id >= control.sizes.w) return;
  const uint g = lbvhSegmentOfRecord(segments, control.sizes.y - 1u, id);
  const BvhSegment segment = segments[g];
  const uint primitive = id - segment.range.x;
  const TraceInstance instance = instances[segment.flags.y];
  const uint base = instance.firstIndex + primitive * 3u;
  if (base + 2u >= control.geometry.x) { lbvhFail(error, kBvhBuildErrorGeometry); return; }
  const uint i0 = indices[base] + instance.vertexOffset;
  const uint i1 = indices[base + 1u] + instance.vertexOffset;
  const uint i2 = indices[base + 2u] + instance.vertexOffset;
  if (max(i0, max(i1, i2)) >= control.geometry.y) { lbvhFail(error, kBvhBuildErrorGeometry); return; }
  const float3 v0 = buildPosition(vertices, i0);
  const float3 v1 = buildPosition(vertices, i1);
  const float3 v2 = buildPosition(vertices, i2);
  if (!all(isfinite(v0)) || !all(isfinite(v1)) || !all(isfinite(v2))) {
    lbvhFail(error, kBvhBuildErrorNonFinite);
    return;
  }
  BvhBuildRecord record;
  record.identity = uint4(primitive, 0u, g, 0u);
  record.low = float4(min(v0, min(v1, v2)), 0.0f);
  record.high = float4(max(v0, max(v1, v2)), 0.0f);
  records[id] = record;
  lbvhGrow(segmentBox, g * 6u, xyz(record.low), xyz(record.high));
}

// 2. Per instance: the world box of its bottom level's object box, as a TLAS record, grown
// into the TLAS segment's box. An empty bottom level's box is the origin, as serially.
kernel void bvh_lbvh_instances(const device BvhSegment *segments [[buffer(0)]],
                               const device TraceInstance *instances [[buffer(1)]],
                               device BvhBuildRecord *records [[buffer(2)]],
                               device atomic_uint *segmentBox [[buffer(3)]],
                               constant BvhLbvhControl &control [[buffer(4)]],
                               uint id [[thread_position_in_grid]]) {
  if (id >= control.range.x) return;
  const uint tlas = control.sizes.y - 1u;
  BvhBuildRecord world;
  world.identity = uint4(id, 0u, tlas, 0u);
  world.low = float4(kLbvhHuge);
  world.high = float4(-kLbvhHuge);
  if (segments[id].range.y == 0u) {
    world.low = float4(0.0f);
    world.high = float4(0.0f);
  } else {
    const uint base = id * 6u;
    const float3 objectLow = float3(ptOrderedFloat(atomic_load_explicit(&segmentBox[base], memory_order_relaxed)),
                                    ptOrderedFloat(atomic_load_explicit(&segmentBox[base + 1u], memory_order_relaxed)),
                                    ptOrderedFloat(atomic_load_explicit(&segmentBox[base + 2u], memory_order_relaxed)));
    const float3 objectHigh = float3(ptOrderedFloat(atomic_load_explicit(&segmentBox[base + 3u], memory_order_relaxed)),
                                     ptOrderedFloat(atomic_load_explicit(&segmentBox[base + 4u], memory_order_relaxed)),
                                     ptOrderedFloat(atomic_load_explicit(&segmentBox[base + 5u], memory_order_relaxed)));
    const TraceInstance instance = instances[id];
    for (uint corner = 0u; corner < 8u; ++corner) {
      float3 point = objectLow;
      if ((corner & 1u) != 0u) point.x = objectHigh.x;
      if ((corner & 2u) != 0u) point.y = objectHigh.y;
      if ((corner & 4u) != 0u) point.z = objectHigh.z;
      const float3 transformed = transformPoint(instance, point);
      world.low = float4(min(xyz(world.low), transformed), world.low.w);
      world.high = float4(max(xyz(world.high), transformed), world.high.w);
    }
  }
  records[control.range.y + id] = world;
  lbvhGrow(segmentBox, tlas * 6u, xyz(world.low), xyz(world.high));
}

// 3. Per record: its Morton code in its segment's box, and the sort's (segment, code, record).
kernel void bvh_lbvh_morton(device BvhBuildRecord *records [[buffer(0)]],
                            const device uint *segmentBox [[buffer(1)]],
                            device uint *keysLo [[buffer(2)]],
                            device uint *keysHi [[buffer(3)]],
                            device uint *values [[buffer(4)]],
                            constant BvhLbvhControl &control [[buffer(5)]],
                            uint index [[thread_position_in_grid]]) {
  const uint id = control.part.x + index;
  if (id >= control.sizes.x) return;
  const BvhBuildRecord record = records[id];
  const uint g = record.identity.z;
  const float3 low = float3(ptOrderedFloat(segmentBox[g * 6u]), ptOrderedFloat(segmentBox[g * 6u + 1u]),
                            ptOrderedFloat(segmentBox[g * 6u + 2u]));
  const float3 high = float3(ptOrderedFloat(segmentBox[g * 6u + 3u]), ptOrderedFloat(segmentBox[g * 6u + 4u]),
                             ptOrderedFloat(segmentBox[g * 6u + 5u]));
  const uint code = morton3((xyz(record.low) + xyz(record.high)) * 0.5f, low, high);
  records[id].identity.y = code;
  keysLo[id] = code;
  keysHi[id] = g;
  values[id] = id;
}

// The serial builder's commonPrefix over (Morton code, identity) at a segment's local sorted
// positions a and b; -1 when b is outside the segment.
inline int lbvhDelta(const device uint *codes, const device uint *sortedIndex,
                     const device BvhBuildRecord *records, uint first, uint count, uint a, int b) {
  if (b < 0 || uint(b) >= count) return -1;
  const uint ma = codes[first + a];
  const uint mb = codes[first + uint(b)];
  if (ma != mb) return int(clz(ma ^ mb));
  const uint ia = records[sortedIndex[first + a]].identity.x;
  const uint ib = records[sortedIndex[first + uint(b)]].identity.x;
  if (ia == ib) return 64;
  return 32 + int(clz(ia ^ ib));
}

// Appends node to a level's queue: level L's nodes are entries[levelBase[L]...], its header
// (groups, 1, 1, count). Every internal node joins exactly one level, so the levels together
// fill the internal node count; the numbering pass walks them again from the top.
inline void lbvhPush(device atomic_uint *headers, device uint *entries, uint level, uint base, uint node) {
  const uint slot = atomic_fetch_add_explicit(&headers[level * 4u + 3u], 1u, memory_order_relaxed);
  entries[base + slot] = node;
  atomic_fetch_max_explicit(&headers[level * 4u], (slot + 64u) / 64u, memory_order_relaxed);
}

// 5. Per internal node: its range (Karras's determineRange), the serial builder's split, its
// children and its internal children's parent. Nodes without internal children seed the fit.
// Children: internal node index, kLbvhLeaf | sorted position, or kLbvhEmpty.
kernel void bvh_lbvh_topology(const device BvhSegment *segments [[buffer(0)]],
                              const device BvhBuildRecord *records [[buffer(1)]],
                              const device uint *codes [[buffer(2)]],
                              const device uint *sortedIndex [[buffer(3)]],
                              device uint4 *nodeChildren [[buffer(4)]],
                              device uint *nodeParent [[buffer(5)]],
                              device uint *nodeCounter [[buffer(6)]],
                              device uint *entries [[buffer(7)]],
                              device atomic_uint *headers [[buffer(8)]],
                              constant BvhLbvhControl &control [[buffer(9)]],
                              uint index [[thread_position_in_grid]]) {
  if (index >= control.part.z) return;
  const uint id = control.part.y + index;
  const uint g = lbvhSegmentOfNode(segments, control.sizes.y, id);
  const BvhSegment segment = segments[g];
  const uint k = id - segment.flags.w;
  const uint first = segment.range.x;
  const uint count = segment.range.y;
  uint left = kLbvhEmpty, right = kLbvhEmpty;
  if (count == 1u) left = kLbvhLeaf | first;
  if (count >= 2u) {
    uint lo = 0u, hi = count - 1u;
    if (k > 0u) {
      const int dLeft = lbvhDelta(codes, sortedIndex, records, first, count, k, int(k) - 1);
      const int dRight = lbvhDelta(codes, sortedIndex, records, first, count, k, int(k) + 1);
      int d = 1;
      int deltaMin = dLeft;
      if (dRight < dLeft) { d = -1; deltaMin = dRight; }
      int lMax = 2;
      while (lbvhDelta(codes, sortedIndex, records, first, count, k, int(k) + lMax * d) > deltaMin) lMax <<= 1;
      int l = 0;
      for (int t = lMax >> 1; t > 0; t >>= 1)
        if (lbvhDelta(codes, sortedIndex, records, first, count, k, int(k) + (l + t) * d) > deltaMin) l += t;
      const uint other = uint(int(k) + l * d);
      lo = min(k, other);
      hi = max(k, other);
    }
    const int common = lbvhDelta(codes, sortedIndex, records, first, count, lo, int(hi));
    uint split = lo;
    uint step = hi - lo;
    do {
      step = (step + 1u) >> 1u;
      const uint candidate = split + step;
      if (candidate < hi && lbvhDelta(codes, sortedIndex, records, first, count, lo, int(candidate)) > common)
        split = candidate;
    } while (step > 1u);
    left = segment.flags.w + split;
    if (split == lo) left = kLbvhLeaf | (first + split);
    right = segment.flags.w + split + 1u;
    if (split + 1u == hi) right = kLbvhLeaf | (first + split + 1u);
  }
  uint internal = 0u;
  if ((left & kLbvhLeaf) == 0u) { nodeParent[left] = id; internal += 1u; }
  if (right != kLbvhEmpty && (right & kLbvhLeaf) == 0u) { nodeParent[right] = id; internal += 1u; }
  if (k == 0u) nodeParent[id] = kLbvhEmpty;
  nodeChildren[id] = uint4(left, right, g, internal);
  nodeCounter[id] = 0u;
  if (internal == 0u) lbvhPush(headers, entries, 0u, 0u, id);
}

inline void lbvhChildBox(uint child, const device float4 *nodeBox, const device BvhBuildRecord *records,
                         const device uint *sortedIndex, thread float3 &low, thread float3 &high) {
  if ((child & kLbvhLeaf) != 0u) {
    const BvhBuildRecord record = records[sortedIndex[child & ~kLbvhLeaf]];
    low = xyz(record.low);
    high = xyz(record.high);
  } else {
    low = xyz(nodeBox[child * 2u]);
    high = xyz(nodeBox[child * 2u + 1u]);
  }
}

// 5b. Per internal node of the part: queue the nodes without internal children for the fit, as
// the topology does; a refit or a TLAS rebuild seeds the fit from the kept topology.
kernel void bvh_lbvh_seed(const device uint4 *nodeChildren [[buffer(0)]],
                          device uint *nodeCounter [[buffer(1)]],
                          device uint *entries [[buffer(2)]],
                          device atomic_uint *headers [[buffer(3)]],
                          constant BvhLbvhControl &control [[buffer(4)]],
                          uint index [[thread_position_in_grid]]) {
  if (index >= control.part.z) return;
  const uint id = control.part.y + index;
  nodeCounter[id] = 0u;
  if (nodeChildren[id].w == 0u) lbvhPush(headers, entries, 0u, 0u, id);
}

// 6. One tree level: iteration i fits the nodes whose internal children finished in earlier
// iterations (their boxes and sizes were written by earlier dispatches), and queues each parent
// whose last internal child this was for iteration i + 1. A node fitted in iteration i has
// height i + 1: the serial builder's depth when it is a root. Its size is the internal nodes
// of its subtree, itself included.
kernel void bvh_lbvh_fit(const device uint4 *nodeChildren [[buffer(0)]],
                         const device uint *nodeParent [[buffer(1)]],
                         device atomic_uint *nodeCounter [[buffer(2)]],
                         const device BvhBuildRecord *records [[buffer(3)]],
                         const device uint *sortedIndex [[buffer(4)]],
                         device float4 *nodeBox [[buffer(5)]],
                         device uint *nodeSize [[buffer(6)]],
                         device uint *entries [[buffer(7)]],
                         device uint *levelBase [[buffer(8)]],
                         device atomic_uint *headers [[buffer(9)]],
                         device uint *segmentHeight [[buffer(10)]],
                         constant BvhLbvhControl &control [[buffer(11)]],
                         uint id [[thread_position_in_grid]]) {
  const uint iteration = control.range.z;
  const uint count = atomic_load_explicit(&headers[iteration * 4u + 3u], memory_order_relaxed);
  if (id >= count) return;
  // This level's entries start at levelBase[i]; the next level's follow them, which the next
  // dispatch reads back.
  const uint base = levelBase[iteration];
  const uint nextBase = base + count;
  if (id == 0u) levelBase[iteration + 1u] = nextBase;
  const uint node = entries[base + id];
  const uint4 children = nodeChildren[node];
  float3 low = float3(kLbvhHuge), high = float3(-kLbvhHuge);
  if (children.x != kLbvhEmpty) lbvhChildBox(children.x, nodeBox, records, sortedIndex, low, high);
  if (children.y != kLbvhEmpty) {
    float3 rightLow = float3(0.0f), rightHigh = float3(0.0f);
    lbvhChildBox(children.y, nodeBox, records, sortedIndex, rightLow, rightHigh);
    low = min(low, rightLow);
    high = max(high, rightHigh);
  }
  nodeBox[node * 2u] = float4(low, 0.0f);
  nodeBox[node * 2u + 1u] = float4(high, 0.0f);
  uint size = 1u;
  if ((children.x & kLbvhLeaf) == 0u) size += nodeSize[children.x];
  if (children.y != kLbvhEmpty && (children.y & kLbvhLeaf) == 0u) size += nodeSize[children.y];
  nodeSize[node] = size;
  const uint parent = nodeParent[node];
  if (parent == kLbvhEmpty) {
    segmentHeight[children.z] = iteration + 1u;
    return;
  }
  const uint arrived = atomic_fetch_add_explicit(&nodeCounter[parent], 1u, memory_order_relaxed) + 1u;
  if (arrived == nodeChildren[parent].w) lbvhPush(headers, entries, iteration + 1u, nextBase, parent);
}

// 7. One tree level, from the highest down: the serial builder's numbering. It numbers a node's
// internal children when it takes the node from its stack (left, then right: consecutive) and
// then empties the right child's subtree before the left child's, so a node is
// (number, first number of its children) = (n, b): with two internal children, right = (b + 1,
// b + 2) and left = (b, b + 1 + size(right)); with one, that child = (b, b + 1). A root is
// (0, 1). A parent sits on a higher level than its children, so its dispatch came first.
kernel void bvh_lbvh_number(const device uint4 *nodeChildren [[buffer(0)]],
                            const device uint *nodeParent [[buffer(1)]],
                            const device uint *nodeSize [[buffer(2)]],
                            device uint2 *nodeNumber [[buffer(3)]],
                            const device uint *entries [[buffer(4)]],
                            const device uint *levelBase [[buffer(5)]],
                            const device uint *headers [[buffer(6)]],
                            constant BvhLbvhControl &control [[buffer(7)]],
                            uint id [[thread_position_in_grid]]) {
  const uint level = control.range.z;
  if (id >= headers[level * 4u + 3u]) return;
  const uint node = entries[levelBase[level] + id];
  uint2 own = uint2(0u, 1u);
  if (nodeParent[node] != kLbvhEmpty) own = nodeNumber[node];
  else nodeNumber[node] = own;
  const uint4 children = nodeChildren[node];
  const bool left = (children.x & kLbvhLeaf) == 0u;
  const bool right = children.y != kLbvhEmpty && (children.y & kLbvhLeaf) == 0u;
  if (left && right) {
    nodeNumber[children.y] = uint2(own.y + 1u, own.y + 2u);
    nodeNumber[children.x] = uint2(own.y, own.y + 1u + nodeSize[children.y]);
  } else if (left) {
    nodeNumber[children.x] = uint2(own.y, own.y + 1u);
  } else if (right) {
    nodeNumber[children.y] = uint2(own.y, own.y + 1u);
  }
}

inline float4 lbvhBits(float3 value, uint data) { return float4(value, as_type<float>(data)); }

// 8. Per internal node: its traversal node at its serial number; per bottom-level sorted
// position: its triangle.
kernel void bvh_lbvh_emit(const device BvhSegment *segments [[buffer(0)]],
                          const device uint4 *nodeChildren [[buffer(1)]],
                          const device float4 *nodeBox [[buffer(2)]],
                          const device uint2 *nodeNumber [[buffer(3)]],
                          const device BvhBuildRecord *records [[buffer(4)]],
                          const device uint *sortedIndex [[buffer(5)]],
                          const device TraceInstance *instances [[buffer(6)]],
                          const device uint *indices [[buffer(7)]],
                          const device float *vertices [[buffer(8)]],
                          device float4 *nodes [[buffer(9)]],
                          device float4 *triangles [[buffer(10)]],
                          constant BvhLbvhControl &control [[buffer(11)]],
                          uint index [[thread_position_in_grid]]) {
  const uint internalCount = control.part.z;
  if (index < internalCount) {
    const uint id = control.part.y + index;
    const uint4 children = nodeChildren[id];
    if (children.z >= control.sizes.y) return;  // a node the PLOC topology never made (it failed)
    const BvhSegment segment = segments[children.z];
    const uint output = segment.range.z + nodeNumber[id].x;
    for (uint side = 0u; side < 2u; ++side) {
      uint child = children.x;
      if (side == 1u) child = children.y;
      float3 low = float3(kLbvhHuge), high = float3(-kLbvhHuge);
      uint data = kLbvhEmpty, count = 0u;
      if (child != kLbvhEmpty) {
        lbvhChildBox(child, nodeBox, records, sortedIndex, low, high);
        if ((child & kLbvhLeaf) != 0u) {
          const uint sorted = child & ~kLbvhLeaf;
          data = segment.range.w + (sorted - segment.range.x);
          if (segment.flags.x != 0u) data = records[sortedIndex[sorted]].identity.x;
          count = 1u;
        } else {
          data = segment.range.z + nodeNumber[child].x;
        }
      }
      nodes[output * 4u + side * 2u] = lbvhBits(low, data);
      nodes[output * 4u + side * 2u + 1u] = lbvhBits(high, count);
    }
    return;
  }
  const uint sorted = index - internalCount;
  if (sorted >= control.part.w) return;
  const BvhBuildRecord record = records[sortedIndex[sorted]];
  const BvhSegment segment = segments[record.identity.z];
  const TraceInstance instance = instances[segment.flags.y];
  const uint primitive = record.identity.x;
  const uint base = instance.firstIndex + primitive * 3u;
  // A geometry error was recorded by the extents kernel; publish nothing for it here.
  if (base + 2u >= control.geometry.x) return;
  const uint i0 = indices[base] + instance.vertexOffset;
  const uint i1 = indices[base + 1u] + instance.vertexOffset;
  const uint i2 = indices[base + 2u] + instance.vertexOffset;
  if (max(i0, max(i1, i2)) >= control.geometry.y) return;
  const uint output = (segment.range.w + sorted - segment.range.x) * 3u;
  triangles[output] = lbvhBits(buildPosition(vertices, i0), primitive);
  triangles[output + 1u] = float4(buildPosition(vertices, i1), 0.0f);
  triangles[output + 2u] = float4(buildPosition(vertices, i2), 0.0f);
}

// 9. One thread: the status record. A root never fitted means a tree deeper than the fit's
// iterations; that and a traversal stack overflow are depth errors, as serially.
kernel void bvh_lbvh_finish(const device uint *segmentHeight [[buffer(0)]],
                            const device uint *headers [[buffer(1)]],
                            const device uint *error [[buffer(2)]],
                            device BvhBuildStatus2 *status [[buffer(3)]],
                            constant BvhLbvhControl &control [[buffer(4)]],
                            uint id [[thread_position_in_grid]]) {
  if (id != 0u) return;
  const uint segmentCount = control.sizes.y;
  uint result = error[0];
  const uint topDepth = segmentHeight[segmentCount - 1u];
  uint bottomDepth = 0u;
  bool unfitted = topDepth == 0u;
  for (uint g = 0u; g + 1u < segmentCount; ++g) {
    bottomDepth = max(bottomDepth, segmentHeight[g]);
    if (segmentHeight[g] == 0u) unfitted = true;
  }
  uint iterations = 0u;
  while (iterations < 64u && headers[iterations * 4u + 3u] != 0u) iterations += 1u;
  if (result == kBvhBuildErrorNone && (unfitted || topDepth + bottomDepth > control.geometry.z))
    result = kBvhBuildErrorDepth;
  BvhBuildStatus2 out;
  out.result = uint4(result, topDepth, bottomDepth, 0u);
  out.counts = uint4(control.sizes.z, control.sizes.w, control.range.x, 0u);
  out.extra = uint4(0u, 0u, iterations, 0u);
  out.cost = float4(0.0f);
  *status = out;
}
