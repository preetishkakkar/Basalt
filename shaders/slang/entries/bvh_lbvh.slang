// Parallel LBVH (Karras 2012) for every bottom level and the TLAS in one set of dispatches. It
// publishes the serial builder's output (bvh_build.slang) byte for byte: the same records,
// Morton codes, stable order, splits and bounds, and the serial numbering, whose top levels sit
// in the first nodes where every ray reads them. Segments are the instances' bottom levels, then
// the TLAS; see BvhSegment. Fence-free: a kernel reads only what earlier dispatches wrote, and
// talks across workgroups only through atomics on uints. Sequence: extents, instances, morton,
// the radix sort (bvh_sort.slang), topology, fit (bottom-up, once per tree level, indirect),
// number (top-down, once per level), emit, finish.
import pt_bvh_build;

static const uint kLbvhEmpty = 0xFFFFFFFFu;
static const uint kLbvhLeaf = 0x80000000u; // child encoding: a leaf at this sorted position
static const float kLbvhHuge = 3.0e38f;

ConstantBuffer<BvhLbvhControl> control;
StructuredBuffer<BvhSegment> segments;
StructuredBuffer<TraceInstance> instances;
StructuredBuffer<uint> indices;
StructuredBuffer<float> vertices;
RWStructuredBuffer<BvhBuildRecord> records;
RWStructuredBuffer<uint> segmentBox;    // six ordered-bit words per segment
RWStructuredBuffer<uint> error;
RWStructuredBuffer<uint> keysLo;
RWStructuredBuffer<uint> keysHi;
RWStructuredBuffer<uint> values;
StructuredBuffer<uint> codes;           // the sorted Morton codes
StructuredBuffer<uint> sortedIndex;     // the sorted record indices
RWStructuredBuffer<uint4> nodeChildren;
RWStructuredBuffer<uint> nodeParent;
RWStructuredBuffer<uint> nodeCounter;
RWStructuredBuffer<uint> entries;
RWStructuredBuffer<uint> headers;       // per level: groups, 1, 1, count
RWStructuredBuffer<float4> nodeBox;
RWStructuredBuffer<uint> nodeSize;
RWStructuredBuffer<uint> levelBase;
RWStructuredBuffer<uint> segmentHeight;
RWStructuredBuffer<uint2> nodeNumber;
RWStructuredBuffer<float4> nodes;
RWStructuredBuffer<float4> triangles;
RWStructuredBuffer<BvhBuildStatus2> status;

// The segment holding record r among the bottom levels: the last whose first record is <= r
// (empty segments share their successor's first record and are skipped that way).
uint segmentOfRecord(uint bottomSegments, uint record) {
  uint low = 0u, high = bottomSegments;
  while (high - low > 1u) {
    const uint middle = (low + high) >> 1u;
    if (segments[middle].range.x <= record) low = middle;
    else high = middle;
  }
  return low;
}

// The segment holding global internal node n: the last whose first internal node is <= n.
uint segmentOfNode(uint segmentCount, uint node) {
  uint low = 0u, high = segmentCount;
  while (high - low > 1u) {
    const uint middle = (low + high) >> 1u;
    if (segments[middle].flags.w <= node) low = middle;
    else high = middle;
  }
  return low;
}

// Grows a segment's six ordered-bit words by a box.
void grow(uint base, float3 low, float3 high) {
  InterlockedMin(segmentBox[base], ptOrderedBits(low.x));
  InterlockedMin(segmentBox[base + 1u], ptOrderedBits(low.y));
  InterlockedMin(segmentBox[base + 2u], ptOrderedBits(low.z));
  InterlockedMax(segmentBox[base + 3u], ptOrderedBits(high.x));
  InterlockedMax(segmentBox[base + 4u], ptOrderedBits(high.y));
  InterlockedMax(segmentBox[base + 5u], ptOrderedBits(high.z));
}

void fail(uint code) { InterlockedMax(error[0], code); }

float3 segmentLow(uint g) {
  return float3(ptOrderedFloat(segmentBox[g * 6u]), ptOrderedFloat(segmentBox[g * 6u + 1u]), ptOrderedFloat(segmentBox[g * 6u + 2u]));
}
float3 segmentHigh(uint g) {
  return float3(ptOrderedFloat(segmentBox[g * 6u + 3u]), ptOrderedFloat(segmentBox[g * 6u + 4u]),
                ptOrderedFloat(segmentBox[g * 6u + 5u]));
}

// 1. Per triangle: its record and its bottom level's object box.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_lbvh_extents(uint3 thread: SV_DispatchThreadID) {
  const uint id = thread.x;
  if (id >= control.sizes.w) return;
  const uint g = segmentOfRecord(control.sizes.y - 1u, id);
  const BvhSegment segment = segments[g];
  const uint primitive = id - segment.range.x;
  const TraceInstance instance = instances[segment.flags.y];
  const uint base = instance.firstIndex + primitive * 3u;
  if (base + 2u >= control.geometry.x) {
    fail(kBvhBuildErrorGeometry);
    return;
  }
  const uint i0 = indices[base] + instance.vertexOffset;
  const uint i1 = indices[base + 1u] + instance.vertexOffset;
  const uint i2 = indices[base + 2u] + instance.vertexOffset;
  if (max(i0, max(i1, i2)) >= control.geometry.y) {
    fail(kBvhBuildErrorGeometry);
    return;
  }
  const float3 v0 = buildPosition(vertices, i0);
  const float3 v1 = buildPosition(vertices, i1);
  const float3 v2 = buildPosition(vertices, i2);
  if (!all(isfinite(v0)) || !all(isfinite(v1)) || !all(isfinite(v2))) {
    fail(kBvhBuildErrorNonFinite);
    return;
  }
  BvhBuildRecord record;
  record.identity = uint4(primitive, 0u, g, 0u);
  record.low = float4(min(v0, min(v1, v2)), 0.0f);
  record.high = float4(max(v0, max(v1, v2)), 0.0f);
  records[id] = record;
  grow(g * 6u, record.low.xyz, record.high.xyz);
}

// 2. Per instance: the world box of its bottom level's object box, as a TLAS record, grown into
// the TLAS segment's box. An empty bottom level's box is the origin, as serially.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_lbvh_instances(uint3 thread: SV_DispatchThreadID) {
  const uint id = thread.x;
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
    const float3 objectLow = segmentLow(id);
    const float3 objectHigh = segmentHigh(id);
    const TraceInstance instance = instances[id];
    for (uint corner = 0u; corner < 8u; ++corner) {
      float3 point = objectLow;
      if ((corner & 1u) != 0u) point.x = objectHigh.x;
      if ((corner & 2u) != 0u) point.y = objectHigh.y;
      if ((corner & 4u) != 0u) point.z = objectHigh.z;
      const float3 transformed = transformPoint(instance, point);
      world.low = float4(min(world.low.xyz, transformed), world.low.w);
      world.high = float4(max(world.high.xyz, transformed), world.high.w);
    }
  }
  records[control.range.y + id] = world;
  grow(tlas * 6u, world.low.xyz, world.high.xyz);
}

// 3. Per record: its Morton code in its segment's box, and the sort's (segment, code, record).
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_lbvh_morton(uint3 thread: SV_DispatchThreadID) {
  const uint id = control.part.x + thread.x;
  if (id >= control.sizes.x) return;
  const BvhBuildRecord record = records[id];
  const uint g = record.identity.z;
  const uint code = morton3((record.low.xyz + record.high.xyz) * 0.5f, segmentLow(g), segmentHigh(g));
  records[id].identity.y = code;
  keysLo[id] = code;
  keysHi[id] = g;
  values[id] = id;
}

// The serial builder's commonPrefix over (Morton code, identity) at a segment's local sorted
// positions a and b; -1 when b is outside the segment.
int delta(uint first, uint count, uint a, int b) {
  if (b < 0 || uint(b) >= count) return -1;
  const uint ma = codes[first + a];
  const uint mb = codes[first + uint(b)];
  if (ma != mb) return int(leadingZeros(ma ^ mb));
  const uint ia = records[sortedIndex[first + a]].identity.x;
  const uint ib = records[sortedIndex[first + uint(b)]].identity.x;
  if (ia == ib) return 64;
  return 32 + int(leadingZeros(ia ^ ib));
}

// Appends node to a level's queue: level L's nodes are entries[levelBase[L]...], its header
// (groups, 1, 1, count). Every internal node joins exactly one level, so the levels together
// fill the internal node count; the numbering pass walks them again from the top.
void push(uint level, uint base, uint node) {
  uint slot;
  InterlockedAdd(headers[level * 4u + 3u], 1u, slot);
  entries[base + slot] = node;
  InterlockedMax(headers[level * 4u], (slot + 64u) / 64u);
}

// 5. Per internal node: its range (Karras's determineRange), the serial builder's split, its
// children and its internal children's parent. Nodes without internal children seed the fit.
// Children: internal node index, kLbvhLeaf | sorted position, or kLbvhEmpty.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_lbvh_topology(uint3 thread: SV_DispatchThreadID) {
  const uint index = thread.x;
  if (index >= control.part.z) return;
  const uint id = control.part.y + index;
  const uint g = segmentOfNode(control.sizes.y, id);
  const BvhSegment segment = segments[g];
  const uint k = id - segment.flags.w;
  const uint first = segment.range.x;
  const uint count = segment.range.y;
  uint left = kLbvhEmpty, right = kLbvhEmpty;
  if (count == 1u) left = kLbvhLeaf | first;
  if (count >= 2u) {
    // Node 0 spans the segment. Any other node's range grows from k towards the neighbour
    // sharing the longer prefix (d): an exponential search bounds its length, a binary search
    // finds it.
    uint lo = 0u, hi = count - 1u;
    if (k > 0u) {
      const int dLeft = delta(first, count, k, int(k) - 1);
      const int dRight = delta(first, count, k, int(k) + 1);
      int d = 1;
      int deltaMin = dLeft;
      if (dRight < dLeft) {
        d = -1;
        deltaMin = dRight;
      }
      int lMax = 2;
      while (delta(first, count, k, int(k) + lMax * d) > deltaMin) lMax <<= 1;
      int l = 0;
      for (int t = lMax >> 1; t > 0; t >>= 1)
        if (delta(first, count, k, int(k) + (l + t) * d) > deltaMin) l += t;
      const uint other = uint(int(k) + l * d);
      lo = min(k, other);
      hi = max(k, other);
    }
    // The split: the last position whose prefix with lo exceeds the range's common prefix
    // (binary search, as the serial builder's splitRange).
    const int common = delta(first, count, lo, int(hi));
    uint split = lo;
    uint step = hi - lo;
    do {
      step = (step + 1u) >> 1u;
      const uint candidate = split + step;
      if (candidate < hi && delta(first, count, lo, int(candidate)) > common) split = candidate;
    } while (step > 1u);
    left = split == lo ? kLbvhLeaf | (first + split) : segment.flags.w + split;
    right = split + 1u == hi ? kLbvhLeaf | (first + split + 1u) : segment.flags.w + split + 1u;
  }
  uint internal = 0u;
  if ((left & kLbvhLeaf) == 0u) {
    nodeParent[left] = id;
    internal += 1u;
  }
  if (right != kLbvhEmpty && (right & kLbvhLeaf) == 0u) {
    nodeParent[right] = id;
    internal += 1u;
  }
  if (k == 0u) nodeParent[id] = kLbvhEmpty;
  nodeChildren[id] = uint4(left, right, g, internal);
  nodeCounter[id] = 0u;
  if (internal == 0u) push(0u, 0u, id);
}

void childBox(uint child, out float3 low, out float3 high) {
  if ((child & kLbvhLeaf) != 0u) {
    const BvhBuildRecord record = records[sortedIndex[child & ~kLbvhLeaf]];
    low = record.low.xyz;
    high = record.high.xyz;
  } else {
    low = nodeBox[child * 2u].xyz;
    high = nodeBox[child * 2u + 1u].xyz;
  }
}

// 5b. Per internal node of the part: queue the nodes without internal children for the fit, as
// the topology does; a refit or a TLAS rebuild seeds the fit from the kept topology.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_lbvh_seed(uint3 thread: SV_DispatchThreadID) {
  if (thread.x >= control.part.z) return;
  const uint id = control.part.y + thread.x;
  nodeCounter[id] = 0u;
  if (nodeChildren[id].w == 0u) push(0u, 0u, id);
}

// 6. One tree level: iteration i fits the nodes whose internal children finished in earlier
// iterations (their boxes and sizes were written by earlier dispatches), and queues each parent
// whose last internal child this was for iteration i + 1. A node fitted in iteration i has height
// i + 1: the serial builder's depth when it is a root. Its size is the internal nodes of its
// subtree, itself included.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_lbvh_fit(uint3 thread: SV_DispatchThreadID) {
  const uint id = thread.x;
  const uint iteration = control.range.z;
  const uint count = headers[iteration * 4u + 3u];
  if (id >= count) return;
  // This level's entries start at levelBase[i]; the next level's follow them, which the next
  // dispatch reads back.
  const uint base = levelBase[iteration];
  const uint nextBase = base + count;
  if (id == 0u) levelBase[iteration + 1u] = nextBase;
  const uint node = entries[base + id];
  const uint4 children = nodeChildren[node];
  float3 low = float3(kLbvhHuge), high = float3(-kLbvhHuge);
  if (children.x != kLbvhEmpty) childBox(children.x, low, high);
  if (children.y != kLbvhEmpty) {
    float3 rightLow, rightHigh;
    childBox(children.y, rightLow, rightHigh);
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
  uint arrived;
  InterlockedAdd(nodeCounter[parent], 1u, arrived);
  if (arrived + 1u == nodeChildren[parent].w) push(iteration + 1u, nextBase, parent);
}

// 7. One tree level, from the highest down: the serial builder's numbering. It numbers a node's
// internal children when it takes the node from its stack (left, then right: consecutive) and
// then empties the right child's subtree before the left child's, so a node is (number, first
// number of its children) = (n, b): with two internal children, right = (b + 1, b + 2) and left
// = (b, b + 1 + size(right)); with one, that child = (b, b + 1). A root is (0, 1). A parent sits
// on a higher level than its children, so its dispatch came first.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_lbvh_number(uint3 thread: SV_DispatchThreadID) {
  const uint id = thread.x;
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

float4 bits(float3 value, uint data) { return float4(value, asfloat(data)); }

// 8. Per internal node: its traversal node at its serial number; per bottom-level sorted
// position: its triangle.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_lbvh_emit(uint3 thread: SV_DispatchThreadID) {
  const uint index = thread.x;
  const uint internalCount = control.part.z;
  if (index < internalCount) {
    const uint id = control.part.y + index;
    const uint4 children = nodeChildren[id];
    if (children.z >= control.sizes.y) return; // a node the PLOC topology never made (it failed)
    const BvhSegment segment = segments[children.z];
    const uint outputNode = segment.range.z + nodeNumber[id].x;
    for (uint side = 0u; side < 2u; ++side) {
      const uint child = side == 1u ? children.y : children.x;
      float3 low = float3(kLbvhHuge), high = float3(-kLbvhHuge);
      uint data = kLbvhEmpty, count = 0u;
      if (child != kLbvhEmpty) {
        childBox(child, low, high);
        if ((child & kLbvhLeaf) != 0u) {
          const uint sorted = child & ~kLbvhLeaf;
          data = segment.flags.x != 0u ? records[sortedIndex[sorted]].identity.x : segment.range.w + (sorted - segment.range.x);
          count = 1u;
        } else {
          data = segment.range.z + nodeNumber[child].x;
        }
      }
      nodes[outputNode * 4u + side * 2u] = bits(low, data);
      nodes[outputNode * 4u + side * 2u + 1u] = bits(high, count);
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
  const uint outputTriangle = (segment.range.w + sorted - segment.range.x) * 3u;
  triangles[outputTriangle] = bits(buildPosition(vertices, i0), primitive);
  triangles[outputTriangle + 1u] = float4(buildPosition(vertices, i1), 0.0f);
  triangles[outputTriangle + 2u] = float4(buildPosition(vertices, i2), 0.0f);
}

// 9. One thread: the status record. A root never fitted means a tree deeper than the fit's
// iterations; that and a traversal stack overflow are depth errors, as serially.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_lbvh_finish(uint3 thread: SV_DispatchThreadID) {
  if (thread.x != 0u) return;
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
  if (result == kBvhBuildErrorNone && (unfitted || topDepth + bottomDepth > control.geometry.z)) result = kBvhBuildErrorDepth;
  BvhBuildStatus2 out;
  out.result = uint4(result, topDepth, bottomDepth, 0u);
  out.counts = uint4(control.sizes.z, control.sizes.w, control.range.x, 0u);
  out.extra = uint4(0u, 0u, iterations, 0u);
  out.cost = float4(0.0f);
  status[0] = out;
}
