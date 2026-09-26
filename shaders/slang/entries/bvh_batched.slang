// HIPRT's batched LBVH (Meister et al. 2024, as HIP-BVH-Construction's BatchedBuildKernelLbvh):
// one workgroup builds a whole small bottom level in shared memory. It bounds the triangles and
// the level, computes their Morton codes, sorts them stably by a one-bit radix split per code bit,
// and climbs the tree bottom-up as the single-pass LBVH does (bvh_apetrei.slang), with the arrival
// counters and node records in shared memory. The result goes out in the formats the parallel
// LBVH's later kernels read: the records and the level's box (the TLAS's instances read it), node
// records queued by height, and the sort's keys as each record's rank in the level, so the global
// sort, which the larger levels and the TLAS need, places the level's records where the workgroup
// sorted them. The same box, Morton codes and stable order as the global kernels give the serial
// builder's tree byte for byte.
//
// A level is batched when it has from 2 to kBatch records and control.range.w is set (the build's
// batched control slot): the extents, Morton and climb kernels then skip it.
import pt_bvh_build;

static const uint kBatch = 256u;       // one thread per record
static const uint kEmpty = 0xFFFFFFFFu;
static const uint kLeaf = 0x80000000u;
static const uint kLevels = 64u;
static const uint kCodeBits = 30u;     // morton3's codes

ConstantBuffer<BvhLbvhControl> control;
StructuredBuffer<BvhSegment> segments;
StructuredBuffer<TraceInstance> instances;
StructuredBuffer<uint> indices;
StructuredBuffer<float> vertices;
RWStructuredBuffer<BvhBuildRecord> records;
StructuredBuffer<BvhReference> references;  // split clipping's, one per bottom-level record (control.source.x)
RWStructuredBuffer<uint> segmentBox;
RWStructuredBuffer<uint> keysLo;
RWStructuredBuffer<uint> keysHi;
RWStructuredBuffer<uint> values;
RWStructuredBuffer<uint4> nodeChildren;
RWStructuredBuffer<uint> nodeParent;
RWStructuredBuffer<float4> nodeBox;
RWStructuredBuffer<uint> nodeSize;
RWStructuredBuffer<uint2> nodeNumber; // (height, slot in its level) until the numbering
RWStructuredBuffer<uint> headers;
RWStructuredBuffer<uint> segmentHeight;
RWStructuredBuffer<uint> error;

groupshared uint sharedBox[6];              // ordered-bit words, as segmentBox
groupshared float3 leafLow[kBatch];         // by local record index
groupshared float3 leafHigh[kBatch];
groupshared uint sortedCode[kBatch];        // by sorted position
groupshared uint sortedLocal[kBatch];       // the local record at each sorted position
groupshared uint scan[kBatch];
// Internal node s splits between sorted positions s and s + 1.
groupshared uint nodeCounter[kBatch];
groupshared uint2 nodeSpanShared[kBatch];   // [x, y) of sorted positions
groupshared uint2 nodeChild[kBatch];        // kLeaf | sorted position, or a node
groupshared uint nodeParentShared[kBatch];
groupshared float3 nodeLow[kBatch];
groupshared float3 nodeHigh[kBatch];
groupshared uint nodeSizeShared[kBatch];
groupshared uint nodeHeight[kBatch];

void fail(uint code) { InterlockedMax(error[0], code); }

// The serial builder's commonPrefix over (Morton code, index in the level) at sorted positions a
// and b of a level of `count` records; the index is the local record index.
int delta(uint count, uint a, int b) {
  if (b < 0 || uint(b) >= count) return -1;
  const uint ma = sortedCode[a], mb = sortedCode[uint(b)];
  if (ma != mb) return int(leadingZeros(ma ^ mb));
  const uint ia = sortedLocal[a], ib = sortedLocal[uint(b)];
  if (ia == ib) return 64;
  return 32 + int(leadingZeros(ia ^ ib));
}

void childFacts(uint child, out float3 low, out float3 high, out uint size, out uint height) {
  if ((child & kLeaf) != 0u) {
    const uint local = sortedLocal[child & ~kLeaf];
    low = leafLow[local];
    high = leafHigh[local];
    size = 0u;
    height = 0u;
  } else {
    low = nodeLow[child];
    high = nodeHigh[child];
    size = nodeSizeShared[child];
    height = nodeHeight[child];
  }
}

// An inclusive prefix sum of `value` over the workgroup (Hillis-Steele, in `scan`).
uint inclusiveSum(uint lane, uint value) {
  scan[lane] = value;
  GroupMemoryBarrierWithGroupSync();
  for (uint offset = 1u; offset < kBatch; offset <<= 1u) {
    const uint add = lane >= offset ? scan[lane - offset] : 0u;
    GroupMemoryBarrierWithGroupSync();
    scan[lane] += add;
    GroupMemoryBarrierWithGroupSync();
  }
  return scan[lane];
}

// One workgroup per bottom level; levels that are not batched return at once (uniformly).
[shader("compute")]
[numthreads(256, 1, 1)]
void bvh_batched_build(uint lane: SV_GroupIndex, uint3 group: SV_GroupID) {
  const uint g = group.x;
  if (g + 1u >= control.sizes.y || control.range.w == 0u) return;
  const BvhSegment segment = segments[g];
  const uint count = segment.range.y;
  if (count < 2u || count > control.range.w) return;
  const uint first = segment.range.x;
  const bool active = lane < count;

  // Bounds: each record's triangle, and the level's box through ordered-bit atomics (exact, as
  // the extents kernel's).
  if (lane < 6u) sharedBox[lane] = lane < 3u ? 0xFFFFFFFFu : 0u;
  GroupMemoryBarrierWithGroupSync();
  // The record's triangle, or with split clipping its reference's.
  const bool referenced = control.source.x != 0u;
  BvhReference reference = {};
  if (referenced && active) reference = references[first + lane];
  const uint primitive = referenced ? asuint(reference.low.w) : lane;
  if (active) {
    const TraceInstance instance = instances[segment.flags.y];
    const uint base = instance.firstIndex + primitive * 3u;
    float3 v0 = float3(0.0f), v1 = float3(0.0f), v2 = float3(0.0f);
    if (base + 2u >= control.geometry.x) {
      fail(kBvhBuildErrorGeometry);
    } else {
      const uint i0 = indices[base] + instance.vertexOffset;
      const uint i1 = indices[base + 1u] + instance.vertexOffset;
      const uint i2 = indices[base + 2u] + instance.vertexOffset;
      if (max(i0, max(i1, i2)) >= control.geometry.y) {
        fail(kBvhBuildErrorGeometry);
      } else {
        v0 = buildPosition(vertices, i0);
        v1 = buildPosition(vertices, i1);
        v2 = buildPosition(vertices, i2);
        if (!all(isfinite(v0)) || !all(isfinite(v1)) || !all(isfinite(v2))) fail(kBvhBuildErrorNonFinite);
      }
    }
    const float3 low = referenced ? reference.low.xyz : min(v0, min(v1, v2));
    const float3 high = referenced ? reference.high.xyz : max(v0, max(v1, v2));
    leafLow[lane] = low;
    leafHigh[lane] = high;
    InterlockedMin(sharedBox[0], ptOrderedBits(low.x));
    InterlockedMin(sharedBox[1], ptOrderedBits(low.y));
    InterlockedMin(sharedBox[2], ptOrderedBits(low.z));
    InterlockedMax(sharedBox[3], ptOrderedBits(high.x));
    InterlockedMax(sharedBox[4], ptOrderedBits(high.y));
    InterlockedMax(sharedBox[5], ptOrderedBits(high.z));
  }
  GroupMemoryBarrierWithGroupSync();
  if (lane < 6u) segmentBox[g * 6u + lane] = sharedBox[lane];
  const float3 levelLow = float3(ptOrderedFloat(sharedBox[0]), ptOrderedFloat(sharedBox[1]), ptOrderedFloat(sharedBox[2]));
  const float3 levelHigh = float3(ptOrderedFloat(sharedBox[3]), ptOrderedFloat(sharedBox[4]), ptOrderedFloat(sharedBox[5]));

  // The record and its Morton code, as the extents and Morton kernels write them.
  uint code = 0xFFFFFFFFu;  // the inactive lanes sort after every code
  if (active) {
    code = morton3((leafLow[lane] + leafHigh[lane]) * 0.5f, levelLow, levelHigh);
    BvhBuildRecord record;
    record.identity = uint4(primitive, code, g, lane);
    record.low = float4(leafLow[lane], 0.0f);
    record.high = float4(leafHigh[lane], 0.0f);
    records[first + lane] = record;
  }

  // The stable sort: one split per code bit, zeros first, each side in its earlier order. The
  // inactive lanes' all-ones keys stay behind the active ones.
  uint key = code, local = lane;
  for (uint bit = 0u; bit < kCodeBits; ++bit) {
    const uint zero = ((key >> bit) & 1u) == 0u ? 1u : 0u;
    const uint zerosThrough = inclusiveSum(lane, zero);
    const uint zeros = scan[kBatch - 1u];
    const uint position = zero != 0u ? zerosThrough - 1u : zeros + lane - zerosThrough;
    GroupMemoryBarrierWithGroupSync();
    sortedCode[position] = key;
    sortedLocal[position] = local;
    GroupMemoryBarrierWithGroupSync();
    key = sortedCode[lane];
    local = sortedLocal[lane];
  }
  // The global sort's keys: (level, rank) moves each record to the position sorted here.
  if (active) {
    keysLo[first + local] = lane;
    keysHi[first + local] = g;
    values[first + local] = first + local;
  }

  // The climb over sorted positions, as the single-pass LBVH's but in shared memory.
  nodeCounter[lane] = 0u;
  GroupMemoryBarrierWithGroupSync();
  if (active) {
    uint current = kLeaf | lane;
    uint lo = lane, hi = lane + 1u;
    for (;;) {
      if (lo == 0u && hi == count) {
        nodeParentShared[current] = kEmpty;
        break;
      }
      const bool leftChild = lo == 0u || (hi != count && delta(count, hi - 1u, int(hi)) > delta(count, lo - 1u, int(lo)));
      const uint parent = leftChild ? hi - 1u : lo - 1u;
      if ((current & kLeaf) == 0u) nodeParentShared[current] = parent;
      if (leftChild) {
        nodeChild[parent].x = current;
        nodeSpanShared[parent].x = lo;
      } else {
        nodeChild[parent].y = current;
        nodeSpanShared[parent].y = hi;
      }
      GroupMemoryBarrier();
      uint arrived;
      InterlockedAdd(nodeCounter[parent], 1u, arrived);
      if (arrived == 0u) break;
      GroupMemoryBarrier();
      const uint2 children = nodeChild[parent];
      float3 leftLow, leftHigh, rightLow, rightHigh;
      uint leftSize, leftHeight, rightSize, rightHeight;
      childFacts(children.x, leftLow, leftHigh, leftSize, leftHeight);
      childFacts(children.y, rightLow, rightHigh, rightSize, rightHeight);
      nodeLow[parent] = min(leftLow, rightLow);
      nodeHigh[parent] = max(leftHigh, rightHigh);
      nodeSizeShared[parent] = 1u + leftSize + rightSize;
      nodeHeight[parent] = 1u + max(leftHeight, rightHeight);
      current = parent;
      const uint2 span = nodeSpanShared[parent];
      lo = span.x;
      hi = span.y;
    }
  }
  GroupMemoryBarrierWithGroupSync();

  // The level's nodes, in the parallel LBVH's records and level queues.
  if (lane + 1u < count) {
    const uint node = segment.flags.w + lane;
    const uint2 children = nodeChild[lane];
    const uint left = (children.x & kLeaf) != 0u ? kLeaf | (first + (children.x & ~kLeaf)) : segment.flags.w + children.x;
    const uint right = (children.y & kLeaf) != 0u ? kLeaf | (first + (children.y & ~kLeaf)) : segment.flags.w + children.y;
    const uint internal = ((children.x & kLeaf) == 0u ? 1u : 0u) + ((children.y & kLeaf) == 0u ? 1u : 0u);
    nodeChildren[node] = uint4(left, right, g, internal);
    const uint parent = nodeParentShared[lane];
    nodeParent[node] = parent == kEmpty ? kEmpty : segment.flags.w + parent;
    nodeBox[node * 2u] = float4(nodeLow[lane], 0.0f);
    nodeBox[node * 2u + 1u] = float4(nodeHigh[lane], 0.0f);
    nodeSize[node] = nodeSizeShared[lane];
    const uint height = nodeHeight[lane];
    if (height > kLevels) {
      fail(kBvhBuildErrorDepth);
    } else {
      uint slot;
      InterlockedAdd(headers[(height - 1u) * 4u + 3u], 1u, slot);
      nodeNumber[node] = uint2(height, slot);
    }
    if (parent == kEmpty) segmentHeight[g] = height;
  }
}
