// Apetrei's single-pass LBVH (2014, "Fast and Simple Agglomerative LBVH Construction"), as the
// topology and fit of the parallel LBVH builder (bvh_lbvh.slang): the same tree, built bottom-up
// in one kernel instead of Karras's per-node ranges followed by one fit dispatch per level.
//
// Over distinct sorted keys the binary radix tree is unique, so building it bottom-up gives
// Karras's tree. A thread starts at a leaf and climbs: a range [lo, hi) of sorted positions is
// the left child of the node splitting at hi - 1 when key hi shares a longer prefix with key
// hi - 1 than key lo - 1 shares with key lo, and otherwise the right child of the node splitting
// at lo - 1 (the two prefixes are never equal: the parent holds both neighbours only if it splits
// on the range's own bit). The first thread to reach a node stops; the second fits it (box,
// subtree size, height) and climbs on. Internal nodes are indexed by their split position; the
// numbering pass that follows gives the serial builder's numbers, so the published tree is the
// serial builder's byte for byte.
//
// The climb reads what other workgroups wrote, so its node buffers are coherent (they bypass
// non-coherent caches) and the arrival counter is ordered by device-scope barriers: a thread
// writes its child's record, then arrives; the second arrival reads both.
//
// Then each fitted node joins its height's level queue, as the fit's iterations queue them, so
// the numbering, emit and finish kernels run unchanged. Sequence: single (per segment), climb (per
// record), levels (one thread), scatter (per internal node).
import pt_bvh_build;

static const uint kEmpty = 0xFFFFFFFFu;
static const uint kLeaf = 0x80000000u; // child encoding: a leaf at this sorted position
static const float kHuge = 3.0e38f;
static const uint kLevels = 64u;       // the fit's iterations: the deepest tree the numbering walks

ConstantBuffer<BvhLbvhControl> control;
StructuredBuffer<BvhSegment> segments;
StructuredBuffer<BvhBuildRecord> records;
StructuredBuffer<uint> codes;       // the sorted Morton codes
StructuredBuffer<uint> sortedIndex; // the sorted record indices
globallycoherent RWStructuredBuffer<uint4> nodeChildren;
globallycoherent RWStructuredBuffer<uint> nodeParent;
globallycoherent RWStructuredBuffer<uint2> nodeSpan; // the node's range of sorted positions, [x, y)
globallycoherent RWStructuredBuffer<float4> nodeBox;
globallycoherent RWStructuredBuffer<uint> nodeSize;
globallycoherent RWStructuredBuffer<uint2> nodeNumber; // (height, slot in its level) until the numbering
RWStructuredBuffer<uint> nodeCounter;                  // arrivals, zero before the climb
RWStructuredBuffer<uint> headers;                      // per level: groups, 1, 1, count
RWStructuredBuffer<uint> levelBase;
RWStructuredBuffer<uint> entries;
RWStructuredBuffer<uint> segmentHeight;
RWStructuredBuffer<uint> error;

void fail(uint code) { InterlockedMax(error[0], code); }

int delta(uint first, uint count, uint a, int b) { return lbvhPrefix(codes, sortedIndex, records, first, count, a, b); }

// A fitted node of `height` joins that height's level queue (its slot counted now, its entry
// placed by the scatter once the levels' bases are known).
void queue(uint node, uint height) {
  uint slot;
  InterlockedAdd(headers[(height - 1u) * 4u + 3u], 1u, slot);
  nodeNumber[node] = uint2(height, slot);
}

// Box, internal-node count and height of a child: a leaf at a sorted position, or a fitted node.
void childFacts(uint child, out float3 low, out float3 high, out uint size, out uint height) {
  if ((child & kLeaf) != 0u) {
    const BvhBuildRecord record = records[sortedIndex[child & ~kLeaf]];
    low = record.low.xyz;
    high = record.high.xyz;
    size = 0u;
    height = 0u;
  } else {
    low = nodeBox[child * 2u].xyz;
    high = nodeBox[child * 2u + 1u].xyz;
    size = nodeSize[child];
    height = nodeNumber[child].x;
  }
}

// 1. Per segment of at most one record: its one node, as the Karras topology and fit make it (an
// empty segment's node has no children and an empty box).
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_apetrei_single(uint3 thread: SV_DispatchThreadID) {
  const uint g = thread.x;
  if (g >= control.sizes.y) return;
  const BvhSegment segment = segments[g];
  if (segment.range.y >= 2u) return;
  const uint node = segment.flags.w;
  float3 low = float3(kHuge), high = float3(-kHuge);
  uint left = kEmpty;
  if (segment.range.y == 1u) {
    left = kLeaf | segment.range.x;
    const BvhBuildRecord record = records[sortedIndex[segment.range.x]];
    low = record.low.xyz;
    high = record.high.xyz;
  }
  nodeChildren[node] = uint4(left, kEmpty, g, 0u);
  nodeParent[node] = kEmpty;
  nodeBox[node * 2u] = float4(low, 0.0f);
  nodeBox[node * 2u + 1u] = float4(high, 0.0f);
  nodeSize[node] = 1u;
  queue(node, 1u);
  segmentHeight[g] = 1u;
}

// 2. Per sorted record of a segment of two or more: the climb from its leaf.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_apetrei_climb(uint3 thread: SV_DispatchThreadID) {
  const uint record = thread.x;
  if (record >= control.sizes.x) return;
  const uint g = lbvhSegmentOfRecord(segments, control.sizes.y, record);
  const BvhSegment segment = segments[g];
  const uint first = segment.range.x;
  const uint count = segment.range.y;
  if (count < 2u || lbvhBatched(segment, control)) return;
  uint current = kLeaf | record;
  uint lo = record - first, hi = lo + 1u;
  for (;;) {
    if (lo == 0u && hi == count) {
      // The segment's root: an internal node (two or more records).
      nodeParent[current] = kEmpty;
      segmentHeight[g] = nodeNumber[current].x;
      return;
    }
    const bool leftChild = lo == 0u || (hi != count && delta(first, count, hi - 1u, int(hi)) > delta(first, count, lo - 1u, int(lo)));
    const uint parent = segment.flags.w + (leftChild ? hi - 1u : lo - 1u);
    if ((current & kLeaf) == 0u) nodeParent[current] = parent;
    if (leftChild) {
      nodeChildren[parent].x = current;
      nodeSpan[parent].x = lo;
    } else {
      nodeChildren[parent].y = current;
      nodeSpan[parent].y = hi;
    }
    // This child's records before the arrival; the second arrival reads both children's.
    DeviceMemoryBarrier();
    uint arrived;
    InterlockedAdd(nodeCounter[parent], 1u, arrived);
    if (arrived == 0u) return;
    DeviceMemoryBarrier();
    const uint4 children = nodeChildren[parent];
    float3 leftLow, leftHigh, rightLow, rightHigh;
    uint leftSize, leftHeight, rightSize, rightHeight;
    childFacts(children.x, leftLow, leftHigh, leftSize, leftHeight);
    childFacts(children.y, rightLow, rightHigh, rightSize, rightHeight);
    const uint height = 1u + max(leftHeight, rightHeight);
    if (height > kLevels) {
      fail(kBvhBuildErrorDepth);  // deeper than the numbering walks, as the fit reports it
      return;
    }
    nodeBox[parent * 2u] = float4(min(leftLow, rightLow), 0.0f);
    nodeBox[parent * 2u + 1u] = float4(max(leftHigh, rightHigh), 0.0f);
    nodeSize[parent] = 1u + leftSize + rightSize;
    const uint internal = ((children.x & kLeaf) == 0u ? 1u : 0u) + ((children.y & kLeaf) == 0u ? 1u : 0u);
    nodeChildren[parent].zw = uint2(g, internal);
    queue(parent, height);
    current = parent;
    const uint2 span = nodeSpan[parent];
    lo = span.x;
    hi = span.y;
  }
}

// 3. One thread: each level's first entry and dispatch size from the counts the climb queued.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_apetrei_levels(uint3 thread: SV_DispatchThreadID) {
  if (thread.x != 0u) return;
  uint running = 0u;
  for (uint level = 0u; level < kLevels; ++level) {
    const uint count = headers[level * 4u + 3u];
    headers[level * 4u] = (count + 63u) / 64u;
    levelBase[level] = running;
    running += count;
  }
  levelBase[kLevels] = running;
}

// 4. Per internal node: its entry in its level's queue.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_apetrei_scatter(uint3 thread: SV_DispatchThreadID) {
  const uint node = thread.x;
  if (node >= control.sizes.z) return;
  // A node the climb never fitted (the build failed, and says so) may hold an earlier build's
  // values: bounded, it writes nothing out of range.
  const uint2 placed = nodeNumber[node];
  if (placed.x == 0u || placed.x > kLevels) return;
  const uint entry = levelBase[placed.x - 1u] + placed.y;
  if (entry < control.sizes.z) entries[entry] = node;
}
