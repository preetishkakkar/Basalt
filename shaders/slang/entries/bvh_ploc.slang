// PLOC (Meister and Bittner 2018) topology for the parallel GPU builder: clusters in Morton order
// merge with their mutual nearest neighbour (smallest union surface area within a window of
// kPlocRadius clusters of the same segment) until each segment is one tree. Only the topology is
// built here (children, parents, internal-child counts); the LBVH's seed, fit, number and emit
// kernels (bvh_lbvh.slang) then fit, number and publish it, so its layout, update paths and status
// are the LBVH's. Fence-free: while clusters are many, one dispatch per step (neighbours with each
// block's kept-flag scan, the block sums' scan with the next iteration's sizes, compact), a number
// of iterations the host sizes from the record count; then one workgroup finishes (iterations with
// workgroup barriers, in shared memory once at most kPlocShared clusters remain, ending when every
// segment is one tree: nearest neighbours while they merge, then pairing clusters by position,
// which ends every segment as one tree whatever the geometry). A segment's internal nodes are its
// LBVH range (flags.w onwards), claimed in merge order by a per-segment cursor: the ids differ run
// to run, the topology (and so the numbered output) does not, and a TLAS rebuild by the LBVH finds
// the TLAS's nodes where it writes them.
// Clusters are (kPlocLeaf | sorted position or internal node, segment) in Morton order, each with
// its box (low, high) alongside, in the two halves of one buffer each (an iteration reads one half
// and writes the other).
import pt_bvh_build;

static const uint kPlocEmpty = 0xFFFFFFFFu; // as bvh_lbvh.slang's kLbvhEmpty
static const uint kPlocLeaf = 0x80000000u;  // as kLbvhLeaf: a leaf at this sorted position
static const uint kPlocRadius = 8u;
static const uint kPlocGroup = 128u; // clusters per workgroup (every PLOC kernel)
// A workgroup's clusters with the halo their neighbours need (kPlocHalo = 2 radii: the kept flag of
// a cluster needs its neighbour's neighbour).
static const uint kPlocHalo = 2u * kPlocRadius;
static const uint kPlocTile = kPlocGroup + 2u * kPlocHalo;
static const uint kPlocShared = 256u;          // clusters the finish holds in shared memory (2 a thread)
static const uint kPlocDeviceGreedy = 16u;     // the finish's device-memory iterations while more remain:
static const uint kPlocDeviceIterations = 32u; // nearest neighbours, then pairing by position
static const uint kPlocFinishGreedy = 48u;     // in shared memory, at most: nearest neighbours,
static const uint kPlocFinishIterations = 96u; // then pairing by position (18 end 256 clusters)

// An iteration's sizes (one slot per parity: an iteration reads its own and writes the other's):
// [0..3] per-cluster dispatch (groups of 128, 1, 1, clusters); [4..6] reserved; [7] iterations so
// far that merged; [8] the slot's parity (the half of the clusters it describes).
static const uint kPlocClusters = 3u;
static const uint kPlocIterations = 7u;
static const uint kPlocParity = 8u;

ConstantBuffer<BvhLbvhControl> control;
StructuredBuffer<BvhSegment> segments;
StructuredBuffer<BvhBuildRecord> records;
StructuredBuffer<uint> sortedIndex;
RWStructuredBuffer<uint4> nodeChildren;
RWStructuredBuffer<uint> nodeParent;
RWStructuredBuffer<uint> segmentNext;
RWStructuredBuffer<uint2> clusters;
RWStructuredBuffer<float4> boxes;
RWStructuredBuffer<uint2> nextClusters;
RWStructuredBuffer<float4> nextBoxes;
RWStructuredBuffer<uint> neighbour;
RWStructuredBuffer<uint> kept;
RWStructuredBuffer<uint> blockSums;
RWStructuredBuffer<uint> current;
RWStructuredBuffer<uint> next;

groupshared uint partial[kPlocGroup];
groupshared float4 lows[kPlocTile];
groupshared float4 highs[kPlocTile];
groupshared uint segmentOf[kPlocTile];
groupshared uint nearest[kPlocTile]; // tile positions; valid for the middle kPlocGroup + 2 radii
groupshared float4 tileLow[kPlocShared];
groupshared float4 tileHigh[kPlocShared];
groupshared uint2 tileCluster[kPlocShared];
groupshared uint tileNeighbour[kPlocShared];
groupshared uint stopSlot;

float plocArea(float3 low, float3 high) {
  const float3 d = high - low;
  return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}

// Whether candidate j (union area `area`) beats the best so far for cluster id: the smaller area,
// then the smaller index xor (a total order on pairs, so the smallest pair of a segment is always
// mutual, and equal boxes pair off as (2k, 2k + 1) rather than all choosing one).
bool plocBetter(uint id, uint j, float area, uint best, float bestArea) {
  return best == id || area < bestArea || (area == bestArea && (j ^ id) < (best ^ id));
}

// A cluster is kept unless it merges into its lower-index mutual neighbour.
uint plocKept(uint id) {
  const uint j = neighbour[id];
  return j < id && neighbour[j] == id ? 0u : 1u;
}

// The internal node of the lower (left) and higher (right) clusters of a mutual pair: its children
// in cluster order and the children's parent. Returns the node.
uint plocNode(uint left, uint right, uint segment) {
  uint node;
  InterlockedAdd(segmentNext[segment], 1u, node);
  uint internal = 0u;
  if ((left & kPlocLeaf) == 0u) {
    nodeParent[left] = node;
    internal += 1u;
  }
  if ((right & kPlocLeaf) == 0u) {
    nodeParent[right] = node;
    internal += 1u;
  }
  nodeChildren[node] = uint4(left, right, segment, internal);
  return node;
}

// The partner of cluster id when pairing by position: (2k, 2k + 1) on even iterations, (2k + 1,
// 2k + 2) on odd ones.
uint plocPairedWith(uint id, uint iteration) {
  if ((iteration & 1u) == 0u) return id ^ 1u;
  return id == 0u ? 0u : ((id - 1u) ^ 1u) + 1u;
}

// Inclusive Hillis-Steele scan of partial[0..127] in place (bvh_sort.slang's sortScanThreads).
void plocScanThreads(uint lane) {
  for (uint offset = 1u; offset < kPlocGroup; offset <<= 1u) {
    uint add = 0u;
    if (lane >= offset) add = partial[lane - offset];
    GroupMemoryBarrierWithGroupSync();
    partial[lane] = partial[lane] + add;
    GroupMemoryBarrierWithGroupSync();
  }
}

// Lane 0's value for every lane of the workgroup: a store, a barrier, a load, a barrier. Loops
// that exit on it keep their barriers under workgroup-uniform control.
uint broadcastFirst(uint value, uint lane) {
  if (lane == 0u) stopSlot = value;
  GroupMemoryBarrierWithGroupSync();
  const uint result = stopSlot;
  GroupMemoryBarrierWithGroupSync();
  return result;
}

// 1. Per sorted record: its leaf cluster and box (in the first half).
[shader("compute")]
[numthreads(128, 1, 1)]
void bvh_ploc_init(uint3 thread: SV_DispatchThreadID) {
  const uint id = thread.x;
  if (id >= control.sizes.x) return;
  const BvhBuildRecord record = records[sortedIndex[id]];
  clusters[id] = uint2(kPlocLeaf | id, record.identity.z);
  boxes[id * 2u] = record.low;
  boxes[id * 2u + 1u] = record.high;
}

// 2. Per segment: its node cursor at its first internal node; a segment of at most one record gets
// its one node there, as the LBVH makes it (the leaf and an empty child, or two empty children),
// and never merges.
[shader("compute")]
[numthreads(128, 1, 1)]
void bvh_ploc_single(uint3 thread: SV_DispatchThreadID) {
  const uint id = thread.x;
  if (id >= control.sizes.y) return;
  const BvhSegment segment = segments[id];
  segmentNext[id] = segment.flags.w;
  if (segment.range.y > 1u) return;
  const uint left = segment.range.y == 1u ? kPlocLeaf | segment.range.x : kPlocEmpty;
  nodeChildren[segment.flags.w] = uint4(left, kPlocEmpty, id, 0u);
}

// 3. Per cluster: its nearest neighbour, the cluster of the same segment within kPlocRadius whose
// union with it has the smallest surface area (plocBetter), itself if none; then, per workgroup of
// 128 clusters (a block), the exclusive scan of their kept flags and the block's total. The kept
// flag needs the neighbour's neighbour, so neighbours are found for the halo of kPlocRadius
// clusters on either side as well (from boxes kPlocHalo out).
[shader("compute")]
[numthreads(128, 1, 1)]
void bvh_ploc_neighbours(uint3 thread: SV_DispatchThreadID, uint lane: SV_GroupIndex, uint3 groupId: SV_GroupID) {
  const uint id = thread.x;
  const uint group = groupId.x;
  const uint count = current[kPlocClusters];
  const int first = int(group * kPlocGroup) - int(kPlocHalo);
  for (uint k = lane; k < kPlocTile; k += kPlocGroup) {
    const int j = first + int(k);
    uint segment = kPlocEmpty;
    float4 low = float4(0.0f), high = float4(0.0f);
    if (j >= 0 && uint(j) < count) {
      segment = clusters[j].y;
      low = boxes[uint(j) * 2u];
      high = boxes[uint(j) * 2u + 1u];
    }
    lows[k] = low;
    highs[k] = high;
    segmentOf[k] = segment;
  }
  GroupMemoryBarrierWithGroupSync();
  // Tile positions kPlocRadius .. kPlocTile - kPlocRadius - 1: this block and a radius each side.
  for (uint k = kPlocRadius + lane; k < kPlocTile - kPlocRadius; k += kPlocGroup) {
    const uint segment = segmentOf[k];
    uint best = k;
    if (segment != kPlocEmpty) {
      const float3 low = lows[k].xyz, high = highs[k].xyz;
      const uint self = uint(first + int(k));
      uint bestIndex = self;
      float bestArea = 0.0f;
      for (uint c = k - kPlocRadius; c <= k + kPlocRadius; ++c) {
        if (c == k || segmentOf[c] != segment) continue;
        const float area = plocArea(min(low, lows[c].xyz), max(high, highs[c].xyz));
        const uint j = uint(first + int(c));
        if (plocBetter(self, j, area, bestIndex, bestArea)) {
          best = c;
          bestIndex = j;
          bestArea = area;
        }
      }
    }
    nearest[k] = best;
  }
  GroupMemoryBarrierWithGroupSync();
  const uint self = kPlocHalo + lane;
  const uint j = nearest[self];
  const uint keep = id < count && !(j < self && nearest[j] == self) ? 1u : 0u;
  if (id < count) neighbour[id] = uint(first + int(j));
  partial[lane] = keep;
  GroupMemoryBarrierWithGroupSync();
  plocScanThreads(lane);
  if (id < count) kept[id] = partial[lane] - keep;
  if (lane == 0u) blockSums[group] = partial[kPlocGroup - 1u];
}

// 4. One workgroup: the block sums (one per 128 clusters) to exclusive offsets, each thread summing
// a contiguous run of them; their total is the next iteration's cluster count, from which its
// sizes follow.
[shader("compute")]
[numthreads(128, 1, 1)]
void bvh_ploc_scan_sums(uint lane: SV_GroupIndex) {
  const uint count = current[kPlocClusters];
  const uint blocks = (count + kPlocGroup - 1u) / kPlocGroup;
  const uint per = (blocks + kPlocGroup - 1u) / kPlocGroup;
  const uint begin = min(lane * per, blocks), end = min(begin + per, blocks);
  uint total = 0u;
  for (uint b = begin; b < end; ++b) total += blockSums[b];
  partial[lane] = total;
  GroupMemoryBarrierWithGroupSync();
  plocScanThreads(lane);
  uint running = partial[lane] - total;
  for (uint b = begin; b < end; ++b) {
    const uint sum = blockSums[b];
    blockSums[b] = running;
    running += sum;
  }
  if (lane == 0u) {
    const uint keptTotal = partial[kPlocGroup - 1u];
    next[0] = (keptTotal + kPlocGroup - 1u) / kPlocGroup;
    next[1] = 1u;
    next[2] = 1u;
    next[kPlocClusters] = keptTotal;
    next[kPlocIterations] = current[kPlocIterations] + (keptTotal < count ? 1u : 0u);
  }
}

// 5. Per cluster: the next clusters and boxes in the same order. A merging cluster makes an
// internal node of its segment in its place, with the union box; a merged partner disappears.
[shader("compute")]
[numthreads(128, 1, 1)]
void bvh_ploc_compact(uint3 thread: SV_DispatchThreadID) {
  const uint id = thread.x;
  if (id >= current[kPlocClusters]) return;
  const uint j = neighbour[id];
  const bool mutual = j != id && neighbour[j] == id;
  if (mutual && j < id) return; // merged into its partner
  uint2 cluster = clusters[id];
  float4 low = boxes[id * 2u], high = boxes[id * 2u + 1u];
  if (mutual) {
    cluster.x = plocNode(cluster.x, clusters[j].x, cluster.y);
    low = min(low, boxes[j * 2u]);
    high = max(high, boxes[j * 2u + 1u]);
  }
  const uint slot = kept[id] + blockSums[id / kPlocGroup];
  nextClusters[slot] = cluster;
  nextBoxes[slot * 2u] = low;
  nextBoxes[slot * 2u + 1u] = high;
}

// 6. One workgroup: the remaining iterations, from the half and count of the current slot; the
// final count and merging iterations back into it. While more than kPlocShared clusters remain,
// iterations run from device memory (the first kPlocDeviceGreedy merge mutual nearest neighbours,
// the rest pair clusters by position so that the count falls below kPlocShared); then the clusters
// and their boxes move into shared memory: nearest-neighbour iterations until one merges nothing
// (every segment is then one tree, since each unfinished segment's smallest pair is mutual) or
// kPlocFinishGreedy have run, then pairing by position, (2k, 2k + 1) and (2k + 1, 2k + 2)
// alternately, until two in a row merge nothing: a segment of c > 1 clusters has a pair in one of
// the two, so it loses at least one every two iterations and about half of them. Every exit is a
// broadcast from lane 0, so the loops' barriers stay under workgroup-uniform control.
[shader("compute")]
[numthreads(128, 1, 1)]
void bvh_ploc_finish(uint lane: SV_GroupIndex) {
  const uint stride = (max(control.sizes.x, 1u) + 31u) & ~31u; // a half: 256-byte aligned, as the host's
  uint from = current[kPlocParity] * stride, to = stride - from;
  uint count = current[kPlocClusters];
  uint merged = current[kPlocIterations];

  // From device memory, while the clusters do not fit the tile.
  for (uint iteration = 0u; iteration < kPlocDeviceIterations; ++iteration) {
    if (broadcastFirst(count <= kPlocShared ? 1u : 0u, lane) != 0u) break;
    for (uint id = lane; id < count; id += kPlocGroup) {
      const uint segment = clusters[from + id].y;
      uint best = id;
      if (iteration < kPlocDeviceGreedy) {
        const float3 low = boxes[(from + id) * 2u].xyz, high = boxes[(from + id) * 2u + 1u].xyz;
        float bestArea = 0.0f;
        const uint first = id > kPlocRadius ? id - kPlocRadius : 0u;
        const uint last = min(id + kPlocRadius, count - 1u);
        for (uint j = first; j <= last; ++j) {
          if (j == id || clusters[from + j].y != segment) continue;
          const float area = plocArea(min(low, boxes[(from + j) * 2u].xyz), max(high, boxes[(from + j) * 2u + 1u].xyz));
          if (plocBetter(id, j, area, best, bestArea)) {
            best = j;
            bestArea = area;
          }
        }
      } else {
        const uint j = plocPairedWith(id, iteration);
        if (j < count && clusters[from + j].y == segment) best = j;
      }
      neighbour[id] = best;
    }
    DeviceMemoryBarrierWithGroupSync();
    // This thread's contiguous range of clusters: its kept count, scanned across the group.
    const uint per = (count + kPlocGroup - 1u) / kPlocGroup;
    const uint begin = min(lane * per, count), end = min(begin + per, count);
    uint total = 0u;
    for (uint id = begin; id < end; ++id) total += plocKept(id);
    partial[lane] = total;
    GroupMemoryBarrierWithGroupSync();
    plocScanThreads(lane);
    uint slot = lane > 0u ? partial[lane - 1u] : 0u;
    const uint keptCount = partial[kPlocGroup - 1u];
    for (uint id = begin; id < end; ++id) {
      const uint j = neighbour[id];
      const bool mutual = j != id && neighbour[j] == id;
      if (mutual && j < id) continue; // merged into its partner
      uint2 cluster = clusters[from + id];
      float4 low = boxes[(from + id) * 2u], high = boxes[(from + id) * 2u + 1u];
      if (mutual) {
        cluster.x = plocNode(cluster.x, clusters[from + j].x, cluster.y);
        low = min(low, boxes[(from + j) * 2u]);
        high = max(high, boxes[(from + j) * 2u + 1u]);
      }
      clusters[to + slot] = cluster;
      boxes[(to + slot) * 2u] = low;
      boxes[(to + slot) * 2u + 1u] = high;
      slot += 1u;
    }
    if (keptCount < count) merged += 1u;
    count = keptCount;
    from = to;
    to = stride - from;
    AllMemoryBarrierWithGroupSync();
  }

  // Into the tile: thread t holds clusters 2t and 2t + 1. Clusters that still do not fit stay as
  // they are (the host then reports the build unfinished).
  const bool fits = count <= kPlocShared;
  for (uint q = 0u; q < 2u; ++q) {
    const uint id = lane * 2u + q;
    if (fits && id < count) {
      tileCluster[id] = clusters[from + id];
      tileLow[id] = boxes[(from + id) * 2u];
      tileHigh[id] = boxes[(from + id) * 2u + 1u];
    }
  }
  GroupMemoryBarrierWithGroupSync();
  bool pairing = false;
  uint pairingRound = 0u, idle = 0u;
  for (uint iteration = 0u; iteration < kPlocFinishIterations; ++iteration) {
    if (broadcastFirst(!fits || idle >= 2u ? 1u : 0u, lane) != 0u) break;
    for (uint q = 0u; q < 2u; ++q) {
      const uint id = lane * 2u + q;
      if (id >= count) continue;
      const uint segment = tileCluster[id].y;
      uint best = id;
      if (!pairing) {
        const float3 low = tileLow[id].xyz, high = tileHigh[id].xyz;
        float bestArea = 0.0f;
        const uint first = id > kPlocRadius ? id - kPlocRadius : 0u;
        const uint last = min(id + kPlocRadius, count - 1u);
        for (uint j = first; j <= last; ++j) {
          if (j == id || tileCluster[j].y != segment) continue;
          const float area = plocArea(min(low, tileLow[j].xyz), max(high, tileHigh[j].xyz));
          if (plocBetter(id, j, area, best, bestArea)) {
            best = j;
            bestArea = area;
          }
        }
      } else {
        const uint j = plocPairedWith(id, pairingRound);
        if (j < count && tileCluster[j].y == segment) best = j;
      }
      tileNeighbour[id] = best;
    }
    GroupMemoryBarrierWithGroupSync();
    // This thread's two clusters: the kept ones (merged if the lower of a mutual pair) read into
    // registers, their count scanned, then written to their new slots (the scan's barriers
    // separate every read from every write).
    bool keep[2] = {};
    uint2 outCluster[2] = {};
    float4 outLow[2] = {};
    float4 outHigh[2] = {};
    for (uint q = 0u; q < 2u; ++q) {
      const uint id = lane * 2u + q;
      if (id >= count) continue;
      const uint j = tileNeighbour[id];
      const bool mutual = j != id && tileNeighbour[j] == id;
      if (mutual && j < id) continue; // merged into its partner
      keep[q] = true;
      uint2 cluster = tileCluster[id];
      float4 low = tileLow[id], high = tileHigh[id];
      if (mutual) {
        cluster.x = plocNode(cluster.x, tileCluster[j].x, cluster.y);
        low = min(low, tileLow[j]);
        high = max(high, tileHigh[j]);
      }
      outCluster[q] = cluster;
      outLow[q] = low;
      outHigh[q] = high;
    }
    partial[lane] = (keep[0] ? 1u : 0u) + (keep[1] ? 1u : 0u);
    GroupMemoryBarrierWithGroupSync();
    plocScanThreads(lane);
    uint slot = lane > 0u ? partial[lane - 1u] : 0u;
    const uint keptCount = partial[kPlocGroup - 1u];
    for (uint q = 0u; q < 2u; ++q) {
      if (!keep[q]) continue;
      tileCluster[slot] = outCluster[q];
      tileLow[slot] = outLow[q];
      tileHigh[slot] = outHigh[q];
      slot += 1u;
    }
    const bool mergedAny = keptCount < count;
    if (mergedAny) merged += 1u;
    count = keptCount;
    if (pairing) {
      pairingRound += 1u;
      idle = mergedAny ? 0u : idle + 1u;
    } else if (!mergedAny || iteration + 1u >= kPlocFinishGreedy) {
      pairing = true;
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (lane == 0u) {
    current[kPlocClusters] = count;
    current[kPlocIterations] = merged;
  }
}
