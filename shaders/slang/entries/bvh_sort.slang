// Stable LSD radix sort of (keyLo, keyHi, value) triples, 8 bits per pass, for the parallel GPU
// builders: key = (segment, Morton code), value = record index. One pass is five dispatches:
// per-tile digit histograms (digit-major), a three-step exclusive scan of them (block sums, one
// workgroup scanning the block sums, blocks scanned and offset) giving each (digit, tile) its
// global position, and a scatter that sorts each tile stably by the digit in shared memory before
// writing it out. Tiles and scan blocks hold 1024 entries (128 threads x 8). No cross-workgroup
// communication inside a dispatch.
import pt_bvh_build;

static const uint kSortThreads = 128u;
static const uint kSortPerThread = 8u;
static const uint kSortTile = 1024u;

ConstantBuffer<BvhSortControl> control;
StructuredBuffer<uint> keysLo;
StructuredBuffer<uint> keysHi;
StructuredBuffer<uint> values;
RWStructuredBuffer<uint> histogram;
RWStructuredBuffer<uint> blockSums;
RWStructuredBuffer<uint> outLo;
RWStructuredBuffer<uint> outHi;
RWStructuredBuffer<uint> outValues;

groupshared uint counts[256];
groupshared uint partial[kSortThreads];
groupshared uint tileA[kSortTile];
groupshared uint tileB[kSortTile];
groupshared uint digitStart[256];

// The pass's digit: 8 bits of the low (key word 0) or high key word at shift.
uint sortDigit(uint lo, uint hi, uint4 pass) {
  const uint key = pass.w != 0u ? hi : lo;
  return (key >> pass.z) & 255u;
}

// histogram[digit * tiles + tile] = keys of this tile with that digit.
[shader("compute")]
[numthreads(128, 1, 1)]
void bvh_sort_histogram(uint lane: SV_GroupIndex, uint3 group: SV_GroupID) {
  const uint tile = group.x;
  counts[lane] = 0u;
  counts[lane + kSortThreads] = 0u;
  GroupMemoryBarrierWithGroupSync();
  const uint count = control.sizes.x;
  for (uint k = 0u; k < kSortPerThread; ++k) {
    const uint index = tile * kSortTile + k * kSortThreads + lane;
    if (index < count) InterlockedAdd(counts[sortDigit(keysLo[index], keysHi[index], control.pass)], 1u);
  }
  GroupMemoryBarrierWithGroupSync();
  histogram[lane * control.sizes.y + tile] = counts[lane];
  histogram[(lane + kSortThreads) * control.sizes.y + tile] = counts[lane + kSortThreads];
}

// Inclusive Hillis-Steele scan of partial[0..127] in place; every thread must call it.
void sortScanThreads(uint lane) {
  for (uint offset = 1u; offset < kSortThreads; offset <<= 1u) {
    uint add = 0u;
    if (lane >= offset) add = partial[lane - offset];
    GroupMemoryBarrierWithGroupSync();
    partial[lane] = partial[lane] + add;
    GroupMemoryBarrierWithGroupSync();
  }
}

// Exclusive scan of the 1024-entry block at data[block * 1024] (entries past length count as zero
// and are not written), plus offset; returns the block's total. Every thread must call it.
uint sortScanBlock(RWStructuredBuffer<uint> data, uint length, uint block, uint offset, uint lane, bool write) {
  uint entries[8] = {};
  uint running[8] = {};
  uint total = 0u;
  const uint base = block * kSortTile + lane * kSortPerThread;
  for (uint k = 0u; k < kSortPerThread; ++k) {
    if (base + k < length) entries[k] = data[base + k];
    running[k] = total;
    total += entries[k];
  }
  partial[lane] = total;
  GroupMemoryBarrierWithGroupSync();
  sortScanThreads(lane);
  uint prefix = offset;
  if (lane > 0u) prefix += partial[lane - 1u];
  const uint blockTotal = partial[kSortThreads - 1u];
  if (write)
    for (uint k = 0u; k < kSortPerThread; ++k)
      if (base + k < length) data[base + k] = prefix + running[k];
  return blockTotal;
}

// Block b of the histogram: its total into blockSums[b].
[shader("compute")]
[numthreads(128, 1, 1)]
void bvh_sort_reduce(uint lane: SV_GroupIndex, uint3 group: SV_GroupID) {
  const uint total = sortScanBlock(histogram, 256u * control.sizes.y, group.x, 0u, lane, false);
  if (lane == 0u) blockSums[group.x] = total;
}

// One workgroup: the block sums (at most 1024 blocks) to exclusive offsets, in place.
[shader("compute")]
[numthreads(128, 1, 1)]
void bvh_sort_blocks(uint lane: SV_GroupIndex) { sortScanBlock(blockSums, control.sizes.z, 0u, 0u, lane, true); }

// Block b of the histogram to exclusive global offsets.
[shader("compute")]
[numthreads(128, 1, 1)]
void bvh_sort_apply(uint lane: SV_GroupIndex, uint3 group: SV_GroupID) {
  sortScanBlock(histogram, 256u * control.sizes.y, group.x, blockSums[group.x], lane, true);
}

// One stable split of the tile on one digit bit, from tileA to tileB (or back): entries with the
// bit clear first. An entry is digit << 16 | its slot in the tile.
void sortSplit(bool fromA, uint bit, uint lane) {
  uint zero[8] = {};
  uint before[8] = {};
  uint entries[8];
  uint zeros = 0u;
  for (uint k = 0u; k < kSortPerThread; ++k) {
    const uint slot = lane * kSortPerThread + k;
    entries[k] = fromA ? tileA[slot] : tileB[slot];
    if (((entries[k] >> (16u + bit)) & 1u) == 0u) zero[k] = 1u;
    before[k] = zeros;
    zeros += zero[k];
  }
  partial[lane] = zeros;
  GroupMemoryBarrierWithGroupSync();
  sortScanThreads(lane);
  uint zerosBefore = 0u;
  if (lane > 0u) zerosBefore = partial[lane - 1u];
  const uint totalZeros = partial[kSortThreads - 1u];
  for (uint k = 0u; k < kSortPerThread; ++k) {
    const uint slot = lane * kSortPerThread + k;
    const uint rank = zerosBefore + before[k];
    // rank counts the zeros before the slot; a one goes after every zero, behind the
    // slot - rank ones before it.
    const uint position = zero[k] != 0u ? rank : totalZeros + slot - rank;
    if (fromA) tileB[position] = entries[k];
    else tileA[position] = entries[k];
  }
  GroupMemoryBarrierWithGroupSync();
}

// Sorts the tile's (digit, slot) entries stably by digit (eight one-bit splits in shared memory),
// then moves each key to its digit's global offset plus its rank among the tile's equal digits,
// reading the key and value again from device memory.
[shader("compute")]
[numthreads(128, 1, 1)]
void bvh_sort_scatter(uint lane: SV_GroupIndex, uint3 group: SV_GroupID) {
  const uint tile = group.x;
  const uint count = control.sizes.x;
  const uint first = tile * kSortTile;
  // Missing keys (past the end) take the largest digit and stay behind the real ones.
  for (uint k = 0u; k < kSortPerThread; ++k) {
    const uint slot = k * kSortThreads + lane;
    uint digit = 255u;
    if (first + slot < count) digit = sortDigit(keysLo[first + slot], keysHi[first + slot], control.pass);
    tileA[slot] = (digit << 16u) | slot;
  }
  GroupMemoryBarrierWithGroupSync();
  for (uint bit = 0u; bit < 8u; bit += 2u) {
    sortSplit(true, bit, lane);
    sortSplit(false, bit + 1u, lane);
  }

  for (uint k = 0u; k < kSortPerThread; ++k) {
    const uint position = lane * kSortPerThread + k;
    const uint digit = tileA[position] >> 16u;
    if (position == 0u || (tileA[position - 1u] >> 16u) != digit) digitStart[digit] = position;
  }
  GroupMemoryBarrierWithGroupSync();
  for (uint k = 0u; k < kSortPerThread; ++k) {
    const uint position = lane * kSortPerThread + k;
    const uint slot = tileA[position] & 0xFFFFu;
    if (first + slot >= count) continue;
    const uint digit = tileA[position] >> 16u;
    const uint target = histogram[digit * control.sizes.y + tile] + position - digitStart[digit];
    outLo[target] = keysLo[first + slot];
    outHi[target] = keysHi[first + slot];
    outValues[target] = values[first + slot];
  }
}
