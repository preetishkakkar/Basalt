// Stable LSD radix sort of (keyLo, keyHi, value) triples, 8 bits per pass, for the parallel
// GPU builders: key = (segment, Morton code), value = record index. One pass is five
// dispatches: per-tile digit histograms (digit-major), a three-step exclusive scan of them
// (block sums, one workgroup scanning the block sums, blocks scanned and offset) giving each
// (digit, tile) its global position, and a scatter that sorts each tile stably by the digit
// in threadgroup memory before writing it out. Tiles and scan blocks hold 1024 entries
// (128 threads x 8): msl2spirv's portable profile allows 128 threads and 16 KiB of
// threadgroup memory per workgroup, and no barrier inside a loop bounded by buffer data.
// No cross-workgroup communication inside a dispatch.
#include "shared/prelude.h"
#include "pt/bvh_build.h"

constant uint kSortThreads = 128u;
constant uint kSortPerThread = 8u;
constant uint kSortTile = 1024u;

// The pass's digit: 8 bits of the low (key word 0) or high key word at shift.
inline uint sortDigit(uint lo, uint hi, uint4 pass) {
  uint key = lo;
  if (pass.w != 0u) key = hi;
  return (key >> pass.z) & 255u;
}

// histogram[digit * tiles + tile] = keys of this tile with that digit.
kernel void bvh_sort_histogram(const device uint *keysLo [[buffer(0)]],
                               const device uint *keysHi [[buffer(1)]],
                               device uint *histogram [[buffer(2)]],
                               constant BvhSortControl &control [[buffer(3)]],
                               uint lane [[thread_index_in_threadgroup]],
                               uint tile [[threadgroup_position_in_grid]]) {
  threadgroup atomic_uint counts[256];
  atomic_store_explicit(&counts[lane], 0u, memory_order_relaxed);
  atomic_store_explicit(&counts[lane + kSortThreads], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint count = control.sizes.x;
  for (uint k = 0u; k < kSortPerThread; ++k) {
    const uint index = tile * kSortTile + k * kSortThreads + lane;
    if (index < count)
      atomic_fetch_add_explicit(&counts[sortDigit(keysLo[index], keysHi[index], control.pass)], 1u,
                                memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  histogram[lane * control.sizes.y + tile] = atomic_load_explicit(&counts[lane], memory_order_relaxed);
  histogram[(lane + kSortThreads) * control.sizes.y + tile] =
      atomic_load_explicit(&counts[lane + kSortThreads], memory_order_relaxed);
}

// Inclusive Hillis-Steele scan of partial[0..127] in place; every thread must call it.
inline void sortScanThreads(threadgroup uint *partial, uint lane) {
  for (uint offset = 1u; offset < kSortThreads; offset <<= 1u) {
    uint add = 0u;
    if (lane >= offset) add = partial[lane - offset];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    partial[lane] = partial[lane] + add;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Exclusive scan of the 1024-entry block at data[block * 1024] (entries past length count as
// zero and are not written), plus offset; returns the block's total. Every thread must call it.
inline uint sortScanBlock(device uint *data, uint length, uint block, uint offset,
                          threadgroup uint *partial, uint lane, bool write) {
  uint values[8] = {};
  uint running[8] = {};
  uint total = 0u;
  const uint base = block * kSortTile + lane * kSortPerThread;
  for (uint k = 0u; k < kSortPerThread; ++k) {
    if (base + k < length) values[k] = data[base + k];
    running[k] = total;
    total += values[k];
  }
  partial[lane] = total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sortScanThreads(partial, lane);
  uint prefix = offset;
  if (lane > 0u) prefix += partial[lane - 1u];
  const uint blockTotal = partial[kSortThreads - 1u];
  if (write)
    for (uint k = 0u; k < kSortPerThread; ++k)
      if (base + k < length) data[base + k] = prefix + running[k];
  return blockTotal;
}

// Block b of the histogram: its total into blockSums[b].
kernel void bvh_sort_reduce(device uint *histogram [[buffer(0)]],
                            device uint *blockSums [[buffer(1)]],
                            constant BvhSortControl &control [[buffer(2)]],
                            uint lane [[thread_index_in_threadgroup]],
                            uint block [[threadgroup_position_in_grid]]) {
  threadgroup uint partial[128];
  const uint total = sortScanBlock(histogram, 256u * control.sizes.y, block, 0u, partial, lane, false);
  if (lane == 0u) blockSums[block] = total;
}

// One workgroup: the block sums (at most 1024 blocks) to exclusive offsets, in place.
kernel void bvh_sort_blocks(device uint *blockSums [[buffer(0)]],
                            constant BvhSortControl &control [[buffer(1)]],
                            uint lane [[thread_index_in_threadgroup]]) {
  threadgroup uint partial[128];
  sortScanBlock(blockSums, control.sizes.z, 0u, 0u, partial, lane, true);
}

// Block b of the histogram to exclusive global offsets.
kernel void bvh_sort_apply(device uint *histogram [[buffer(0)]],
                           const device uint *blockSums [[buffer(1)]],
                           constant BvhSortControl &control [[buffer(2)]],
                           uint lane [[thread_index_in_threadgroup]],
                           uint block [[threadgroup_position_in_grid]]) {
  threadgroup uint partial[128];
  sortScanBlock(histogram, 256u * control.sizes.y, block, blockSums[block], partial, lane, true);
}

// One stable split of the tile on one digit bit: entries with the bit clear first. An entry
// is digit << 16 | its slot in the tile.
inline void sortSplit(threadgroup uint *from, threadgroup uint *to, threadgroup uint *partial, uint bit,
                      uint lane) {
  uint zero[8] = {};
  uint before[8] = {};
  uint zeros = 0u;
  for (uint k = 0u; k < kSortPerThread; ++k) {
    const uint slot = lane * kSortPerThread + k;
    if (((from[slot] >> (16u + bit)) & 1u) == 0u) zero[k] = 1u;
    before[k] = zeros;
    zeros += zero[k];
  }
  partial[lane] = zeros;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sortScanThreads(partial, lane);
  uint zerosBefore = 0u;
  if (lane > 0u) zerosBefore = partial[lane - 1u];
  const uint totalZeros = partial[kSortThreads - 1u];
  for (uint k = 0u; k < kSortPerThread; ++k) {
    const uint slot = lane * kSortPerThread + k;
    const uint rank = zerosBefore + before[k];
    uint position = totalZeros + slot - rank;
    if (zero[k] != 0u) position = rank;
    to[position] = from[slot];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Sorts the tile's (digit, slot) entries stably by digit (eight one-bit splits in threadgroup
// memory), then moves each key to its digit's global offset plus its rank among the tile's
// equal digits, reading the key and value again from device memory.
kernel void bvh_sort_scatter(const device uint *keysLo [[buffer(0)]],
                             const device uint *keysHi [[buffer(1)]],
                             const device uint *values [[buffer(2)]],
                             device uint *outLo [[buffer(3)]],
                             device uint *outHi [[buffer(4)]],
                             device uint *outValues [[buffer(5)]],
                             const device uint *histogram [[buffer(6)]],
                             constant BvhSortControl &control [[buffer(7)]],
                             uint lane [[thread_index_in_threadgroup]],
                             uint tile [[threadgroup_position_in_grid]]) {
  threadgroup uint a[1024];
  threadgroup uint b[1024];
  threadgroup uint partial[128];
  threadgroup uint digitStart[256];
  const uint count = control.sizes.x;
  const uint first = tile * kSortTile;
  // Missing keys (past the end) take the largest digit and stay behind the real ones.
  for (uint k = 0u; k < kSortPerThread; ++k) {
    const uint slot = k * kSortThreads + lane;
    uint digit = 255u;
    if (first + slot < count) digit = sortDigit(keysLo[first + slot], keysHi[first + slot], control.pass);
    a[slot] = (digit << 16u) | slot;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint bit = 0u; bit < 8u; bit += 2u) {
    sortSplit(a, b, partial, bit, lane);
    sortSplit(b, a, partial, bit + 1u, lane);
  }

  for (uint k = 0u; k < kSortPerThread; ++k) {
    const uint position = lane * kSortPerThread + k;
    const uint digit = a[position] >> 16u;
    if (position == 0u || (a[position - 1u] >> 16u) != digit) digitStart[digit] = position;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 0u; k < kSortPerThread; ++k) {
    const uint position = lane * kSortPerThread + k;
    const uint slot = a[position] & 0xFFFFu;
    if (first + slot >= count) continue;
    const uint digit = a[position] >> 16u;
    const uint target = histogram[digit * control.sizes.y + tile] + position - digitStart[digit];
    outLo[target] = keysLo[first + slot];
    outHi[target] = keysHi[first + slot];
    outValues[target] = values[first + slot];
  }
}
