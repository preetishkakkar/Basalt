// Collapse of a resident binary tree into quantized BVH4/BVH8 nodes: pt::buildWideBvh
// (src/pt/BvhVariants.cpp) on the GPU, publishing its nodes byte for byte and in its order.
// Every float operation of the host's rule is done on bits (pt/exact_float.h), so rounding and
// subnormals match the CPU's IEEE arithmetic.
//
// A task is one wide node: the binary node its frontier starts from, and whether it is in a
// bottom level. The TLAS and the bottom levels form one tree here: a TLAS leaf's task has the
// instance's bottom-level root as its child, so one bottom-up pass gives the host's traversal
// stack bound. Tasks are queued per tree level (task 0, the TLAS root, is level 0); fence-free
// like the LBVH: a kernel reads what earlier dispatches wrote, and workgroups meet only in
// atomics. Sequence: gather (top-down, once per level: frontier, slots, children queued),
// size (bottom-up: wide nodes per subtree and stack bound), roots (one workgroup: numbers of
// the TLAS root and, after the whole TLAS, each bottom level's root in instance order, as the
// host emits them), emit (top-down: each node numbers its children in slot order, depth first,
// and writes itself).
#include "shared/prelude.h"
#include "pt/bvh_build.h"
#include "pt/bvh_layout.h"
#include "pt/exact_float.h"

constant uint kCollapseLevels = 64u;  // PT_BVH_STACK: deeper trees cannot be traversed
constant uint kCollapseBottom = 0x80000000u;  // task word: a bottom-level node
constant uint kCollapseMaximumSteps = 1024u;  // frontier expansions; more means a malformed tree
constant uint kExactHalf = 0x3F000000u;        // 0.5f
constant uint kExactTwo = 0x40000000u;         // 2.0f
constant uint kExactMinusOne = 0xBF800000u;    // -1.0f
constant uint kExactQuantumStep = 0x37800080u; // 1.0f / 65535.0f

inline void collapseFail(device atomic_uint *error, uint code) {
  atomic_fetch_max_explicit(&error[0], code, memory_order_relaxed);
}

// Binary child entry `reference` = node * 2 + side: its box words and data / count.
inline uint4 collapseLow(const device uint4 *binary, uint reference) { return binary[reference * 2u]; }
inline uint4 collapseHigh(const device uint4 *binary, uint reference) { return binary[reference * 2u + 1u]; }

// The host's area(): max(0, 2 * (dx * dy + dy * dz + dz * dx)).
inline uint collapseArea(uint4 low, uint4 high) {
  const uint dx = ptExactSubtract(high.x, low.x);
  const uint dy = ptExactSubtract(high.y, low.y);
  const uint dz = ptExactSubtract(high.z, low.z);
  const uint sum = ptExactAdd(ptExactAdd(ptExactMultiply(dx, dy), ptExactMultiply(dy, dz)), ptExactMultiply(dz, dx));
  const uint area = ptExactMultiply(kExactTwo, sum);
  if (ptExactLess(0u, area)) return area;
  return 0u;
}

// A uint whose unsigned order is ptExactLess's on non-NaN floats (the zeros equal); a NaN maps
// above +infinity, so like the host's `cost < best` it is never chosen over infinity.
inline uint collapseOrderKey(uint a) {
  if (ptExactIsNaN(a)) return 0xFFFFFFFFu;
  if (a == 0x80000000u) a = 0u;
  if ((a & 0x80000000u) != 0u) return ~a;
  return a | 0x80000000u;
}

// The host's packBounds().
inline uint collapsePack(uint low, uint high, uint origin, uint scale) {
  if (!ptExactLess(0u, scale)) return 0u;
  uint lo = ptExactQuantiseQuotient(ptExactSubtract(low, origin), scale, false);
  uint hi = ptExactQuantiseQuotient(ptExactSubtract(high, origin), scale, true);
  if (lo > 0u) lo -= 1u;
  if (hi < 65535u) hi += 1u;
  return lo | (hi << 16u);
}

// Queues a task on `level`; kBvhEmpty (and an error) when the level or the capacity runs out.
inline uint collapsePush(device atomic_uint *headers, device atomic_uint *error, uint level, uint levels,
                         uint nextBase, uint capacity) {
  if (level >= levels) {
    collapseFail(error, kBvhBuildErrorDepth);
    return kBvhEmpty;
  }
  const uint slot = atomic_fetch_add_explicit(&headers[level * 4u + 3u], 1u, memory_order_relaxed);
  atomic_fetch_max_explicit(&headers[level * 4u], (slot + 64u) / 64u, memory_order_relaxed);
  if (nextBase + slot >= capacity) {
    collapseFail(error, kBvhBuildErrorCapacity);
    return kBvhEmpty;
  }
  return nextBase + slot;
}

// 1. One level, top-down: the task's frontier (the host's rule: expand the interior entry of
// largest area, first on ties, until `width` entries or only leaves), its slot order (BVH8:
// octant slots, greedy on the cheapest (entry, slot) pair) and its children's tasks: interior
// entries, and at the TLAS each leaf's bottom-level root.
kernel void bvh_collapse_gather(const device uint4 *binary [[buffer(0)]],
                                const device uint *instanceRoots [[buffer(1)]],
                                device uint *taskNode [[buffer(2)]],
                                device uint *taskSlots [[buffer(3)]],
                                device uint *taskChild [[buffer(4)]],
                                device atomic_uint *rootTask [[buffer(5)]],
                                device uint *levelBase [[buffer(6)]],
                                device atomic_uint *headers [[buffer(7)]],
                                device atomic_uint *error [[buffer(8)]],
                                constant BvhCollapseControl &control [[buffer(9)]],
                                uint id [[thread_position_in_grid]]) {
  const uint level = control.level.x;
  const uint count = atomic_load_explicit(&headers[level * 4u + 3u], memory_order_relaxed);
  if (id >= count) return;
  const uint base = levelBase[level];
  const uint nextBase = base + count;
  if (id == 0u) levelBase[level + 1u] = nextBase;
  const uint task = base + id;
  const uint word = taskNode[task];
  const uint node = word & ~kCollapseBottom;
  const bool bottom = (word & kCollapseBottom) != 0u;
  const uint binaryNodes = control.sizes.y, width = control.sizes.z;
  for (uint slot = 0u; slot < 8u; ++slot) {
    taskSlots[task * 8u + slot] = kBvhEmpty;
    taskChild[task * 8u + slot] = kBvhEmpty;
  }
  if (node >= binaryNodes) {
    collapseFail(error, kBvhBuildErrorCapacity);
    return;
  }

  uint entries[8] = {};
  uint areas[8] = {};
  uint n = 0u;
  // Append the node's non-empty children; while below `width`, replace the interior entry of
  // largest area (the first on ties) by its children.
  uint expanded = node;
  for (uint step = 0u;; ++step) {
    if (step == kCollapseMaximumSteps) {
      collapseFail(error, kBvhBuildErrorCapacity);
      return;
    }
    for (uint side = 0u; side < 2u; ++side) {
      const uint reference = expanded * 2u + side;
      const uint4 low = collapseLow(binary, reference);
      if (low.w == kBvhEmpty) continue;
      entries[n] = reference;
      areas[n] = collapseArea(low, collapseHigh(binary, reference));
      n += 1u;
    }
    if (n >= width) break;
    uint chosen = n;
    uint largest = kExactMinusOne;
    for (uint i = 0u; i < n; ++i)
      if (collapseHigh(binary, entries[i]).w == 0u && ptExactLess(largest, areas[i])) {
        chosen = i;
        largest = areas[i];
      }
    if (chosen == n) break;
    expanded = collapseLow(binary, entries[chosen]).w;
    if (expanded >= binaryNodes) {
      collapseFail(error, kBvhBuildErrorCapacity);
      return;
    }
    for (uint i = chosen; i + 1u < n; ++i) {
      entries[i] = entries[i + 1u];
      areas[i] = areas[i + 1u];
    }
    n -= 1u;
  }

  // Slot s holds entry entryOfSlot[s]; BVH4 keeps the collapse order.
  uint entryOfSlot[8] = {kBvhEmpty, kBvhEmpty, kBvhEmpty, kBvhEmpty, kBvhEmpty, kBvhEmpty, kBvhEmpty, kBvhEmpty};
  if (width != 8u) {
    for (uint i = 0u; i < n; ++i) entryOfSlot[i] = i;
  } else if (n > 0u) {
    uint low[3] = {kExactPositiveInfinity, kExactPositiveInfinity, kExactPositiveInfinity};
    uint high[3] = {kExactNegativeInfinity, kExactNegativeInfinity, kExactNegativeInfinity};
    for (uint i = 0u; i < n; ++i) {
      const uint4 childLow = collapseLow(binary, entries[i]), childHigh = collapseHigh(binary, entries[i]);
      low[0] = ptExactMin(low[0], childLow.x);
      low[1] = ptExactMin(low[1], childLow.y);
      low[2] = ptExactMin(low[2], childLow.z);
      high[0] = ptExactMax(high[0], childHigh.x);
      high[1] = ptExactMax(high[1], childHigh.y);
      high[2] = ptExactMax(high[2], childHigh.z);
    }
    uint centre[3] = {};
    for (uint a = 0u; a < 3u; ++a) centre[a] = ptExactMultiply(ptExactAdd(low[a], high[a]), kExactHalf);
    // key[i * 8 + s] orders dot(entry centre - node centre, direction of s), summed in the host's
    // order: RN(RN(+-x +- y) +- z). Rounding to nearest is symmetric, so the eight are four
    // sums and their negations (up to the sign of an exact zero, which compares equal).
    uint key[64] = {};
    for (uint i = 0u; i < n; ++i) {
      const uint4 childLow = collapseLow(binary, entries[i]), childHigh = collapseHigh(binary, entries[i]);
      const uint ox = ptExactSubtract(ptExactMultiply(ptExactAdd(childLow.x, childHigh.x), kExactHalf), centre[0]);
      const uint oy = ptExactSubtract(ptExactMultiply(ptExactAdd(childLow.y, childHigh.y), kExactHalf), centre[1]);
      const uint oz = ptExactSubtract(ptExactMultiply(ptExactAdd(childLow.z, childHigh.z), kExactHalf), centre[2]);
      const uint sum = ptExactAdd(ox, oy), difference = ptExactSubtract(ox, oy);
      const uint a = ptExactAdd(sum, oz), b = ptExactSubtract(sum, oz);
      const uint c = ptExactAdd(difference, oz), d = ptExactSubtract(difference, oz);
      key[i * 8u + 0u] = collapseOrderKey(a);
      key[i * 8u + 1u] = collapseOrderKey(d ^ 0x80000000u);
      key[i * 8u + 2u] = collapseOrderKey(c);
      key[i * 8u + 3u] = collapseOrderKey(b ^ 0x80000000u);
      key[i * 8u + 4u] = collapseOrderKey(b);
      key[i * 8u + 5u] = collapseOrderKey(c ^ 0x80000000u);
      key[i * 8u + 6u] = collapseOrderKey(d);
      key[i * 8u + 7u] = collapseOrderKey(a ^ 0x80000000u);
    }
    // Greedy as the host: the first strictly cheapest (entry, slot), entries then slots in order.
    uint placed = 0u, taken = 0u;  // bit masks of assigned entries and slots
    for (uint assigned = 0u; assigned < n; ++assigned) {
      uint best = collapseOrderKey(kExactPositiveInfinity), bestEntry = 0u, bestSlot = 0u;
      for (uint i = 0u; i < n; ++i) {
        if ((placed & (1u << i)) != 0u) continue;
        for (uint s = 0u; s < 8u; ++s) {
          if ((taken & (1u << s)) != 0u) continue;
          if (key[i * 8u + s] < best) {
            best = key[i * 8u + s];
            bestEntry = i;
            bestSlot = s;
          }
        }
      }
      placed |= 1u << bestEntry;
      taken |= 1u << bestSlot;
      entryOfSlot[bestSlot] = bestEntry;
    }
  }

  for (uint slot = 0u; slot < 8u; ++slot) {
    if (entryOfSlot[slot] == kBvhEmpty) continue;
    const uint reference = entries[entryOfSlot[slot]];
    taskSlots[task * 8u + slot] = reference;
    const uint data = collapseLow(binary, reference).w;
    uint childWord = kBvhEmpty;
    if (collapseHigh(binary, reference).w == 0u) {
      childWord = data;
      if (bottom) childWord |= kCollapseBottom;
    } else if (!bottom) {
      if (data >= control.sizes.w) {
        collapseFail(error, kBvhBuildErrorCapacity);
        continue;
      }
      childWord = instanceRoots[data] | kCollapseBottom;
    }
    if (childWord == kBvhEmpty) continue;
    const uint child = collapsePush(headers, error, level + 1u, control.level.y, nextBase, control.sizes.x);
    if (child == kBvhEmpty) continue;
    taskNode[child] = childWord;
    taskChild[task * 8u + slot] = child;
    // Each instance's bottom level is reached from exactly one TLAS leaf.
    if (!bottom && collapseHigh(binary, reference).w != 0u &&
        atomic_exchange_explicit(&rootTask[data], child, memory_order_relaxed) != kBvhEmpty)
      collapseFail(error, kBvhBuildErrorCapacity);
  }
}

// 2. One level, bottom-up: wide nodes in the task's own tree below and including it, and the
// host's traversal stack bound for its subtree (maximumTraversalStack's stack when the task
// is popped alone): max(1, k, max over the j-th of its k occupied slots of j + the child's),
// a bottom-level leaf counting 1 and a TLAS leaf its bottom level's root.
kernel void bvh_collapse_size(const device uint4 *binary [[buffer(0)]],
                              const device uint *taskNode [[buffer(1)]],
                              const device uint *taskSlots [[buffer(2)]],
                              const device uint *taskChild [[buffer(3)]],
                              device uint *taskSize [[buffer(4)]],
                              device uint *taskStack [[buffer(5)]],
                              const device uint *levelBase [[buffer(6)]],
                              const device uint *headers [[buffer(7)]],
                              constant BvhCollapseControl &control [[buffer(8)]],
                              uint id [[thread_position_in_grid]]) {
  const uint level = control.level.x;
  if (id >= headers[level * 4u + 3u]) return;
  const uint task = levelBase[level] + id;
  const bool bottom = (taskNode[task] & kCollapseBottom) != 0u;
  uint size = 1u, stack = 1u, occupied = 0u;
  for (uint slot = 0u; slot < 8u; ++slot) {
    const uint reference = taskSlots[task * 8u + slot];
    if (reference == kBvhEmpty) continue;
    const uint child = taskChild[task * 8u + slot];
    uint childStack = 1u;
    if (child != kBvhEmpty) {
      childStack = taskStack[child];
      if (bottom || collapseHigh(binary, reference).w == 0u) size += taskSize[child];
    }
    stack = max(stack, occupied + childStack);
    occupied += 1u;
  }
  taskSize[task] = size;
  taskStack[task] = max(stack, occupied);
}

// Inclusive Hillis-Steele scan of partial[0..127] in place (bvh_sort.metal's sortScanThreads).
inline void collapseScanThreads(threadgroup uint *partial, uint lane) {
  for (uint offset = 1u; offset < 128u; offset <<= 1u) {
    uint add = 0u;
    if (lane >= offset) add = partial[lane - offset];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    partial[lane] = partial[lane] + add;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// 3. One workgroup of 128: the TLAS root is node 0 and bottom level i's root follows the TLAS
// and the bottom levels before it; the status record.
kernel void bvh_collapse_roots(const device uint *taskSize [[buffer(0)]],
                               const device uint *taskStack [[buffer(1)]],
                               const device uint *rootTask [[buffer(2)]],
                               device uint *taskNumber [[buffer(3)]],
                               device uint *rootNumber [[buffer(4)]],
                               const device uint *headers [[buffer(5)]],
                               const device uint *error [[buffer(6)]],
                               device BvhCollapseStatus *status [[buffer(7)]],
                               constant BvhCollapseControl &control [[buffer(8)]],
                               uint lane [[thread_index_in_threadgroup]]) {
  threadgroup uint partial[128];
  threadgroup atomic_uint missing;  // an instance no TLAS leaf reached
  if (lane == 0u) atomic_store_explicit(&missing, 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint instances = control.sizes.w;
  uint carry = taskSize[0];
  for (uint first = 0u; first < instances; first += 1024u) {
    uint values[8] = {};
    uint running[8] = {};
    uint total = 0u;
    const uint base = first + lane * 8u;
    for (uint k = 0u; k < 8u; ++k) {
      if (base + k < instances) {
        const uint root = rootTask[base + k];
        if (root == kBvhEmpty) atomic_fetch_max_explicit(&missing, 1u, memory_order_relaxed);
        else values[k] = taskSize[root];
      }
      running[k] = total;
      total += values[k];
    }
    partial[lane] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    collapseScanThreads(partial, lane);
    uint prefix = carry;
    if (lane > 0u) prefix += partial[lane - 1u];
    for (uint k = 0u; k < 8u; ++k)
      if (base + k < instances && rootTask[base + k] != kBvhEmpty) {
        taskNumber[rootTask[base + k]] = prefix + running[k];
        rootNumber[base + k] = prefix + running[k];
      }
    carry += partial[127];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane != 0u) return;
  taskNumber[0] = 0u;
  uint levels = 0u;
  while (levels < kCollapseLevels && headers[levels * 4u + 3u] != 0u) levels += 1u;
  uint result = error[0];
  if (result == kBvhBuildErrorNone && atomic_load_explicit(&missing, memory_order_relaxed) != 0u)
    result = kBvhBuildErrorCapacity;
  const uint stack = taskStack[0];
  if (result == kBvhBuildErrorNone && stack > uint(PT_WIDE_BVH_STACK_DEEP)) result = kBvhBuildErrorDepth;
  BvhCollapseStatus out;
  out.result = uint4(result, levels, carry, stack);
  *status = out;
}

// 4. One level, top-down, one thread per (task, slot): the task's node at its number, the slot's
// interior child numbered after the node and the subtrees of the interior children in earlier
// slots (the host's depth-first emission). Every thread of a task derives the node's bounds.
kernel void bvh_collapse_emit(const device uint4 *binary [[buffer(0)]],
                              const device uint *taskNode [[buffer(1)]],
                              const device uint *taskSlots [[buffer(2)]],
                              const device uint *taskChild [[buffer(3)]],
                              const device uint *taskSize [[buffer(4)]],
                              device uint *taskNumber [[buffer(5)]],
                              const device uint *levelBase [[buffer(6)]],
                              const device uint *headers [[buffer(7)]],
                              device uint4 *nodes [[buffer(8)]],
                              constant BvhCollapseControl &control [[buffer(9)]],
                              uint id [[thread_position_in_grid]]) {
  const uint level = control.level.x;
  if (id / 8u >= headers[level * 4u + 3u]) return;
  const uint task = levelBase[level] + id / 8u, own = id % 8u;
  const bool bottom = (taskNode[task] & kCollapseBottom) != 0u;
  const uint number = taskNumber[task];
  const uint width = control.sizes.z;
  const uint output = number * 10u;  // PtWideNode: origin, scale, eight children

  uint low[3] = {kExactPositiveInfinity, kExactPositiveInfinity, kExactPositiveInfinity};
  uint high[3] = {kExactNegativeInfinity, kExactNegativeInfinity, kExactNegativeInfinity};
  uint occupied = 0u, next = number + 1u;
  for (uint slot = 0u; slot < 8u; ++slot) {
    const uint reference = taskSlots[task * 8u + slot];
    if (reference == kBvhEmpty) continue;
    const uint4 childLow = collapseLow(binary, reference), childHigh = collapseHigh(binary, reference);
    low[0] = ptExactMin(low[0], childLow.x);
    low[1] = ptExactMin(low[1], childLow.y);
    low[2] = ptExactMin(low[2], childLow.z);
    high[0] = ptExactMax(high[0], childHigh.x);
    high[1] = ptExactMax(high[1], childHigh.y);
    high[2] = ptExactMax(high[2], childHigh.z);
    occupied += 1u;
    const uint child = taskChild[task * 8u + slot];
    if (slot < own && childHigh.w == 0u && child != kBvhEmpty) next += taskSize[child];
  }
  if (occupied == 0u) {
    nodes[output + 2u + own] = uint4(0u);
    if (own < 2u) nodes[output + own] = uint4(0u);
    return;
  }
  uint scale[3] = {};
  for (uint a = 0u; a < 3u; ++a) {
    low[a] = ptExactNextAfter(low[a], false);
    high[a] = ptExactNextAfter(high[a], true);
    scale[a] = ptExactNextAfter(ptExactMultiply(ptExactSubtract(high[a], low[a]), kExactQuantumStep), true);
  }
  if (own == 0u) {
    uint slots = occupied;
    if (width == 8u) slots = 8u | kWideOctantOrdered;
    nodes[output] = uint4(low[0], low[1], low[2], slots);
  } else if (own == 1u) {
    nodes[output + 1u] = uint4(scale[0], scale[1], scale[2], 0u);
  }
  const uint reference = taskSlots[task * 8u + own];
  if (reference == kBvhEmpty) {
    uint4 empty = uint4(0u);
    if (width == 8u) empty.w = kBvhEmpty;
    nodes[output + 2u + own] = empty;
    return;
  }
  const uint4 childLow = collapseLow(binary, reference), childHigh = collapseHigh(binary, reference);
  uint data = kBvhLeafTag | childLow.w;
  if (childHigh.w == 0u) {
    const uint child = taskChild[task * 8u + own];
    data = next;
    if (child != kBvhEmpty) taskNumber[child] = next;
  } else if (bottom) {
    data = kBvhLeafTag | (childHigh.w << kBvhCountShift) | childLow.w;
  }
  nodes[output + 2u + own] = uint4(collapsePack(childLow.x, childHigh.x, low[0], scale[0]),
                                   collapsePack(childLow.y, childHigh.y, low[1], scale[1]),
                                   collapsePack(childLow.z, childHigh.z, low[2], scale[2]), data);
}

// 5. One thread per level: the emit's indirect dispatch (eight threads per task) from the level's
// task count, for a collapse recorded without reading the counts back.
kernel void bvh_collapse_emit_args(const device uint *headers [[buffer(0)]],
                                   device uint4 *emitHeaders [[buffer(1)]],
                                   uint id [[thread_position_in_grid]]) {
  if (id > kCollapseLevels) return;
  emitHeaders[id] = uint4((headers[id * 4u + 3u] * 8u + 63u) / 64u, 1u, 1u, 0u);
}

// 6. Per instance: the traced row, the input row (the binary tree's roots, the current
// transforms) with its bottom level's wide root.
kernel void bvh_collapse_rows(const device TraceInstance *input [[buffer(0)]],
                              device TraceInstance *output [[buffer(1)]],
                              const device uint *rootNumber [[buffer(2)]],
                              constant BvhCollapseControl &control [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
  if (id >= control.sizes.w) return;
  TraceInstance row = input[id];
  row.blasRoot = rootNumber[id];
  output[id] = row;
}
