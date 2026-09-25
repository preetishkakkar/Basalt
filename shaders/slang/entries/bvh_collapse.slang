// Collapse of a resident binary tree into quantized BVH4/BVH8 nodes: pt::buildWideBvh
// (src/pt/BvhVariants.cpp) on the GPU, publishing its nodes byte for byte and in its order.
// Every float operation of the host's rule is done on bits (pt_exact_float), so rounding and
// subnormals match the CPU's IEEE arithmetic.
//
// A task is one wide node: the binary node its frontier starts from, and whether it is in a bottom
// level. The TLAS and the bottom levels form one tree here: a TLAS leaf's task has the instance's
// bottom-level root as its child, so one bottom-up pass gives the host's traversal stack bound.
// Tasks are queued per tree level (task 0, the TLAS root, is level 0); fence-free like the LBVH: a
// kernel reads what earlier dispatches wrote, and workgroups meet only in atomics. Sequence: gather
// (top-down, once per level: frontier, slots, children queued), size (bottom-up: wide nodes per
// subtree and stack bound), roots (one workgroup: numbers of the TLAS root and, after the whole
// TLAS, each bottom level's root in instance order, as the host emits them), emit (top-down: each
// node numbers its children in slot order, depth first, and writes itself).
import pt_bvh_build;
import pt_bvh;
import pt_wide_bvh;
import pt_exact_float;

static const uint kCollapseLevels = uint(kBvhStack); // deeper trees cannot be traversed
static const uint kCollapseBottom = 0x80000000u; // task word: a bottom-level node
static const uint kCollapseMaximumSteps = 1024u; // frontier expansions; more means a malformed tree
static const uint kExactHalf = 0x3F000000u;        // 0.5f
static const uint kExactTwo = 0x40000000u;         // 2.0f
static const uint kExactMinusOne = 0xBF800000u;    // -1.0f
static const uint kExactQuantumStep = 0x37800080u; // 1.0f / 65535.0f

ConstantBuffer<BvhCollapseControl> control;
StructuredBuffer<uint4> binary; // the binary tree's child entries, low and high per entry
StructuredBuffer<uint> instanceRoots;
RWStructuredBuffer<uint> taskNode;
RWStructuredBuffer<uint> taskSlots;
RWStructuredBuffer<uint> taskChild;
RWStructuredBuffer<uint> taskSize;
RWStructuredBuffer<uint> taskStack;
RWStructuredBuffer<uint> taskNumber;
RWStructuredBuffer<uint> rootTask;
RWStructuredBuffer<uint> rootNumber;
RWStructuredBuffer<uint> levelBase;
RWStructuredBuffer<uint> headers;
RWStructuredBuffer<uint> error;
RWStructuredBuffer<BvhCollapseStatus> status;
RWStructuredBuffer<uint4> nodes;
RWStructuredBuffer<uint4> emitHeaders;
StructuredBuffer<TraceInstance> input;
RWStructuredBuffer<TraceInstance> output;

groupshared uint partial[128];
groupshared uint missing; // an instance no TLAS leaf reached

void collapseFail(uint code) { InterlockedMax(error[0], code); }

// Binary child entry `reference` = node * 2 + side: its box words and data / count.
uint4 collapseLow(uint reference) { return binary[reference * 2u]; }
uint4 collapseHigh(uint reference) { return binary[reference * 2u + 1u]; }

// The host's area(): max(0, 2 * (dx * dy + dy * dz + dz * dx)).
uint collapseArea(uint4 low, uint4 high) {
  const uint dx = ptExactSubtract(high.x, low.x);
  const uint dy = ptExactSubtract(high.y, low.y);
  const uint dz = ptExactSubtract(high.z, low.z);
  const uint sum = ptExactAdd(ptExactAdd(ptExactMultiply(dx, dy), ptExactMultiply(dy, dz)), ptExactMultiply(dz, dx));
  const uint area = ptExactMultiply(kExactTwo, sum);
  if (ptExactLess(0u, area)) return area;
  return 0u;
}

// A uint whose unsigned order is ptExactLess's on non-NaN floats (the zeros equal); a NaN maps above
// +infinity, so like the host's `cost < best` it is never chosen over infinity.
uint collapseOrderKey(uint a) {
  if (ptExactIsNaN(a)) return 0xFFFFFFFFu;
  if (a == 0x80000000u) a = 0u;
  if ((a & 0x80000000u) != 0u) return ~a;
  return a | 0x80000000u;
}

// The host's packBounds().
uint collapsePack(uint low, uint high, uint origin, uint scale) {
  if (!ptExactLess(0u, scale)) return 0u;
  uint lo = ptExactQuantiseQuotient(ptExactSubtract(low, origin), scale, false);
  uint hi = ptExactQuantiseQuotient(ptExactSubtract(high, origin), scale, true);
  if (lo > 0u) lo -= 1u;
  if (hi < 65535u) hi += 1u;
  return lo | (hi << 16u);
}

// Queues a task on `level`; kBvhEmpty (and an error) when the level or the capacity runs out.
uint collapsePush(uint level, uint levels, uint nextBase, uint capacity) {
  if (level >= levels) {
    collapseFail(kBvhBuildErrorDepth);
    return kBvhEmpty;
  }
  uint slot;
  InterlockedAdd(headers[level * 4u + 3u], 1u, slot);
  InterlockedMax(headers[level * 4u], (slot + 64u) / 64u);
  if (nextBase + slot >= capacity) {
    collapseFail(kBvhBuildErrorCapacity);
    return kBvhEmpty;
  }
  return nextBase + slot;
}

// 1. One level, top-down: the task's frontier (the host's rule: expand the interior entry of
// largest area, first on ties, until `width` entries or only leaves), its slot order (BVH8: octant
// slots, greedy on the cheapest (entry, slot) pair) and its children's tasks: interior entries, and
// at the TLAS each leaf's bottom-level root.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_collapse_gather(uint3 thread: SV_DispatchThreadID) {
  const uint id = thread.x;
  const uint level = control.level.x;
  const uint count = headers[level * 4u + 3u];
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
    collapseFail(kBvhBuildErrorCapacity);
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
      collapseFail(kBvhBuildErrorCapacity);
      return;
    }
    for (uint side = 0u; side < 2u; ++side) {
      const uint reference = expanded * 2u + side;
      const uint4 low = collapseLow(reference);
      if (low.w == kBvhEmpty) continue;
      entries[n] = reference;
      areas[n] = collapseArea(low, collapseHigh(reference));
      n += 1u;
    }
    if (n >= width) break;
    uint chosen = n;
    uint largest = kExactMinusOne;
    for (uint i = 0u; i < n; ++i)
      if (collapseHigh(entries[i]).w == 0u && ptExactLess(largest, areas[i])) {
        chosen = i;
        largest = areas[i];
      }
    if (chosen == n) break;
    expanded = collapseLow(entries[chosen]).w;
    if (expanded >= binaryNodes) {
      collapseFail(kBvhBuildErrorCapacity);
      return;
    }
    for (uint i = chosen; i + 1u < n; ++i) {
      entries[i] = entries[i + 1u];
      areas[i] = areas[i + 1u];
    }
    n -= 1u;
  }

  // Slot s holds entry entryOfSlot[s]; BVH4 keeps the collapse order.
  uint entryOfSlot[8] = { kBvhEmpty, kBvhEmpty, kBvhEmpty, kBvhEmpty, kBvhEmpty, kBvhEmpty, kBvhEmpty, kBvhEmpty };
  if (width != 8u) {
    for (uint i = 0u; i < n; ++i) entryOfSlot[i] = i;
  } else if (n > 0u) {
    uint low[3] = { kExactPositiveInfinity, kExactPositiveInfinity, kExactPositiveInfinity };
    uint high[3] = { kExactNegativeInfinity, kExactNegativeInfinity, kExactNegativeInfinity };
    for (uint i = 0u; i < n; ++i) {
      const uint4 childLow = collapseLow(entries[i]), childHigh = collapseHigh(entries[i]);
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
    // order: RN(RN(+-x +- y) +- z). Rounding to nearest is symmetric, so the eight are four sums
    // and their negations (up to the sign of an exact zero, which compares equal).
    uint key[64] = {};
    for (uint i = 0u; i < n; ++i) {
      const uint4 childLow = collapseLow(entries[i]), childHigh = collapseHigh(entries[i]);
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
    uint placed = 0u, taken = 0u; // bit masks of assigned entries and slots
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
    const uint data = collapseLow(reference).w;
    uint childWord = kBvhEmpty;
    if (collapseHigh(reference).w == 0u) {
      childWord = data;
      if (bottom) childWord |= kCollapseBottom;
    } else if (!bottom) {
      if (data >= control.sizes.w) {
        collapseFail(kBvhBuildErrorCapacity);
        continue;
      }
      childWord = instanceRoots[data] | kCollapseBottom;
    }
    if (childWord == kBvhEmpty) continue;
    const uint child = collapsePush(level + 1u, control.level.y, nextBase, control.sizes.x);
    if (child == kBvhEmpty) continue;
    taskNode[child] = childWord;
    taskChild[task * 8u + slot] = child;
    // Each instance's bottom level is reached from exactly one TLAS leaf.
    if (!bottom && collapseHigh(reference).w != 0u) {
      uint previous;
      InterlockedExchange(rootTask[data], child, previous);
      if (previous != kBvhEmpty) collapseFail(kBvhBuildErrorCapacity);
    }
  }
}

// 2. One level, bottom-up: wide nodes in the task's own tree below and including it, and the
// host's traversal stack bound for its subtree (maximumTraversalStack's stack when the task is
// popped alone): max(1, k, max over the j-th of its k occupied slots of j + the child's), a
// bottom-level leaf counting 1 and a TLAS leaf its bottom level's root.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_collapse_size(uint3 thread: SV_DispatchThreadID) {
  const uint id = thread.x;
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
      if (bottom || collapseHigh(reference).w == 0u) size += taskSize[child];
    }
    stack = max(stack, occupied + childStack);
    occupied += 1u;
  }
  taskSize[task] = size;
  taskStack[task] = max(stack, occupied);
}

// Inclusive Hillis-Steele scan of partial[0..127] in place (bvh_sort.slang's sortScanThreads).
void collapseScanThreads(uint lane) {
  for (uint offset = 1u; offset < 128u; offset <<= 1u) {
    uint add = 0u;
    if (lane >= offset) add = partial[lane - offset];
    GroupMemoryBarrierWithGroupSync();
    partial[lane] = partial[lane] + add;
    GroupMemoryBarrierWithGroupSync();
  }
}

// 3. One workgroup of 128: the TLAS root is node 0 and bottom level i's root follows the TLAS and
// the bottom levels before it; the status record.
[shader("compute")]
[numthreads(128, 1, 1)]
void bvh_collapse_roots(uint lane: SV_GroupIndex) {
  if (lane == 0u) missing = 0u;
  GroupMemoryBarrierWithGroupSync();
  const uint instances = control.sizes.w;
  uint carry = taskSize[0];
  for (uint first = 0u; first < instances; first += 1024u) {
    uint sizes[8] = {};
    uint running[8] = {};
    uint total = 0u;
    const uint base = first + lane * 8u;
    for (uint k = 0u; k < 8u; ++k) {
      if (base + k < instances) {
        const uint root = rootTask[base + k];
        if (root == kBvhEmpty) InterlockedMax(missing, 1u);
        else sizes[k] = taskSize[root];
      }
      running[k] = total;
      total += sizes[k];
    }
    partial[lane] = total;
    GroupMemoryBarrierWithGroupSync();
    collapseScanThreads(lane);
    uint prefix = carry;
    if (lane > 0u) prefix += partial[lane - 1u];
    for (uint k = 0u; k < 8u; ++k)
      if (base + k < instances && rootTask[base + k] != kBvhEmpty) {
        taskNumber[rootTask[base + k]] = prefix + running[k];
        rootNumber[base + k] = prefix + running[k];
      }
    carry += partial[127];
    GroupMemoryBarrierWithGroupSync();
  }
  if (lane != 0u) return;
  taskNumber[0] = 0u;
  uint levels = 0u;
  while (levels < kCollapseLevels && headers[levels * 4u + 3u] != 0u) levels += 1u;
  uint result = error[0];
  if (result == kBvhBuildErrorNone && missing != 0u) result = kBvhBuildErrorCapacity;
  const uint stack = taskStack[0];
  if (result == kBvhBuildErrorNone && stack > uint(kWideStackDeep)) result = kBvhBuildErrorDepth;
  BvhCollapseStatus out;
  out.result = uint4(result, levels, carry, stack);
  status[0] = out;
}

// 4. One level, top-down, one thread per (task, slot): the task's node at its number, the slot's
// interior child numbered after the node and the subtrees of the interior children in earlier slots
// (the host's depth-first emission). Every thread of a task derives the node's bounds.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_collapse_emit(uint3 thread: SV_DispatchThreadID) {
  const uint id = thread.x;
  const uint level = control.level.x;
  if (id / 8u >= headers[level * 4u + 3u]) return;
  const uint task = levelBase[level] + id / 8u, own = id % 8u;
  const bool bottom = (taskNode[task] & kCollapseBottom) != 0u;
  const uint number = taskNumber[task];
  const uint width = control.sizes.z;
  const uint outputNode = number * 10u; // PtWideNode: origin, scale, eight children

  uint low[3] = { kExactPositiveInfinity, kExactPositiveInfinity, kExactPositiveInfinity };
  uint high[3] = { kExactNegativeInfinity, kExactNegativeInfinity, kExactNegativeInfinity };
  uint occupied = 0u, next = number + 1u;
  for (uint slot = 0u; slot < 8u; ++slot) {
    const uint reference = taskSlots[task * 8u + slot];
    if (reference == kBvhEmpty) continue;
    const uint4 childLow = collapseLow(reference), childHigh = collapseHigh(reference);
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
    nodes[outputNode + 2u + own] = uint4(0u);
    if (own < 2u) nodes[outputNode + own] = uint4(0u);
    return;
  }
  uint scale[3] = {};
  for (uint a = 0u; a < 3u; ++a) {
    low[a] = ptExactNextAfter(low[a], false);
    high[a] = ptExactNextAfter(high[a], true);
    scale[a] = ptExactNextAfter(ptExactMultiply(ptExactSubtract(high[a], low[a]), kExactQuantumStep), true);
  }
  if (own == 0u) {
    const uint slots = width == 8u ? 8u | kWideOctantOrdered : occupied;
    nodes[outputNode] = uint4(low[0], low[1], low[2], slots);
  } else if (own == 1u) {
    nodes[outputNode + 1u] = uint4(scale[0], scale[1], scale[2], 0u);
  }
  const uint reference = taskSlots[task * 8u + own];
  if (reference == kBvhEmpty) {
    uint4 empty = uint4(0u);
    if (width == 8u) empty.w = kBvhEmpty;
    nodes[outputNode + 2u + own] = empty;
    return;
  }
  const uint4 childLow = collapseLow(reference), childHigh = collapseHigh(reference);
  uint data = kBvhLeafTag | childLow.w;
  if (childHigh.w == 0u) {
    const uint child = taskChild[task * 8u + own];
    data = next;
    if (child != kBvhEmpty) taskNumber[child] = next;
  } else if (bottom) {
    data = kBvhLeafTag | (childHigh.w << kBvhCountShift) | childLow.w;
  }
  nodes[outputNode + 2u + own] = uint4(collapsePack(childLow.x, childHigh.x, low[0], scale[0]),
                                       collapsePack(childLow.y, childHigh.y, low[1], scale[1]),
                                       collapsePack(childLow.z, childHigh.z, low[2], scale[2]), data);
}

// 5. One thread per level: the emit's indirect dispatch (eight threads per task) from the level's
// task count, for a collapse recorded without reading the counts back.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_collapse_emit_args(uint3 thread: SV_DispatchThreadID) {
  const uint id = thread.x;
  if (id > kCollapseLevels) return;
  emitHeaders[id] = uint4((headers[id * 4u + 3u] * 8u + 63u) / 64u, 1u, 1u, 0u);
}

// 6. Per instance: the traced row, the input row (the binary tree's roots, the current transforms)
// with its bottom level's wide root.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_collapse_rows(uint3 thread: SV_DispatchThreadID) {
  const uint id = thread.x;
  if (id >= control.sizes.w) return;
  TraceInstance row = input[id];
  row.blasRoot = rootNumber[id];
  output[id] = row;
}
