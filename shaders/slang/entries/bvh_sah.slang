// A GPU binned SAH builder: the CPU builder's rule (src/pt/Bvh.cpp buildTree: 16 bins on each
// axis over the centroid bounds, a traversal step and a triangle test costing the same, leaves of
// up to 8 triangles in the bottom levels and 1 instance in the TLAS) built top-down on the GPU,
// one tree level per dispatch. The records (triangles, then the instances' world boxes) come from
// the parallel LBVH's extents and instances kernels.
//
// A level's tasks are node ranges of an order of the records, in segment order. One subgroup
// (of at least 32 lanes, up to kSahTeams to a workgroup) serves one task: it bounds the range's
// boxes and centroids (exact min/max of ordered bits, reduced across the subgroup), bins it on
// three axes in shared memory, evaluates the split with the CPU builder's arithmetic (a lane per
// candidate boundary), and either makes the range a leaf or partitions it stably into the next
// order and queues its two halves. Subgroups take tickets in the order they start and serve the
// task of their ticket, so a decoupled look-back over the tasks' split counts (each waiting only
// on lower tickets, which have started) numbers the next level's tasks and the new nodes in task
// order: the build is the same bytes every run. The partition is stable and a range no boundary
// separates is halved in order, as the CPU builder does, so the tree is the CPU builder's.
//
// A large task (above kSahLarge records, in the first control.source.y levels) would leave one
// workgroup to walk a whole mesh, so its records are split into chunks of kSahChunk, and the
// level's chunks run before and after its task's workgroup: bvh_sah_large_bins bins them (and at
// level 0 bvh_sah_large_bounds bounds them first) into the task's slot through global atomics,
// exact and order-free as the shared ones; the task's workgroup reads the slot instead of walking
// the records, decides the same split and queues the children, allocating the large ones a slot;
// and bvh_sah_large_scatter partitions the chunks stably (a decoupled look-back over each task's
// chunks numbers their left records) while bounding each large child for the next level.
//
// Nodes are stored in their segment's range of the LBVH's internal nodes (flags.w onwards), by
// their rank in the segment's breadth-first creation order (the root first); a node's slots hold
// what the published node will, except that an internal child is its storage index. A climb then
// sizes every subtree, and the emit numbers each segment's nodes in preorder (the CPU builder's
// layout: a left child follows its parent, a right child its left sibling's subtree), places the
// segments contiguously, the TLAS first, and publishes the triangles in the leaves' order.
import pt_bvh_build;
import pt_exact_float;

static const uint kSahGroup = 128u;
static const uint kSahBins = 16u;
static const uint kSahMaxBottomLeaf = 8u;
static const uint kSahEmpty = 0xFFFFFFFFu;
static const float kSahHuge = 3.0e38f;
static const uint kSahLevels = 64u;
// A task's look-back state: (level + 1) << 25 | aggregate or prefix | splits. The level tag makes
// an earlier level's states read as unpublished, so the states are never cleared between levels.
static const uint kLookBackLevel = 25u;
static const uint kLookBackAggregate = 1u << 23u;
static const uint kLookBackPrefix = 1u << 24u;
static const uint kLookBackValue = kLookBackAggregate - 1u;
static const uint kSegmentStartLevel = 23u; // segmentStart: (level + 1) << 23 | the level's splits before the segment
static const uint kArrivalShift = 30u;      // arrivals: children arrived << 30 | their subtrees' nodes
static const uint kSahTeams = 4u;           // subgroups (tasks) per level workgroup of 128
static const uint kSahBinWords = 3u * kSahBins * 7u;
static const uint kSahCandidates = 3u * (kSahBins - 1u);  // (axis, boundary): the left side's last bin
static const uint kSahChunk = 256u;        // a large task's records per chunk workgroup
static const uint kSahLarge = 4u * kSahChunk;
static const uint kSahLargeLevels = 16u;    // levels with large-task slots (control.source.y of them used)
// A large task's slot: the box and centroid box (12 ordered-bit words), the bins (as binWords),
// the decision (as sahDecision), its task, its first chunk and its chunks.
static const uint kSlotBins = 12u;
static const uint kSlotDecision = kSlotBins + 3u * kSahBins * 7u;
static const uint kSlotTask = kSlotDecision + 4u;
static const uint kSlotChunkBase = kSlotTask + 1u;
static const uint kSlotChunks = kSlotTask + 2u;
static const uint kSahSlotWords = kSlotTask + 4u;

// A task: its segment, its range [first, first + count) of the order, and the node slot it fills:
// parent storage index * 2 + side, or kSahEmpty for a segment's root.
struct SahTask {
  uint4 range;
};

ConstantBuffer<BvhLbvhControl> control;
StructuredBuffer<BvhSegment> segments;
StructuredBuffer<BvhBuildRecord> records;
StructuredBuffer<TraceInstance> instances;
StructuredBuffer<uint> indices;
StructuredBuffer<float> vertices;
StructuredBuffer<SahTask> tasks;              // this level's
RWStructuredBuffer<SahTask> nextTasks;
StructuredBuffer<uint> order;                 // this level's order of the records
RWStructuredBuffer<uint> nextOrder;
RWStructuredBuffer<uint> finalOrder;          // each leaf's records, where it placed them
RWStructuredBuffer<float4> nodeSlots;         // per node storage index: four slots, as published
RWStructuredBuffer<uint> nodeParent;          // per node storage index: its slot in its parent, or kSahEmpty
RWStructuredBuffer<uint> subtree;             // per node storage index: its subtree's nodes
RWStructuredBuffer<uint> arrivals;            // per node storage index: see kArrivalShift
RWStructuredBuffer<uint> segmentNodes;        // per segment: nodes made (the highest rank + 1)
RWStructuredBuffer<uint> segmentDepth;        // per segment: its depth, as the CPU builder counts it
RWStructuredBuffer<uint> levelArgs;           // per level: groups, 1, 1, tasks (the next level's written by the last task)
RWStructuredBuffer<uint> error;
globallycoherent RWStructuredBuffer<uint> lookback; // [level] its tickets, [kSahLevels + t] task t's state
globallycoherent RWStructuredBuffer<uint> segmentStart;  // per segment, see kSegmentStartLevel
RWStructuredBuffer<uint> segmentCursor;       // per segment: nodes made in earlier levels (read here)
RWStructuredBuffer<uint> nextSegmentCursor;   // per segment: the same after this level
RWStructuredBuffer<uint> segmentBase;         // per segment: its first output node (the emit's)
RWStructuredBuffer<float4> nodes;
RWStructuredBuffer<float4> triangles;
RWStructuredBuffer<BvhBuildStatus2> status;
RWStructuredBuffer<uint> large;               // the large tasks' slots, level after level
StructuredBuffer<uint> largeBase;             // per level: its first slot (the next level's first: its end)
StructuredBuffer<uint> largeOf;               // per task of this level: its slot, or kSahEmpty
RWStructuredBuffer<uint> nextLargeOf;         // the same for the next level's tasks
RWStructuredBuffer<uint> largeArgs;           // per level: chunks, 1, 1, slots (the chunk kernels' dispatch)
StructuredBuffer<uint> chunkSlot;             // per chunk of this level: its task's slot
RWStructuredBuffer<uint> nextChunkSlot;
globallycoherent RWStructuredBuffer<uint> chunkState; // [level] the scatter's tickets, [kSahLargeLevels + c] chunk c's state

groupshared uint boxWords[12];                 // box then centroid box, ordered bits
groupshared uint binWords[3u * kSahBins * 7u]; // per axis and bin: box (6 words), count
groupshared uint teamBins[kSahTeams * kSahBinWords];  // the level kernel's, a task's per subgroup
groupshared uint teamCounter;
groupshared uint scanWords[kSahGroup];
groupshared float axisScale[3];                // per axis: bins per unit of centroid extent, 0 when flat
groupshared uint ticketSlot;

void fail(uint code) { InterlockedMax(error[0], code); }

float axisOf(float3 v, uint axis) { return axis == 0u ? v.x : (axis == 1u ? v.y : v.z); }

float boxArea(float3 low, float3 high) {
  if (!(low.x <= high.x)) return 0.0f;
  const float3 d = high - low;
  return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}

float3 orderedLow(uint base) {
  return float3(ptOrderedFloat(boxWords[base]), ptOrderedFloat(boxWords[base + 1u]), ptOrderedFloat(boxWords[base + 2u]));
}
float3 orderedHigh(uint base) {
  return float3(ptOrderedFloat(boxWords[base + 3u]), ptOrderedFloat(boxWords[base + 4u]), ptOrderedFloat(boxWords[base + 5u]));
}

// a / b as the host divides: Vulkan allows fp32 division an error of 3 ULP, which moves centres
// across bin boundaries (pt_exact_float).
float divideExact(float a, float b) { return asfloat(ptExactDivide(asuint(a), asuint(b))); }

// The CPU builder's bin of a centre on an axis.
uint binOf(float centre, float low, float scale) { return min(kSahBins - 1u, uint((centre - low) * scale)); }

// An empty slot for `task`, its chunks from chunkBase.
void initSlot(uint slot, uint task, uint chunkBase, uint chunks) {
  const uint base = slot * kSahSlotWords;
  for (uint w = 0u; w < 12u; ++w) large[base + w] = w % 6u < 3u ? 0xFFFFFFFFu : 0u;
  for (uint w = 0u; w < 3u * kSahBins * 7u; ++w) large[base + kSlotBins + w] = w % 7u < 3u ? 0xFFFFFFFFu : 0u;
  large[base + kSlotTask] = task;
  large[base + kSlotChunkBase] = chunkBase;
  large[base + kSlotChunks] = chunks;
}

// Grows a slot's box (words 0-5) and centroid box (6-11) by a shared box's twelve words.
void growSlot(uint slot, uint lane, uint word) {
  if (lane < 12u) {
    if (lane % 6u < 3u) InterlockedMin(large[slot * kSahSlotWords + lane], word);
    else InterlockedMax(large[slot * kSahSlotWords + lane], word);
  }
}

// Inclusive scan of scanWords over the workgroup.
void scanThreads(uint lane) {
  for (uint offset = 1u; offset < kSahGroup; offset <<= 1u) {
    const uint add = lane >= offset ? scanWords[lane - offset] : 0u;
    GroupMemoryBarrierWithGroupSync();
    scanWords[lane] += add;
    GroupMemoryBarrierWithGroupSync();
  }
}

float3 wordsLow(uint box[12], uint base) {
  return float3(ptOrderedFloat(box[base]), ptOrderedFloat(box[base + 1u]), ptOrderedFloat(box[base + 2u]));
}
float3 wordsHigh(uint box[12], uint base) {
  return float3(ptOrderedFloat(box[base + 3u]), ptOrderedFloat(box[base + 4u]), ptOrderedFloat(box[base + 5u]));
}

// A bin's box, from teamBins.
float3 binLow(uint word) {
  return float3(ptOrderedFloat(teamBins[word]), ptOrderedFloat(teamBins[word + 1u]), ptOrderedFloat(teamBins[word + 2u]));
}
float3 binHigh(uint word) {
  return float3(ptOrderedFloat(teamBins[word + 3u]), ptOrderedFloat(teamBins[word + 4u]),
                ptOrderedFloat(teamBins[word + 5u]));
}

// A subgroup per task (control.range.z: the level). The subgroups of a workgroup share nothing
// after it starts, so they sync only among their own lanes.
[shader("compute")]
[numthreads(128, 1, 1)]
void bvh_sah_level(uint index: SV_GroupIndex) {
  const uint level = control.range.z;
  if (index == 0u) teamCounter = 0u;
  GroupMemoryBarrierWithGroupSync();
  const uint lane = WaveGetLaneIndex(), width = WaveGetLaneCount();
  uint team = 0u;
  if (lane == 0u) InterlockedAdd(teamCounter, 1u, team);
  team = WaveReadLaneFirst(team);
  if (team >= kSahTeams) {
    if (lane == 0u) fail(kBvhBuildErrorCapacity);  // subgroups under 32 lanes: the host refuses them
    return;
  }
  uint ticket = 0u;
  if (lane == 0u) InterlockedAdd(lookback[level], 1u, ticket);
  const uint t = WaveReadLaneFirst(ticket);
  const uint taskCount = levelArgs[level * 4u + 3u];
  if (t >= taskCount) return;
  const uint bins = team * kSahBinWords;
  const SahTask task = tasks[t];
  const uint g = task.range.x, first = task.range.y, count = task.range.z, slot = task.range.w;
  const BvhSegment segment = segments[g];
  const uint maxLeaf = segment.flags.x != 0u ? 1u : kSahMaxBottomLeaf;
  // A large task's bounds and bins are in its slot, the chunk kernels' (the same values).
  const uint largeSlot = level < control.source.y ? largeOf[t] : kSahEmpty;
  const bool isLarge = largeSlot != kSahEmpty;

  // Bounds: the boxes and the centroids, each lane's then the subgroup's.
  uint box[12];
  if (isLarge) {
    for (uint w = 0u; w < 12u; ++w) box[w] = large[largeSlot * kSahSlotWords + w];
    for (uint w = lane; w < kSahBinWords; w += width) teamBins[bins + w] = large[largeSlot * kSahSlotWords + kSlotBins + w];
  } else {
    for (uint w = 0u; w < 12u; ++w) box[w] = w % 6u < 3u ? 0xFFFFFFFFu : 0u;
    for (uint p = lane; p < count; p += width) {
      const BvhBuildRecord record = records[order[first + p]];
      const float3 centre = (record.low.xyz + record.high.xyz) * 0.5f;
      const float values[12] = {record.low.x, record.low.y, record.low.z, record.high.x, record.high.y, record.high.z,
                                centre.x, centre.y, centre.z, centre.x, centre.y, centre.z};
      for (uint w = 0u; w < 12u; ++w)
        box[w] = w % 6u < 3u ? min(box[w], ptOrderedBits(values[w])) : max(box[w], ptOrderedBits(values[w]));
    }
    for (uint w = 0u; w < 12u; ++w) box[w] = w % 6u < 3u ? WaveActiveMin(box[w]) : WaveActiveMax(box[w]);
    for (uint w = lane; w < kSahBinWords; w += width) teamBins[bins + w] = w % 7u < 3u ? 0xFFFFFFFFu : 0u;
  }
  float3 low = wordsLow(box, 0u), high = wordsHigh(box, 0u);
  if (count == 0u) {
    low = float3(kSahHuge);
    high = float3(-kSahHuge);
  }
  const float3 centreLow = wordsLow(box, 6u), centreHigh = wordsHigh(box, 6u);
  const float3 extent = centreHigh - centreLow;
  float laneScale = 0.0f;  // lanes 0-2: an axis's bins per unit of centroid extent
  if (lane < 3u && axisOf(extent, lane) > 0.0f) laneScale = divideExact(float(kSahBins), axisOf(extent, lane));
  const float3 scale = float3(WaveReadLaneAt(laneScale, 0u), WaveReadLaneAt(laneScale, 1u), WaveReadLaneAt(laneScale, 2u));
  GroupMemoryBarrierWithWaveSync();

  // Bins on every axis with a centroid extent.
  if (count > 1u && !isLarge) {
    for (uint p = lane; p < count; p += width) {
      const BvhBuildRecord record = records[order[first + p]];
      const float3 centre = (record.low.xyz + record.high.xyz) * 0.5f;
      for (uint axis = 0u; axis < 3u; ++axis) {
        if (!(axisOf(extent, axis) > 0.0f)) continue;
        const uint base = bins + (axis * kSahBins + binOf(axisOf(centre, axis), axisOf(centreLow, axis), axisOf(scale, axis))) * 7u;
        InterlockedMin(teamBins[base], ptOrderedBits(record.low.x));
        InterlockedMin(teamBins[base + 1u], ptOrderedBits(record.low.y));
        InterlockedMin(teamBins[base + 2u], ptOrderedBits(record.low.z));
        InterlockedMax(teamBins[base + 3u], ptOrderedBits(record.high.x));
        InterlockedMax(teamBins[base + 4u], ptOrderedBits(record.high.y));
        InterlockedMax(teamBins[base + 5u], ptOrderedBits(record.high.z));
        InterlockedAdd(teamBins[base + 6u], 1u);
      }
    }
  }
  GroupMemoryBarrierWithWaveSync();

  // The split, as the CPU builder chooses it: the cheapest bin boundary over the three axes (its
  // first in axis then bin order among equal costs: the lowest candidate), a leaf when small
  // enough and no cheaper than its triangles, and halving when the chosen boundary would separate
  // nothing. Each lane costs its candidates (the boxes of the bins on either side, their counts).
  uint decision = 4u, bestBin = 0u, leftCount = count;
  if (count > 1u) {
    uint bestBits = 0xFFFFFFFFu, bestIndex = 0xFFFFFFFFu;
    for (uint c = lane; c < kSahCandidates; c += width) {
      const uint axis = c / (kSahBins - 1u), boundary = c % (kSahBins - 1u);
      if (!(axisOf(extent, axis) > 0.0f)) continue;
      float3 leftLow = float3(kSahHuge), leftHigh = float3(-kSahHuge), rightLow = float3(kSahHuge), rightHigh = float3(-kSahHuge);
      uint leftSide = 0u, rightSide = 0u;
      for (uint bin = 0u; bin < kSahBins; ++bin) {
        const uint word = bins + (axis * kSahBins + bin) * 7u;
        const uint binCount = teamBins[word + 6u];
        if (binCount == 0u) continue;
        if (bin <= boundary) {
          leftLow = min(leftLow, binLow(word));
          leftHigh = max(leftHigh, binHigh(word));
          leftSide += binCount;
        } else {
          rightLow = min(rightLow, binLow(word));
          rightHigh = max(rightHigh, binHigh(word));
          rightSide += binCount;
        }
      }
      if (leftSide == 0u || rightSide == 0u) continue;
      const float cost = boxArea(leftLow, leftHigh) * float(leftSide) + boxArea(rightLow, rightHigh) * float(rightSide);
      if (!(cost < kSahHuge)) continue;  // the CPU builder's first best must be below kHuge
      // Non-negative floats order as their bits.
      if (asuint(cost) < bestBits) {
        bestBits = asuint(cost);
        bestIndex = c;
      }
    }
    const uint minBits = WaveActiveMin(bestBits);
    const uint minIndex = WaveActiveMin(bestBits == minBits ? bestIndex : 0xFFFFFFFFu);
    int bestAxis = -1;
    float bestCost = kSahHuge;
    if (minBits != 0xFFFFFFFFu) {
      bestAxis = int(minIndex / (kSahBins - 1u));
      bestBin = minIndex % (kSahBins - 1u);
      bestCost = asfloat(minBits);
    }
    const float parentArea = max(boxArea(low, high), 1e-30f);
    const float splitCost = 1.0f + divideExact(bestCost, parentArea);
    if (!(count <= maxLeaf && (bestAxis < 0 || float(count) <= splitCost))) {
      decision = 3u;  // halve, unless the best boundary separates something
      leftCount = count / 2u;
      if (bestAxis >= 0) {
        uint left = 0u;
        for (uint bin = 0u; bin <= bestBin; ++bin) left += teamBins[bins + (uint(bestAxis) * kSahBins + bin) * 7u + 6u];
        if (left != 0u && left != count) {
          decision = uint(bestAxis);
          leftCount = left;
        }
      }
    }
  }
  const uint splits = decision < 4u ? 1u : 0u;

  uint before = 0u;
  if (lane == 0u) {
    // The task's place among the level's: its splits (0 or 1) published by decoupled look-back.
    const uint tag = (level + 1u) << kLookBackLevel;
    if (t == 0u) {
      InterlockedExchange(lookback[kSahLevels], tag | kLookBackPrefix | splits);
    } else {
      InterlockedExchange(lookback[kSahLevels + t], tag | kLookBackAggregate | splits);
      for (int b = int(t) - 1; b >= 0;) {
        uint state;
        InterlockedAdd(lookback[kSahLevels + uint(b)], 0u, state);
        if ((state >> kLookBackLevel) != level + 1u) continue;
        before += state & kLookBackValue;
        if ((state & kLookBackPrefix) != 0u) break;
        b -= 1;
      }
      InterlockedExchange(lookback[kSahLevels + t], tag | kLookBackPrefix | (before + splits));
    }
    if (isLarge) {
      const uint base = largeSlot * kSahSlotWords + kSlotDecision;
      large[base] = decision;
      large[base + 1u] = bestBin;
      large[base + 2u] = leftCount;
      large[base + 3u] = before;
    }

    // The node this task makes (every root does; below the root, a split), ranked in its
    // segment's breadth-first order: the segment's nodes of earlier levels, then its splits in
    // this level before this task (the level's splits before the segment's first task, which
    // that task publishes and later ones wait for).
    const bool firstOfSegment = t == 0u || tasks[t - 1u].range.x != g;
    const bool lastOfSegment = t + 1u == taskCount || tasks[t + 1u].range.x != g;
    uint segmentBefore = before;
    if (level > 0u) {
      if (firstOfSegment) {
        InterlockedExchange(segmentStart[g], ((level + 1u) << kSegmentStartLevel) | before);
      } else {
        for (;;) {
          uint state;
          InterlockedAdd(segmentStart[g], 0u, state);
          if ((state >> kSegmentStartLevel) == level + 1u) {
            segmentBefore = state & ((1u << kSegmentStartLevel) - 1u);
            break;
          }
        }
      }
    }
    const uint cursor = level == 0u ? 0u : segmentCursor[g];
    uint node = kSahEmpty;
    if (level == 0u) node = segment.flags.w;  // every root: rank 0
    else if (splits != 0u) node = segment.flags.w + cursor + (before - segmentBefore);
    if (node != kSahEmpty) {
      nodeParent[node] = slot;
      InterlockedMax(segmentNodes[g], node - segment.flags.w + 1u);
    }
    if (lastOfSegment) {
      const uint made = level == 0u ? 1u : cursor + (before + splits - segmentBefore);
      nextSegmentCursor[g] = made;
    }
    InterlockedMax(segmentDepth[g], level + 1u);
    if (level + 1u >= kSahLevels && splits != 0u) fail(kBvhBuildErrorDepth);

    // This task in its parent's slot: a leaf, or its node.
    if (slot != kSahEmpty) {
      uint data = kSahEmpty, leafCount = 0u;
      if (decision == 4u) {
        leafCount = count;
        data = segment.flags.x != 0u ? records[order[first]].identity.x : segment.range.w + (first - segment.range.x);
      } else {
        data = node;
      }
      nodeSlots[slot * 2u] = float4(low, asfloat(data));
      nodeSlots[slot * 2u + 1u] = float4(high, asfloat(leafCount));
    } else if (decision == 4u) {
      // A root that is a leaf (or an empty segment): its node holds it and an empty child, as the
      // CPU builder emits it.
      const uint rootSlot = segment.flags.w * 2u;
      uint data = kSahEmpty;
      if (count > 0u) data = segment.flags.x != 0u ? records[order[first]].identity.x : segment.range.w;
      nodeSlots[rootSlot * 2u] = count > 0u ? float4(low, asfloat(data)) : float4(float3(kSahHuge), asfloat(kSahEmpty));
      nodeSlots[rootSlot * 2u + 1u] = count > 0u ? float4(high, asfloat(count)) : float4(float3(-kSahHuge), asfloat(0u));
      nodeSlots[rootSlot * 2u + 2u] = float4(float3(kSahHuge), asfloat(kSahEmpty));
      nodeSlots[rootSlot * 2u + 3u] = float4(float3(-kSahHuge), asfloat(0u));
    }

    // The two halves queued for the next level; the last task sizes it.
    if (splits != 0u) {
      nextTasks[before * 2u].range = uint4(g, first, leftCount, node * 2u);
      nextTasks[before * 2u + 1u].range = uint4(g, first + leftCount, count - leftCount, node * 2u + 1u);
      // A large child's slot, its chunks and their entries in the next level's chunk table.
      for (uint side = 0u; side < 2u; ++side) {
        const uint childCount = side == 0u ? leftCount : count - leftCount;
        uint childSlot = kSahEmpty;
        if (level + 1u < control.source.y && childCount > kSahLarge) {
          uint index, chunkBase;
          const uint chunks = (childCount + kSahChunk - 1u) / kSahChunk;
          InterlockedAdd(largeArgs[(level + 1u) * 4u + 3u], 1u, index);
          InterlockedAdd(largeArgs[(level + 1u) * 4u], chunks, chunkBase);
          childSlot = largeBase[level + 1u] + index;
          if (childSlot < largeBase[level + 2u]) {
            initSlot(childSlot, before * 2u + side, chunkBase, chunks);
            for (uint c = 0u; c < chunks; ++c) nextChunkSlot[chunkBase + c] = childSlot;
          } else {
            fail(kBvhBuildErrorCapacity);  // the host's bound does not hold: a builder bug
            childSlot = kSahEmpty;
          }
        }
        nextLargeOf[before * 2u + side] = childSlot;
      }
    }
    if (t + 1u == taskCount) {
      // A workgroup per kSahTeams tasks (128 / the subgroup size of them run in one).
      const uint nextCount = (before + splits) * 2u;
      const uint teams = min(kSahTeams, 128u / width);
      levelArgs[(level + 1u) * 4u] = (nextCount + teams - 1u) / teams;
      levelArgs[(level + 1u) * 4u + 1u] = 1u;
      levelArgs[(level + 1u) * 4u + 2u] = 1u;
      levelArgs[(level + 1u) * 4u + 3u] = nextCount;
    }
  }
  before = WaveReadLaneFirst(before);

  // A leaf's records are final where they are; a split's go to the next order, a stable
  // partition by the chosen boundary (or the range halved): a large task's in its chunks.
  if (isLarge) return;
  if (decision == 4u) {
    for (uint p = lane; p < count; p += width) finalOrder[first + p] = order[first + p];
    return;
  }
  const float axisLow = decision < 3u ? axisOf(centreLow, decision) : 0.0f;
  const float axisScaled = decision < 3u ? axisOf(scale, decision) : 0.0f;
  uint leftBefore = 0u;
  for (uint round = 0u; round < count; round += width) {
    const uint p = round + lane;
    uint record = 0u;
    bool left = false;
    if (p < count) {
      record = order[first + p];
      if (decision < 3u) {
        const BvhBuildRecord r = records[record];
        left = binOf(axisOf((r.low.xyz + r.high.xyz) * 0.5f, decision), axisLow, axisScaled) <= bestBin;
      } else {
        left = p < leftCount;
      }
    }
    const uint leftAhead = WavePrefixCountBits(left);  // this round's left records before this lane
    if (p < count) {
      const uint lefts = leftBefore + leftAhead;
      nextOrder[first + (left ? lefts : leftCount + (p - lefts))] = record;
    }
    leftBefore += WaveActiveCountBits(left);
  }
}

// A chunk of a large task: its slot and its range of the task's records.
struct SahChunk {
  uint slot;
  uint base;   // the slot's first word
  uint first;  // the chunk's first position in the order
  uint count;  // its records
  uint index;  // within its task
  SahTask task;
};

SahChunk chunkAt(uint c) {
  SahChunk chunk;
  chunk.slot = chunkSlot[c];
  chunk.base = chunk.slot * kSahSlotWords;
  chunk.task = tasks[large[chunk.base + kSlotTask]];
  chunk.index = c - large[chunk.base + kSlotChunkBase];
  chunk.first = chunk.task.range.y + chunk.index * kSahChunk;
  chunk.count = min(kSahChunk, chunk.task.range.z - chunk.index * kSahChunk);
  return chunk;
}

// Level 0: a large root's chunk grows its slot's boxes (deeper levels' come from the scatter).
[shader("compute")]
[numthreads(128, 1, 1)]
void bvh_sah_large_bounds(uint lane: SV_GroupIndex, uint3 group: SV_GroupID) {
  if (lane < 12u) boxWords[lane] = lane % 6u < 3u ? 0xFFFFFFFFu : 0u;
  GroupMemoryBarrierWithGroupSync();
  const SahChunk chunk = chunkAt(group.x);
  for (uint p = lane; p < chunk.count; p += kSahGroup) {
    const BvhBuildRecord record = records[order[chunk.first + p]];
    const float3 centre = (record.low.xyz + record.high.xyz) * 0.5f;
    InterlockedMin(boxWords[0], ptOrderedBits(record.low.x));
    InterlockedMin(boxWords[1], ptOrderedBits(record.low.y));
    InterlockedMin(boxWords[2], ptOrderedBits(record.low.z));
    InterlockedMax(boxWords[3], ptOrderedBits(record.high.x));
    InterlockedMax(boxWords[4], ptOrderedBits(record.high.y));
    InterlockedMax(boxWords[5], ptOrderedBits(record.high.z));
    InterlockedMin(boxWords[6], ptOrderedBits(centre.x));
    InterlockedMin(boxWords[7], ptOrderedBits(centre.y));
    InterlockedMin(boxWords[8], ptOrderedBits(centre.z));
    InterlockedMax(boxWords[9], ptOrderedBits(centre.x));
    InterlockedMax(boxWords[10], ptOrderedBits(centre.y));
    InterlockedMax(boxWords[11], ptOrderedBits(centre.z));
  }
  GroupMemoryBarrierWithGroupSync();
  growSlot(chunk.slot, lane, boxWords[min(lane, 11u)]);
}

// A large task's chunk bins its records on the axes with a centroid extent, into its slot.
[shader("compute")]
[numthreads(128, 1, 1)]
void bvh_sah_large_bins(uint lane: SV_GroupIndex, uint3 group: SV_GroupID) {
  for (uint w = lane; w < 3u * kSahBins * 7u; w += kSahGroup) binWords[w] = w % 7u < 3u ? 0xFFFFFFFFu : 0u;
  const SahChunk chunk = chunkAt(group.x);
  if (lane < 12u) boxWords[lane] = large[chunk.base + lane];
  GroupMemoryBarrierWithGroupSync();
  const float3 centreLow = orderedLow(6u), centreHigh = orderedHigh(6u);
  if (lane < 3u) {
    const float extent = axisOf(centreHigh, lane) - axisOf(centreLow, lane);
    axisScale[lane] = extent > 0.0f ? divideExact(float(kSahBins), extent) : 0.0f;
  }
  GroupMemoryBarrierWithGroupSync();
  for (uint p = lane; p < chunk.count; p += kSahGroup) {
    const BvhBuildRecord record = records[order[chunk.first + p]];
    const float3 centre = (record.low.xyz + record.high.xyz) * 0.5f;
    for (uint axis = 0u; axis < 3u; ++axis) {
      const float scale = axisScale[axis];
      if (scale == 0.0f) continue;
      const uint base = (axis * kSahBins + binOf(axisOf(centre, axis), axisOf(centreLow, axis), scale)) * 7u;
      InterlockedMin(binWords[base], ptOrderedBits(record.low.x));
      InterlockedMin(binWords[base + 1u], ptOrderedBits(record.low.y));
      InterlockedMin(binWords[base + 2u], ptOrderedBits(record.low.z));
      InterlockedMax(binWords[base + 3u], ptOrderedBits(record.high.x));
      InterlockedMax(binWords[base + 4u], ptOrderedBits(record.high.y));
      InterlockedMax(binWords[base + 5u], ptOrderedBits(record.high.z));
      InterlockedAdd(binWords[base + 6u], 1u);
    }
  }
  GroupMemoryBarrierWithGroupSync();
  for (uint w = lane; w < 3u * kSahBins * 7u; w += kSahGroup) {
    const uint value = binWords[w];
    const uint word = chunk.base + kSlotBins + w;
    if (w % 7u == 6u) {
      if (value != 0u) InterlockedAdd(large[word], value);
    } else if (w % 7u < 3u) {
      if (value != 0xFFFFFFFFu) InterlockedMin(large[word], value);
    } else if (value != 0u) {
      InterlockedMax(large[word], value);
    }
  }
}

groupshared uint childWords[24];  // the scatter's two children: box and centroid box each

// Whether a large task's record at task position p goes left.
bool goesLeft(SahChunk chunk, uint decision, uint bestBin, uint leftCount, float axisLow, float scale, uint p,
              uint record) {
  if (decision >= 3u) return chunk.index * kSahChunk + p < leftCount;
  const BvhBuildRecord r = records[record];
  return binOf(axisOf((r.low.xyz + r.high.xyz) * 0.5f, decision), axisLow, scale) <= bestBin;
}

// A large task's chunk, in ticket order: its left records counted and numbered after the earlier
// chunks' (decoupled look-back within the task), then scattered stably into the next order; each
// large child's slot grows by the chunk's records on its side.
[shader("compute")]
[numthreads(128, 1, 1)]
void bvh_sah_large_scatter(uint lane: SV_GroupIndex) {
  const uint level = control.range.z;
  if (lane == 0u) {
    uint ticket;
    InterlockedAdd(chunkState[level], 1u, ticket);
    ticketSlot = ticket;
  }
  if (lane < 24u) childWords[lane] = lane % 6u < 3u ? 0xFFFFFFFFu : 0u;
  if (lane < 12u) boxWords[lane] = 0u;
  GroupMemoryBarrierWithGroupSync();
  const uint c = ticketSlot;
  const SahChunk chunk = chunkAt(c);
  if (lane < 12u) boxWords[lane] = large[chunk.base + lane];
  const uint decision = large[chunk.base + kSlotDecision];
  const uint bestBin = large[chunk.base + kSlotDecision + 1u];
  const uint leftCount = large[chunk.base + kSlotDecision + 2u];
  const uint before = large[chunk.base + kSlotDecision + 3u];
  const uint first = chunk.task.range.y;
  GroupMemoryBarrierWithGroupSync();
  const float3 centreLow = orderedLow(6u), centreHigh = orderedHigh(6u);
  const float axisLow = decision < 3u ? axisOf(centreLow, decision) : 0.0f;
  const float scale = decision < 3u ? divideExact(float(kSahBins), axisOf(centreHigh, decision) - axisLow) : 0.0f;

  // The chunk's left records, and those of the task's chunks before it.
  uint mine = 0u;
  for (uint p = lane; p < chunk.count; p += kSahGroup)
    mine += goesLeft(chunk, decision, bestBin, leftCount, axisLow, scale, p, order[chunk.first + p]) ? 1u : 0u;
  scanWords[lane] = mine;
  GroupMemoryBarrierWithGroupSync();
  scanThreads(lane);
  if (lane == 0u) {
    const uint lefts = scanWords[kSahGroup - 1u];
    const uint tag = (level + 1u) << kLookBackLevel;
    uint prior = 0u;
    if (chunk.index == 0u) {
      InterlockedExchange(chunkState[kSahLargeLevels + c], tag | kLookBackPrefix | lefts);
    } else {
      InterlockedExchange(chunkState[kSahLargeLevels + c], tag | kLookBackAggregate | lefts);
      for (uint b = c - 1u;;) {
        uint state;
        InterlockedAdd(chunkState[kSahLargeLevels + b], 0u, state);
        if ((state >> kLookBackLevel) != level + 1u) continue;
        prior += state & kLookBackValue;
        if ((state & kLookBackPrefix) != 0u) break;
        b -= 1u;
      }
      InterlockedExchange(chunkState[kSahLargeLevels + c], tag | kLookBackPrefix | (prior + lefts));
    }
    ticketSlot = prior;
  }
  GroupMemoryBarrierWithGroupSync();
  uint leftBefore = ticketSlot;  // left records at task positions before the chunk
  const uint start = chunk.index * kSahChunk;  // the chunk's first task position

  // The stable scatter, a round of kSahGroup records at a time.
  for (uint round = 0u; round < chunk.count; round += kSahGroup) {
    const uint p = round + lane;
    uint record = 0u;
    bool left = false;
    if (p < chunk.count) {
      record = order[chunk.first + p];
      left = goesLeft(chunk, decision, bestBin, leftCount, axisLow, scale, p, record);
    }
    scanWords[lane] = left ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    scanThreads(lane);
    const uint leftThrough = scanWords[lane];
    const uint roundLeft = scanWords[kSahGroup - 1u];
    GroupMemoryBarrierWithGroupSync();
    if (p < chunk.count) {
      const uint position = start + p;  // in the task
      const uint lefts = leftBefore + leftThrough - (left ? 1u : 0u);
      nextOrder[first + (left ? lefts : leftCount + (position - lefts))] = record;
      const BvhBuildRecord r = records[record];
      const float3 centre = (r.low.xyz + r.high.xyz) * 0.5f;
      const uint box = left ? 0u : 12u;
      InterlockedMin(childWords[box], ptOrderedBits(r.low.x));
      InterlockedMin(childWords[box + 1u], ptOrderedBits(r.low.y));
      InterlockedMin(childWords[box + 2u], ptOrderedBits(r.low.z));
      InterlockedMax(childWords[box + 3u], ptOrderedBits(r.high.x));
      InterlockedMax(childWords[box + 4u], ptOrderedBits(r.high.y));
      InterlockedMax(childWords[box + 5u], ptOrderedBits(r.high.z));
      InterlockedMin(childWords[box + 6u], ptOrderedBits(centre.x));
      InterlockedMin(childWords[box + 7u], ptOrderedBits(centre.y));
      InterlockedMin(childWords[box + 8u], ptOrderedBits(centre.z));
      InterlockedMax(childWords[box + 9u], ptOrderedBits(centre.x));
      InterlockedMax(childWords[box + 10u], ptOrderedBits(centre.y));
      InterlockedMax(childWords[box + 11u], ptOrderedBits(centre.z));
    }
    leftBefore += roundLeft;
  }
  GroupMemoryBarrierWithGroupSync();
  for (uint side = 0u; side < 2u; ++side) {
    const uint childSlot = nextLargeOf[before * 2u + side];
    if (childSlot != kSahEmpty) growSlot(childSlot, lane, childWords[side * 12u + min(lane, 11u)]);
  }
}

// One thread: every segment's first output node (the TLAS first, then the bottom levels in
// order), and the status.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_sah_bases(uint3 thread: SV_DispatchThreadID) {
  if (thread.x != 0u) return;
  const uint segmentCount = control.sizes.y;
  const uint top = segmentCount - 1u;
  segmentBase[top] = 0u;
  uint running = segmentNodes[top];
  uint bottomDepth = 0u;
  for (uint g = 0u; g < top; ++g) {
    segmentBase[g] = running;
    running += segmentNodes[g];
    bottomDepth = max(bottomDepth, segmentDepth[g]);
  }
  segmentBase[segmentCount] = running;  // every node
  uint result = error[0];
  const uint topDepth = segmentDepth[top];
  if (result == kBvhBuildErrorNone && topDepth + bottomDepth > control.geometry.z) result = kBvhBuildErrorDepth;
  BvhBuildStatus2 out;
  out.result = uint4(result, topDepth, bottomDepth, 0u);
  out.counts = uint4(running, control.sizes.w, control.range.x, 0u);
  out.extra = uint4(0u, 0u, 0u, 0u);
  out.cost = float4(0.0f);
  status[0] = out;
}

uint segmentOfNode(uint segmentCount, uint node) {
  uint lowIndex = 0u, highIndex = segmentCount;
  while (highIndex - lowIndex > 1u) {
    const uint middle = (lowIndex + highIndex) >> 1u;
    if (segments[middle].flags.w <= node) lowIndex = middle;
    else highIndex = middle;
  }
  return lowIndex;
}

// A node's child on a side: its storage index when an internal node, else kSahEmpty.
uint internalChild(uint storage, uint side) {
  const uint data = asuint(nodeSlots[storage * 4u + side * 2u].w);
  return asuint(nodeSlots[storage * 4u + side * 2u + 1u].w) == 0u ? data : kSahEmpty;
}

uint internalChildren(uint storage) {
  return (internalChild(storage, 0u) != kSahEmpty ? 1u : 0u) + (internalChild(storage, 1u) != kSahEmpty ? 1u : 0u);
}

// The storage index made for a segment's rank, or kSahEmpty past its nodes.
uint storageNode(uint storage, out uint g) {
  g = segmentOfNode(control.sizes.y, storage);
  return storage - segments[g].flags.w < segmentNodes[g] ? storage : kSahEmpty;
}

// Per node without internal children: climbs to the root, sizing subtrees. A parent's two
// arrivals and their sizes add into one word, so the second arrival reads both sizes with the
// same atomic that tells it it is second.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_sah_sizes(uint3 thread: SV_DispatchThreadID) {
  if (thread.x >= control.sizes.z) return;
  uint g;
  uint node = storageNode(thread.x, g);
  if (node == kSahEmpty || internalChildren(node) != 0u) return;
  uint size = 1u;
  for (;;) {
    subtree[node] = size;
    const uint parent = nodeParent[node];
    if (parent == kSahEmpty) return;
    node = parent >> 1u;
    uint prior;
    InterlockedAdd(arrivals[node], (1u << kArrivalShift) + size, prior);
    if ((prior >> kArrivalShift) + 1u < internalChildren(node)) return;
    size += 1u + (prior & ((1u << kArrivalShift) - 1u));
  }
}

// A node's preorder rank in its segment: one past its parent's, and past the left sibling's
// subtree for a right child.
uint preorderRank(uint node) {
  uint rank = 0u;
  for (uint parent = nodeParent[node]; parent != kSahEmpty; parent = nodeParent[node]) {
    node = parent >> 1u;
    rank += 1u;
    if ((parent & 1u) != 0u) {
      const uint left = internalChild(node, 0u);
      if (left != kSahEmpty) rank += subtree[left];
    }
  }
  return rank;
}

// Per node storage index: its published node at its preorder rank, internal children as their
// output nodes; per position of the final order: its triangle (bvh_lbvh_emit's layout).
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_sah_emit(uint3 thread: SV_DispatchThreadID) {
  uint g = 0u;
  const uint storage = thread.x < control.sizes.z ? storageNode(thread.x, g) : kSahEmpty;
  if (storage != kSahEmpty) {
    const uint rank = preorderRank(storage);
    const uint output = segmentBase[g] + rank;
    const uint left = internalChild(storage, 0u);
    for (uint side = 0u; side < 2u; ++side) {
      float4 low = nodeSlots[storage * 4u + side * 2u];
      const float4 high = nodeSlots[storage * 4u + side * 2u + 1u];
      if (internalChild(storage, side) != kSahEmpty)
        low.w = asfloat(output + 1u + (side == 1u && left != kSahEmpty ? subtree[left] : 0u));
      nodes[output * 4u + side * 2u] = low;
      nodes[output * 4u + side * 2u + 1u] = high;
    }
  }
  const uint position = thread.x;
  if (position >= control.sizes.w) return;
  const BvhBuildRecord record = records[finalOrder[position]];
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
  const uint outputTriangle = (segment.range.w + position - segment.range.x) * 3u;
  triangles[outputTriangle] = float4(buildPosition(vertices, i0), asfloat(primitive));
  triangles[outputTriangle + 1u] = float4(buildPosition(vertices, i1), 0.0f);
  triangles[outputTriangle + 2u] = float4(buildPosition(vertices, i2), 0.0f);
}
