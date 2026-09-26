// GPU-resident LBVH construction. One compute invocation deliberately performs the baseline build
// serially: its stable radix passes are linear, deterministic and easy to validate. The parallel
// builders compete with this correctness baseline. A build is several dispatches, each its own
// submission (control.part: some bottom levels, then the TLAS), so none outlasts the driver's
// timeout; each resumes from the status the one before left.
import pt_bvh_build;

static const uint kEmpty = 0xFFFFFFFFu;
static const float kHuge = 3.0e38f;
static const uint kBuilderStack = 64u;

StructuredBuffer<BvhBuildDescriptor> descriptors;
StructuredBuffer<TraceInstance> instances;
StructuredBuffer<uint> indices;
StructuredBuffer<float> vertices;
RWStructuredBuffer<float4> nodes;
RWStructuredBuffer<float4> triangles;
RWStructuredBuffer<BvhBuildRecord> scratchA;
RWStructuredBuffer<BvhBuildRecord> scratchB;
RWStructuredBuffer<BvhBuildStatus> status;
RWStructuredBuffer<BvhBuildRecord> instanceRecords;
StructuredBuffer<BvhReference> references;  // split clipping's, when a descriptor's output.w is 1
ConstantBuffer<BvhBuildControl> control;

// Six stable 5-bit passes over the Morton codes, scratchA to scratchB and back.
void stableRadix(uint count, inout uint passes) {
  for (uint pass = 0u; pass < 6u; ++pass) {
    uint counts[32] = {};
    for (uint i = 0u; i < count; ++i) {
      const BvhBuildRecord value = (pass & 1u) == 0u ? scratchA[i] : scratchB[i];
      counts[(value.identity.y >> (pass * 5u)) & 31u] += 1u;
    }
    uint offsets[32] = {};
    uint sum = 0u;
    for (uint digit = 0u; digit < 32u; ++digit) {
      offsets[digit] = sum;
      sum += counts[digit];
    }
    for (uint i = 0u; i < count; ++i) {
      const BvhBuildRecord value = (pass & 1u) == 0u ? scratchA[i] : scratchB[i];
      const uint digit = (value.identity.y >> (pass * 5u)) & 31u;
      if ((pass & 1u) == 0u) scratchB[offsets[digit]++] = value;
      else scratchA[offsets[digit]++] = value;
    }
    passes += 1u;
  }
}

// The records are scratchA's, in stable Morton order. The leading bits records a and b share
// in their (Morton code, index in the level) key; 64 when identical.
uint commonPrefix(uint a, uint b) {
  const uint mortonDifference = scratchA[a].identity.y ^ scratchA[b].identity.y;
  if (mortonDifference != 0u) return leadingZeros(mortonDifference);
  const uint identityDifference = scratchA[a].identity.w ^ scratchA[b].identity.w;
  return identityDifference == 0u ? 64u : 32u + leadingZeros(identityDifference);
}

// The last position in [first, last) sharing more than the range's common prefix with first
// (binary search): the left child is [first, split], the right [split + 1, last].
uint splitRange(uint first, uint last) {
  const uint common = commonPrefix(first, last);
  uint split = first;
  uint step = last - first;
  do {
    step = (step + 1u) >> 1u;
    const uint candidate = split + step;
    if (candidate < last && commonPrefix(first, candidate) > common) split = candidate;
  } while (step > 1u);
  return split;
}

float4 bits(float3 value, uint data) { return float4(value, asfloat(data)); }

// Emits the two-child traversal layout of scratchA's first `count` records.
bool emitTree(uint count, uint nodeBase, uint leafBase, bool instanceLeaves, uint nodeCapacity, inout uint nodesWritten,
              inout uint treeDepth, inout uint maximumStack) {
  if (nodeBase >= nodeCapacity) return false;
  if (count == 0u) {
    nodes[nodeBase * 4u] = bits(float3(kHuge), kEmpty);
    nodes[nodeBase * 4u + 1u] = bits(float3(-kHuge), 0u);
    nodes[nodeBase * 4u + 2u] = bits(float3(kHuge), kEmpty);
    nodes[nodeBase * 4u + 3u] = bits(float3(-kHuge), 0u);
    nodesWritten += 1u;
    treeDepth = 1u;
    return true;
  }
  if (count == 1u) {
    const uint data = instanceLeaves ? scratchA[0].identity.x : leafBase;
    nodes[nodeBase * 4u] = bits(scratchA[0].low.xyz, data);
    nodes[nodeBase * 4u + 1u] = bits(scratchA[0].high.xyz, 1u);
    nodes[nodeBase * 4u + 2u] = bits(float3(kHuge), kEmpty);
    nodes[nodeBase * 4u + 3u] = bits(float3(-kHuge), 0u);
    nodesWritten += 1u;
    treeDepth = 1u;
    return true;
  }

  uint firstStack[64] = {};
  uint lastStack[64] = {};
  uint nodeStack[64] = {};
  uint depthStack[64] = {};
  uint stackSize = 1u;
  firstStack[0] = 0u;
  lastStack[0] = count - 1u;
  nodeStack[0] = nodeBase;
  depthStack[0] = 1u;
  uint nextNode = nodeBase + 1u;
  while (stackSize > 0u) {
    maximumStack = max(maximumStack, stackSize);
    stackSize -= 1u;
    const uint first = firstStack[stackSize];
    const uint last = lastStack[stackSize];
    const uint node = nodeStack[stackSize];
    const uint depth = depthStack[stackSize];
    treeDepth = max(treeDepth, depth);
    const uint split = splitRange(first, last);
    for (uint side = 0u; side < 2u; ++side) {
      const uint begin = side == 0u ? first : split + 1u;
      const uint end = side == 0u ? split : last;
      const bool leaf = begin == end;
      uint data = 0u;
      if (leaf) {
        data = instanceLeaves ? scratchA[begin].identity.x : leafBase + begin;
      } else {
        if (nextNode >= nodeCapacity || stackSize >= kBuilderStack) return false;
        data = nextNode++;
        firstStack[stackSize] = begin;
        lastStack[stackSize] = end;
        nodeStack[stackSize] = data;
        depthStack[stackSize] = depth + 1u;
        stackSize += 1u;
      }
      // Interior bounds are published by the reverse-order bottom-up pass below.
      const float3 low = leaf ? scratchA[begin].low.xyz : float3(kHuge);
      const float3 high = leaf ? scratchA[begin].high.xyz : float3(-kHuge);
      const uint slot = node * 4u + side * 2u;
      nodes[slot] = bits(low, data);
      nodes[slot + 1u] = bits(high, leaf ? 1u : 0u);
    }
    nodesWritten += 1u;
  }
  // Children are always allocated after their parent. Reversing node order therefore makes every
  // child's bounds available before its parent consumes them.
  for (uint cursor = nextNode; cursor > nodeBase; --cursor) {
    const uint node = cursor - 1u;
    for (uint side = 0u; side < 2u; ++side) {
      const uint slot = node * 4u + side * 2u;
      const uint child = asuint(nodes[slot].w);
      const uint countValue = asuint(nodes[slot + 1u].w);
      if (countValue == 0u && child != kEmpty) {
        const uint childSlot = child * 4u;
        const float3 low = min(nodes[childSlot].xyz, nodes[childSlot + 2u].xyz);
        const float3 high = max(nodes[childSlot + 1u].xyz, nodes[childSlot + 3u].xyz);
        nodes[slot] = bits(low, child);
        nodes[slot + 1u] = bits(high, countValue);
      }
    }
  }
  return true;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void bvh_build(uint3 id: SV_DispatchThreadID) {
  if (id.x != 0u) return;
  BvhBuildStatus result = status[0];  // the earlier parts' (zero before the first)
  if (result.result.x != kBvhBuildErrorNone) return;
  if (control.geometry.w != kBvhBuildLayoutVersion || control.sizes.z == 0u || control.sizes.w == 0u) {
    result.result.x = kBvhBuildErrorCapacity;
    status[0] = result;
    return;
  }

  const uint instanceCount = control.sizes.x;
  uint radixPasses = result.counts.w;
  // This part's BLASes, retaining one world-space bound record per instance.
  for (uint descriptorIndex = control.part.x; descriptorIndex < min(control.part.y, instanceCount); ++descriptorIndex) {
    const BvhBuildDescriptor descriptor = descriptors[descriptorIndex];
    const uint count = descriptor.geometry.z;
    if (count > control.sizes.z || descriptor.output.x >= control.sizes.w) {
      result.result.x = kBvhBuildErrorCapacity;
      status[0] = result;
      return;
    }
    // The records: the triangles, or with split clipping the references (from the triangle base).
    const bool referenced = descriptor.output.w != 0u;
    float3 objectLow = float3(kHuge), objectHigh = float3(-kHuge);
    for (uint index = 0u; index < count; ++index) {
      BvhReference reference = {};
      if (referenced) reference = references[descriptor.output.y + index];
      const uint primitive = referenced ? asuint(reference.low.w) : index;
      const uint base = descriptor.geometry.x + primitive * 3u;
      if (base + 2u >= control.geometry.x) {
        result.result.x = kBvhBuildErrorGeometry;
        status[0] = result;
        return;
      }
      const uint i0 = indices[base] + descriptor.geometry.y;
      const uint i1 = indices[base + 1u] + descriptor.geometry.y;
      const uint i2 = indices[base + 2u] + descriptor.geometry.y;
      if (max(i0, max(i1, i2)) >= control.geometry.y) {
        result.result.x = kBvhBuildErrorGeometry;
        status[0] = result;
        return;
      }
      const float3 v0 = buildPosition(vertices, i0);
      const float3 v1 = buildPosition(vertices, i1);
      const float3 v2 = buildPosition(vertices, i2);
      if (!all(isfinite(v0)) || !all(isfinite(v1)) || !all(isfinite(v2))) {
        result.result.x = kBvhBuildErrorNonFinite;
        status[0] = result;
        return;
      }
      BvhBuildRecord record;
      record.identity = uint4(primitive, 0u, 0u, index);
      record.low = float4(referenced ? reference.low.xyz : min(v0, min(v1, v2)), 0.0f);
      record.high = float4(referenced ? reference.high.xyz : max(v0, max(v1, v2)), 0.0f);
      scratchA[index] = record;
      objectLow = min(objectLow, record.low.xyz);
      objectHigh = max(objectHigh, record.high.xyz);
    }
    for (uint index = 0u; index < count; ++index) {
      BvhBuildRecord record = scratchA[index];
      record.identity.y = morton3((record.low.xyz + record.high.xyz) * 0.5f, objectLow, objectHigh);
      scratchA[index] = record;
    }
    stableRadix(count, radixPasses);
    for (uint sorted = 0u; sorted < count; ++sorted) {
      const uint primitive = scratchA[sorted].identity.x;
      const uint base = descriptor.geometry.x + primitive * 3u;
      const uint outputIndex = (descriptor.output.y + sorted) * 3u;
      const float3 v0 = buildPosition(vertices, indices[base] + descriptor.geometry.y);
      const float3 v1 = buildPosition(vertices, indices[base + 1u] + descriptor.geometry.y);
      const float3 v2 = buildPosition(vertices, indices[base + 2u] + descriptor.geometry.y);
      triangles[outputIndex] = bits(v0, primitive);
      triangles[outputIndex + 1u] = float4(v1, 0.0f);
      triangles[outputIndex + 2u] = float4(v2, 0.0f);
    }
    uint depth = 0u;
    uint nodesWritten = result.counts.x;
    uint maximumStack = result.result.w;
    if (!emitTree(count, descriptor.output.x, descriptor.output.y, false, control.sizes.w, nodesWritten, depth, maximumStack)) {
      result.result.x = kBvhBuildErrorDepth;
      status[0] = result;
      return;
    }
    result.counts.x = nodesWritten;
    result.result.w = maximumStack;
    result.result.z = max(result.result.z, depth);
    result.counts.y += count;

    BvhBuildRecord world;
    world.identity = uint4(descriptor.geometry.w, 0u, 0u, descriptorIndex);
    world.low = float4(kHuge);
    world.high = float4(-kHuge);
    if (count == 0u) {
      world.low = float4(0.0f);
      world.high = float4(0.0f);
    } else {
      const TraceInstance instance = instances[descriptor.geometry.w];
      for (uint corner = 0u; corner < 8u; ++corner) {
        const float3 point = float3((corner & 1u) != 0u ? objectHigh.x : objectLow.x, (corner & 2u) != 0u ? objectHigh.y : objectLow.y,
                                    (corner & 4u) != 0u ? objectHigh.z : objectLow.z);
        const float3 transformed = transformPoint(instance, point);
        world.low.xyz = min(world.low.xyz, transformed);
        world.high.xyz = max(world.high.xyz, transformed);
      }
    }
    instanceRecords[descriptorIndex] = world;
  }

  if (control.part.z == 0u) {
    result.counts.w = radixPasses;
    status[0] = result;
    return;
  }

  // Stable Morton order for the TLAS. Copy the instance records because scratchA may still hold
  // the last BLAS's sorted records.
  float3 sceneLow = float3(kHuge), sceneHigh = float3(-kHuge);
  for (uint i = 0u; i < instanceCount; ++i) {
    scratchA[i] = instanceRecords[i];
    sceneLow = min(sceneLow, scratchA[i].low.xyz);
    sceneHigh = max(sceneHigh, scratchA[i].high.xyz);
  }
  for (uint i = 0u; i < instanceCount; ++i) {
    BvhBuildRecord record = scratchA[i];
    record.identity.y = morton3((record.low.xyz + record.high.xyz) * 0.5f, sceneLow, sceneHigh);
    scratchA[i] = record;
  }
  stableRadix(instanceCount, radixPasses);
  uint topDepth = 0u;
  uint nodesWritten = result.counts.x;
  uint maximumStack = result.result.w;
  if (!emitTree(instanceCount, 0u, 0u, true, control.sizes.w, nodesWritten, topDepth, maximumStack)) {
    result.result.x = kBvhBuildErrorDepth;
    status[0] = result;
    return;
  }
  result.counts.x = nodesWritten;
  result.result.w = maximumStack;
  result.result.y = topDepth;
  if (topDepth + result.result.z > control.geometry.z) result.result.x = kBvhBuildErrorDepth;
  result.counts.z = instanceCount;
  result.counts.w = radixPasses;
  status[0] = result;
}
