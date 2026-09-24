// GPU-resident LBVH construction. One compute invocation deliberately performs the
// baseline build serially: its stable radix passes are linear, deterministic and easy
// to validate. Later parallel builders compete with this correctness baseline.
#include "shared/prelude.h"
#include "pt/bvh_build.h"

constant uint kEmpty = 0xFFFFFFFFu;
constant float kHuge = 3.0e38f;
constant uint kBuilderStack = 64u;

inline float3 buildPosition(const device float *vertices, uint index) {
  const uint base = index * kVertexFloats;
  return float3(vertices[base], vertices[base + 1u], vertices[base + 2u]);
}

inline uint expandMorton(uint value) {
  value &= 0x000003ffu;
  value = (value | (value << 16u)) & 0x030000FFu;
  value = (value | (value << 8u)) & 0x0300F00Fu;
  value = (value | (value << 4u)) & 0x030C30C3u;
  value = (value | (value << 2u)) & 0x09249249u;
  return value;
}

inline uint morton3(float3 point, float3 low, float3 high) {
  const float3 extent = high - low;
  float3 unit = float3(0.5f);
  if (extent.x > 0.0f) unit.x = (point.x - low.x) / extent.x;
  if (extent.y > 0.0f) unit.y = (point.y - low.y) / extent.y;
  if (extent.z > 0.0f) unit.z = (point.z - low.z) / extent.z;
  unit = clamp(unit, float3(0.0f), float3(0.999999f));
  const uint x = uint(unit.x * 1024.0f);
  const uint y = uint(unit.y * 1024.0f);
  const uint z = uint(unit.z * 1024.0f);
  return expandMorton(x) | (expandMorton(y) << 1u) | (expandMorton(z) << 2u);
}

inline void stableRadix(device BvhBuildRecord *a, device BvhBuildRecord *b, uint count,
                        thread uint &passes) {
  for (uint pass = 0u; pass < 6u; ++pass) {
    uint counts[32] = {};
    for (uint i = 0u; i < count; ++i) {
      BvhBuildRecord value;
      if ((pass & 1u) == 0u) value = a[i];
      else value = b[i];
      counts[(value.identity.y >> (pass * 5u)) & 31u] += 1u;
    }
    uint offsets[32] = {};
    uint sum = 0u;
    for (uint digit = 0u; digit < 32u; ++digit) {
      offsets[digit] = sum;
      sum += counts[digit];
    }
    for (uint i = 0u; i < count; ++i) {
      BvhBuildRecord value;
      if ((pass & 1u) == 0u) value = a[i];
      else value = b[i];
      const uint digit = (value.identity.y >> (pass * 5u)) & 31u;
      if ((pass & 1u) == 0u) b[offsets[digit]++] = value;
      else a[offsets[digit]++] = value;
    }
    passes += 1u;
  }
}

inline uint commonPrefix(const device BvhBuildRecord *records, uint a, uint b) {
  const uint mortonDifference = records[a].identity.y ^ records[b].identity.y;
  if (mortonDifference != 0u) return clz(mortonDifference);
  const uint identityDifference = records[a].identity.x ^ records[b].identity.x;
  return identityDifference == 0u ? 64u : 32u + clz(identityDifference);
}

inline uint splitRange(const device BvhBuildRecord *records, uint first, uint last) {
  const uint common = commonPrefix(records, first, last);
  uint split = first;
  uint step = last - first;
  do {
    step = (step + 1u) >> 1u;
    const uint candidate = split + step;
    if (candidate < last && commonPrefix(records, first, candidate) > common) split = candidate;
  } while (step > 1u);
  return split;
}

inline float4 bits(float3 value, uint data) { return float4(value, as_type<float>(data)); }

// Emits the two-child traversal layout. Records are already in stable Morton order.
inline bool emitTree(const device BvhBuildRecord *records, uint count, uint nodeBase,
                     uint leafBase, bool instanceLeaves, uint nodeCapacity,
                     device float4 *nodes, thread uint &nodesWritten,
                     thread uint &treeDepth, thread uint &maximumStack) {
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
    const uint data = instanceLeaves ? records[0].identity.x : leafBase;
    nodes[nodeBase * 4u] = bits(records[0].low.xyz, data);
    nodes[nodeBase * 4u + 1u] = bits(records[0].high.xyz, 1u);
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
    const uint split = splitRange(records, first, last);
    const uint childFirst[2] = {first, split + 1u};
    const uint childLast[2] = {split, last};
    for (uint side = 0u; side < 2u; ++side) {
      const uint begin = childFirst[side], end = childLast[side];
      const bool leaf = begin == end;
      uint data = 0u;
      if (leaf) data = instanceLeaves ? records[begin].identity.x : leafBase + begin;
      else {
        if (nextNode >= nodeCapacity || stackSize >= kBuilderStack) return false;
        data = nextNode++;
        firstStack[stackSize] = begin;
        lastStack[stackSize] = end;
        nodeStack[stackSize] = data;
        depthStack[stackSize] = depth + 1u;
        stackSize += 1u;
      }
      // Interior bounds are published by the reverse-order bottom-up pass below.
      const float3 low = leaf ? records[begin].low.xyz : float3(kHuge);
      const float3 high = leaf ? records[begin].high.xyz : float3(-kHuge);
      const uint slot = node * 4u + side * 2u;
      nodes[slot] = bits(low, data);
      nodes[slot + 1u] = bits(high, leaf ? 1u : 0u);
    }
    nodesWritten += 1u;
  }
  // Children are always allocated after their parent. Reversing node order therefore
  // makes every child's bounds available before its parent consumes them.
  for (uint cursor = nextNode; cursor > nodeBase; --cursor) {
    const uint node = cursor - 1u;
    for (uint side = 0u; side < 2u; ++side) {
      const uint slot = node * 4u + side * 2u;
      const uint child = as_type<uint>(nodes[slot].w);
      const uint countValue = as_type<uint>(nodes[slot + 1u].w);
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

inline float3 transformPoint(TraceInstance instance, float3 point) {
  const float4 p = float4(point.x, point.y, point.z, 1.0f);
  return float3(dot(instance.objectToWorld0, p), dot(instance.objectToWorld1, p),
                dot(instance.objectToWorld2, p));
}

kernel void bvh_build(const device BvhBuildDescriptor *descriptors [[buffer(0)]],
                      const device TraceInstance *instances [[buffer(1)]],
                      const device uint *indices [[buffer(2)]],
                      const device float *vertices [[buffer(3)]],
                      device float4 *nodes [[buffer(4)]],
                      device float4 *triangles [[buffer(5)]],
                      device BvhBuildRecord *scratchA [[buffer(6)]],
                      device BvhBuildRecord *scratchB [[buffer(7)]],
                      device BvhBuildStatus *status [[buffer(8)]],
                      device BvhBuildRecord *instanceRecords [[buffer(9)]],
                      constant BvhBuildControl &control [[buffer(10)]],
                      uint id [[thread_position_in_grid]]) {
  if (id != 0u) return;
  BvhBuildStatus result;
  result.result = uint4(0u);
  result.counts = uint4(0u);
  if (control.geometry.w != kBvhBuildLayoutVersion || control.sizes.z == 0u ||
      control.sizes.w == 0u) {
    result.result.x = kBvhBuildErrorCapacity;
    *status = result;
    return;
  }

  const uint instanceCount = control.sizes.x;
  uint radixPasses = 0u;
  // BLAS construction, retaining one world-space bound record per instance in scratchB.
  for (uint descriptorIndex = 0u; descriptorIndex < instanceCount; ++descriptorIndex) {
    const BvhBuildDescriptor descriptor = descriptors[descriptorIndex];
    const uint count = descriptor.geometry.z;
    if (count > control.sizes.z || descriptor.output.x >= control.sizes.w) {
      result.result.x = kBvhBuildErrorCapacity;
      *status = result;
      return;
    }
    float3 objectLow = float3(kHuge), objectHigh = float3(-kHuge);
    for (uint primitive = 0u; primitive < count; ++primitive) {
      const uint base = descriptor.geometry.x + primitive * 3u;
      if (base + 2u >= control.geometry.x) {
        result.result.x = kBvhBuildErrorGeometry;
        *status = result;
        return;
      }
      const uint i0 = indices[base] + descriptor.geometry.y;
      const uint i1 = indices[base + 1u] + descriptor.geometry.y;
      const uint i2 = indices[base + 2u] + descriptor.geometry.y;
      if (max(i0, max(i1, i2)) >= control.geometry.y) {
        result.result.x = kBvhBuildErrorGeometry;
        *status = result;
        return;
      }
      const float3 v0 = buildPosition(vertices, i0);
      const float3 v1 = buildPosition(vertices, i1);
      const float3 v2 = buildPosition(vertices, i2);
      if (!all(isfinite(v0)) || !all(isfinite(v1)) || !all(isfinite(v2))) {
        result.result.x = kBvhBuildErrorNonFinite;
        *status = result;
        return;
      }
      BvhBuildRecord record;
      record.identity = uint4(primitive, 0u, 0u, 0u);
      record.low = float4(min(v0, min(v1, v2)), 0.0f);
      record.high = float4(max(v0, max(v1, v2)), 0.0f);
      scratchA[primitive] = record;
      objectLow = min(objectLow, record.low.xyz);
      objectHigh = max(objectHigh, record.high.xyz);
    }
    for (uint primitive = 0u; primitive < count; ++primitive) {
      BvhBuildRecord record = scratchA[primitive];
      record.identity.y = morton3((record.low.xyz + record.high.xyz) * 0.5f, objectLow, objectHigh);
      scratchA[primitive] = record;
    }
    stableRadix(scratchA, scratchB, count, radixPasses);
    for (uint sorted = 0u; sorted < count; ++sorted) {
      const uint primitive = scratchA[sorted].identity.x;
      const uint base = descriptor.geometry.x + primitive * 3u;
      const uint output = (descriptor.output.y + sorted) * 3u;
      const float3 v0 = buildPosition(vertices, indices[base] + descriptor.geometry.y);
      const float3 v1 = buildPosition(vertices, indices[base + 1u] + descriptor.geometry.y);
      const float3 v2 = buildPosition(vertices, indices[base + 2u] + descriptor.geometry.y);
      triangles[output] = bits(v0, primitive);
      triangles[output + 1u] = float4(v1, 0.0f);
      triangles[output + 2u] = float4(v2, 0.0f);
    }
    uint depth = 0u;
    uint nodesWritten = result.counts.x;
    uint maximumStack = result.result.w;
    if (!emitTree(scratchA, count, descriptor.output.x, descriptor.output.y, false,
                  control.sizes.w, nodes, nodesWritten, depth, maximumStack)) {
      result.result.x = kBvhBuildErrorDepth;
      *status = result;
      return;
    }
    result.counts.x = nodesWritten;
    result.result.w = maximumStack;
    result.result.z = max(result.result.z, depth);
    result.counts.y += count;

    BvhBuildRecord world;
    world.identity = uint4(descriptor.geometry.w, 0u, 0u, 0u);
    world.low = float4(kHuge);
    world.high = float4(-kHuge);
    if (count == 0u) {
      world.low = float4(0.0f);
      world.high = float4(0.0f);
    } else {
      const TraceInstance instance = instances[descriptor.geometry.w];
      for (uint corner = 0u; corner < 8u; ++corner) {
        const float3 point = float3((corner & 1u) != 0u ? objectHigh.x : objectLow.x,
                                    (corner & 2u) != 0u ? objectHigh.y : objectLow.y,
                                    (corner & 4u) != 0u ? objectHigh.z : objectLow.z);
        const float3 transformed = transformPoint(instance, point);
        world.low.xyz = min(world.low.xyz, transformed);
        world.high.xyz = max(world.high.xyz, transformed);
      }
    }
    instanceRecords[descriptorIndex] = world;
  }

  // Stable Morton order for the TLAS. Copy scratchB because scratchA may still contain
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
  stableRadix(scratchA, scratchB, instanceCount, radixPasses);
  uint topDepth = 0u;
  uint nodesWritten = result.counts.x;
  uint maximumStack = result.result.w;
  if (!emitTree(scratchA, instanceCount, 0u, 0u, true, control.sizes.w, nodes,
                nodesWritten, topDepth, maximumStack)) {
    result.result.x = kBvhBuildErrorDepth;
    *status = result;
    return;
  }
  result.counts.x = nodesWritten;
  result.result.w = maximumStack;
  result.result.y = topDepth;
  if (topDepth + result.result.z > control.geometry.z) result.result.x = kBvhBuildErrorDepth;
  result.counts.z = instanceCount;
  result.counts.w = radixPasses;
  *status = result;
}
