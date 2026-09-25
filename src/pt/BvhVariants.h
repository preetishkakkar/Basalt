#pragma once
#include "pt/Bvh.h"

#include <array>

namespace pt {

struct alignas(16) QuantizedWideNode {
  float4 origin; // xyz quantization origin, w child count as uint bits
  float4 scale;  // xyz one conservative 16-bit quantization step
  std::array<uint4, 8> children{}; // xyz packed low/high bounds, w child data
};
static_assert(sizeof(QuantizedWideNode) == 160);

struct WideBvh {
  std::vector<QuantizedWideNode> nodes;
  std::vector<float4> triangles;
  std::vector<TraceInstance> instances;
  uint width = 0;
  uint maximumStack = 0; // proven upper bound when every child bound intersects (the wide kernels' stack must hold it)
  double milliseconds = 0.0;
};

WideBvh buildWideBvh(const Bvh &binary, const std::vector<TraceInstance> &instances, uint width);

// SAH cost of a published tree, by walking it from its roots: per tree, 1 for an interior
// root plus each reachable child's area relative to the root's times one (interior) or its
// count (leaf). bottom sums the instances' bottom levels, as BvhStatistics::sahCost does.
// Wide nodes use their decoded, conservative child boxes. The reference for bvh_cost.metal.
struct LayoutCost {
  double top = 0.0;
  double bottom = 0.0;
};
LayoutCost binaryLayoutCost(const std::vector<float4> &nodes, const std::vector<TraceInstance> &instances);
LayoutCost wideLayoutCost(const std::vector<QuantizedWideNode> &nodes, const std::vector<TraceInstance> &instances);
PtHit traceWideBvh(const WideBvh &bvh, const Material *materials, const uint *indices,
                   const float *vertices, const HostTextures &textures, float3 origin,
                   float3 direction, float tMax, uint mask, uint seed, float2 cone, uint anyHit);
bool cpuAvx2Available();
PtHit traceWideBvhAvx2(const WideBvh &bvh, const Material *materials, const uint *indices,
                       const float *vertices, const HostTextures &textures, float3 origin,
                       float3 direction, float tMax, uint mask, uint seed, float2 cone, uint anyHit);

} // namespace pt
