#pragma once
#include "pt/Bvh.h"

#include <array>

namespace pt {

// The wide tracers' node (pt_wide_bvh.slang's PtWideNode): origin.xyz the quantization origin, w
// the slot count (kWideSlotMask) and kWideOctantOrdered as uint bits; scale.xyz one conservative
// 16-bit step; per child the packed low and high bounds in xyz and its data in w.
using QuantizedWideNode = PtWideNode;
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
// Wide nodes use their decoded, conservative child boxes. The reference for bvh_cost.slang.
struct LayoutCost {
  double top = 0.0;
  double bottom = 0.0;
};
LayoutCost binaryLayoutCost(const std::vector<float4> &nodes, const std::vector<TraceInstance> &instances);
LayoutCost wideLayoutCost(const std::vector<QuantizedWideNode> &nodes, const std::vector<TraceInstance> &instances);
// The wide tree traced with the shared tracer (pt_wide_bvh.slang, children sorted far to near),
// with view's materials, geometry and textures and the tree's own instances and triangles.
PtHit traceWideBvh(const WideBvh &bvh, const TraceView &view, float3 origin, float3 direction, float tMax, uint mask,
                   uint seed, float2 cone, uint anyHit);
bool cpuAvx2Available();
// The same traversal with the eight slab tests in AVX2.
PtHit traceWideBvhAvx2(const WideBvh &bvh, const TraceView &view, float3 origin, float3 direction, float tMax, uint mask,
                       uint seed, float2 cone, uint anyHit);

} // namespace pt
