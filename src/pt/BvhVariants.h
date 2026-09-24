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
  uint maximumStack = 0; // proven upper bound when every child bound intersects
  double milliseconds = 0.0;
};

WideBvh buildWideBvh(const Bvh &binary, const std::vector<TraceInstance> &instances, uint width);
PtHit traceWideBvh(const WideBvh &bvh, const Material *materials, const uint *indices,
                   const float *vertices, const HostTextures &textures, float3 origin,
                   float3 direction, float tMax, uint mask, uint seed, float2 cone, uint anyHit);
bool cpuAvx2Available();
PtHit traceWideBvhAvx2(const WideBvh &bvh, const Material *materials, const uint *indices,
                       const float *vertices, const HostTextures &textures, float3 origin,
                       float3 direction, float tMax, uint mask, uint seed, float2 cone, uint anyHit);

} // namespace pt
