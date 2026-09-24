// Builds the software BVH in the layout shaders/pt/bvh.h traverses: one bottom level per
// instance over its triangles in object space, one top level over the instances' world
// bounds, as the hardware structure is built. Binned SAH, run across threads.
#pragma once
#include "pt/Tracing.h"

#include <cstdint>
#include <vector>

namespace pt {

struct Bvh {
  std::vector<float4> nodes;      // the top level first, root at 0, then each bottom level
  std::vector<float4> triangles;  // object-space vertices in leaf order, three per triangle
};

struct BvhStatistics {
  uint instances = 0;
  uint triangles = 0;
  uint topNodes = 0;
  uint bottomNodes = 0;
  uint topDepth = 0;
  uint bottomDepth = 0;   // the deepest bottom level
  double sahCost = 0.0;   // summed over bottom levels, relative to one triangle test
  double milliseconds = 0.0;
};

// Fills instances[i].blasRoot and triangleOffset. triangleCounts[i] is the number of
// triangles behind instance i, starting at its firstIndex.
BvhStatistics buildBvh(const std::vector<float> &vertices, const std::vector<uint> &indices,
                       std::vector<TraceInstance> &instances, const std::vector<uint> &triangleCounts, Bvh &bvh,
                       unsigned threads);

// Updates triangle vertices and every bound without changing topology or leaf order.
// Returns elapsed milliseconds. Intended for V6 refit evaluation; callers rebuild when
// topology, primitive counts, or instance ordering changes.
double refitBvh(const std::vector<float> &vertices, const std::vector<uint> &indices,
                const std::vector<TraceInstance> &instances, Bvh &bvh);

std::vector<PtEmissiveTriangle> buildEmissiveTriangles(const std::vector<float> &vertices,
    const std::vector<uint> &indices, const std::vector<TraceInstance> &instances,
    const std::vector<uint> &triangleCounts, const std::vector<Material> &materials);

} // namespace pt
