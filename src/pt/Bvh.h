// Builds the software BVH in the layout pt_bvh.slang traverses: one bottom level per
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

// Early split clipping (Ernst, Greiner 2007): the bottom levels' triangles as references, each
// with the box of its part of the triangle. A part whose box's surface area is above `factor`
// times its bottom level's mean triangle box area is halved at the middle of its box's longest
// axis, each half the triangle clipped to it, until small enough or six halvings deep (at most
// kMaxClipReferences parts; a triangle's in depth-first order, the low half first). The boxes
// cover every triangle: clipped in double precision, rounded outwards to float. The GPU makes the
// same references (bvh_clip.slang, GpuLbvhBuilder::clipReferences).
inline constexpr uint kMaxClipReferences = 64;
// --bvh-split-clipping on: on Sponza the Morton-order builders' trees trace 13-19% faster with it.
inline constexpr float kDefaultClipFactor = 1.0f;
struct BvhReferences {
  std::vector<BvhReference> references;  // each bottom level's in instance order, a triangle's together
  std::vector<uint> counts;              // per instance
  double milliseconds = 0.0;
};
BvhReferences clipReferences(const std::vector<float> &vertices, const std::vector<uint> &indices,
                             const std::vector<TraceInstance> &instances, const std::vector<uint> &triangleCounts,
                             float factor, unsigned threads = 1);

// Fills instances[i].blasRoot and triangleOffset. triangleCounts[i] is the number of
// triangles behind instance i, starting at its firstIndex. With references, the bottom levels
// are built over them and publish a triangle per reference (statistics.triangles counts them).
BvhStatistics buildBvh(const std::vector<float> &vertices, const std::vector<uint> &indices,
                       std::vector<TraceInstance> &instances, const std::vector<uint> &triangleCounts, Bvh &bvh,
                       unsigned threads, const BvhReferences *references = nullptr);

// Updates triangle vertices and every bound without changing topology or leaf order.
// Returns elapsed milliseconds. Intended for V6 refit evaluation; callers rebuild when
// topology, primitive counts, or instance ordering changes.
double refitBvh(const std::vector<float> &vertices, const std::vector<uint> &indices,
                const std::vector<TraceInstance> &instances, Bvh &bvh);

std::vector<PtEmissiveTriangle> buildEmissiveTriangles(const std::vector<float> &vertices,
    const std::vector<uint> &indices, const std::vector<TraceInstance> &instances,
    const std::vector<uint> &triangleCounts, const std::vector<Material> &materials);

} // namespace pt
