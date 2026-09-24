#pragma once
#include "gpu/Resources.h"
#include "pt/Bvh.h"
#include "render/TraceScene.h"

namespace basalt {

class Context;
class Uploader;

struct GpuBvhBuildResult {
  Buffer nodes;
  Buffer triangles;
  std::vector<pt::TraceInstance> instances;
  pt::BvhStatistics statistics{};
  VkDeviceSize scratchBytes = 0;
  VkDeviceSize outputBytes = 0;
  double milliseconds = 0.0;
  std::uint32_t radixPasses = 0;
  std::uint32_t maximumBuilderStack = 0;
};

// Builds the traversal structure entirely from the resident GPU geometry. The host only
// supplies primitive counts/offsets and reads the small failure/status record.
GpuBvhBuildResult buildGpuBvh(const Context &context, Uploader &uploader,
                              const Scene &scene, const TraceScene &trace);

} // namespace basalt

