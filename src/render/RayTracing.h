// One BLAS per trace instance and a TLAS over them, in the TraceScene's order, so a hit's
// instance index is the row of the instance table.
#pragma once
#include "gpu/Resources.h"
#include "gpu/Uploader.h"
#include "render/TraceScene.h"
#include "scene/Scene.h"

#include <vector>

namespace basalt {

class SceneAccelerationStructure {
public:
  SceneAccelerationStructure(const Context &context, Uploader &uploader, const Scene &scene,
                             const TraceScene &trace);
  ~SceneAccelerationStructure();
  SceneAccelerationStructure(const SceneAccelerationStructure &) = delete;
  SceneAccelerationStructure &operator=(const SceneAccelerationStructure &) = delete;

  VkAccelerationStructureKHR topLevel = VK_NULL_HANDLE;
  std::uint32_t instanceCount = 0;

private:
  void build(Uploader &uploader, const Scene &scene, const TraceScene &trace);
  void release();
  const Context &context;
  std::vector<VkAccelerationStructureKHR> bottomLevels;
  std::vector<Buffer> bottomBuffers;
  Buffer topBuffer;
  Buffer instanceBuffer;
};

} // namespace basalt
