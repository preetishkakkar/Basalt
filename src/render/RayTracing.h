// One BLAS per primitive, a TLAS with an instance each, and the table that turns a hit into indices, a material and texture slots.
#pragma once
#include "gpu/Resources.h"
#include "gpu/Uploader.h"
#include "scene/Scene.h"

#include <vector>

namespace basalt {

// Mirrors PrimitiveInfo in shaders/common.metal; indexed by the hit's instance id.
struct PrimitiveInfo {
  std::uint32_t firstIndex;     // Into the shared index buffer.
  std::uint32_t vertexOffset;   // Added to every index.
  std::uint32_t material;       // Into the material buffer.
  std::uint32_t slots;          // Texture table slots: base colour, metallic-roughness, emissive, 8 bits each.
};

class SceneAccelerationStructure {
public:
  SceneAccelerationStructure(const Context &context, Uploader &uploader, const Scene &scene,
                             const std::vector<std::uint32_t> &materialSlots);
  ~SceneAccelerationStructure();
  SceneAccelerationStructure(const SceneAccelerationStructure &) = delete;
  SceneAccelerationStructure &operator=(const SceneAccelerationStructure &) = delete;

  VkAccelerationStructureKHR topLevel = VK_NULL_HANDLE;
  Buffer primitiveInfo;
  std::uint32_t instanceCount = 0;

private:
  void build(Uploader &uploader, const Scene &scene, const std::vector<std::uint32_t> &materialSlots);
  void release();
  const Context &context;
  std::vector<VkAccelerationStructureKHR> bottomLevels;
  std::vector<Buffer> bottomBuffers;
  Buffer topBuffer;
  Buffer instanceBuffer;
};

} // namespace basalt
