#pragma once
#include "gpu/Descriptors.h"

#include <memory>

namespace basalt {

struct RayStageDescription {
  std::string entry;
  VkShaderStageFlagBits stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
};

// A shader group: the indices of its stages in the pipeline's stage list.
struct RayShaderGroup {
  VkRayTracingShaderGroupTypeKHR type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  std::uint32_t general = VK_SHADER_UNUSED_KHR, closestHit = VK_SHADER_UNUSED_KHR, anyHit = VK_SHADER_UNUSED_KHR,
                intersection = VK_SHADER_UNUSED_KHR;
};

// Which groups the shader binding table's records hold: one raygen record, then the miss, hit and
// callable regions, a record per listed group. Records carry the group's handle only.
struct ShaderBindingRecords {
  std::uint32_t raygen = 0;
  std::vector<std::uint32_t> miss, hit, callable;
};

// Owns one ray pipeline, its stages' ShaderLayout (sets by ParameterBlock, as a Program's) and an
// aligned, immutable SBT. Replacing one of these while work is in flight requires the caller to
// wait or retire it.
class RayPipeline {
public:
  RayPipeline(const Context &context, const std::vector<RayStageDescription> &stageDescriptions,
              const std::vector<RayShaderGroup> &groups, const ShaderBindingRecords &records,
              std::uint32_t recursionDepth = 1);
  ~RayPipeline();
  RayPipeline(const RayPipeline &) = delete;
  RayPipeline &operator=(const RayPipeline &) = delete;

  const Shader &shader(std::size_t index) const { return *shaders.at(index); }
  // Its sets are allocated, written and bound through this.
  const ShaderLayout &shaderLayout() const { return *sets; }

  // Binds the pipeline; the caller binds its sets through shaderLayout(), then traces.
  void bind(VkCommandBuffer command) const;
  void traceRays(VkCommandBuffer command, std::uint32_t width, std::uint32_t height = 1, std::uint32_t depth = 1) const;
  // Launch dimensions read on the device from a VkTraceRaysIndirectCommandKHR at `address`,
  // written by an earlier stage. Requires Context::rayPipelineIndirectSupported; the caller
  // bounds the dimensions by maxRayDispatchInvocationCount.
  void traceRaysIndirect(VkCommandBuffer command, VkDeviceAddress address) const;

  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline handle = VK_NULL_HANDLE;
  VkStridedDeviceAddressRegionKHR raygenRegion{}, missRegion{}, hitRegion{}, callableRegion{};
  VkDeviceSize sbtBytes = 0;

private:
  void checkDimensions(std::uint32_t width, std::uint32_t height, std::uint32_t depth) const;
  const Context &context;
  std::vector<std::unique_ptr<Shader>> shaders;
  std::unique_ptr<ShaderLayout> sets;
  Buffer sbt;
};

} // namespace basalt
