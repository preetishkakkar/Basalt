#pragma once
#include "gpu/Descriptors.h"

#include <memory>

namespace basalt {

struct RayStageDescription {
  std::string entry;
  VkShaderStageFlagBits stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
};

// Owns one ray pipeline, its merged set-0 layout and an aligned, immutable SBT.
// Replacing one of these while work is in flight requires the caller to wait or retire it.
class RayPipeline {
public:
  RayPipeline(const Context &context, const std::vector<RayStageDescription> &stageDescriptions,
              const std::vector<m2v::host::RayShaderGroup> &groups,
              const m2v::host::ShaderBindingRecord &raygen,
              const std::vector<m2v::host::ShaderBindingRecord> &miss,
              const std::vector<m2v::host::ShaderBindingRecord> &hit,
              const std::vector<m2v::host::ShaderBindingRecord> &callable = {},
              std::uint32_t recursionDepth = 1);
  ~RayPipeline();
  RayPipeline(const RayPipeline &) = delete;
  RayPipeline &operator=(const RayPipeline &) = delete;

  const Shader &shader(std::size_t index) const { return *shaders.at(index); }
  void trace(VkCommandBuffer command, VkDescriptorSet set, std::uint32_t width,
             std::uint32_t height = 1, std::uint32_t depth = 1) const;
  // Launch dimensions read on the device from a VkTraceRaysIndirectCommandKHR at `address`,
  // written by an earlier stage. Requires Context::rayPipelineIndirectSupported; the caller
  // bounds the dimensions by maxRayDispatchInvocationCount.
  void traceIndirect(VkCommandBuffer command, VkDescriptorSet set, VkDeviceAddress address) const;

  VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline handle = VK_NULL_HANDLE;
  VkStridedDeviceAddressRegionKHR raygenRegion{}, missRegion{}, hitRegion{}, callableRegion{};
  VkDeviceSize sbtBytes = 0;

private:
  const Context &context;
  std::vector<std::unique_ptr<Shader>> shaders;
  std::vector<VkSampler> immutableSamplers;
  Buffer sbt;
};

} // namespace basalt
