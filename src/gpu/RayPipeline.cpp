#include "gpu/RayPipeline.h"

#include "core/Log.h"

#include <algorithm>
#include <cstring>

namespace basalt {
namespace {

VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) {
  if (!alignment || (alignment & (alignment - 1)) != 0)
    throw Error("ray pipeline reported an invalid SBT base alignment");
  return (value + alignment - 1) & ~(alignment - 1);
}

} // namespace

RayPipeline::RayPipeline(const Context &ctx,
                         const std::vector<RayStageDescription> &stageDescriptions,
                         const std::vector<m2v::host::RayShaderGroup> &groups,
                         const m2v::host::ShaderBindingRecord &raygen,
                         const std::vector<m2v::host::ShaderBindingRecord> &miss,
                         const std::vector<m2v::host::ShaderBindingRecord> &hit,
                         const std::vector<m2v::host::ShaderBindingRecord> &callable,
                         std::uint32_t recursionDepth)
    : context(ctx) {
  if (!ctx.rayPipelineSupported) throw Error("the selected device does not support Vulkan ray pipelines");
  if (stageDescriptions.empty()) throw Error("a ray pipeline needs shader stages");

  std::vector<m2v::host::Stage> reflections;
  std::vector<VkPipelineShaderStageCreateInfo> stageInfos;
  shaders.reserve(stageDescriptions.size());
  reflections.reserve(stageDescriptions.size());
  stageInfos.reserve(stageDescriptions.size());
  for (const RayStageDescription &description : stageDescriptions) {
    shaders.push_back(std::make_unique<Shader>(ctx, description.entry, description.stage));
    const Shader &stage = *shaders.back();
    for (const std::string &feature : stage.reflection.requiredFeatures)
      if (!ctx.enabledFeatures.contains(feature))
        throw Error("shader " + stage.entry + " requires disabled device feature " + feature);
    for (const std::string &property : stage.reflection.requiredProperties)
      if (!m2v::host::supportsProperty(ctx.physical, property))
        throw Error("shader " + stage.entry + " requires device property " + property);
    reflections.push_back(stage.reflection);
    stageInfos.push_back(stage.stageInfo());
  }

  const std::vector<std::string> roles = m2v::host::checkRayShaderGroups(reflections, groups);
  (void)roles;
  m2v::host::checkRayRecursionDepth(reflections, recursionDepth, ctx.rayPipelineProperties);

  const auto merged = m2v::host::rayDescriptorBindings(reflections);
  std::vector<VkDescriptorSetLayoutBinding> bindings;
  bindings.reserve(merged.size());
  for (const auto &binding : merged)
    bindings.push_back({binding.binding, binding.type, binding.count, binding.stages, nullptr});
  std::size_t immutableCount = 0;
  for (const auto &stage : reflections)
    for (const auto &sampler : stage.samplers)
      if (sampler.constexprSampler) ++immutableCount;
  immutableSamplers.reserve(immutableCount);
  for (const auto &stage : reflections) {
    for (const auto &sampler : stage.samplers) {
      if (!sampler.constexprSampler) continue;
      auto binding = std::find_if(bindings.begin(), bindings.end(), [&](const auto &candidate) {
        return candidate.binding == sampler.binding;
      });
      if (binding == bindings.end()) throw Error("constexpr sampler binding is absent from the merged layout");
      if (binding->pImmutableSamplers) continue;
      VkSampler samplerHandle = VK_NULL_HANDLE;
      const VkSamplerCreateInfo info = m2v::host::samplerCreateInfo(sampler);
      check(vkCreateSampler(ctx.device, &info, nullptr, &samplerHandle),
            "vkCreateSampler (ray pipeline constexpr)");
      immutableSamplers.push_back(samplerHandle);
      binding->pImmutableSamplers = &immutableSamplers.back();
    }
  }

  VkDescriptorSetLayoutCreateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  setInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
  setInfo.pBindings = bindings.data();
  check(vkCreateDescriptorSetLayout(ctx.device, &setInfo, nullptr, &setLayout),
        "vkCreateDescriptorSetLayout (ray pipeline)");

  VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layoutInfo.setLayoutCount = 1;
  layoutInfo.pSetLayouts = &setLayout;
  check(vkCreatePipelineLayout(ctx.device, &layoutInfo, nullptr, &layout),
        "vkCreatePipelineLayout (ray pipeline)");

  std::vector<VkRayTracingShaderGroupCreateInfoKHR> groupInfos;
  groupInfos.reserve(groups.size());
  for (const auto &group : groups) {
    VkRayTracingShaderGroupCreateInfoKHR info{
        VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    info.type = group.type;
    info.generalShader = group.general;
    info.closestHitShader = group.closestHit;
    info.anyHitShader = group.anyHit;
    info.intersectionShader = group.intersection;
    groupInfos.push_back(info);
  }

  VkRayTracingPipelineCreateInfoKHR pipelineInfo{
      VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  pipelineInfo.stageCount = static_cast<std::uint32_t>(stageInfos.size());
  pipelineInfo.pStages = stageInfos.data();
  pipelineInfo.groupCount = static_cast<std::uint32_t>(groupInfos.size());
  pipelineInfo.pGroups = groupInfos.data();
  pipelineInfo.maxPipelineRayRecursionDepth = recursionDepth;
  pipelineInfo.layout = layout;
  check(ctx.rt.createPipelines(ctx.device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineInfo,
                               nullptr, &handle),
        "vkCreateRayTracingPipelinesKHR");
  ctx.nameObject(handle, VK_OBJECT_TYPE_PIPELINE, "ray.pipeline");

  const std::size_t handleBytes = groups.size() * ctx.rayPipelineProperties.shaderGroupHandleSize;
  std::vector<std::byte> handles(handleBytes);
  check(ctx.rt.getShaderGroupHandles(ctx.device, handle, 0,
                                     static_cast<std::uint32_t>(groups.size()),
                                     handles.size(), handles.data()),
        "vkGetRayTracingShaderGroupHandlesKHR");
  m2v::host::ShaderBindingTable packed = m2v::host::buildShaderBindingTable(
      reflections, groups, ctx.rayPipelineProperties, handles, raygen, miss, hit, callable);

  const VkDeviceSize baseAlignment = ctx.rayPipelineProperties.shaderGroupBaseAlignment;
  const VkDeviceSize allocationBytes = packed.bytes.size() + baseAlignment - 1;
  sbt = Buffer(ctx, allocationBytes,
               VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
               VMA_MEMORY_USAGE_AUTO,
               VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                   VMA_ALLOCATION_CREATE_MAPPED_BIT,
               "ray.sbt");
  const VkDeviceAddress raw = sbt.deviceAddress();
  const VkDeviceAddress base = alignUp(raw, baseAlignment);
  const VkDeviceSize offset = base - raw;
  sbt.write(packed.bytes.data(), packed.bytes.size(), static_cast<std::size_t>(offset));
  auto relocate = [base](VkStridedDeviceAddressRegionKHR region) {
    if (region.size) region.deviceAddress += base;
    return region;
  };
  raygenRegion = relocate(packed.raygen);
  missRegion = relocate(packed.miss);
  hitRegion = relocate(packed.hit);
  callableRegion = relocate(packed.callable);
  sbtBytes = packed.bytes.size();
}

RayPipeline::~RayPipeline() {
  // Buffer and modules are destroyed after the pipeline objects by member destruction order.
  if (handle) vkDestroyPipeline(context.device, handle, nullptr);
  if (layout) vkDestroyPipelineLayout(context.device, layout, nullptr);
  if (setLayout) vkDestroyDescriptorSetLayout(context.device, setLayout, nullptr);
  for (VkSampler sampler : immutableSamplers) vkDestroySampler(context.device, sampler, nullptr);
}

void RayPipeline::trace(VkCommandBuffer command, VkDescriptorSet set, std::uint32_t width,
                        std::uint32_t height, std::uint32_t depth) const {
  if (!width || !height || !depth) throw Error("ray trace dimensions must be nonzero");
  if (width > context.rayPipelineProperties.maxRayDispatchInvocationCount / height / depth)
    throw Error("ray trace dimensions exceed maxRayDispatchInvocationCount");
  vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, handle);
  vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, layout, 0, 1,
                          &set, 0, nullptr);
  context.rt.traceRays(command, &raygenRegion, &missRegion, &hitRegion, &callableRegion,
                       width, height, depth);
}

void RayPipeline::traceIndirect(VkCommandBuffer command, VkDescriptorSet set, VkDeviceAddress address) const {
  if (!context.rayPipelineIndirectSupported || !context.rt.traceRaysIndirect)
    throw Error("indirect ray tracing is not enabled on this device");
  vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, handle);
  vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, layout, 0, 1,
                          &set, 0, nullptr);
  context.rt.traceRaysIndirect(command, &raygenRegion, &missRegion, &hitRegion, &callableRegion, address);
}

} // namespace basalt
