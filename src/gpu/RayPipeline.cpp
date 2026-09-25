#include "gpu/RayPipeline.h"

#include "core/Log.h"

#include <algorithm>
#include <cstring>

namespace basalt {
namespace {

VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) {
  if (!alignment || (alignment & (alignment - 1)) != 0) throw Error("ray pipeline reported an invalid alignment");
  return (value + alignment - 1) & ~(alignment - 1);
}

bool isGeneral(VkShaderStageFlagBits stage) {
  return stage == VK_SHADER_STAGE_RAYGEN_BIT_KHR || stage == VK_SHADER_STAGE_MISS_BIT_KHR ||
         stage == VK_SHADER_STAGE_CALLABLE_BIT_KHR;
}

// Each group's role ("raygen", "miss", "callable" or "hit"), after checking its stages fill the
// right slots.
std::vector<std::string> groupRoles(const std::vector<std::unique_ptr<Shader>> &shaders,
                                    const std::vector<RayShaderGroup> &groups) {
  auto stageAt = [&](std::uint32_t index, const char *slot) -> const Shader * {
    if (index == VK_SHADER_UNUSED_KHR) return nullptr;
    if (index >= shaders.size())
      throw Error(std::string("a ray shader group's ") + slot + " names stage " + std::to_string(index) + " of " +
                  std::to_string(shaders.size()));
    return shaders[index].get();
  };
  std::vector<std::string> roles;
  bool raygen = false;
  for (const RayShaderGroup &group : groups) {
    const Shader *general = stageAt(group.general, "general stage");
    const Shader *closest = stageAt(group.closestHit, "closest-hit stage");
    const Shader *any = stageAt(group.anyHit, "any-hit stage");
    const Shader *intersection = stageAt(group.intersection, "intersection stage");
    if (group.type == VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR) {
      if (!general || !isGeneral(general->stageFlag) || closest || any || intersection)
        throw Error("a general ray shader group needs one raygen, miss or callable stage and nothing else");
      const VkShaderStageFlagBits stage = general->stageFlag;
      roles.push_back(stage == VK_SHADER_STAGE_RAYGEN_BIT_KHR ? "raygen" : stage == VK_SHADER_STAGE_MISS_BIT_KHR ? "miss" : "callable");
      raygen = raygen || stage == VK_SHADER_STAGE_RAYGEN_BIT_KHR;
      continue;
    }
    const bool procedural = group.type == VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
    if (general || (closest && closest->stageFlag != VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR) ||
        (any && any->stageFlag != VK_SHADER_STAGE_ANY_HIT_BIT_KHR) || (procedural != (intersection != nullptr)) ||
        (intersection && intersection->stageFlag != VK_SHADER_STAGE_INTERSECTION_BIT_KHR))
      throw Error("a hit group holds closest-hit and any-hit stages, and an intersection stage exactly when procedural");
    roles.push_back("hit");
  }
  if (!raygen) throw Error("a ray pipeline needs a ray-generation group");
  return roles;
}

} // namespace

RayPipeline::RayPipeline(const Context &ctx, const std::vector<RayStageDescription> &stageDescriptions,
                         const std::vector<RayShaderGroup> &groups, const ShaderBindingRecords &records,
                         std::uint32_t recursionDepth)
    : context(ctx) {
  if (!ctx.rayPipelineSupported) throw Error("the selected device does not support Vulkan ray pipelines");
  if (stageDescriptions.empty()) throw Error("a ray pipeline needs shader stages");

  std::vector<VkPipelineShaderStageCreateInfo> stageInfos;
  std::vector<const Shader *> stages;
  shaders.reserve(stageDescriptions.size());
  for (const RayStageDescription &description : stageDescriptions) {
    shaders.push_back(std::make_unique<Shader>(ctx, description.entry, description.stage));
    const Shader &stage = *shaders.back();
    requireDeviceSupport(ctx, stage);
    stages.push_back(&stage);
    stageInfos.push_back(stage.stageInfo());
  }
  const std::vector<std::string> roles = groupRoles(shaders, groups);

  // A trace from a closest-hit or miss stage is a second level of recursion.
  std::uint32_t minimumDepth = 1;
  for (const Shader *stage : stages)
    if (stage->tracesRays &&
        (stage->stageFlag == VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR || stage->stageFlag == VK_SHADER_STAGE_MISS_BIT_KHR))
      minimumDepth = 2;
  if (recursionDepth < minimumDepth || recursionDepth > ctx.rayPipelineProperties.maxRayRecursionDepth)
    throw Error("maxPipelineRayRecursionDepth " + std::to_string(recursionDepth) + " is outside " +
                std::to_string(minimumDepth) + ".." + std::to_string(ctx.rayPipelineProperties.maxRayRecursionDepth));

  sets = std::make_unique<ShaderLayout>(ctx, stages, "ray pipeline " + shaders.front()->entry,
                                        VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR);
  layout = sets->layout;

  std::vector<VkRayTracingShaderGroupCreateInfoKHR> groupInfos;
  groupInfos.reserve(groups.size());
  for (const RayShaderGroup &group : groups) {
    VkRayTracingShaderGroupCreateInfoKHR info{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    info.type = group.type;
    info.generalShader = group.general;
    info.closestHitShader = group.closestHit;
    info.anyHitShader = group.anyHit;
    info.intersectionShader = group.intersection;
    groupInfos.push_back(info);
  }

  VkRayTracingPipelineCreateInfoKHR pipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  pipelineInfo.stageCount = static_cast<std::uint32_t>(stageInfos.size());
  pipelineInfo.pStages = stageInfos.data();
  pipelineInfo.groupCount = static_cast<std::uint32_t>(groupInfos.size());
  pipelineInfo.pGroups = groupInfos.data();
  pipelineInfo.maxPipelineRayRecursionDepth = recursionDepth;
  pipelineInfo.layout = layout;
  if (ctx.pipelineStatisticsEnabled) pipelineInfo.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
  check(ctx.rt.createPipelines(ctx.device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &handle),
        "vkCreateRayTracingPipelinesKHR");
  ctx.nameObject(handle, VK_OBJECT_TYPE_PIPELINE, "ray.pipeline");

  // The shader binding table: each region's records are the groups' handles at the handle
  // alignment; regions start base-aligned, and the raygen region is one record whose size is its
  // stride.
  const VkPhysicalDeviceRayTracingPipelinePropertiesKHR &properties = ctx.rayPipelineProperties;
  const VkDeviceSize handleSize = properties.shaderGroupHandleSize;
  const VkDeviceSize baseAlignment = properties.shaderGroupBaseAlignment;
  std::vector<std::byte> handles(groups.size() * handleSize);
  check(ctx.rt.getShaderGroupHandles(ctx.device, handle, 0, static_cast<std::uint32_t>(groups.size()), handles.size(),
                                     handles.data()),
        "vkGetRayTracingShaderGroupHandlesKHR");
  const VkDeviceSize stride = alignUp(handleSize, properties.shaderGroupHandleAlignment);
  if (stride > properties.maxShaderGroupStride) throw Error("the device's shader group handles exceed its maximum stride");
  VkDeviceSize offset = 0;
  auto region = [&](const std::vector<std::uint32_t> &list, const char *role, VkDeviceSize regionStride) {
    VkStridedDeviceAddressRegionKHR result{};
    for (const std::uint32_t group : list) {
      if (group >= groups.size() || roles[group] != role)
        throw Error(std::string("a ") + role + " record names group " + std::to_string(group) + ", which is not a " + role +
                    " group");
    }
    if (list.empty()) return result;
    offset = alignUp(offset, baseAlignment);
    result = {offset, regionStride, regionStride * list.size()};
    offset += result.size;
    return result;
  };
  raygenRegion = region({records.raygen}, "raygen", alignUp(stride, baseAlignment));
  missRegion = region(records.miss, "miss", stride);
  hitRegion = region(records.hit, "hit", stride);
  callableRegion = region(records.callable, "callable", stride);
  std::vector<std::byte> table(static_cast<std::size_t>(offset), std::byte{0});
  auto fill = [&](const VkStridedDeviceAddressRegionKHR &target, const std::vector<std::uint32_t> &list) {
    for (std::size_t i = 0; i < list.size(); ++i)
      std::memcpy(table.data() + target.deviceAddress + target.stride * i, handles.data() + list[i] * handleSize,
                  static_cast<std::size_t>(handleSize));
  };
  fill(raygenRegion, {records.raygen});
  fill(missRegion, records.miss);
  fill(hitRegion, records.hit);
  fill(callableRegion, records.callable);

  // A buffer's address need not meet shaderGroupBaseAlignment: allocate the slack, write the table
  // at the first aligned address and rebase the regions, whose deviceAddress held offsets until now.
  sbt = Buffer(ctx, table.size() + baseAlignment - 1,
               VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_AUTO,
               VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, "ray.sbt");
  const VkDeviceAddress raw = sbt.deviceAddress();
  const VkDeviceAddress base = alignUp(raw, baseAlignment);
  sbt.write(table.data(), table.size(), static_cast<std::size_t>(base - raw));
  for (VkStridedDeviceAddressRegionKHR *target : {&raygenRegion, &missRegion, &hitRegion, &callableRegion})
    if (target->size) target->deviceAddress += base;
  sbtBytes = table.size();
}

RayPipeline::~RayPipeline() {
  // Buffer and modules are destroyed after the pipeline objects by member destruction order.
  if (handle) vkDestroyPipeline(context.device, handle, nullptr);
}

void RayPipeline::checkDimensions(std::uint32_t width, std::uint32_t height, std::uint32_t depth) const {
  if (!width || !height || !depth) throw Error("ray trace dimensions must be nonzero");
  if (width > context.rayPipelineProperties.maxRayDispatchInvocationCount / height / depth)
    throw Error("ray trace dimensions exceed maxRayDispatchInvocationCount");
}

void RayPipeline::bind(VkCommandBuffer command) const {
  vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, handle);
}

void RayPipeline::traceRays(VkCommandBuffer command, std::uint32_t width, std::uint32_t height, std::uint32_t depth) const {
  checkDimensions(width, height, depth);
  context.rt.traceRays(command, &raygenRegion, &missRegion, &hitRegion, &callableRegion, width, height, depth);
}

void RayPipeline::traceRaysIndirect(VkCommandBuffer command, VkDeviceAddress address) const {
  if (!context.rayPipelineIndirectSupported || !context.rt.traceRaysIndirect)
    throw Error("indirect ray tracing is not enabled on this device");
  context.rt.traceRaysIndirect(command, &raygenRegion, &missRegion, &hitRegion, &callableRegion, address);
}

} // namespace basalt
