// Instance, device, allocator and the state every other file needs.
#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <set>
#include <string>
#include <vector>

namespace basalt {

class Window;

struct DeviceInfo {
  std::string name;
  VkPhysicalDeviceType type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
  std::uint32_t driverVersion = 0;
  std::uint32_t apiVersion = 0;
};

class Context {
public:
  // With validation on, any reported error fails the run.
  Context(const Window &window, bool validation);
  ~Context();
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  VkInstance instance = VK_NULL_HANDLE;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  std::uint32_t queueFamily = 0;
  VmaAllocator allocator = VK_NULL_HANDLE;

  VkPhysicalDeviceProperties properties{};
  VkPhysicalDeviceMemoryProperties memoryProperties{};
  DeviceInfo info;
  bool validationEnabled = false;
  mutable bool sawValidationError = false;

  // The inline ray-query capability.
  bool rayTracingSupported = false;
  bool accelerationStructureSupported = false;
  bool rayQuerySupported = false;
  // VK_KHR_pipeline_executable_properties, enabled only for BASALT_PIPELINE_STATISTICS=1
  // (tools/shader_stats.cpp): the driver's register and spill counts per pipeline.
  bool pipelineStatisticsEnabled = false;
  bool rayPipelineSupported = false;
  // vkCmdTraceRaysIndirectKHR, queried and enabled separately; BASALT_NO_INDIRECT_TRACE forces
  // the direct fallback.
  bool rayPipelineIndirectSupported = false;
  // What the device was created with, checked against shader reflection.
  std::set<std::string> enabledFeatures;
  // The alignment an acceleration structure build's scratch address needs.
  VkDeviceSize scratchAlignment = 256;
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayPipelineProperties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  struct RayTracingFunctions {
    PFN_vkGetAccelerationStructureBuildSizesKHR getBuildSizes = nullptr;
    PFN_vkCreateAccelerationStructureKHR create = nullptr;
    PFN_vkDestroyAccelerationStructureKHR destroy = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR build = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR address = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR createPipelines = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR getShaderGroupHandles = nullptr;
    PFN_vkCmdTraceRaysKHR traceRays = nullptr;
    PFN_vkCmdTraceRaysIndirectKHR traceRaysIndirect = nullptr;
    PFN_vkGetRayTracingShaderGroupStackSizeKHR getShaderGroupStackSize = nullptr;
    PFN_vkCmdSetRayTracingPipelineStackSizeKHR setPipelineStackSize = nullptr;
  } rt;

  // The ray-tracing shader stage, or none when the device was created without the ray
  // pipeline feature: naming the stage in a barrier is invalid there (VUID-07946).
  VkPipelineStageFlags2 rayTracingShaderStage() const {
    return rayPipelineSupported ? VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR : VkPipelineStageFlags2{0};
  }

  void waitIdle() const;

  VkFormat selectFormat(const std::vector<VkFormat> &candidates, VkFormatFeatureFlags features) const;

  void setName(std::uint64_t handle, VkObjectType type, const std::string &name) const;
  template <class T> void nameObject(T handle, VkObjectType type, const std::string &name) const {
    setName(reinterpret_cast<std::uint64_t>(handle), type, name);
  }

private:
  VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
  bool debugUtils = false;
};

} // namespace basalt
