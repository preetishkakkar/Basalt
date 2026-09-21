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

  // Ray queries when the extensions and features exist; the rasterised paths stay either way.
  bool rayTracingSupported = false;
  // What the device was created with, checked against shader reflection.
  std::set<std::string> enabledFeatures;
  // The alignment an acceleration structure build's scratch address needs.
  VkDeviceSize scratchAlignment = 256;
  struct RayTracingFunctions {
    PFN_vkGetAccelerationStructureBuildSizesKHR getBuildSizes = nullptr;
    PFN_vkCreateAccelerationStructureKHR create = nullptr;
    PFN_vkDestroyAccelerationStructureKHR destroy = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR build = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR address = nullptr;
  } rt;

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
