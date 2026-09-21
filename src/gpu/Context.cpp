#include "gpu/Context.h"

#include "core/Log.h"
#include "platform/Window.h"

#include <vulkan/vulkan_win32.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>

namespace basalt {
namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT *data,
                                             void *user) {
  auto *context = static_cast<Context *>(user);
  const std::string text = data && data->pMessage ? data->pMessage : "(no message)";
  if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    if (context) context->sawValidationError = true;
    logError("validation: {}", text);
  } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    logWarning("validation: {}", text);
  }
  return VK_FALSE;
}

bool hasLayer(const char *name) {
  std::uint32_t count = 0;
  vkEnumerateInstanceLayerProperties(&count, nullptr);
  std::vector<VkLayerProperties> layers(count);
  vkEnumerateInstanceLayerProperties(&count, layers.data());
  return std::any_of(layers.begin(), layers.end(),
                     [&](const VkLayerProperties &l) { return std::strcmp(l.layerName, name) == 0; });
}

bool hasInstanceExtension(const char *name) {
  std::uint32_t count = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
  std::vector<VkExtensionProperties> extensions(count);
  vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data());
  return std::any_of(extensions.begin(), extensions.end(), [&](const VkExtensionProperties &e) {
    return std::strcmp(e.extensionName, name) == 0;
  });
}

bool hasDeviceExtension(VkPhysicalDevice physical, const char *name) {
  std::uint32_t count = 0;
  vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
  std::vector<VkExtensionProperties> extensions(count);
  vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, extensions.data());
  return std::any_of(extensions.begin(), extensions.end(), [&](const VkExtensionProperties &e) {
    return std::strcmp(e.extensionName, name) == 0;
  });
}

} // namespace

Context::Context(const Window &window, bool validation) {
  validationEnabled = validation && hasLayer("VK_LAYER_KHRONOS_validation");
  if (validation && !validationEnabled)
    logWarning("the Khronos validation layer is not installed; running without it");

  VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  application.pApplicationName = "Basalt";
  application.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
  application.pEngineName = "Basalt";
  application.apiVersion = VK_API_VERSION_1_3;

  std::vector<const char *> layers;
  std::vector<const char *> extensions{VK_KHR_SURFACE_EXTENSION_NAME,
                                       VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
  debugUtils = hasInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  if (debugUtils) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  if (validationEnabled) layers.push_back("VK_LAYER_KHRONOS_validation");

  VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instanceInfo.pApplicationInfo = &application;
  instanceInfo.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
  instanceInfo.ppEnabledLayerNames = layers.data();
  instanceInfo.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
  instanceInfo.ppEnabledExtensionNames = extensions.data();
  check(vkCreateInstance(&instanceInfo, nullptr, &instance), "vkCreateInstance");

  if (debugUtils) {
    auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (create) {
      VkDebugUtilsMessengerCreateInfoEXT messengerInfo{
          VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
      messengerInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
      messengerInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
      messengerInfo.pfnUserCallback = debugCallback;
      messengerInfo.pUserData = this;
      create(instance, &messengerInfo, nullptr, &messenger);
    }
  }

  VkWin32SurfaceCreateInfoKHR surfaceInfo{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
  surfaceInfo.hinstance = window.moduleHandle();
  surfaceInfo.hwnd = window.handle();
  check(vkCreateWin32SurfaceKHR(instance, &surfaceInfo, nullptr, &surface), "vkCreateWin32SurfaceKHR");

  std::uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
  if (deviceCount == 0) throw Error("no Vulkan device is installed");
  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

  // First discrete device with graphics and present on one family; one queue draws, presents and uploads.
  auto suitable = [&](VkPhysicalDevice candidate, std::uint32_t &family) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(candidate, &props);
    if (props.apiVersion < VK_API_VERSION_1_3) return false;
    if (!hasDeviceExtension(candidate, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) return false;
    std::uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, families.data());
    for (std::uint32_t i = 0; i < familyCount; ++i) {
      VkBool32 present = VK_FALSE;
      vkGetPhysicalDeviceSurfaceSupportKHR(candidate, i, surface, &present);
      if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && present) {
        family = i;
        return true;
      }
    }
    return false;
  };

  for (int pass = 0; pass < 2 && physical == VK_NULL_HANDLE; ++pass) {
    for (VkPhysicalDevice candidate : devices) {
      VkPhysicalDeviceProperties props{};
      vkGetPhysicalDeviceProperties(candidate, &props);
      const bool discrete = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
      if (pass == 0 && !discrete) continue;
      std::uint32_t family = 0;
      if (suitable(candidate, family)) {
        physical = candidate;
        queueFamily = family;
        break;
      }
    }
  }
  if (physical == VK_NULL_HANDLE)
    throw Error("no Vulkan 1.3 device offers graphics, compute and presentation on this window");

  vkGetPhysicalDeviceProperties(physical, &properties);
  vkGetPhysicalDeviceMemoryProperties(physical, &memoryProperties);
  {
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationProperties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties2.pNext = &accelerationProperties;
    vkGetPhysicalDeviceProperties2(physical, &properties2);
    if (accelerationProperties.minAccelerationStructureScratchOffsetAlignment > 0)
      scratchAlignment = accelerationProperties.minAccelerationStructureScratchOffsetAlignment;
  }
  info.name = properties.deviceName;
  info.type = properties.deviceType;
  info.driverVersion = properties.driverVersion;
  info.apiVersion = properties.apiVersion;

  // shaderDrawParameters carries [[instance_id]], which is how a draw finds its record.
  VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  VkPhysicalDeviceVulkan11Features features11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  features.pNext = &features11;
  features11.pNext = &features12;
  features12.pNext = &features13;
  // Queried only where the extensions exist; asking otherwise is invalid.
  const bool rayExtensions = hasDeviceExtension(physical, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
                             hasDeviceExtension(physical, VK_KHR_RAY_QUERY_EXTENSION_NAME) &&
                             hasDeviceExtension(physical, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
  if (rayExtensions) {
    features13.pNext = &accelerationFeatures;
    accelerationFeatures.pNext = &rayQueryFeatures;
  }
  vkGetPhysicalDeviceFeatures2(physical, &features);

  VkPhysicalDeviceFeatures enabled{};
  enabled.samplerAnisotropy = features.features.samplerAnisotropy;
  enabled.fillModeNonSolid = features.features.fillModeNonSolid;
  enabled.depthClamp = features.features.depthClamp;
  enabled.independentBlend = features.features.independentBlend;
  enabled.shaderStorageImageWriteWithoutFormat = features.features.shaderStorageImageWriteWithoutFormat;
  enabled.imageCubeArray = features.features.imageCubeArray;
  enabled.textureCompressionBC = features.features.textureCompressionBC;
  enabled.shaderSampledImageArrayDynamicIndexing = features.features.shaderSampledImageArrayDynamicIndexing;

  VkPhysicalDeviceVulkan11Features enable11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  enable11.shaderDrawParameters = features11.shaderDrawParameters;
  VkPhysicalDeviceVulkan12Features enable12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  enable12.pNext = &enable11;
  enable12.scalarBlockLayout = features12.scalarBlockLayout;
  enable12.hostQueryReset = features12.hostQueryReset;
  enable12.shaderSampledImageArrayNonUniformIndexing = features12.shaderSampledImageArrayNonUniformIndexing;
  enable12.descriptorIndexing = features12.descriptorIndexing;
  enable12.runtimeDescriptorArray = features12.runtimeDescriptorArray;
  rayTracingSupported = rayExtensions && features12.bufferDeviceAddress &&
                        accelerationFeatures.accelerationStructure && rayQueryFeatures.rayQuery;
  // The hit texture table needs 128 sampled images per stage; without them, the rasterised pair.
  if (rayTracingSupported && (properties.limits.maxPerStageDescriptorSampledImages < 128 ||
                              properties.limits.maxDescriptorSetSampledImages < 128)) {
    logWarning("the device allows only {} sampled images per stage; ray tracing is off",
               properties.limits.maxPerStageDescriptorSampledImages);
    rayTracingSupported = false;
  }
  // BASALT_NO_RAY_TRACING forces the rasterised pair, for testing.
  if (rayTracingSupported && std::getenv("BASALT_NO_RAY_TRACING")) {
    logInfo("ray tracing is available but disabled by BASALT_NO_RAY_TRACING");
    rayTracingSupported = false;
  }
  enable12.bufferDeviceAddress = rayTracingSupported ? VK_TRUE : VK_FALSE;
  VkPhysicalDeviceAccelerationStructureFeaturesKHR enableAcceleration{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  enableAcceleration.accelerationStructure = VK_TRUE;
  VkPhysicalDeviceRayQueryFeaturesKHR enableRayQuery{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  enableRayQuery.rayQuery = VK_TRUE;
  enableAcceleration.pNext = &enableRayQuery;
  VkPhysicalDeviceVulkan13Features enable13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  enable13.pNext = &enable12;
  enable13.dynamicRendering = VK_TRUE;
  enable13.synchronization2 = VK_TRUE;
  // discard_fragment() lowers to OpDemoteToHelperInvocation.
  enable13.shaderDemoteToHelperInvocation = features13.shaderDemoteToHelperInvocation;
  if (!features13.dynamicRendering || !features13.synchronization2)
    throw Error("the device does not offer dynamic rendering and synchronization2");
  if (!features13.shaderDemoteToHelperInvocation)
    throw Error("the device does not offer shaderDemoteToHelperInvocation, which alpha-masked "
                "materials need for discard_fragment()");

  // What was actually asked for, by the names the reflection uses.
  auto record = [&](const char *name, VkBool32 on) {
    if (on) enabledFeatures.insert(name);
  };
  record("samplerAnisotropy", enabled.samplerAnisotropy);
  record("fillModeNonSolid", enabled.fillModeNonSolid);
  record("depthClamp", enabled.depthClamp);
  record("independentBlend", enabled.independentBlend);
  record("shaderStorageImageWriteWithoutFormat", enabled.shaderStorageImageWriteWithoutFormat);
  record("imageCubeArray", enabled.imageCubeArray);
  record("textureCompressionBC", enabled.textureCompressionBC);
  record("shaderSampledImageArrayDynamicIndexing", enabled.shaderSampledImageArrayDynamicIndexing);
  record("shaderDrawParameters", enable11.shaderDrawParameters);
  record("scalarBlockLayout", enable12.scalarBlockLayout);
  record("hostQueryReset", enable12.hostQueryReset);
  record("shaderSampledImageArrayNonUniformIndexing", enable12.shaderSampledImageArrayNonUniformIndexing);
  record("descriptorIndexing", enable12.descriptorIndexing);
  record("runtimeDescriptorArray", enable12.runtimeDescriptorArray);
  record("bufferDeviceAddress", enable12.bufferDeviceAddress);
  record("dynamicRendering", enable13.dynamicRendering);
  record("synchronization2", enable13.synchronization2);
  record("shaderDemoteToHelperInvocation", enable13.shaderDemoteToHelperInvocation);
  record("accelerationStructure", rayTracingSupported ? VK_TRUE : VK_FALSE);
  record("rayQuery", rayTracingSupported ? VK_TRUE : VK_FALSE);

  const float priority = 1.0f;
  VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queueInfo.queueFamilyIndex = queueFamily;
  queueInfo.queueCount = 1;
  queueInfo.pQueuePriorities = &priority;

  std::vector<const char *> deviceExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  if (rayTracingSupported) {
    deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    enable11.pNext = &enableAcceleration;
  }
  VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  deviceInfo.pNext = &enable13;
  deviceInfo.queueCreateInfoCount = 1;
  deviceInfo.pQueueCreateInfos = &queueInfo;
  deviceInfo.pEnabledFeatures = &enabled;
  deviceInfo.enabledExtensionCount = static_cast<std::uint32_t>(deviceExtensions.size());
  deviceInfo.ppEnabledExtensionNames = deviceExtensions.data();
  check(vkCreateDevice(physical, &deviceInfo, nullptr, &device), "vkCreateDevice");
  vkGetDeviceQueue(device, queueFamily, 0, &queue);

  VmaAllocatorCreateInfo allocatorInfo{};
  allocatorInfo.physicalDevice = physical;
  allocatorInfo.device = device;
  allocatorInfo.instance = instance;
  allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
  if (rayTracingSupported) allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  check(vmaCreateAllocator(&allocatorInfo, &allocator), "vmaCreateAllocator");

  if (rayTracingSupported) {
    // The extension entry points come through the device, not the loader.
    auto load = [&](const char *name) { return vkGetDeviceProcAddr(device, name); };
    rt.getBuildSizes = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
        load("vkGetAccelerationStructureBuildSizesKHR"));
    rt.create = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(load("vkCreateAccelerationStructureKHR"));
    rt.destroy = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(load("vkDestroyAccelerationStructureKHR"));
    rt.build = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(load("vkCmdBuildAccelerationStructuresKHR"));
    rt.address = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
        load("vkGetAccelerationStructureDeviceAddressKHR"));
    if (!rt.getBuildSizes || !rt.create || !rt.destroy || !rt.build || !rt.address) {
      logWarning("the acceleration structure entry points could not be loaded; ray tracing is off");
      rayTracingSupported = false;
    }
  }

  logInfo("device: {} (Vulkan {}.{}.{}, driver {}), ray tracing {}", info.name,
          VK_API_VERSION_MAJOR(info.apiVersion), VK_API_VERSION_MINOR(info.apiVersion),
          VK_API_VERSION_PATCH(info.apiVersion), info.driverVersion,
          rayTracingSupported ? "available" : "not available");
}

Context::~Context() {
  if (allocator) vmaDestroyAllocator(allocator);
  if (device) vkDestroyDevice(device, nullptr);
  if (messenger) {
    auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (destroy) destroy(instance, messenger, nullptr);
  }
  if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
  if (instance) vkDestroyInstance(instance, nullptr);
}

void Context::waitIdle() const {
  if (device) vkDeviceWaitIdle(device);
}

VkFormat Context::selectFormat(const std::vector<VkFormat> &candidates,
                               VkFormatFeatureFlags features) const {
  for (VkFormat format : candidates) {
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(physical, format, &props);
    if ((props.optimalTilingFeatures & features) == features) return format;
  }
  throw Error("no candidate format supports the requested features on this device");
}

void Context::setName(std::uint64_t handle, VkObjectType type, const std::string &name) const {
  if (!debugUtils || !device) return;
  static auto setNameEXT = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
      vkGetInstanceProcAddr(instance, "vkSetDebugUtilsObjectNameEXT"));
  if (!setNameEXT) return;
  VkDebugUtilsObjectNameInfoEXT nameInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
  nameInfo.objectType = type;
  nameInfo.objectHandle = handle;
  nameInfo.pObjectName = name.c_str();
  setNameEXT(device, &nameInfo);
}

} // namespace basalt
