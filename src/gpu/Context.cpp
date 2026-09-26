#include "gpu/Context.h"

#include "core/Log.h"
#include "platform/Window.h"

#include <vulkan/vulkan_win32.h>
#include <algorithm>
#include <array>
#include <cctype>
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

  const char *selectionEnvironment = std::getenv("BASALT_VULKAN_DEVICE");
  const std::string selection = selectionEnvironment ? selectionEnvironment : "";
  if (!selection.empty()) {
    auto lower = [](std::string value) {
      std::transform(value.begin(), value.end(), value.begin(),
                     [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      return value;
    };
    char *end = nullptr;
    const unsigned long requestedIndex = std::strtoul(selection.c_str(), &end, 10);
    const bool byIndex = end != selection.c_str() && *end == '\0';
    const std::string requestedName = lower(selection);
    for (std::size_t index = 0; index < devices.size(); ++index) {
      VkPhysicalDeviceProperties props{};
      vkGetPhysicalDeviceProperties(devices[index], &props);
      const bool match = byIndex ? index == requestedIndex
                                 : lower(props.deviceName).find(requestedName) != std::string::npos;
      std::uint32_t family = 0;
      if (match && suitable(devices[index], family)) {
        physical = devices[index];
        queueFamily = family;
        break;
      }
    }
    if (physical == VK_NULL_HANDLE)
      throw Error("requested Vulkan device '" + selection +
                  "' was not found or lacks Vulkan 1.3 graphics, compute and presentation");
    logInfo("selected Vulkan device using BASALT_VULKAN_DEVICE={}", selection);
  } else {
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
  }
  if (physical == VK_NULL_HANDLE)
    throw Error("no Vulkan 1.3 device offers graphics, compute and presentation on this window");

  vkGetPhysicalDeviceProperties(physical, &properties);
  vkGetPhysicalDeviceMemoryProperties(physical, &memoryProperties);
  const bool accelerationExtensions =
      hasDeviceExtension(physical, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
      hasDeviceExtension(physical, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
  const bool rayQueryExtension =
      accelerationExtensions && hasDeviceExtension(physical, VK_KHR_RAY_QUERY_EXTENSION_NAME);
  const bool rayPipelineExtension =
      accelerationExtensions && hasDeviceExtension(physical, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
  {
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationProperties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    if (accelerationExtensions) {
      properties2.pNext = &accelerationProperties;
      if (rayPipelineExtension) accelerationProperties.pNext = &rayPipelineProperties;
    }
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
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayPipelineFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  features.pNext = &features11;
  features11.pNext = &features12;
  features12.pNext = &features13;
  // Queried only where the extensions exist; asking otherwise is invalid.
  if (accelerationExtensions) {
    features13.pNext = &accelerationFeatures;
    if (rayQueryExtension) accelerationFeatures.pNext = &rayQueryFeatures;
    if (rayPipelineExtension) {
      if (rayQueryExtension) rayQueryFeatures.pNext = &rayPipelineFeatures;
      else accelerationFeatures.pNext = &rayPipelineFeatures;
    }
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
  enabled.geometryShader = features.features.geometryShader;
  // Optional: double precision and 64-bit integers in shaders (the GPU split clipping, exact as
  // the host's).
  enabled.shaderFloat64 = features.features.shaderFloat64;
  enabled.shaderInt64 = features.features.shaderInt64;
  if (!enabled.geometryShader)
    throw Error("the device does not offer geometryShader, required for fragment primitive identity");

  VkPhysicalDeviceVulkan11Features enable11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  enable11.shaderDrawParameters = features11.shaderDrawParameters;
  VkPhysicalDeviceVulkan12Features enable12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  enable12.pNext = &enable11;
  enable12.scalarBlockLayout = features12.scalarBlockLayout;
  enable12.hostQueryReset = features12.hostQueryReset;
  enable12.shaderSampledImageArrayNonUniformIndexing = features12.shaderSampledImageArrayNonUniformIndexing;
  enable12.descriptorIndexing = features12.descriptorIndexing;
  enable12.runtimeDescriptorArray = features12.runtimeDescriptorArray;
  accelerationStructureSupported = accelerationExtensions && features12.bufferDeviceAddress &&
                                   accelerationFeatures.accelerationStructure;
  rayQuerySupported = accelerationStructureSupported && rayQueryExtension && rayQueryFeatures.rayQuery;
  rayPipelineSupported = accelerationStructureSupported && rayPipelineExtension &&
                         rayPipelineFeatures.rayTracingPipeline;
  // The hit texture table needs 128 sampled images per stage; without them, the rasterised pair.
  if ((rayQuerySupported || rayPipelineSupported) &&
      (properties.limits.maxPerStageDescriptorSampledImages < 128 ||
       properties.limits.maxDescriptorSetSampledImages < 128)) {
    logWarning("the device allows only {} sampled images per stage; path tracing is off",
               properties.limits.maxPerStageDescriptorSampledImages);
    rayQuerySupported = false;
    rayPipelineSupported = false;
  }
  // BASALT_NO_RAY_TRACING forces the rasterised pair, for testing.
  if ((rayQuerySupported || rayPipelineSupported) && std::getenv("BASALT_NO_RAY_TRACING")) {
    logInfo("ray tracing is available but disabled by BASALT_NO_RAY_TRACING");
    rayQuerySupported = false;
    rayPipelineSupported = false;
    accelerationStructureSupported = false;
  }
  if (rayPipelineSupported && std::getenv("BASALT_NO_RAY_PIPELINE")) {
    logInfo("ray pipelines are available but disabled by BASALT_NO_RAY_PIPELINE");
    rayPipelineSupported = false;
  }
  rayTracingSupported = rayQuerySupported;
  enable12.bufferDeviceAddress = accelerationStructureSupported ? VK_TRUE : VK_FALSE;
  VkPhysicalDeviceAccelerationStructureFeaturesKHR enableAcceleration{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  enableAcceleration.accelerationStructure = VK_TRUE;
  VkPhysicalDeviceRayQueryFeaturesKHR enableRayQuery{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  enableRayQuery.rayQuery = rayQuerySupported ? VK_TRUE : VK_FALSE;
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR enableRayPipeline{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  enableRayPipeline.rayTracingPipeline = rayPipelineSupported ? VK_TRUE : VK_FALSE;
  enableRayPipeline.rayTracingPipelineTraceRaysIndirect =
      rayPipelineSupported && rayPipelineFeatures.rayTracingPipelineTraceRaysIndirect ? VK_TRUE : VK_FALSE;
  rayPipelineIndirectSupported = enableRayPipeline.rayTracingPipelineTraceRaysIndirect == VK_TRUE &&
                                 !std::getenv("BASALT_NO_INDIRECT_TRACE");
  if (rayQuerySupported) enableAcceleration.pNext = &enableRayQuery;
  if (rayPipelineSupported) {
    if (rayQuerySupported) enableRayQuery.pNext = &enableRayPipeline;
    else enableAcceleration.pNext = &enableRayPipeline;
  }
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
  record("geometryShader", enabled.geometryShader);
  record("shaderFloat64", enabled.shaderFloat64);
  record("shaderInt64", enabled.shaderInt64);
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
  record("accelerationStructure", accelerationStructureSupported ? VK_TRUE : VK_FALSE);
  record("rayQuery", rayQuerySupported ? VK_TRUE : VK_FALSE);
  record("rayTracingPipeline", rayPipelineSupported ? VK_TRUE : VK_FALSE);
  record("rayTracingPipelineTraceRaysIndirect", enableRayPipeline.rayTracingPipelineTraceRaysIndirect);

  const float priority = 1.0f;
  VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queueInfo.queueFamilyIndex = queueFamily;
  queueInfo.queueCount = 1;
  queueInfo.pQueuePriorities = &priority;

  std::vector<const char *> deviceExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  if (accelerationStructureSupported) {
    deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    if (rayQuerySupported) deviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    if (rayPipelineSupported) deviceExtensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    enable11.pNext = &enableAcceleration;
  }
  VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR enableExecutables{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
  VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  deviceInfo.pNext = &enable13;
  if (std::getenv("BASALT_PIPELINE_STATISTICS") &&
      hasDeviceExtension(physical, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)) {
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR available{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    query.pNext = &available;
    vkGetPhysicalDeviceFeatures2(physical, &query);
    if (available.pipelineExecutableInfo) {
      deviceExtensions.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
      enableExecutables.pipelineExecutableInfo = VK_TRUE;
      enableExecutables.pNext = const_cast<void *>(deviceInfo.pNext);
      deviceInfo.pNext = &enableExecutables;
      pipelineStatisticsEnabled = true;
    }
  }
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
  if (accelerationStructureSupported) allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  check(vmaCreateAllocator(&allocatorInfo, &allocator), "vmaCreateAllocator");

  if (accelerationStructureSupported) {
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
      accelerationStructureSupported = false;
      rayQuerySupported = false;
      rayPipelineSupported = false;
      rayTracingSupported = false;
    }
  }

  if (rayPipelineSupported) {
    auto load = [&](const char *name) { return vkGetDeviceProcAddr(device, name); };
    rt.createPipelines = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(
        load("vkCreateRayTracingPipelinesKHR"));
    rt.getShaderGroupHandles = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(
        load("vkGetRayTracingShaderGroupHandlesKHR"));
    rt.traceRays = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(load("vkCmdTraceRaysKHR"));
    rt.traceRaysIndirect = reinterpret_cast<PFN_vkCmdTraceRaysIndirectKHR>(
        load("vkCmdTraceRaysIndirectKHR"));
    rt.getShaderGroupStackSize = reinterpret_cast<PFN_vkGetRayTracingShaderGroupStackSizeKHR>(
        load("vkGetRayTracingShaderGroupStackSizeKHR"));
    rt.setPipelineStackSize = reinterpret_cast<PFN_vkCmdSetRayTracingPipelineStackSizeKHR>(
        load("vkCmdSetRayTracingPipelineStackSizeKHR"));
    if (!rt.createPipelines || !rt.getShaderGroupHandles || !rt.traceRays) {
      logWarning("the ray-pipeline entry points could not be loaded; ray pipelines are off");
      rayPipelineSupported = false;
    }
  }
  rayTracingSupported = rayQuerySupported;

  logInfo("device: {} (Vulkan {}.{}.{}, driver {}), acceleration structures {}, ray queries {}, ray pipelines {}", info.name,
          VK_API_VERSION_MAJOR(info.apiVersion), VK_API_VERSION_MINOR(info.apiVersion),
          VK_API_VERSION_PATCH(info.apiVersion), info.driverVersion,
          accelerationStructureSupported ? "available" : "not available",
          rayQuerySupported ? "available" : "not available",
          rayPipelineSupported ? "available" : "not available");
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

void Context::beginLabel(VkCommandBuffer command, const char *name) const {
  if (!debugUtils) return;
  static auto begin = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
      vkGetInstanceProcAddr(instance, "vkCmdBeginDebugUtilsLabelEXT"));
  if (!begin) return;
  VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
  label.pLabelName = name;
  begin(command, &label);
}

void Context::endLabel(VkCommandBuffer command) const {
  if (!debugUtils) return;
  static auto end = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
      vkGetInstanceProcAddr(instance, "vkCmdEndDebugUtilsLabelEXT"));
  if (end) end(command);
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
