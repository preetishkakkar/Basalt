#include "gpu/Swapchain.h"

#include "core/Log.h"

#include <algorithm>

namespace basalt {

Swapchain::Swapchain(const Context &ctx, std::uint32_t width, std::uint32_t height, bool useVsync)
    : context(ctx), vsync(useVsync) {
  VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = ctx.queueFamily;
  check(vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &commandPool), "vkCreateCommandPool");

  for (FrameData &data : frames) {
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = commandPool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(ctx.device, &allocate, &data.command), "vkAllocateCommandBuffers");

    VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    check(vkCreateSemaphore(ctx.device, &semaphoreInfo, nullptr, &data.acquired), "vkCreateSemaphore");

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    check(vkCreateFence(ctx.device, &fenceInfo, nullptr, &data.inFlight), "vkCreateFence");
  }

  build(width, height, VK_NULL_HANDLE);
}

Swapchain::~Swapchain() {
  vkDeviceWaitIdle(context.device);
  destroyViews();
  for (VkSemaphore semaphore : rendered) vkDestroySemaphore(context.device, semaphore, nullptr);
  if (swapchain) vkDestroySwapchainKHR(context.device, swapchain, nullptr);
  for (FrameData &data : frames) {
    if (data.acquired) vkDestroySemaphore(context.device, data.acquired, nullptr);
    if (data.inFlight) vkDestroyFence(context.device, data.inFlight, nullptr);
  }
  if (commandPool) vkDestroyCommandPool(context.device, commandPool, nullptr);
}

void Swapchain::destroyViews() {
  for (VkImageView view : views) vkDestroyImageView(context.device, view, nullptr);
  views.clear();
  images.clear();
}

void Swapchain::build(std::uint32_t width, std::uint32_t height, VkSwapchainKHR old) {
  VkSurfaceCapabilitiesKHR capabilities{};
  check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(context.physical, context.surface, &capabilities),
        "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

  std::uint32_t formatCount = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(context.physical, context.surface, &formatCount, nullptr);
  std::vector<VkSurfaceFormatKHR> formats(formatCount);
  vkGetPhysicalDeviceSurfaceFormatsKHR(context.physical, context.surface, &formatCount, formats.data());
  if (formats.empty()) throw Error("the surface offers no format");

  // sRGB surface: the presentation engine encodes the linear output.
  surfaceFormat = formats[0];
  for (const VkSurfaceFormatKHR &candidate : formats) {
    if ((candidate.format == VK_FORMAT_B8G8R8A8_SRGB || candidate.format == VK_FORMAT_R8G8B8A8_SRGB) &&
        candidate.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      surfaceFormat = candidate;
      break;
    }
  }

  std::uint32_t modeCount = 0;
  vkGetPhysicalDeviceSurfacePresentModesKHR(context.physical, context.surface, &modeCount, nullptr);
  std::vector<VkPresentModeKHR> modes(modeCount);
  vkGetPhysicalDeviceSurfacePresentModesKHR(context.physical, context.surface, &modeCount, modes.data());
  presentMode = VK_PRESENT_MODE_FIFO_KHR;
  if (!vsync) {
    for (VkPresentModeKHR mode : modes)
      if (mode == VK_PRESENT_MODE_MAILBOX_KHR) presentMode = mode;
    if (presentMode == VK_PRESENT_MODE_FIFO_KHR)
      for (VkPresentModeKHR mode : modes)
        if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR) presentMode = mode;
  }

  currentExtent = capabilities.currentExtent;
  if (currentExtent.width == UINT32_MAX) {
    currentExtent.width = std::clamp(width, capabilities.minImageExtent.width,
                                     capabilities.maxImageExtent.width);
    currentExtent.height = std::clamp(height, capabilities.minImageExtent.height,
                                      capabilities.maxImageExtent.height);
  }
  if (currentExtent.width == 0 || currentExtent.height == 0) {
    // No area: the old swapchain goes too, its images and views already gone; beginFrame declines until there is one.
    if (old) vkDestroySwapchainKHR(context.device, old, nullptr);
    swapchain = VK_NULL_HANDLE;
    return;
  }

  std::uint32_t desired = capabilities.minImageCount + 1;
  if (capabilities.maxImageCount > 0) desired = std::min(desired, capabilities.maxImageCount);

  VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  info.surface = context.surface;
  info.minImageCount = desired;
  info.imageFormat = surfaceFormat.format;
  info.imageColorSpace = surfaceFormat.colorSpace;
  info.imageExtent = currentExtent;
  info.imageArrayLayers = 1;
  info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.preTransform = capabilities.currentTransform;
  info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  info.presentMode = presentMode;
  info.clipped = VK_TRUE;
  info.oldSwapchain = old;
  check(vkCreateSwapchainKHR(context.device, &info, nullptr, &swapchain), "vkCreateSwapchainKHR");
  if (old) vkDestroySwapchainKHR(context.device, old, nullptr);

  std::uint32_t count = 0;
  vkGetSwapchainImagesKHR(context.device, swapchain, &count, nullptr);
  images.resize(count);
  vkGetSwapchainImagesKHR(context.device, swapchain, &count, images.data());

  views.resize(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = images[i];
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = surfaceFormat.format;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    check(vkCreateImageView(context.device, &viewInfo, nullptr, &views[i]), "vkCreateImageView");
  }

  for (VkSemaphore semaphore : rendered) vkDestroySemaphore(context.device, semaphore, nullptr);
  rendered.resize(count);
  for (VkSemaphore &semaphore : rendered) {
    VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    check(vkCreateSemaphore(context.device, &semaphoreInfo, nullptr, &semaphore), "vkCreateSemaphore");
  }
}

void Swapchain::recreate(std::uint32_t width, std::uint32_t height, bool useVsync) {
  vkDeviceWaitIdle(context.device);
  vsync = useVsync;
  destroyViews();
  build(width, height, swapchain);
}

bool Swapchain::beginFrame(VkCommandBuffer &command, std::uint32_t &imageIndex) {
  if (!swapchain) return false;
  FrameData &data = frames[frame];
  check(vkWaitForFences(context.device, 1, &data.inFlight, VK_TRUE, UINT64_MAX), "vkWaitForFences");

  const VkResult acquire = vkAcquireNextImageKHR(context.device, swapchain, UINT64_MAX,
                                                 data.acquired, VK_NULL_HANDLE, &imageIndex);
  if (acquire == VK_ERROR_OUT_OF_DATE_KHR) return false;
  if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) check(acquire, "vkAcquireNextImageKHR");

  check(vkResetFences(context.device, 1, &data.inFlight), "vkResetFences");
  check(vkResetCommandBuffer(data.command, 0), "vkResetCommandBuffer");

  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check(vkBeginCommandBuffer(data.command, &begin), "vkBeginCommandBuffer");
  command = data.command;
  return true;
}

bool Swapchain::endFrame(VkCommandBuffer command, std::uint32_t imageIndex) {
  FrameData &data = frames[frame];
  check(vkEndCommandBuffer(command), "vkEndCommandBuffer");

  VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
  wait.semaphore = data.acquired;
  wait.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

  VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
  signal.semaphore = rendered[imageIndex];
  signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

  VkCommandBufferSubmitInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
  commandInfo.commandBuffer = command;

  VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  submit.waitSemaphoreInfoCount = 1;
  submit.pWaitSemaphoreInfos = &wait;
  submit.commandBufferInfoCount = 1;
  submit.pCommandBufferInfos = &commandInfo;
  submit.signalSemaphoreInfoCount = 1;
  submit.pSignalSemaphoreInfos = &signal;
  check(vkQueueSubmit2(context.queue, 1, &submit, data.inFlight), "vkQueueSubmit2");

  VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  present.waitSemaphoreCount = 1;
  present.pWaitSemaphores = &rendered[imageIndex];
  present.swapchainCount = 1;
  present.pSwapchains = &swapchain;
  present.pImageIndices = &imageIndex;
  const VkResult result = vkQueuePresentKHR(context.queue, &present);

  frame = (frame + 1) % kFramesInFlight;
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) return false;
  check(result, "vkQueuePresentKHR");
  return true;
}

} // namespace basalt
