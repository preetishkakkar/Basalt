// Swapchain and per-frame sync; two frames in flight.
#pragma once
#include "gpu/Context.h"

#include <cstdint>
#include <vector>

namespace basalt {

inline constexpr std::uint32_t kFramesInFlight = 2;

class Swapchain {
public:
  Swapchain(const Context &context, std::uint32_t width, std::uint32_t height, bool vsync);
  ~Swapchain();
  Swapchain(const Swapchain &) = delete;
  Swapchain &operator=(const Swapchain &) = delete;

  void recreate(std::uint32_t width, std::uint32_t height, bool vsync);

  // Returns false when out of date; the caller recreates and skips the frame.
  bool beginFrame(VkCommandBuffer &command, std::uint32_t &imageIndex);
  bool endFrame(VkCommandBuffer command, std::uint32_t imageIndex);

  VkFormat format() const { return surfaceFormat.format; }
  VkExtent2D extent() const { return currentExtent; }
  std::uint32_t imageCount() const { return static_cast<std::uint32_t>(images.size()); }
  VkImage image(std::uint32_t index) const { return images[index]; }
  VkImageView view(std::uint32_t index) const { return views[index]; }
  std::uint32_t frameIndex() const { return frame; }
  bool vsyncEnabled() const { return vsync; }

private:
  void build(std::uint32_t width, std::uint32_t height, VkSwapchainKHR old);
  void destroyViews();

  const Context &context;
  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  VkSurfaceFormatKHR surfaceFormat{};
  VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
  VkExtent2D currentExtent{};
  bool vsync = true;
  std::vector<VkImage> images;
  std::vector<VkImageView> views;

  VkCommandPool commandPool = VK_NULL_HANDLE;
  struct FrameData {
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkSemaphore acquired = VK_NULL_HANDLE;
    VkFence inFlight = VK_NULL_HANDLE;
  };
  FrameData frames[kFramesInFlight]{};
  // One per image: presentation must wait on the semaphore that drew that image.
  std::vector<VkSemaphore> rendered;
  std::uint32_t frame = 0;
};

} // namespace basalt
