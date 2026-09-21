#pragma once
#include "gpu/Context.h"

#include <cstdint>
#include <string>
#include <vector>

namespace basalt {

class Buffer {
public:
  Buffer() = default;
  Buffer(const Context &context, VkDeviceSize size, VkBufferUsageFlags usage,
         VmaMemoryUsage memory = VMA_MEMORY_USAGE_AUTO,
         VmaAllocationCreateFlags flags = 0, const std::string &name = {});
  ~Buffer();
  Buffer(Buffer &&other) noexcept { *this = std::move(other); }
  Buffer &operator=(Buffer &&other) noexcept;
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  VkBuffer handle = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  void *mapped = nullptr; // Non-null when created with a host-visible flag.

  explicit operator bool() const { return handle != VK_NULL_HANDLE; }
  void write(const void *data, std::size_t bytes, std::size_t offset = 0);
  VkDeviceAddress deviceAddress() const;
  VkDescriptorBufferInfo descriptor(VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE) const {
    return {handle, offset, range};
  }
  void reset();

private:
  const Context *context = nullptr;
};

struct ImageDescription {
  VkFormat format = VK_FORMAT_UNDEFINED;
  std::uint32_t width = 1, height = 1, depth = 1;
  std::uint32_t mipLevels = 1;
  std::uint32_t arrayLayers = 1;
  VkImageUsageFlags usage = 0;
  VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
  bool cube = false;
  std::string name;
};

class Image {
public:
  Image() = default;
  Image(const Context &context, const ImageDescription &description);
  ~Image();
  Image(Image &&other) noexcept { *this = std::move(other); }
  Image &operator=(Image &&other) noexcept;
  Image(const Image &) = delete;
  Image &operator=(const Image &) = delete;

  VkImage handle = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;   // The whole resource, as the shader samples it.
  ImageDescription description;
  // Tracked layout; one queue and one thread keep it accurate.
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;

  explicit operator bool() const { return handle != VK_NULL_HANDLE; }
  VkImageAspectFlags aspect() const;
  VkImageView createView(std::uint32_t baseMip, std::uint32_t mipCount, std::uint32_t baseLayer,
                         std::uint32_t layerCount, VkImageViewType type) const;
  void reset();

private:
  const Context *context = nullptr;
};

void transitionImage(VkCommandBuffer command, Image &image, VkImageLayout newLayout,
                     VkPipelineStageFlags2 sourceStage, VkAccessFlags2 sourceAccess,
                     VkPipelineStageFlags2 destinationStage, VkAccessFlags2 destinationAccess,
                     std::uint32_t baseMip = 0, std::uint32_t mipCount = VK_REMAINING_MIP_LEVELS);

void computeImageBarrier(VkCommandBuffer command, VkImage image, VkImageAspectFlags aspect,
                         std::uint32_t baseMip, std::uint32_t mipCount, std::uint32_t layerCount);

std::uint32_t mipLevelsFor(std::uint32_t width, std::uint32_t height);

} // namespace basalt
