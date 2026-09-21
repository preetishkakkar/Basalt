#include "gpu/Resources.h"

#include "core/Log.h"

#include <cstring>
#include <utility>

namespace basalt {

Buffer::Buffer(const Context &ctx, VkDeviceSize bytes, VkBufferUsageFlags usage,
               VmaMemoryUsage memory, VmaAllocationCreateFlags flags, const std::string &name)
    : size(bytes), context(&ctx) {
  if (bytes == 0) throw Error("a buffer of zero bytes was requested: " + name);
  VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bufferInfo.size = bytes;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo allocationInfo{};
  allocationInfo.usage = memory;
  allocationInfo.flags = flags;

  VmaAllocationInfo result{};
  check(vmaCreateBuffer(ctx.allocator, &bufferInfo, &allocationInfo, &handle, &allocation, &result),
        "vmaCreateBuffer " + name);
  mapped = result.pMappedData;
  if (!name.empty()) ctx.nameObject(handle, VK_OBJECT_TYPE_BUFFER, name);
}

Buffer &Buffer::operator=(Buffer &&other) noexcept {
  if (this == &other) return *this;
  reset();
  handle = std::exchange(other.handle, VK_NULL_HANDLE);
  allocation = std::exchange(other.allocation, VK_NULL_HANDLE);
  size = std::exchange(other.size, 0);
  mapped = std::exchange(other.mapped, nullptr);
  context = std::exchange(other.context, nullptr);
  return *this;
}

Buffer::~Buffer() { reset(); }

void Buffer::reset() {
  if (handle && context) vmaDestroyBuffer(context->allocator, handle, allocation);
  handle = VK_NULL_HANDLE;
  allocation = VK_NULL_HANDLE;
  size = 0;
  mapped = nullptr;
}

VkDeviceAddress Buffer::deviceAddress() const {
  VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  info.buffer = handle;
  return vkGetBufferDeviceAddress(context->device, &info);
}

void Buffer::write(const void *data, std::size_t bytes, std::size_t offset) {
  if (!mapped) throw Error("write to a buffer that is not host visible");
  if (offset + bytes > size) throw Error("write past the end of a buffer");
  std::memcpy(static_cast<std::uint8_t *>(mapped) + offset, data, bytes);
}

Image::Image(const Context &ctx, const ImageDescription &desc) : description(desc), context(&ctx) {
  VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  imageInfo.imageType = desc.depth > 1 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
  imageInfo.format = desc.format;
  imageInfo.extent = {desc.width, desc.height, desc.depth};
  imageInfo.mipLevels = desc.mipLevels;
  imageInfo.arrayLayers = desc.arrayLayers;
  imageInfo.samples = desc.samples;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.usage = desc.usage;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (desc.cube) imageInfo.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
  // A cube a kernel writes needs a 2D array view of the same memory.
  if (desc.cube && (desc.usage & VK_IMAGE_USAGE_STORAGE_BIT))
    imageInfo.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;

  VmaAllocationCreateInfo allocationInfo{};
  allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
  check(vmaCreateImage(ctx.allocator, &imageInfo, &allocationInfo, &handle, &allocation, nullptr),
        "vmaCreateImage " + desc.name);

  const VkImageViewType type = desc.cube ? (desc.arrayLayers > 6 ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY
                                                                 : VK_IMAGE_VIEW_TYPE_CUBE)
                               : desc.depth > 1 ? VK_IMAGE_VIEW_TYPE_3D
                               : desc.arrayLayers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                                                      : VK_IMAGE_VIEW_TYPE_2D;
  view = createView(0, desc.mipLevels, 0, desc.arrayLayers, type);
  if (!desc.name.empty()) {
    ctx.nameObject(handle, VK_OBJECT_TYPE_IMAGE, desc.name);
    ctx.nameObject(view, VK_OBJECT_TYPE_IMAGE_VIEW, desc.name + ".view");
  }
}

Image &Image::operator=(Image &&other) noexcept {
  if (this == &other) return *this;
  reset();
  handle = std::exchange(other.handle, VK_NULL_HANDLE);
  allocation = std::exchange(other.allocation, VK_NULL_HANDLE);
  view = std::exchange(other.view, VK_NULL_HANDLE);
  description = std::move(other.description);
  layout = std::exchange(other.layout, VK_IMAGE_LAYOUT_UNDEFINED);
  context = std::exchange(other.context, nullptr);
  return *this;
}

Image::~Image() { reset(); }

void Image::reset() {
  if (context) {
    if (view) vkDestroyImageView(context->device, view, nullptr);
    if (handle) vmaDestroyImage(context->allocator, handle, allocation);
  }
  view = VK_NULL_HANDLE;
  handle = VK_NULL_HANDLE;
  allocation = VK_NULL_HANDLE;
  layout = VK_IMAGE_LAYOUT_UNDEFINED;
}

VkImageAspectFlags Image::aspect() const {
  switch (description.format) {
  case VK_FORMAT_D16_UNORM:
  case VK_FORMAT_D32_SFLOAT:
  case VK_FORMAT_X8_D24_UNORM_PACK32:
    return VK_IMAGE_ASPECT_DEPTH_BIT;
  case VK_FORMAT_D16_UNORM_S8_UINT:
  case VK_FORMAT_D24_UNORM_S8_UINT:
  case VK_FORMAT_D32_SFLOAT_S8_UINT:
    return VK_IMAGE_ASPECT_DEPTH_BIT;
  default:
    return VK_IMAGE_ASPECT_COLOR_BIT;
  }
}

VkImageView Image::createView(std::uint32_t baseMip, std::uint32_t mipCount, std::uint32_t baseLayer,
                              std::uint32_t layerCount, VkImageViewType type) const {
  VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  viewInfo.image = handle;
  viewInfo.viewType = type;
  viewInfo.format = description.format;
  viewInfo.subresourceRange.aspectMask = aspect();
  viewInfo.subresourceRange.baseMipLevel = baseMip;
  viewInfo.subresourceRange.levelCount = mipCount;
  viewInfo.subresourceRange.baseArrayLayer = baseLayer;
  viewInfo.subresourceRange.layerCount = layerCount;
  VkImageView created = VK_NULL_HANDLE;
  check(vkCreateImageView(context->device, &viewInfo, nullptr, &created),
        "vkCreateImageView " + description.name);
  return created;
}

void transitionImage(VkCommandBuffer command, Image &image, VkImageLayout newLayout,
                     VkPipelineStageFlags2 sourceStage, VkAccessFlags2 sourceAccess,
                     VkPipelineStageFlags2 destinationStage, VkAccessFlags2 destinationAccess,
                     std::uint32_t baseMip, std::uint32_t mipCount) {
  VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  barrier.srcStageMask = sourceStage;
  barrier.srcAccessMask = sourceAccess;
  barrier.dstStageMask = destinationStage;
  barrier.dstAccessMask = destinationAccess;
  barrier.oldLayout = image.layout;
  barrier.newLayout = newLayout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image.handle;
  barrier.subresourceRange.aspectMask = image.aspect();
  barrier.subresourceRange.baseMipLevel = baseMip;
  barrier.subresourceRange.levelCount = mipCount;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.imageMemoryBarrierCount = 1;
  dependency.pImageMemoryBarriers = &barrier;
  vkCmdPipelineBarrier2(command, &dependency);
  image.layout = newLayout;
}

void computeImageBarrier(VkCommandBuffer command, VkImage image, VkImageAspectFlags aspect,
                         std::uint32_t baseMip, std::uint32_t mipCount, std::uint32_t layerCount) {
  VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
  barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
  barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange = {aspect, baseMip, mipCount, 0, layerCount};

  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.imageMemoryBarrierCount = 1;
  dependency.pImageMemoryBarriers = &barrier;
  vkCmdPipelineBarrier2(command, &dependency);
}

std::uint32_t mipLevelsFor(std::uint32_t width, std::uint32_t height) {
  std::uint32_t levels = 1;
  std::uint32_t extent = width > height ? width : height;
  while (extent > 1) {
    extent >>= 1;
    ++levels;
  }
  return levels;
}

} // namespace basalt
