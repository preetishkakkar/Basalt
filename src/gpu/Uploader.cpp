#include "gpu/Uploader.h"

#include "core/Log.h"

#include <cstring>

namespace basalt {

Uploader::Uploader(const Context &ctx) : context(ctx) {
  VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = ctx.queueFamily;
  check(vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &pool), "vkCreateCommandPool (upload)");

  VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocate.commandPool = pool;
  allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocate.commandBufferCount = 1;
  check(vkAllocateCommandBuffers(ctx.device, &allocate, &command), "vkAllocateCommandBuffers (upload)");

  VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  check(vkCreateFence(ctx.device, &fenceInfo, nullptr, &fence), "vkCreateFence (upload)");
}

Uploader::~Uploader() {
  if (fence) vkDestroyFence(context.device, fence, nullptr);
  if (pool) vkDestroyCommandPool(context.device, pool, nullptr);
}

void Uploader::runImmediate(const std::function<void(VkCommandBuffer)> &record) {
  check(vkResetCommandBuffer(command, 0), "vkResetCommandBuffer (upload)");
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer (upload)");
  record(command);
  check(vkEndCommandBuffer(command), "vkEndCommandBuffer (upload)");

  VkCommandBufferSubmitInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
  commandInfo.commandBuffer = command;
  VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  submit.commandBufferInfoCount = 1;
  submit.pCommandBufferInfos = &commandInfo;
  check(vkQueueSubmit2(context.queue, 1, &submit, fence), "vkQueueSubmit2 (upload)");
  check(vkWaitForFences(context.device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences (upload)");
  check(vkResetFences(context.device, 1, &fence), "vkResetFences (upload)");
  staging.clear();
}

Buffer Uploader::createBuffer(const void *data, VkDeviceSize bytes, VkBufferUsageFlags usage,
                              const std::string &name) {
  Buffer source(context, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT,
                name + ".staging");
  source.write(data, static_cast<std::size_t>(bytes));

  Buffer destination(context, bytes, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VMA_MEMORY_USAGE_AUTO, 0, name);
  const VkBuffer from = source.handle, to = destination.handle;
  runImmediate([&](VkCommandBuffer cmd) {
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(cmd, from, to, 1, &region);
  });
  return destination;
}

Image Uploader::createTexture(const void *texels, VkDeviceSize bytes, std::uint32_t width,
                              std::uint32_t height, VkFormat format, std::uint32_t mipLevels,
                              const std::string &name) {
  const bool generateMips = mipLevels == 0;
  const std::uint32_t levels = generateMips ? mipLevelsFor(width, height) : mipLevels;

  ImageDescription description;
  description.format = format;
  description.width = width;
  description.height = height;
  description.mipLevels = levels;
  description.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  description.name = name;
  Image image(context, description);

  Buffer source(context, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT,
                name + ".staging");
  source.write(texels, static_cast<std::size_t>(bytes));

  runImmediate([&](VkCommandBuffer cmd) {
    transitionImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COPY_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(cmd, source.handle, image.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &region);

    std::int32_t mipWidth = static_cast<std::int32_t>(width);
    std::int32_t mipHeight = static_cast<std::int32_t>(height);
    for (std::uint32_t level = 1; level < levels; ++level) {
      VkImageMemoryBarrier2 toSource{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
      // Level zero came from a copy, the rest from blits.
      toSource.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
      toSource.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
      toSource.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
      toSource.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
      toSource.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      toSource.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      toSource.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      toSource.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      toSource.image = image.handle;
      toSource.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 1, 0, 1};
      VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      dependency.imageMemoryBarrierCount = 1;
      dependency.pImageMemoryBarriers = &toSource;
      vkCmdPipelineBarrier2(cmd, &dependency);

      const std::int32_t nextWidth = mipWidth > 1 ? mipWidth / 2 : 1;
      const std::int32_t nextHeight = mipHeight > 1 ? mipHeight / 2 : 1;
      VkImageBlit blit{};
      blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1};
      blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
      blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
      blit.dstOffsets[1] = {nextWidth, nextHeight, 1};
      vkCmdBlitImage(cmd, image.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image.handle,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
      mipWidth = nextWidth;
      mipHeight = nextHeight;
    }

    // All but the last level are in TRANSFER_SRC, the last in TRANSFER_DST.
    VkImageMemoryBarrier2 barriers[2]{};
    for (VkImageMemoryBarrier2 &barrier : barriers) {
      barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
      barrier.srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
      barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
      barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image = image.handle;
    }
    barriers[0].srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    barriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, levels - 1, 0, 1};
    barriers[1].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barriers[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, levels - 1, 1, 0, 1};

    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.imageMemoryBarrierCount = levels > 1 ? 2u : 1u;
    dependency.pImageMemoryBarriers = levels > 1 ? barriers : &barriers[1];
    vkCmdPipelineBarrier2(cmd, &dependency);
  });

  image.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  return image;
}

} // namespace basalt
