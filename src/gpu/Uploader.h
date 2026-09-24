// One-shot GPU work at load time: staging uploads, mips, immediate commands.
#pragma once
#include "gpu/Resources.h"

#include <functional>
#include <vector>

namespace basalt {

class Uploader {
public:
  explicit Uploader(const Context &context);
  ~Uploader();
  Uploader(const Uploader &) = delete;
  Uploader &operator=(const Uploader &) = delete;

  void runImmediate(const std::function<void(VkCommandBuffer)> &record);

  Buffer createBuffer(const void *data, VkDeviceSize bytes, VkBufferUsageFlags usage,
                      const std::string &name);

  // mipLevels 0 generates a full chain.
  Image createTexture(const void *texels, VkDeviceSize bytes, std::uint32_t width,
                      std::uint32_t height, VkFormat format, std::uint32_t mipLevels,
                      const std::string &name);

  // Copies back to the host, waiting for the device: a buffer's first `bytes`, or an
  // image's level zero as tightly packed texels. The buffer needs TRANSFER_SRC usage.
  std::vector<std::uint8_t> readBuffer(const Buffer &buffer, VkDeviceSize bytes);
  std::vector<std::uint8_t> readImage(Image &image, std::uint32_t texelBytes);

private:
  const Context &context;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkCommandBuffer command = VK_NULL_HANDLE;
  // Staging buffers live until the work that reads them has finished.
  std::vector<Buffer> staging;
};

} // namespace basalt
