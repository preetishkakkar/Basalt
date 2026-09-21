// Dear ImGui through the engine's own Metal-compiled pipeline, replacing the library's GLSL backend.
#pragma once
#include "gpu/Descriptors.h"
#include "gpu/Pipeline.h"
#include "gpu/Uploader.h"

#include <memory>
#include <vector>

struct ImDrawData;

namespace basalt {

class UiPass {
public:
  UiPass(const Context &context, Uploader &uploader, VkFormat colorFormat);
  ~UiPass();
  UiPass(const UiPass &) = delete;
  UiPass &operator=(const UiPass &) = delete;

  void record(VkCommandBuffer command, ImDrawData *drawData, std::uint32_t frameIndex);

private:
  struct FrameBuffers {
    Buffer vertices;
    Buffer indices;
    Buffer uniforms;
    std::size_t vertexCapacity = 0;
    std::size_t indexCapacity = 0;
    VkDescriptorSet vertexSet = VK_NULL_HANDLE;
  };

  const Context &context;
  std::unique_ptr<Program> program;
  Pipeline pipeline;
  std::unique_ptr<DescriptorPool> pool;
  Image fontAtlas;
  VkSampler sampler = VK_NULL_HANDLE;
  VkDescriptorSet fontSet = VK_NULL_HANDLE;
  std::vector<FrameBuffers> frames;
};

} // namespace basalt
