#include "render/UiPass.h"

#include "core/Log.h"
#include "gpu/Swapchain.h"

#include <imgui.h>

#include <algorithm>
#include <cstring>

namespace basalt {
namespace {

// Mirrors UiTransform in shaders/slang/entries/ui.slang: the vertex stage's push constants.
struct UiTransform {
  float scaleAndTranslate[4];
};

} // namespace

UiPass::UiPass(const Context &ctx, Uploader &uploader, VkFormat colorFormat) : context(ctx) {
  program = std::make_unique<Program>(ctx, "ui_vertex", "ui_fragment");

  GraphicsPipelineDescription description;
  description.program = program.get();
  description.colorFormats = {colorFormat};
  description.depthFormat = VK_FORMAT_UNDEFINED;
  description.bindings = {{0, sizeof(ImDrawVert), VK_VERTEX_INPUT_RATE_VERTEX}};
  description.attributes = {
      {0, 0, offsetof(ImDrawVert, pos), VK_FORMAT_R32G32_SFLOAT},
      {1, 0, offsetof(ImDrawVert, uv), VK_FORMAT_R32G32_SFLOAT},
      // Four bytes read as a float4, the compiler's packed format for a float attribute.
      {2, 0, offsetof(ImDrawVert, col), VK_FORMAT_R8G8B8A8_UNORM},
  };
  description.cullMode = VK_CULL_MODE_NONE;
  description.depthTest = false;
  description.depthWrite = false;
  description.blend = true;
  description.name = "ui";
  pipeline = Pipeline(ctx, description);

  pool = std::make_unique<DescriptorPool>(ctx, 32);

  // Expanded to four channels, so one shader serves fonts and user textures.
  ImGuiIO &io = ImGui::GetIO();
  unsigned char *pixels = nullptr;
  int width = 0, height = 0;
  io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
  const VkDeviceSize bytes = static_cast<VkDeviceSize>(width) * height * 4;
  fontAtlas = uploader.createTexture(pixels, bytes, static_cast<std::uint32_t>(width),
                                     static_cast<std::uint32_t>(height), VK_FORMAT_R8G8B8A8_UNORM, 1,
                                     "ui.font");

  VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  check(vkCreateSampler(ctx.device, &samplerInfo, nullptr, &sampler), "vkCreateSampler (ui)");

  fontSet = program->allocate(*pool);
  DescriptorWriter(ctx, *program, fontSet)
      .texture("atlas", fontAtlas)
      .sampler("atlasSampler", sampler)
      .apply();
  io.Fonts->SetTexID(reinterpret_cast<ImTextureID>(fontSet));

  frames.resize(kFramesInFlight);
}

UiPass::~UiPass() {
  if (sampler) vkDestroySampler(context.device, sampler, nullptr);
}

void UiPass::record(VkCommandBuffer command, ImDrawData *drawData, std::uint32_t frameIndex) {
  if (!drawData || drawData->TotalVtxCount == 0) return;
  const int framebufferWidth = static_cast<int>(drawData->DisplaySize.x * drawData->FramebufferScale.x);
  const int framebufferHeight = static_cast<int>(drawData->DisplaySize.y * drawData->FramebufferScale.y);
  if (framebufferWidth <= 0 || framebufferHeight <= 0) return;

  FrameBuffers &frame = frames[frameIndex % frames.size()];

  const std::size_t vertexBytes = static_cast<std::size_t>(drawData->TotalVtxCount) * sizeof(ImDrawVert);
  const std::size_t indexBytes = static_cast<std::size_t>(drawData->TotalIdxCount) * sizeof(ImDrawIdx);
  if (frame.vertexCapacity < vertexBytes) {
    frame.vertexCapacity = std::max<std::size_t>(vertexBytes * 2, 64 * 1024);
    frame.vertices = Buffer(context, frame.vertexCapacity, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            VMA_MEMORY_USAGE_AUTO,
                            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                VMA_ALLOCATION_CREATE_MAPPED_BIT,
                            "ui.vertices");
  }
  if (frame.indexCapacity < indexBytes) {
    frame.indexCapacity = std::max<std::size_t>(indexBytes * 2, 32 * 1024);
    frame.indices = Buffer(context, frame.indexCapacity, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                           VMA_MEMORY_USAGE_AUTO,
                           VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                               VMA_ALLOCATION_CREATE_MAPPED_BIT,
                           "ui.indices");
  }

  auto *vertexTarget = static_cast<ImDrawVert *>(frame.vertices.mapped);
  auto *indexTarget = static_cast<ImDrawIdx *>(frame.indices.mapped);
  for (int list = 0; list < drawData->CmdListsCount; ++list) {
    const ImDrawList *commands = drawData->CmdLists[list];
    std::memcpy(vertexTarget, commands->VtxBuffer.Data,
                static_cast<std::size_t>(commands->VtxBuffer.Size) * sizeof(ImDrawVert));
    std::memcpy(indexTarget, commands->IdxBuffer.Data,
                static_cast<std::size_t>(commands->IdxBuffer.Size) * sizeof(ImDrawIdx));
    vertexTarget += commands->VtxBuffer.Size;
    indexTarget += commands->IdxBuffer.Size;
  }

  UiTransform transform{};
  transform.scaleAndTranslate[0] = 2.0f / drawData->DisplaySize.x;
  transform.scaleAndTranslate[1] = 2.0f / drawData->DisplaySize.y;
  transform.scaleAndTranslate[2] = -1.0f - drawData->DisplayPos.x * transform.scaleAndTranslate[0];
  transform.scaleAndTranslate[3] = -1.0f - drawData->DisplayPos.y * transform.scaleAndTranslate[1];

  vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.handle);
  program->push(command, transform);
  const VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(command, 0, 1, &frame.vertices.handle, &offset);
  vkCmdBindIndexBuffer(command, frame.indices.handle, 0,
                       sizeof(ImDrawIdx) == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);

  VkViewport viewport{0.0f, 0.0f, static_cast<float>(framebufferWidth),
                      static_cast<float>(framebufferHeight), 0.0f, 1.0f};
  vkCmdSetViewport(command, 0, 1, &viewport);

  const ImVec2 clipOffset = drawData->DisplayPos;
  const ImVec2 clipScale = drawData->FramebufferScale;
  std::uint32_t vertexOffset = 0, indexOffset = 0;
  VkDescriptorSet boundTexture = VK_NULL_HANDLE;
  for (int list = 0; list < drawData->CmdListsCount; ++list) {
    const ImDrawList *commands = drawData->CmdLists[list];
    for (int i = 0; i < commands->CmdBuffer.Size; ++i) {
      const ImDrawCmd &draw = commands->CmdBuffer[i];
      if (draw.UserCallback) {
        draw.UserCallback(commands, &draw);
        continue;
      }
      // Clamped so a window dragged off the edge gives no negative offset.
      ImVec2 minimum((draw.ClipRect.x - clipOffset.x) * clipScale.x,
                     (draw.ClipRect.y - clipOffset.y) * clipScale.y);
      ImVec2 maximum((draw.ClipRect.z - clipOffset.x) * clipScale.x,
                     (draw.ClipRect.w - clipOffset.y) * clipScale.y);
      minimum.x = std::max(minimum.x, 0.0f);
      minimum.y = std::max(minimum.y, 0.0f);
      maximum.x = std::min(maximum.x, static_cast<float>(framebufferWidth));
      maximum.y = std::min(maximum.y, static_cast<float>(framebufferHeight));
      if (maximum.x <= minimum.x || maximum.y <= minimum.y) continue;

      VkRect2D scissor{{static_cast<std::int32_t>(minimum.x), static_cast<std::int32_t>(minimum.y)},
                       {static_cast<std::uint32_t>(maximum.x - minimum.x),
                        static_cast<std::uint32_t>(maximum.y - minimum.y)}};
      vkCmdSetScissor(command, 0, 1, &scissor);

      auto texture = reinterpret_cast<VkDescriptorSet>(draw.GetTexID());
      if (!texture) texture = fontSet;
      if (texture != boundTexture) {
        program->bind(command, texture);
        boundTexture = texture;
      }
      vkCmdDrawIndexed(command, draw.ElemCount, 1, draw.IdxOffset + indexOffset,
                       static_cast<std::int32_t>(draw.VtxOffset + vertexOffset), 0);
    }
    indexOffset += static_cast<std::uint32_t>(commands->IdxBuffer.Size);
    vertexOffset += static_cast<std::uint32_t>(commands->VtxBuffer.Size);
  }
}

} // namespace basalt
