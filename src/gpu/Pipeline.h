// Dynamic rendering: a pipeline names attachment formats, not a render pass; the vertex layout comes from the reflection.
#pragma once
#include "gpu/Descriptors.h"

#include <vector>

namespace basalt {

struct VertexBinding {
  std::uint32_t binding = 0;
  std::uint32_t stride = 0;
  VkVertexInputRate rate = VK_VERTEX_INPUT_RATE_VERTEX;
};

// A pass reading a subset of the vertex must still find its fields at the shared buffer's offsets.
struct VertexAttributeLayout {
  std::uint32_t location = 0;
  std::uint32_t binding = 0;
  std::uint32_t offset = 0;
  VkFormat format = VK_FORMAT_UNDEFINED; // Undefined takes the reflected type's format.
};

struct GraphicsPipelineDescription {
  const Program *program = nullptr;
  std::vector<VkFormat> colorFormats;
  VkFormat depthFormat = VK_FORMAT_UNDEFINED;
  // Empty bindings: no vertex buffers, for fullscreen passes.
  std::vector<VertexBinding> bindings;
  // Empty: packed in location order; otherwise every declared attribute must appear.
  std::vector<VertexAttributeLayout> attributes;
  VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;
  VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  VkPolygonMode polygonMode = VK_POLYGON_MODE_FILL;
  bool depthTest = true;
  bool depthWrite = true;
  VkCompareOp depthCompare = VK_COMPARE_OP_GREATER_OR_EQUAL; // Reverse-Z everywhere but the shadow pass.
  bool depthBias = false;
  // No fragment stage: the profile has no `fragment void`, so a depth-only pass would otherwise write a colour nothing receives.
  bool depthOnly = false;
  bool blend = false;                 // Straight alpha over the destination.
  bool additive = false;              // Add, for the bloom composite.
  VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
  const VkSpecializationInfo *vertexSpecialization = nullptr;
  const VkSpecializationInfo *fragmentSpecialization = nullptr;
  std::string name;
};

class Pipeline {
public:
  Pipeline() = default;
  Pipeline(const Context &context, const GraphicsPipelineDescription &description);
  Pipeline(const Context &context, const Program &program, const std::string &name);
  ~Pipeline();
  Pipeline(Pipeline &&other) noexcept { *this = std::move(other); }
  Pipeline &operator=(Pipeline &&other) noexcept;
  Pipeline(const Pipeline &) = delete;
  Pipeline &operator=(const Pipeline &) = delete;

  VkPipeline handle = VK_NULL_HANDLE;
  VkPipelineBindPoint bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  explicit operator bool() const { return handle != VK_NULL_HANDLE; }

private:
  const Context *context = nullptr;
};

} // namespace basalt
