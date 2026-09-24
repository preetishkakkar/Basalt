#include "gpu/Pipeline.h"

#include "core/Log.h"

#include <algorithm>

#include <array>
#include <utility>

namespace basalt {

Pipeline::Pipeline(const Context &ctx, const GraphicsPipelineDescription &description)
    : bindPoint(VK_PIPELINE_BIND_POINT_GRAPHICS), context(&ctx) {
  const Program &program = *description.program;
  std::vector<VkPipelineShaderStageCreateInfo> stages{
      program.vertex().stageInfo(description.vertexSpecialization)};
  if (!description.depthOnly)
    stages.push_back(program.fragment().stageInfo(description.fragmentSpecialization));

  // Attributes from the reflection, packed in location order into the bindings.
  std::vector<VkVertexInputAttributeDescription> attributes;
  std::vector<VkVertexInputBindingDescription> bindings;
  if (!description.bindings.empty() && !description.attributes.empty()) {
    // An explicit layout must place every attribute the entry declares.
    for (const VertexInput &input : program.vertex().vertexInputs) {
      const auto found = std::find_if(description.attributes.begin(), description.attributes.end(),
                                      [&](const VertexAttributeLayout &candidate) {
                                        return candidate.location == input.location;
                                      });
      if (found == description.attributes.end())
        throw Error("pipeline " + description.name + " has no layout for vertex attribute " +
                    input.name + " at location " + std::to_string(input.location));
      attributes.push_back({input.location, found->binding,
                            found->format == VK_FORMAT_UNDEFINED ? input.format() : found->format,
                            found->offset});
    }
    for (const VertexBinding &binding : description.bindings)
      bindings.push_back({binding.binding, binding.stride, binding.rate});
  } else if (!description.bindings.empty()) {
    std::vector<std::uint32_t> offsets(description.bindings.size(), 0);
    for (const VertexInput &input : program.vertex().vertexInputs) {
      std::size_t target = 0;
      for (std::size_t i = 0; i < description.bindings.size(); ++i) {
        if (offsets[i] + input.byteSize() <= description.bindings[i].stride) {
          target = i;
          break;
        }
      }
      if (offsets[target] + input.byteSize() > description.bindings[target].stride)
        throw Error("vertex attribute " + input.name + " does not fit any binding of " +
                    description.name);
      attributes.push_back({input.location, description.bindings[target].binding, input.format(),
                            offsets[target]});
      offsets[target] += input.byteSize();
    }
    for (const VertexBinding &binding : description.bindings)
      bindings.push_back({binding.binding, binding.stride, binding.rate});
  }

  VkPipelineVertexInputStateCreateInfo vertexInput{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vertexInput.vertexBindingDescriptionCount = static_cast<std::uint32_t>(bindings.size());
  vertexInput.pVertexBindingDescriptions = bindings.data();
  vertexInput.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size());
  vertexInput.pVertexAttributeDescriptions = attributes.data();

  VkPipelineInputAssemblyStateCreateInfo assembly{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  assembly.topology = description.topology;

  VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  viewport.viewportCount = 1;
  viewport.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo raster{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  raster.polygonMode = description.polygonMode;
  raster.cullMode = description.cullMode;
  raster.frontFace = description.frontFace;
  raster.lineWidth = 1.0f;
  raster.depthBiasEnable = description.depthBias ? VK_TRUE : VK_FALSE;
  raster.depthClampEnable = description.depthBias ? VK_TRUE : VK_FALSE;

  VkPipelineMultisampleStateCreateInfo multisample{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  multisample.rasterizationSamples = description.samples;

  VkPipelineDepthStencilStateCreateInfo depthStencil{
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  depthStencil.depthTestEnable = description.depthTest ? VK_TRUE : VK_FALSE;
  depthStencil.depthWriteEnable = description.depthWrite ? VK_TRUE : VK_FALSE;
  depthStencil.depthCompareOp = description.depthCompare;

  std::vector<VkPipelineColorBlendAttachmentState> blendStates;
  for (std::size_t i = 0; i < description.colorFormats.size(); ++i) {
    VkPipelineColorBlendAttachmentState state{};
    state.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    // Only the first attachment blends; the G-buffer attachments replace.
    if (description.blend && i == 0) {
      state.blendEnable = VK_TRUE;
      state.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      state.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      state.colorBlendOp = VK_BLEND_OP_ADD;
      state.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      state.alphaBlendOp = VK_BLEND_OP_ADD;
    } else if (description.additive) {
      state.blendEnable = VK_TRUE;
      state.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
      state.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
      state.colorBlendOp = VK_BLEND_OP_ADD;
      state.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      state.alphaBlendOp = VK_BLEND_OP_ADD;
    }
    blendStates.push_back(state);
  }
  VkPipelineColorBlendStateCreateInfo blend{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  blend.attachmentCount = static_cast<std::uint32_t>(blendStates.size());
  blend.pAttachments = blendStates.data();

  std::vector<VkDynamicState> dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  if (description.depthBias) dynamicStates.push_back(VK_DYNAMIC_STATE_DEPTH_BIAS);
  if (description.dynamicFrontFace) dynamicStates.push_back(VK_DYNAMIC_STATE_FRONT_FACE);
  VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
  dynamic.pDynamicStates = dynamicStates.data();

  VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rendering.colorAttachmentCount = static_cast<std::uint32_t>(description.colorFormats.size());
  rendering.pColorAttachmentFormats = description.colorFormats.data();
  rendering.depthAttachmentFormat = description.depthFormat;

  VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  info.pNext = &rendering;
  info.stageCount = static_cast<std::uint32_t>(stages.size());
  info.pStages = stages.data();
  info.pVertexInputState = &vertexInput;
  info.pInputAssemblyState = &assembly;
  info.pViewportState = &viewport;
  info.pRasterizationState = &raster;
  info.pMultisampleState = &multisample;
  info.pDepthStencilState = &depthStencil;
  info.pColorBlendState = &blend;
  info.pDynamicState = &dynamic;
  info.layout = program.layout;
  check(vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &info, nullptr, &handle),
        "vkCreateGraphicsPipelines " + description.name);
  ctx.nameObject(handle, VK_OBJECT_TYPE_PIPELINE, description.name);
}

Pipeline::Pipeline(const Context &ctx, const Program &program, const std::string &name)
    : bindPoint(VK_PIPELINE_BIND_POINT_COMPUTE), context(&ctx) {
  VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  info.stage = program.compute().stageInfo();
  info.layout = program.layout;
  check(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &info, nullptr, &handle),
        "vkCreateComputePipelines " + name);
  ctx.nameObject(handle, VK_OBJECT_TYPE_PIPELINE, name);
}

Pipeline &Pipeline::operator=(Pipeline &&other) noexcept {
  if (this == &other) return *this;
  if (handle && context) vkDestroyPipeline(context->device, handle, nullptr);
  handle = std::exchange(other.handle, VK_NULL_HANDLE);
  bindPoint = other.bindPoint;
  context = std::exchange(other.context, nullptr);
  return *this;
}

Pipeline::~Pipeline() {
  if (handle && context) vkDestroyPipeline(context->device, handle, nullptr);
}

} // namespace basalt
