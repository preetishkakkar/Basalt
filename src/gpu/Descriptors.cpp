#include "gpu/Descriptors.h"

#include "core/Log.h"

#include <array>

namespace basalt {

DescriptorPool::DescriptorPool(const Context &ctx, std::uint32_t perPool)
    : context(ctx), setsPerPool(perPool) {
  pools.push_back(createPool());
}

DescriptorPool::~DescriptorPool() {
  for (VkDescriptorPool pool : pools) vkDestroyDescriptorPool(context.device, pool, nullptr);
}

VkDescriptorPool DescriptorPool::createPool() const {
  // Generous; a second pool is allocated the moment one fills.
  std::vector<VkDescriptorPoolSize> sizes{
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, setsPerPool * 4},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, setsPerPool * 6},
      // Sized for the 120-slot texture table on a ray tracing device.
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, setsPerPool * (context.rayTracingSupported ? 144 : 16)},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, setsPerPool * 2},
      {VK_DESCRIPTOR_TYPE_SAMPLER, setsPerPool * 4},
  };
  if (context.rayTracingSupported)
    sizes.push_back({VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, setsPerPool});
  VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  info.maxSets = setsPerPool;
  info.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
  info.pPoolSizes = sizes.data();
  VkDescriptorPool pool = VK_NULL_HANDLE;
  check(vkCreateDescriptorPool(context.device, &info, nullptr, &pool), "vkCreateDescriptorPool");
  return pool;
}

VkDescriptorSet DescriptorPool::allocate(VkDescriptorSetLayout layout) {
  for (;;) {
    VkDescriptorSetAllocateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    info.descriptorPool = pools[current];
    info.descriptorSetCount = 1;
    info.pSetLayouts = &layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    const VkResult result = vkAllocateDescriptorSets(context.device, &info, &set);
    if (result == VK_SUCCESS) return set;
    if (result != VK_ERROR_OUT_OF_POOL_MEMORY && result != VK_ERROR_FRAGMENTED_POOL)
      check(result, "vkAllocateDescriptorSets");
    ++current;
    if (current == pools.size()) pools.push_back(createPool());
  }
}

void DescriptorPool::reset() {
  for (VkDescriptorPool pool : pools) vkResetDescriptorPool(context.device, pool, 0);
  current = 0;
}

Program::Program(const Context &ctx, std::string vertexEntry, std::string fragmentEntry)
    : context(ctx) {
  vertexShader = std::make_unique<Shader>(ctx, vertexEntry, VK_SHADER_STAGE_VERTEX_BIT);
  fragmentShader = std::make_unique<Shader>(ctx, fragmentEntry, VK_SHADER_STAGE_FRAGMENT_BIT);
  build();
}

Program::Program(const Context &ctx, std::string computeEntry) : context(ctx) {
  computeShader = std::make_unique<Shader>(ctx, computeEntry, VK_SHADER_STAGE_COMPUTE_BIT);
  build();
}

Program::~Program() {
  for (VkDescriptorSetLayout setLayout : setLayouts)
    if (setLayout) vkDestroyDescriptorSetLayout(context.device, setLayout, nullptr);
  if (layout) vkDestroyPipelineLayout(context.device, layout, nullptr);
}

void Program::build() {
  requireDeviceSupport();

  // Vertex or compute stage in set 0, fragment in set 1: the compiler's binding contract.
  auto bindingsFor = [](const Shader &shader) {
    return m2v::host::setLayoutBindings(m2v::host::descriptorBindings(shader.reflection),
                                        shader.reflection.set, shader.stageFlag);
  };

  std::vector<VkDescriptorSetLayoutBinding> perSet[2];
  if (computeShader) {
    perSet[0] = bindingsFor(*computeShader);
  } else {
    perSet[0] = bindingsFor(*vertexShader);
    perSet[1] = bindingsFor(*fragmentShader);
  }

  for (int set = 0; set < 2; ++set) {
    VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.bindingCount = static_cast<std::uint32_t>(perSet[set].size());
    info.pBindings = perSet[set].data();
    check(vkCreateDescriptorSetLayout(context.device, &info, nullptr, &setLayouts[set]),
          "vkCreateDescriptorSetLayout");
  }

  VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layoutInfo.setLayoutCount = computeShader ? 1u : 2u;
  layoutInfo.pSetLayouts = setLayouts;
  check(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &layout),
        "vkCreatePipelineLayout");
}

void Program::requireDeviceSupport() const {
  auto checkStage = [&](const Shader *shader) {
    if (!shader) return;
    for (const std::string &feature : shader->reflection.requiredFeatures) {
      // Checked against what the device was created with, so a missing feature fails by name.
      if (!context.enabledFeatures.contains(feature))
        throw Error("shader " + shader->entry + " requires the device feature " + feature +
                    ", which this device does not offer or this engine does not enable");
    }
    for (const std::string &property : shader->reflection.requiredProperties)
      if (!m2v::host::supportsProperty(context.physical, property))
        throw Error("shader " + shader->entry + " requires the device property " + property);
  };
  checkStage(vertexShader.get());
  checkStage(fragmentShader.get());
  checkStage(computeShader.get());
}

DescriptorWriter::DescriptorWriter(const Context &ctx, const Shader &sh, VkDescriptorSet target)
    : context(ctx), shader(sh), set(target) {}

DescriptorWriter &DescriptorWriter::buffer(const std::string &name, const Buffer &resource,
                                           VkDeviceSize offset, VkDeviceSize range) {
  const m2v::host::Buffer &declared = shader.buffer(name);
  if (!resource) throw Error("no buffer bound for " + name + " in " + shader.entry);
  bufferInfos.push_back(std::make_unique<VkDescriptorBufferInfo>(
      VkDescriptorBufferInfo{resource.handle, offset, range}));

  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = set;
  write.dstBinding = declared.binding;
  write.descriptorCount = 1;
  write.descriptorType = declared.uniform ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                          : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.pBufferInfo = bufferInfos.back().get();
  writes.push_back(write);
  return *this;
}

DescriptorWriter &DescriptorWriter::texture(const std::string &name, VkImageView view,
                                            VkImageLayout imageLayout) {
  return textureArray(name, {view}, imageLayout);
}

DescriptorWriter &DescriptorWriter::texture(const std::string &name, const Image &image) {
  return textureArray(name, {image.view}, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

DescriptorWriter &DescriptorWriter::storageTexture(const std::string &name, VkImageView view) {
  return textureArray(name, {view}, VK_IMAGE_LAYOUT_GENERAL);
}

DescriptorWriter &DescriptorWriter::textureArray(const std::string &name,
                                                 const std::vector<VkImageView> &views,
                                                 VkImageLayout imageLayout) {
  const m2v::host::Handle &declared = shader.texture(name);
  if (views.size() != declared.count)
    throw Error("texture " + name + " in " + shader.entry + " takes " +
                std::to_string(declared.count) + " descriptors, " + std::to_string(views.size()) +
                " were bound");
  auto infos = std::make_unique<std::vector<VkDescriptorImageInfo>>();
  for (VkImageView view : views) {
    if (view == VK_NULL_HANDLE) throw Error("a null image view was bound to " + name);
    infos->push_back({VK_NULL_HANDLE, view, imageLayout});
  }

  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = set;
  write.dstBinding = declared.binding;
  write.descriptorCount = static_cast<std::uint32_t>(infos->size());
  write.descriptorType = declared.descriptorType();
  write.pImageInfo = infos->data();
  imageInfos.push_back(std::move(infos));
  writes.push_back(write);
  return *this;
}

DescriptorWriter &DescriptorWriter::sampler(const std::string &name, VkSampler handle) {
  const m2v::host::Handle &declared = shader.sampler(name);
  auto infos = std::make_unique<std::vector<VkDescriptorImageInfo>>();
  infos->push_back({handle, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});

  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = set;
  write.dstBinding = declared.binding;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
  write.pImageInfo = infos->data();
  imageInfos.push_back(std::move(infos));
  writes.push_back(write);
  return *this;
}

DescriptorWriter &DescriptorWriter::accelerationStructure(const std::string &name,
                                                          VkAccelerationStructureKHR structure) {
  const m2v::host::AccelerationStructure &declared = shader.accelerationStructure(name);
  if (structure == VK_NULL_HANDLE) throw Error("a null acceleration structure was bound to " + name);
  auto write = std::make_unique<StructureWrite>();
  write->structure = structure;
  write->info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
  write->info.accelerationStructureCount = 1;
  write->info.pAccelerationStructures = &write->structure;

  VkWriteDescriptorSet descriptor{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  descriptor.pNext = &write->info;
  descriptor.dstSet = set;
  descriptor.dstBinding = declared.binding;
  descriptor.descriptorCount = 1;
  descriptor.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  structureInfos.push_back(std::move(write));
  writes.push_back(descriptor);
  return *this;
}

void DescriptorWriter::apply() {
  if (writes.empty()) return;
  vkUpdateDescriptorSets(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0,
                         nullptr);
  writes.clear();
  imageInfos.clear();
  bufferInfos.clear();
  structureInfos.clear();
}

} // namespace basalt
