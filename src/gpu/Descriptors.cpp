#include "gpu/Descriptors.h"

#include "core/Log.h"

#include <algorithm>
#include <array>
#include <format>

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
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, setsPerPool * 16},
      // Sized for the 120-slot texture table used by both GPU path tracers.
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, setsPerPool * 144},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, setsPerPool * 2},
      {VK_DESCRIPTOR_TYPE_SAMPLER, setsPerPool * 4},
  };
  if (context.accelerationStructureSupported)
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

Program::~Program() = default;

std::string Program::describe() const {
  if (computeShader) return computeShader->entry;
  return vertexShader->entry + "/" + fragmentShader->entry;
}

std::vector<const Shader *> Program::stages() const {
  if (computeShader) return {computeShader.get()};
  return {vertexShader.get(), fragmentShader.get()};
}

void Program::build() {
  for (const Shader *stage : stages()) requireDeviceSupport(context, *stage);
  sets = std::make_unique<ShaderLayout>(context, stages(), describe(),
                                        computeShader ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS);
  layout = sets->layout;
}

ShaderLayout::ShaderLayout(const Context &ctx, const std::vector<const Shader *> &stages, std::string layoutName,
                           VkPipelineBindPoint point)
    : name(std::move(layoutName)), context(ctx), bindPoint(point) {
  for (const Shader *stage : stages) {
    const ShaderReflection &reflection = stage->reflection;
    mergeSets(sets, reflection.sets, name);
    if (reflection.pushConstantBytes > 0) {
      pushConstantBytes = std::max(pushConstantBytes, reflection.pushConstantBytes);
      pushConstantStages |= stage->stageFlag;
    }
  }

  // Every index up to the highest gets a layout; one Slang leaves empty stays empty.
  const std::uint32_t count = sets.empty() ? 0u : sets.back().index + 1;
  setLayouts.assign(count, VK_NULL_HANDLE);
  for (std::uint32_t index = 0; index < count; ++index) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (const ShaderSet &set : sets) {
      if (set.index != index) continue;
      // What no stage reads stays out of the layout.
      for (const ShaderBinding &binding : set.bindings)
        if (binding.stages != 0)
          bindings.push_back({binding.binding, binding.type, binding.count, binding.stages, nullptr});
    }
    VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.bindingCount = static_cast<std::uint32_t>(bindings.size());
    info.pBindings = bindings.data();
    check(vkCreateDescriptorSetLayout(context.device, &info, nullptr, &setLayouts[index]),
          "vkCreateDescriptorSetLayout " + name);
  }

  const VkPushConstantRange range{pushConstantStages, 0, pushConstantBytes};
  VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layoutInfo.setLayoutCount = count;
  layoutInfo.pSetLayouts = setLayouts.data();
  layoutInfo.pushConstantRangeCount = pushConstantBytes > 0 ? 1u : 0u;
  layoutInfo.pPushConstantRanges = &range;
  check(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &layout), "vkCreatePipelineLayout " + name);
}

ShaderLayout::~ShaderLayout() {
  for (VkDescriptorSetLayout setLayout : setLayouts)
    if (setLayout) vkDestroyDescriptorSetLayout(context.device, setLayout, nullptr);
  if (layout) vkDestroyPipelineLayout(context.device, layout, nullptr);
}

const ShaderSet &ShaderLayout::set(std::string_view block) const {
  for (const ShaderSet &candidate : sets)
    if (candidate.block == block) return candidate;
  throw Error(name + " has no " + (block.empty() ? std::string("default set") : "ParameterBlock named " + std::string(block)));
}

bool ShaderLayout::declares(std::string_view block) const {
  return std::any_of(sets.begin(), sets.end(), [&](const ShaderSet &candidate) { return candidate.block == block; });
}

bool ShaderLayout::uses(std::string_view block) const {
  const ShaderSet &target = set(block);
  return std::any_of(target.bindings.begin(), target.bindings.end(),
                     [](const ShaderBinding &binding) { return binding.stages != 0; });
}

VkDescriptorSet ShaderLayout::allocate(DescriptorPool &pool, std::string_view block) const {
  return pool.allocate(setLayouts.at(set(block).index));
}

void ShaderLayout::bind(VkCommandBuffer command, VkDescriptorSet descriptorSet, std::string_view block) const {
  vkCmdBindDescriptorSets(command, bindPoint, layout, set(block).index, 1, &descriptorSet, 0, nullptr);
}

void ShaderLayout::push(VkCommandBuffer command, const void *data, std::uint32_t bytes) const {
  if (bytes != pushConstantBytes)
    throw Error(name + " takes " + std::to_string(pushConstantBytes) + " bytes of push constants, " +
                std::to_string(bytes) + " were given");
  vkCmdPushConstants(command, layout, pushConstantStages, 0, bytes, data);
}

namespace {

// The subgroup operation classes a stage uses that the device lacks in that stage.
VkSubgroupFeatureFlags missingSubgroupOperations(const Context &context, const Shader &shader) {
  const VkSubgroupFeatureFlags needed = shader.reflection.subgroupOperations;
  if (needed == 0) return 0;
  VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  properties.pNext = &subgroup;
  vkGetPhysicalDeviceProperties2(context.physical, &properties);
  if ((subgroup.supportedStages & shader.stageFlag) == 0) return needed;
  return needed & ~subgroup.supportedOperations;
}

} // namespace

bool deviceSupports(const Context &context, const Shader &shader) {
  for (const std::string &feature : shader.reflection.requiredFeatures)
    if (!context.enabledFeatures.contains(feature)) return false;
  return missingSubgroupOperations(context, shader) == 0;
}

void requireDeviceSupport(const Context &context, const Shader &shader) {
  for (const std::string &feature : shader.reflection.requiredFeatures) {
    // Checked against what the device was created with, so a missing feature fails by name.
    if (!context.enabledFeatures.contains(feature))
      throw Error("shader " + shader.entry + " requires the device feature " + feature +
                  ", which this device does not offer or this engine does not enable");
  }
  if (const VkSubgroupFeatureFlags missing = missingSubgroupOperations(context, shader))
    throw Error("shader " + shader.entry + " uses subgroup operations (VkSubgroupFeatureFlags 0x" +
                std::format("{:x}", missing) + ") this device does not offer in its stage");
}

namespace {

const char *kindName(VkDescriptorType type) {
  switch (type) {
  case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: return "a uniform buffer";
  case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: return "a storage buffer";
  case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: return "a sampled texture";
  case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: return "a storage texture";
  case VK_DESCRIPTOR_TYPE_SAMPLER: return "a sampler";
  case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: return "an acceleration structure";
  default: return "an unsupported descriptor";
  }
}

std::string displayName(const std::string &name) { return name.empty() ? "(block uniforms)" : name; }

} // namespace

DescriptorWriter::DescriptorWriter(const Context &ctx, const ShaderLayout &shaderLayout, VkDescriptorSet descriptorSet,
                                   std::string_view block)
    : context(ctx), layout(shaderLayout), target(shaderLayout.set(block)), set(descriptorSet) {}

std::string DescriptorWriter::owner() const {
  return layout.name + (target.block.empty() ? "" : " block " + target.block);
}

std::optional<DescriptorWriter::Slot> DescriptorWriter::resolve(const std::string &name,
                                                                std::initializer_list<VkDescriptorType> kinds) const {
  const auto accepts = [&](VkDescriptorType type) { return std::find(kinds.begin(), kinds.end(), type) != kinds.end(); };
  const ShaderBinding *binding = target.find(name);
  if (!binding) throw Error(owner() + " declares nothing named " + displayName(name));
  const Slot slot{binding->binding, binding->count, binding->type, binding->uniformBytes};
  if (!accepts(slot.type))
    throw Error(owner() + " declares " + displayName(name) + " as " + kindName(slot.type) + ", not " +
                kindName(*kinds.begin()));
  if (binding->stages == 0) return std::nullopt; // declared, read by none of the stages
  return slot;
}

DescriptorWriter &DescriptorWriter::bufferAt(const std::string &name, const Slot &slot, const Buffer &resource,
                                             VkDeviceSize offset, VkDeviceSize range) {
  if (!resource) throw Error("no buffer bound for " + displayName(name) + " in " + owner());
  if (slot.uniformBytes > 0) {
    const VkDeviceSize bytes = range == VK_WHOLE_SIZE ? resource.size - offset : range;
    if (bytes < slot.uniformBytes)
      throw Error(owner() + " reads " + std::to_string(slot.uniformBytes) + " bytes from " + displayName(name) +
                  ", the buffer bound has " + std::to_string(bytes));
  }
  bufferInfos.push_back(std::make_unique<VkDescriptorBufferInfo>(
      VkDescriptorBufferInfo{resource.handle, offset, range}));

  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = set;
  write.dstBinding = slot.binding;
  write.descriptorCount = 1;
  write.descriptorType = slot.type;
  write.pBufferInfo = bufferInfos.back().get();
  writes.push_back(write);
  return *this;
}

DescriptorWriter &DescriptorWriter::buffer(const std::string &name, const Buffer &resource,
                                           VkDeviceSize offset, VkDeviceSize range) {
  const std::optional<Slot> slot =
      resolve(name, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER});
  if (!slot) return *this;
  return bufferAt(name, *slot, resource, offset, range);
}

DescriptorWriter &DescriptorWriter::uniforms(const Buffer &resource, VkDeviceSize offset, VkDeviceSize range) {
  if (target.block.empty()) throw Error(owner() + ": only a ParameterBlock has block uniforms");
  const std::optional<Slot> slot = resolve("", {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER});
  if (!slot) return *this;
  return bufferAt("", *slot, resource, offset, range);
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
  const std::optional<Slot> slot =
      resolve(name, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE});
  if (!slot) return *this;
  if (views.size() != slot->count)
    throw Error("texture " + name + " in " + owner() + " takes " + std::to_string(slot->count) +
                " descriptors, " + std::to_string(views.size()) + " were bound");
  auto infos = std::make_unique<std::vector<VkDescriptorImageInfo>>();
  for (VkImageView view : views) {
    if (view == VK_NULL_HANDLE) throw Error("a null image view was bound to " + name);
    infos->push_back({VK_NULL_HANDLE, view, imageLayout});
  }

  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = set;
  write.dstBinding = slot->binding;
  write.descriptorCount = static_cast<std::uint32_t>(infos->size());
  write.descriptorType = slot->type;
  write.pImageInfo = infos->data();
  imageInfos.push_back(std::move(infos));
  writes.push_back(write);
  return *this;
}

DescriptorWriter &DescriptorWriter::sampler(const std::string &name, VkSampler handle) {
  const std::optional<Slot> slot = resolve(name, {VK_DESCRIPTOR_TYPE_SAMPLER});
  if (!slot) return *this;
  auto infos = std::make_unique<std::vector<VkDescriptorImageInfo>>();
  infos->push_back({handle, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});

  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = set;
  write.dstBinding = slot->binding;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
  write.pImageInfo = infos->data();
  imageInfos.push_back(std::move(infos));
  writes.push_back(write);
  return *this;
}

DescriptorWriter &DescriptorWriter::accelerationStructure(const std::string &name,
                                                          VkAccelerationStructureKHR structure) {
  const std::optional<Slot> slot = resolve(name, {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR});
  if (!slot) return *this;
  if (structure == VK_NULL_HANDLE) throw Error("a null acceleration structure was bound to " + name);
  auto write = std::make_unique<StructureWrite>();
  write->structure = structure;
  write->info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
  write->info.accelerationStructureCount = 1;
  write->info.pAccelerationStructures = &write->structure;

  VkWriteDescriptorSet descriptor{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  descriptor.pNext = &write->info;
  descriptor.dstSet = set;
  descriptor.dstBinding = slot->binding;
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
