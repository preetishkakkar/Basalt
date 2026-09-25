// Descriptor layouts from shader reflection; resources are bound by the shader's names.
#pragma once
#include "gpu/Resources.h"
#include "gpu/Shader.h"

#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string_view>
#include <string>
#include <vector>

namespace basalt {

class DescriptorPool {
public:
  explicit DescriptorPool(const Context &context, std::uint32_t setsPerPool = 256);
  ~DescriptorPool();
  DescriptorPool(const DescriptorPool &) = delete;
  DescriptorPool &operator=(const DescriptorPool &) = delete;

  VkDescriptorSet allocate(VkDescriptorSetLayout layout);
  void reset();

private:
  VkDescriptorPool createPool() const;
  const Context &context;
  std::uint32_t setsPerPool;
  std::vector<VkDescriptorPool> pools;
  std::size_t current = 0;
};

// The descriptor sets and pipeline layout of Slang stages that run together (a program's, a ray
// pipeline's). The sets are Slang's (see gpu/SlangReflection.h) with every stage's bindings
// merged: a file's loose globals share the default set, and each ParameterBlock is a set of its
// own, found by the block's name.
class ShaderLayout {
public:
  ShaderLayout(const Context &context, const std::vector<const Shader *> &stages, std::string name,
               VkPipelineBindPoint bindPoint);
  ~ShaderLayout();
  ShaderLayout(const ShaderLayout &) = delete;
  ShaderLayout &operator=(const ShaderLayout &) = delete;

  const std::string name;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  // By set index; an index Slang leaves empty has an empty layout.
  std::vector<VkDescriptorSetLayout> setLayouts;
  // The merged sets, by index, and the push constants' size and stages.
  std::vector<ShaderSet> sets;
  std::uint32_t pushConstantBytes = 0;
  VkShaderStageFlags pushConstantStages = 0;

  // The set holding a ParameterBlock, or the default set for an empty name; throws when absent.
  const ShaderSet &set(std::string_view block = {}) const;
  // Whether the stages' files declare the block at all.
  bool declares(std::string_view block) const;
  // Whether any stage reads something in the block's set: a traced variant's scene block is
  // declared by its raster twin too, and there reads nothing.
  bool uses(std::string_view block) const;
  VkDescriptorSet allocate(DescriptorPool &pool, std::string_view block = {}) const;
  void bind(VkCommandBuffer command, VkDescriptorSet descriptorSet, std::string_view block = {}) const;
  // Writes the entry points' uniform parameters; the size must be what the shaders declare.
  void push(VkCommandBuffer command, const void *data, std::uint32_t bytes) const;

private:
  const Context &context;
  VkPipelineBindPoint bindPoint;
};

// Whether the device was created with every feature a stage needs and offers its subgroup
// operations in the stage's own stage; requireDeviceSupport throws, naming what is missing.
bool deviceSupports(const Context &context, const Shader &shader);
void requireDeviceSupport(const Context &context, const Shader &shader);

// A graphics or compute pipeline's stages and their ShaderLayout.
class Program {
public:
  Program(const Context &context, std::string vertexEntry, std::string fragmentEntry);
  Program(const Context &context, std::string computeEntry);
  ~Program();
  Program(const Program &) = delete;
  Program &operator=(const Program &) = delete;

  const Shader &vertex() const { return *vertexShader; }
  const Shader &fragment() const { return *fragmentShader; }
  const Shader &compute() const { return *computeShader; }
  bool isCompute() const { return computeShader != nullptr; }
  const ShaderLayout &shaderLayout() const { return *sets; }

  VkPipelineLayout layout = VK_NULL_HANDLE;

  const ShaderSet &set(std::string_view block = {}) const { return sets->set(block); }
  bool uses(std::string_view block) const { return sets->uses(block); }
  VkDescriptorSet allocate(DescriptorPool &pool, std::string_view block = {}) const { return sets->allocate(pool, block); }
  void bind(VkCommandBuffer command, VkDescriptorSet descriptorSet, std::string_view block = {}) const {
    sets->bind(command, descriptorSet, block);
  }
  template <class T> void push(VkCommandBuffer command, const T &value) const {
    sets->push(command, &value, static_cast<std::uint32_t>(sizeof(T)));
  }

  std::string describe() const;

private:
  std::vector<const Shader *> stages() const;
  void build();
  const Context &context;
  std::unique_ptr<Shader> vertexShader, fragmentShader, computeShader;
  std::unique_ptr<ShaderLayout> sets;
};

// Writes a set by the shaders' own names; set and binding come from the reflection.
class DescriptorWriter {
public:
  // A ParameterBlock's set by the block's name, else the default set. Names are the loose
  // globals', or the block's fields' ("maps", "maps.table"). Writing one the file declares but
  // none of the stages reads is a no-op, so variants share host code.
  DescriptorWriter(const Context &context, const ShaderLayout &layout, VkDescriptorSet set,
                   std::string_view block = {});
  DescriptorWriter(const Context &context, const Program &program, VkDescriptorSet set, std::string_view block = {})
      : DescriptorWriter(context, program.shaderLayout(), set, block) {}

  // A ParameterBlock's own data: the implicit uniform buffer Slang gives a block with any.
  DescriptorWriter &uniforms(const Buffer &buffer, VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE);

  DescriptorWriter &buffer(const std::string &name, const Buffer &buffer, VkDeviceSize offset = 0,
                           VkDeviceSize range = VK_WHOLE_SIZE);
  DescriptorWriter &texture(const std::string &name, VkImageView view, VkImageLayout layout);
  DescriptorWriter &texture(const std::string &name, const Image &image);
  DescriptorWriter &storageTexture(const std::string &name, VkImageView view);
  DescriptorWriter &textureArray(const std::string &name, const std::vector<VkImageView> &views,
                                 VkImageLayout layout);
  DescriptorWriter &sampler(const std::string &name, VkSampler sampler);
  DescriptorWriter &accelerationStructure(const std::string &name, VkAccelerationStructureKHR structure);
  void apply();

private:
  struct Slot {
    std::uint32_t binding = 0;
    std::uint32_t count = 1;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
    std::uint32_t uniformBytes = 0;
  };
  // The named binding; nullopt when the program declares it but never reads it. Throws for a name
  // that is not declared, or one of another kind than `kinds`.
  std::optional<Slot> resolve(const std::string &name, std::initializer_list<VkDescriptorType> kinds) const;
  std::string owner() const;
  DescriptorWriter &bufferAt(const std::string &name, const Slot &slot, const Buffer &buffer, VkDeviceSize offset,
                             VkDeviceSize range);

  const Context &context;
  const ShaderLayout &layout;
  const ShaderSet &target;
  VkDescriptorSet set;
  // Stable storage: the descriptor infos must outlive the write array.
  std::vector<std::unique_ptr<std::vector<VkDescriptorImageInfo>>> imageInfos;
  std::vector<std::unique_ptr<VkDescriptorBufferInfo>> bufferInfos;
  struct StructureWrite {
    VkWriteDescriptorSetAccelerationStructureKHR info{};
    VkAccelerationStructureKHR structure = VK_NULL_HANDLE;
  };
  std::vector<std::unique_ptr<StructureWrite>> structureInfos;
  std::vector<VkWriteDescriptorSet> writes;
};

} // namespace basalt
