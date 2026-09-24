// Descriptor layouts from shader reflection; resources are bound by the shader's names.
#pragma once
#include "gpu/Resources.h"
#include "gpu/Shader.h"

#include <map>
#include <memory>
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

// Set 0 is the vertex or compute stage, set 1 the fragment: the compiler's binding contract.
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

  VkPipelineLayout layout = VK_NULL_HANDLE;
  // An empty set is still created, so the layout stays two sets wide.
  VkDescriptorSetLayout setLayouts[2]{VK_NULL_HANDLE, VK_NULL_HANDLE};

  void requireDeviceSupport() const;

private:
  void build();
  const Context &context;
  std::unique_ptr<Shader> vertexShader, fragmentShader, computeShader;
  std::vector<VkSampler> immutableSamplers;
};

// Writes by the shader's own names; set and binding come from the reflection.
class DescriptorWriter {
public:
  DescriptorWriter(const Context &context, const Shader &shader, VkDescriptorSet set);

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
  const Context &context;
  const Shader &shader;
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
