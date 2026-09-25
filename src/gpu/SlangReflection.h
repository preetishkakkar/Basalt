// The host's view of a Slang entry point, read from slangc's -reflection-json: which descriptor
// sets it uses and what sits in them, its push constants, vertex inputs and device requirements.
// Slang assigns every binding; the host never numbers one itself.
//
// Sets follow Slang's parameter model: the loose globals of a file (resources and ConstantBuffers
// declared at global scope) share the default set, and every ParameterBlock<T> gets a set of its
// own, named after the block. A block's ordinary data sits in an implicit uniform buffer at the
// block's first binding. Uniform parameters of an entry point, and a global push-constant
// buffer's, are its push constants.
#pragma once
#include "core/Json.h"

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace basalt {

struct VertexInput {
  std::uint32_t location = 0;
  std::string name;
  std::string type; // f32x3, f32x4, u32, ...
  VkFormat format() const;
  std::uint32_t byteSize() const;
};

struct ShaderBinding {
  // The loose global's name, or the field's path inside its ParameterBlock ("maps", "maps.table").
  // Empty for a block's implicit uniform buffer.
  std::string name;
  std::uint32_t binding = 0;
  std::uint32_t count = 1;
  VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
  // The stages whose module declares it; none when the file declares it but no entry here reads it.
  VkShaderStageFlags stages = 0;
  // Uniform buffers: the size of what the shader reads from it.
  std::uint32_t uniformBytes = 0;
};

struct ShaderSet {
  std::uint32_t index = 0;
  // The ParameterBlock held here; empty for the default set of loose globals.
  std::string block;
  std::vector<ShaderBinding> bindings;
  const ShaderBinding *find(std::string_view name) const;
};

struct ShaderReflection {
  // Every set the file declares, sorted by index. Slang lays out all of a file's parameters for
  // each of its entries but emits only what an entry reads, so the module's own DescriptorSet and
  // Binding decorations say which bindings this entry uses (the JSON's "used" flags are not
  // reliable for ParameterBlocks).
  std::vector<ShaderSet> sets;
  std::uint32_t pushConstantBytes = 0;
  std::array<std::uint32_t, 3> threadGroupSize{1, 1, 1};
  std::vector<VertexInput> vertexInputs; // vertex entries, by location
  std::vector<std::string> requiredFeatures;    // VkPhysicalDevice*Features member names
  VkSubgroupFeatureFlags subgroupOperations = 0; // the subgroup operation classes it uses
};

ShaderReflection readSlangReflection(const Json &document, const std::vector<std::uint32_t> &spirv,
                                     VkShaderStageFlagBits stage, const std::string &entry);

// Folds one entry's sets into a program's: a binding two stages share must agree on name and
// type, and ends up visible to both. what names the program in errors.
void mergeSets(std::vector<ShaderSet> &into, const std::vector<ShaderSet> &from, const std::string &what);

// The OpCapability operands of a SPIR-V module.
std::vector<std::uint32_t> spirvCapabilities(const std::vector<std::uint32_t> &words);
// Whether a SPIR-V module traces rays (OpTraceRayKHR).
bool spirvTracesRays(const std::vector<std::uint32_t> &words);
// The (set, binding) pairs a SPIR-V module decorates its resource variables with.
std::set<std::pair<std::uint32_t, std::uint32_t>> spirvBindings(const std::vector<std::uint32_t> &words);

} // namespace basalt
