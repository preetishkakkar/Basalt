#include "gpu/SlangReflection.h"

#include "core/Log.h"

#include <spirv-headers/spirv.hpp>

#include <algorithm>
#include <map>

namespace basalt {
namespace {

std::string scalarName(const std::string &scalarType) {
  if (scalarType == "float32") return "f32";
  if (scalarType == "uint32") return "u32";
  if (scalarType == "int32") return "i32";
  if (scalarType == "float16") return "f16";
  throw Error("unsupported reflected scalar type " + scalarType);
}

// "f32", "f32x3", "u32x4" ... from a scalar or vector type.
std::string valueTypeName(const Json &type) {
  const std::string &kind = type.text("kind");
  if (kind == "scalar") return scalarName(type.text("scalarType"));
  if (kind == "vector") {
    const std::string element = scalarName(type.at("elementType").text("scalarType"));
    return element + "x" + std::to_string(type.integer("elementCount", 4));
  }
  throw Error("unsupported reflected value type " + kind);
}

// A variable layout's binding of the given kind: its one "binding", or one of its "bindings".
const Json *bindingOfKind(const Json &variable, std::string_view kind) {
  if (const Json *single = variable.find("binding"))
    return single->text("kind") == kind ? single : nullptr;
  for (const Json &binding : variable.list("bindings"))
    if (binding.text("kind") == kind) return &binding;
  return nullptr;
}

std::uint32_t uniformBytes(const Json &variableLayout) {
  const Json *uniform = bindingOfKind(variableLayout, "uniform");
  return uniform ? static_cast<std::uint32_t>(uniform->integer("size", UINT32_MAX)) : 0u;
}

std::uint32_t slotIndex(const Json &binding) {
  return static_cast<std::uint32_t>(binding.integer("index", UINT32_MAX));
}

std::uint32_t space(const Json &binding) {
  return binding.find("space") ? static_cast<std::uint32_t>(binding.integer("space", UINT32_MAX)) : 0u;
}

void addUnique(std::vector<std::string> &list, const std::string &value) {
  if (std::find(list.begin(), list.end(), value) == list.end()) list.push_back(value);
}

class Reader {
public:
  Reader(ShaderReflection &out, VkShaderStageFlagBits stage, const std::string &entry)
      : result(out), stage(stage), entry(entry) {}

  ShaderSet &set(std::uint32_t index, const std::string &block) {
    for (ShaderSet &existing : result.sets) {
      if (existing.index != index) continue;
      if (existing.block != block)
        throw Error("shader " + entry + " puts both '" + existing.block + "' and '" + block + "' in set " +
                    std::to_string(index));
      return existing;
    }
    ShaderSet &added = result.sets.emplace_back();
    added.index = index;
    added.block = block;
    return added;
  }

  // The descriptors of a variable of this type at binding `base`: one for a resource or an array
  // of them, one per resource field for a struct.
  void collect(ShaderSet &target, const std::string &path, const Json &type, std::uint32_t base) {
    const Json *element = &type;
    std::uint32_t count = 1;
    if (type.text("kind") == "array") {
      count = static_cast<std::uint32_t>(type.integer("elementCount", UINT32_MAX));
      if (count == 0) throw Error("shader " + entry + ": " + path + " is an unsized array, which the host cannot bind");
      element = &type.at("elementType");
    }
    const std::string &kind = element->text("kind");
    if (kind == "struct") {
      if (count != 1) throw Error("shader " + entry + ": " + path + " is an array of structs holding resources");
      for (const Json &field : element->list("fields"))
        if (const Json *slot = bindingOfKind(field, "descriptorTableSlot"))
          collect(target, path + "." + field.text("name"), field.at("type"), base + slotIndex(*slot));
      return;
    }

    ShaderBinding binding;
    binding.name = path;
    binding.binding = base;
    binding.count = count;
    binding.stages = stage;
    if (kind == "constantBuffer") {
      binding.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      binding.uniformBytes = uniformBytes(element->at("elementVarLayout"));
    } else if (kind == "samplerState") {
      binding.type = VK_DESCRIPTOR_TYPE_SAMPLER;
    } else if (kind == "resource") {
      const std::string &shape = element->text("baseShape");
      const bool writable = element->find("access") && element->text("access") != "read";
      if (shape == "structuredBuffer" || shape == "byteAddressBuffer") {
        binding.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      } else if (shape == "accelerationStructure") {
        binding.type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
      } else if (shape == "texture2D" || shape == "textureCube" || shape == "texture3D") {
        binding.type = writable ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        if (count > 1) dynamicArrays = true;
      } else {
        throw Error("shader " + entry + ": " + path + " has the unsupported resource shape " + shape);
      }
    } else {
      throw Error("shader " + entry + ": " + path + " has the unsupported kind " + kind);
    }
    target.bindings.push_back(std::move(binding));
  }

  void globals(const Json &document) {
    for (const Json &parameter : document.list("parameters")) {
      const std::string &name = parameter.text("name");
      const Json &type = parameter.at("type");
      if (const Json *block = bindingOfKind(parameter, "subElementRegisterSpace")) {
        // A ParameterBlock: a set of its own.
        if (type.text("kind") != "parameterBlock")
          throw Error("shader " + entry + ": " + name + " takes a set but is not a ParameterBlock");
        // The block's set: its register-space index, offset by any enclosing space.
        ShaderSet &target = set(slotIndex(*block) + space(*block), name);
        const std::uint32_t bytes = uniformBytes(type.at("elementVarLayout"));
        if (bytes > 0) {
          const Json *container = bindingOfKind(type.at("containerVarLayout"), "descriptorTableSlot");
          if (!container) throw Error("shader " + entry + ": block " + name + " has data but no uniform buffer");
          ShaderBinding uniforms;
          uniforms.binding = slotIndex(*container);
          uniforms.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
          uniforms.stages = stage;
          uniforms.uniformBytes = bytes;
          target.bindings.push_back(std::move(uniforms));
        }
        const Json &element = type.at("elementType");
        if (element.text("kind") != "struct")
          throw Error("shader " + entry + ": block " + name + " does not hold a struct");
        for (const Json &field : element.list("fields"))
          if (const Json *slot = bindingOfKind(field, "descriptorTableSlot"))
            collect(target, field.text("name"), field.at("type"), slotIndex(*slot));
        continue;
      }
      if (const Json *push = bindingOfKind(parameter, "pushConstantBuffer")) {
        (void)push;
        result.pushConstantBytes = std::max(result.pushConstantBytes, uniformBytes(type.at("elementVarLayout")));
        continue;
      }
      if (bindingOfKind(parameter, "uniform"))
        throw Error("shader " + entry + ": the global " + name +
                    " is ordinary data; put it in a ConstantBuffer or a ParameterBlock");
      const Json *slot = bindingOfKind(parameter, "descriptorTableSlot");
      if (!slot) throw Error("shader " + entry + ": the global " + name + " has no descriptor binding");
      collect(set(space(*slot), ""), name, type, slotIndex(*slot));
    }
  }

  bool dynamicArrays = false;

private:
  ShaderReflection &result;
  VkShaderStageFlagBits stage;
  const std::string &entry;
};

} // namespace

VkFormat VertexInput::format() const {
  if (type == "f32") return VK_FORMAT_R32_SFLOAT;
  if (type == "f32x2") return VK_FORMAT_R32G32_SFLOAT;
  if (type == "f32x3") return VK_FORMAT_R32G32B32_SFLOAT;
  if (type == "f32x4") return VK_FORMAT_R32G32B32A32_SFLOAT;
  if (type == "u32") return VK_FORMAT_R32_UINT;
  if (type == "u32x2") return VK_FORMAT_R32G32_UINT;
  if (type == "u32x4") return VK_FORMAT_R32G32B32A32_UINT;
  if (type == "i32") return VK_FORMAT_R32_SINT;
  if (type == "i32x4") return VK_FORMAT_R32G32B32A32_SINT;
  throw Error("no vertex format for attribute type " + type);
}

std::uint32_t VertexInput::byteSize() const {
  const std::size_t x = type.find('x');
  const std::uint32_t lanes = x == std::string::npos ? 1u : static_cast<std::uint32_t>(type[x + 1] - '0');
  return 4u * lanes;
}

const ShaderBinding *ShaderSet::find(std::string_view name) const {
  for (const ShaderBinding &binding : bindings)
    if (binding.name == name) return &binding;
  return nullptr;
}

std::vector<std::uint32_t> spirvCapabilities(const std::vector<std::uint32_t> &words) {
  std::vector<std::uint32_t> capabilities;
  for (std::size_t i = 5; i < words.size();) {
    const std::uint32_t count = words[i] >> 16, opcode = words[i] & 0xFFFFu;
    if (count == 0 || i + count > words.size()) throw Error("malformed SPIR-V instruction stream");
    if (opcode == spv::OpCapability) capabilities.push_back(words[i + 1]);
    // Capabilities come first; stop at the memory model.
    if (opcode == spv::OpMemoryModel) break;
    i += count;
  }
  return capabilities;
}

bool spirvTracesRays(const std::vector<std::uint32_t> &words) {
  for (std::size_t i = 5; i < words.size();) {
    const std::uint32_t count = words[i] >> 16, opcode = words[i] & 0xFFFFu;
    if (count == 0 || i + count > words.size()) throw Error("malformed SPIR-V instruction stream");
    if (opcode == spv::OpTraceRayKHR) return true;
    i += count;
  }
  return false;
}

std::set<std::pair<std::uint32_t, std::uint32_t>> spirvBindings(const std::vector<std::uint32_t> &words) {
  std::map<std::uint32_t, std::uint32_t> sets, bindings;
  for (std::size_t i = 5; i < words.size();) {
    const std::uint32_t count = words[i] >> 16, opcode = words[i] & 0xFFFFu;
    if (count == 0 || i + count > words.size()) throw Error("malformed SPIR-V instruction stream");
    // Annotations precede the types; the first function ends them.
    if (opcode == spv::OpFunction) break;
    if (opcode == spv::OpDecorate && count >= 4) {
      if (words[i + 2] == spv::DecorationDescriptorSet) sets[words[i + 1]] = words[i + 3];
      else if (words[i + 2] == spv::DecorationBinding) bindings[words[i + 1]] = words[i + 3];
    }
    i += count;
  }
  std::set<std::pair<std::uint32_t, std::uint32_t>> result;
  for (const auto &[id, binding] : bindings) {
    const auto set = sets.find(id);
    result.insert({set == sets.end() ? 0u : set->second, binding});
  }
  return result;
}

ShaderReflection readSlangReflection(const Json &document, const std::vector<std::uint32_t> &spirv,
                                     VkShaderStageFlagBits stage, const std::string &entry) {
  ShaderReflection result;

  const Json *point = nullptr;
  for (const Json &candidate : document.list("entryPoints"))
    if (candidate.text("name") == entry) point = &candidate;
  if (!point) throw Error("reflection for " + entry + " describes no entry point of that name");

  if (const Json *size = point->find("threadGroupSize"))
    for (std::size_t axis = 0; axis < 3 && axis < size->array.size(); ++axis)
      result.threadGroupSize[axis] = static_cast<std::uint32_t>(size->array[axis].number);

  Reader reader(result, stage, entry);
  reader.globals(document);

  // What this entry reads is what its module declares.
  const std::set<std::pair<std::uint32_t, std::uint32_t>> declared = spirvBindings(spirv);
  for (ShaderSet &set : result.sets)
    for (ShaderBinding &binding : set.bindings)
      if (!declared.contains({set.index, binding.binding})) binding.stages = 0;

  for (const Json &parameter : point->list("parameters")) {
    // The entry's uniform parameters are its push constants.
    if (const Json *uniform = bindingOfKind(parameter, "uniform"))
      result.pushConstantBytes =
          std::max(result.pushConstantBytes, static_cast<std::uint32_t>(uniform->integer("offset", UINT32_MAX) +
                                                                        uniform->integer("size", UINT32_MAX)));
  }

  std::sort(result.sets.begin(), result.sets.end(),
            [](const ShaderSet &a, const ShaderSet &b) { return a.index < b.index; });
  for (ShaderSet &set : result.sets)
    std::sort(set.bindings.begin(), set.bindings.end(),
              [](const ShaderBinding &a, const ShaderBinding &b) { return a.binding < b.binding; });

  // Vertex attributes: the entry's varying inputs, by location.
  if (stage == VK_SHADER_STAGE_VERTEX_BIT) {
    auto addInput = [&](const Json &value) {
      const Json *input = bindingOfKind(value, "varyingInput");
      if (!input) return;
      VertexInput attribute;
      attribute.location = slotIndex(*input);
      attribute.name = value.text("name");
      attribute.type = valueTypeName(value.at("type"));
      result.vertexInputs.push_back(std::move(attribute));
    };
    for (const Json &parameter : point->list("parameters")) {
      const Json &type = parameter.at("type");
      if (type.text("kind") == "struct") {
        for (const Json &field : type.list("fields")) addInput(field);
      } else {
        addInput(parameter);
      }
    }
    std::sort(result.vertexInputs.begin(), result.vertexInputs.end(),
              [](const VertexInput &a, const VertexInput &b) { return a.location < b.location; });
  }

  // Device features and properties the module needs, from its capabilities.
  std::vector<std::string> &features = result.requiredFeatures;
  if (reader.dynamicArrays) addUnique(features, "shaderSampledImageArrayDynamicIndexing");
  for (const std::uint32_t capability : spirvCapabilities(spirv)) {
    switch (static_cast<spv::Capability>(capability)) {
    case spv::CapabilityRayQueryKHR:
      addUnique(features, "rayQuery");
      addUnique(features, "accelerationStructure");
      break;
    case spv::CapabilityRayTracingKHR:
      addUnique(features, "rayTracingPipeline");
      addUnique(features, "accelerationStructure");
      break;
    case spv::CapabilityStorageImageWriteWithoutFormat: addUnique(features, "shaderStorageImageWriteWithoutFormat"); break;
    case spv::CapabilitySampledImageArrayNonUniformIndexing:
      addUnique(features, "shaderSampledImageArrayNonUniformIndexing");
      break;
    case spv::CapabilityDemoteToHelperInvocation: addUnique(features, "shaderDemoteToHelperInvocation"); break;
    case spv::CapabilityDrawParameters: addUnique(features, "shaderDrawParameters"); break;
    case spv::CapabilityImageCubeArray:
    case spv::CapabilitySampledCubeArray: addUnique(features, "imageCubeArray"); break;
    case spv::CapabilityInt64: addUnique(features, "shaderInt64"); break;
    case spv::CapabilityFloat64: addUnique(features, "shaderFloat64"); break;
    // A fragment shader reading its primitive index.
    case spv::CapabilityGeometry: addUnique(features, "geometryShader"); break;
    case spv::CapabilityGroupNonUniform: result.subgroupOperations |= VK_SUBGROUP_FEATURE_BASIC_BIT; break;
    case spv::CapabilityGroupNonUniformVote: result.subgroupOperations |= VK_SUBGROUP_FEATURE_VOTE_BIT; break;
    case spv::CapabilityGroupNonUniformArithmetic: result.subgroupOperations |= VK_SUBGROUP_FEATURE_ARITHMETIC_BIT; break;
    case spv::CapabilityGroupNonUniformBallot: result.subgroupOperations |= VK_SUBGROUP_FEATURE_BALLOT_BIT; break;
    case spv::CapabilityGroupNonUniformShuffle: result.subgroupOperations |= VK_SUBGROUP_FEATURE_SHUFFLE_BIT; break;
    case spv::CapabilityGroupNonUniformShuffleRelative: result.subgroupOperations |= VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT; break;
    case spv::CapabilityGroupNonUniformClustered: result.subgroupOperations |= VK_SUBGROUP_FEATURE_CLUSTERED_BIT; break;
    case spv::CapabilityGroupNonUniformQuad: result.subgroupOperations |= VK_SUBGROUP_FEATURE_QUAD_BIT; break;
    default: break;
    }
  }
  return result;
}

void mergeSets(std::vector<ShaderSet> &into, const std::vector<ShaderSet> &from, const std::string &what) {
  for (const ShaderSet &incoming : from) {
    auto existing = std::find_if(into.begin(), into.end(),
                                 [&](const ShaderSet &set) { return set.index == incoming.index; });
    if (existing == into.end()) {
      into.insert(std::upper_bound(into.begin(), into.end(), incoming,
                                   [](const ShaderSet &a, const ShaderSet &b) { return a.index < b.index; }),
                  incoming);
      continue;
    }
    if (existing->block != incoming.block)
      throw Error(what + ": set " + std::to_string(incoming.index) + " holds '" + existing->block +
                  "' in one stage and '" + incoming.block + "' in another");
    for (const ShaderBinding &binding : incoming.bindings) {
      auto match = std::find_if(existing->bindings.begin(), existing->bindings.end(),
                                [&](const ShaderBinding &b) { return b.binding == binding.binding; });
      if (match == existing->bindings.end()) {
        existing->bindings.insert(
            std::upper_bound(existing->bindings.begin(), existing->bindings.end(), binding,
                             [](const ShaderBinding &a, const ShaderBinding &b) { return a.binding < b.binding; }),
            binding);
        continue;
      }
      if (match->name != binding.name || match->type != binding.type || match->count != binding.count)
        throw Error(what + ": set " + std::to_string(incoming.index) + " binding " + std::to_string(binding.binding) +
                    " is '" + match->name + "' in one stage and '" + binding.name + "' in another");
      match->stages |= binding.stages;
      match->uniformBytes = std::max(match->uniformBytes, binding.uniformBytes);
    }
  }
}

} // namespace basalt
