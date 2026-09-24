#include "gpu/Shader.h"

#include "core/Log.h"

#include <windows.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <utility>

namespace basalt {
namespace {

std::vector<std::uint32_t> readWords(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw Error("cannot open " + path);
  const auto bytes = static_cast<std::size_t>(file.tellg());
  if (bytes == 0 || bytes % 4 != 0) throw Error("not a SPIR-V module: " + path);
  std::vector<std::uint32_t> words(bytes / 4);
  file.seekg(0);
  file.read(reinterpret_cast<char *>(words.data()), static_cast<std::streamsize>(bytes));
  if (words[0] != 0x07230203u) throw Error("wrong SPIR-V magic in " + path);
  return words;
}

const char *stageName(VkShaderStageFlagBits stage) {
  switch (stage) {
  case VK_SHADER_STAGE_VERTEX_BIT: return "vertex";
  case VK_SHADER_STAGE_FRAGMENT_BIT: return "fragment";
  case VK_SHADER_STAGE_COMPUTE_BIT: return "compute";
  case VK_SHADER_STAGE_RAYGEN_BIT_KHR: return "ray_generation";
  case VK_SHADER_STAGE_MISS_BIT_KHR: return "miss";
  case VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR: return "closest_hit";
  case VK_SHADER_STAGE_ANY_HIT_BIT_KHR: return "any_hit";
  case VK_SHADER_STAGE_INTERSECTION_BIT_KHR: return "intersection";
  case VK_SHADER_STAGE_CALLABLE_BIT_KHR: return "callable";
  default: throw Error("unsupported shader stage");
  }
}

} // namespace

const std::string &shaderDirectory() {
  static const std::string directory = [] {
    // Diagnostics (tools/shader_stats.cpp) read another build's shaders.
    if (const char *override = std::getenv("BASALT_SHADER_DIRECTORY")) return std::string(override);
    std::filesystem::path built = std::filesystem::path(BASALT_SHADER_DIR);
    if (std::filesystem::exists(built)) return built.string();
    // A packaged build has no build tree: its shaders sit beside the executable, whatever the
    // working directory it was started from.
    wchar_t executable[MAX_PATH] = {};
    const DWORD length = GetModuleFileNameW(nullptr, executable, MAX_PATH);
    if (length > 0 && length < MAX_PATH) {
      const std::filesystem::path packaged = std::filesystem::path(executable).parent_path() / "shaders";
      if (std::filesystem::exists(packaged)) return packaged.string();
    }
    return std::string("shaders");
  }();
  return directory;
}

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

Shader::Shader(const Context &ctx, const std::string &name, VkShaderStageFlagBits stage)
    : stageFlag(stage), entry(name), context(&ctx) {
  const std::string base = shaderDirectory() + "/" + name;
  const std::vector<std::uint32_t> words = readWords(base + ".spv");

  VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  moduleInfo.codeSize = words.size() * sizeof(std::uint32_t);
  moduleInfo.pCode = words.data();
  check(vkCreateShaderModule(ctx.device, &moduleInfo, nullptr, &module), "vkCreateShaderModule " + name);
  ctx.nameObject(module, VK_OBJECT_TYPE_SHADER_MODULE, name);

  reflection = m2v::host::readStage(base + ".json", stageName(stage), name);

  // vertex_inputs is not in the host library's Stage; read it from the same JSON.
  if (stage == VK_SHADER_STAGE_VERTEX_BIT) {
    const m2v::host::Json document = m2v::host::Json::parseFile(base + ".json");
    if (const m2v::host::Json *inputs = document.find("vertex_inputs")) {
      for (const m2v::host::Json &input : inputs->array) {
        VertexInput attribute;
        attribute.location = static_cast<std::uint32_t>(input.integer("location", 31));
        attribute.name = input.text("name");
        attribute.type = input.text("type");
        vertexInputs.push_back(std::move(attribute));
      }
      std::sort(vertexInputs.begin(), vertexInputs.end(),
                [](const VertexInput &a, const VertexInput &b) { return a.location < b.location; });
    }
  }
}

Shader &Shader::operator=(Shader &&other) noexcept {
  if (this == &other) return *this;
  if (module && context) vkDestroyShaderModule(context->device, module, nullptr);
  module = std::exchange(other.module, VK_NULL_HANDLE);
  stageFlag = other.stageFlag;
  entry = std::move(other.entry);
  reflection = std::move(other.reflection);
  vertexInputs = std::move(other.vertexInputs);
  context = std::exchange(other.context, nullptr);
  return *this;
}

Shader::~Shader() {
  if (module && context) vkDestroyShaderModule(context->device, module, nullptr);
}

const m2v::host::Buffer &Shader::buffer(const std::string &name) const {
  for (const m2v::host::Buffer &candidate : reflection.buffers)
    if (candidate.name == name) return candidate;
  throw Error("shader " + entry + " declares no buffer named " + name);
}

const m2v::host::Handle &Shader::texture(const std::string &name) const {
  for (const m2v::host::Handle &candidate : reflection.textures)
    if (candidate.name == name) return candidate;
  throw Error("shader " + entry + " declares no texture named " + name);
}

bool Shader::hasTexture(const std::string &name) const {
  for (const m2v::host::Handle &candidate : reflection.textures)
    if (candidate.name == name) return true;
  return false;
}

const m2v::host::AccelerationStructure &Shader::accelerationStructure(const std::string &name) const {
  for (const m2v::host::AccelerationStructure &candidate : reflection.accelerationStructures)
    if (candidate.name == name) return candidate;
  throw Error("shader " + entry + " declares no acceleration structure named " + name);
}

const m2v::host::Handle &Shader::sampler(const std::string &name) const {
  for (const m2v::host::Handle &candidate : reflection.samplers)
    if (candidate.name == name) return candidate;
  throw Error("shader " + entry + " declares no sampler named " + name);
}

VkPipelineShaderStageCreateInfo Shader::stageInfo(const VkSpecializationInfo *specialization) const {
  VkPipelineShaderStageCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  info.stage = stageFlag;
  info.module = module;
  info.pName = entry.c_str();
  info.pSpecializationInfo = specialization;
  return info;
}

} // namespace basalt
