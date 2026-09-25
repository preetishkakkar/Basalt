#include "gpu/Shader.h"

#include "core/Log.h"
#include "gpu/SlangReflection.h"

#include <windows.h>

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

Shader::Shader(const Context &ctx, const std::string &name, VkShaderStageFlagBits stage)
    : stageFlag(stage), entry(name), context(&ctx) {
  const std::string base = shaderDirectory() + "/" + name;
  const std::vector<std::uint32_t> words = readWords(base + ".spv");

  VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  moduleInfo.codeSize = words.size() * sizeof(std::uint32_t);
  moduleInfo.pCode = words.data();
  check(vkCreateShaderModule(ctx.device, &moduleInfo, nullptr, &module), "vkCreateShaderModule " + name);
  ctx.nameObject(module, VK_OBJECT_TYPE_SHADER_MODULE, name);

  tracesRays = spirvTracesRays(words);
  reflection = readSlangReflection(Json::parseFile(base + ".json"), words, stage, name);
}

Shader &Shader::operator=(Shader &&other) noexcept {
  if (this == &other) return *this;
  if (module && context) vkDestroyShaderModule(context->device, module, nullptr);
  module = std::exchange(other.module, VK_NULL_HANDLE);
  stageFlag = other.stageFlag;
  entry = std::move(other.entry);
  tracesRays = other.tracesRays;
  reflection = std::move(other.reflection);
  context = std::exchange(other.context, nullptr);
  return *this;
}

Shader::~Shader() {
  if (module && context) vkDestroyShaderModule(context->device, module, nullptr);
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
