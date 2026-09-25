// A shader module and slangc's reflection of it, so binding numbers are never hard-coded.
#pragma once
#include "gpu/Context.h"
#include "gpu/SlangReflection.h"

#include <string>
#include <vector>

namespace basalt {

class Shader {
public:
  // name is the entry point and the file stem: <shader dir>/<name>.spv and .json.
  Shader(const Context &context, const std::string &name, VkShaderStageFlagBits stage);
  ~Shader();
  Shader(Shader &&other) noexcept { *this = std::move(other); }
  Shader &operator=(Shader &&other) noexcept;
  Shader(const Shader &) = delete;
  Shader &operator=(const Shader &) = delete;

  VkShaderModule module = VK_NULL_HANDLE;
  VkShaderStageFlagBits stageFlag = VK_SHADER_STAGE_VERTEX_BIT;
  std::string entry;
  // A ray stage that traces further rays.
  bool tracesRays = false;
  ShaderReflection reflection;

  VkPipelineShaderStageCreateInfo stageInfo(const VkSpecializationInfo *specialization = nullptr) const;

private:
  const Context *context = nullptr;
};

const std::string &shaderDirectory();

} // namespace basalt
