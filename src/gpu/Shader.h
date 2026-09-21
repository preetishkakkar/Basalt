// Modules plus the compiler's reflection, so binding numbers are never hard-coded.
#pragma once
#include "gpu/Context.h"

#include <m2v_host.h>

#include <string>
#include <vector>

namespace basalt {

struct VertexInput {
  std::uint32_t location = 0;
  std::string name;
  std::string type; // f32x3, f32x4, u32, ...
  VkFormat format() const;
  std::uint32_t byteSize() const;
};

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
  m2v::host::Stage reflection;
  std::vector<VertexInput> vertexInputs;

  // Throws for an unknown name: binding the wrong resource silently draws garbage.
  const m2v::host::Buffer &buffer(const std::string &name) const;
  const m2v::host::Handle &texture(const std::string &name) const;
  const m2v::host::Handle &sampler(const std::string &name) const;
  bool hasTexture(const std::string &name) const;
  const m2v::host::AccelerationStructure &accelerationStructure(const std::string &name) const;

  VkPipelineShaderStageCreateInfo stageInfo(const VkSpecializationInfo *specialization = nullptr) const;

private:
  const Context *context = nullptr;
};

const std::string &shaderDirectory();

} // namespace basalt
