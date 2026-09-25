// Prints the driver's own statistics (registers, spills, ...) for Basalt's compute and ray
// pipelines: what decides a kernel's occupancy, which the SPIR-V does not show.
//
// usage: shader-stats [entry ...]   (default: every path-tracer compute entry and ray pipeline)
// A ray pipeline is named by its ray-generation entry. Set BASALT_VULKAN_DEVICE to pick the
// device. Needs VK_KHR_pipeline_executable_properties.
#include "core/Log.h"
#include "gpu/Context.h"
#include "gpu/Descriptors.h"
#include "gpu/RayPipeline.h"
#include "gpu/Shader.h"
#include "platform/Window.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace basalt;

namespace {

// The path tracers' ray pipelines, as the renderer builds them: raygen, miss, closest hit (none
// for shadow rays), any hit.
struct RayPipelineStages {
  const char *raygen, *miss, *closestHit, *anyHit;
};
constexpr RayPipelineStages kRayPipelines[] = {
    {"path_trace_pipeline", "path_pipeline_miss", "path_pipeline_closest", "path_pipeline_alpha"},
    {"path_wavefront_intersect_pipeline", "path_wave_pipeline_miss", "path_wave_pipeline_closest",
     "path_wave_pipeline_alpha"},
    {"path_wavefront_shadow_pipeline", "path_wave_shadow_pipeline_miss", nullptr, "path_wave_shadow_pipeline_alpha"},
};

const RayPipelineStages *findRayPipeline(const std::string &entry) {
  for (const RayPipelineStages &stages : kRayPipelines)
    if (entry == stages.raygen) return &stages;
  return nullptr;
}

// One line per executable: a compute pipeline has one, a ray pipeline one per stage.
void printStatistics(const Context &context, const std::string &entry, VkPipeline pipeline) {
  auto properties = reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(
      vkGetDeviceProcAddr(context.device, "vkGetPipelineExecutablePropertiesKHR"));
  auto statistics = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(
      vkGetDeviceProcAddr(context.device, "vkGetPipelineExecutableStatisticsKHR"));
  VkPipelineInfoKHR pipelineInfo{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
  pipelineInfo.pipeline = pipeline;
  std::uint32_t executableCount = 0;
  properties(context.device, &pipelineInfo, &executableCount, nullptr);
  std::vector<VkPipelineExecutablePropertiesKHR> executables(executableCount,
                                                             {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
  properties(context.device, &pipelineInfo, &executableCount, executables.data());
  for (std::uint32_t e = 0; e < executableCount; ++e) {
    VkPipelineExecutableInfoKHR executable{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
    executable.pipeline = pipeline;
    executable.executableIndex = e;
    std::uint32_t count = 0;
    statistics(context.device, &executable, &count, nullptr);
    std::vector<VkPipelineExecutableStatisticKHR> values(count, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
    statistics(context.device, &executable, &count, values.data());
    std::string line = entry + (executableCount > 1 ? std::string(" [") + executables[e].name + "]:" : ":");
    for (const VkPipelineExecutableStatisticKHR &value : values) {
      line += std::string("  ") + value.name + "=";
      switch (value.format) {
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: line += value.value.b32 ? "true" : "false"; break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: line += std::to_string(value.value.i64); break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: line += std::to_string(value.value.u64); break;
        default: line += std::to_string(value.value.f64); break;
      }
    }
    std::printf("%s\n", line.c_str());
  }
}

void printRayPipeline(const Context &context, const RayPipelineStages &names) {
  if (!context.rayPipelineSupported) {
    std::printf("%s: skipped (no ray pipelines)\n", names.raygen);
    return;
  }
  std::vector<RayStageDescription> stages{{names.raygen, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
                                          {names.miss, VK_SHADER_STAGE_MISS_BIT_KHR}};
  std::uint32_t closestHit = VK_SHADER_UNUSED_KHR;
  if (names.closestHit) {
    closestHit = static_cast<std::uint32_t>(stages.size());
    stages.push_back({names.closestHit, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR});
  }
  const auto anyHit = static_cast<std::uint32_t>(stages.size());
  stages.push_back({names.anyHit, VK_SHADER_STAGE_ANY_HIT_BIT_KHR});
  const std::vector<RayShaderGroup> groups{
      {VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 0u},
      {VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 1u},
      {VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, closestHit, anyHit, VK_SHADER_UNUSED_KHR}};
  const RayPipeline pipeline(context, stages, groups, ShaderBindingRecords{0u, {1u}, {2u}});
  // Drivers that report no executables for a ray pipeline still report each stage's stack: what the
  // raygen stage keeps live across a trace, and each hit stage's own state.
  if (context.rt.getShaderGroupStackSize) {
    auto stack = [&](std::uint32_t group, VkShaderGroupShaderKHR shader) {
      return context.rt.getShaderGroupStackSize(context.device, pipeline.handle, group, shader);
    };
    std::printf("%s [stack]:  raygen=%llu  miss=%llu  closest hit=%llu  any hit=%llu\n", names.raygen,
                static_cast<unsigned long long>(stack(0u, VK_SHADER_GROUP_SHADER_GENERAL_KHR)),
                static_cast<unsigned long long>(stack(1u, VK_SHADER_GROUP_SHADER_GENERAL_KHR)),
                static_cast<unsigned long long>(names.closestHit ? stack(2u, VK_SHADER_GROUP_SHADER_CLOSEST_HIT_KHR) : 0u),
                static_cast<unsigned long long>(stack(2u, VK_SHADER_GROUP_SHADER_ANY_HIT_KHR)));
  }
  printStatistics(context, names.raygen, pipeline.handle);
}

void printCompute(const Context &context, const std::string &entry) {
  std::unique_ptr<Program> program;
  try {
    program = std::make_unique<Program>(context, entry);
  } catch (const std::exception &error) {
    std::printf("%s: skipped (%s)\n", entry.c_str(), error.what());
    return;
  }
  VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  info.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
  info.stage = program->compute().stageInfo();
  info.layout = program->layout;
  VkPipeline pipeline = VK_NULL_HANDLE;
  if (vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) != VK_SUCCESS)
    throw std::runtime_error("vkCreateComputePipelines " + entry);
  printStatistics(context, entry, pipeline);
  vkDestroyPipeline(context.device, pipeline, nullptr);
}

} // namespace

int main(int argc, char **argv) {
  _putenv_s("BASALT_PIPELINE_STATISTICS", "1");
  std::vector<std::string> entries;
  for (int i = 1; i < argc; ++i) entries.emplace_back(argv[i]);
  if (entries.empty()) {
    entries = {"path_trace", "path_trace_emissive", "path_trace_wide", "path_trace_wide_emissive",
               "path_trace_rt", "path_trace_hybrid_rt", "path_wavefront_shade_atomic", "path_wavefront_fused",
               "path_wavefront_fused_wide", "path_wavefront_fused_rt", "path_wavefront_intersect",
               "path_wavefront_shadow"};
    for (const RayPipelineStages &stages : kRayPipelines) entries.emplace_back(stages.raygen);
  }
  try {
    Window window("shader statistics", 64, 64, false);
    Context context(window, false);
    if (!context.pipelineStatisticsEnabled) throw std::runtime_error("no VK_KHR_pipeline_executable_properties");
    std::printf("device: %s\n", context.info.name.c_str());
    for (const std::string &entry : entries) {
      if (const RayPipelineStages *stages = findRayPipeline(entry))
        printRayPipeline(context, *stages);
      else
        printCompute(context, entry);
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "shader-stats: %s\n", error.what());
    return 1;
  }
  return 0;
}
