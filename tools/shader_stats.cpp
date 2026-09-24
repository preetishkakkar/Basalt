// Prints the driver's own statistics (registers, spills, ...) for Basalt's compute pipelines:
// what decides a kernel's occupancy, which neither the SPIR-V nor msl2spirv's budget shows.
//
// usage: shader-stats [entry ...]   (default: every path-tracer compute entry)
// Set BASALT_VULKAN_DEVICE to pick the device. Needs VK_KHR_pipeline_executable_properties.
#include "core/Log.h"
#include "gpu/Context.h"
#include "gpu/Descriptors.h"
#include "gpu/Shader.h"
#include "platform/Window.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace basalt;

int main(int argc, char **argv) {
  _putenv_s("BASALT_PIPELINE_STATISTICS", "1");
  std::vector<std::string> entries;
  for (int i = 1; i < argc; ++i) entries.emplace_back(argv[i]);
  if (entries.empty())
    entries = {"path_trace", "path_trace_emissive", "path_trace_wide", "path_trace_wide_emissive",
               "path_trace_rt", "path_trace_hybrid_rt", "path_wavefront_shade_atomic", "path_wavefront_fused",
               "path_wavefront_fused_wide", "path_wavefront_fused_rt", "path_wavefront_intersect",
               "path_wavefront_shadow"};
  try {
    Window window("shader statistics", 64, 64, false);
    Context context(window, false);
    if (!context.pipelineStatisticsEnabled) throw std::runtime_error("no VK_KHR_pipeline_executable_properties");
    auto properties = reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(
        vkGetDeviceProcAddr(context.device, "vkGetPipelineExecutablePropertiesKHR"));
    auto statistics = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(
        vkGetDeviceProcAddr(context.device, "vkGetPipelineExecutableStatisticsKHR"));
    std::printf("device: %s\n", context.info.name.c_str());
    for (const std::string &entry : entries) {
      std::unique_ptr<Program> program;
      try {
        program = std::make_unique<Program>(context, entry);
      } catch (const std::exception &error) {
        std::printf("%s: skipped (%s)\n", entry.c_str(), error.what());
        continue;
      }
      VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
      info.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
      info.stage = program->compute().stageInfo();
      info.layout = program->layout;
      VkPipeline pipeline = VK_NULL_HANDLE;
      if (vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error("vkCreateComputePipelines " + entry);
      VkPipelineInfoKHR pipelineInfo{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
      pipelineInfo.pipeline = pipeline;
      std::uint32_t executableCount = 0;
      properties(context.device, &pipelineInfo, &executableCount, nullptr);
      std::string line = entry + ":";
      for (std::uint32_t e = 0; e < executableCount; ++e) {
        VkPipelineExecutableInfoKHR executable{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
        executable.pipeline = pipeline;
        executable.executableIndex = e;
        std::uint32_t count = 0;
        statistics(context.device, &executable, &count, nullptr);
        std::vector<VkPipelineExecutableStatisticKHR> values(count, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
        statistics(context.device, &executable, &count, values.data());
        for (const VkPipelineExecutableStatisticKHR &value : values) {
          line += std::string("  ") + value.name + "=";
          switch (value.format) {
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: line += value.value.b32 ? "true" : "false"; break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: line += std::to_string(value.value.i64); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: line += std::to_string(value.value.u64); break;
            default: line += std::to_string(value.value.f64); break;
          }
        }
      }
      std::printf("%s\n", line.c_str());
      vkDestroyPipeline(context.device, pipeline, nullptr);
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "shader-stats: %s\n", error.what());
    return 1;
  }
  return 0;
}
