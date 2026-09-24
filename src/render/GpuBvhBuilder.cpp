#include "render/GpuBvhBuilder.h"

#include "core/Log.h"
#include "gpu/Descriptors.h"
#include "gpu/Pipeline.h"
#include "gpu/Shader.h"
#include "gpu/Uploader.h"
#include "pt/Shared.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

#define device
#define thread
namespace pt {
#include "pt/bvh_build.h"
}
#undef device
#undef thread

namespace basalt {
namespace {

static_assert(sizeof(pt::BvhBuildDescriptor) == 32);
static_assert(sizeof(pt::BvhBuildControl) == 32);
static_assert(sizeof(pt::BvhBuildRecord) == 48);
static_assert(sizeof(pt::BvhBuildStatus) == 32);

const char *errorName(std::uint32_t error) {
  switch (error) {
  case pt::kBvhBuildErrorCapacity: return "buffer capacity or layout contract";
  case pt::kBvhBuildErrorGeometry: return "geometry index";
  case pt::kBvhBuildErrorNonFinite: return "non-finite vertex";
  case pt::kBvhBuildErrorDepth: return "builder or traversal depth";
  default: return "unknown";
  }
}

std::uint32_t nodeCount(std::uint32_t leaves) { return leaves > 1 ? leaves - 1 : 1; }

} // namespace

GpuBvhBuildResult buildGpuBvh(const Context &context, Uploader &uploader,
                              const Scene &scene, const TraceScene &trace) {
  const auto started = std::chrono::steady_clock::now();
  GpuBvhBuildResult result;
  result.instances = trace.instances;
  const std::uint32_t instanceCount = static_cast<std::uint32_t>(result.instances.size());
  std::vector<pt::BvhBuildDescriptor> descriptors;
  descriptors.reserve(instanceCount);
  std::uint32_t nodeOffset = nodeCount(instanceCount);
  const std::uint32_t topNodes = nodeOffset;
  std::uint32_t triangleOffset = 0;
  std::uint32_t maximumTriangles = instanceCount;
  for (std::uint32_t i = 0; i < instanceCount; ++i) {
    const std::uint32_t primitiveIndex = trace.primitives[i];
    if (primitiveIndex >= scene.primitives.size())
      throw std::runtime_error("GPU BVH build received an invalid primitive mapping");
    const Primitive &primitive = scene.primitives[primitiveIndex];
    const std::uint32_t triangles = primitive.indexCount / 3u;
    pt::BvhBuildDescriptor descriptor{};
    descriptor.geometry = {result.instances[i].firstIndex, result.instances[i].vertexOffset,
                           triangles, i};
    descriptor.output = {nodeOffset, triangleOffset, 0u, 0u};
    descriptors.push_back(descriptor);
    result.instances[i].blasRoot = nodeOffset;
    result.instances[i].triangleOffset = triangleOffset;
    nodeOffset += nodeCount(triangles);
    triangleOffset += triangles;
    maximumTriangles = std::max(maximumTriangles, triangles);
  }

  const std::uint32_t scratchRecords = std::max(1u, maximumTriangles);
  const std::uint32_t nodeCapacity = std::max(1u, nodeOffset);
  const std::uint32_t triangleCapacity = std::max(1u, triangleOffset);
  result.nodes = Buffer(context, static_cast<VkDeviceSize>(nodeCapacity) * 4u * sizeof(pt::float4),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VMA_MEMORY_USAGE_AUTO, 0, "path.gpu-bvh.nodes");
  result.triangles = Buffer(context, static_cast<VkDeviceSize>(triangleCapacity) * 3u * sizeof(pt::float4),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VMA_MEMORY_USAGE_AUTO, 0, "path.gpu-bvh.triangles");
  Buffer scratchA(context, static_cast<VkDeviceSize>(scratchRecords) * sizeof(pt::BvhBuildRecord),
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, 0, "path.gpu-bvh.scratch-a");
  Buffer scratchB(context, static_cast<VkDeviceSize>(scratchRecords) * sizeof(pt::BvhBuildRecord),
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, 0, "path.gpu-bvh.scratch-b");
  Buffer instanceRecords(context, static_cast<VkDeviceSize>(std::max(1u, instanceCount)) * sizeof(pt::BvhBuildRecord),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, 0,
                         "path.gpu-bvh.instance-records");

  const pt::BvhBuildDescriptor emptyDescriptor{};
  Buffer descriptorBuffer = uploader.createBuffer(
      descriptors.empty() ? static_cast<const void *>(&emptyDescriptor) : descriptors.data(),
      std::max<std::size_t>(1, descriptors.size()) * sizeof(pt::BvhBuildDescriptor),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "path.gpu-bvh.descriptors");
  const pt::TraceInstance emptyInstance{};
  Buffer instanceBuffer = uploader.createBuffer(
      result.instances.empty() ? static_cast<const void *>(&emptyInstance) : result.instances.data(),
      std::max<std::size_t>(1, result.instances.size()) * sizeof(pt::TraceInstance),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "path.gpu-bvh.instances");
  const pt::BvhBuildStatus emptyStatus{};
  Buffer statusBuffer = uploader.createBuffer(&emptyStatus, sizeof(emptyStatus),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, "path.gpu-bvh.status");
  pt::BvhBuildControl control{};
  control.sizes = {instanceCount, triangleOffset, scratchRecords, nodeCapacity};
  control.geometry = {scene.indexCount, scene.vertexCount, PT_BVH_STACK, pt::kBvhBuildLayoutVersion};
  Buffer controlBuffer = uploader.createBuffer(&control, sizeof(control),
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, "path.gpu-bvh.control");

  Program program(context, "bvh_build");
  Pipeline pipeline(context, program, "GPU LBVH build");
  DescriptorPool pool(context, 1);
  const VkDescriptorSet set = pool.allocate(program.setLayouts[0]);
  DescriptorWriter(context, program.compute(), set)
      .buffer("descriptors", descriptorBuffer)
      .buffer("instances", instanceBuffer)
      .buffer("indices", scene.indexBuffer)
      .buffer("vertices", scene.vertexBuffer)
      .buffer("nodes", result.nodes)
      .buffer("triangles", result.triangles)
      .buffer("scratchA", scratchA)
      .buffer("scratchB", scratchB)
      .buffer("status", statusBuffer)
      .buffer("instanceRecords", instanceRecords)
      .buffer("control", controlBuffer)
      .apply();
  uploader.runImmediate([&](VkCommandBuffer command) {
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, program.layout, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(command, 1, 1, 1);
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(command, &dependency);
  });
  const std::vector<std::uint8_t> statusBytes = uploader.readBuffer(statusBuffer, sizeof(pt::BvhBuildStatus));
  pt::BvhBuildStatus status{};
  std::memcpy(&status, statusBytes.data(), sizeof(status));
  if (status.result.x != pt::kBvhBuildErrorNone)
    throw std::runtime_error(std::string("GPU LBVH build failed: ") + errorName(status.result.x) +
                             " (code " + std::to_string(status.result.x) + ")");
  if (status.counts.x != nodeCapacity || status.counts.y != triangleOffset ||
      status.counts.z != instanceCount)
    throw std::runtime_error("GPU LBVH build published incomplete output counts");

  result.statistics.instances = instanceCount;
  result.statistics.triangles = triangleOffset;
  result.statistics.topNodes = topNodes;
  result.statistics.bottomNodes = nodeCapacity - topNodes;
  result.statistics.topDepth = status.result.y;
  result.statistics.bottomDepth = status.result.z;
  result.statistics.milliseconds =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  result.milliseconds = result.statistics.milliseconds;
  result.scratchBytes = static_cast<VkDeviceSize>(scratchRecords) * sizeof(pt::BvhBuildRecord) * 2u +
                        static_cast<VkDeviceSize>(std::max(1u, instanceCount)) * sizeof(pt::BvhBuildRecord);
  result.outputBytes = result.nodes.size + result.triangles.size;
  result.radixPasses = status.counts.w;
  result.maximumBuilderStack = status.result.w;
  return result;
}

} // namespace basalt
