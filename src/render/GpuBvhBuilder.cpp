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

namespace basalt {
namespace {

static_assert(sizeof(pt::BvhBuildDescriptor) == 32);
static_assert(sizeof(pt::BvhBuildControl) == 48);
static_assert(sizeof(pt::BvhBuildRecord) == 48);
static_assert(sizeof(pt::BvhBuildStatus) == 32);
static_assert(sizeof(pt::BvhBuildStatus2) == 64);

// The serial build's parts: its one thread takes about 5 us per record on an RTX 4090, so a part
// of kSerialPartRecords takes about 0.35 s, and a bottom level of kSerialMeshRecords (the largest
// it takes, alone in its part) about 1.4 s, under the driver's ~2 s timeout (which resets the
// device).
constexpr std::uint32_t kSerialPartRecords = 65536, kSerialMeshRecords = 262144;

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

// Orders compute writes before later compute reads and host readback copies.
void computeBarrier(VkCommandBuffer command) {
  VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
  barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.memoryBarrierCount = 1;
  dependency.pMemoryBarriers = &barrier;
  vkCmdPipelineBarrier2(command, &dependency);
}

} // namespace

GpuBvhReferences uploadReferences(Uploader &uploader, const pt::BvhReferences &references) {
  const auto started = std::chrono::steady_clock::now();
  GpuBvhReferences result;
  const pt::BvhReference empty{};
  result.references = uploader.createBuffer(
      references.references.empty() ? static_cast<const void *>(&empty) : references.references.data(),
      std::max<std::size_t>(1, references.references.size()) * sizeof(pt::BvhReference), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT, "path.gpu-bvh.references");
  result.counts = references.counts;
  result.milliseconds = references.milliseconds +
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  return result;
}

GpuBvhBuildResult buildGpuBvh(const Context &context, Uploader &uploader,
                              const Scene &scene, const TraceScene &trace, const GpuBvhReferences *references) {
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
  if (references && references->counts.size() != instanceCount)
    throw std::runtime_error("GPU BVH build received references for other instances");
  for (std::uint32_t i = 0; i < instanceCount; ++i) {
    const std::uint32_t primitiveIndex = trace.primitives[i];
    if (primitiveIndex >= scene.primitives.size())
      throw std::runtime_error("GPU BVH build received an invalid primitive mapping");
    const Primitive &primitive = scene.primitives[primitiveIndex];
    // Its records: its triangles, or its references (the table's from its triangle base).
    const std::uint32_t triangles = references ? references->counts[i] : primitive.indexCount / 3u;
    pt::BvhBuildDescriptor descriptor{};
    descriptor.geometry = {result.instances[i].firstIndex, result.instances[i].vertexOffset,
                           triangles, i};
    descriptor.output = {nodeOffset, triangleOffset, 0u, references ? 1u : 0u};
    descriptors.push_back(descriptor);
    result.instances[i].blasRoot = nodeOffset;
    result.instances[i].triangleOffset = triangleOffset;
    nodeOffset += nodeCount(triangles);
    triangleOffset += triangles;
    maximumTriangles = std::max(maximumTriangles, triangles);
  }

  for (std::uint32_t i = 0; i < instanceCount; ++i)
    if (descriptors[i].geometry.z > kSerialMeshRecords)
      throw std::runtime_error("the serial GPU builder builds bottom levels of up to " + std::to_string(kSerialMeshRecords) +
                               " triangles within the driver's timeout (instance " + std::to_string(i) + " has " +
                               std::to_string(descriptors[i].geometry.z) + "); use a parallel builder");
  // The parts: runs of bottom levels up to kSerialPartRecords records (a larger one alone), then
  // the TLAS.
  std::vector<pt::uint4> parts;
  for (std::uint32_t first = 0; first < instanceCount;) {
    std::uint32_t last = first, records = 0;
    while (last < instanceCount && (last == first || records + descriptors[last].geometry.z <= kSerialPartRecords))
      records += descriptors[last++].geometry.z;
    parts.push_back({first, last, 0u, 0u});
    first = last;
  }
  parts.push_back({instanceCount, instanceCount, 1u, 0u});

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
  const pt::BvhReference emptyReference{};
  Buffer placeholder;  // bound when there are no references
  if (!references)
    placeholder = uploader.createBuffer(&emptyReference, sizeof(emptyReference), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                        "path.gpu-bvh.no-references");
  const Buffer &referenceBuffer = references ? references->references : placeholder;
  const pt::BvhBuildStatus emptyStatus{};
  Buffer statusBuffer = uploader.createBuffer(&emptyStatus, sizeof(emptyStatus),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, "path.gpu-bvh.status");
  pt::BvhBuildControl control{};
  control.sizes = {instanceCount, triangleOffset, scratchRecords, nodeCapacity};
  control.geometry = {scene.indexCount, scene.vertexCount, pt::kBvhStackDeep, pt::kBvhBuildLayoutVersion};

  Program program(context, "bvh_build");
  Pipeline pipeline(context, program, "GPU LBVH build");
  DescriptorPool pool(context, static_cast<std::uint32_t>(parts.size()));
  // Each part its own submission (the status carries over).
  for (const pt::uint4 &part : parts) {
    control.part = part;
    Buffer controlBuffer = uploader.createBuffer(&control, sizeof(control), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                 "path.gpu-bvh.control");
    const VkDescriptorSet set = program.allocate(pool);
    DescriptorWriter(context, program, set)
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
        .buffer("references", referenceBuffer)
        .buffer("control", controlBuffer)
        .apply();
    uploader.runImmediate([&](VkCommandBuffer command) {
      vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
      program.bind(command, set);
      vkCmdDispatch(command, 1, 1, 1);
      computeBarrier(command);
    });
  }
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
  result.status.error = status.result.x;
  result.status.topDepth = status.result.y;
  result.status.bottomDepth = status.result.z;
  result.status.maximumBuilderStack = status.result.w;
  result.status.nodes = status.counts.x;
  result.status.triangles = status.counts.y;
  result.status.instances = status.counts.z;
  result.status.sortPasses = status.counts.w;
  result.status.dispatches = static_cast<std::uint32_t>(parts.size());

  const pt::LayoutCost cost = gpuBvhSahCost(context, uploader, result.nodes, nodeCapacity, result.instances, false);
  result.status.topCost = cost.top;
  result.status.bottomCost = cost.bottom;
  result.statistics.sahCost = cost.bottom;
  return result;
}

pt::LayoutCost gpuBvhSahCost(const Context &context, Uploader &uploader, const Buffer &nodes,
                             std::uint32_t nodeCount, const std::vector<pt::TraceInstance> &instances, bool wide) {
  pt::LayoutCost cost;
  if (nodeCount == 0) return cost;
  // Trees are contiguous node ranges starting at their roots: the TLAS at 0, then the
  // distinct bottom-level roots in ascending order.
  std::vector<std::uint32_t> roots;
  roots.reserve(instances.size());
  for (const pt::TraceInstance &instance : instances) roots.push_back(instance.blasRoot);
  std::sort(roots.begin(), roots.end());
  roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
  if (!roots.empty() && (roots.front() == 0u || roots.back() >= nodeCount))
    throw std::runtime_error("BVH cost: a bottom-level root is outside the published node range");
  std::vector<pt::uint4> trees;
  trees.reserve(roots.size() + 1u);
  trees.push_back({0u, roots.empty() ? nodeCount : roots.front(), 0u, 0u});
  for (std::size_t i = 0; i < roots.size(); ++i) {
    const std::uint32_t end = i + 1u < roots.size() ? roots[i + 1u] : nodeCount;
    trees.push_back({roots[i], end - roots[i], 0u, 0u});
  }

  Buffer treeBuffer = uploader.createBuffer(trees.data(), trees.size() * sizeof(pt::uint4),
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "bvh-cost.trees");
  Buffer contributions(context, static_cast<VkDeviceSize>(nodeCount) * 2u * sizeof(float),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VMA_MEMORY_USAGE_AUTO, 0, "bvh-cost.contributions");
  const pt::uint4 control{nodeCount, static_cast<std::uint32_t>(trees.size()), 0u, 0u};
  Buffer controlBuffer = uploader.createBuffer(&control, sizeof(control), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                               "bvh-cost.control");
  Program program(context, wide ? "bvh_wide_cost" : "bvh_cost");
  Pipeline pipeline(context, program, wide ? "BVH wide SAH cost" : "BVH SAH cost");
  DescriptorPool pool(context, 1);
  const VkDescriptorSet set = program.allocate(pool);
  DescriptorWriter(context, program, set)
      .buffer(wide ? "wideNodes" : "nodes", nodes)
      .buffer("trees", treeBuffer)
      .buffer("contributions", contributions)
      .buffer("control", controlBuffer)
      .apply();
  uploader.runImmediate([&](VkCommandBuffer command) {
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
    program.bind(command, set);
    vkCmdDispatch(command, (nodeCount + 63u) / 64u, 1, 1);
    computeBarrier(command);
  });
  const std::vector<std::uint8_t> bytes = uploader.readBuffer(contributions, contributions.size);
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), bytes.size());

  // Per tree, in node order: the root term, then every node's children.
  std::vector<double> treeCost(trees.size(), 0.0);
  for (std::size_t t = 0; t < trees.size(); ++t) {
    double sum = values[static_cast<std::size_t>(trees[t].x) * 2u + 1u];
    for (std::uint32_t node = trees[t].x; node < trees[t].x + trees[t].y; ++node)
      sum += values[static_cast<std::size_t>(node) * 2u];
    treeCost[t] = sum;
  }
  cost.top = treeCost[0];
  for (const pt::TraceInstance &instance : instances) {
    const auto found = std::lower_bound(roots.begin(), roots.end(), instance.blasRoot);
    cost.bottom += treeCost[1u + static_cast<std::size_t>(found - roots.begin())];
  }
  return cost;
}

} // namespace basalt
