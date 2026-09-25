#include "render/RayTracing.h"

#include "core/Log.h"

#include <algorithm>

namespace basalt {
namespace {

VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

} // namespace

SceneAccelerationStructure::SceneAccelerationStructure(const Context &ctx, Uploader &uploader,
                                                       const Scene &scene, const TraceScene &trace)
    : context(ctx) {
  if (!ctx.accelerationStructureSupported)
    throw Error("acceleration structures need an acceleration-structure capable device");
  try {
    build(uploader, scene, trace);
  } catch (...) {
    release();
    throw;
  }
}

void SceneAccelerationStructure::release() {
  context.waitIdle();
  if (topLevel) context.rt.destroy(context.device, topLevel, nullptr);
  topLevel = VK_NULL_HANDLE;
  for (VkAccelerationStructureKHR structure : bottomLevels)
    context.rt.destroy(context.device, structure, nullptr);
  bottomLevels.clear();
}

void SceneAccelerationStructure::build(Uploader &uploader, const Scene &scene, const TraceScene &trace) {
  const Context &ctx = context;
  if (trace.instances.empty()) throw Error("no triangles to build an acceleration structure over");

  const VkDeviceAddress vertexAddress = scene.vertexBuffer.deviceAddress();
  const VkDeviceAddress indexAddress = scene.indexBuffer.deviceAddress();
  const VkDeviceSize scratchAlignment = std::max<VkDeviceSize>(ctx.scratchAlignment, 16);

  // One BLAS per instance. Masked and blended ones are non-opaque, so queries run the candidate
  // test; single-sided back faces are culled by the path tracers' rays (kPtRayFlags), below.
  struct Build {
    VkAccelerationStructureGeometryKHR geometry{};
    VkAccelerationStructureBuildRangeInfoKHR range{};
    VkAccelerationStructureBuildGeometryInfoKHR info{};
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    std::uint32_t primitive = 0;
  };
  std::vector<Build> builds;
  VkDeviceSize scratchBytes = 0;
  for (std::size_t b = 0; b < trace.instances.size(); ++b) {
    const std::uint32_t i = trace.primitives[b];
    const Primitive &primitive = scene.primitives[i];
    const bool opaque = (trace.instances[b].flags & (pt::kInstanceMasked | pt::kInstanceBlended)) == 0;

    Build build;
    build.primitive = i;
    build.geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    build.geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    build.geometry.flags = opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;
    auto &triangles = build.geometry.geometry.triangles;
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = vertexAddress;
    triangles.vertexStride = sizeof(Vertex);
    triangles.maxVertex = scene.vertexCount - 1;
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = indexAddress;

    build.range.primitiveCount = primitive.indexCount / 3;
    build.range.primitiveOffset = primitive.firstIndex * sizeof(std::uint32_t);
    build.range.firstVertex = static_cast<std::uint32_t>(primitive.vertexOffset);

    build.info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build.info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build.info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build.info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build.info.geometryCount = 1;
    build.sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    // pGeometries is set after the vector stops moving.
    builds.push_back(build);
  }

  for (Build &build : builds) {
    build.info.pGeometries = &build.geometry;
    const std::uint32_t count = build.range.primitiveCount;
    ctx.rt.getBuildSizes(ctx.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build.info,
                         &count, &build.sizes);
    scratchBytes = std::max(scratchBytes, alignUp(build.sizes.buildScratchSize, scratchAlignment));

    Buffer storage(ctx, build.sizes.accelerationStructureSize,
                   VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                   VMA_MEMORY_USAGE_AUTO, 0, "blas." + scene.primitives[build.primitive].name);
    VkAccelerationStructureCreateInfoKHR createInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfo.buffer = storage.handle;
    createInfo.size = build.sizes.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    VkAccelerationStructureKHR structure = VK_NULL_HANDLE;
    check(ctx.rt.create(ctx.device, &createInfo, nullptr, &structure), "vkCreateAccelerationStructureKHR");
    build.info.dstAccelerationStructure = structure;
    bottomLevels.push_back(structure);
    bottomBuffers.push_back(std::move(storage));
  }

  // Slack for aligning the address up.
  Buffer scratch(ctx, scratchBytes + scratchAlignment,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                 VMA_MEMORY_USAGE_AUTO, 0, "as.scratch");
  const VkDeviceAddress scratchAddress = alignUp(scratch.deviceAddress(), scratchAlignment);

  uploader.runImmediate([&](VkCommandBuffer command) {
    // One scratch buffer serves each build in turn.
    for (Build &build : builds) {
      build.info.scratchData.deviceAddress = scratchAddress;
      const VkAccelerationStructureBuildRangeInfoKHR *range = &build.range;
      ctx.rt.build(command, 1, &build.info, &range);
      VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
      barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                              VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
      barrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                              VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
      VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      dependency.memoryBarrierCount = 1;
      dependency.pMemoryBarriers = &barrier;
      vkCmdPipelineBarrier2(command, &dependency);
    }
  });

  std::vector<VkAccelerationStructureInstanceKHR> instances;
  for (std::size_t b = 0; b < builds.size(); ++b) {
    const Primitive &primitive = scene.primitives[builds[b].primitive];
    VkAccelerationStructureInstanceKHR instance{};
    // Row-major three by four: the rows of the model matrix.
    for (int row = 0; row < 3; ++row) {
      const Vec4 r = primitive.transform.row(row);
      instance.transform.matrix[row][0] = r.x;
      instance.transform.matrix[row][1] = r.y;
      instance.transform.matrix[row][2] = r.z;
      instance.transform.matrix[row][3] = r.w;
    }
    instance.instanceCustomIndex = builds[b].primitive;
    instance.mask = trace.instances[b].mask;
    // A path tracer's ray skips a single-sided instance's back faces, as ptCandidateSolid would. A
    // blended instance keeps both: its back faces still mark the hit ambiguous. Rays that set no
    // cull flag (the rasteriser's) see every face either way.
    if ((trace.instances[b].flags & (pt::kInstanceDoubleSided | pt::kInstanceBlended)) != 0)
      instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    addressInfo.accelerationStructure = bottomLevels[b];
    instance.accelerationStructureReference = ctx.rt.address(ctx.device, &addressInfo);
    instances.push_back(instance);
  }
  instanceCount = static_cast<std::uint32_t>(instances.size());

  instanceBuffer = uploader.createBuffer(
      instances.data(), instances.size() * sizeof(VkAccelerationStructureInstanceKHR),
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      "tlas.instances");

  VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  geometry.geometry.instances.data.deviceAddress = instanceBuffer.deviceAddress();

  VkAccelerationStructureBuildGeometryInfoKHR topInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  topInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  topInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  topInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  topInfo.geometryCount = 1;
  topInfo.pGeometries = &geometry;
  VkAccelerationStructureBuildSizesInfoKHR topSizes{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  ctx.rt.getBuildSizes(ctx.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &topInfo,
                       &instanceCount, &topSizes);

  topBuffer = Buffer(ctx, topSizes.accelerationStructureSize,
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                     VMA_MEMORY_USAGE_AUTO, 0, "tlas");
  VkAccelerationStructureCreateInfoKHR topCreate{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  topCreate.buffer = topBuffer.handle;
  topCreate.size = topSizes.accelerationStructureSize;
  topCreate.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  check(ctx.rt.create(ctx.device, &topCreate, nullptr, &topLevel), "vkCreateAccelerationStructureKHR (top)");
  topInfo.dstAccelerationStructure = topLevel;

  Buffer topScratch(ctx, alignUp(topSizes.buildScratchSize, scratchAlignment) + scratchAlignment,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO, 0, "tlas.scratch");
  topInfo.scratchData.deviceAddress = alignUp(topScratch.deviceAddress(), scratchAlignment);
  VkAccelerationStructureBuildRangeInfoKHR topRange{};
  topRange.primitiveCount = instanceCount;
  uploader.runImmediate([&](VkCommandBuffer command) {
    const VkAccelerationStructureBuildRangeInfoKHR *range = &topRange;
    ctx.rt.build(command, 1, &topInfo, &range);
  });

  logInfo("acceleration structures: {} bottom-level, {} instances", bottomLevels.size(), instanceCount);
}

SceneAccelerationStructure::~SceneAccelerationStructure() { release(); }

} // namespace basalt
