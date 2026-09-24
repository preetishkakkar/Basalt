#include "render/PathReconstruction.h"
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace basalt {
namespace {
void barrier(VkCommandBuffer command) {
  VkMemoryBarrier2 b{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  b.srcStageMask = b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  b.srcAccessMask = b.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;
  VkDependencyInfo d{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  d.memoryBarrierCount = 1; d.pMemoryBarriers = &b;
  vkCmdPipelineBarrier2(command, &d);
}
void dispatch(VkCommandBuffer command, const Program &p, const Pipeline &pipeline, VkDescriptorSet set,
              std::uint32_t width, std::uint32_t height) {
  vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
  vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0, nullptr);
  vkCmdDispatch(command, (width + 7) / 8, (height + 7) / 8, 1);
}
}

PathReconstruction::PathReconstruction(Context &ctx, std::uint32_t w, std::uint32_t h)
    : context(ctx), width(w), height(h), temporalProgram(ctx, "path_temporal"),
      atrousProgram(ctx, "path_atrous"), compositeProgram(ctx, "path_reconstruct"),
      temporalPipeline(ctx, temporalProgram, "path.temporal"), atrousPipeline(ctx, atrousProgram, "path.atrous"),
      compositePipeline(ctx, compositeProgram, "path.reconstruct"), pool(ctx, 32) {
  const VkDeviceSize pixels = static_cast<VkDeviceSize>(w) * h;
  if (pixels == 0 || pixels * 704 > 2ull * 1024 * 1024 * 1024)
    throw std::runtime_error("path reconstruction exceeds its 2 GiB budget; use a smaller window or raw display");
  constexpr VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  auto buffer = [&](VkDeviceSize bytes, const char *name) { return Buffer(ctx, bytes, usage, VMA_MEMORY_USAGE_AUTO, 0, name); };
  sampleBuffer = buffer(pixels * sizeof(pt::PtReconstructionSample), "path.samples");
  previousSamples = buffer(sampleBuffer.size, "path.previous.samples");
  previousHistory = buffer(pixels * sizeof(pt::PtTemporalHistory), "path.history");
  temporalBuffer = buffer(previousHistory.size, "path.temporal");
  scratchBuffer = buffer(previousHistory.size, "path.scratch");
  filteredBuffer = buffer(previousHistory.size, "path.filtered");
  for (auto &f : frames) {
    for (auto &u : f.uniforms) u = Buffer(ctx, sizeof(pt::PtTemporalUniforms), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT, "path.reconstruction.uniforms");
    f.temporal = pool.allocate(temporalProgram.setLayouts[0]);
    DescriptorWriter(ctx, temporalProgram.compute(), f.temporal).buffer("uniforms", f.uniforms[0])
        .buffer("samples", sampleBuffer).buffer("previousSamples", previousSamples)
        .buffer("previous", previousHistory).buffer("output", temporalBuffer).apply();
    const Buffer *inputs[] = {&temporalBuffer, &scratchBuffer, &filteredBuffer};
    const Buffer *outputs[] = {&scratchBuffer, &filteredBuffer, &scratchBuffer};
    for (std::uint32_t i = 0; i < 3; ++i) {
      f.atrous[i] = pool.allocate(atrousProgram.setLayouts[0]);
      DescriptorWriter(ctx, atrousProgram.compute(), f.atrous[i]).buffer("uniforms", f.uniforms[i + 1])
          .buffer("samples", sampleBuffer).buffer("input", *inputs[i]).buffer("output", *outputs[i]).apply();
    }
    f.composite = pool.allocate(compositeProgram.setLayouts[0]);
  }
  VkQueryPoolCreateInfo q{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  q.queryType = VK_QUERY_TYPE_TIMESTAMP; q.queryCount = kFramesInFlight * 2;
  if (ctx.properties.limits.timestampComputeAndGraphics) {
    if (vkCreateQueryPool(ctx.device, &q, nullptr, &timestamps) != VK_SUCCESS) timestamps = VK_NULL_HANDLE;
    if (vkCreateQueryPool(ctx.device, &q, nullptr, &uploadTimestamps) != VK_SUCCESS) uploadTimestamps = VK_NULL_HANDLE;
  }
}

PathReconstruction::~PathReconstruction() {
  if (timestamps) vkDestroyQueryPool(context.device, timestamps, nullptr);
  if (uploadTimestamps) vkDestroyQueryPool(context.device, uploadTimestamps, nullptr);
}

void PathReconstruction::upload(VkCommandBuffer command, std::uint32_t slot,
                              const std::vector<pt::PtReconstructionSample> &samples) {
  if (samples.size() * sizeof(pt::PtReconstructionSample) != sampleBuffer.size)
    throw std::runtime_error("incoherent CPU reconstruction publication extent");
  auto &f = frames.at(slot);
  if (!f.staging) f.staging = Buffer(context, sampleBuffer.size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
      VMA_ALLOCATION_CREATE_MAPPED_BIT, "path.guide.upload");
  const auto started = std::chrono::steady_clock::now();
  f.staging.write(samples.data(), static_cast<std::size_t>(sampleBuffer.size));
  uploadMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - started).count();
  if (uploadTimestamps) {
    std::uint64_t t[2]{};
    if (uploadTimestampWritten[slot] && vkGetQueryPoolResults(context.device, uploadTimestamps, slot * 2, 2,
        sizeof(t), t, sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
      transferMs = static_cast<float>(static_cast<double>(t[1] - t[0]) * context.properties.limits.timestampPeriod * 1e-6);
    vkCmdResetQueryPool(command, uploadTimestamps, slot * 2, 2);
    vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, uploadTimestamps, slot * 2);
    uploadTimestampWritten[slot] = true;
  }
  barrier(command);
  const VkBufferCopy copy{0, 0, sampleBuffer.size};
  vkCmdCopyBuffer(command, f.staging.handle, sampleBuffer.handle, 1, &copy);
  if (uploadTimestamps)
    vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, uploadTimestamps, slot * 2 + 1);
}

bool PathReconstruction::record(VkCommandBuffer command, std::uint32_t slot, Image &lit,
    const pt::PathUniforms &camera, std::uint64_t key, std::uint32_t rawSamples, bool fresh, float cutDistance) {
  auto &f = frames.at(slot);
  if (key != previousKey) valid = false;
  if (!fresh && !valid) return false;
  pt::PtTemporalUniforms u{};
  u.previousCamera = valid ? previousCamera : camera;
  u.image = pt::uint4(width, height, valid ? 0u : 1u, 1u);
  u.control = pt::float4(static_cast<float>(rawSamples), 0.0f, 0.0f, 0.0f);
  u.outputExtent = pt::uint4(lit.description.width, lit.description.height, 0, 0);
  if (fresh) {
    if (pt::dot(pt::xyz(camera.cameraForward), pt::xyz(previousCamera.cameraForward)) < 0.85f ||
        pt::length(pt::xyz(camera.cameraPosition) - pt::xyz(previousCamera.cameraPosition)) > cutDistance ||
        std::abs(pt::length(pt::xyz(camera.cameraRight)) - pt::length(pt::xyz(previousCamera.cameraRight))) > 1e-6f ||
        std::abs(pt::length(pt::xyz(camera.cameraUp)) - pt::length(pt::xyz(previousCamera.cameraUp))) > 1e-6f ||
        camera.lens.x != previousCamera.lens.x || camera.lens.y != previousCamera.lens.y)  // aperture or focus
      u.image.z = 1u;
    resetLast = u.image.z != 0;
  }
  for (std::uint32_t i = 0; i < 4; ++i) {
    u.image.w = i == 0 ? 1u : 1u << (i - 1);
    f.uniforms[i].write(&u, sizeof(u));
  }
  if (timestamps) {
    const std::uint32_t first = slot * 2;
    std::uint64_t t[2]{};
    if (timestampWritten[slot] && timestampFresh[slot] && vkGetQueryPoolResults(context.device, timestamps, first, 2, sizeof(t), t,
        sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
      filterMs = static_cast<float>(static_cast<double>(t[1] - t[0]) * context.properties.limits.timestampPeriod * 1e-6);
    vkCmdResetQueryPool(command, timestamps, first, 2);
    vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, timestamps, first);
    timestampWritten[slot] = true;
    timestampFresh[slot] = fresh;
  }
  barrier(command);
  if (fresh) {
    dispatch(command, temporalProgram, temporalPipeline, f.temporal, width, height);
    for (std::uint32_t i = 0; i < 3; ++i) {
      barrier(command);
      dispatch(command, atrousProgram, atrousPipeline, f.atrous[i], width, height);
    }
    barrier(command);
    VkBufferCopy copy{0, 0, sampleBuffer.size};
    vkCmdCopyBuffer(command, sampleBuffer.handle, previousSamples.handle, 1, &copy);
    copy.size = temporalBuffer.size;
    vkCmdCopyBuffer(command, temporalBuffer.handle, previousHistory.handle, 1, &copy);
    vkCmdCopyBuffer(command, scratchBuffer.handle, filteredBuffer.handle, 1, &copy);
    previousCamera = camera; previousKey = key; valid = true; ++updateCount;
    barrier(command);
  }
  DescriptorWriter(context, compositeProgram.compute(), f.composite).buffer("uniforms", f.uniforms[0])
      .buffer("filtered", filteredBuffer).storageTexture("output", lit.view).apply();
  transitionImage(command, lit, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
  dispatch(command, compositeProgram, compositePipeline, f.composite, lit.description.width, lit.description.height);
  transitionImage(command, lit, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  if (timestamps) vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, timestamps, slot * 2 + 1);
  return true;
}

std::uint64_t PathReconstruction::bytes() const {
  std::uint64_t total = sampleBuffer.size + previousSamples.size + previousHistory.size + temporalBuffer.size +
                        scratchBuffer.size + filteredBuffer.size;
  for (const auto &f : frames) { total += f.staging.size; for (const auto &u : f.uniforms) total += u.size; }
  return total;
}
} // namespace basalt
