#pragma once
#include "gpu/Descriptors.h"
#include "gpu/Pipeline.h"
#include "gpu/Swapchain.h"
#include "pt/Tracing.h"
#include <array>

namespace basalt {
// Display-only reconstruction on the render queue. Caller waits for slot fences before
// uploading/recording and for queue idle before destruction.
class PathReconstruction {
public:
  PathReconstruction(Context &context, std::uint32_t width, std::uint32_t height);
  ~PathReconstruction();
  Buffer &samples() { return sampleBuffer; }
  const Buffer &history() const { return previousHistory; }
  const Buffer &filtered() const { return filteredBuffer; }
  void upload(VkCommandBuffer command, std::uint32_t slot, const std::vector<pt::PtReconstructionSample> &samples);
  bool record(VkCommandBuffer command, std::uint32_t slot, Image &lit, const pt::PathUniforms &camera,
              std::uint64_t resetKey, std::uint32_t rawSamples, bool fresh, float cutDistance);
  std::uint64_t bytes() const;
  std::uint64_t updates() const { return updateCount; }
  bool lastReset() const { return resetLast; }
  float filterMilliseconds() const { return filterMs; }
  float uploadMilliseconds() const { return uploadMs; }
  float transferMilliseconds() const { return transferMs; }
  bool fits(std::uint32_t w, std::uint32_t h) const { return w == width && h == height; }
private:
  Context &context;
  std::uint32_t width, height;
  Program temporalProgram, atrousProgram, compositeProgram;
  Pipeline temporalPipeline, atrousPipeline, compositePipeline;
  DescriptorPool pool;
  Buffer sampleBuffer, previousSamples, previousHistory, temporalBuffer, scratchBuffer, filteredBuffer;
  struct Frame {
    std::array<Buffer, 4> uniforms;
    Buffer staging;
    VkDescriptorSet temporal{}, composite{};
    std::array<VkDescriptorSet, 3> atrous{};
  };
  std::array<Frame, kFramesInFlight> frames;
  pt::PathUniforms previousCamera{};
  std::uint64_t previousKey = 0, updateCount = 0;
  bool valid = false, resetLast = true;
  VkQueryPool timestamps = VK_NULL_HANDLE;
  VkQueryPool uploadTimestamps = VK_NULL_HANDLE;
  std::array<bool, kFramesInFlight> timestampWritten{};
  std::array<bool, kFramesInFlight> timestampFresh{};
  std::array<bool, kFramesInFlight> uploadTimestampWritten{};
  float filterMs = 0.0f, uploadMs = 0.0f, transferMs = 0.0f;
};
} // namespace basalt
