#include "render/GpuProfiler.h"

#include "core/Log.h"

namespace basalt {

const char *gpuStageName(GpuStage stage) {
  switch (stage) {
  case GpuStage::Setup: return "setup";
  case GpuStage::Generate: return "generate";
  case GpuStage::Intersect: return "intersect";
  case GpuStage::Shade: return "shade";
  case GpuStage::Shadow: return "shadow";
  case GpuStage::Resolve: return "resolve";
  case GpuStage::Trace: return "trace";
  case GpuStage::Sort: return "sort";
  default: return "unknown";
  }
}

GpuProfiler::GpuProfiler(const Context &ctx) : context(ctx) {
  for (Slot &slot : slots) {
    VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = kMarks;
    check(vkCreateQueryPool(context.device, &info, nullptr, &slot.pool), "vkCreateQueryPool (profiler)");
    vkResetQueryPool(context.device, slot.pool, 0, kMarks);
    slot.counters = Buffer(context, kCounters * 2 * sizeof(std::uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           VMA_MEMORY_USAGE_AUTO,
                           VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                           "profiler.counters");
  }
}

GpuProfiler::~GpuProfiler() {
  for (Slot &slot : slots)
    if (slot.pool) vkDestroyQueryPool(context.device, slot.pool, nullptr);
}

void GpuProfiler::collect(Slot &slot) {
  if (!slot.pending) return;
  slot.pending = false;
  if (slot.overflowWatched) {
    // The last element of the counter readback; written before this slot's fence.
    const std::uint32_t refused = static_cast<const std::uint32_t *>(slot.counters.mapped)[kCounters * 2 - 1];
    if (refused) logError("wavefront queues refused {} reservations in one frame", refused);
    overflowSum += refused;
  }
  const std::uint32_t marks = static_cast<std::uint32_t>(slot.stages.size());
  if (marks < 2) return;
  std::vector<std::uint64_t> stamps(marks);
  if (vkGetQueryPoolResults(context.device, slot.pool, 0, marks, stamps.size() * sizeof(std::uint64_t),
                            stamps.data(), sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
    return;
  const double period = context.properties.limits.timestampPeriod * 1e-6;  // ticks to ms
  GpuProfile profile;
  profile.valid = true;
  profile.truncated = slot.truncated;
  profile.dispatches = marks - 1;
  profile.traceMilliseconds = static_cast<double>(stamps[marks - 1] - stamps[0]) * period;
  for (std::uint32_t i = 1; i < marks; ++i)
    profile.stageMilliseconds[static_cast<std::size_t>(slot.stages[i])] +=
        static_cast<double>(stamps[i] - stamps[i - 1]) * period;
  const auto *values = static_cast<const std::uint32_t *>(slot.counters.mapped);
  for (std::size_t i = 0; i < slot.counterKinds.size(); ++i) {
    if (slot.counterKinds[i] == 0) {
      profile.liveByBounce.push_back(values[i * 2]);
    } else {
      profile.shadowByBounce.push_back(values[i * 2]);
    }
  }
  if (slot.fresh) series.push_back(profile.traceMilliseconds);
  newest = std::move(profile);
}

void GpuProfiler::begin(VkCommandBuffer command, std::uint32_t frameSlot) {
  current = frameSlot % kFramesInFlight;
  Slot &slot = slots[current];
  collect(slot);
  slot.stages.clear();
  slot.counterKinds.clear();
  slot.overflowWatched = false;
  slot.truncated = false;
  slot.fresh = false;
  slot.pending = true;
  vkCmdResetQueryPool(command, slot.pool, 0, kMarks);
}

void GpuProfiler::mark(VkCommandBuffer command, GpuStage stage) {
  Slot &slot = slots[current];
  if (slot.stages.size() >= kMarks) {
    slot.truncated = true;
    return;
  }
  vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, slot.pool,
                       static_cast<std::uint32_t>(slot.stages.size()));
  slot.stages.push_back(stage);
}

void GpuProfiler::copyCounter(VkCommandBuffer command, VkBuffer source, VkDeviceSize offset, int kind) {
  Slot &slot = slots[current];
  if (slot.counterKinds.size() >= kCounters - 1) {
    slot.truncated = true;
    return;
  }
  VkBufferCopy region{offset, slot.counterKinds.size() * 2 * sizeof(std::uint32_t), 2 * sizeof(std::uint32_t)};
  vkCmdCopyBuffer(command, source, slot.counters.handle, 1, &region);
  slot.counterKinds.push_back(kind);
}

void GpuProfiler::watchOverflow(VkCommandBuffer command, VkBuffer source, VkDeviceSize offset) {
  Slot &slot = slots[current];
  VkBufferCopy region{offset, (kCounters * 2 - 1) * sizeof(std::uint32_t), sizeof(std::uint32_t)};
  vkCmdCopyBuffer(command, source, slot.counters.handle, 1, &region);
  slot.overflowWatched = true;
}

void GpuProfiler::end(VkCommandBuffer command) {
  if (slots[current].counterKinds.empty() && !slots[current].overflowWatched) return;
  VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  barrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
  barrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.memoryBarrierCount = 1;
  dependency.pMemoryBarriers = &barrier;
  vkCmdPipelineBarrier2(command, &dependency);
}

void GpuProfiler::finish() {
  context.waitIdle();
  // Oldest first: the slot after the current one was begun earlier.
  for (std::uint32_t i = 1; i <= kFramesInFlight; ++i) collect(slots[(current + i) % kFramesInFlight]);
}

} // namespace basalt
