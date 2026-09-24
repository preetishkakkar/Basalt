// Optional GPU stage timing and queue counters for the path tracers.
//
// Each frame in flight owns a timestamp pool and a host-visible counter buffer. Marks and
// counter copies are recorded into the frame's command buffer; their results are read when
// that frame slot is next begun, after its fence, so profiling adds no CPU/GPU wait to the
// frame it measures. finish() waits for the device and collects every pending slot, for
// captures. Timings are attributed to the stage that ends at each mark: consecutive marks
// separated by full barriers bracket one stage's execution.
#pragma once
#include "gpu/Context.h"
#include "gpu/Resources.h"
#include "gpu/Swapchain.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace basalt {

enum class GpuStage : std::uint8_t {
  Setup,      // counter/control transfers and their barrier
  Generate,   // camera-path generation (wavefront init)
  Intersect,  // nearest-hit traversal
  Shade,      // material, light sampling and queue compaction
  Shadow,     // occlusion traversal
  Resolve,    // per-pixel reduction into the accumulator
  Trace,      // a whole monolithic trace (megakernel or iterative raygen)
  Sort,       // optional shading-order sort (key, scan, scatter)
  Count
};

const char *gpuStageName(GpuStage stage);

struct GpuProfile {
  bool valid = false;
  double traceMilliseconds = 0.0;  // first mark to last mark
  std::array<double, static_cast<std::size_t>(GpuStage::Count)> stageMilliseconds{};
  std::uint32_t dispatches = 0;    // marks after the first: one per recorded stage
  std::vector<std::uint32_t> liveByBounce;    // continuation queue length after each bounce's shade
  std::vector<std::uint32_t> shadowByBounce;  // shadow queue length after each bounce's shade
  bool truncated = false;                     // more marks than the pool holds
};

class GpuProfiler {
public:
  explicit GpuProfiler(const Context &context);
  ~GpuProfiler();
  GpuProfiler(const GpuProfiler &) = delete;
  GpuProfiler &operator=(const GpuProfiler &) = delete;

  // Collects this slot's previous results (its fence has signalled) and starts a new frame.
  void begin(VkCommandBuffer command, std::uint32_t frameSlot);
  // Writes a timestamp once all earlier work finishes; the interval since the previous
  // mark is attributed to `stage`.
  void mark(VkCommandBuffer command, GpuStage stage);
  // Copies the queue count at `offset` of `source` for readback: continuation (kind 0) or
  // shadow (kind 1) queue length after a bounce's shade.
  void copyCounter(VkCommandBuffer command, VkBuffer source, VkDeviceSize offset, int kind);
  // Copies one uint that must stay zero (refused queue reservations); collected values add
  // to overflowTotal(). Recorded in every frame, profiled or not.
  void watchOverflow(VkCommandBuffer command, VkBuffer source, VkDeviceSize offset);
  std::uint64_t overflowTotal() const { return overflowSum; }
  // Makes this frame's counter copies visible to the host once its fence signals.
  void end(VkCommandBuffer command);
  // Waits for the device and collects every slot; the newest complete profile wins.
  void finish();

  const GpuProfile &latest() const { return newest; }
  // Trace times of every collected frame that carried fresh samples, oldest first.
  const std::vector<double> &traceSeries() const { return series; }
  void clearSeries() { series.clear(); }
  void setFresh(bool fresh) { slots[current].fresh = fresh; }

private:
  static constexpr std::uint32_t kMarks = 4096;
  static constexpr std::uint32_t kCounters = 1024;
  struct Slot {
    VkQueryPool pool = VK_NULL_HANDLE;
    Buffer counters;
    std::vector<GpuStage> stages;       // stage ending at each mark
    std::vector<int> counterKinds;      // per counter copy
    bool overflowWatched = false;
    bool pending = false, fresh = false, truncated = false;
  };
  void collect(Slot &slot);

  const Context &context;
  std::array<Slot, kFramesInFlight> slots;
  std::uint32_t current = 0;
  GpuProfile newest;
  std::vector<double> series;
  std::uint64_t overflowSum = 0;
};

} // namespace basalt
