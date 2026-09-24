#include "pt/WavefrontPlan.h"

#include <algorithm>
#include <limits>

namespace pt {

WavefrontPlan planWavefront(std::uint64_t pixels, std::uint32_t samplesPerPixel, std::uint64_t requestedPaths,
                            std::uint64_t guidePixels, const WavefrontLimits &limits) {
  WavefrontPlan plan;
  if (pixels == 0 || samplesPerPixel == 0) {
    plan.error = "a wavefront frame needs at least one pixel and one sample";
    return plan;
  }
  if (pixels > std::numeric_limits<std::uint32_t>::max()) {
    plan.error = "the image has more pixels than the 32-bit path numbering allows";
    return plan;
  }
  const std::uint64_t total = pixels * samplesPerPixel;  // < 2^32 * 2^32: no overflow
  auto tighten = [&](std::uint64_t limit, const char *name) {
    if (limit < plan.limitPaths || plan.limitedBy.empty()) {
      plan.limitPaths = limit;
      plan.limitedBy = name;
    }
  };
  tighten(requestedPaths ? requestedPaths : kWavefrontDefaultPaths,
          requestedPaths ? "requested capacity" : "default capacity");
  tighten(limits.maxStorageBufferRange / kWavefrontLargestElement, "maxStorageBufferRange");
  tighten(static_cast<std::uint64_t>(limits.maxWorkGroupCountX) * kWavefrontGroup, "maxComputeWorkGroupCount[0]");
  if (limits.maxRayDispatchInvocations) tighten(limits.maxRayDispatchInvocations, "maxRayDispatchInvocationCount");
  tighten(std::numeric_limits<std::uint32_t>::max() / kWavefrontGroup * kWavefrontGroup, "32-bit queue indices");
  const std::uint64_t perPath = kWavefrontPathBytes;
  const std::uint64_t fixed = kWavefrontFixedBytes + guidePixels * kWavefrontGuidePixelBytes;
  if (limits.budgetBytes <= fixed) {
    plan.error = "the wavefront budget cannot hold the per-pixel guide storage";
    return plan;
  }
  tighten((limits.budgetBytes - fixed) / perPath, "allocation budget");
  const std::uint64_t usable = std::min(plan.limitPaths, total);
  const std::uint64_t capacity = usable / samplesPerPixel * samplesPerPixel;
  if (capacity == 0) {
    plan.error = "one pixel's " + std::to_string(samplesPerPixel) + " samples exceed the wavefront capacity of " +
                 std::to_string(plan.limitPaths) + " paths (" + plan.limitedBy + ")";
    return plan;
  }
  plan.capacity = static_cast<std::uint32_t>(capacity);
  plan.batches = static_cast<std::uint32_t>((total + capacity - 1) / capacity);
  plan.bytes = capacity * perPath + fixed;
  return plan;
}

} // namespace pt
