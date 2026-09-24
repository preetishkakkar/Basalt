// Capacity planning for the GPU wavefront queues (V6.1). Pure arithmetic, independent of
// Vulkan, so the bounds are unit-tested on the CPU.
//
// A frame traces pixels x S paths, numbered pixel-major. They run in batches of at most
// `capacity` paths, each batch a whole number of pixels, so every per-path and queue array
// holds `capacity` elements. Each queued path emits at most one continuation and at most
// one shadow ray per bounce, so neither queue can exceed the batch; the counters still
// record overflow and a nonzero value fails validation tests.
#pragma once
#include <cstdint>
#include <string>

namespace pt {

// Bytes per path slot: two ping-pong states (48 each), hit (24), result (48), shadow (48)
// and traversal costs (8). Per pixel, when temporal guides are collected: 48.
inline constexpr std::uint64_t kWavefrontPathBytes = 48 * 2 + 24 + 48 + 48 + 8;
inline constexpr std::uint64_t kWavefrontLargestElement = 48;
inline constexpr std::uint64_t kWavefrontGuidePixelBytes = 48;
inline constexpr std::uint64_t kWavefrontFixedBytes = 128;  // counters and control
inline constexpr std::uint32_t kWavefrontGroup = 64;
inline constexpr std::uint32_t kWavefrontDefaultPaths = 1u << 22;
inline constexpr std::uint64_t kWavefrontBudgetBytes = 3ull * 1024 * 1024 * 1024;

struct WavefrontLimits {
  std::uint64_t maxStorageBufferRange = 1ull << 27;  // Vulkan's guaranteed minimum
  std::uint32_t maxWorkGroupCountX = 65535;           // likewise
  std::uint32_t maxRayDispatchInvocations = 0;        // ray pipeline launches; 0 when unused
  std::uint64_t budgetBytes = kWavefrontBudgetBytes;
};

struct WavefrontPlan {
  std::uint32_t capacity = 0;       // paths per queue array (a multiple of the planned S)
  std::uint32_t batches = 0;        // batches for the planned S
  std::uint64_t bytes = 0;          // every wavefront allocation
  std::uint64_t limitPaths = 0;     // the tightest device/request limit applied
  std::string limitedBy;            // which limit that was
  std::string error;                // nonempty: nothing can be traced within the limits
};

// requestedPaths 0 selects kWavefrontDefaultPaths. guidePixels is 0 without temporal guides.
WavefrontPlan planWavefront(std::uint64_t pixels, std::uint32_t samplesPerPixel, std::uint64_t requestedPaths,
                            std::uint64_t guidePixels, const WavefrontLimits &limits);

// Paths in one batch for S samples per pixel: whole pixels, never more than capacity.
inline std::uint32_t wavefrontBatchPaths(std::uint32_t capacity, std::uint32_t samplesPerPixel) {
  return samplesPerPixel == 0 ? 0u : capacity / samplesPerPixel * samplesPerPixel;
}

} // namespace pt
