#include "core/Log.h"
#include "platform/Window.h"
#include "gpu/Uploader.h"
#include "render/PathReconstruction.h"
#include "temporal_fixture.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>

using namespace basalt;
namespace {
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
void compare(Uploader &uploader, const Buffer &buffer, const std::vector<pt::PtTemporalHistory> &expected) {
  const auto bytes = uploader.readBuffer(buffer, buffer.size);
  std::vector<pt::PtTemporalHistory> actual(expected.size());
  std::memcpy(actual.data(), bytes.data(), bytes.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float *a = reinterpret_cast<const float *>(&actual[i]), *e = reinterpret_cast<const float *>(&expected[i]);
    for (int c = 0; c < 12; ++c) {
      if (!std::isfinite(a[c]) || std::abs(a[c] - e[c]) > 2e-4f * std::max(1.0f, std::abs(e[c]))) {
        std::fprintf(stderr, "pixel %zu component %d: GPU %.9g CPU %.9g\n", i, c, a[c], e[c]);
        throw std::runtime_error("GPU reconstruction disagrees with CPU equations");
      }
    }
  }
}
int run() {
  Window window("Path reconstruction tests", 64, 64, false);
  Context context(window, true);
  require(context.validationEnabled, "GPU reconstruction tests require Vulkan validation");
  {
    Uploader uploader(context);
    constexpr pt::uint w = 17, h = 11;
    ImageDescription d;
    d.width = w; d.height = h; d.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    d.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
              VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    Image lit(context, d);
    PathReconstruction filter(context, w, h);
    auto camera = temporal_fixture::camera(w, h);
    auto samples = temporal_fixture::plane(w, h, 0), old = samples;
    std::vector<pt::PtTemporalHistory> history(w * h);
    for (pt::uint f = 0; f < 10; ++f) {
      samples = temporal_fixture::plane(w, h, f);
      if (f >= 5) for (auto &s : samples) { s.diffuse = pt::float4(0); s.specular = pt::float4(0); }
      const std::uint64_t key = f < 5 ? 1 : 2;
      pt::PtTemporalUniforms u{};
      u.previousCamera = camera; u.image = pt::uint4(w, h, f == 0 || f == 5 ? 1 : 0, 1);
      history = temporal_fixture::temporal(u, samples, old, history);
      const auto filtered = temporal_fixture::filter(u, samples, history);
      uploader.runImmediate([&](VkCommandBuffer cmd) {
        transitionImage(cmd, lit, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_CLEAR_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT);
        const VkClearColorValue color{{0, 0, 0, 1}};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, lit.handle, lit.layout, &color, 1, &range);
        filter.upload(cmd, f % kFramesInFlight, samples);
        require(filter.record(cmd, f % kFramesInFlight, lit, camera, key, 1, true, 1), "fresh filter did not dispatch");
      });
      compare(uploader, filter.history(), history);
      compare(uploader, filter.filtered(), filtered);
      const auto count = filter.updates();
      uploader.runImmediate([&](VkCommandBuffer cmd) {
        filter.record(cmd, f % kFramesInFlight, lit, camera, key, 1, false, 1);
      });
      require(filter.updates() == count, "duplicate display advanced reconstruction history");
      compare(uploader, filter.history(), history);
      old = samples;
    }
    require(filter.bytes() <= static_cast<std::uint64_t>(w) * h * 704 + 4096, "history allocation is unbounded");
    // Resets are tested at the dispatch owner, not only in shared filter equations.
    std::uint64_t key = 2;
    for (int change = 0; change < 5; ++change) {
      if (change == 0) camera.cameraPosition.x += 2; // translation cut
      if (change == 1) camera.cameraForward = pt::float4(1, 0, 0, 0); // rotation cut
      if (change == 2) camera.cameraRight.x *= 1.2f; // projection change
      if (change >= 3) ++key; // scene/light/material/environment or backend/integrator epoch
      uploader.runImmediate([&](VkCommandBuffer cmd) {
        filter.upload(cmd, 0, samples);
        filter.record(cmd, 0, lit, camera, key, 1, true, 1);
      });
      require(filter.lastReset(), "camera/projection/scene change failed to reset history");
      const auto bytes = uploader.readBuffer(filter.history(), filter.history().size);
      std::vector<pt::PtTemporalHistory> hs(w * h); std::memcpy(hs.data(), bytes.data(), bytes.size());
      for (const auto &p : hs) require(p.diffuse.w == 1 && p.specular.w == 1,
          "reset accepted old history samples");
    }
    // After convergence the display must equal raw, even when history is black.
    uploader.runImmediate([&](VkCommandBuffer cmd) {
      transitionImage(cmd, lit, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
          VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_CLEAR_BIT,
          VK_ACCESS_2_TRANSFER_WRITE_BIT);
      const VkClearColorValue raw{{0.125f, 0.25f, 0.5f, 1}};
      const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      vkCmdClearColorImage(cmd, lit.handle, lit.layout, &raw, 1, &range);
      filter.record(cmd, 0, lit, camera, key, 256, false, 1);
    });
    const auto displayed = uploader.readImage(lit, 16);
    std::vector<pt::float4> colors(w * h); std::memcpy(colors.data(), displayed.data(), displayed.size());
    for (const auto &p : colors) require(std::abs(p.x - 0.125f) + std::abs(p.y - 0.25f) + std::abs(p.z - 0.5f) < 1e-6f,
        "stationary reconstruction did not return to raw at 256 real SPP");
    // New extent/new owner has no access to old history; low-resolution guides fill
    // the complete display extent (including odd sizes), without increasing raw SPP.
    {
      PathReconstruction resized(context, 5, 3);
      auto low = temporal_fixture::plane(5, 3, 0);
      for (auto &s : low) { s.diffuse = pt::float4(0); s.specular = pt::float4(0); }
      uploader.runImmediate([&](VkCommandBuffer cmd) {
        resized.upload(cmd, 0, low);
        resized.record(cmd, 0, lit, temporal_fixture::camera(5, 3), 9, 0, true, 1);
      });
      require(resized.lastReset() && resized.updates() == 1, "resize carried history across extents");
      const auto result = uploader.readImage(lit, 16);
      std::memcpy(colors.data(), result.data(), result.size());
      for (const auto &p : colors) require(pt::length(pt::xyz(p)) < 1e-6f, "preview upscale left stale display pixels");
      context.waitIdle();
    }
    context.waitIdle();
    std::printf("reconstruction: %.3f ms filter, %.3f ms host guide copy, %llu bytes\n",
        filter.filterMilliseconds(), filter.uploadMilliseconds(), static_cast<unsigned long long>(filter.bytes()));
  }
  if (context.sawValidationError)
    for (const auto &entry : logEntries()) if (entry.level == LogEntry::Level::ErrorLevel)
      std::fprintf(stderr, "%s\n", entry.text.c_str());
  require(!context.sawValidationError, "Vulkan validation error during reconstruction tests");
  std::printf("PASS GPU reconstruction, reset and duplicate-publication tests on %s\n", context.info.name.c_str());
  return 0;
}
}
int main() {
  try { return run(); }
  catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
