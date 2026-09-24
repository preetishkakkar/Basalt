// The ray-cone level-of-detail functions of shaders/pt/raycone.h are
// bitwise identical on the CPU and the GPU over a fixed corpus of 10,000 hits.
#include "core/Log.h"
#include "platform/Window.h"
#include "gpu/Uploader.h"
#include "gpu/Pipeline.h"
#include "gpu/Descriptors.h"
#include "pt/Tracing.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

using namespace basalt;
namespace {
struct RayConeCase {
  pt::float4 row0, row1, row2, p0, p1, p2, uv01, uv2Cone, directionT, misc;
};
struct RayConeResult {
  pt::float4 lod, cones;
};

void require(bool condition, const char *why) { if (!condition) throw std::runtime_error(why); }

// Scaled and sheared instances, triangles from slivers to large, grazing to head-on
// directions, distances over nine decades, a tenth of the cones level zero, and power-of-two
// textures from 1 to 4096 texels a side.
std::vector<RayConeCase> corpus() {
  std::mt19937 random(0x7C0E5u);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f), zeroOne(0.0f, 1.0f);
  auto logUniform = [&](float low, float high) {
    return std::exp2(std::log2(low) + zeroOne(random) * (std::log2(high) - std::log2(low)));
  };
  std::vector<RayConeCase> cases(10000);
  for (RayConeCase &c : cases) {
    auto row = [&] { return pt::float4(2.0f * unit(random), 2.0f * unit(random), 2.0f * unit(random), 10.0f * unit(random)); };
    c.row0 = row(); c.row1 = row(); c.row2 = row();
    const float size = logUniform(1e-3f, 1e2f);
    auto corner = [&](float w) { return pt::float4(size * unit(random), size * unit(random), size * unit(random), w); };
    const float texels = std::exp2(static_cast<float>(random() % 13u)), other = std::exp2(static_cast<float>(random() % 13u));
    c.p0 = corner(texels); c.p1 = corner(other);
    c.p2 = corner(std::log2(std::max(texels, other)) + 1.0f);
    const float tiling = logUniform(1e-2f, 1e2f);
    c.uv01 = pt::float4(tiling * unit(random), tiling * unit(random), tiling * unit(random), tiling * unit(random));
    const bool levelZero = random() % 10u == 0u;
    c.uv2Cone = pt::float4(tiling * unit(random), tiling * unit(random), levelZero ? -1.0f : logUniform(1e-6f, 1.0f),
                           levelZero ? 0.0f : logUniform(1e-5f, 1.5f));
    pt::float3 d(unit(random), unit(random), unit(random));
    if (pt::dot(d, d) < 1e-4f) d = pt::float3(0.0f, 0.0f, 1.0f);
    d = pt::normalize(d);
    c.directionT = pt::float4(d.x, d.y, d.z, logUniform(1e-3f, 1e6f));
    c.misc = pt::float4(zeroOne(random), logUniform(1e-30f, 1e30f), 0.0f, 0.0f);
  }
  // Edge cases the random draws almost never reach: a grazing direction below the 1e-4 cosine
  // floor, a degenerate UV mapping, collinear corners (no world area), a zero cone width.
  cases[0].row0 = pt::float4(1.0f, 0.0f, 0.0f, 0.0f); cases[0].row1 = pt::float4(0.0f, 1.0f, 0.0f, 0.0f);
  cases[0].row2 = pt::float4(0.0f, 0.0f, 1.0f, 0.0f);
  cases[0].p0 = pt::float4(0.0f, 0.0f, 0.0f, 256.0f); cases[0].p1 = pt::float4(0.0f, 1.0f, 0.0f, 256.0f);
  cases[0].p2 = pt::float4(0.0f, 0.0f, 1.0f, 9.0f);  // normal +x
  cases[0].directionT = pt::float4(0.0f, 0.70710678f, 0.70710678f, 3.0f);  // in the triangle's plane
  cases[1].uv01 = pt::float4(0.0f, 0.0f, 0.0f, 0.0f); cases[1].uv2Cone.x = 0.0f; cases[1].uv2Cone.y = 0.0f;
  cases[2].p0 = pt::float4(0.0f, 0.0f, 0.0f, 64.0f); cases[2].p1 = pt::float4(1.0f, 1.0f, 1.0f, 64.0f);
  cases[2].p2 = pt::float4(2.0f, 2.0f, 2.0f, 7.0f);
  cases[3].uv2Cone.z = 0.0f;
  return cases;
}

RayConeResult expected(const RayConeCase &c) {
  using namespace pt;
  const float2 cone(c.uv2Cone.z, c.uv2Cone.w);
  const float width = ptConeWidthOrLevelZero(cone, c.directionT.w);
  const float lodBase = ptConeLodBase(ptUvCross(float2(c.uv01.x, c.uv01.y), float2(c.uv01.z, c.uv01.w), float2(c.uv2Cone.x, c.uv2Cone.y)),
                                      ptWorldCross(c.row0, c.row1, c.row2, xyz(c.p0), xyz(c.p1), xyz(c.p2)),
                                      xyz(c.directionT), width);
  const float2 scattered = ptConeScattered(cone, width, c.misc.x), shadow = ptConeShadow(cone, width);
  return {float4(width, lodBase, ptTextureLevel(lodBase, c.p0.w, c.p1.w, c.p2.w), ptLog2(c.misc.y)),
          float4(scattered.x, scattered.y, shadow.x, shadow.y)};
}

int run() {
  openLogFile("gpu-raycone-validation.log");
  Window window("Ray-cone oracle", 64, 64, false);
  Context context(window, true);
  require(context.validationEnabled, "GPU ray-cone tests require Vulkan validation");
  std::size_t mismatches = 0;
  const std::vector<RayConeCase> cases = corpus();
  {
    Uploader uploader(context);
    Program program(context, "raycone_oracle");
    Pipeline pipeline(context, program, "raycone oracle");
    Buffer caseBuffer = uploader.createBuffer(cases.data(), cases.size() * sizeof(RayConeCase),
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "raycone.cases");
    std::vector<RayConeResult> zeros(cases.size());
    Buffer resultBuffer = uploader.createBuffer(zeros.data(), zeros.size() * sizeof(RayConeResult),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, "raycone.results");
    const pt::uint4 control(static_cast<std::uint32_t>(cases.size()), 0u, 0u, 0u);
    Buffer controlBuffer = uploader.createBuffer(&control, sizeof(control), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                 "raycone.control");
    DescriptorPool pool(context, 1u);
    const VkDescriptorSet set = pool.allocate(program.setLayouts[0]);
    DescriptorWriter(context, program.compute(), set)
        .buffer("cases", caseBuffer).buffer("results", resultBuffer).buffer("control", controlBuffer).apply();
    uploader.runImmediate([&](VkCommandBuffer cmd) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, program.layout, 0, 1, &set, 0, nullptr);
      vkCmdDispatch(cmd, static_cast<std::uint32_t>((cases.size() + 63) / 64), 1, 1);
    });
    const auto bytes = uploader.readBuffer(resultBuffer, resultBuffer.size);
    std::vector<RayConeResult> actual(cases.size());
    std::memcpy(actual.data(), bytes.data(), actual.size() * sizeof(RayConeResult));
    const char *names[8] = {"width", "lodBase", "level", "log2", "scattered width", "scattered spread",
                            "shadow width", "shadow spread"};
    for (std::size_t i = 0; i < cases.size(); ++i) {
      const RayConeResult e = expected(cases[i]);
      std::uint32_t a[8], b[8];
      std::memcpy(a, &actual[i], sizeof(a));
      std::memcpy(b, &e, sizeof(b));
      for (int k = 0; k < 8; ++k)
        if (a[k] != b[k] && mismatches++ < 10) {
          float fa, fb;
          std::memcpy(&fa, &a[k], 4);
          std::memcpy(&fb, &b[k], 4);
          std::fprintf(stderr, "hit %zu %s: GPU %.9g (0x%08X) CPU %.9g (0x%08X)\n", i, names[k], fa, a[k], fb, b[k]);
        }
    }
    context.waitIdle();
  }
  require(!context.sawValidationError, "Vulkan validation error during ray-cone tests");
  if (mismatches) {
    std::fprintf(stderr, "%zu ray-cone values differ between the CPU and the GPU\n", mismatches);
    return 1;
  }
  std::printf("PASS ray-cone level of detail bitwise on %zu hits on %s\n", cases.size(), context.info.name.c_str());
  return 0;
}
}  // namespace

int main() {
  try { return run(); }
  catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
