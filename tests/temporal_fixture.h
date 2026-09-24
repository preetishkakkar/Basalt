#pragma once
#include "pt/Tracing.h"
#include <vector>

namespace temporal_fixture {
inline pt::PathUniforms camera(pt::uint width, pt::uint height) {
  pt::PathUniforms u{};
  u.cameraPosition = pt::float4(0, 0, 2, 1);
  u.cameraForward = pt::float4(0, 0, -1, 0);
  u.cameraRight = pt::float4(0.5f * float(width) / float(height), 0, 0, 0);
  u.cameraUp = pt::float4(0, 0.5f, 0, 0);
  u.image = pt::float4(float(width), float(height), 1, 1);
  return u;
}
inline std::vector<pt::PtReconstructionSample> plane(pt::uint width, pt::uint height, pt::uint seed) {
  using namespace pt;
  const auto c = camera(width, height);
  std::vector<PtReconstructionSample> samples(width * height);
  for (uint y = 0; y < height; ++y) for (uint x = 0; x < width; ++x) {
    auto &s = samples[y * width + x];
    s = ptEmptyReconstructionSample();
    const float nx = (float(x) + 0.5f) / float(width) * 2 - 1;
    const float ny = 1 - (float(y) + 0.5f) / float(height) * 2;
    const float3 ray = xyz(c.cameraForward) + xyz(c.cameraRight) * nx + xyz(c.cameraUp) * ny;
    s.positionDepth = float4(xyz(c.cameraPosition) + ray * 2, 2);
    s.geometricNormal = float4(0, 0, 1, 0);
    s.normalRoughness = float4(0, 0, 1, 0.6f);
    s.albedo = float4(0.5f);
    s.viewDistance = float4(normalize(ray), -1);
    s.identity = uint4(1, 1, 1, 0);
    const float noise = pathRandom4(pathSeed(x, y, seed, 11), 0, 0).x * 2;
    s.diffuse = float4(float3(noise), 0);
    s.specular = float4(float3(noise * 0.1f), 0);
  }
  return samples;
}
inline std::vector<pt::PtTemporalHistory> temporal(pt::PtTemporalUniforms u,
    const std::vector<pt::PtReconstructionSample> &s, const std::vector<pt::PtReconstructionSample> &old,
    const std::vector<pt::PtTemporalHistory> &h) {
  std::vector<pt::PtTemporalHistory> result(s.size());
  for (pt::uint y = 0; y < u.image.y; ++y) for (pt::uint x = 0; x < u.image.x; ++x)
    result[y * u.image.x + x] = pt::ptTemporalPixel(u, x, y, s.data(), old.data(), h.data());
  return result;
}
inline std::vector<pt::PtTemporalHistory> filter(pt::PtTemporalUniforms u,
    const std::vector<pt::PtReconstructionSample> &s, std::vector<pt::PtTemporalHistory> h) {
  std::vector<pt::PtTemporalHistory> next(h.size());
  for (pt::uint stride : {1u, 2u, 4u}) {
    u.image.w = stride;
    for (pt::uint y = 0; y < u.image.y; ++y) for (pt::uint x = 0; x < u.image.x; ++x)
      next[y * u.image.x + x] = pt::ptAtrousPixel(u, x, y, s.data(), h.data());
    h.swap(next);
  }
  return h;
}
} // namespace temporal_fixture
