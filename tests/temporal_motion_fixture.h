#pragma once
#include "pt_fixture.h"
#include "temporal_fixture.h"
#include <array>

// Frozen V3.1 scene and trajectories. Do not edit to make a failed quality gate pass.
namespace temporal_motion {
constexpr pt::uint width = 32, height = 24, frames = 12, referenceSamples = 4096;
constexpr std::array<pt::uint, 4> seeds{11, 22, 33, 44};
constexpr std::array<const char *, 3> sequences{"translation", "rotation", "stop-start"};

inline void scene(pt_fixture::Builder &b) {
  using namespace pt;
  const auto floor = b.material(float3(0.55f, 0.6f, 0.65f), 0, 0.9f);
  b.floor(5, 0, floor);
  b.wall(3, -1.5f, b.material(float3(0.6f, 0.5f, 0.4f), 0, 0.8f));
  b.sphere(float3(-0.8f, 0.5f, 0), 0.5f, b.material(float3(0.8f, 0.3f, 0.1f), 0, 0.8f), 32);
  b.sphere(float3(0.35f, 0.5f, 0), 0.5f, b.material(float3(0.65f, 0.75f, 0.85f), 1, 0.3f), 32);
  b.sphere(float3(1.15f, 0.3f, 0.65f), 0.3f, b.material(float3(0.9f), 1, 0.01f), 32);
  const auto thin = b.material(float3(0.1f, 0.4f, 0.1f), 0, 0.7f);
  const auto blend = b.material(float3(0.7f, 0.2f, 0.7f), 0, 0.7f, 0.35f, 2);
  const auto mask = b.material(float3(0.4f, 0.8f, 0.2f), 0, 0.7f, 1, 1);
  const std::vector<float3> normal(4, float3(0, 0, 1));
  for (uint i = 0; i < 3; ++i) {
    const float x = -0.35f + float(i) * 0.24f;
    const float half = i == 0 ? 0.015f : 0.08f;
    b.mesh({{x-half,0,1}, {x+half,0,1}, {x+half,1.4f,1}, {x-half,1.4f,1}}, normal,
        {0,1,2,0,2,3}, i == 0 ? thin : i == 1 ? blend : mask,
        pt_fixture::affine(0, 1, float3(0)), kRayMaskScene,
        i == 0 ? 0 : i == 1 ? kInstanceBlended : kInstanceMasked);
  }
  HostTexture checker;
  checker.width = checker.height = 8;
  checker.texels.resize(64);
  for (uint y = 0; y < 8; ++y) for (uint x = 0; x < 8; ++x)
    checker.texels[y * 8 + x] = ((x + y) % 2 ? 0xff000000u : 0u) | 0x00ffffffu;
  b.scene.textures.textures.push_back(checker);
  b.scene.textures.slots[2] = static_cast<uint>(b.scene.textures.textures.size() - 1);
  b.scene.instances.back().slots = 2u | (1u << 24);
  b.environment([](float3 d) {
    const float bright = dot(d, normalize(float3(-0.2f, 0.8f, 0.5f))) > 0.94f ? 12.0f : 0.0f;
    return float3(0.05f + 0.3f * max(d.y, 0.0f) + bright);
  }, 128, 64);
  b.camera(float3(0, 1.7f, 4.2f), float3(0, 0.6f, 0), 0.8f, width, height);
  b.finish(4, 0, 11);
}

inline void camera(pt_fixture::Builder &b, pt::uint sequence, pt::uint frame) {
  using namespace pt;
  float time = float(frame) / float(frames - 1);
  if (sequence == 2) time = frame < 4 ? float(frame) / 11 : frame < 8 ? 3.0f / 11 : float(frame - 4) / 11;
  float3 eye(-0.3f + time * 0.6f, 1.7f, 4.2f), target(-0.3f + time * 0.6f, 0.6f, 0);
  if (sequence == 1) { eye = float3(0, 1.7f, 4.2f); target = float3(-0.35f + time * 0.7f, 0.6f, 0); }
  b.camera(eye, target, 0.8f, width, height);
  b.frame.uniforms.image.z = 1;
}
}
