// The CPU tracer's side of the shared Slang code: the host copies of the hit texture table and
// the environment that the generated code samples, and TraceView, a frame as that code sees it.
#pragma once
#include "pt/Shared.h"

#include <array>
#include <cstdint>
#include <vector>

namespace pt {

// One texture of the hit table, sRGB-decoded on read when srgb is set: level zero in texels,
// the smaller levels in mips.
struct HostTexture {
  uint width = 1;
  uint height = 1;
  bool srgb = false;
  std::vector<std::uint32_t> texels{0xFFFFFFFFu};  // RGBA8, red in the low byte
  // Levels 1..n (buildMipChain): 2x2 box filtered in linear space, as the GPU's linear blits.
  struct Level {
    uint width = 1, height = 1;
    std::vector<std::uint32_t> texels;
  };
  std::vector<Level> mips;
};

// Builds a texture's mip chain down to 1x1 (ray-cone filtering).
void buildMipChain(HostTexture &texture);

class HostTextures {
public:
  HostTextures();
  std::vector<HostTexture> textures;               // distinct textures
  std::array<std::uint32_t, kHitTextureSlots> slots{};  // slot -> index into textures
  // lodBase from ptConeLodBase; the default (kPtLevelZero, -1e30) samples level 0 exactly.
  float4 sample(uint slot, float2 uv, uint samplerCode = 0, float lodBase = -1e30f) const;
  float footprint(uint slot, float lodBase) const;  // ptTextureFootprint of the slot's texture
};

// The environment's equirect as the GPU's environment sampler reads it: bilinear, U
// repeating, V clamped.
struct HostEnvironment {
  uint width = 1;
  uint height = 1;
  std::vector<float> texels{0.0f, 0.0f, 0.0f, 0.0f};  // RGBA32F
  float3 sample(float2 uv) const;
};

// A frame as the generated code sees it: `scene` holds the host's arrays as buffers, the
// uniforms and the intersector (0 the binary software BVH); the code calls back through it
// for texture reads and, for any other intersector, to `trace`. Fixed in memory: scene.host is
// this address, and scenes from withInstances carry it, so the view must outlive them.
class TraceView {
public:
  using Trace = PtHit (*)(const TraceView &view, float3 origin, float3 direction, float tMax, uint mask, uint seed,
                          float2 cone, uint anyHit);
  TraceView(const HostTextures &textures, const HostEnvironment *environment = nullptr);
  TraceView(const TraceView &) = delete;
  TraceView &operator=(const TraceView &) = delete;

  PtCpuScene scene{};
  const HostTextures &textures;
  const HostEnvironment *environment;
  // The host intersector for scene.intersector != 0, and what it traces.
  Trace trace = nullptr;
  const void *tracer = nullptr;

  PtCpuScene *get() const { return const_cast<PtCpuScene *>(&scene); }
  // This view's scene with other instance rows, for an intersector with its own (the wide BVH).
  PtCpuScene withInstances(const std::vector<TraceInstance> &instances) const;
};

// The alpha test every intersector applies to masked and blended candidates.
inline bool ptCandidateSolid(const PtCpuScene &scene, uint instance, uint primitive, float2 barycentric, uint frontFacing,
                             uint seed, float3 direction, float coneWidth) {
  return gen::ptCpuCandidateSolid(const_cast<PtCpuScene *>(&scene), instance, primitive, barycentric, frontFacing, seed, direction,
                             coneWidth);
}
// The watertight triangle test, for the host's intersectors.
inline PtRayPrep ptPrepareRay(float3 origin, float3 direction) { return gen::ptCpuPrepareRay(origin, direction); }
inline bool ptIntersectTriangle(const PtRayPrep &ray, float3 v0, float3 v1, float3 v2, float tMax, float &t, float2 &barycentric,
                                uint &candidateFlags) {
  return gen::ptCpuIntersectTriangle(const_cast<PtRayPrep *>(&ray), v0, v1, v2, tMax, &t, &barycentric, &candidateFlags);
}
// The binary software BVH over a view's scene.bvhNodes and bvhTriangles.
inline PtHit ptTraceBvh(const TraceView &view, float3 origin, float3 direction, float tMax, uint mask, uint seed, float2 cone,
                        uint anyHit) {
  return gen::ptCpuTraceBvh(view.get(), origin, direction, tMax, mask, seed, cone, anyHit != 0u);
}
// A pixel's surface at a hit.
inline PtSurface ptSurfaceAt(const TraceView &view, PtHit hit, float3 rayDirection, float coneWidth) {
  return gen::ptCpuSurfaceAt(view.get(), &hit, rayDirection, coneWidth);
}

// ReSTIR DI's resampling over a view's lights and emitters.
inline PtReservoir ptRestirInitial(const TraceView &view, PtBsdf bsdf, float3 position, float3 geometricNormal, float2 cone,
                                   uint seed, uint candidates) {
  return gen::ptCpuRestirInitial(view.get(), &bsdf, position, geometricNormal, cone, seed, candidates);
}
inline float3 ptRestirIntegrand(const TraceView &view, PtBsdf bsdf, float3 position, float3 geometricNormal, uint light,
                                float2 params, float2 cone, float3 &direction, float &reach, float &sourcePdf,
                                float &solidAnglePdf, float &bsdfPdf) {
  return gen::ptCpuRestirIntegrand(view.get(), &bsdf, position, geometricNormal, light, params, cone, &direction, &reach,
                                   &sourcePdf, &solidAnglePdf, &bsdfPdf);
}

} // namespace pt
