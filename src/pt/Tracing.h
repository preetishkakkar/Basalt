// The path tracer's shared code (shaders/pt) compiled for the CPU: the host copies of the
// hit texture table and the environment that the shared code samples, then the headers
// themselves inside namespace pt.
#pragma once
#include "pt/Shared.h"

#include <array>
#include <cstdint>
#include <vector>

namespace pt {

// One texture of the hit table at level zero, sampled as the GPU's table sampler samples
// it: bilinear, repeating, after decoding sRGB.
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

} // namespace pt

#define PT_TEXTURE_PARAMS const HostTextures &maps
#define PT_TEXTURE_ARGS maps
#define PT_ENVIRONMENT_PARAMS const HostEnvironment &environmentMap
#define PT_ENVIRONMENT_ARGS environmentMap

namespace pt {
inline float4 ptSampleTexture(PT_TEXTURE_PARAMS, uint slot, float2 uv, uint samplerCode, float lodBase) {
  return maps.sample(slot, uv, samplerCode, lodBase);
}
inline float ptTextureFootprintOf(PT_TEXTURE_PARAMS, uint slot, float lodBase) { return maps.footprint(slot, lodBase); }
inline float3 ptSampleEnvironment(PT_ENVIRONMENT_PARAMS, float2 uv) { return environmentMap.sample(uv); }
} // namespace pt

#define device
#define thread
namespace pt {
#include "../../shaders/pt/path.h"
#include "../../shaders/pt/raycone.h"
#include "../../shaders/pt/bvh.h"
#include "../../shaders/pt/surface.h"
#include "../../shaders/pt/bsdf.h"
#include "../../shaders/pt/reconstruction.h"
#include "../../shaders/pt/temporal.h"
#include "../../shaders/pt/lights.h"
#include "../../shaders/pt/restir.h"
} // namespace pt
#undef device
#undef thread

namespace pt {
static_assert(sizeof(PathUniforms) == 224, "PathUniforms must match the MSL constant-buffer layout");
static_assert(sizeof(PtEmissiveTriangle) == 112, "emissive triangle layout must match MSL");
static_assert(sizeof(PtReconstructionSample) == 128, "Reconstruction sample must match MSL");
static_assert(sizeof(PtReservoir) == 32, "ReSTIR reservoir must match MSL");
static_assert(sizeof(PtRestirSurface) == 64, "ReSTIR surface must match MSL");
}
