#pragma once
#include "path.h"

// Wavefront queue ABI (V6.1). A frame's paths are numbered pixel-major: path g is sample
// g % S of pixel g / S, S being the frame's samples per pixel. Paths run in batches of up to
// `capacity` consecutive g, capacity a multiple of S, so each pixel's samples share one batch
// and the resolve sums them in sample order without atomics. Batch-local index i = g - base
// addresses the per-path arrays (results, costs); queue slots are compacted each bounce and
// carry i. Camera rays are regenerated from g at bounce 0, so no state is stored for them.
//
// Counters: three 8-uint groups, [0] continuation queue parity 0, [8] parity 1, [16] shadow
// queue. In each: x = dispatch groups of 64, 1, 1, then count, 1, 1 (the indirect trace
// width, height, depth), two reserved. [24] counts, for the whole frame, reservations a full
// queue refused; the host reads it back and any nonzero value fails the frame's capture.
// Control: [0] = parity, queue capacity, batch base g, batch count; [1] = bounce, samples per
// pixel S, flags (bit 0: accumulate this frame), first pixel. Bounce 0 processes the batch
// count the host wrote; later stages clamp live counts to the queue capacity.

PT_CONSTANT uint kWaveGroup = 64u;
PT_CONSTANT uint kWaveQueueCount = 3u;       // uint offset of a group's count
PT_CONSTANT uint kWaveFrameOverflow = 24u;   // uint index of the frame's refused reservations
PT_CONSTANT uint kWaveShadowGroup = 16u;     // uint offset of the shadow group

// A continuing path. origin.w: BSDF pdf of the sampled direction; direction.w: the path's
// batch-local index (as uint); throughput.w: the path's ray cone (ptWavePackCone; the bounce is
// the control word's).
struct PtWaveState {
  float4 origin;
  float4 direction;
  float4 throughput;
};

// The nearest hit of one queue slot. flags: bit 0 found, bit 1 a blended candidate was seen.
struct PtWaveHit {
  float t;
  float barycentricX;
  float barycentricY;
  uint instance;
  uint primitive;
  uint flags;
};

// One path's outcome: radiance, and the first hit's albedo and shading normal (the denoiser
// guides). Written whole at bounce 0; radiance is updated only by nonzero contributions.
struct PtWaveResult {
  float4 radiance;
  float4 albedo;
  float4 normal;
};

// A next-event shadow ray. originReach.w: the finite extent; direction.w: the path's
// batch-local index; contribution.w: the shadow ray's cone (ptWavePackCone). The any-hit seed
// is derived from the path index and the bounce, as the shade stage derives it.
struct PtWaveShadow {
  float4 originReach;
  float4 direction;
  float4 contribution;
};

// Guide-collecting path only (the last sample of a pixel, temporal display on): its
// continuation and pending-shadow diffuse fractions and its diffuse radiance, per pixel.
struct PtWaveGuide {
  float4 diffuseFraction;
  float4 shadowFraction;
  float4 diffuseRadiance;
};

// PT_WAVE_RESERVE(counters, group, capacity, slot) reserves one slot of queue group
// `group` for the calling lane (a macro: the compiler keeps atomics in the entry). A slot >=
// capacity was not reserved and is counted as overflow. With PT_WAVE_SUBGROUP_ALLOCATION the
// active lanes of a subgroup reserve together, in lane order: one atomic add and one atomic
// max per subgroup instead of per path. Only the lanes that enqueue may expand it.
#ifdef PT_WAVE_SUBGROUP_ALLOCATION
#define PT_WAVE_RESERVE(counters, group, capacity, slot)                                          \
  {                                                                                              \
    const uint ptOffset = simd_prefix_exclusive_sum(1u);                                         \
    const uint ptTotal = simd_sum(1u);                                                           \
    uint ptBase = 0u;                                                                            \
    if (simd_is_first()) {                                                                       \
      ptBase = atomic_fetch_add_explicit(&counters[(group) + kWaveQueueCount], ptTotal,          \
                                         memory_order_relaxed);                                  \
      const uint ptEnd = min(ptBase + ptTotal, capacity);                                        \
      if (ptEnd > ptBase)                                                                        \
        atomic_fetch_max_explicit(&counters[group], (ptEnd + kWaveGroup - 1u) / kWaveGroup,      \
                                  memory_order_relaxed);                                         \
      if (ptBase + ptTotal > capacity)                                                           \
        atomic_fetch_add_explicit(&counters[kWaveFrameOverflow],                       \
                                  ptBase + ptTotal - max(ptBase, capacity), memory_order_relaxed); \
    }                                                                                            \
    slot = simd_broadcast_first(ptBase) + ptOffset;                                              \
  }
#else
#define PT_WAVE_RESERVE(counters, group, capacity, slot)                                          \
  {                                                                                              \
    slot = atomic_fetch_add_explicit(&counters[(group) + kWaveQueueCount], 1u, memory_order_relaxed); \
    if (slot < capacity)                                                                         \
      atomic_fetch_max_explicit(&counters[group], (slot + kWaveGroup) / kWaveGroup,              \
                                memory_order_relaxed);                                           \
    else                                                                                         \
      atomic_fetch_add_explicit(&counters[kWaveFrameOverflow], 1u, memory_order_relaxed); \
  }
#endif

inline uint ptWaveAsUint(float value) { return as_type<uint>(value); }
// A cone as two bfloat16 halves (width high, spread low): the upper 16 bits of each float,
// rounded, so the width keeps its full range and the level-zero sentinel (-1, 0) is exact. No
// half types: those would need the shaderFloat16 device feature.
inline float ptWavePackCone(float2 cone) {
  const uint width = (as_type<uint>(cone.x) + 0x8000u) & 0xFFFF0000u;
  const uint spread = (as_type<uint>(cone.y) + 0x8000u) >> 16u;
  return as_type<float>(width | spread);
}
inline float2 ptWaveUnpackCone(float packed) {
  const uint bits = as_type<uint>(packed);
  return float2(as_type<float>(bits & 0xFFFF0000u), as_type<float>(bits << 16u));
}
inline float ptWaveAsFloat(uint value) { return as_type<float>(value); }

// The camera ray of a pixel's sample, exactly as integrator.inc derives it, lens included.
inline float3 ptWaveCameraRay(constant PathUniforms &uniforms, uint pixelX, uint pixelY, uint pathSeedValue,
                              thread float3 &origin) {
  const float4 jitter = pathRandom4(pathSeedValue, 0u, 0u);
  const float ndcX = (float(pixelX) + jitter.x) / uniforms.image.x * 2.0f - 1.0f;
  const float ndcY = 1.0f - (float(pixelY) + jitter.y) / uniforms.image.y * 2.0f;
  float3 direction =
      normalize(uniforms.cameraForward.xyz + uniforms.cameraRight.xyz * ndcX + uniforms.cameraUp.xyz * ndcY);
  origin = uniforms.cameraPosition.xyz;
  ptApplyThinLens(uniforms.lens, uniforms.cameraForward.xyz, uniforms.cameraRight.xyz, uniforms.cameraUp.xyz,
                  float2(jitter.z, jitter.w), origin, direction);
  return direction;
}
