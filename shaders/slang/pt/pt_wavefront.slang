// Wavefront queue ABI. A frame's paths are numbered pixel-major: path g is sample
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
module pt_wavefront;

import pt_path;

public static const uint kWaveGroup = 64u;
public static const uint kWaveQueueCount = 3u;     // uint offset of a group's count
public static const uint kWaveFrameOverflow = 24u; // uint index of the frame's refused reservations
public static const uint kWaveShadowGroup = 16u;   // uint offset of the shadow group

// A continuing path. origin.w: BSDF pdf of the sampled direction; direction.w: the path's
// batch-local index (as uint); throughput.w: the path's ray cone (ptWavePackCone; the bounce is
// the control word's).
public struct PtWaveState {
  float4 origin;
  float4 direction;
  float4 throughput;
};

// The nearest hit of one queue slot. flags: bit 0 found, bit 1 a blended candidate was seen.
public struct PtWaveHit {
  float t;
  float barycentricX;
  float barycentricY;
  uint instance;
  uint primitive;
  uint flags;
};

// One path's outcome: radiance, and the first hit's albedo and shading normal (the denoiser
// guides). Written whole at bounce 0; radiance is updated only by nonzero contributions.
public struct PtWaveResult {
  float4 radiance;
  float4 albedo;
  float4 normal;
};

// A next-event shadow ray. originReach.w: the finite extent; direction.w: the path's
// batch-local index; contribution.w: the shadow ray's cone (ptWavePackCone). The any-hit seed
// is derived from the path index and the bounce, as the shade stage derives it.
public struct PtWaveShadow {
  float4 originReach;
  float4 direction;
  float4 contribution;
};

// Guide-collecting path only (the last sample of a pixel, temporal display on): its
// continuation and pending-shadow diffuse fractions and its diffuse radiance, per pixel.
public struct PtWaveGuide {
  float4 diffuseFraction;
  float4 shadowFraction;
  float4 diffuseRadiance;
};

// A cone as two bfloat16 halves (width high, spread low): the upper 16 bits of each float,
// rounded, so the width keeps its full range and the level-zero sentinel (-1, 0) is exact. No
// half types: those would need the shaderFloat16 device feature.
public float ptWavePackCone(float2 cone) {
  const uint width = (asuint(cone.x) + 0x8000u) & 0xFFFF0000u;
  const uint spread = (asuint(cone.y) + 0x8000u) >> 16u;
  return asfloat(width | spread);
}
public float2 ptWaveUnpackCone(float packed) {
  const uint bits = asuint(packed);
  return float2(asfloat(bits & 0xFFFF0000u), asfloat(bits << 16u));
}
