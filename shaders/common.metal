// The rasteriser's structures and helpers; the maths it shares with the path tracer is
// under shared/. Two layout rules from the compiler: a transform
// in a storage buffer is stored as float4 rows (no matrix arrays), and a `constant T&`
// block holds only float4 and float4x4 so it lowers to a uniform buffer.
#pragma once
#include "shared/prelude.h"
#include "shared/types.h"
#include "shared/random.h"
#include "shared/shading.h"

constant uint kCascadeCount = 4u;

struct FrameUniforms {
  float4x4 viewProjection;
  float4x4 view;
  float4x4 inverseViewProjection;
  float4 cameraPosition;  // xyz eye in world space, w exposure
  float4 sunDirection;    // xyz unit vector towards the sun, w angular softness
  float4 sunColor;        // rgb radiance, w intensity multiplier
  float4 environment;     // x IBL intensity, y prefiltered mip count, z ray instance mask, w time
  float4 shadowParameters;// x depth bias, y normal bias, z texel world size, w softness
  float4 cascadeSplits;   // view-space far distance of each cascade
  float4 viewportAndLights; // xy target size, z light count, w debug view
  float4 rays;            // x shadow mode (0 none, 1 cascades, 2 traced), y traced shadow samples,
                          // z sun angular radius, w ambient occlusion mode (0 texture, 1 traced)
  float4 occlusion;       // x traced occlusion radius, y traced occlusion samples, z noise frame (0 when still), w scene radius
  float4 options;         // x punctual lights cast traced shadows, y near plane, z far plane, w lights are clustered
  float4 clusters;        // x tiles across, y tiles down, z depth slices, w lights a cell can hold
  // Irradiance as nine SH coefficients, premultiplied so shIrradiance() returns radiance.
  float4 sh0;
  float4 sh1;
  float4 sh2;
  float4 sh3;
  float4 sh4;
  float4 sh5;
  float4 sh6;
  float4 sh7;
  float4 sh8;
};

// Per-draw record, selected by [[instance_id]] from the draw's first instance: no push constants.
struct Instance {
  float4 modelRow0, modelRow1, modelRow2;
  float4 normalRow0, normalRow1, normalRow2;
  float4 materialAndFlags; // x material index, y visible, zw unused
};

inline float interleavedGradientNoise(float2 pixel) {
  const float3 magic = float3(0.06711056f, 0.00583715f, 52.9829189f);
  return fract(magic.z * fract(dot(pixel, magic.xy)));
}

inline float animatedNoise(float2 pixel, float frame) {
  return fract(interleavedGradientNoise(pixel) + frameRotation(frame).x);
}

// Ramamoorthi-Hanrahan irradiance from SH, clamped: a bright sun rings below zero.
inline float3 shIrradiance(float3 n, float4 c0, float4 c1, float4 c2, float4 c3, float4 c4, float4 c5,
                           float4 c6, float4 c7, float4 c8) {
  const float3 result = c0.rgb + c1.rgb * n.y + c2.rgb * n.z + c3.rgb * n.x + c4.rgb * (n.x * n.y) +
                        c5.rgb * (n.y * n.z) + c6.rgb * (3.0f * n.z * n.z - 1.0f) + c7.rgb * (n.x * n.z) +
                        c8.rgb * (n.x * n.x - n.y * n.y);
  return max(float3(0.0f), result);
}

// The inverse must be the jittered one the depth was rendered with.
inline float3 reconstructWorld(float2 uv, float depth, float4x4 inverseViewProjection) {
  const float4 clip = float4(uv.x * 2.0f - 1.0f, uv.y * 2.0f - 1.0f, depth, 1.0f);
  const float4 world = inverseViewProjection * clip;
  return world.xyz / world.w;
}

// Alpha test for a masked candidate, so rays pass through cut-outs.
inline bool candidateIsSolid(uint instance, uint primitive, float2 barycentric,
                             const device TraceInstance* traceInstances, const device Material* materials,
                             const device uint* indices, const device float* vertices,
                             array<texture2d<float>, kHitTextureSlots> maps, sampler materialSampler) {
  const TraceInstance info = traceInstances[instance];
  const Material material = materials[info.material];
  if (material.alpha.y < 0.5f || material.alpha.y > 1.5f) return true;
  const uint base = info.firstIndex + primitive * 3u;
  const uint i0 = indices[base] + info.vertexOffset;
  const uint i1 = indices[base + 1u] + info.vertexOffset;
  const uint i2 = indices[base + 2u] + info.vertexOffset;
  const float b0 = 1.0f - barycentric.x - barycentric.y;
  const float2 uv = readUV(vertices, i0) * b0 + readUV(vertices, i1) * barycentric.x +
                    readUV(vertices, i2) * barycentric.y;
  const float alpha = maps[info.slots & 0xFFu].sample(materialSampler, uv, level(0.0f)).a *
                      material.baseColorFactor.a;
  return alpha >= material.alpha.x;
}

// Narkowicz's ACES fit.
inline float3 tonemapACES(float3 colour) {
  const float3 x = colour * 0.6f;
  const float3 numerator = x * (x * 2.51f + float3(0.03f));
  const float3 denominator = x * (x * 2.43f + float3(0.59f)) + float3(0.14f);
  return saturate(numerator / denominator);
}

inline float3 tonemapReinhard(float3 colour, float whitePoint) {
  const float3 numerator = colour * (float3(1.0f) + colour / (whitePoint * whitePoint));
  return numerator / (float3(1.0f) + colour);
}

// Face order +X, -X, +Y, -Y, +Z, -Z.
inline float3 cubeDirection(uint face, float2 uv) {
  const float2 c = uv * 2.0f - 1.0f;
  float3 direction = float3(1.0f, -c.y, -c.x);
  if (face == 1u) direction = float3(-1.0f, -c.y, c.x);
  else if (face == 2u) direction = float3(c.x, 1.0f, c.y);
  else if (face == 3u) direction = float3(c.x, -1.0f, -c.y);
  else if (face == 4u) direction = float3(c.x, -c.y, 1.0f);
  else if (face == 5u) direction = float3(-c.x, -c.y, -1.0f);
  return normalize(direction);
}

