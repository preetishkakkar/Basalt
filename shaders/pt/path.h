// The path tracer's shared types and parameter lists. Compiled as MSL by msl2spirv and as
// C++ by MSVC (shared/prelude.h has the rules this code keeps).
//
// Each backend defines, before including the headers in this directory:
//   PT_TEXTURE_PARAMS / PT_TEXTURE_ARGS       the hit texture table
//   PT_ENVIRONMENT_PARAMS / PT_ENVIRONMENT_ARGS the environment's equirect
//   float4 ptSampleTexture(PT_TEXTURE_PARAMS, uint slot, float2 uv, uint samplerCode)
//   float3 ptSampleEnvironment(PT_ENVIRONMENT_PARAMS, float2 uv)
#pragma once
#include "../shared/prelude.h"
#include "../shared/types.h"
#include "../shared/random.h"
#include "../shared/shading.h"

// Everything a path needs besides geometry. Four-lane fields keep the C++ and MSL
// layouts identical. Values whose individual bits matter stay integer all the way to
// the shader: float cannot distinguish adjacent uints above 2^24.
struct PathUniforms {
  float4 cameraPosition;  // xyz the eye
  float4 cameraForward;   // xyz unit forward
  float4 cameraRight;     // xyz right, scaled to the half-width of the image at unit distance
  float4 cameraUp;        // xyz up, scaled to the half-height of the image at unit distance
  float4 image;           // x width, y height, z 0 raw / 1 guides / 2 node cost / 3 triangle cost, w pass samples
  float4 path;            // x maximum bounces, y Russian roulette from this bounce, z firefly clamp (0 off),
                          // w strategy: 0 MIS, 1 BSDF sampling only, 2 light sampling only
  float4 environment;     // x intensity, y equirect width, z equirect height, w draw it behind the scene
  float4 distribution;    // x columns, y rows, z integral, w present
  float4 sunDirection;    // xyz unit vector towards the sun, w cosine of its angular radius (0: no sun)
  float4 sunRadiance;     // rgb radiance of the disc, w its solid angle
  uint4 counts;           // x punctual lights, y ray mask, z seed, w first sample index
  uint4 emissive;         // x emissive triangle count, yzw reserved
  float4 lens;            // x aperture radius (0 pinhole), y focus distance along the view axis,
                          // z texture filter (0 level-zero bilinear, 1 ray cones), w pixel spread angle
  uint4 estimator;        // x direct light (0 NEE, 1 ReSTIR DI), y ReSTIR candidates, z reuse bits, w frame
};

// Shirley and Chiu's concentric map of the unit square onto the unit disk.
inline float2 ptConcentricDisk(float2 u) {
  const float2 offset = u * 2.0f - float2(1.0f);
  if (offset.x == 0.0f && offset.y == 0.0f) return float2(0.0f);
  float radius = offset.y;
  float angle = kPi * 0.5f - (kPi / 4.0f) * (offset.x / offset.y);
  if (abs(offset.x) > abs(offset.y)) {
    radius = offset.x;
    angle = (kPi / 4.0f) * (offset.y / offset.x);
  }
  return float2(cos(angle), sin(angle)) * radius;
}

// Thin-lens depth of field. The pinhole ray
// through the pixel sample fixes the focus point at the focus distance along the view axis;
// the ray then starts at a lens point from the concentric map of `xi` (random group 0 of
// bounce 0, lanes zw) and passes through that point. Aperture 0 leaves the ray untouched.
inline void ptApplyThinLens(float4 lens, float3 forward, float3 right, float3 up, float2 xi, thread float3 &origin,
                            thread float3 &direction) {
  if (lens.x > 0.0f) {
    const float3 focus = origin + direction * (lens.y / dot(direction, forward));
    const float2 disk = ptConcentricDisk(xi) * lens.x;
    origin = origin + normalize(right) * disk.x + normalize(up) * disk.y;
    direction = normalize(focus - origin);
  }
}

// Compact world-space area-light metadata. Textured radiance is evaluated through the
// ordinary surface path at the sampled barycentrics; the CDF only chooses a triangle.
struct PtEmissiveTriangle {
  float4 v0Area;           // xyz first vertex, w area
  float4 edge1Probability; // xyz edge 1, w discrete selection probability
  float4 edge2Cdf;         // xyz edge 2, w inclusive normalized CDF
  float4 normal;           // xyz geometric world normal
  float4 uv01;             // sampled emissive UV at vertices 0 and 1
  float4 uv2;              // xy sampled emissive UV at vertex 2
  uint4 identity;          // x instance, y primitive, zw reserved
};

// A ray's closest hit, in the terms the hardware reports it: the instance, the triangle
// within the instance's primitive, and the barycentrics of the second and third vertices.
struct PtHit {
  uint ambiguous; // a stochastic blended candidate was encountered, even if rejected
  float t;
  float2 barycentric;
  uint instance;
  uint primitive;
  uint found;
  uint nodeVisits;     // software-BVH diagnostic; zero for other intersectors
  uint triangleTests;  // software-BVH diagnostic; zero for other intersectors
};

#define PT_SCENE_PARAMS                                                                           \
  device const TraceInstance *traceInstances, device const Material *materials,                   \
      device const uint *indices, device const float *vertices
#define PT_SCENE_ARGS traceInstances, materials, indices, vertices

PT_CONSTANT float kPtInfinity = 3.0e38f;

inline float ptMaxComponent(float3 v) { return max(max(v.x, v.y), v.z); }
inline float ptLuminance(float3 v) { return dot(v, float3(0.2126f, 0.7152f, 0.0722f)); }

// Power heuristic in ratio form, so a near-delta pdf cannot overflow its square.
inline float ptPowerHeuristic(float chosen, float other) {
  if (chosen <= 0.0f) return 0.0f;
  const float ratio = other / chosen;
  return 1.0f / (1.0f + ratio * ratio);
}

// Scales a contribution down to the clamp; a limit of zero leaves it alone.
inline float3 ptClampContribution(float3 contribution, float limit) {
  float3 result = contribution;
  const float largest = ptMaxComponent(contribution);
  if (limit > 0.0f && largest > limit) result = contribution * (limit / largest);
  return result;
}

// Waechter and Binder's offset along the geometric normal, in units of the position's own
// floating-point spacing, so it holds at any scene scale; near the origin, where that
// spacing vanishes, a small fixed step instead.
inline float3 ptOffsetRay(float3 position, float3 normal) {
  const int3 steps = int3(normal * 256.0f);
  float3 result = as_type<float3>(as_type<int3>(position) + steps * int3(sign(position)));
  const float3 nudged = position + normal * (1.0f / 65536.0f);
  if (abs(position.x) < 1.0f / 32.0f) result.x = nudged.x;
  if (abs(position.y) < 1.0f / 32.0f) result.y = nudged.y;
  if (abs(position.z) < 1.0f / 32.0f) result.z = nudged.z;
  return result;
}

// A 32-bit seed for a trace's stochastic decisions (blended surfaces), distinct per bounce.
inline uint ptTraceSeed(uint pathSeedValue, uint bounce, uint purpose) {
  return pcgHash(pathSeedValue ^ pcgHash(bounce * 16u + purpose + 0x51ED270Bu));
}
