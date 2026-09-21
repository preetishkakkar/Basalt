// Shared structures and shading maths. Two layout rules from the compiler: a transform
// in a storage buffer is stored as float4 rows (no matrix arrays), and a `constant T&`
// block holds only float4 and float4x4 so it lowers to a uniform buffer.
#pragma once
#include <metal_stdlib>
using namespace metal;

constant float kPi = 3.14159265358979323846f;
constant float kMinRoughness = 0.045f;
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

// One per top-level instance, indexed by the hit's instance id.
struct PrimitiveInfo {
  uint firstIndex;
  uint vertexOffset;
  uint material;
  uint slots;   // Texture table slots, 8 bits each: base colour, metallic-roughness, emissive, normal.
};
constant uint kHitTextureSlots = 120u; // Bound at texture(7); indices must stay within 0..127.

// Vertex layout as floats: position 3, normal 3, tangent 4, uv0 2, uv1 2.
constant uint kVertexFloats = 14u;

// Per-draw record, selected by [[instance_id]] from the draw's first instance: no push constants.
struct Instance {
  float4 modelRow0, modelRow1, modelRow2;
  float4 normalRow0, normalRow1, normalRow2;
  float4 materialAndFlags; // x material index, y visible, zw unused
};

struct Material {
  float4 baseColorFactor;
  float4 emissive;   // rgb factor, w strength
  float4 factors;    // x metallic, y roughness, z normal scale, w occlusion strength
  float4 alpha;      // x cutoff, y mode (0 opaque, 1 mask, 2 blend), z double sided, w unused
};

struct Light {
  float4 position;   // xyz world position, w range
  float4 color;      // rgb colour, w intensity in candela
  float4 direction;  // xyz spot direction, w inner cone cosine
  float4 cone;       // x outer cone cosine, y type (0 point, 1 spot), zw unused
};

inline float3 applyRows(float4 row0, float4 row1, float4 row2, float3 point) {
  const float4 p = float4(point, 1.0f);
  return float3(dot(row0, p), dot(row1, p), dot(row2, p));
}

inline float3 applyRowsToDirection(float4 row0, float4 row1, float4 row2, float3 direction) {
  const float4 d = float4(direction, 0.0f);
  return float3(dot(row0, d), dot(row1, d), dot(row2, d));
}

inline float distributionGGX(float normalDotHalf, float alpha) {
  const float a2 = alpha * alpha;
  const float d = normalDotHalf * normalDotHalf * (a2 - 1.0f) + 1.0f;
  return a2 / max(kPi * d * d, 1e-7f);
}

// Height-correlated Smith visibility; includes the 1/(4 NdotL NdotV) term.
inline float visibilitySmith(float normalDotView, float normalDotLight, float alpha) {
  const float a2 = alpha * alpha;
  const float v = normalDotLight * sqrt(normalDotView * normalDotView * (1.0f - a2) + a2);
  const float l = normalDotView * sqrt(normalDotLight * normalDotLight * (1.0f - a2) + a2);
  return 0.5f / max(v + l, 1e-7f);
}

inline float3 fresnelSchlick(float3 f0, float viewDotHalf) {
  const float f = pow(saturate(1.0f - viewDotHalf), 5.0f);
  return f0 + (float3(1.0f) - f0) * f;
}

inline float3 fresnelSchlickRoughness(float3 f0, float normalDotView, float roughness) {
  const float f = pow(saturate(1.0f - normalDotView), 5.0f);
  const float3 ceiling = max(float3(1.0f - roughness), f0);
  return f0 + (ceiling - f0) * f;
}

inline float3 shadeDirect(float3 normal, float3 view, float3 lightDirection, float3 radiance,
                          float3 diffuseColor, float3 f0, float roughness) {
  const float3 halfVector = normalize(view + lightDirection);
  const float normalDotLight = saturate(dot(normal, lightDirection));
  const float normalDotView = saturate(dot(normal, view)) + 1e-5f;
  const float normalDotHalf = saturate(dot(normal, halfVector));
  const float viewDotHalf = saturate(dot(view, halfVector));

  const float alpha = roughness * roughness;
  const float3 fresnel = fresnelSchlick(f0, viewDotHalf);
  const float3 specular = fresnel * (distributionGGX(normalDotHalf, alpha) *
                                     visibilitySmith(normalDotView, normalDotLight, alpha));
  const float3 diffuse = (float3(1.0f) - fresnel) * diffuseColor * (1.0f / kPi);
  return (diffuse + specular) * radiance * normalDotLight;
}

inline float distanceAttenuation(float distanceToLight, float range) {
  const float inverseSquare = 1.0f / max(distanceToLight * distanceToLight, 1e-4f);
  if (range <= 0.0f) return inverseSquare;
  const float ratio = distanceToLight / range;
  const float ratio4 = ratio * ratio * ratio * ratio;
  const float window = saturate(1.0f - ratio4);
  return inverseSquare * window * window;
}

// Split-sum environment BRDF, Lazarov's analytic fit: the entry's eight texture slots
// leave no room for a lookup table.
inline float2 environmentBRDF(float roughness, float normalDotView) {
  const float4 c0 = float4(-1.0f, -0.0275f, -0.572f, 0.022f);
  const float4 c1 = float4(1.0f, 0.0425f, 1.04f, -0.04f);
  const float4 r = float4(roughness, roughness, roughness, roughness) * c0 + c1;
  const float a004 = min(r.x * r.x, exp2(-9.28f * normalDotView)) * r.x + r.y;
  return float2(-1.04f, 1.04f) * a004 + float2(r.z, r.w);
}

// Fdez-Aguera compensation for the energy single-scatter GGX loses at high roughness.
struct EnergyCompensation {
  float3 direct;   // Multiplier on the single-scattering specular lobe.
  float3 ambient;  // Extra image-based term, multiplied by the irradiance.
};
inline EnergyCompensation energyCompensation(float3 f0, float2 brdf) {
  const float singleScatter = brdf.x + brdf.y;
  const float3 averageFresnel = f0 + (float3(1.0f) - f0) * (1.0f / 21.0f);
  const float3 multiScatter = averageFresnel * (1.0f - singleScatter) /
                              max(float3(1.0f) - averageFresnel * (1.0f - singleScatter), float3(1e-4f));
  EnergyCompensation result;
  result.direct = float3(1.0f) + multiScatter / max(singleScatter, 1e-4f);
  result.ambient = multiScatter;
  return result;
}

// Lagarde's specular occlusion from ambient occlusion.
inline float specularOcclusion(float normalDotView, float ambientOcclusion, float roughness) {
  return saturate(pow(normalDotView + ambientOcclusion, exp2(-16.0f * roughness - 1.0f)) - 1.0f +
                  ambientOcclusion);
}

// Fades reflections that dip below the geometric surface after normal mapping.
inline float horizonOcclusion(float3 reflection, float3 geometricNormal) {
  const float horizon = saturate(1.0f + dot(reflection, geometricNormal));
  return horizon * horizon;
}

inline float interleavedGradientNoise(float2 pixel) {
  const float3 magic = float3(0.06711056f, 0.00583715f, 52.9829189f);
  return fract(magic.z * fract(dot(pixel, magic.xy)));
}

inline float hashFloat(uint value) {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  value ^= value >> 16u;
  return float(value) * 2.3283064365386963e-10f;
}

// R2 low-discrepancy rotation for a frame, in exact 32-bit fixed point so it neither
// repeats nor loses precision over thousands of frames.
inline float3 frameRotation(float frame) {
  const uint n = uint(frame);
  return float3(float(n * 3242174889u), float(n * 2447445413u), float(n * 2654435769u)) *
         2.3283064365386963e-10f;
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

inline float3 readPosition(const device float* vertices, uint index) {
  const uint base = index * kVertexFloats;
  return float3(vertices[base + 0u], vertices[base + 1u], vertices[base + 2u]);
}
inline float3 readNormal(const device float* vertices, uint index) {
  const uint base = index * kVertexFloats;
  return float3(vertices[base + 3u], vertices[base + 4u], vertices[base + 5u]);
}
inline float4 readTangent(const device float* vertices, uint index) {
  const uint base = index * kVertexFloats;
  return float4(vertices[base + 6u], vertices[base + 7u], vertices[base + 8u], vertices[base + 9u]);
}
inline float2 readUV(const device float* vertices, uint index) {
  const uint base = index * kVertexFloats;
  return float2(vertices[base + 10u], vertices[base + 11u]);
}

// Alpha test for a masked candidate, so rays pass through cut-outs.
inline bool candidateIsSolid(uint instance, uint primitive, float2 barycentric,
                             const device PrimitiveInfo* primitives, const device Material* materials,
                             const device uint* indices, const device float* vertices,
                             array<texture2d<float>, kHitTextureSlots> maps, sampler materialSampler) {
  const PrimitiveInfo info = primitives[instance];
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

inline void orthonormalBasis(float3 normal, thread float3 &tangent, thread float3 &bitangent) {
  const float3 up = abs(normal.y) < 0.999f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
  tangent = normalize(cross(up, normal));
  bitangent = cross(normal, tangent);
}

inline float3 cosineSampleHemisphere(float3 normal, float2 xi) {
  const float radius = sqrt(xi.x);
  const float angle = 2.0f * kPi * xi.y;
  float3 tangent = float3(0.0f), bitangent = float3(0.0f);
  orthonormalBasis(normal, tangent, bitangent);
  const float3 local = float3(radius * cos(angle), radius * sin(angle), sqrt(max(0.0f, 1.0f - xi.x)));
  return normalize(tangent * local.x + bitangent * local.y + normal * local.z);
}

inline float3 sampleCone(float3 axis, float angularRadius, float2 xi) {
  float3 tangent = float3(0.0f), bitangent = float3(0.0f);
  orthonormalBasis(axis, tangent, bitangent);
  const float spread = tan(angularRadius) * sqrt(xi.x);
  const float angle = 2.0f * kPi * xi.y;
  return normalize(axis + tangent * (spread * cos(angle)) + bitangent * (spread * sin(angle)));
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

// Equirectangular lookup for a direction, with v = 0 at the top of the image.
inline float2 equirectangularUV(float3 direction) {
  const float u = atan2(direction.z, direction.x) / (2.0f * kPi) + 0.5f;
  const float v = acos(clamp(direction.y, -1.0f, 1.0f)) / kPi;
  return float2(u, v);
}

inline float radicalInverse(uint bits) {
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return float(bits) * 2.3283064365386963e-10f;
}

inline float2 hammersley(uint index, uint count) {
  return float2(float(index) / float(count), radicalInverse(index));
}

inline float3 importanceSampleGGX(float2 xi, float3 normal, float roughness) {
  const float alpha = roughness * roughness;
  const float phi = 2.0f * kPi * xi.x;
  const float cosTheta = sqrt((1.0f - xi.y) / (1.0f + (alpha * alpha - 1.0f) * xi.y));
  const float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
  const float3 halfVector = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

  const float3 up = abs(normal.z) < 0.999f ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
  const float3 tangent = normalize(cross(up, normal));
  const float3 bitangent = cross(normal, tangent);
  return normalize(tangent * halfVector.x + bitangent * halfVector.y + normal * halfVector.z);
}
