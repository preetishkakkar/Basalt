// Shading maths shared by the rasteriser and the path tracer: the BRDF terms, light
// falloff, vertex reads and the sampling routines.
#pragma once
#include "prelude.h"
#include "types.h"

PT_CONSTANT float kPi = 3.14159265358979323846f;
PT_CONSTANT float kMinRoughness = 0.045f;

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

// Equirectangular lookup for a direction, with v = 0 at the top of the image.
inline float2 equirectangularUV(float3 direction) {
  const float u = atan2(direction.z, direction.x) / (2.0f * kPi) + 0.5f;
  const float v = acos(clamp(direction.y, -1.0f, 1.0f)) / kPi;
  return float2(u, v);
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
