// A hit turned into a shading point: position, normals and the material's values there,
// read the way the rasteriser's forward pass reads them.
#pragma once
#include "path.h"
#include "raycone.h"

struct PtSurface {
  float3 position;
  float3 geometricNormal;  // faces the incoming ray
  float3 normal;           // shading normal, after the normal map, on the same side
  float3 baseColor;
  float3 emissive;
  float metallic;
  float roughness;
  // transmission.x factor, y IOR, z 1 thin-walled /
  // 0 closed dielectric, w 1 when the ray arrived from the front (outside) face.
  float4 transmission;
  float4 clearcoat;          // x factor, y roughness (>= kMinRoughness), zw unused
  float3 clearcoatNormal;    // the interpolated (or clearcoat-mapped) normal, on the ray's side
};

// The material's IOR (KHR_materials_ior); unset (0) reads as the glTF default 1.5. It sets
// the dielectric F0 even without transmission.
inline float ptMaterialIor(float4 transmission) {
  return transmission.y > 0.0f ? max(transmission.y, 1.0f) : 1.5f;
}

// A surface without V7 layers: no transmission, no clearcoat, the material's IOR.
inline void ptClearV7Layers(thread PtSurface &surface, float ior, float entering) {
  surface.transmission = float4(0.0f, ior, 1.0f, entering);
  surface.clearcoat = float4(0.0f, kMinRoughness, 0.0f, 0.0f);
  surface.clearcoatNormal = surface.normal;
}

#define PT_VERTEX_UV(selector)                                                                     \
  mix(float2(dot(weights, float3(vertices[v0 + 10u], vertices[v1 + 10u], vertices[v2 + 10u])),    \
             dot(weights, float3(vertices[v0 + 11u], vertices[v1 + 11u], vertices[v2 + 11u]))),   \
      float2(dot(weights, float3(vertices[v0 + 12u], vertices[v1 + 12u], vertices[v2 + 12u])),    \
             dot(weights, float3(vertices[v0 + 13u], vertices[v1 + 13u], vertices[v2 + 13u]))),   \
      float((selector) == 1u))

// coneWidth: the ray cone's width at the hit (ptConeWidthOrLevelZero); -1 samples level zero.
inline PtSurface ptSurfaceAt(PT_SCENE_PARAMS, PT_TEXTURE_PARAMS, PtHit hit, float3 rayDirection, float coneWidth) {
  const uint instance = hit.instance;
  const uint material = traceInstances[instance].material;
  const uint slots = traceInstances[instance].slots;
  const uint vertexOffset = traceInstances[instance].vertexOffset;
  const uint base = traceInstances[instance].firstIndex + hit.primitive * 3u;
  const uint v0 = (indices[base] + vertexOffset) * kVertexFloats;
  const uint v1 = (indices[base + 1u] + vertexOffset) * kVertexFloats;
  const uint v2 = (indices[base + 2u] + vertexOffset) * kVertexFloats;
  const float3 weights = float3(1.0f - hit.barycentric.x - hit.barycentric.y, hit.barycentric.x, hit.barycentric.y);

  // Vertex fields: position 0, normal 3, tangent 6, UV0 10, UV1 12, colour 14.
  const float3 p0 = float3(vertices[v0], vertices[v0 + 1u], vertices[v0 + 2u]);
  const float3 p1 = float3(vertices[v1], vertices[v1 + 1u], vertices[v1 + 2u]);
  const float3 p2 = float3(vertices[v2], vertices[v2 + 1u], vertices[v2 + 2u]);
  // Ray-cone level of detail per UV set: the same world-space triangle, each set's
  // UVs. Level-zero mode (coneWidth < 0) skips the loads and arithmetic.
  float lod0 = kPtLevelZero, lod1 = kPtLevelZero;
  if (coneWidth >= 0.0f) {
    const float3 worldCross = ptWorldCross(traceInstances[instance].objectToWorld0, traceInstances[instance].objectToWorld1,
                                           traceInstances[instance].objectToWorld2, p0, p1, p2);
    lod0 = ptConeLodBase(ptUvCross(float2(vertices[v0 + 10u], vertices[v0 + 11u]),
                                   float2(vertices[v1 + 10u], vertices[v1 + 11u]),
                                   float2(vertices[v2 + 10u], vertices[v2 + 11u])),
                         worldCross, rayDirection, coneWidth);
    lod1 = ptConeLodBase(ptUvCross(float2(vertices[v0 + 12u], vertices[v0 + 13u]),
                                   float2(vertices[v1 + 12u], vertices[v1 + 13u]),
                                   float2(vertices[v2 + 12u], vertices[v2 + 13u])),
                         worldCross, rayDirection, coneWidth);
  }
// "+ 0.0f" makes the operands values: msl2spirv has no conditional lvalues (exact for any lod).
#define PT_VERTEX_LOD(selector) ((selector) == 1u ? lod1 + 0.0f : lod0 + 0.0f)
  const float3 objectNormal =
      float3(dot(weights, float3(vertices[v0 + 3u], vertices[v1 + 3u], vertices[v2 + 3u])),
             dot(weights, float3(vertices[v0 + 4u], vertices[v1 + 4u], vertices[v2 + 4u])),
             dot(weights, float3(vertices[v0 + 5u], vertices[v1 + 5u], vertices[v2 + 5u])));
  const float4 objectTangent =
      float4(dot(weights, float3(vertices[v0 + 6u], vertices[v1 + 6u], vertices[v2 + 6u])),
             dot(weights, float3(vertices[v0 + 7u], vertices[v1 + 7u], vertices[v2 + 7u])),
             dot(weights, float3(vertices[v0 + 8u], vertices[v1 + 8u], vertices[v2 + 8u])),
             dot(weights, float3(vertices[v0 + 9u], vertices[v1 + 9u], vertices[v2 + 9u])));

  PtSurface surface;
  // The position from the vertices, not origin + t * direction: it lies on the triangle to
  // the precision of its own coordinates, which the ray offset relies on.
  surface.position = applyRows(traceInstances[instance].objectToWorld0, traceInstances[instance].objectToWorld1,
                               traceInstances[instance].objectToWorld2, p0 * weights.x + p1 * weights.y + p2 * weights.z);
  const float4 n0 = traceInstances[instance].normalToWorld0;
  const float4 n1 = traceInstances[instance].normalToWorld1;
  const float4 n2 = traceInstances[instance].normalToWorld2;
  const float3 face = cross(p1 - p0, p2 - p0);
  float3 geometric = normalize(float3(dot(xyz(n0), face), dot(xyz(n1), face), dot(xyz(n2), face)));
  float3 normal = float3(dot(xyz(n0), objectNormal), dot(xyz(n1), objectNormal), dot(xyz(n2), objectNormal));
  if (dot(normal, normal) > 1e-12f) normal = normalize(normal);
  else normal = geometric;
  // Rays hit either side; the shading frame faces the ray, as the forward pass faces a
  // double-sided back face towards the eye.
  // Arrived from the front (the face the winding and determinant make front) or the back.
  const float entering = dot(geometric, rayDirection) < 0.0f ? 1.0f : 0.0f;
  if (dot(geometric, rayDirection) > 0.0f) geometric = -geometric;
  if (dot(normal, geometric) < 0.0f) normal = -normal;
  surface.clearcoatNormal = normal;  // before the base normal map, as glTF specifies

  float3 tangent = applyRowsToDirection(traceInstances[instance].objectToWorld0, traceInstances[instance].objectToWorld1,
                                        traceInstances[instance].objectToWorld2, xyz(objectTangent));
  tangent = tangent - normal * dot(normal, tangent);
  if (dot(tangent, tangent) > 1e-12f) {
    tangent = normalize(tangent);
    float handedness = 1.0f;
    if (objectTangent.w < 0.0f) handedness = -1.0f;
    const float3 sampled = xyz(ptSampleTexture(PT_TEXTURE_ARGS, (slots >> 24u) & 0xFFu,
                                               PT_VERTEX_UV((materials[material].texture.x >> 24u) & 0xFFu),
                                               (materials[material].texture.y >> 24u) & 0xFFu, PT_VERTEX_LOD((materials[material].texture.x >> 24u) & 0xFFu))) * 2.0f - 1.0f;
    const float scale = materials[material].factors.z;
    const float3 perturbed = normalize(tangent * (sampled.x * scale) +
                                       cross(normal, tangent) * (sampled.y * scale * handedness) + normal * sampled.z);
    // A normal map may tilt the normal past the geometric horizon; keep the unperturbed one then.
    if (dot(perturbed, geometric) > 0.0f) normal = perturbed;
  }
  surface.geometricNormal = geometric;
  surface.normal = normal;

  const float4 metallicRoughness = ptSampleTexture(PT_TEXTURE_ARGS, (slots >> 8u) & 0xFFu,
                                                   PT_VERTEX_UV((materials[material].texture.x >> 8u) & 0xFFu),
                                                   (materials[material].texture.y >> 8u) & 0xFFu, PT_VERTEX_LOD((materials[material].texture.x >> 8u) & 0xFFu));
  surface.baseColor = xyz(ptSampleTexture(PT_TEXTURE_ARGS, slots & 0xFFu,
                                          PT_VERTEX_UV(materials[material].texture.x & 0xFFu),
                                          materials[material].texture.y & 0xFFu, PT_VERTEX_LOD(materials[material].texture.x & 0xFFu)) *
                          materials[material].baseColorFactor *
                          float4(dot(weights, float3(vertices[v0 + 14u], vertices[v1 + 14u], vertices[v2 + 14u])),
                                 dot(weights, float3(vertices[v0 + 15u], vertices[v1 + 15u], vertices[v2 + 15u])),
                                 dot(weights, float3(vertices[v0 + 16u], vertices[v1 + 16u], vertices[v2 + 16u])),
                                 dot(weights, float3(vertices[v0 + 17u], vertices[v1 + 17u], vertices[v2 + 17u]))));
  surface.emissive = xyz(ptSampleTexture(PT_TEXTURE_ARGS, (slots >> 16u) & 0xFFu,
                                         PT_VERTEX_UV((materials[material].texture.x >> 16u) & 0xFFu),
                                         (materials[material].texture.y >> 16u) & 0xFFu, PT_VERTEX_LOD((materials[material].texture.x >> 16u) & 0xFFu))) * xyz(materials[material].emissive) *
                     materials[material].emissive.w;
  surface.metallic = saturate(materials[material].factors.x * metallicRoughness.z);
  surface.roughness = clamp(materials[material].factors.y * metallicRoughness.y, kMinRoughness, 1.0f);
  const uint4 extension = materials[material].extensionTextures;
  surface.transmission = float4(0.0f, ptMaterialIor(materials[material].transmission), 1.0f, entering);
  if (materials[material].transmission.x > 0.0f)
    surface.transmission = float4(saturate(materials[material].transmission.x *
                                           ptSampleTexture(PT_TEXTURE_ARGS, extension.x & 0xFFu,
                                                           PT_VERTEX_UV((extension.x >> 8u) & 0xFFu),
                                                           (extension.x >> 16u) & 0xFFu, PT_VERTEX_LOD((extension.x >> 8u) & 0xFFu)).x),
                                  surface.transmission.y, materials[material].transmission.z > 0.0f ? 0.0f : 1.0f,
                                  entering);
  surface.clearcoat = float4(0.0f, kMinRoughness, 0.0f, 0.0f);
  if (materials[material].clearcoat.x > 0.0f) {
    surface.clearcoat.x = saturate(materials[material].clearcoat.x *
                                   ptSampleTexture(PT_TEXTURE_ARGS, extension.y & 0xFFu,
                                                   PT_VERTEX_UV((extension.y >> 8u) & 0xFFu), (extension.y >> 16u) & 0xFFu, PT_VERTEX_LOD((extension.y >> 8u) & 0xFFu)).x);
    surface.clearcoat.y = clamp(materials[material].clearcoat.y *
                                    ptSampleTexture(PT_TEXTURE_ARGS, extension.z & 0xFFu,
                                                    PT_VERTEX_UV((extension.z >> 8u) & 0xFFu), (extension.z >> 16u) & 0xFFu, PT_VERTEX_LOD((extension.z >> 8u) & 0xFFu)).y,
                                kMinRoughness, 1.0f);
    if (dot(tangent, tangent) > 1e-12f && (extension.w & 0xFFu) != 1u) {
      const float3 coatSample = xyz(ptSampleTexture(PT_TEXTURE_ARGS, extension.w & 0xFFu,
                                                    PT_VERTEX_UV((extension.w >> 8u) & 0xFFu),
                                                    (extension.w >> 16u) & 0xFFu, PT_VERTEX_LOD((extension.w >> 8u) & 0xFFu))) * 2.0f - 1.0f;
      const float coatScale = materials[material].clearcoat.z;
      const float3 coatNormal = normalize(tangent * (coatSample.x * coatScale) +
                                          cross(surface.clearcoatNormal, tangent) * (coatSample.y * coatScale *
                                                                                     (objectTangent.w < 0.0f ? -1.0f : 1.0f)) +
                                          surface.clearcoatNormal * coatSample.z);
      if (dot(coatNormal, geometric) > 0.0f) surface.clearcoatNormal = coatNormal;
    }
  }
  return surface;
}

#undef PT_VERTEX_UV
#undef PT_VERTEX_LOD
