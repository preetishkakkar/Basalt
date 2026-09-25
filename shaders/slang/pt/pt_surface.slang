// A hit turned into a shading point: position, normals and the material's values there, read
// the way the rasteriser's forward pass reads them.
module pt_surface;

import pt_path;
import pt_raycone;

public __extern_cpp struct PtSurface {
  __extern_cpp float3 position;
  __extern_cpp float3 geometricNormal; // faces the incoming ray
  __extern_cpp float3 normal;          // shading normal, after the normal map, on the same side
  __extern_cpp float3 baseColor;
  __extern_cpp float3 emissive;
  __extern_cpp float metallic;
  __extern_cpp float roughness;
  // x transmission factor, y IOR, z 1 thin-walled / 0 closed dielectric, w 1 when the ray
  // arrived from the front (outside) face.
  __extern_cpp float4 transmission;
  __extern_cpp float4 clearcoat;       // x factor, y roughness (>= kMinRoughness), zw unused
  __extern_cpp float3 clearcoatNormal; // the interpolated (or clearcoat-mapped) normal, on the ray's side

  // Without the transmission/clearcoat layers: no transmission, no clearcoat, the material's IOR.
  [mutating]
  void clearLayers(float ior, float entering) {
    transmission = float4(0.0f, ior, 1.0f, entering);
    clearcoat = float4(0.0f, kMinRoughness, 0.0f, 0.0f);
    clearcoatNormal = normal;
  }
};

// The material's IOR (KHR_materials_ior); unset (0) reads as the glTF default 1.5. It sets the
// dielectric F0 even without transmission.
public float ptMaterialIor(float4 transmission) { return transmission.y > 0.0f ? max(transmission.y, 1.0f) : 1.5f; }

// A hit's texture reads: the UV set and level of detail a material's texture selects.
struct HitTexels<M : IPtMaps> {
  M maps;
  float2 uv0, uv1;
  float lod0, lod1;

  float4 read(uint slot, uint uvSet, uint samplerCode) {
    const float2 uv = lerp(uv0, uv1, float(uvSet == 1u));
    return maps.sampleTexture(slot, uv, samplerCode, uvSet == 1u ? lod1 : lod0);
  }
  // An extension texture, packed slot | UV set << 8 | sampler code << 16.
  float4 readPacked(uint packed) { return read(packed & 0xFFu, (packed >> 8u) & 0xFFu, (packed >> 16u) & 0xFFu); }
};

// coneWidth: the ray cone's width at the hit (ptConeWidthOrLevelZero); -1 samples level zero.
public PtSurface ptSurfaceAt<M : IPtMaps>(PtGeometry geometry, M maps, PtHit hit, float3 rayDirection, float coneWidth) {
  const uint instance = hit.instance;
  const TraceInstance info = geometry.traceInstances[instance];
  const Material material = geometry.materials[info.material];
  const uint slots = info.slots;
  const uint3 v = geometry.triangleVertices(instance, hit.primitive);
  const float3 weights = float3(1.0f - hit.barycentric.x - hit.barycentric.y, hit.barycentric.x, hit.barycentric.y);

  // Vertex fields: position 0, normal 3, tangent 6, UV0 10, UV1 12, colour 14.
  const float3 p0 = geometry.position(v.x);
  const float3 p1 = geometry.position(v.y);
  const float3 p2 = geometry.position(v.z);
  HitTexels<M> texels;
  texels.maps = maps;
  texels.uv0 = geometry.interpolate2(v, 10u, weights);
  texels.uv1 = geometry.interpolate2(v, 12u, weights);
  // Ray-cone level of detail per UV set: the same world-space triangle, each set's UVs.
  // Level-zero mode (coneWidth < 0) skips the loads and arithmetic.
  texels.lod0 = kPtLevelZero;
  texels.lod1 = kPtLevelZero;
  if (coneWidth >= 0.0f) {
    const float3 worldCross = ptWorldCross(info.objectToWorld0, info.objectToWorld1, info.objectToWorld2, p0, p1, p2);
    texels.lod0 = ptConeLodBase(ptUvCross(geometry.uv(v.x, 10u), geometry.uv(v.y, 10u), geometry.uv(v.z, 10u)), worldCross,
                                rayDirection, coneWidth);
    texels.lod1 = ptConeLodBase(ptUvCross(geometry.uv(v.x, 12u), geometry.uv(v.y, 12u), geometry.uv(v.z, 12u)), worldCross,
                                rayDirection, coneWidth);
  }
  const float3 objectNormal = geometry.interpolate3(v, 3u, weights);
  const float4 objectTangent = float4(geometry.interpolate3(v, 6u, weights), geometry.interpolate1(v, 9u, weights));

  PtSurface surface;
  // The position from the vertices, not origin + t * direction: it lies on the triangle to the
  // precision of its own coordinates, which the ray offset relies on.
  surface.position = applyRows(info.objectToWorld0, info.objectToWorld1, info.objectToWorld2,
                               p0 * weights.x + p1 * weights.y + p2 * weights.z);
  const float3 n0 = info.normalToWorld0.xyz;
  const float3 n1 = info.normalToWorld1.xyz;
  const float3 n2 = info.normalToWorld2.xyz;
  const float3 face = cross(p1 - p0, p2 - p0);
  float3 geometric = normalize(float3(dot(n0, face), dot(n1, face), dot(n2, face)));
  float3 normal = float3(dot(n0, objectNormal), dot(n1, objectNormal), dot(n2, objectNormal));
  if (dot(normal, normal) > 1e-12f) normal = normalize(normal);
  else normal = geometric;
  // Rays hit either side; the shading frame faces the ray, as the forward pass faces a
  // double-sided back face towards the eye. Arrived from the front (the face the winding and
  // determinant make front) or the back.
  const float entering = dot(geometric, rayDirection) < 0.0f ? 1.0f : 0.0f;
  if (dot(geometric, rayDirection) > 0.0f) geometric = -geometric;
  if (dot(normal, geometric) < 0.0f) normal = -normal;
  surface.clearcoatNormal = normal; // before the base normal map, as glTF specifies

  float3 tangent = applyRowsToDirection(info.objectToWorld0, info.objectToWorld1, info.objectToWorld2, objectTangent.xyz);
  tangent = tangent - normal * dot(normal, tangent);
  const bool hasTangent = dot(tangent, tangent) > 1e-12f;
  const float handedness = objectTangent.w < 0.0f ? -1.0f : 1.0f;
  if (hasTangent) {
    tangent = normalize(tangent);
    const float3 sampled =
        texels.read((slots >> 24u) & 0xFFu, (material.texture.x >> 24u) & 0xFFu, (material.texture.y >> 24u) & 0xFFu).xyz * 2.0f -
        1.0f;
    const float scale = material.factors.z;
    const float3 perturbed = normalize(tangent * (sampled.x * scale) + cross(normal, tangent) * (sampled.y * scale * handedness) +
                                       normal * sampled.z);
    // A normal map may tilt the normal past the geometric horizon; keep the unperturbed one then.
    if (dot(perturbed, geometric) > 0.0f) normal = perturbed;
  }
  surface.geometricNormal = geometric;
  surface.normal = normal;

  const float4 metallicRoughness =
      texels.read((slots >> 8u) & 0xFFu, (material.texture.x >> 8u) & 0xFFu, (material.texture.y >> 8u) & 0xFFu);
  const float4 vertexColor = float4(geometry.interpolate3(v, 14u, weights), geometry.interpolate1(v, 17u, weights));
  surface.baseColor =
      (texels.read(slots & 0xFFu, material.texture.x & 0xFFu, material.texture.y & 0xFFu) * material.baseColorFactor * vertexColor)
          .xyz;
  surface.emissive = texels.read((slots >> 16u) & 0xFFu, (material.texture.x >> 16u) & 0xFFu, (material.texture.y >> 16u) & 0xFFu)
                         .xyz *
                     material.emissive.xyz * material.emissive.w;
  surface.metallic = saturate(material.factors.x * metallicRoughness.z);
  surface.roughness = clamp(material.factors.y * metallicRoughness.y, kMinRoughness, 1.0f);
  const uint4 extension = material.extensionTextures;
  surface.transmission = float4(0.0f, ptMaterialIor(material.transmission), 1.0f, entering);
  if (material.transmission.x > 0.0f)
    surface.transmission = float4(saturate(material.transmission.x * texels.readPacked(extension.x).x), surface.transmission.y,
                                  material.transmission.z > 0.0f ? 0.0f : 1.0f, entering);
  surface.clearcoat = float4(0.0f, kMinRoughness, 0.0f, 0.0f);
  if (material.clearcoat.x > 0.0f) {
    surface.clearcoat.x = saturate(material.clearcoat.x * texels.readPacked(extension.y).x);
    surface.clearcoat.y = clamp(material.clearcoat.y * texels.readPacked(extension.z).y, kMinRoughness, 1.0f);
    if (hasTangent && (extension.w & 0xFFu) != 1u) {
      const float3 coatSample = texels.readPacked(extension.w).xyz * 2.0f - 1.0f;
      const float coatScale = material.clearcoat.z;
      const float3 coatNormal = normalize(tangent * (coatSample.x * coatScale) +
                                          cross(surface.clearcoatNormal, tangent) * (coatSample.y * coatScale * handedness) +
                                          surface.clearcoatNormal * coatSample.z);
      if (dot(coatNormal, geometric) > 0.0f) surface.clearcoatNormal = coatNormal;
    }
  }
  return surface;
}
