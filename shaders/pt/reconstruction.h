// Float4/uint4 lanes only: shared CPU/MSL storage-buffer ABI, 128 bytes per sample.
#pragma once
struct PtReconstructionSample {
  float4 positionDepth;       // world position, forward linear depth
  float4 geometricNormal;     // world normal, V7 layer flags (ptGuideLayers)
  float4 normalRoughness;     // world shading normal, perceptual roughness
  float4 albedo;              // linear base color, reserved
  float4 viewDistance;        // primary ray direction, secondary hit distance (-1 miss)
  float4 diffuse;             // fresh diffuse/residual radiance, reserved
  float4 specular;            // fresh specular radiance, reserved
  uint4 identity;             // instance, material, coverage (0 miss, 1 opaque, 2 blend, 3 mask), primitive
};

// V7 layers whose specular correspondence primary-surface reprojection cannot follow (the
// V3.1 mirror policy): 1 a transmissive surface (what shows through moves with
// the refraction, not with the surface), 2 a mirror-like clearcoat (coat roughness below the
// 0.15 mirror threshold).
inline float ptGuideLayers(float4 transmission, float4 clearcoat) {
  float layers = 0.0f;
  if (transmission.x > 0.0f) layers = layers + 1.0f;
  if (clearcoat.x > 0.0f && clearcoat.y < 0.15f) layers = layers + 2.0f;
  return layers;
}

inline PtReconstructionSample ptEmptyReconstructionSample() {
  PtReconstructionSample s;
  s.positionDepth = float4(0.0f);
  s.geometricNormal = float4(0.0f);
  s.normalRoughness = float4(0.0f);
  s.albedo = float4(0.0f);
  s.viewDistance = float4(0.0f, 0.0f, 0.0f, -1.0f);
  s.diffuse = float4(0.0f);
  s.specular = float4(0.0f);
  s.identity = uint4(0u);
  return s;
}

struct PtTemporalHistory {
  float4 diffuse;             // RGB estimate, effective history length
  float4 specular;            // RGB estimate, effective history length
  float4 moments;             // diffuse first/second moment, specular first/second moment
};

struct PtTemporalUniforms {
  PathUniforms previousCamera;
  uint4 image;                // width, height, reset, a-trous stride
  float4 control;             // stable raw SPP, reserved
  uint4 outputExtent;         // display width/height, reserved (can exceed guide extent for CPU preview)
};
