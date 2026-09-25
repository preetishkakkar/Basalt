// The rasteriser's structures and helpers; the maths it shares with the path tracer is in the
// basalt_types, basalt_random and basalt_shading modules, which this one re-exports.
module basalt_common;

__exported import basalt_types;
__exported import basalt_random;
__exported import basalt_shading;

public static const uint kCascadeCount = 4u;

// Irradiance as nine SH coefficients.
public struct SphericalHarmonics {
  float4 coefficients[9];

  // Ramamoorthi-Hanrahan irradiance, clamped: a bright sun rings below zero.
  float3 irradiance(float3 n) {
    const float4 c[9] = coefficients;
    const float3 result = c[0].rgb + c[1].rgb * n.y + c[2].rgb * n.z + c[3].rgb * n.x + c[4].rgb * (n.x * n.y) +
                          c[5].rgb * (n.y * n.z) + c[6].rgb * (3.0f * n.z * n.z - 1.0f) + c[7].rgb * (n.x * n.z) +
                          c[8].rgb * (n.x * n.x - n.y * n.y);
    return max(float3(0.0f), result);
  }
};

public struct FrameUniforms {
  float4x4 viewProjection;
  float4x4 view;
  float4x4 inverseViewProjection;
  float4 cameraPosition;    // xyz eye in world space, w exposure
  float4 sunDirection;      // xyz unit vector towards the sun, w cosine of the disc the sky draws (0: none)
  float4 sunColor;          // rgb radiance, w intensity multiplier
  float4 environment;       // x IBL intensity, y prefiltered mip count, z ray instance mask, w time
  float4 shadowParameters;  // x depth bias, y normal bias, z texel world size, w softness
  float4 cascadeSplits;     // view-space far distance of each cascade
  float4 viewportAndLights; // xy target size, z light count, w debug view
  float4 rays;      // x shadow mode (0 none, 1 cascades, 2 traced), y traced shadow samples,
                    // z sun angular radius, w ambient occlusion mode (0 texture, 1 traced
                    // contact, 2 traced sky visibility)
  float4 occlusion; // x traced occlusion radius, y traced occlusion samples, z noise frame (0 when still), w scene radius
  float4 options;   // x punctual lights cast traced shadows, y near plane, z far plane, w lights are clustered
  float4 clusters;  // x tiles across, y tiles down, z depth slices, w lights a cell can hold
  // Irradiance, premultiplied so irradiance() returns radiance.
  SphericalHarmonics sh;
};

// Per-draw record, selected by the instance index from the draw's first instance.
public struct Instance {
  float4 modelRow0, modelRow1, modelRow2;
  float4 normalRow0, normalRow1, normalRow2;
  float4 materialAndFlags; // x material index, y visible, zw unused
};

public float interleavedGradientNoise(float2 pixel) {
  const float3 magic = float3(0.06711056f, 0.00583715f, 52.9829189f);
  return frac(magic.z * frac(dot(pixel, magic.xy)));
}

public float animatedNoise(float2 pixel, float frame) {
  return frac(interleavedGradientNoise(pixel) + frameRotation(frame).x);
}

// The inverse must be the jittered one the depth was rendered with.
public float3 reconstructWorld(float2 uv, float depth, float4x4 inverseViewProjection) {
  const float4 clip = float4(uv.x * 2.0f - 1.0f, uv.y * 2.0f - 1.0f, depth, 1.0f);
  const float4 world = mul(inverseViewProjection, clip);
  return world.xyz / world.w;
}

// Narkowicz's ACES fit.
public float3 tonemapACES(float3 colour) {
  const float3 x = colour * 0.6f;
  const float3 numerator = x * (x * 2.51f + float3(0.03f));
  const float3 denominator = x * (x * 2.43f + float3(0.59f)) + float3(0.14f);
  return saturate(numerator / denominator);
}

public float3 tonemapReinhard(float3 colour, float whitePoint) {
  const float3 numerator = colour * (float3(1.0f) + colour / (whitePoint * whitePoint));
  return numerator / (float3(1.0f) + colour);
}

// Face order +X, -X, +Y, -Y, +Z, -Z.
public float3 cubeDirection(uint face, float2 uv) {
  const float2 c = uv * 2.0f - 1.0f;
  float3 direction = float3(1.0f, -c.y, -c.x);
  if (face == 1u) direction = float3(-1.0f, -c.y, c.x);
  else if (face == 2u) direction = float3(c.x, 1.0f, c.y);
  else if (face == 3u) direction = float3(c.x, -1.0f, -c.y);
  else if (face == 4u) direction = float3(c.x, -c.y, 1.0f);
  else if (face == 5u) direction = float3(-c.x, -c.y, -1.0f);
  return normalize(direction);
}
