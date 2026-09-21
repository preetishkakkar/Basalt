// Sky: a screen triangle at the far plane sampling the environment cube along the view
// ray. Writes the same three targets, with zero reflection weight.
#include "common.metal"

struct SkyVaryings {
  float4 position [[position]];
  float2 ndc      [[user(locn0)]];
};

struct SkyOutput {
  float4 color            [[color(0)]];
  float4 normalRoughness  [[color(1)]];
  float4 reflectionWeight [[color(2)]];
};

vertex SkyVaryings sky_vertex(uint vertexIndex [[vertex_id]]) {
  // One oversized triangle: no seam across the screen.
  const float2 corner = vertexIndex == 0u   ? float2(-1.0f, -1.0f)
                        : vertexIndex == 1u ? float2(3.0f, -1.0f)
                                            : float2(-1.0f, 3.0f);
  SkyVaryings out;
  // Reverse-Z puts the far plane at zero, which is where the sky belongs.
  out.position = float4(corner, 0.0f, 1.0f);
  out.ndc = corner;
  return out;
}

fragment SkyOutput sky_fragment(SkyVaryings input [[stage_in]],
                                constant FrameUniforms& frame [[buffer(0)]],
                                texturecube<float> environmentCube [[texture(0)]],
                                sampler clampSampler [[sampler(0)]]) {
  const float4 far = frame.inverseViewProjection * float4(input.ndc, 1.0e-6f, 1.0f);
  const float3 direction = normalize(far.xyz / far.w - frame.cameraPosition.xyz);
  const float3 radiance = environmentCube.sample(clampSampler, direction, level(0.0f)).rgb;
  SkyOutput out;
  out.color = float4(radiance * frame.environment.x, 1.0f);
  out.normalRoughness = float4(0.0f, 0.0f, 0.0f, 1.0f);
  out.reflectionWeight = float4(0.0f);
  return out;
}
