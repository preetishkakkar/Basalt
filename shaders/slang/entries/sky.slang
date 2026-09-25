// Sky: a screen triangle at the far plane sampling the environment cube along the view
// ray. Writes the forward pass's targets, with zero reflection weight.
import basalt_common;

ConstantBuffer<FrameUniforms> frame;
TextureCube<float4> environmentCube;
SamplerState clampSampler;

struct SkyVaryings {
  float4 position : SV_Position;
  float2 ndc : NDC;
};

struct SkyOutput {
  float4 color : SV_Target0;
  float4 normalRoughness : SV_Target1;
  float4 reflectionWeight : SV_Target2;
  float4 baseMetallic : SV_Target3;
  float4 geometricCoverage : SV_Target4;
  float4 emissive : SV_Target5;
  uint4 identity : SV_Target6;
};

[shader("vertex")]
SkyVaryings sky_vertex(uint vertexIndex: SV_VulkanVertexID) {
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

[shader("fragment")]
SkyOutput sky_fragment(SkyVaryings input) {
  const float4 far = mul(frame.inverseViewProjection, float4(input.ndc, 1.0e-6f, 1.0f));
  const float3 direction = normalize(far.xyz / far.w - frame.cameraPosition.xyz);
  float3 radiance = environmentCube.SampleLevel(clampSampler, direction, 0.0f).rgb * frame.environment.x;
  // The analytic sun of an .hdr it was extracted from: its irradiance spread over the disc,
  // whose solid angle 4 pi sin^2(r / 2) keeps its precision for a sub-degree radius.
  if (frame.sunDirection.w > 0.0f && dot(direction, normalize(frame.sunDirection.xyz)) >= frame.sunDirection.w) {
    const float halfRadius = sin(max(frame.rays.z, 1e-4f) * 0.5f);
    radiance += frame.sunColor.rgb * frame.sunColor.w / max(4.0f * kPi * halfRadius * halfRadius, 1e-12f);
  }
  SkyOutput out;
  out.color = float4(radiance, 1.0f);
  out.normalRoughness = float4(0.0f, 0.0f, 0.0f, 1.0f);
  out.reflectionWeight = float4(0.0f);
  out.baseMetallic = float4(0.0f);
  out.geometricCoverage = float4(0.0f);
  out.emissive = float4(0.0f);
  out.identity = uint4(0u);
  return out;
}
