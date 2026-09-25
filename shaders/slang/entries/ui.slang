// Dear ImGui drawing.

Texture2D<float4> atlas;
SamplerState atlasSampler;

struct UiTransform {
  float4 scaleAndTranslate; // xy scale, zw translate, framebuffer space to clip
};

struct UiVertexInput {
  float2 position : POSITION;
  float2 uv : TEXCOORD0;
  float4 color : COLOR; // fetched from a UNORM8 vertex format
};

struct UiVaryings {
  float4 position : SV_Position;
  float4 color : COLOR;
  float2 uv : TEXCOORD0;
};

[shader("vertex")]
UiVaryings ui_vertex(UiVertexInput input, uniform UiTransform ui) {
  UiVaryings out;
  out.position = float4(input.position * ui.scaleAndTranslate.xy + ui.scaleAndTranslate.zw, 0.0f, 1.0f);
  out.color = input.color;
  out.uv = input.uv;
  return out;
}

[shader("fragment")]
float4 ui_fragment(UiVaryings input) : SV_Target0 {
  // ImGui colours are sRGB bytes; the target is linear.
  const float4 texel = atlas.Sample(atlasSampler, input.uv);
  const float3 linearColor = pow(input.color.rgb, float3(2.2f));
  return float4(linearColor, input.color.a) * texel;
}
