// Dear ImGui drawing, in Metal like everything else.
#include <metal_stdlib>
using namespace metal;

struct UiUniforms {
  float4 scaleAndTranslate; // xy scale, zw translate, framebuffer space to clip
};

struct UiVertexInput {
  float2 position [[attribute(0)]];
  float2 uv       [[attribute(1)]];
  float4 color    [[attribute(2)]]; // fetched from a UNORM8 vertex format
};

struct UiVaryings {
  float4 position [[position]];
  float4 color    [[user(locn0)]];
  float2 uv       [[user(locn1)]];
};

vertex UiVaryings ui_vertex(UiVertexInput input [[stage_in]],
                            constant UiUniforms& ui [[buffer(0)]]) {
  UiVaryings out;
  out.position = float4(input.position * ui.scaleAndTranslate.xy + ui.scaleAndTranslate.zw,
                        0.0f, 1.0f);
  out.color = input.color;
  out.uv = input.uv;
  return out;
}

fragment float4 ui_fragment(UiVaryings input [[stage_in]],
                            texture2d<float> atlas [[texture(0)]],
                            sampler atlasSampler [[sampler(0)]]) {
  // ImGui colours are sRGB bytes; the target is linear.
  const float4 texel = atlas.sample(atlasSampler, input.uv);
  const float3 linearColor = pow(input.color.rgb, float3(2.2f));
  return float4(linearColor, input.color.a) * texel;
}
