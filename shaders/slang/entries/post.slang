// Post: exposure, bloom, tone mapping, grade, optional FXAA. The swapchain is sRGB, so this writes linear.
import basalt_common;

struct PostUniforms {
  float4 parameters;  // x exposure, y bloom strength, z tonemap (0 ACES, 1 Reinhard, 2 none), w white point
  float4 target;      // xy target size, zw texel size
  float4 effects;     // x vignette, y grain, z sharpen, w time
  float4 antialias;   // x FXAA on, y edge threshold, z subpixel blend, w unused
};

ConstantBuffer<PostUniforms> post;
SamplerState clampSampler;
Texture2D<float4> hdrColor;
Texture2D<float4> bloom;

struct PostVaryings {
  float4 position : SV_Position;
  float2 uv : TEXCOORD0;
};

[shader("vertex")]
PostVaryings post_vertex(uint vertexIndex: SV_VulkanVertexID) {
  const float2 corner = vertexIndex == 0u   ? float2(-1.0f, -1.0f)
                        : vertexIndex == 1u ? float2(3.0f, -1.0f)
                                            : float2(-1.0f, 3.0f);
  PostVaryings result;
  result.position = float4(corner, 0.0f, 1.0f);
  result.uv = corner * 0.5f + 0.5f;
  return result;
}

float luminance(float3 colour) { return dot(colour, float3(0.299f, 0.587f, 0.114f)); }

float3 grade(float3 colour, float2 uv) {
  // Neither curve is defined below zero.
  colour = max(float3(0.0f), colour) * post.parameters.x;
  // Mode two passes through, for the debug views.
  colour = post.parameters.z > 1.5f   ? saturate(colour)
           : post.parameters.z > 0.5f ? tonemapReinhard(colour, post.parameters.w)
                                      : tonemapACES(colour);
  if (post.effects.x > 0.0f) {
    const float2 centred = uv * 2.0f - 1.0f;
    const float falloff = 1.0f - post.effects.x * dot(centred, centred) * 0.25f;
    colour *= saturate(falloff);
  }
  if (post.effects.y > 0.0f) {
    const float noise = frac(sin(dot(uv + float2(post.effects.w, post.effects.w), float2(12.9898f, 78.233f))) *
                             43758.5453f);
    colour += float3((noise - 0.5f) * post.effects.y);
  }
  return colour;
}

// Tone-mapped taps for FXAA, which must see display values.
float3 mapped(float2 uv) {
  const float3 colour = hdrColor.Sample(clampSampler, uv).rgb + bloom.Sample(clampSampler, uv).rgb * post.parameters.y;
  return grade(colour, uv);
}

[shader("fragment")]
float4 post_fragment(PostVaryings input) : SV_Target0 {
  float3 colour = hdrColor.Sample(clampSampler, input.uv).rgb;

  // Unsharp mask on radiance, before tone mapping.
  if (post.effects.z > 0.0f) {
    const float2 texel = post.target.zw;
    float3 blurred = float3(0.0f);
    for (int y = -1; y <= 1; ++y) {
      for (int x = -1; x <= 1; ++x) {
        blurred += hdrColor.Sample(clampSampler, input.uv + float2(float(x), float(y)) * texel).rgb;
      }
    }
    blurred *= (1.0f / 9.0f);
    colour = max(float3(0.0f), colour + (colour - blurred) * post.effects.z);
  }

  colour += bloom.Sample(clampSampler, input.uv).rgb * post.parameters.y;
  colour = grade(colour, input.uv);

  // FXAA: the four neighbours' contrast finds an edge, the luma gradient its direction,
  // two taps along it blend across.
  if (post.antialias.x > 0.5f) {
    const float2 texel = post.target.zw;
    const float3 north = mapped(input.uv + float2(0.0f, -texel.y));
    const float3 south = mapped(input.uv + float2(0.0f, texel.y));
    const float3 west = mapped(input.uv + float2(-texel.x, 0.0f));
    const float3 east = mapped(input.uv + float2(texel.x, 0.0f));
    const float lumaCentre = luminance(colour);
    const float lumaN = luminance(north), lumaS = luminance(south);
    const float lumaW = luminance(west), lumaE = luminance(east);
    const float lumaMin = min(lumaCentre, min(min(lumaN, lumaS), min(lumaW, lumaE)));
    const float lumaMax = max(lumaCentre, max(max(lumaN, lumaS), max(lumaW, lumaE)));
    const float contrast = lumaMax - lumaMin;
    if (contrast >= max(post.antialias.y, lumaMax * 0.125f)) {
      const float3 northWest = mapped(input.uv + float2(-texel.x, -texel.y));
      const float3 northEast = mapped(input.uv + float2(texel.x, -texel.y));
      const float3 southWest = mapped(input.uv + float2(-texel.x, texel.y));
      const float3 southEast = mapped(input.uv + float2(texel.x, texel.y));
      const float lumaNW = luminance(northWest), lumaNE = luminance(northEast);
      const float lumaSW = luminance(southWest), lumaSE = luminance(southEast);

      float2 gradient = float2(-((lumaNW + lumaNE) - (lumaSW + lumaSE)), (lumaNW + lumaSW) - (lumaNE + lumaSE));
      const float reduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.03125f, 1e-4f);
      const float scale = 1.0f / (min(abs(gradient.x), abs(gradient.y)) + reduce);
      gradient = clamp(gradient * scale, float2(-8.0f), float2(8.0f)) * texel;

      const float3 tapA = 0.5f * (mapped(input.uv + gradient * (1.0f / 3.0f - 0.5f)) +
                                  mapped(input.uv + gradient * (2.0f / 3.0f - 0.5f)));
      const float3 tapB = tapA * 0.5f + 0.25f * (mapped(input.uv - gradient * 0.5f) + mapped(input.uv + gradient * 0.5f));
      const float lumaB = luminance(tapB);
      const float useWide = (lumaB < lumaMin || lumaB > lumaMax) ? 0.0f : 1.0f;
      const float3 antialiased = lerp(tapA, tapB, useWide);
      colour = lerp(colour, antialiased, post.antialias.z);
    }
  }
  return float4(colour, 1.0f);
}
