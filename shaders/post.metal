// Post: exposure, bloom, tone mapping, grade, optional FXAA. The swapchain is sRGB, so this writes linear.
#include "common.metal"

struct PostUniforms {
  float4 parameters;  // x exposure, y bloom strength, z tonemap (0 ACES, 1 Reinhard, 2 none), w white point
  float4 target;      // xy target size, zw texel size
  float4 effects;     // x vignette, y grain, z sharpen, w time
  float4 antialias;   // x FXAA on, y edge threshold, z subpixel blend, w unused
};

struct PostVaryings {
  float4 position [[position]];
  float2 uv       [[user(locn0)]];
};

vertex PostVaryings post_vertex(uint vertexIndex [[vertex_id]]) {
  const float2 corner = vertexIndex == 0u   ? float2(-1.0f, -1.0f)
                        : vertexIndex == 1u ? float2(3.0f, -1.0f)
                                            : float2(-1.0f, 3.0f);
  PostVaryings out;
  out.position = float4(corner, 0.0f, 1.0f);
  out.uv = corner * 0.5f + 0.5f;
  return out;
}

static float luminance(float3 colour) { return dot(colour, float3(0.299f, 0.587f, 0.114f)); }

static float3 grade(float3 colour, constant PostUniforms& post, float2 uv) {
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
    const float noise = fract(sin(dot(uv + float2(post.effects.w, post.effects.w),
                                      float2(12.9898f, 78.233f))) * 43758.5453f);
    colour += float3((noise - 0.5f) * post.effects.y);
  }
  return colour;
}

// Tone-mapped taps for FXAA, which must see display values.
static float3 mapped(texture2d<float> hdr, texture2d<float> bloom, sampler s, float2 uv,
                     constant PostUniforms& post) {
  float3 colour = hdr.sample(s, uv).rgb + bloom.sample(s, uv).rgb * post.parameters.y;
  return grade(colour, post, uv);
}

fragment float4 post_fragment(PostVaryings input [[stage_in]],
                              constant PostUniforms& post [[buffer(0)]],
                              texture2d<float> hdrColor [[texture(0)]],
                              texture2d<float> bloom [[texture(1)]],
                              sampler clampSampler [[sampler(0)]]) {
  float3 colour = hdrColor.sample(clampSampler, input.uv).rgb;

  // Unsharp mask on radiance, before tone mapping.
  if (post.effects.z > 0.0f) {
    const float2 texel = post.target.zw;
    float3 blurred = float3(0.0f);
    for (int y = -1; y <= 1; ++y) {
      for (int x = -1; x <= 1; ++x) {
        blurred += hdrColor.sample(clampSampler, input.uv + float2(float(x), float(y)) * texel).rgb;
      }
    }
    blurred *= (1.0f / 9.0f);
    colour = max(float3(0.0f), colour + (colour - blurred) * post.effects.z);
  }

  colour += bloom.sample(clampSampler, input.uv).rgb * post.parameters.y;
  colour = grade(colour, post, input.uv);

  // FXAA: the four neighbours' contrast finds an edge, the luma gradient its direction,
  // two taps along it blend across.
  if (post.antialias.x > 0.5f) {
    const float2 texel = post.target.zw;
    const float3 north = mapped(hdrColor, bloom, clampSampler, input.uv + float2(0.0f, -texel.y), post);
    const float3 south = mapped(hdrColor, bloom, clampSampler, input.uv + float2(0.0f, texel.y), post);
    const float3 west = mapped(hdrColor, bloom, clampSampler, input.uv + float2(-texel.x, 0.0f), post);
    const float3 east = mapped(hdrColor, bloom, clampSampler, input.uv + float2(texel.x, 0.0f), post);
    const float lumaCentre = luminance(colour);
    const float lumaN = luminance(north), lumaS = luminance(south);
    const float lumaW = luminance(west), lumaE = luminance(east);
    const float lumaMin = min(lumaCentre, min(min(lumaN, lumaS), min(lumaW, lumaE)));
    const float lumaMax = max(lumaCentre, max(max(lumaN, lumaS), max(lumaW, lumaE)));
    const float contrast = lumaMax - lumaMin;
    if (contrast >= max(post.antialias.y, lumaMax * 0.125f)) {
      const float3 northWest = mapped(hdrColor, bloom, clampSampler, input.uv + float2(-texel.x, -texel.y), post);
      const float3 northEast = mapped(hdrColor, bloom, clampSampler, input.uv + float2(texel.x, -texel.y), post);
      const float3 southWest = mapped(hdrColor, bloom, clampSampler, input.uv + float2(-texel.x, texel.y), post);
      const float3 southEast = mapped(hdrColor, bloom, clampSampler, input.uv + float2(texel.x, texel.y), post);
      const float lumaNW = luminance(northWest), lumaNE = luminance(northEast);
      const float lumaSW = luminance(southWest), lumaSE = luminance(southEast);

      float2 gradient = float2(-((lumaNW + lumaNE) - (lumaSW + lumaSE)),
                               (lumaNW + lumaSW) - (lumaNE + lumaSE));
      const float reduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.03125f, 1e-4f);
      const float scale = 1.0f / (min(abs(gradient.x), abs(gradient.y)) + reduce);
      gradient = clamp(gradient * scale, float2(-8.0f), float2(8.0f)) * texel;

      const float3 tapA = 0.5f * (mapped(hdrColor, bloom, clampSampler, input.uv + gradient * (1.0f / 3.0f - 0.5f), post) +
                                  mapped(hdrColor, bloom, clampSampler, input.uv + gradient * (2.0f / 3.0f - 0.5f), post));
      const float3 tapB = tapA * 0.5f + 0.25f * (mapped(hdrColor, bloom, clampSampler, input.uv - gradient * 0.5f, post) +
                                                 mapped(hdrColor, bloom, clampSampler, input.uv + gradient * 0.5f, post));
      const float lumaB = luminance(tapB);
      const float useWide = (lumaB < lumaMin || lumaB > lumaMax) ? 0.0f : 1.0f;
      const float3 antialiased = mix(tapA, tapB, useWide);
      colour = mix(colour, antialiased, post.antialias.z);
    }
  }
  return float4(colour, 1.0f);
}
