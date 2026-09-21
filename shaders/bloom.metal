// Bloom mip chain: 13-tap downsample, 9-tap tent upsample, both compute.
#include "common.metal"

struct BloomUniforms {
  float4 parameters; // xy destination size, z threshold, w filter radius in texels of the source
  float4 source;     // xy source size, z mip being read, w knee
};

// Soft knee, so bloom does not pop as the exposure changes.
static float3 applyThreshold(float3 colour, float threshold, float knee) {
  const float brightness = max(colour.r, max(colour.g, colour.b));
  const float soft = clamp(brightness - threshold + knee, 0.0f, 2.0f * knee);
  const float contribution = max(soft * soft / (4.0f * knee + 1e-5f), brightness - threshold);
  return colour * (contribution / max(brightness, 1e-5f));
}

kernel void bloom_downsample(texture2d<float, access::write> destination [[texture(0)]],
                             texture2d<float> source [[texture(1)]],
                             constant BloomUniforms& bloom [[buffer(0)]],
                             sampler clampSampler [[sampler(0)]],
                             uint2 id [[thread_position_in_grid]]) {
  if (float(id.x) >= bloom.parameters.x || float(id.y) >= bloom.parameters.y) return;
  const float2 uv = (float2(float(id.x), float(id.y)) + 0.5f) /
                    float2(bloom.parameters.x, bloom.parameters.y);
  const float2 texel = 1.0f / float2(bloom.source.x, bloom.source.y);
  const float sourceMip = bloom.source.z;

  // Jimenez's thirteen taps.
  const float3 a = source.sample(clampSampler, uv + float2(-2.0f, 2.0f) * texel, level(sourceMip)).rgb;
  const float3 b = source.sample(clampSampler, uv + float2(0.0f, 2.0f) * texel, level(sourceMip)).rgb;
  const float3 c = source.sample(clampSampler, uv + float2(2.0f, 2.0f) * texel, level(sourceMip)).rgb;
  const float3 d = source.sample(clampSampler, uv + float2(-2.0f, 0.0f) * texel, level(sourceMip)).rgb;
  const float3 e = source.sample(clampSampler, uv, level(sourceMip)).rgb;
  const float3 f = source.sample(clampSampler, uv + float2(2.0f, 0.0f) * texel, level(sourceMip)).rgb;
  const float3 g = source.sample(clampSampler, uv + float2(-2.0f, -2.0f) * texel, level(sourceMip)).rgb;
  const float3 h = source.sample(clampSampler, uv + float2(0.0f, -2.0f) * texel, level(sourceMip)).rgb;
  const float3 i = source.sample(clampSampler, uv + float2(2.0f, -2.0f) * texel, level(sourceMip)).rgb;
  const float3 j = source.sample(clampSampler, uv + float2(-1.0f, 1.0f) * texel, level(sourceMip)).rgb;
  const float3 k = source.sample(clampSampler, uv + float2(1.0f, 1.0f) * texel, level(sourceMip)).rgb;
  const float3 l = source.sample(clampSampler, uv + float2(-1.0f, -1.0f) * texel, level(sourceMip)).rgb;
  const float3 m = source.sample(clampSampler, uv + float2(1.0f, -1.0f) * texel, level(sourceMip)).rgb;

  float3 colour = e * 0.125f;
  colour += (a + c + g + i) * 0.03125f;
  colour += (b + d + f + h) * 0.0625f;
  colour += (j + k + l + m) * 0.125f;

  // Threshold on the first level only.
  if (bloom.parameters.z >= 0.0f) colour = applyThreshold(colour, bloom.parameters.z, bloom.source.w);
  destination.write(float4(colour, 1.0f), id);
}

kernel void bloom_upsample(texture2d<float, access::read_write> destination [[texture(0)]],
                           texture2d<float> source [[texture(1)]],
                           constant BloomUniforms& bloom [[buffer(0)]],
                           sampler clampSampler [[sampler(0)]],
                           uint2 id [[thread_position_in_grid]]) {
  if (float(id.x) >= bloom.parameters.x || float(id.y) >= bloom.parameters.y) return;
  const float2 uv = (float2(float(id.x), float(id.y)) + 0.5f) /
                    float2(bloom.parameters.x, bloom.parameters.y);
  const float2 texel = (1.0f / float2(bloom.source.x, bloom.source.y)) * bloom.parameters.w;
  const float sourceMip = bloom.source.z;

  const float3 a = source.sample(clampSampler, uv + float2(-1.0f, 1.0f) * texel, level(sourceMip)).rgb;
  const float3 b = source.sample(clampSampler, uv + float2(0.0f, 1.0f) * texel, level(sourceMip)).rgb;
  const float3 c = source.sample(clampSampler, uv + float2(1.0f, 1.0f) * texel, level(sourceMip)).rgb;
  const float3 d = source.sample(clampSampler, uv + float2(-1.0f, 0.0f) * texel, level(sourceMip)).rgb;
  const float3 e = source.sample(clampSampler, uv, level(sourceMip)).rgb;
  const float3 f = source.sample(clampSampler, uv + float2(1.0f, 0.0f) * texel, level(sourceMip)).rgb;
  const float3 g = source.sample(clampSampler, uv + float2(-1.0f, -1.0f) * texel, level(sourceMip)).rgb;
  const float3 h = source.sample(clampSampler, uv + float2(0.0f, -1.0f) * texel, level(sourceMip)).rgb;
  const float3 i = source.sample(clampSampler, uv + float2(1.0f, -1.0f) * texel, level(sourceMip)).rgb;

  const float3 tent = (e * 4.0f + (b + d + f + h) * 2.0f + (a + c + g + i)) * (1.0f / 16.0f);
  const float4 existing = destination.read(id);
  destination.write(float4(existing.rgb + tent, 1.0f), id);
}
