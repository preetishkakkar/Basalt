// Bloom mip chain: 13-tap downsample, 9-tap tent upsample, both compute. Each dispatch reads
// one mip of the chain and writes another; which ones and how come in as push constants.

struct BloomUniforms {
  float4 parameters; // xy destination size, z threshold (negative: none), w filter radius in texels of the source
  float4 source;     // xy source size, z mip being read, w knee
};

SamplerState clampSampler;
// The chain is half-float RGBA; the upsample reads what it adds to, so the format is spelled out.
[format("rgba16f")] RWTexture2D<float4> destination;
Texture2D<float4> source;

// Soft knee, so bloom does not pop as the exposure changes.
float3 applyThreshold(float3 colour, float threshold, float knee) {
  const float brightness = max(colour.r, max(colour.g, colour.b));
  const float soft = clamp(brightness - threshold + knee, 0.0f, 2.0f * knee);
  const float contribution = max(soft * soft / (4.0f * knee + 1e-5f), brightness - threshold);
  return colour * (contribution / max(brightness, 1e-5f));
}

// The source's colour at uv + offset texels, from the mip being read.
float3 tap(float2 uv, float2 offset, float2 texel, float mip) {
  return source.SampleLevel(clampSampler, uv + offset * texel, mip).rgb;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void bloom_downsample(uint3 thread : SV_DispatchThreadID, uniform BloomUniforms bloom) {
  const uint2 id = thread.xy;
  if (float(id.x) >= bloom.parameters.x || float(id.y) >= bloom.parameters.y) return;
  const float2 uv = (float2(float(id.x), float(id.y)) + 0.5f) / bloom.parameters.xy;
  const float2 texel = 1.0f / bloom.source.xy;
  const float mip = bloom.source.z;

  // Jimenez's thirteen taps.
  const float3 a = tap(uv, float2(-2.0f, 2.0f), texel, mip);
  const float3 b = tap(uv, float2(0.0f, 2.0f), texel, mip);
  const float3 c = tap(uv, float2(2.0f, 2.0f), texel, mip);
  const float3 d = tap(uv, float2(-2.0f, 0.0f), texel, mip);
  const float3 e = source.SampleLevel(clampSampler, uv, mip).rgb;
  const float3 f = tap(uv, float2(2.0f, 0.0f), texel, mip);
  const float3 g = tap(uv, float2(-2.0f, -2.0f), texel, mip);
  const float3 h = tap(uv, float2(0.0f, -2.0f), texel, mip);
  const float3 i = tap(uv, float2(2.0f, -2.0f), texel, mip);
  const float3 j = tap(uv, float2(-1.0f, 1.0f), texel, mip);
  const float3 k = tap(uv, float2(1.0f, 1.0f), texel, mip);
  const float3 l = tap(uv, float2(-1.0f, -1.0f), texel, mip);
  const float3 m = tap(uv, float2(1.0f, -1.0f), texel, mip);

  float3 colour = e * 0.125f;
  colour += (a + c + g + i) * 0.03125f;
  colour += (b + d + f + h) * 0.0625f;
  colour += (j + k + l + m) * 0.125f;

  // Threshold on the first level only.
  if (bloom.parameters.z >= 0.0f) colour = applyThreshold(colour, bloom.parameters.z, bloom.source.w);
  destination[id] = float4(colour, 1.0f);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void bloom_upsample(uint3 thread : SV_DispatchThreadID, uniform BloomUniforms bloom) {
  const uint2 id = thread.xy;
  if (float(id.x) >= bloom.parameters.x || float(id.y) >= bloom.parameters.y) return;
  const float2 uv = (float2(float(id.x), float(id.y)) + 0.5f) / bloom.parameters.xy;
  const float2 texel = (1.0f / bloom.source.xy) * bloom.parameters.w;
  const float mip = bloom.source.z;

  const float3 a = tap(uv, float2(-1.0f, 1.0f), texel, mip);
  const float3 b = tap(uv, float2(0.0f, 1.0f), texel, mip);
  const float3 c = tap(uv, float2(1.0f, 1.0f), texel, mip);
  const float3 d = tap(uv, float2(-1.0f, 0.0f), texel, mip);
  const float3 e = source.SampleLevel(clampSampler, uv, mip).rgb;
  const float3 f = tap(uv, float2(1.0f, 0.0f), texel, mip);
  const float3 g = tap(uv, float2(-1.0f, -1.0f), texel, mip);
  const float3 h = tap(uv, float2(0.0f, -1.0f), texel, mip);
  const float3 i = tap(uv, float2(1.0f, -1.0f), texel, mip);

  const float3 tent = (e * 4.0f + (b + d + f + h) * 2.0f + (a + c + g + i)) * (1.0f / 16.0f);
  const float4 existing = destination[id];
  destination[id] = float4(existing.rgb + tent, 1.0f);
}
