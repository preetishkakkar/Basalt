// IBL bake: equirect to cube, then a GGX prefiltered mip chain. A cube a kernel writes is
// bound as a 2D array view of six layers. Each dispatch's parameters are push constants.
import basalt_common;

struct Bake {
  float4 parameters; // x face size, y roughness, z sample count, w source mip count
};

WTexture2DArray<float4> destination;
Texture2D<float4> equirectangular;
TextureCube<float4> source;
SamplerState linearSampler;

[shader("compute")]
[numthreads(8, 8, 1)]
void equirect_to_cube(uint3 id: SV_DispatchThreadID, uniform Bake bake) {
  const float size = bake.parameters.x;
  if (float(id.x) >= size || float(id.y) >= size) return;
  const float2 uv = (float2(id.xy) + 0.5f) / size;
  const float3 direction = cubeDirection(id.z, uv);
  const float4 radiance = equirectangular.SampleLevel(linearSampler, equirectangularUV(direction), 0.0f);
  // Half-float cube: clamp below the half range and drop non-finite texels, or an
  // infinity reaches every pass.
  float3 stored = min(radiance.rgb, float3(60000.0f));
  if (any(isnan(radiance.rgb)) || any(isinf(radiance.rgb))) stored = float3(0.0f);
  destination.Store(id, float4(stored, 1.0f));
}

// GGX prefilter per mip, importance sampled, with the split-sum assumption view = normal.
[shader("compute")]
[numthreads(8, 8, 1)]
void prefilter_specular(uint3 id: SV_DispatchThreadID, uniform Bake bake) {
  const float size = bake.parameters.x;
  if (float(id.x) >= size || float(id.y) >= size) return;
  const float2 uv = (float2(id.xy) + 0.5f) / size;
  const float3 normal = cubeDirection(id.z, uv);
  const float roughness = bake.parameters.y;
  const uint samples = uint(bake.parameters.z);

  if (roughness <= 0.0f) {
    destination.Store(id, float4(source.SampleLevel(linearSampler, normal, 0.0f).rgb, 1.0f));
    return;
  }

  float3 total = float3(0.0f);
  float weight = 0.0f;
  for (uint i = 0u; i < samples; ++i) {
    const float2 xi = hammersley(i, samples);
    const float3 halfVector = importanceSampleGGX(xi, normal, roughness);
    const float3 lightDirection = normalize(halfVector * (2.0f * dot(normal, halfVector)) - normal);
    const float normalDotLight = dot(normal, lightDirection);
    if (normalDotLight <= 0.0f) continue;

    // The mip from the sample's solid angle keeps a bright environment from sparkling.
    const float normalDotHalf = saturate(dot(normal, halfVector));
    const float alpha = roughness * roughness;
    const float distribution = distributionGGX(normalDotHalf, alpha);
    const float pdf = distribution * normalDotHalf / (4.0f * normalDotHalf) + 1e-4f;
    const float texelSolidAngle = 4.0f * kPi / (6.0f * size * size);
    const float sampleSolidAngle = 1.0f / (float(samples) * pdf + 1e-4f);
    const float mip = 0.5f * log2(sampleSolidAngle / texelSolidAngle) + 1.0f;

    total += source.SampleLevel(linearSampler, lightDirection, clamp(mip, 0.0f, bake.parameters.w - 1.0f)).rgb *
             normalDotLight;
    weight += normalDotLight;
  }
  const float3 prefiltered = weight > 0.0f ? total / weight : float3(0.0f);
  destination.Store(id, float4(prefiltered, 1.0f));
}
