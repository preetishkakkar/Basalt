// Reflection resolve: per pixel the environment, a screen-space march or a ray, as radiance
// plus confidence; the composite adds it by weight and blurs by roughness through the mips.
// resolve_reflections_rt, where the device has ray queries, adds the traced mode; both come
// from resolve<TRACED>.
import basalt_common;
import basalt_trace;

struct ReflectionUniforms {
  float4x4 viewProjection;
  float4x4 inverseViewProjection;
  float4 camera;      // xyz eye, w mode (0 environment, 1 screen space, 2 traced)
  float4 march;       // x max distance, y thickness, z steps, w near plane
  float4 sun;         // xyz direction towards the sun, w intensity
  float4 sunColor;    // rgb, w environment intensity
  float4 target;      // xy size, zw texel size
  float4 environment; // x prefiltered mip count, y reflection mip count, z scene scale, w frame
  float4 debug;       // x debug view (11 resolved reflection, 12 trace confidence), y ray instance mask,
                      // z frames accumulated so far, w punctual light count
  float4 sampling;    // x traced reflections sample the GGX lobe (1) or take the mirror direction (0),
                      // y noise frame (0 when still), zw unused
  SphericalHarmonics sh;
};

ConstantBuffer<ReflectionUniforms> uniforms;
Texture2D<float4> depthBuffer;
Texture2D<float4> normalRoughness;
Texture2D<float4> reflectionWeight;
Texture2D<float4> sceneColor;
TextureCube<float4> prefilteredCube;
WTexture2D<float4> reflection; // resolve: written
Texture2D<float4> resolvedReflection; // composite: the resolve's output, through its mips
WTexture2D<float4> lit;
StructuredBuffer<Light> lights;
SamplerState clampSampler;
SamplerState pointSampler;

ParameterBlock<TraceScene> trace;

// Reverse-Z with an infinite far plane: linear depth is near over the stored value.
float linearDepth(float stored, float nearPlane) { return nearPlane / max(stored, 1e-7f); }

// World-space march against the depth buffer with a binary refine; returns the hit uv and a confidence.
float screenSpaceTrace(float3 origin, float3 direction, float4x4 viewProjection, float maxDistance, float thickness,
                       uint steps, float nearPlane, float3 eye, float jitter, inout float2 hitUV) {
  const float stepLength = maxDistance / float(steps);
  float travelled = stepLength * (0.5f + jitter);
  float previousTravelled = 0.0f;
  float coarseGap = 0.0f;
  bool found = false;
  for (uint i = 0u; i < steps && !found; ++i) {
    const float3 point = origin + direction * travelled;
    const float4 clip = mul(viewProjection, float4(point, 1.0f));
    if (clip.w <= 0.0f) return 0.0f;
    const float3 ndc = clip.xyz / clip.w;
    const float2 uv = float2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f);
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) return 0.0f;
    const float stored = depthBuffer.SampleLevel(pointSampler, uv, 0.0f).r;
    if (stored <= 0.0f) {
      previousTravelled = travelled;
      travelled += stepLength;
      continue;
    }
    const float rayLinear = linearDepth(ndc.z, nearPlane);
    const float sceneLinear = linearDepth(stored, nearPlane);
    if (rayLinear > sceneLinear) {
      found = true;
      hitUV = uv;
      coarseGap = rayLinear - sceneLinear;
    } else {
      previousTravelled = travelled;
      travelled += stepLength;
    }
  }
  if (!found) return 0.0f;

  // Refine the crossing, then test thickness there: a real surface leaves a tiny residual,
  // a silhouette the depth of whatever lies beyond.
  float low = previousTravelled, high = travelled;
  float residual = coarseGap;
  for (uint i = 0u; i < 6u; ++i) {
    const float middle = (low + high) * 0.5f;
    const float3 point = origin + direction * middle;
    const float4 clip = mul(viewProjection, float4(point, 1.0f));
    const float3 ndc = clip.xyz / max(clip.w, 1e-6f);
    const float2 uv = float2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f);
    const float stored = depthBuffer.SampleLevel(pointSampler, uv, 0.0f).r;
    const float gap = linearDepth(ndc.z, nearPlane) - linearDepth(max(stored, 1e-7f), nearPlane);
    if (gap > 0.0f) {
      high = middle;
      hitUV = uv;
      residual = gap;
    } else {
      low = middle;
    }
  }
  if (residual > thickness * (1.0f + high / maxDistance)) return 0.0f;

  // Fade at the screen edge and for rays heading back at the camera.
  const float2 edge = min(hitUV, float2(1.0f) - hitUV) * 8.0f;
  const float edgeFade = saturate(min(edge.x, edge.y));
  const float3 toEye = normalize(eye - origin);
  const float backFade = saturate(1.0f - dot(direction, toEye));
  return edgeFade * backFade;
}

float3 shadeSurface(HitSurface surface, float3 rayDirection, float visible) {
  const float3 toEye = -rayDirection;
  const float3 diffuseColor = surface.baseColor * (1.0f - surface.metallic);
  const float3 f0 = lerp(float3(0.04f), surface.baseColor, surface.metallic);
  const float3 sunDirection = normalize(uniforms.sun.xyz);
  float3 colour = shadeDirect(surface.normal, toEye, sunDirection, uniforms.sunColor.rgb * uniforms.sun.w, diffuseColor,
                              f0, surface.roughness) *
                  visible;
  // Unshadowed: a shadow ray per light per hit is not worth it.
  const uint lightCount = uint(uniforms.debug.w);
  for (uint i = 0u; i < lightCount; ++i) {
    const Light light = lights[i];
    const float3 toLight = light.position.xyz - surface.position;
    const float distanceToLight = length(toLight);
    if (distanceToLight <= 1e-4f) continue;
    const float3 lightDirection = toLight / distanceToLight;
    float attenuation = distanceAttenuation(distanceToLight, light.position.w);
    if (light.cone.y > 0.5f) {
      const float cosine = dot(normalize(light.direction.xyz), -lightDirection);
      const float t = saturate((cosine - light.cone.x) / max(light.direction.w - light.cone.x, 1e-4f));
      attenuation *= t * t;
    }
    if (attenuation <= 0.0f) continue;
    colour += shadeDirect(surface.normal, toEye, lightDirection, light.color.rgb * light.color.w * attenuation,
                          diffuseColor, f0, surface.roughness);
  }
  const float normalDotView = saturate(dot(surface.normal, toEye)) + 1e-5f;
  const float2 brdf = environmentBRDF(surface.roughness, normalDotView);
  const float3 fresnel = fresnelSchlickRoughness(f0, normalDotView, surface.roughness);
  const float3 irradiance = uniforms.sh.irradiance(surface.normal);
  const float3 reflected = reflect(rayDirection, surface.normal);
  const float3 prefiltered =
      prefilteredCube.SampleLevel(clampSampler, reflected, surface.roughness * (uniforms.environment.x - 1.0f)).rgb;
  colour += (irradiance * diffuseColor * (float3(1.0f) - fresnel) + prefiltered * (fresnel * brdf.x + float3(brdf.y))) *
            uniforms.sunColor.w;
  colour += surface.emissive;
  return colour;
}

void resolve<let TRACED : bool>(uint2 id) {
  if (float(id.x) >= uniforms.target.x || float(id.y) >= uniforms.target.y) return;
  const float2 uv = (float2(id) + 0.5f) * uniforms.target.zw;

  float depth = depthBuffer.SampleLevel(pointSampler, uv, 0.0f).r;
  const float4 weight = reflectionWeight.SampleLevel(pointSampler, uv, 0.0f);
  // A blended surface's depth rides in weight.w and reads larger (nearer) under reverse Z;
  // the margin covers half-float rounding.
  if (weight.w > depth * 1.001f) depth = weight.w;
  // The sky, and any pixel whose reflection would be multiplied by nothing.
  if (depth <= 0.0f || dot(weight.rgb, float3(1.0f)) <= 1e-5f) {
    reflection.Store(id, float4(0.0f));
    return;
  }

  const float4 nr = normalRoughness.SampleLevel(pointSampler, uv, 0.0f);
  const float3 normal = normalize(nr.xyz);
  const float roughness = nr.w;
  // The y flip: the forward pass renders Y up in clip space; the grid's row 0 is the top.
  const float3 world = reconstructWorld(float2(uv.x, 1.0f - uv.y), depth, uniforms.inverseViewProjection);
  const float3 toEye = normalize(uniforms.camera.xyz - world);
  const float3 direction = reflect(-toEye, normal);

  const float envMip = roughness * (uniforms.environment.x - 1.0f);
  const float3 fallback = prefilteredCube.SampleLevel(clampSampler, direction, envMip).rgb * uniforms.sunColor.w;
  float3 colour = fallback;
  float confidence = 0.0f;
  const uint mode = uint(uniforms.camera.w);

  if (mode == 1u) {
    float2 hitUV = uv;
    const float jitter = animatedNoise(float2(id), uniforms.sampling.y);
    confidence = screenSpaceTrace(world + normal * (uniforms.environment.z * 1e-3f), direction, uniforms.viewProjection,
                                  uniforms.march.x, uniforms.march.y, uint(uniforms.march.z), uniforms.march.w,
                                  uniforms.camera.xyz, jitter, hitUV);
    if (confidence > 0.0f) {
      const float3 hit = sceneColor.SampleLevel(clampSampler, hitUV, 0.0f).rgb;
      colour = lerp(fallback, hit, confidence);
    }
  } else if (TRACED && mode == 2u) {
    const float scale = uniforms.environment.z;
    const uint mask = uint(uniforms.debug.y);
    // While accumulating, sample the GGX lobe per frame; otherwise the mirror direction and a blurred mip.
    float3 rayDirection = direction;
    bool sendRay = true;
    bool keepEscape = true;
    if (uniforms.sampling.x > 0.5f) {
      const float2 pixel = float2(id);
      const float3 rotation = frameRotation(uniforms.sampling.y);
      const float2 xi = float2(frac(interleavedGradientNoise(pixel) + rotation.x),
                               frac(hashFloat(id.x * 1973u + id.y * 9277u) + rotation.y));
      float3 sampled = reflect(-toEye, importanceSampleGGX(xi, normal, roughness));
      // Redraw once below the horizon, so grazing rough surfaces do not sharpen to the mirror.
      if (dot(sampled, normal) <= 1e-3f) {
        const float2 xi2 = float2(frac(xi.x + 0.618034f), frac(xi.y + 0.324717f));
        sampled = reflect(-toEye, importanceSampleGGX(xi2, normal, roughness));
      }
      // Every ray is traced, and one that hits is kept whatever its angle: the environment
      // cannot stand in for the scene. An escaping ray is kept with probability cosine, the
      // prefiltered environment standing in otherwise, which matches how that is weighted.
      const float cosine = dot(sampled, normal);
      // Its own sequence, uncorrelated with the direction draw.
      const float accept = frac(hashFloat(id.x * 26699u + id.y * 15485863u) + rotation.z);
      sendRay = cosine > 1e-3f;
      keepEscape = accept <= cosine;
      if (sendRay) rayDirection = sampled;
    }
    if (sendRay) {
      if (let surface = trace.nearestSurface(world + normal * (scale * 2e-3f), rayDirection, scale * 1e-3f,
                                             uniforms.march.x, mask)) {
        // Sun shadow ray from the hit, alpha-tested.
        const bool shadowed = trace.occluded(surface.position + surface.normal * (scale * 2e-3f),
                                             normalize(uniforms.sun.xyz), scale * 1e-3f, scale * 1.0e4f, mask);
        colour = shadeSurface(surface, rayDirection, shadowed ? 0.0f : 1.0f);
        confidence = 1.0f;
      } else if (uniforms.sampling.x > 0.5f && keepEscape) {
        // An escaped sampled ray reads the sharp environment: the lobe itself is being averaged.
        colour = prefilteredCube.SampleLevel(clampSampler, rayDirection, 0.0f).rgb * uniforms.sunColor.w;
        confidence = 1.0f;
      }
    }
  }
  reflection.Store(id, float4(colour, confidence));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void resolve_reflections(uint3 id: SV_DispatchThreadID) { resolve<false>(id.xy); }

[shader("compute")]
[numthreads(8, 8, 1)]
void resolve_reflections_rt(uint3 id: SV_DispatchThreadID) { resolve<true>(id.xy); }

// Adds the resolved reflection by weight, blurred through the mips by roughness;
// the environment fallback is already prefiltered.
[shader("compute")]
[numthreads(8, 8, 1)]
void composite_reflections(uint3 thread: SV_DispatchThreadID) {
  const uint2 id = thread.xy;
  if (float(id.x) >= uniforms.target.x || float(id.y) >= uniforms.target.y) return;
  const float2 uv = (float2(id) + 0.5f) * uniforms.target.zw;
  const float3 colour = sceneColor.SampleLevel(pointSampler, uv, 0.0f).rgb;
  const float3 weight = reflectionWeight.SampleLevel(pointSampler, uv, 0.0f).rgb;
  const float roughness = normalRoughness.SampleLevel(pointSampler, uv, 0.0f).w;

  // No mip blur when the rays sampled the lobe themselves.
  const bool sampledLobe = uniforms.sampling.x > 0.5f && uniforms.camera.w > 1.5f;
  const float mip = sampledLobe ? 0.0f : sqrt(roughness) * (uniforms.environment.y - 1.0f);
  const float4 blurred = resolvedReflection.SampleLevel(clampSampler, uv, mip);
  const float4 sharp = resolvedReflection.SampleLevel(pointSampler, uv, 0.0f);
  const float3 resolved = lerp(sharp.rgb, blurred.rgb, saturate(blurred.a));
  float3 shown = colour + weight * resolved;
  // Debug views 11 and 12: the resolved reflection and its confidence.
  if (uniforms.debug.x > 11.5f && uniforms.debug.x < 12.5f) shown = float3(sharp.a);
  else if (uniforms.debug.x > 10.5f && uniforms.debug.x < 11.5f) shown = resolved;
  lit.Store(id, float4(shown, 1.0f));
}
