// Forward pass: PBR with the sun, punctual lights and SH diffuse. Writes colour without
// specular reflection, normal and roughness, and the reflection weight the resolve applies.
// Included twice: forward.metal rasterised, forward_rt.metal with BASALT_RAY_TRACING.
#include "common.metal"
#ifdef BASALT_RAY_TRACING
#include <metal_raytracing>
using namespace metal::raytracing;
#endif

struct VertexInput {
  float3 position [[attribute(0)]];
  float3 normal   [[attribute(1)]];
  float4 tangent  [[attribute(2)]]; // xyz tangent, w handedness
  float2 uv0      [[attribute(3)]];
  float2 uv1      [[attribute(4)]];
  float4 color    [[attribute(5)]];
};

struct Varyings {
  float4 position  [[position]];
  float3 world     [[user(locn0)]];
  float3 normal    [[user(locn1)]];
  float4 tangent   [[user(locn2)]];
  float2 uv0       [[user(locn3)]];
  float2 uv1       [[user(locn4)]];
  float4 color     [[user(locn5)]];
  float  viewDepth [[user(locn6)]];
  uint2  identity  [[user(locn7)]] [[flat]]; // material, instance
};

struct ForwardOutput {
  float4 color          [[color(0)]]; // Lit radiance without the specular reflection; alpha for blending.
  float4 normalRoughness [[color(1)]]; // World shading normal, roughness.
  float4 reflectionWeight [[color(2)]]; // What the reflection is multiplied by; w is the surface depth.
  float4 baseMetallic [[color(3)]]; // Linear base colour, metallic.
  float4 geometricCoverage [[color(4)]]; // World geometric normal, coverage class.
  float4 emissive [[color(5)]]; // Primary emissive radiance.
  uint4 identity [[color(6)]]; // Instance, material, primitive, coverage class.
};

vertex Varyings FORWARD_VERTEX(VertexInput input [[stage_in]],
                               uint instanceIndex [[instance_id]],
                               constant FrameUniforms& frame [[buffer(0)]],
                               const device Instance* instances [[buffer(1)]]) {
  const Instance instance = instances[instanceIndex];
  const float3 world = applyRows(instance.modelRow0, instance.modelRow1, instance.modelRow2,
                                 input.position);
  const float3 normal = applyRowsToDirection(instance.normalRow0, instance.normalRow1,
                                             instance.normalRow2, input.normal);
  const float3 tangent = applyRowsToDirection(instance.modelRow0, instance.modelRow1,
                                              instance.modelRow2, input.tangent.xyz);

  Varyings out;
  out.position = frame.viewProjection * float4(world, 1.0f);
  out.world = world;
  out.normal = normal;
  out.tangent = float4(tangent, input.tangent.w);
  out.uv0 = input.uv0;
  out.uv1 = input.uv1;
  out.color = input.color;
  out.viewDepth = -(frame.view * float4(world, 1.0f)).z;
  out.identity = uint2(uint(instance.materialAndFlags.x), instanceIndex);
  return out;
}

// Cascade by view depth, 3x3 PCF, blended into the next cascade over the last tenth.
static float sampleCascade(depth2d_array<float> shadowMap, sampler shadowSampler,
                           const device float4* cascadeRows, uint cascade, float3 world,
                           float3 normal, float3 lightDirection, float4 shadowParameters) {
  const float slope = 1.0f - saturate(dot(normal, lightDirection));
  const float3 offsetWorld = world + normal * (shadowParameters.y * (1.0f + slope * 2.0f));
  const float4 point = float4(offsetWorld, 1.0f);
  const uint base = cascade * 4u;
  const float4 clip = float4(dot(cascadeRows[base + 0u], point), dot(cascadeRows[base + 1u], point),
                             dot(cascadeRows[base + 2u], point), dot(cascadeRows[base + 3u], point));
  if (clip.w <= 0.0f) return 1.0f;
  const float3 ndc = clip.xyz / clip.w;
  const float2 uv = ndc.xy * 0.5f + 0.5f;
  if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) return 1.0f;

  const float reference = ndc.z - shadowParameters.x * (1.0f + slope * 3.0f);
  const float texel = shadowParameters.z * shadowParameters.w;
  float visible = 0.0f;
  for (int y = -1; y <= 1; ++y) {
    for (int x = -1; x <= 1; ++x) {
      const float2 tap = uv + float2(float(x), float(y)) * texel;
      visible += shadowMap.sample_compare(shadowSampler, tap, cascade, reference);
    }
  }
  return visible * (1.0f / 9.0f);
}

static float cascadedSunVisibility(depth2d_array<float> shadowMap, sampler shadowSampler,
                                   const device float4* cascadeRows, float3 world, float3 normal,
                                   float3 lightDirection, float viewDepth, float4 splits,
                                   float4 shadowParameters, thread uint& chosen) {
  uint cascade = 3u;
  if (viewDepth < splits.x) cascade = 0u;
  else if (viewDepth < splits.y) cascade = 1u;
  else if (viewDepth < splits.z) cascade = 2u;
  chosen = cascade;
  float visible = sampleCascade(shadowMap, shadowSampler, cascadeRows, cascade, world, normal,
                                lightDirection, shadowParameters);
  const float far = cascade == 0u ? splits.x : cascade == 1u ? splits.y
                    : cascade == 2u ? splits.z : splits.w;
  const float near = cascade == 0u ? 0.0f : cascade == 1u ? splits.x
                     : cascade == 2u ? splits.y : splits.z;
  const float fadeStart = far - (far - near) * 0.1f;
  if (cascade + 1u < kCascadeCount && viewDepth > fadeStart) {
    const float blend = saturate((viewDepth - fadeStart) / max(far - fadeStart, 1e-4f));
    const float next = sampleCascade(shadowMap, shadowSampler, cascadeRows, cascade + 1u, world,
                                     normal, lightDirection, shadowParameters);
    visible = mix(visible, next, blend);
  }
  return visible;
}

fragment ForwardOutput FORWARD_FRAGMENT(Varyings input [[stage_in]],
                                 uint primitive [[primitive_id]],
                                 constant FrameUniforms& frame [[buffer(0)]],
                                 const device Material* materials [[buffer(1)]],
                                 const device float4* cascadeRows [[buffer(2)]],
                                 const device Light* lights [[buffer(3)]],
                                 // Light grid: a count per cell, then each cell's fixed run of indices.
                                 const device uint* lightClusters [[buffer(8)]],
#ifdef BASALT_RAY_TRACING
                                 instance_acceleration_structure scene [[buffer(4)]],
                                 const device TraceInstance* traceInstances [[buffer(5)]],
                                 const device uint* indices [[buffer(6)]],
                                 const device float* vertices [[buffer(7)]],
#endif
                                 texture2d<float> baseColorMap [[texture(0)]],
                                 texture2d<float> metallicRoughnessMap [[texture(1)]],
                                 texture2d<float> normalMap [[texture(2)]],
                                 texture2d<float> occlusionMap [[texture(3)]],
                                 texture2d<float> emissiveMap [[texture(4)]],
                                 texturecube<float> prefilteredCube [[texture(5)]],
                                 depth2d_array<float> shadowMap [[texture(6)]],
#ifdef BASALT_RAY_TRACING
                                 // Last, because an array occupies consecutive texture indices.
                                 array<texture2d<float>, kHitTextureSlots> maps [[texture(7)]],
#endif
                                 sampler materialSampler [[sampler(0)]],
                                 sampler clampSampler [[sampler(1)]],
                                 sampler shadowSampler [[sampler(2)]]
#ifdef BASALT_RAY_TRACING
                                 // Own repeating sampler: a cut-out test must not depend on the shaded material's.
                                 , sampler tableSampler [[sampler(3)]]
#endif
                                 ) {
  const Material material = materials[input.identity.x];

  const uint uvBits = material.texture.x;
  float2 baseUv = input.uv0, metallicUv = input.uv0, emissiveUv = input.uv0, normalUv = input.uv0;
  if ((uvBits & 0xFFu) == 1u) baseUv = input.uv1;
  if (((uvBits >> 8u) & 0xFFu) == 1u) metallicUv = input.uv1;
  if (((uvBits >> 16u) & 0xFFu) == 1u) emissiveUv = input.uv1;
  if (((uvBits >> 24u) & 0xFFu) == 1u) normalUv = input.uv1;
  const float4 baseSample = baseColorMap.sample(materialSampler, baseUv);
  const float4 base = baseSample * material.baseColorFactor * input.color;
  if (material.alpha.y < 1.5f && material.alpha.y > 0.5f && base.a < material.alpha.x)
    discard_fragment();

  const float4 metallicRoughness = metallicRoughnessMap.sample(materialSampler, metallicUv);
  const float metallic = saturate(metallicRoughness.b * material.factors.x);
  const float roughness = clamp(metallicRoughness.g * material.factors.y, kMinRoughness, 1.0f);
  float occlusion = mix(1.0f, occlusionMap.sample(materialSampler, input.uv1).r, material.factors.w);
  const float3 emissive = emissiveMap.sample(materialSampler, emissiveUv).rgb *
                          material.emissive.rgb * material.emissive.w;

  float3 geometricNormal = normalize(input.normal);
  const float3 view = normalize(frame.cameraPosition.xyz - input.world);
  if (material.alpha.z > 0.5f && dot(geometricNormal, view) < 0.0f) geometricNormal = -geometricNormal;
  float3 shadingNormal = geometricNormal;
  const float3 tangent = input.tangent.xyz;
  if (dot(tangent, tangent) > 1e-8f) {
    const float3 t = normalize(tangent - geometricNormal * dot(geometricNormal, tangent));
    const float3 b = cross(geometricNormal, t) * input.tangent.w;
    const float3 raw = normalMap.sample(materialSampler, normalUv).rgb * 2.0f - 1.0f;
    const float3 sampled = float3(raw.x * material.factors.z, raw.y * material.factors.z, raw.z);
    shadingNormal = normalize(t * sampled.x + b * sampled.y + geometricNormal * sampled.z);
  }

  const float3 diffuseColor = base.rgb * (1.0f - metallic);
  const float3 f0 = mix(float3(0.04f), base.rgb, metallic);
  const float normalDotView = saturate(dot(shadingNormal, view)) + 1e-5f;
  const float2 brdf = environmentBRDF(roughness, normalDotView);
  const EnergyCompensation energy = energyCompensation(f0, brdf);
  const float2 pixel = input.position.xy;
  const float sceneScale = max(frame.occlusion.w, 1e-3f);

  uint cascade = 0u;
  const float3 sunDirection = normalize(frame.sunDirection.xyz);
  float visible = 1.0f;
#ifdef BASALT_RAY_TRACING
  // Ray loops live in the entry: the compiler refuses to traverse an acceleration structure in a helper.
  const float3 rotation = frameRotation(frame.occlusion.z);
  const float noise = fract(interleavedGradientNoise(pixel) + rotation.x);
  const float3 rayOrigin = input.world + geometricNormal * (sceneScale * 2e-3f);
  const uint rayMask = max(uint(frame.environment.z), 1u); // Bit one is the ground plane.
  // Accept-any: stop at the first solid hit; masked candidates are alpha-tested.
  intersection_params anyHit;
  anyHit.accept_any_intersection(true);
  if (frame.rays.x > 1.5f) {
    const uint samples = max(uint(frame.rays.y), 1u);
    float lit = 0.0f;
    for (uint i = 0u; i < samples; ++i) {
      const float2 xi = float2(fract(noise + float(i) * 0.618034f),
                               fract(hashFloat(uint(pixel.x) * 1973u + uint(pixel.y) * 9277u + i * 26699u) +
                                     rotation.y));
      float3 direction = sunDirection;
      if (samples > 1u) direction = sampleCone(sunDirection, frame.rays.z, xi);
      ray towardsSun(rayOrigin, direction, sceneScale * 1e-3f, sceneScale * 1.0e4f);
      intersection_query<triangle_data, instancing> query(towardsSun, scene, rayMask, anyHit);
      while (query.next()) {
        if (query.get_candidate_intersection_type() == intersection_type::triangle &&
            candidateIsSolid(query.get_candidate_instance_id(), query.get_candidate_primitive_id(),
                             query.get_candidate_triangle_barycentric_coord(), traceInstances, materials,
                             indices, vertices, maps, tableSampler))
          query.commit_triangle_intersection();
      }
      lit += query.get_committed_intersection_type() == intersection_type::none ? 1.0f : 0.0f;
    }
    visible = lit / float(samples);
  } else
#endif
  // Mode zero is no shadow; the cascades are not drawn then and must not be read.
  if (frame.rays.x > 0.5f) {
    visible = cascadedSunVisibility(shadowMap, shadowSampler, cascadeRows, input.world,
                                    geometricNormal, sunDirection, input.viewDepth,
                                    frame.cascadeSplits, frame.shadowParameters, cascade);
  }
#ifdef BASALT_RAY_TRACING
  // Traced occlusion, min'd with the material's map. Contact mode fades an occluder by its
  // distance over a short radius. Sky-visibility mode counts a cosine-weighted ray as open only
  // if it leaves the model: the share of the environment the diffuse term can see, which is
  // what a path tracer's environment lighting sees. Its radius spans the scene and any hit ends
  // it. The ground plane is not an occluder there: the environment's own lower half already
  // stands for the ground in the irradiance this scales, as the lit plane does in a path tracer.
  if (frame.rays.w > 0.5f) {
    const bool skyVisibility = frame.rays.w > 1.5f;
    const uint samples = max(uint(frame.occlusion.y), 1u);
    const float radius = max(frame.occlusion.x, sceneScale * 1e-2f);
    intersection_params occlusionParams;
    uint occlusionMask = rayMask;
    if (skyVisibility) {
      occlusionParams.accept_any_intersection(true);
      occlusionMask = max(rayMask & ~kRayMaskGround, kRayMaskScene);
    }
    float open = 0.0f;
    for (uint i = 0u; i < samples; ++i) {
      const float2 xi = float2(fract(noise + float(i) * 0.618034f),
                               fract(hashFloat(uint(pixel.x) * 7919u + uint(pixel.y) * 104729u + i * 15485863u) +
                                     rotation.y));
      const float3 direction = cosineSampleHemisphere(geometricNormal, xi);
      ray probe(rayOrigin, direction, sceneScale * 1e-3f, radius);
      // Contact mode needs the nearest occluder for its falloff; sky visibility, any.
      intersection_query<triangle_data, instancing> query(probe, scene, occlusionMask, occlusionParams);
      while (query.next()) {
        if (query.get_candidate_intersection_type() == intersection_type::triangle &&
            candidateIsSolid(query.get_candidate_instance_id(), query.get_candidate_primitive_id(),
                             query.get_candidate_triangle_barycentric_coord(), traceInstances, materials,
                             indices, vertices, maps, tableSampler))
          query.commit_triangle_intersection();
      }
      if (query.get_committed_intersection_type() == intersection_type::none) open += 1.0f;
      else if (!skyVisibility) open += saturate(query.get_committed_distance() / radius);
    }
    occlusion = min(occlusion, open / float(samples));
  }
#endif

  const float3 sunRadiance = frame.sunColor.rgb * frame.sunColor.w;
  float3 colour = shadeDirect(shadingNormal, view, sunDirection, sunRadiance, diffuseColor, f0,
                              roughness) * visible;

  // Only the lights in this pixel's grid cell.
  const uint lightTotal = uint(frame.viewportAndLights.z);
  const bool clustered = frame.options.w > 0.5f;
  uint lightCount = lightTotal;
  uint clusterBase = 0u;
  if (clustered) {
    const uint tilesX = uint(frame.clusters.x);
    const uint tilesY = uint(frame.clusters.y);
    const uint slices = uint(frame.clusters.z);
    const uint capacity = uint(frame.clusters.w);
    const uint tileX = min(uint(pixel.x * frame.clusters.x / frame.viewportAndLights.x), tilesX - 1u);
    const uint tileY = min(uint(pixel.y * frame.clusters.y / frame.viewportAndLights.y), tilesY - 1u);
    // The same exponential spacing the culling pass laid the slices out with.
    const float nearPlane = frame.options.y;
    const float ratio = max(frame.options.z / nearPlane, 1.0f + 1e-4f);
    const float depth = max(input.viewDepth, nearPlane);
    const uint slice = min(uint(log(depth / nearPlane) / log(ratio) * frame.clusters.z), slices - 1u);
    const uint cell = (slice * tilesY + tileY) * tilesX + tileX;
    const uint cellCount = tilesX * tilesY * slices;
    lightCount = min(lightClusters[cell], capacity);
    clusterBase = cellCount + cell * capacity;
  }
  for (uint i = 0u; i < lightCount; ++i) {
    uint index = i;
    if (clustered) index = lightClusters[clusterBase + i];
    const Light light = lights[index];
    const float3 toLight = light.position.xyz - input.world;
    const float distanceToLight = length(toLight);
    if (distanceToLight <= 1e-4f) continue;
    const float3 lightDirection = toLight / distanceToLight;
    float attenuation = distanceAttenuation(distanceToLight, light.position.w);
    if (light.cone.y > 0.5f) {
      // dot(spot axis, light-to-surface); lightDirection points at the light.
      const float cosine = dot(normalize(light.direction.xyz), -lightDirection);
      const float t = saturate((cosine - light.cone.x) / max(light.direction.w - light.cone.x, 1e-4f));
      attenuation *= t * t;
    }
    if (attenuation <= 0.0f) continue;
    float lightVisible = 1.0f;
#ifdef BASALT_RAY_TRACING
    // Shadow ray to the light, stopping short of it.
    if (frame.options.x > 0.5f) {
      const float reach = max(distanceToLight - sceneScale * 4e-3f, sceneScale * 2e-3f);
      ray towardsLight(rayOrigin, lightDirection, sceneScale * 1e-3f, reach);
      intersection_query<triangle_data, instancing> lightQuery(towardsLight, scene, rayMask, anyHit);
      while (lightQuery.next()) {
        if (lightQuery.get_candidate_intersection_type() == intersection_type::triangle &&
            candidateIsSolid(lightQuery.get_candidate_instance_id(), lightQuery.get_candidate_primitive_id(),
                             lightQuery.get_candidate_triangle_barycentric_coord(), traceInstances, materials,
                             indices, vertices, maps, tableSampler))
          lightQuery.commit_triangle_intersection();
      }
      lightVisible = lightQuery.get_committed_intersection_type() == intersection_type::none ? 1.0f : 0.0f;
    }
#endif
    if (lightVisible <= 0.0f) continue;
    colour += shadeDirect(shadingNormal, view, lightDirection,
                          light.color.rgb * light.color.w * attenuation, diffuseColor, f0, roughness) *
              lightVisible;
  }
  // Multi-scatter compensation on the whole direct term; slightly over on diffuse, invisible.
  colour *= energy.direct.x * (1.0f - metallic) + metallic * dot(energy.direct, float3(1.0f / 3.0f));

  const float3 irradiance = shIrradiance(shadingNormal, frame.sh0, frame.sh1, frame.sh2, frame.sh3, frame.sh4, frame.sh5, frame.sh6, frame.sh7, frame.sh8);
  const float3 fresnel = fresnelSchlickRoughness(f0, normalDotView, roughness);
  const float3 ambientDiffuse = irradiance * diffuseColor * (float3(1.0f) - fresnel) * frame.environment.x;
  const float3 ambientMultiScatter = irradiance * energy.ambient * frame.environment.x;
  colour += (ambientDiffuse + ambientMultiScatter) * occlusion;
  colour += emissive;

  const float3 reflection = reflect(-view, shadingNormal);
  const float specularVisibility = specularOcclusion(normalDotView, occlusion, roughness) *
                                   horizonOcclusion(reflection, geometricNormal);
  // Environment intensity is applied in the resolve, so screen and scene hits keep their radiance.
  const float3 reflectionWeight = (fresnel * brdf.x + float3(brdf.y)) * specularVisibility;

  ForwardOutput out;
  out.color = float4(colour, material.alpha.y > 1.5f ? base.a : 1.0f);
  out.normalRoughness = float4(shadingNormal, roughness);
  // w carries depth: a blended surface writes none to the depth buffer.
  out.reflectionWeight = float4(reflectionWeight, input.position.z);
  const uint coverage = max(max(emissive.x, emissive.y), emissive.z) > 0.0f ? 4u :
                        material.alpha.y > 1.5f ? 2u : material.alpha.y > 0.5f ? 3u : 1u;
  out.baseMetallic = float4(base.rgb, metallic);
  // Reverse-Z depth is duplicated here so the hybrid shader stays within the 128-index
  // texture namespace while the integer identity carries the coverage class.
  out.geometricCoverage = float4(geometricNormal, input.position.z);
  out.emissive = float4(emissive, 0.0f);
  out.identity = uint4(input.identity.y, input.identity.x, primitive, coverage);

  // Debug views show the raw quantity.
  const uint debug = uint(frame.viewportAndLights.w);
  if (debug != 0u) {
    float4 shown = float4(colour, 1.0f);
    if (debug == 1u) shown = float4(base.rgb, 1.0f);
    else if (debug == 2u) shown = float4(shadingNormal * 0.5f + 0.5f, 1.0f);
    else if (debug == 3u) shown = float4(float3(metallic), 1.0f);
    else if (debug == 4u) shown = float4(float3(roughness), 1.0f);
    else if (debug == 5u) shown = float4(float3(occlusion), 1.0f);
    else if (debug == 6u) shown = float4(float3(visible), 1.0f);
    else if (debug == 7u) {
      const float3 tint = cascade == 0u ? float3(1.0f, 0.3f, 0.3f)
                          : cascade == 1u ? float3(0.3f, 1.0f, 0.3f)
                          : cascade == 2u ? float3(0.3f, 0.5f, 1.0f)
                                          : float3(1.0f, 1.0f, 0.3f);
      shown = float4(tint * (0.3f + 0.7f * visible), 1.0f);
    }
    else if (debug == 8u) shown = float4(emissive, 1.0f);
    else if (debug == 9u) shown = float4(float3(input.uv0, 0.0f), 1.0f);
    else if (debug == 10u) shown = float4(reflectionWeight, 1.0f);
    else if (debug == 13u) shown = float4(geometricNormal * 0.5f + 0.5f, 1.0f);
    else if (debug == 14u) shown = float4(float3(float(coverage) / 4.0f), 1.0f);
    else if (debug == 15u)
      shown = float4(hashFloat(input.identity.y), hashFloat(input.identity.x), hashFloat(primitive), 1.0f);
    out.color = shown;
    // Views 11 and 12 are the reflection's own and keep the weight so the resolve runs.
    if (debug != 11u && debug != 12u)
      out.reflectionWeight = float4(0.0f, 0.0f, 0.0f, input.position.z);
  }
  return out;
}
