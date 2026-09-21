// Reflection resolve: per pixel the environment, a screen-space march or a ray, as radiance
// plus confidence; the composite adds it by weight and blurs by roughness through the mips.
// Included twice; reflect_rt.metal adds the traced mode.
#include "common.metal"
#ifdef BASALT_RAY_TRACING
#include <metal_raytracing>
using namespace metal::raytracing;
#endif

struct ReflectionUniforms {
  float4x4 viewProjection;
  float4x4 inverseViewProjection;
  float4 camera;       // xyz eye, w mode (0 environment, 1 screen space, 2 traced)
  float4 march;        // x max distance, y thickness, z steps, w near plane
  float4 sun;          // xyz direction towards the sun, w intensity
  float4 sunColor;     // rgb, w environment intensity
  float4 target;       // xy size, zw texel size
  float4 environment;  // x prefiltered mip count, y reflection mip count, z scene scale, w frame
  float4 debug;        // x debug view (11 resolved reflection, 12 trace confidence), y ray instance mask,
                       // z frames accumulated so far, w punctual light count
  float4 sampling;     // x traced reflections sample the GGX lobe (1) or take the mirror direction (0),
                       // y noise frame (0 when still), zw unused
  float4 sh0;
  float4 sh1;
  float4 sh2;
  float4 sh3;
  float4 sh4;
  float4 sh5;
  float4 sh6;
  float4 sh7;
  float4 sh8;
};

// Reverse-Z with an infinite far plane: linear depth is near over the stored value.
static float linearDepth(float stored, float nearPlane) {
  return nearPlane / max(stored, 1e-7f);
}

// World-space march against the depth buffer with a binary refine; returns the hit uv and a confidence.
static float screenSpaceTrace(float3 origin, float3 direction, float4x4 viewProjection,
                              texture2d<float> depthBuffer, sampler pointSampler, float maxDistance,
                              float thickness, uint steps, float nearPlane, float3 eye,
                              float jitter, thread float2 &hitUV) {
  const float stepLength = maxDistance / float(steps);
  float travelled = stepLength * (0.5f + jitter);
  float previousTravelled = 0.0f;
  float coarseGap = 0.0f;
  bool found = false;
  for (uint i = 0u; i < steps && !found; ++i) {
    const float3 point = origin + direction * travelled;
    const float4 clip = viewProjection * float4(point, 1.0f);
    if (clip.w <= 0.0f) return 0.0f;
    const float3 ndc = clip.xyz / clip.w;
    const float2 uv = float2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f);
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) return 0.0f;
    const float stored = depthBuffer.sample(pointSampler, uv, level(0.0f)).r;
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
    const float4 clip = viewProjection * float4(point, 1.0f);
    const float3 ndc = clip.xyz / max(clip.w, 1e-6f);
    const float2 uv = float2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f);
    const float stored = depthBuffer.sample(pointSampler, uv, level(0.0f)).r;
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

#ifdef BASALT_RAY_TRACING
struct HitSurface {
  float3 position;
  float3 normal;
  float3 baseColor;
  float3 emissive;
  float metallic;
  float roughness;
};

static HitSurface surfaceAt(uint instance, uint primitive, float2 barycentric, float3 rayOrigin,
                            float3 rayDirection, float distance, float4x3 objectToWorld,
                            const device PrimitiveInfo* primitives, const device Material* materials,
                            const device uint* indices, const device float* vertices,
                            array<texture2d<float>, kHitTextureSlots> maps, sampler materialSampler) {
  const PrimitiveInfo info = primitives[instance];
  const Material material = materials[info.material];
  const uint base = info.firstIndex + primitive * 3u;
  const uint i0 = indices[base] + info.vertexOffset;
  const uint i1 = indices[base + 1u] + info.vertexOffset;
  const uint i2 = indices[base + 2u] + info.vertexOffset;
  const float b0 = 1.0f - barycentric.x - barycentric.y;

  const float2 uv = readUV(vertices, i0) * b0 + readUV(vertices, i1) * barycentric.x +
                    readUV(vertices, i2) * barycentric.y;
  const float3 objectNormal = normalize(readNormal(vertices, i0) * b0 +
                                        readNormal(vertices, i1) * barycentric.x +
                                        readNormal(vertices, i2) * barycentric.y);
  const float4 objectTangent = readTangent(vertices, i0) * b0 + readTangent(vertices, i1) * barycentric.x +
                               readTangent(vertices, i2) * barycentric.y;
  // Rotation only; a non-uniform scale would need the inverse transpose.
  float3 normal = normalize(objectToWorld[0] * objectNormal.x + objectToWorld[1] * objectNormal.y +
                            objectToWorld[2] * objectNormal.z);
  if (dot(normal, -rayDirection) < 0.0f) normal = -normal;
  float3 tangent = objectToWorld[0] * objectTangent.x + objectToWorld[1] * objectTangent.y +
                   objectToWorld[2] * objectTangent.z;
  tangent = tangent - normal * dot(normal, tangent);
  // A collapsed tangent leaves the normal unperturbed, as the forward pass does.
  if (dot(tangent, tangent) > 1e-8f) {
    tangent = normalize(tangent);
    const float3 bitangent = cross(normal, tangent) * (objectTangent.w < 0.0f ? -1.0f : 1.0f);
    const float3 normalSample = maps[(info.slots >> 24u) & 0xFFu].sample(materialSampler, uv, level(0.0f)).xyz;
    const float3 perturbed = (normalSample * 2.0f - 1.0f) * float3(material.factors.z, material.factors.z, 1.0f);
    normal = normalize(tangent * perturbed.x + bitangent * perturbed.y + normal * perturbed.z);
  }

  // Slot zero is white, so a missing map leaves the factor alone.
  const float4 base4 = maps[info.slots & 0xFFu].sample(materialSampler, uv, level(0.0f)) *
                       material.baseColorFactor;
  const float4 metallicRoughness = maps[(info.slots >> 8u) & 0xFFu].sample(materialSampler, uv, level(0.0f));
  const float3 emissive = maps[(info.slots >> 16u) & 0xFFu].sample(materialSampler, uv, level(0.0f)).rgb;
  HitSurface surface;
  surface.position = rayOrigin + rayDirection * distance;
  surface.normal = normal;
  surface.baseColor = base4.rgb;
  surface.emissive = emissive * material.emissive.rgb * material.emissive.w;
  surface.metallic = saturate(material.factors.x * metallicRoughness.b);
  surface.roughness = clamp(material.factors.y * metallicRoughness.g, kMinRoughness, 1.0f);
  return surface;
}

static float3 shadeSurface(HitSurface surface, float3 rayDirection, float visible,
                           constant ReflectionUniforms& uniforms, const device Light* lights,
                           texturecube<float> prefilteredCube, sampler clampSampler) {
  const float3 view = -rayDirection;
  const float3 diffuseColor = surface.baseColor * (1.0f - surface.metallic);
  const float3 f0 = mix(float3(0.04f), surface.baseColor, surface.metallic);
  const float3 sunDirection = normalize(uniforms.sun.xyz);
  float3 colour = shadeDirect(surface.normal, view, sunDirection, uniforms.sunColor.rgb * uniforms.sun.w,
                              diffuseColor, f0, surface.roughness) * visible;
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
    colour += shadeDirect(surface.normal, view, lightDirection,
                          light.color.rgb * light.color.w * attenuation, diffuseColor, f0, surface.roughness);
  }
  const float normalDotView = saturate(dot(surface.normal, view)) + 1e-5f;
  const float2 brdf = environmentBRDF(surface.roughness, normalDotView);
  const float3 fresnel = fresnelSchlickRoughness(f0, normalDotView, surface.roughness);
  const float3 irradiance = shIrradiance(surface.normal, uniforms.sh0, uniforms.sh1, uniforms.sh2, uniforms.sh3, uniforms.sh4, uniforms.sh5, uniforms.sh6, uniforms.sh7, uniforms.sh8);
  const float3 reflected = reflect(rayDirection, surface.normal);
  const float3 prefiltered = prefilteredCube.sample(clampSampler, reflected,
                                                    level(surface.roughness * (uniforms.environment.x - 1.0f))).rgb;
  colour += (irradiance * diffuseColor * (float3(1.0f) - fresnel) +
             prefiltered * (fresnel * brdf.x + float3(brdf.y))) * uniforms.sunColor.w;
  colour += surface.emissive;
  return colour;
}
#endif

kernel void RESOLVE_REFLECTIONS(texture2d<float> depthBuffer [[texture(0)]],
                                texture2d<float> normalRoughness [[texture(1)]],
                                texture2d<float> reflectionWeight [[texture(2)]],
                                texture2d<float> sceneColor [[texture(3)]],
                                texturecube<float> prefilteredCube [[texture(4)]],
                                texture2d<float, access::write> reflection [[texture(6)]],
#ifdef BASALT_RAY_TRACING
                                array<texture2d<float>, kHitTextureSlots> maps [[texture(7)]],
#endif
                                constant ReflectionUniforms& uniforms [[buffer(0)]],
#ifdef BASALT_RAY_TRACING
                                instance_acceleration_structure scene [[buffer(1)]],
                                const device PrimitiveInfo* primitives [[buffer(2)]],
                                const device Material* materials [[buffer(3)]],
                                const device uint* indices [[buffer(4)]],
                                const device float* vertices [[buffer(5)]],
                                const device Light* lights [[buffer(6)]],
#endif
                                sampler clampSampler [[sampler(0)]],
                                sampler pointSampler [[sampler(1)]],
                                sampler materialSampler [[sampler(2)]],
                                uint2 id [[thread_position_in_grid]]) {
  if (float(id.x) >= uniforms.target.x || float(id.y) >= uniforms.target.y) return;
  const float2 uv = (float2(float(id.x), float(id.y)) + 0.5f) * uniforms.target.zw;

  float depth = depthBuffer.sample(pointSampler, uv, level(0.0f)).r;
  const float4 weight = reflectionWeight.sample(pointSampler, uv, level(0.0f));
  // A blended surface's depth rides in weight.w and reads larger (nearer) under reverse Z;
  // the margin covers half-float rounding.
  if (weight.w > depth * 1.001f) depth = weight.w;
  // The sky, and any pixel whose reflection would be multiplied by nothing.
  if (depth <= 0.0f || dot(weight.rgb, float3(1.0f)) <= 1e-5f) {
    reflection.write(float4(0.0f), id);
    return;
  }

  const float4 nr = normalRoughness.sample(pointSampler, uv, level(0.0f));
  const float3 normal = normalize(nr.xyz);
  const float roughness = nr.w;
  // The y flip: the forward pass renders Y up in clip space; the grid's row 0 is the top.
  const float3 world = reconstructWorld(float2(uv.x, 1.0f - uv.y), depth, uniforms.inverseViewProjection);
  const float3 view = normalize(uniforms.camera.xyz - world);
  const float3 direction = reflect(-view, normal);

  const float envMip = roughness * (uniforms.environment.x - 1.0f);
  float3 fallback = prefilteredCube.sample(clampSampler, direction, level(envMip)).rgb * uniforms.sunColor.w;
  float3 colour = fallback;
  float confidence = 0.0f;
  const uint mode = uint(uniforms.camera.w);

  if (mode == 1u) {
    float2 hitUV = uv;
    const float jitter = animatedNoise(float2(float(id.x), float(id.y)), uniforms.sampling.y);
    confidence = screenSpaceTrace(world + normal * (uniforms.environment.z * 1e-3f), direction,
                                  uniforms.viewProjection, depthBuffer, pointSampler,
                                  uniforms.march.x, uniforms.march.y, uint(uniforms.march.z),
                                  uniforms.march.w, uniforms.camera.xyz, jitter, hitUV);
    if (confidence > 0.0f) {
      const float3 hit = sceneColor.sample(clampSampler, hitUV, level(0.0f)).rgb;
      colour = mix(fallback, hit, confidence);
    }
  }
#ifdef BASALT_RAY_TRACING
  else if (mode == 2u) {
    // Stepped so masked candidates are alpha-tested.
    const float scale = uniforms.environment.z;
    // While accumulating, sample the GGX lobe per frame; otherwise the mirror direction and a blurred mip.
    float3 rayDirection = direction;
    bool sendRay = true;
    bool keepEscape = true;
    if (uniforms.sampling.x > 0.5f) {
      const float2 pixel = float2(float(id.x), float(id.y));
      const float3 rotation = frameRotation(uniforms.sampling.y);
      const float2 xi = float2(fract(interleavedGradientNoise(pixel) + rotation.x),
                               fract(hashFloat(id.x * 1973u + id.y * 9277u) + rotation.y));
      float3 sampled = reflect(-view, importanceSampleGGX(xi, normal, roughness));
      // Redraw once below the horizon, so grazing rough surfaces do not sharpen to the mirror.
      if (dot(sampled, normal) <= 1e-3f) {
        const float2 xi2 = float2(fract(xi.x + 0.618034f), fract(xi.y + 0.324717f));
        sampled = reflect(-view, importanceSampleGGX(xi2, normal, roughness));
      }
      // Every ray is traced, and one that hits is kept whatever its angle: the environment
      // cannot stand in for the scene. An escaping ray is kept with probability cosine, the
      // prefiltered environment standing in otherwise, which matches how that is weighted.
      const float cosine = dot(sampled, normal);
      // Its own sequence, uncorrelated with the direction draw.
      const float accept = fract(hashFloat(id.x * 26699u + id.y * 15485863u) + rotation.z);
      sendRay = cosine > 1e-3f;
      keepEscape = accept <= cosine;
      if (sendRay) rayDirection = sampled;
    }
    if (sendRay) {
      ray r(world + normal * (scale * 2e-3f), rayDirection, scale * 1e-3f, uniforms.march.x);
      intersection_query<triangle_data, instancing> query(r, scene, uint(uniforms.debug.y));
      while (query.next()) {
        if (query.get_candidate_intersection_type() == intersection_type::triangle) {
          if (candidateIsSolid(query.get_candidate_instance_id(), query.get_candidate_primitive_id(),
                               query.get_candidate_triangle_barycentric_coord(), primitives, materials,
                               indices, vertices, maps, materialSampler))
            query.commit_triangle_intersection();
        }
      }
      if (query.get_committed_intersection_type() == intersection_type::triangle) {
        const HitSurface surface = surfaceAt(
            query.get_committed_instance_id(), query.get_committed_primitive_id(),
            query.get_committed_triangle_barycentric_coord(), r.origin, rayDirection,
            query.get_committed_distance(), query.get_committed_object_to_world_transform(),
            primitives, materials, indices, vertices, maps, materialSampler);
        // Sun shadow ray from the hit, alpha-tested.
        ray towardsSun(surface.position + surface.normal * (scale * 2e-3f), normalize(uniforms.sun.xyz),
                       scale * 1e-3f, scale * 1.0e4f);
        intersection_params anyHit;
        anyHit.accept_any_intersection(true);
        intersection_query<triangle_data, instancing> shadowQuery(towardsSun, scene,
                                                                  uint(uniforms.debug.y), anyHit);
        while (shadowQuery.next()) {
          if (shadowQuery.get_candidate_intersection_type() == intersection_type::triangle &&
              candidateIsSolid(shadowQuery.get_candidate_instance_id(), shadowQuery.get_candidate_primitive_id(),
                               shadowQuery.get_candidate_triangle_barycentric_coord(), primitives, materials,
                               indices, vertices, maps, materialSampler))
            shadowQuery.commit_triangle_intersection();
        }
        const float visible =
            shadowQuery.get_committed_intersection_type() == intersection_type::none ? 1.0f : 0.0f;
        colour = shadeSurface(surface, rayDirection, visible, uniforms, lights, prefilteredCube,
                              clampSampler);
        confidence = 1.0f;
      } else if (uniforms.sampling.x > 0.5f && keepEscape) {
        // An escaped sampled ray reads the sharp environment: the lobe itself is being averaged.
        colour = prefilteredCube.sample(clampSampler, rayDirection, level(0.0f)).rgb * uniforms.sunColor.w;
        confidence = 1.0f;
      }
    }
  }
#endif
  reflection.write(float4(colour, confidence), id);
}

// Adds the resolved reflection by weight, blurred through the mips by roughness;
// the environment fallback is already prefiltered.
kernel void COMPOSITE_REFLECTIONS(texture2d<float> sceneColor [[texture(0)]],
                                  texture2d<float> reflectionWeight [[texture(1)]],
                                  texture2d<float> normalRoughness [[texture(2)]],
                                  texture2d<float> reflection [[texture(3)]],
                                  texture2d<float, access::write> lit [[texture(4)]],
                                  constant ReflectionUniforms& uniforms [[buffer(0)]],
                                  sampler clampSampler [[sampler(0)]],
                                  sampler pointSampler [[sampler(1)]],
                                  uint2 id [[thread_position_in_grid]]) {
  if (float(id.x) >= uniforms.target.x || float(id.y) >= uniforms.target.y) return;
  const float2 uv = (float2(float(id.x), float(id.y)) + 0.5f) * uniforms.target.zw;
  const float3 colour = sceneColor.sample(pointSampler, uv, level(0.0f)).rgb;
  const float3 weight = reflectionWeight.sample(pointSampler, uv, level(0.0f)).rgb;
  const float roughness = normalRoughness.sample(pointSampler, uv, level(0.0f)).w;

  // No mip blur when the rays sampled the lobe themselves.
  const bool sampledLobe = uniforms.sampling.x > 0.5f && uniforms.camera.w > 1.5f;
  const float mip = sampledLobe ? 0.0f : sqrt(roughness) * (uniforms.environment.y - 1.0f);
  const float4 blurred = reflection.sample(clampSampler, uv, level(mip));
  const float4 sharp = reflection.sample(pointSampler, uv, level(0.0f));
  const float3 resolved = mix(sharp.rgb, blurred.rgb, saturate(blurred.a));
  float3 shown = colour + weight * resolved;
  // Debug views 11 and 12: the resolved reflection and its confidence.
  if (uniforms.debug.x > 11.5f) shown = float3(sharp.a);
  else if (uniforms.debug.x > 10.5f) shown = resolved;
  lit.write(float4(shown, 1.0f), id);
}
