// The scene as the rasteriser's ray queries see it: the acceleration structure plus what an
// alpha test needs. Passes hold one in a ParameterBlock; entries that trace nothing never reach
// it, so their modules and layouts carry none of it.
module basalt_trace;

import basalt_types;
import basalt_shading;

// A traced hit, shaded with the material's maps.
public struct HitSurface {
  float3 position;
  float3 normal;
  float3 baseColor;
  float3 emissive;
  float metallic;
  float roughness;
};

public struct TraceScene {
  RaytracingAccelerationStructure scene;
  StructuredBuffer<TraceInstance> traceInstances;
  StructuredBuffer<Material> materials;
  StructuredBuffer<uint> indices;
  StructuredBuffer<float> vertices;
  Texture2D<float4> maps[kHitTextureSlots];
  // Its own repeating sampler: a cut-out test must not depend on the shaded material's.
  SamplerState tableSampler;

  // Alpha test for a masked candidate, so rays pass through cut-outs.
  bool candidateIsSolid(uint instance, uint primitive, float2 barycentric) {
    const TraceInstance info = traceInstances[instance];
    const Material material = materials[info.material];
    if (material.alpha.y < 0.5f || material.alpha.y > 1.5f) return true;
    const uint base = info.firstIndex + primitive * 3u;
    const uint i0 = indices[base] + info.vertexOffset;
    const uint i1 = indices[base + 1u] + info.vertexOffset;
    const uint i2 = indices[base + 2u] + info.vertexOffset;
    const float b0 = 1.0f - barycentric.x - barycentric.y;
    const float2 uv = readUV(vertices, i0) * b0 + readUV(vertices, i1) * barycentric.x + readUV(vertices, i2) * barycentric.y;
    const float alpha =
        maps[NonUniformResourceIndex(info.slots & 0xFFu)].SampleLevel(tableSampler, uv, 0.0f).a * material.baseColorFactor.a;
    return alpha >= material.alpha.x;
  }

  // The nearest solid hit's distance along the ray, masked candidates alpha-tested; none when
  // the ray escapes. With firstHit, any solid hit ends the search (its distance is then not the
  // nearest).
  Optional<float> trace(float3 origin, float3 direction, float tMin, float tMax, uint mask, bool firstHit) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tMin;
    ray.TMax = tMax;
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(scene, firstHit ? RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH : RAY_FLAG_NONE, mask, ray);
    while (query.Proceed()) {
      if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
          candidateIsSolid(query.CandidateInstanceIndex(), query.CandidatePrimitiveIndex(),
                           query.CandidateTriangleBarycentrics()))
        query.CommitNonOpaqueTriangleHit();
    }
    if (query.CommittedStatus() == COMMITTED_NOTHING) return none;
    return query.CommittedRayT();
  }

  // The nearest solid surface along the ray, shaded; none when the ray escapes.
  Optional<HitSurface> nearestSurface(float3 origin, float3 direction, float tMin, float tMax, uint mask) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tMin;
    ray.TMax = tMax;
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, mask, ray);
    while (query.Proceed()) {
      if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
          candidateIsSolid(query.CandidateInstanceIndex(), query.CandidatePrimitiveIndex(),
                           query.CandidateTriangleBarycentrics()))
        query.CommitNonOpaqueTriangleHit();
    }
    if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) return none;
    return surfaceAt(query.CommittedInstanceIndex(), query.CommittedPrimitiveIndex(),
                     query.CommittedTriangleBarycentrics(), origin, direction, query.CommittedRayT(),
                     query.CommittedObjectToWorld3x4());
  }

  // Whether any solid surface lies between tMin and tMax.
  bool occluded(float3 origin, float3 direction, float tMin, float tMax, uint mask) {
    return trace(origin, direction, tMin, tMax, mask, true).hasValue;
  }

  // The surface a ray hit, normal facing the ray and normal-mapped. objectToWorld is the
  // instance's transform as the ray query reports it (3 rows of 4).
  HitSurface surfaceAt(uint instance, uint primitive, float2 barycentric, float3 rayOrigin, float3 rayDirection,
                              float distance, float3x4 objectToWorld) {
    const TraceInstance info = traceInstances[instance];
    const Material material = materials[info.material];
    const uint base = info.firstIndex + primitive * 3u;
    const uint i0 = indices[base] + info.vertexOffset;
    const uint i1 = indices[base + 1u] + info.vertexOffset;
    const uint i2 = indices[base + 2u] + info.vertexOffset;
    const float b0 = 1.0f - barycentric.x - barycentric.y;

    const float2 uv = readUV(vertices, i0) * b0 + readUV(vertices, i1) * barycentric.x + readUV(vertices, i2) * barycentric.y;
    const float3 objectNormal = normalize(readNormal(vertices, i0) * b0 + readNormal(vertices, i1) * barycentric.x +
                                          readNormal(vertices, i2) * barycentric.y);
    const float4 objectTangent =
        readTangent(vertices, i0) * b0 + readTangent(vertices, i1) * barycentric.x + readTangent(vertices, i2) * barycentric.y;
    // Rotation only; a non-uniform scale would need the inverse transpose.
    const float3x3 rotation = float3x3(objectToWorld[0].xyz, objectToWorld[1].xyz, objectToWorld[2].xyz);
    float3 normal = normalize(mul(rotation, objectNormal));
    if (dot(normal, -rayDirection) < 0.0f) normal = -normal;
    float3 tangent = mul(rotation, objectTangent.xyz);
    tangent = tangent - normal * dot(normal, tangent);
    // A collapsed tangent leaves the normal unperturbed, as the forward pass does.
    if (dot(tangent, tangent) > 1e-8f) {
      tangent = normalize(tangent);
      const float3 bitangent = cross(normal, tangent) * (objectTangent.w < 0.0f ? -1.0f : 1.0f);
      const float3 normalSample = maps[NonUniformResourceIndex((info.slots >> 24u) & 0xFFu)].SampleLevel(tableSampler, uv, 0.0f).xyz;
      const float3 perturbed = (normalSample * 2.0f - 1.0f) * float3(material.factors.z, material.factors.z, 1.0f);
      normal = normalize(tangent * perturbed.x + bitangent * perturbed.y + normal * perturbed.z);
    }

    // Slot zero is white, so a missing map leaves the factor alone.
    const float4 base4 = maps[NonUniformResourceIndex(info.slots & 0xFFu)].SampleLevel(tableSampler, uv, 0.0f) * material.baseColorFactor;
    const float4 metallicRoughness = maps[NonUniformResourceIndex((info.slots >> 8u) & 0xFFu)].SampleLevel(tableSampler, uv, 0.0f);
    const float3 emissive = maps[NonUniformResourceIndex((info.slots >> 16u) & 0xFFu)].SampleLevel(tableSampler, uv, 0.0f).rgb;
    HitSurface surface;
    surface.position = rayOrigin + rayDirection * distance;
    surface.normal = normal;
    surface.baseColor = base4.rgb;
    surface.emissive = emissive * material.emissive.rgb * material.emissive.w;
    surface.metallic = saturate(material.factors.x * metallicRoughness.b);
    surface.roughness = clamp(material.factors.y * metallicRoughness.g, kMinRoughness, 1.0f);
    return surface;
  }
};
