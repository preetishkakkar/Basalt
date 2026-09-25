// The GPU's side of the path tracer's interfaces: the hit texture table and the environment as
// bound textures, and the hardware intersector through ray queries.
module pt_gpu;

import pt_path;
import pt_raycone;
import pt_bvh;

// The path tracers' hardware rays drop single-sided back faces in the traversal. The host
// disables facing culling only on double-sided and blended instances (a blended candidate marks
// the hit ambiguous whichever way it faces), so opaque single-sided geometry is built opaque and
// never reaches a candidate test. ptCandidateSolid still tests facing for the other intersectors.
public static const uint kPtRayFlags = RAY_FLAG_CULL_BACK_FACING_TRIANGLES;

// The hit texture table with the six glTF sampler modes, and the environment's equirect.
public struct GpuMaps : IPtMaps {
  Texture2D<float4> table[kHitTextureSlots];
  // Wrap by sampler code & 3 (0 repeat, 1 clamp, 2 mirror), filtered linearly or, with
  // (code & 16), nearest; mips always linear.
  SamplerState repeatLinear, clampLinear, mirrorLinear;
  SamplerState repeatNearest, clampNearest, mirrorNearest;
  Texture2D<float4> environmentMap;
  SamplerState environmentSampler;

  float4 sampleTexture(uint slot, float2 uv, uint samplerCode, float lodBase) {
    const Texture2D<float4> map = table[NonUniformResourceIndex(slot)];
    float lod = 0.0f;
    if (lodBase > kPtLevelZero) {
      uint width, height, levels;
      map.GetDimensions(0u, width, height, levels);
      lod = ptTextureLevel(lodBase, float(width), float(height), float(levels));
    }
    const uint wrap = samplerCode & 3u;
    if ((samplerCode & 16u) != 0u) {
      if (wrap == 1u) return map.SampleLevel(clampNearest, uv, lod);
      if (wrap == 2u) return map.SampleLevel(mirrorNearest, uv, lod);
      return map.SampleLevel(repeatNearest, uv, lod);
    }
    if (wrap == 1u) return map.SampleLevel(clampLinear, uv, lod);
    if (wrap == 2u) return map.SampleLevel(mirrorLinear, uv, lod);
    return map.SampleLevel(repeatLinear, uv, lod);
  }

  float textureFootprint(uint slot, float lodBase) {
    uint width, height, levels;
    table[NonUniformResourceIndex(slot)].GetDimensions(0u, width, height, levels);
    return ptTextureFootprint(lodBase, float(width), float(height));
  }

  float3 sampleEnvironment(float2 uv) { return environmentMap.SampleLevel(environmentSampler, uv, 0.0f).xyz; }
};

// The hardware acceleration structure through ray queries. Masked and blended candidates run
// the alpha test every intersector uses; an occlusion ray stops at the first solid one.
public struct RayQueryTracer<M : IPtMaps> : IPtTracer {
  RaytracingAccelerationStructure scene;
  PtGeometry geometry;
  M maps;

  PtHit trace(float3 origin, float3 direction, float tMax, uint mask, uint seed, float2 cone, bool anyHit) {
    PtHit hit = PtHit(0.0f);
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0f;
    ray.TMax = tMax;
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(scene, kPtRayFlags | (anyHit ? RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH : RAY_FLAG_NONE), mask, ray);
    while (query.Proceed()) {
      if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) continue;
      const uint instance = query.CandidateInstanceIndex();
      if ((geometry.traceInstances[instance].flags & kInstanceBlended) != 0u) hit.ambiguous = 1u;
      if (ptCandidateSolid(geometry, maps, instance, query.CandidatePrimitiveIndex(), query.CandidateTriangleBarycentrics(),
                           query.CandidateTriangleFrontFace() ? 1u : 0u, seed, direction,
                           ptConeWidthOrLevelZero(cone, query.CandidateTriangleRayT())))
        query.CommitNonOpaqueTriangleHit();
    }
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
      hit.found = 1u;
      hit.t = query.CommittedRayT();
      hit.barycentric = query.CommittedTriangleBarycentrics();
      hit.instance = query.CommittedInstanceIndex();
      hit.primitive = query.CommittedPrimitiveIndex();
    }
    return hit;
  }
};
