// The GPU path tracer: the shared path loop (pt/integrator.inc) with one trace macro, over
// the software BVH (path_trace.metal) or, with BASALT_RAY_TRACING, the rasteriser's own TLAS
// through ray queries (path_trace_rt.metal). One thread per pixel, `image.w` samples per
// dispatch, added to float accumulators; this frame's lit image receives the mean.
#include "shared/prelude.h"
#include "pt/path.h"
#ifdef BASALT_RAY_PIPELINE
#include <m2v_ray_pipeline>
#elif defined(BASALT_RAY_TRACING)
#include <metal_raytracing>
using namespace metal::raytracing;
#endif

#define PT_TEXTURE_PARAMS array<texture2d<float>, kHitTextureSlots> maps, sampler materialSampler
#define PT_TEXTURE_ARGS maps, materialSampler
#include "pt/texture_msl.h"

#define PT_ENVIRONMENT_PARAMS texture2d<float> environmentMap, sampler environmentSampler
#define PT_ENVIRONMENT_ARGS environmentMap, environmentSampler
inline float3 ptSampleEnvironment(PT_ENVIRONMENT_PARAMS, float2 uv) {
  return environmentMap.sample(environmentSampler, uv, level(0.0f)).xyz;
}

#include "pt/bvh.h"
#ifdef BASALT_WIDE_BVH
#include "pt/wide_bvh.h"
#endif
#include "pt/surface.h"
#include "pt/bsdf.h"
#include "pt/reconstruction.h"
#include "pt/lights.h"
#ifndef PT_PRIMARY_GBUFFER
#include "pt/wavefront.h"  // ReSTIR DI's per-pixel results
#endif

#ifdef BASALT_RAY_PIPELINE
struct PtPipelinePayload {
  uint ambiguous;
  float t;
  float2 barycentric;
  uint instance;
  uint primitive;
  uint found;
  uint seed;
  uint anyHit;
  float2 cone;  // the ray cone where the ray left, for alpha tests
};

[[m2v::ray_stage(miss)]]
void path_pipeline_miss(thread PtPipelinePayload &payload [[m2v::payload]]) {
  payload.found = 0u;
}

[[m2v::ray_stage(closest_hit)]]
void path_pipeline_closest(thread PtPipelinePayload &payload [[m2v::payload]],
                           float2 barycentric [[m2v::hit_attribute]]) {
  payload.t = m2v::ray_tmax();
  payload.barycentric = barycentric;
  payload.instance = m2v::instance_id();
  payload.primitive = m2v::primitive_id();
  payload.found = 1u;
}

[[m2v::ray_stage(any_hit)]]
void path_pipeline_alpha(thread PtPipelinePayload &payload [[m2v::payload]],
                         float2 barycentric [[m2v::hit_attribute]],
                         const device TraceInstance *traceInstances [[buffer(1)]],
                         const device Material *materials [[buffer(2)]],
                         const device uint *indices [[buffer(3)]],
                         const device float *vertices [[buffer(4)]],
                         array<texture2d<float>, kHitTextureSlots> maps [[texture(7)]],
                         sampler environmentSampler [[sampler(0)]],
                         sampler materialSampler [[sampler(1)]]) {
  const uint instance = m2v::instance_id();
  if ((traceInstances[instance].flags & kInstanceBlended) != 0u) payload.ambiguous = 1u;
  const uint front = (m2v::hit_kind() == m2v::hit_kind_front_facing_triangle ? 1u : 0u);
  if (!ptCandidateSolid(PT_SCENE_ARGS, PT_TEXTURE_ARGS, instance, m2v::primitive_id(),
                        barycentric, front, payload.seed, m2v::world_ray_direction(),
                        ptConeWidthOrLevelZero(payload.cone, m2v::ray_tmax())))
    m2v::ignore_intersection();
  if (payload.anyHit != 0u) {
    payload.t = m2v::ray_tmax();
    payload.barycentric = barycentric;
    payload.instance = instance;
    payload.primitive = m2v::primitive_id();
    payload.found = 1u;
    m2v::terminate_ray();
  }
}
#endif

#ifdef BASALT_RAY_PIPELINE
[[m2v::ray_stage(raygen)]]
#else
kernel
#endif
void PATH_TRACE_ENTRY(constant PathUniforms& uniforms [[buffer(0)]],
                             const device TraceInstance* traceInstances [[buffer(1)]],
                             const device Material* materials [[buffer(2)]],
                             const device uint* indices [[buffer(3)]],
                             const device float* vertices [[buffer(4)]],
                             const device Light* lights [[buffer(5)]],
                             const device float* environmentDistribution [[buffer(6)]],
                             const device float* specularAlbedo [[buffer(7)]],
#if defined(BASALT_RAY_TRACING) || defined(BASALT_RAY_PIPELINE)
                             instance_acceleration_structure scene [[buffer(8)]],
#elif defined(BASALT_WIDE_BVH)
                             const device PtWideNode* bvhNodes [[buffer(8)]],
                             const device float4* bvhTriangles [[buffer(9)]],
#else
                             const device float4* bvhNodes [[buffer(8)]],
                             const device float4* bvhTriangles [[buffer(9)]],
#endif
                             device PtReconstructionSample* reconstructionSamples [[buffer(10)]],
                             const device PtEmissiveTriangle* emissiveTriangles [[buffer(11)]],
#ifdef PT_PRIMARY_GBUFFER
                             constant float4x4& hybridInverseViewProjection [[buffer(12)]],
#else
                             // ReSTIR DI: the primary vertex's visible direct light
                             // and its diffuse part, per pixel, from the reservoir passes.
                             const device PtWaveResult* restirResults [[buffer(13)]],
                             const device PtWaveGuide* restirGuides [[buffer(14)]],
#endif
                             texture2d<float> environmentMap [[texture(0)]],
                             texture2d<float, access::read_write> accumulation [[texture(1)]],
                             texture2d<float, access::write> output [[texture(2)]],
#ifndef PT_NO_AUX_ACCUMULATION
                             texture2d<float, access::read_write> albedoAccumulation [[texture(3)]],
                             texture2d<float, access::read_write> normalAccumulation [[texture(4)]],
#endif
                             array<texture2d<float>, kHitTextureSlots> maps [[texture(7)]],
#ifdef PT_PRIMARY_GBUFFER
                             texture2d<float> hybridNormalRoughness [[texture(3)]],
                             texture2d<float> hybridBaseMetallic [[texture(4)]],
                             texture2d<float> hybridGeometricDepth [[texture(5)]],
                             texture2d<uint> hybridIdentity [[texture(127)]],
#endif
                             sampler environmentSampler [[sampler(0)]],
                             sampler materialSampler [[sampler(1)]]
#ifndef BASALT_RAY_PIPELINE
                             , uint2 id [[thread_position_in_grid]]
#endif
                             ) {
#ifdef BASALT_RAY_PIPELINE
  const uint2 id = m2v::launch_id().xy;
#endif
  if (float(id.x) >= uniforms.image.x || float(id.y) >= uniforms.image.y) return;
  const uint pixelX = id.x;
  const uint pixelY = id.y;
  const uint firstSample = uniforms.counts.w;
  // Zero samples only rewrites the mean: the frame's lit image still needs it.
  const uint samples = uint(uniforms.image.w);

#ifdef PT_PRIMARY_GBUFFER
  const float4 primaryGeometric = hybridGeometricDepth.read(id);
  const float primaryDepth = primaryGeometric.w;
  const uint4 primaryIdentity = hybridIdentity.read(id);
  // Blended and masked/subpixel coverage takes the ordinary traced-primary fallback.
  bool primaryAvailable = primaryDepth > 0.0f && primaryIdentity.w == 1u;
  PtSurface primarySurface;
  PtHit primaryHit;
  primarySurface.position = float3(0.0f);
  primarySurface.geometricNormal = float3(0.0f, 1.0f, 0.0f);
  primarySurface.normal = primarySurface.geometricNormal;
  primarySurface.baseColor = float3(0.0f);
  primarySurface.emissive = float3(0.0f);
  primarySurface.metallic = 0.0f;
  primarySurface.roughness = 1.0f;
  ptClearV7Layers(primarySurface, 1.5f, 1.0f);
  // The raster G-buffer has no transmission or clearcoat layer (the rasteriser does not render
  // them) and no barycentrics to sample their textures: such pixels trace their primary ray,
  // as blended coverage does.
  float4 primaryTransmission = float4(0.0f, 1.5f, 1.0f, 0.0f);
  if (primaryAvailable) {
    primaryTransmission = materials[primaryIdentity.y].transmission;
    if (primaryTransmission.x > 0.0f || materials[primaryIdentity.y].clearcoat.x > 0.0f) primaryAvailable = false;
  }
  primaryHit.ambiguous = 0u;
  primaryHit.t = 0.0f;
  primaryHit.barycentric = float2(0.0f);
  primaryHit.instance = 0u;
  primaryHit.primitive = 0u;
  primaryHit.found = 0u;
  primaryHit.nodeVisits = 0u;
  primaryHit.triangleTests = 0u;
  if (primaryAvailable) {
    const float2 primaryUv = (float2(id) + 0.5f) / uniforms.image.xy;
    const float4 primaryClip = float4(primaryUv.x * 2.0f - 1.0f,
                                     (1.0f - primaryUv.y) * 2.0f - 1.0f,
                                     primaryDepth, 1.0f);
    const float4 primaryWorld = hybridInverseViewProjection * primaryClip;
    const float4 primaryNormal = hybridNormalRoughness.read(id);
    const float4 primaryBase = hybridBaseMetallic.read(id);
    primarySurface.position = primaryWorld.xyz / primaryWorld.w;
    primarySurface.geometricNormal = normalize(primaryGeometric.xyz);
    primarySurface.normal = normalize(primaryNormal.xyz);
    primarySurface.baseColor = primaryBase.xyz;
    primarySurface.emissive = float3(0.0f);
    primarySurface.metallic = primaryBase.w;
    primarySurface.roughness = primaryNormal.w;
    // The raster faces every primary towards the eye, so the ray enters from the front.
    ptClearV7Layers(primarySurface, ptMaterialIor(primaryTransmission), 1.0f);
    primaryHit.t = length(primarySurface.position - uniforms.cameraPosition.xyz);
    primaryHit.instance = primaryIdentity.x;
    primaryHit.primitive = primaryIdentity.z;
    primaryHit.found = 1u;
    primaryHit.nodeVisits = 0u;
    primaryHit.triangleTests = 0u;
  }
#endif

  float3 sum = float3(0.0f);
#ifndef PT_NO_AUX_ACCUMULATION
  float3 albedoSum = float3(0.0f);
  float3 normalSum = float3(0.0f);
#endif
  for (uint s = 0u; s < samples; ++s) {
    const uint sampleIndex = firstSample + s;
    float3 pathRadiance = float3(0.0f);
    float3 pathAlbedo = float3(0.0f);
    float3 pathNormal = float3(0.0f);
    PtReconstructionSample pathGuide = ptEmptyReconstructionSample();
    uint2 pathTraversalCosts = uint2(0u);
#define PT_ACCUMULATE_COST(hit) pathTraversalCosts = pathTraversalCosts + uint2(hit.nodeVisits, hit.triangleTests)
#ifdef BASALT_RAY_PIPELINE
#define PT_TRACE(origin, direction, tMax, mask, seedValue, coneValue, anyHitValue, hit)           \
  {                                                                                                \
    PtPipelinePayload ptPayload = {};                                                              \
    ptPayload.seed = seedValue;                                                                    \
    ptPayload.anyHit = anyHitValue;                                                                \
    ptPayload.cone = coneValue;                                                                    \
    raytracing::ray ptRay(origin, direction, 0.0f, tMax);                                         \
    m2v::trace_ray(scene, m2v::ray_flags::none, mask, 0u, 1u, 0u, ptRay, ptPayload);              \
    hit.ambiguous = ptPayload.ambiguous;                                                           \
    hit.t = ptPayload.t;                                                                           \
    hit.barycentric = ptPayload.barycentric;                                                       \
    hit.instance = ptPayload.instance;                                                             \
    hit.primitive = ptPayload.primitive;                                                           \
    hit.found = ptPayload.found;                                                                   \
    hit.nodeVisits = 0u;                                                                           \
    hit.triangleTests = 0u;                                                                        \
  }
#elif defined(BASALT_RAY_TRACING)
    // Masked and blended candidates run the same test every backend uses; an occlusion ray
    // stops at the first solid one.
#define PT_TRACE(origin, direction, tMax, mask, seed, cone, anyHit, hit)                           \
  {                                                                                                \
    ray ptRay(origin, direction, 0.0f, tMax);                                                      \
    intersection_params ptParams;                                                                  \
    ptParams.accept_any_intersection(anyHit != 0u);                                                \
    intersection_query<triangle_data, instancing> ptQuery(ptRay, scene, mask, ptParams);          \
    while (ptQuery.next()) {                                                                       \
      if (ptQuery.get_candidate_intersection_type() == intersection_type::triangle &&              \
          (traceInstances[ptQuery.get_candidate_instance_id()].flags & kInstanceBlended) != 0u)    \
        hit.ambiguous = 1u;                                                                        \
      if (ptQuery.get_candidate_intersection_type() == intersection_type::triangle &&              \
          ptCandidateSolid(PT_SCENE_ARGS, PT_TEXTURE_ARGS, ptQuery.get_candidate_instance_id(),    \
                           ptQuery.get_candidate_primitive_id(),                                   \
                           ptQuery.get_candidate_triangle_barycentric_coord(),                     \
                           (ptQuery.is_candidate_triangle_front_facing() ? 1u : 0u), seed, direction,\
                           ptConeWidthOrLevelZero(cone, ptQuery.get_candidate_triangle_distance()))) \
        ptQuery.commit_triangle_intersection();                                                    \
    }                                                                                              \
    if (ptQuery.get_committed_intersection_type() == intersection_type::triangle) {                \
      hit.found = 1u;                                                                              \
      hit.t = ptQuery.get_committed_distance();                                                    \
      hit.barycentric = ptQuery.get_committed_triangle_barycentric_coord();                       \
      hit.instance = ptQuery.get_committed_instance_id();                                          \
      hit.primitive = ptQuery.get_committed_primitive_id();                                        \
    }                                                                                              \
    hit.nodeVisits = 0u;                                                                           \
    hit.triangleTests = 0u;                                                                        \
  }
#elif defined(BASALT_WIDE_BVH)
#define PT_TRACE(origin, direction, tMax, mask, seed, cone, anyHit, hit)                           \
  {                                                                                                \
    hit = ptTraceWideBvh(bvhNodes, bvhTriangles, PT_SCENE_ARGS, PT_TEXTURE_ARGS, origin, direction, \
                         tMax, mask, seed, cone, anyHit);                                          \
    PT_ACCUMULATE_COST(hit);                                                                       \
  }
#else
#define PT_TRACE(origin, direction, tMax, mask, seed, cone, anyHit, hit)                           \
  {                                                                                                \
    hit = ptTraceBvh(bvhNodes, bvhTriangles, PT_SCENE_ARGS, PT_TEXTURE_ARGS, origin, direction,    \
                     tMax, mask, seed, cone, anyHit);                                               \
    PT_ACCUMULATE_COST(hit);                                                                       \
  }
#endif
#ifndef PT_PRIMARY_GBUFFER
#define PT_RESTIR_DIRECT xyz(restirResults[pixelY * uint(uniforms.image.x) + pixelX].radiance)
#define PT_RESTIR_DIFFUSE xyz(restirGuides[pixelY * uint(uniforms.image.x) + pixelX].diffuseRadiance)
#endif
#include "pt/integrator.inc"
#ifndef PT_PRIMARY_GBUFFER
#undef PT_RESTIR_DIRECT
#undef PT_RESTIR_DIFFUSE
#endif
#ifndef BASALT_RAY_TRACING
    if (uniforms.image.z > 1.5f) {
      const float cost = uniforms.image.z < 2.5f ? float(pathTraversalCosts.x) / 128.0f
                                                  : float(pathTraversalCosts.y) / 64.0f;
      // A fixed false-colour scale makes captures comparable between runs: black is no
      // work, blue-to-green is below budget, and red is at or above the declared scale.
      const float value = saturate(cost);
      pathRadiance = float3(saturate(value * 3.0f - 1.5f),
                            saturate(1.0f - abs(value * 3.0f - 1.5f)),
                            saturate(1.5f - value * 3.0f));
      pathAlbedo = float3(0.0f);
      pathNormal = float3(0.0f);
    }
#endif
#undef PT_ACCUMULATE_COST
    // One coherent, genuinely fresh path per reconstruction update. Multi-SPP batches
    // still use every path in the independent raw accumulator.
    // Mode 1 only: modes 2 and 3 are BVH cost views, whose guide binding is one element.
    if (uniforms.image.z > 0.5f && uniforms.image.z < 1.5f && s + 1u == samples)
      reconstructionSamples[id.y * uint(uniforms.image.x) + id.x] = pathGuide;
    // A non-finite path gets no weight rather than poisoning the pixel for good.
    if (!(isnan(pathRadiance.x) || isnan(pathRadiance.y) || isnan(pathRadiance.z) || isinf(pathRadiance.x) ||
          isinf(pathRadiance.y) || isinf(pathRadiance.z)))
      sum = sum + pathRadiance;
#ifndef PT_NO_AUX_ACCUMULATION
    albedoSum = albedoSum + pathAlbedo;
    normalSum = normalSum + pathNormal;
#endif
  }

  float4 total = float4(sum, float(samples));
#ifndef PT_NO_AUX_ACCUMULATION
  float4 albedoTotal = float4(albedoSum, 0.0f);
  float4 normalTotal = float4(normalSum, 0.0f);
#endif
  if (firstSample > 0u) {
    total = total + accumulation.read(id);
#ifndef PT_NO_AUX_ACCUMULATION
    albedoTotal = albedoTotal + albedoAccumulation.read(id);
    normalTotal = normalTotal + normalAccumulation.read(id);
#endif
  }
  if (samples > 0u) {
    accumulation.write(total, id);
#ifndef PT_NO_AUX_ACCUMULATION
    albedoAccumulation.write(albedoTotal, id);
    normalAccumulation.write(normalTotal, id);
#endif
  }
  output.write(float4(xyz(total) / max(total.w, 1.0f), 1.0f), id);
}
