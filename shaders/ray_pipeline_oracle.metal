// Full-pipeline form of the independent GPU ray corpus. It intentionally shares the
// same candidate policy and records as ray_query_oracle.metal so the host oracle can
// compare traversal APIs without changing its tolerances.
#include "shared/prelude.h"
#include "pt/path.h"
#include <m2v_ray_pipeline>

#define PT_TEXTURE_PARAMS array<texture2d<float>, kHitTextureSlots> maps, sampler materialSampler
#define PT_TEXTURE_ARGS maps, materialSampler
#include "pt/texture_msl.h"
#include "pt/bvh.h"

struct OracleRay {
  float4 originAndMin;
  float4 directionAndMax;
  uint4 control;
};

struct OracleHit {
  float4 distanceAndBarycentric;
  uint4 identity;
};

struct PipelineOraclePayload {
  float t;
  float2 barycentric;
  uint found;
  uint instance;
  uint primitive;
  uint mask;
  uint alphaSeed;
  uint occlusion;
};

[[m2v::ray_stage(raygen)]]
void pipeline_oracle_generate(const device OracleRay *rays [[buffer(0)]],
                              device OracleHit *hits [[buffer(1)]],
                              constant uint4 &control [[buffer(6)]],
                              instance_acceleration_structure scene [[buffer(7)]]) {
  const uint id = m2v::launch_id().x;
  if (id >= control.x) return;
  const OracleRay input = rays[id];
  PipelineOraclePayload payload = {};
  payload.t = input.directionAndMax.w;
  payload.instance = 0xFFFFFFFFu;
  payload.primitive = 0xFFFFFFFFu;
  payload.mask = input.control.x;
  payload.alphaSeed = input.control.y;
  payload.occlusion = input.control.z;
  raytracing::ray ray(input.originAndMin.xyz, input.directionAndMax.xyz,
                      input.originAndMin.w, input.directionAndMax.w);
  m2v::trace_ray(scene, m2v::ray_flags::none, input.control.x, 0u, 1u, 0u, ray, payload);
  OracleHit output;
  output.distanceAndBarycentric = float4(payload.t, payload.barycentric, 0.0f);
  output.identity = uint4(payload.found, payload.instance, payload.primitive, 0u);
  hits[id] = output;
}

[[m2v::ray_stage(miss)]]
void pipeline_oracle_miss(thread PipelineOraclePayload &payload [[m2v::payload]]) {
  payload.found = 0u;
}

[[m2v::ray_stage(closest_hit)]]
void pipeline_oracle_closest(thread PipelineOraclePayload &payload [[m2v::payload]],
                             float2 barycentric [[m2v::hit_attribute]]) {
  payload.t = m2v::ray_tmax();
  payload.barycentric = barycentric;
  payload.instance = m2v::instance_id();
  payload.primitive = m2v::primitive_id();
  payload.found = 1u;
}

[[m2v::ray_stage(any_hit)]]
void pipeline_oracle_alpha(thread PipelineOraclePayload &payload [[m2v::payload]],
                           float2 barycentric [[m2v::hit_attribute]],
                           const device TraceInstance *traceInstances [[buffer(2)]],
                           const device Material *materials [[buffer(3)]],
                           const device uint *indices [[buffer(4)]],
                           const device float *vertices [[buffer(5)]],
                           array<texture2d<float>, kHitTextureSlots> maps [[texture(7)]],
                           sampler materialSampler [[sampler(0)]]) {
  const uint instance = m2v::instance_id();
  const uint front = (m2v::hit_kind() == m2v::hit_kind_front_facing_triangle ? 1u : 0u);
  if (!ptCandidateSolid(PT_SCENE_ARGS, PT_TEXTURE_ARGS, instance, m2v::primitive_id(),
                        barycentric, front, payload.alphaSeed, float3(0.0f), -1.0f))  // level zero
    m2v::ignore_intersection();
  if (payload.occlusion != 0u) {
    payload.t = m2v::ray_tmax();
    payload.barycentric = barycentric;
    payload.instance = instance;
    payload.primitive = m2v::primitive_id();
    payload.found = 1u;
    m2v::terminate_ray();
  }
}
