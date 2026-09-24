// Basalt's V6.2 R1 feasibility probe. This deliberately uses the compact fields from
// pt/path.h's PtHit rather than the compiler example's payload. The alpha decision is
// deterministic and candidate-local, like ptCandidateSolid's masked-material branch.
#include <metal_stdlib>
#include <m2v_ray_pipeline>
using namespace metal;

struct ProbeRay {
  float4 originTmax;
  float4 directionCutoff;
  uint4 control; // x stable round-trip token
};

struct ProbeResult {
  float4 hit; // t, barycentric xy, found
  uint4 identity; // instance, primitive, ambiguous, round-trip token
};

struct ProbePayload {
  uint ambiguous;
  float t;
  float2 barycentric;
  uint instance;
  uint primitive;
  uint found;
  uint token;
  float cutoff;
};

[[m2v::ray_stage(raygen)]]
void probeGenerate(instance_acceleration_structure scene [[buffer(0)]],
                   const device ProbeRay *rays [[buffer(1)]],
                   device ProbeResult *results [[buffer(2)]]) {
  const uint3 id = m2v::launch_id();
  const uint3 size = m2v::launch_size();
  const uint index = id.y * size.x + id.x;
  const ProbeRay input = rays[index];
  ProbePayload payload = {};
  payload.t = -2.0f;
  payload.token = input.control.x;
  payload.cutoff = input.directionCutoff.w;
  raytracing::ray ray(input.originTmax.xyz, input.directionCutoff.xyz, 0.0f,
                      input.originTmax.w);
  m2v::trace_ray(scene, m2v::ray_flags::none, 0xFFu, 0u, 1u, 0u, ray, payload);
  ProbeResult result;
  result.hit = float4(payload.t, payload.barycentric, float(payload.found));
  result.identity = uint4(payload.instance, payload.primitive, payload.ambiguous,
                          payload.token);
  results[index] = result;
}

[[m2v::ray_stage(miss)]]
void probeMiss(thread ProbePayload &payload [[m2v::payload]]) {
  payload.t = -1.0f;
  payload.found = 0u;
}

[[m2v::ray_stage(closest_hit)]]
void probeClosest(thread ProbePayload &payload [[m2v::payload]],
                  float2 barycentric [[m2v::hit_attribute]]) {
  payload.t = m2v::ray_tmax();
  payload.barycentric = barycentric;
  payload.instance = m2v::instance_custom_index();
  payload.primitive = m2v::primitive_id();
  payload.found = 1u;
}

[[m2v::ray_stage(any_hit)]]
void probeAlpha(thread ProbePayload &payload [[m2v::payload]],
                float2 barycentric [[m2v::hit_attribute]],
                const device float4 *vertexAlpha [[buffer(3)]]) {
  const float3 weights = float3(1.0f - barycentric.x - barycentric.y,
                                barycentric.x, barycentric.y);
  const float alpha = dot(weights, vertexAlpha[0].xyz);
  if (alpha < payload.cutoff) m2v::ignore_intersection();
}
