// Batched hardware ray-query oracle. The host supplies an independently generated ray
// corpus and compares these records with double-precision world-space intersections.
#include "shared/prelude.h"
#include "pt/path.h"
#define BASALT_RAY_TRACING 1
#include <metal_raytracing>
using namespace metal::raytracing;

#define PT_TEXTURE_PARAMS array<texture2d<float>, kHitTextureSlots> maps, sampler materialSampler
#define PT_TEXTURE_ARGS maps, materialSampler
#include "pt/texture_msl.h"

#include "pt/bvh.h"

struct OracleRay {
  float4 originAndMin;
  float4 directionAndMax;
  uint4 control; // x mask, y alpha seed, z nonzero for occlusion, w unused
};

struct OracleHit {
  float4 distanceAndBarycentric; // x distance, yz barycentrics
  uint4 identity;                // x found, y instance, z primitive, w unused
};

kernel void ray_query_oracle(const device OracleRay* rays [[buffer(0)]],
                             device OracleHit* hits [[buffer(1)]],
                             const device TraceInstance* traceInstances [[buffer(2)]],
                             const device Material* materials [[buffer(3)]],
                             const device uint* indices [[buffer(4)]],
                             const device float* vertices [[buffer(5)]],
                             constant uint4& control [[buffer(6)]],
                             instance_acceleration_structure scene [[buffer(7)]],
                             array<texture2d<float>, kHitTextureSlots> maps [[texture(7)]],
                             sampler materialSampler [[sampler(0)]],
                             uint id [[thread_position_in_grid]]) {
  if (id >= control.x) return;
  const OracleRay input = rays[id];
  OracleHit output;
  output.distanceAndBarycentric = float4(input.directionAndMax.w, 0.0f, 0.0f, 0.0f);
  output.identity = uint4(0u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0u);

  ray queryRay(input.originAndMin.xyz, input.directionAndMax.xyz, input.originAndMin.w,
               input.directionAndMax.w);
  intersection_params parameters;
  parameters.accept_any_intersection(input.control.z != 0u);
  intersection_query<triangle_data, instancing> query(queryRay, scene, input.control.x, parameters);
  while (query.next()) {
    if (query.get_candidate_intersection_type() == intersection_type::triangle &&
        ptCandidateSolid(PT_SCENE_ARGS, PT_TEXTURE_ARGS, query.get_candidate_instance_id(),
                         query.get_candidate_primitive_id(),
                         query.get_candidate_triangle_barycentric_coord(),
                         (query.is_candidate_triangle_front_facing() ? 1u : 0u),
                         input.control.y, float3(0.0f), -1.0f))  // level zero
      query.commit_triangle_intersection();
  }
  if (query.get_committed_intersection_type() == intersection_type::triangle) {
    output.distanceAndBarycentric =
        float4(query.get_committed_distance(), query.get_committed_triangle_barycentric_coord(), 0.0f);
    output.identity = uint4(1u, query.get_committed_instance_id(),
                            query.get_committed_primitive_id(), 0u);
  }
  hits[id] = output;
}
