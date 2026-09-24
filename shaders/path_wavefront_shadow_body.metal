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
#include "pt/bvh.h"
#ifdef BASALT_WIDE_BVH
#include "pt/wide_bvh.h"
#endif
#include "pt/wavefront.h"

#ifdef BASALT_RAY_PIPELINE
struct PtWaveShadowPayload { uint found; uint seed; float2 cone; };
[[m2v::ray_stage(miss)]] void path_wave_shadow_pipeline_miss(thread PtWaveShadowPayload& p [[m2v::payload]]) { p.found=0u; }
[[m2v::ray_stage(closest_hit)]] void path_wave_shadow_pipeline_closest(thread PtWaveShadowPayload& p [[m2v::payload]], float2 b [[m2v::hit_attribute]]) { p.found=1u; }
[[m2v::ray_stage(any_hit)]] void path_wave_shadow_pipeline_alpha(thread PtWaveShadowPayload& p [[m2v::payload]], float2 b [[m2v::hit_attribute]],
    const device TraceInstance* traceInstances [[buffer(1)]], const device Material* materials [[buffer(2)]],
    const device uint* indices [[buffer(3)]], const device float* vertices [[buffer(4)]],
    array<texture2d<float>, kHitTextureSlots> maps [[texture(7)]], sampler materialSampler [[sampler(1)]]) {
  const uint i=m2v::instance_id();const uint front=(m2v::hit_kind()==m2v::hit_kind_front_facing_triangle?1u:0u);
  if(!ptCandidateSolid(PT_SCENE_ARGS,PT_TEXTURE_ARGS,i,m2v::primitive_id(),b,front,p.seed,m2v::world_ray_direction(),ptConeWidthOrLevelZero(p.cone,m2v::ray_tmax())))m2v::ignore_intersection();
  p.found=1u;m2v::terminate_ray();
}
#endif

// Occlusion for every queued shadow ray of this bounce; an unoccluded one adds its
// contribution to its own path's radiance (and diffuse guide for the guide path).
#ifdef BASALT_RAY_PIPELINE
[[m2v::ray_stage(raygen)]]
#else
kernel
#endif
void PATH_WAVEFRONT_SHADOW(constant PathUniforms& uniforms [[buffer(0)]],
    const device TraceInstance* traceInstances [[buffer(1)]], const device Material* materials [[buffer(2)]],
    const device uint* indices [[buffer(3)]], const device float* vertices [[buffer(4)]],
#if defined(BASALT_RAY_TRACING) || defined(BASALT_RAY_PIPELINE)
    instance_acceleration_structure scene [[buffer(8)]],
#elif defined(BASALT_WIDE_BVH)
    const device PtWideNode* bvhNodes [[buffer(8)]], const device float4* bvhTriangles [[buffer(9)]],
#else
    const device float4* bvhNodes [[buffer(8)]], const device float4* bvhTriangles [[buffer(9)]],
#endif
    const device uint* counters [[buffer(14)]], const device uint4* waveControl [[buffer(15)]],
    device PtWaveResult* results [[buffer(16)]], const device PtWaveShadow* shadows [[buffer(17)]],
    device uint2* costs [[buffer(18)]], device PtWaveGuide* waveGuides [[buffer(19)]],
    array<texture2d<float>, kHitTextureSlots> maps [[texture(7)]], sampler materialSampler [[sampler(1)]]
#ifndef BASALT_RAY_PIPELINE
    , uint id [[thread_position_in_grid]]
#endif
    ) {
#ifdef BASALT_RAY_PIPELINE
  const uint id=m2v::launch_id().x;
#endif
  const uint4 control=waveControl[0];const uint4 frame=waveControl[1];
  if(id>=min(counters[kWaveShadowGroup+kWaveQueueCount],control.y))return;
  const PtWaveShadow work=shadows[id];const uint local=ptWaveAsUint(work.direction.w);
  const uint g=control.z+local,pixel=g/frame.y,px=pixel%uint(uniforms.image.x),py=pixel/uint(uniforms.image.x);
  const uint seed=ptTraceSeed(pathSeed(px,py,uniforms.counts.w+g%frame.y,uniforms.counts.z),frame.x,1u);
  const float2 cone=ptWaveUnpackCone(work.contribution.w);
  PtHit hit;hit.ambiguous=0u;hit.found=0u;hit.nodeVisits=0u;hit.triangleTests=0u;
#ifdef BASALT_RAY_PIPELINE
  PtWaveShadowPayload p={};p.seed=seed;p.cone=cone;raytracing::ray r(work.originReach.xyz,work.direction.xyz,0.0f,work.originReach.w);
  m2v::trace_ray(scene,m2v::ray_flags::none,uniforms.counts.y,0u,1u,0u,r,p);hit.found=p.found;
#elif defined(BASALT_RAY_TRACING)
  ray r(work.originReach.xyz,work.direction.xyz,0.0f,work.originReach.w);intersection_params p;p.accept_any_intersection(true);
  intersection_query<triangle_data,instancing> q(r,scene,uniforms.counts.y,p);
  while(q.next())if(q.get_candidate_intersection_type()==intersection_type::triangle&&ptCandidateSolid(PT_SCENE_ARGS,PT_TEXTURE_ARGS,q.get_candidate_instance_id(),q.get_candidate_primitive_id(),q.get_candidate_triangle_barycentric_coord(),(q.is_candidate_triangle_front_facing()?1u:0u),seed,work.direction.xyz,ptConeWidthOrLevelZero(cone,q.get_candidate_triangle_distance())))q.commit_triangle_intersection();
  if(q.get_committed_intersection_type()==intersection_type::triangle)hit.found=1u;
#elif defined(BASALT_WIDE_BVH)
  hit=ptTraceWideBvh(bvhNodes,bvhTriangles,PT_SCENE_ARGS,PT_TEXTURE_ARGS,work.originReach.xyz,work.direction.xyz,work.originReach.w,uniforms.counts.y,seed,cone,1u);
#else
  hit=ptTraceBvh(bvhNodes,bvhTriangles,PT_SCENE_ARGS,PT_TEXTURE_ARGS,work.originReach.xyz,work.direction.xyz,work.originReach.w,uniforms.counts.y,seed,cone,1u);
#endif
  if(uniforms.image.z>1.5f)costs[local]+=uint2(hit.nodeVisits,hit.triangleTests);
  if(hit.found!=0u)return;
  results[local].radiance+=float4(work.contribution.xyz,0.0f);
  if(uniforms.image.z>0.5f&&uniforms.image.z<1.5f&&g%frame.y+1u==frame.y){
    waveGuides[pixel].diffuseRadiance+=float4(work.contribution.xyz*waveGuides[pixel].shadowFraction.xyz,0.0f);
  }
}
