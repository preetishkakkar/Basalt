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
struct PtWavePipelinePayload { uint ambiguous; float t; float2 barycentric; uint instance; uint primitive; uint found; uint seed; float2 cone; };
[[m2v::ray_stage(miss)]] void path_wave_pipeline_miss(thread PtWavePipelinePayload& p [[m2v::payload]]) { p.found=0u; }
[[m2v::ray_stage(closest_hit)]] void path_wave_pipeline_closest(thread PtWavePipelinePayload& p [[m2v::payload]], float2 b [[m2v::hit_attribute]]) { p.t=m2v::ray_tmax();p.barycentric=b;p.instance=m2v::instance_id();p.primitive=m2v::primitive_id();p.found=1u; }
[[m2v::ray_stage(any_hit)]] void path_wave_pipeline_alpha(thread PtWavePipelinePayload& p [[m2v::payload]], float2 b [[m2v::hit_attribute]],
    const device TraceInstance* traceInstances [[buffer(1)]], const device Material* materials [[buffer(2)]],
    const device uint* indices [[buffer(3)]], const device float* vertices [[buffer(4)]],
    array<texture2d<float>, kHitTextureSlots> maps [[texture(7)]], sampler materialSampler [[sampler(1)]]) {
  const uint i=m2v::instance_id();if((traceInstances[i].flags&kInstanceBlended)!=0u)p.ambiguous=1u;
  const uint front=(m2v::hit_kind()==m2v::hit_kind_front_facing_triangle?1u:0u);
  if(!ptCandidateSolid(PT_SCENE_ARGS,PT_TEXTURE_ARGS,i,m2v::primitive_id(),b,front,p.seed,m2v::world_ray_direction(),ptConeWidthOrLevelZero(p.cone,m2v::ray_tmax())))m2v::ignore_intersection();
}
#endif

// Nearest hit for every live queue slot of this bounce. At bounce 0 the slots are the
// batch's paths in order and their camera rays are generated here.
#ifdef BASALT_RAY_PIPELINE
[[m2v::ray_stage(raygen)]]
#else
kernel
#endif
void PATH_WAVEFRONT_INTERSECT(constant PathUniforms& uniforms [[buffer(0)]],
    const device TraceInstance* traceInstances [[buffer(1)]], const device Material* materials [[buffer(2)]],
    const device uint* indices [[buffer(3)]], const device float* vertices [[buffer(4)]],
#if defined(BASALT_RAY_TRACING) || defined(BASALT_RAY_PIPELINE)
    instance_acceleration_structure scene [[buffer(8)]],
#elif defined(BASALT_WIDE_BVH)
    const device PtWideNode* bvhNodes [[buffer(8)]], const device float4* bvhTriangles [[buffer(9)]],
#else
    const device float4* bvhNodes [[buffer(8)]], const device float4* bvhTriangles [[buffer(9)]],
#endif
    const device PtWaveState* statesA [[buffer(12)]], const device PtWaveState* statesB [[buffer(13)]],
    const device uint* counters [[buffer(14)]], const device uint4* waveControl [[buffer(15)]],
    device uint2* costs [[buffer(16)]], device PtWaveHit* hits [[buffer(18)]],
    array<texture2d<float>, kHitTextureSlots> maps [[texture(7)]], sampler materialSampler [[sampler(1)]]
#ifndef BASALT_RAY_PIPELINE
    , uint id [[thread_position_in_grid]]
#endif
    ) {
#ifdef BASALT_RAY_PIPELINE
  const uint id=m2v::launch_id().x;
#endif
  const uint4 control=waveControl[0];const uint4 frame=waveControl[1];const uint parity=control.x&1u;
  const uint bounce=frame.x;
  if(id>=(bounce==0u?control.w:min(counters[parity*8u+kWaveQueueCount],control.y)))return;
  float3 origin=uniforms.cameraPosition.xyz,direction=float3(0.0f,0.0f,1.0f);uint local=id;float2 cone=ptCameraCone(uniforms.lens);
  if(bounce==0u){
    const uint g=control.z+id,pixel=g/frame.y,px=pixel%uint(uniforms.image.x),py=pixel/uint(uniforms.image.x);
    direction=ptWaveCameraRay(uniforms,px,py,pathSeed(px,py,uniforms.counts.w+g%frame.y,uniforms.counts.z),origin);
  }else{
    PtWaveState state;if(parity==0u)state=statesA[id];else state=statesB[id];
    origin=state.origin.xyz;direction=state.direction.xyz;local=ptWaveAsUint(state.direction.w);cone=ptWaveUnpackCone(state.throughput.w);
  }
  const uint g=control.z+local,pixel=g/frame.y,px=pixel%uint(uniforms.image.x),py=pixel/uint(uniforms.image.x);
  const uint seed=ptTraceSeed(pathSeed(px,py,uniforms.counts.w+g%frame.y,uniforms.counts.z),bounce,0u);
  PtHit hit;hit.ambiguous=0u;hit.t=0.0f;hit.barycentric=float2(0.0f);hit.instance=0u;hit.primitive=0u;hit.found=0u;hit.nodeVisits=0u;hit.triangleTests=0u;
#ifdef BASALT_RAY_PIPELINE
  PtWavePipelinePayload p={};p.seed=seed;p.cone=cone;raytracing::ray r(origin,direction,0.0f,kPtInfinity);
  m2v::trace_ray(scene,m2v::ray_flags::none,uniforms.counts.y,0u,1u,0u,r,p);
  hit.ambiguous=p.ambiguous;hit.t=p.t;hit.barycentric=p.barycentric;hit.instance=p.instance;hit.primitive=p.primitive;hit.found=p.found;
#else
#include "pt/wavefront_nearest.inc"
#endif
  PtWaveHit compact;compact.t=hit.t;compact.barycentricX=hit.barycentric.x;compact.barycentricY=hit.barycentric.y;
  compact.instance=hit.instance;compact.primitive=hit.primitive;compact.flags=(hit.found!=0u?1u:0u)|(hit.ambiguous!=0u?2u:0u);
  hits[id]=compact;
  // BVH cost views only: per-path traversal work, the path's own slot, integer and exact.
  if(uniforms.image.z>1.5f){if(bounce==0u)costs[local]=uint2(hit.nodeVisits,hit.triangleTests);else costs[local]+=uint2(hit.nodeVisits,hit.triangleTests);}
}
