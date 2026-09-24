#include "shared/prelude.h"
#include "pt/path.h"
#if defined(BASALT_WAVE_FUSED) && defined(BASALT_RAY_TRACING)
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
#include "pt/surface.h"
#include "pt/bsdf.h"
#include "pt/reconstruction.h"
#include "pt/lights.h"
#include "pt/wavefront.h"
#ifdef BASALT_WAVE_FUSED
#include "pt/bvh.h"
#ifdef BASALT_WIDE_BVH
#include "pt/wide_bvh.h"
#endif
#endif

// One bounce of every live queue slot: escape or emission, the vertex's light sample (queued
// as a shadow ray), and the BSDF continuation (queued for the next bounce). The transport is
// integrator.inc's, statement for statement; only the storage of the path between bounces
// differs. Radiance goes to the path's own result, written whole at bounce 0 and afterwards
// only when a contribution is nonzero.
kernel void PATH_WAVEFRONT_SHADE(constant PathUniforms& uniforms [[buffer(0)]],
    const device TraceInstance* traceInstances [[buffer(1)]], const device Material* materials [[buffer(2)]],
    const device uint* indices [[buffer(3)]], const device float* vertices [[buffer(4)]],
    const device Light* lights [[buffer(5)]], const device float* environmentDistribution [[buffer(6)]],
    const device float* specularAlbedo [[buffer(7)]],
    device PtReconstructionSample* guides [[buffer(10)]],
    const device PtEmissiveTriangle* emissiveTriangles [[buffer(11)]],
    device PtWaveState* statesA [[buffer(12)]], device PtWaveState* statesB [[buffer(13)]],
    device atomic_uint* counters [[buffer(14)]], const device uint4* waveControl [[buffer(15)]],
    device PtWaveResult* results [[buffer(16)]], device PtWaveShadow* shadows [[buffer(17)]],
    const device PtWaveHit* hits [[buffer(18)]], device PtWaveGuide* waveGuides [[buffer(19)]],
    // ReSTIR DI: the primary vertex's visible direct light and its diffuse part.
    const device PtWaveResult* restirResults [[buffer(22)]], const device PtWaveGuide* restirGuides [[buffer(23)]],
#ifdef BASALT_WAVE_FUSED
    // Fused mode traces the slot's nearest hit here instead of reading the intersect stage's.
#if defined(BASALT_RAY_TRACING)
    instance_acceleration_structure scene [[buffer(8)]],
#elif defined(BASALT_WIDE_BVH)
    const device PtWideNode* bvhNodes [[buffer(8)]], const device float4* bvhTriangles [[buffer(9)]],
#else
    const device float4* bvhNodes [[buffer(8)]], const device float4* bvhTriangles [[buffer(9)]],
#endif
    device uint2* costs [[buffer(21)]],
#endif
    texture2d<float> environmentMap [[texture(0)]],
    array<texture2d<float>, kHitTextureSlots> maps [[texture(7)]],
    sampler environmentSampler [[sampler(0)]], sampler materialSampler [[sampler(1)]],
    uint id [[thread_position_in_grid]]) {
  const uint4 control = waveControl[0];
  const uint4 frame = waveControl[1];
  const uint parity = control.x & 1u;
  const uint capacity = control.y;
  const uint bounce = frame.x;
  if (id >= (bounce == 0u ? control.w : min(atomic_load_explicit(&counters[parity * 8u + kWaveQueueCount], memory_order_relaxed), capacity))) return;
  PtWaveState state;
  state.origin = float4(uniforms.cameraPosition.xyz, 0.0f);
  state.direction = float4(0.0f, 0.0f, 1.0f, 0.0f);
  state.throughput = float4(1.0f);
  uint local = id;
  if (bounce > 0u) {
    if (parity == 0u) state = statesA[id]; else state = statesB[id];
    local = ptWaveAsUint(state.direction.w);
  }
  const uint g = control.z + local, pixel = g / frame.y, sampleInPixel = g % frame.y;
  const uint pixelX = pixel % uint(uniforms.image.x), pixelY = pixel / uint(uniforms.image.x);
  const uint sampleIndex = uniforms.counts.w + sampleInPixel;
  const uint pathSeedValue = pathSeed(pixelX, pixelY, sampleIndex, uniforms.counts.z);
  // The path's ray cone: the camera's at bounce 0, else the state's.
  const float2 cone = bounce == 0u ? ptCameraCone(uniforms.lens) : ptWaveUnpackCone(state.throughput.w);
  if (bounce == 0u) {
    float3 cameraOrigin = state.origin.xyz;
    state.direction = float4(ptWaveCameraRay(uniforms, pixelX, pixelY, pathSeedValue, cameraOrigin), 0.0f);
    state.origin = float4(cameraOrigin, 0.0f);
  }
  const bool collectGuides = uniforms.image.z > 0.5f && uniforms.image.z < 1.5f && sampleInPixel + 1u == frame.y;
  const float3 direction = state.direction.xyz;
  float3 diffuseFraction = float3(1.0f);
  if (collectGuides && bounce > 0u) diffuseFraction = waveGuides[pixel].diffuseFraction.xyz;
  float3 guideDiffuse = float3(0.0f);
  float3 radiance = float3(0.0f), albedo = float3(0.0f), normal = float3(0.0f);
#ifdef BASALT_WAVE_FUSED
  PtWaveHit waveHit;
  {
    const float3 origin = state.origin.xyz;
    const uint seed = ptTraceSeed(pathSeedValue, bounce, 0u);
    PtHit hit;
    hit.ambiguous = 0u; hit.t = 0.0f; hit.barycentric = float2(0.0f); hit.instance = 0u; hit.primitive = 0u;
    hit.found = 0u; hit.nodeVisits = 0u; hit.triangleTests = 0u;
#include "pt/wavefront_nearest.inc"
    waveHit.t = hit.t; waveHit.barycentricX = hit.barycentric.x; waveHit.barycentricY = hit.barycentric.y;
    waveHit.instance = hit.instance; waveHit.primitive = hit.primitive;
    waveHit.flags = (hit.found != 0u ? 1u : 0u) | (hit.ambiguous != 0u ? 2u : 0u);
    if (uniforms.image.z > 1.5f) {
      if (bounce == 0u) costs[local] = uint2(hit.nodeVisits, hit.triangleTests);
      else costs[local] += uint2(hit.nodeVisits, hit.triangleTests);
    }
  }
#else
  const PtWaveHit waveHit = hits[id];
#endif
  bool continuing = false;
  if ((waveHit.flags & 1u) == 0u) {
    if (bounce == 0u && uniforms.environment.w > 0.5f)
      albedo = saturate(ptSampleEnvironment(PT_ENVIRONMENT_ARGS,equirectangularUV(direction))*uniforms.environment.x);
    if (bounce == 0u && collectGuides) guides[pixel] = ptEmptyReconstructionSample();
    if (bounce > 0u || uniforms.environment.w > 0.5f) {
      uint kindCount=0u; if(uint(uniforms.path.w)!=1u&&uniforms.distribution.w>0.5f)++kindCount;
      if(uint(uniforms.path.w)!=1u&&uniforms.sunDirection.w>0.0f)++kindCount;
      if(uint(uniforms.path.w)!=1u&&uniforms.counts.x>0u)++kindCount;
      if(uint(uniforms.path.w)!=1u&&uniforms.emissive.x>0u)++kindCount;
      float ew=1.0f,sw=1.0f;
      if(bounce>0u&&uint(uniforms.path.w)==2u){ew=0.0f;sw=0.0f;}
      if(bounce>0u&&uint(uniforms.path.w)==0u&&uniforms.distribution.w>0.5f) ew=ptPowerHeuristic(state.origin.w,(1.0f/max(float(kindCount),1.0f))*ptEnvironmentPdf(environmentDistribution,uniforms.distribution,direction));
      if(bounce>0u&&uint(uniforms.path.w)==0u&&uniforms.sunDirection.w>0.0f) sw=ptPowerHeuristic(state.origin.w,(1.0f/max(float(kindCount),1.0f))*(1.0f/max(uniforms.sunRadiance.w,1e-12f)));
      float3 escaped=ptSampleEnvironment(PT_ENVIRONMENT_ARGS,equirectangularUV(direction))*(uniforms.environment.x*ew);
      if(uniforms.sunDirection.w>0.0f&&dot(direction,uniforms.sunDirection.xyz)>=uniforms.sunDirection.w) escaped+=uniforms.sunRadiance.xyz*sw;
      escaped*=state.throughput.xyz; if(bounce>0u) escaped=ptClampContribution(escaped,uniforms.path.z);
      radiance+=escaped; guideDiffuse+=escaped*diffuseFraction;
    }
  } else {
    PtHit hit; hit.ambiguous=(waveHit.flags>>1u)&1u; hit.t=waveHit.t; hit.barycentric=float2(waveHit.barycentricX,waveHit.barycentricY);
    hit.instance=waveHit.instance; hit.primitive=waveHit.primitive; hit.found=1u; hit.nodeVisits=0u; hit.triangleTests=0u;
    const float hitWidth=ptConeWidthOrLevelZero(cone,hit.t);
    const PtSurface surface=ptSurfaceAt(PT_SCENE_ARGS,PT_TEXTURE_ARGS,hit,direction,hitWidth);
    if(bounce==0u){
      albedo=surface.baseColor; normal=surface.normal;
      if(collectGuides){const uint materialId=traceInstances[hit.instance].material; const uint coverage=uint(materials[materialId].alpha.y); PtReconstructionSample gs=ptEmptyReconstructionSample(); gs.positionDepth=float4(surface.position,dot(surface.position-uniforms.cameraPosition.xyz,uniforms.cameraForward.xyz)); gs.geometricNormal=float4(surface.geometricNormal,ptGuideLayers(surface.transmission,surface.clearcoat)); gs.normalRoughness=float4(surface.normal,surface.roughness); gs.albedo=float4(surface.baseColor,0.0f); gs.viewDistance=float4(direction,-1.0f); gs.identity=uint4(hit.instance,materialId,hit.ambiguous!=0u||coverage==2u?2u:coverage==1u?3u:1u,hit.primitive); guides[pixel]=gs;}
    } else if(collectGuides&&bounce==1u){PtReconstructionSample gs=guides[pixel];gs.viewDistance.w=hit.t;guides[pixel]=gs;}
    uint kindCount=0u; if(uint(uniforms.path.w)!=1u&&uniforms.distribution.w>0.5f)++kindCount; if(uint(uniforms.path.w)!=1u&&uniforms.sunDirection.w>0.0f)++kindCount; if(uint(uniforms.path.w)!=1u&&uniforms.counts.x>0u)++kindCount; if(uint(uniforms.path.w)!=1u&&uniforms.emissive.x>0u)++kindCount;
    float emissiveWeight=uint(uniforms.path.w)==2u&&bounce>0u?0.0f:1.0f;  // as integrator.inc when no emitter has power
    if(bounce>0u&&uint(uniforms.path.w)==0u&&uniforms.emissive.x>0u) emissiveWeight=ptPowerHeuristic(state.origin.w,(1.0f/max(float(kindCount),1.0f))*ptEmissivePdf(emissiveTriangles,uniforms.emissive.x,traceInstances,hit,direction));
    const float3 emitted=state.throughput.xyz*surface.emissive*emissiveWeight; radiance+=emitted; guideDiffuse+=emitted*diffuseFraction;
    if(bounce<uint(uniforms.path.x)){
      const PtBsdf bsdf=ptMakeBsdf(surface,-direction,specularAlbedo); const float3 rayOrigin=ptOffsetRay(surface.position,surface.geometricNormal);
      const float4 lr=pathRandom4(pathSeedValue,bounce+1u,0u), br=pathRandom4(pathSeedValue,bounce+1u,1u);
      uint kind=4u,slot=kindCount>0u?min(uint(lr.z*float(kindCount)),kindCount-1u):0u;
      if(kindCount>0u&&uint(uniforms.path.w)!=1u&&uniforms.distribution.w>0.5f){if(slot==0u)kind=0u;slot--;}
      if(kindCount>0u&&uint(uniforms.path.w)!=1u&&uniforms.sunDirection.w>0.0f&&kind==4u){if(slot==0u)kind=1u;slot--;}
      if(kindCount>0u&&uint(uniforms.path.w)!=1u&&uniforms.counts.x>0u&&kind==4u){if(slot==0u)kind=2u;slot--;}
      if(kindCount>0u&&uint(uniforms.path.w)!=1u&&uniforms.emissive.x>0u&&kind==4u)kind=3u;
      // ReSTIR DI replaces the primary vertex's light sample with its reservoir's visible light.
      if(bounce==0u&&uniforms.estimator.x==1u){kind=4u;radiance+=restirResults[pixel].radiance.xyz;if(collectGuides)guideDiffuse+=restirGuides[pixel].diffuseRadiance.xyz;}
      float3 ld=float3(0.0f,1.0f,0.0f),arr=float3(0.0f);float lp=0.0f,reach=kPtInfinity;
      if(kind==0u){ld=ptSampleEnvironmentDirection(environmentDistribution,uniforms.distribution,lr.xy,lp);lp/=max(float(kindCount),1.0f);arr=ptSampleEnvironment(PT_ENVIRONMENT_ARGS,equirectangularUV(ld))*uniforms.environment.x;}
      else if(kind==1u){ld=ptSampleCone(uniforms.sunDirection.xyz,uniforms.sunDirection.w,lr.xy);lp=(1.0f/max(uniforms.sunRadiance.w,1e-12f))/max(float(kindCount),1.0f);arr=uniforms.sunRadiance.xyz;}
      else if(kind==2u){const uint pick=min(uint(lr.w*float(uniforms.counts.x)),uniforms.counts.x-1u);float dist=0.0f;arr=ptPunctualLight(lights[pick],surface.position,ld,dist)*(float(uniforms.counts.x)*float(kindCount));reach=dist*0.999f;}
      else if(kind==3u){arr=ptSampleEmissiveTriangle(emissiveTriangles,uniforms.emissive.x,traceInstances,materials,PT_TEXTURE_ARGS,surface.position,float3(lr.w,lr.x,lr.y),ptConeScattered(cone,hitWidth,ptBsdfSampledAlpha(bsdf,br.z)),ld,reach,lp);lp/=max(float(kindCount),1.0f);reach*=0.9999f;}
      const float3 sd=ptBsdfSampleDirection(bsdf,br.xyz);float lbp=0.0f,sp=0.0f;const float3 lv=ptBsdfEvaluate(bsdf,surface.geometricNormal,ld,lbp);const float3 sv=ptBsdfEvaluate(bsdf,surface.geometricNormal,sd,sp);
      if(kind!=4u){float3 contribution=state.throughput.xyz*lv*arr;if(kind!=2u){float weight=lp>0.0f?1.0f/lp:0.0f;if(uint(uniforms.path.w)==0u&&lp>0.0f)weight=ptPowerHeuristic(lp,lbp)/lp;contribution*=weight;}
        if(ptMaxComponent(contribution)>0.0f){
          contribution=ptClampContribution(contribution,uniforms.path.z);
          uint queued=0u;PT_WAVE_RESERVE(counters,kWaveShadowGroup,capacity,queued);
          if(queued<capacity){
            PtWaveShadow work;work.originReach=float4(dot(ld,surface.geometricNormal)<0.0f?ptOffsetRay(surface.position,-surface.geometricNormal):rayOrigin,reach);work.direction=float4(ld,ptWaveAsFloat(local));
            // contribution.w: the shadow ray's cone; the shadow stage derives the any-hit seed.
            work.contribution=float4(contribution,ptWavePackCone(ptConeShadow(cone,hitWidth)));shadows[queued]=work;
            if(collectGuides)waveGuides[pixel].shadowFraction=float4(bounce==0u?ptDiffuseFraction(bsdf,ld,lv):diffuseFraction,0.0f);
          }
        }}
      continuing=sp>0.0f&&ptMaxComponent(sv)>0.0f;
      if(continuing){
        if(collectGuides&&bounce==0u)waveGuides[pixel].diffuseFraction=float4(ptDiffuseFraction(bsdf,sd,sv),0.0f);
        state.throughput.xyz*=sv/sp;state.origin=float4(dot(sd,surface.geometricNormal)<0.0f?ptOffsetRay(surface.position,-surface.geometricNormal):rayOrigin,sp);state.direction=float4(sd,ptWaveAsFloat(local));
        state.throughput.w=ptWavePackCone(ptConeScattered(cone,hitWidth,ptBsdfSampledAlpha(bsdf,br.z)));
        if(bounce+1u>=uint(uniforms.path.y)){const float survive=min(ptMaxComponent(state.throughput.xyz),0.95f);if(br.w>=survive)continuing=false;else state.throughput.xyz/=survive;}
      }
    }
  }
  if(bounce==0u){PtWaveResult result;result.radiance=float4(radiance,0.0f);result.albedo=float4(albedo,0.0f);result.normal=float4(normal,0.0f);results[local]=result;}
  else if(radiance.x!=0.0f||radiance.y!=0.0f||radiance.z!=0.0f)results[local].radiance+=float4(radiance,0.0f);
  if(collectGuides){if(bounce==0u)waveGuides[pixel].diffuseRadiance=float4(guideDiffuse,0.0f);
    else if(guideDiffuse.x!=0.0f||guideDiffuse.y!=0.0f||guideDiffuse.z!=0.0f)waveGuides[pixel].diffuseRadiance+=float4(guideDiffuse,0.0f);}
  if(continuing){const uint next=1u-parity;uint slotOut=0u;PT_WAVE_RESERVE(counters,next*8u,capacity,slotOut);if(slotOut<capacity){if(next==0u)statesA[slotOut]=state;else statesB[slotOut]=state;}}
}
