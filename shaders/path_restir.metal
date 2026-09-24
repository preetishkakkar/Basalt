// ReSTIR DI's reservoir passes, one thread per pixel, on every GPU backend. A
// frame runs: the backend's wavefront intersect stage at bounce 0 into per-pixel hits;
// path_restir_initial (initial resampling and temporal reuse); path_restir_spatial (spatial
// reuse, then the final reservoir's light as one shadow record per pixel); the backend's
// wavefront shadow stage, which adds the visible light to per-pixel results; then the path
// tracer, which takes those in place of the primary vertex's light sample. The per-pixel
// passes are shaders/pt/restir.h's, shared with the CPU tracer.
#include "shared/prelude.h"
#include "pt/path.h"
#define PT_TEXTURE_PARAMS array<texture2d<float>, kHitTextureSlots> maps, sampler materialSampler
#define PT_TEXTURE_ARGS maps, materialSampler
#include "pt/texture_msl.h"
#define PT_ENVIRONMENT_PARAMS texture2d<float> environmentMap, sampler environmentSampler
#define PT_ENVIRONMENT_ARGS environmentMap, environmentSampler
inline float3 ptSampleEnvironment(PT_ENVIRONMENT_PARAMS, float2 uv) {
  return environmentMap.sample(environmentSampler, uv, level(0.0f)).xyz;
}
#include "pt/restir.h"
#include "pt/wavefront.h"

inline PtRestirLights ptRestirLightsOf(constant PathUniforms &u) {
  return ptRestirLights(u.environment, u.distribution, u.sunDirection, u.sunRadiance, u.path, u.counts, u.emissive);
}

// Initial resampling at the pixel's primary hit, then temporal reuse from the previous
// frame's final reservoirs (restirCamera: the previous camera, history validity).
kernel void path_restir_initial(constant PathUniforms& uniforms [[buffer(0)]],
    const device TraceInstance* traceInstances [[buffer(1)]], const device Material* materials [[buffer(2)]],
    const device uint* indices [[buffer(3)]], const device float* vertices [[buffer(4)]],
    const device Light* lights [[buffer(5)]], const device float* environmentDistribution [[buffer(6)]],
    const device float* specularAlbedo [[buffer(7)]],
    const device PtEmissiveTriangle* emissiveTriangles [[buffer(11)]],
    const device PtWaveHit* hits [[buffer(18)]],
    const device PtRestirCamera* restirCamera [[buffer(20)]],
    const device PtRestirSurface* previousSurfaces [[buffer(21)]],
    const device PtReservoir* previousReservoirs [[buffer(22)]],
    device PtRestirSurface* surfaces [[buffer(23)]], device PtReservoir* reservoirs [[buffer(24)]],
    texture2d<float> environmentMap [[texture(0)]],
    array<texture2d<float>, kHitTextureSlots> maps [[texture(7)]],
    sampler environmentSampler [[sampler(0)]], sampler materialSampler [[sampler(1)]],
    uint id [[thread_position_in_grid]]) {
  const uint width = uint(uniforms.image.x), height = uint(uniforms.image.y);
  if (id >= width * height) return;
  const uint pixelX = id % width, pixelY = id / width;
  const uint candidates = min(uniforms.estimator.y, kPtRestirMaxCandidates);
  const uint seed = pathSeed(pixelX, pixelY, uniforms.counts.w, uniforms.counts.z);
  const PtRestirLights l = ptRestirLightsOf(uniforms);
  const float2 cameraCone = ptCameraCone(uniforms.lens);
  const PtWaveHit waveHit = hits[id];
  PtRestirSurface record;
  record.positionDepth = float4(0.0f);
  record.normalMaterial = float4(0.0f);
  record.view = float4(0.0f);
  record.hit = uint4(0u);
  PtReservoir r = ptEmptyReservoir();
  r.count = float(candidates);
  if ((waveHit.flags & 1u) != 0u) {
    float3 origin = float3(0.0f);
    const float3 direction = ptWaveCameraRay(uniforms, pixelX, pixelY, seed, origin);
    PtHit hit;
    hit.ambiguous = (waveHit.flags >> 1u) & 1u;
    hit.t = waveHit.t;
    hit.barycentric = float2(waveHit.barycentricX, waveHit.barycentricY);
    hit.instance = waveHit.instance;
    hit.primitive = waveHit.primitive;
    hit.found = 1u;
    hit.nodeVisits = 0u;
    hit.triangleTests = 0u;
    const float hitWidth = ptConeWidthOrLevelZero(cameraCone, hit.t);
    const PtSurface surface = ptSurfaceAt(PT_SCENE_ARGS, PT_TEXTURE_ARGS, hit, direction, hitWidth);
    record = ptRestirSurfaceOf(surface, hit, traceInstances[hit.instance].material, direction, hitWidth,
                               uniforms.cameraPosition.xyz, uniforms.cameraForward.xyz);
    const PtBsdf bsdf = ptMakeBsdf(surface, -direction, specularAlbedo);
    const PtReservoir initial = ptRestirInitial(PT_RESTIR_LIGHT_ARGS, l, bsdf, surface.position, surface.geometricNormal,
                                                ptConeShadow(cameraCone, hitWidth), seed, candidates);
    r = ptRestirTemporal(PT_RESTIR_PASS_ARGS, l, restirCamera[0], cameraCone, id, uniforms.counts.w, candidates, record,
                         initial, previousSurfaces, previousReservoirs);
  }
  surfaces[id] = record;
  reservoirs[id] = r;
}

// Spatial reuse (when enabled), the final reservoir kept for the next frame, and its light
// queued as the pixel's shadow record: the wavefront shadow stage adds it to results[pixel]
// (zeroed here) and its diffuse part to waveGuides[pixel] when guides are collected.
kernel void path_restir_spatial(constant PathUniforms& uniforms [[buffer(0)]],
    const device TraceInstance* traceInstances [[buffer(1)]], const device Material* materials [[buffer(2)]],
    const device uint* indices [[buffer(3)]], const device float* vertices [[buffer(4)]],
    const device Light* lights [[buffer(5)]], const device float* environmentDistribution [[buffer(6)]],
    const device float* specularAlbedo [[buffer(7)]],
    const device PtEmissiveTriangle* emissiveTriangles [[buffer(11)]],
    device PtWaveResult* results [[buffer(16)]], device PtWaveShadow* shadows [[buffer(17)]],
    device PtWaveGuide* waveGuides [[buffer(19)]],
    const device PtRestirSurface* surfaces [[buffer(23)]], const device PtReservoir* reservoirs [[buffer(24)]],
    device PtReservoir* finals [[buffer(25)]],
    texture2d<float> environmentMap [[texture(0)]],
    array<texture2d<float>, kHitTextureSlots> maps [[texture(7)]],
    sampler environmentSampler [[sampler(0)]], sampler materialSampler [[sampler(1)]],
    uint id [[thread_position_in_grid]]) {
  const uint width = uint(uniforms.image.x), height = uint(uniforms.image.y);
  if (id >= width * height) return;
  const PtRestirLights l = ptRestirLightsOf(uniforms);
  const float2 cameraCone = ptCameraCone(uniforms.lens);
  PtReservoir final = reservoirs[id];
  if ((uniforms.estimator.z & 2u) != 0u)
    final = ptRestirSpatial(PT_RESTIR_PASS_ARGS, l, cameraCone, id, uniforms.counts.w, width, height, surfaces, reservoirs);
  finals[id] = final;
  float3 shadowOrigin = float3(0.0f), shadowDirection = float3(0.0f, 1.0f, 0.0f), diffuseFraction = float3(1.0f);
  float shadowReach = 0.0f;
  float2 shadowCone = cameraCone;
  const float3 contribution = ptRestirShade(PT_RESTIR_PASS_ARGS, l, cameraCone, surfaces[id], final,
                                            pathSeed(id % width, id / width, uniforms.counts.w, uniforms.counts.z),
                                            shadowOrigin, shadowDirection, shadowReach, shadowCone, diffuseFraction);
  PtWaveShadow work;
  work.originReach = float4(shadowOrigin, ptMaxComponent(contribution) > 0.0f ? shadowReach : 0.0f);
  work.direction = float4(shadowDirection, ptWaveAsFloat(id));
  work.contribution = float4(contribution, ptWavePackCone(shadowCone));
  shadows[id] = work;
  PtWaveResult result;
  result.radiance = float4(0.0f);
  result.albedo = float4(0.0f);
  result.normal = float4(0.0f);
  results[id] = result;
  if (uniforms.image.z > 0.5f && uniforms.image.z < 1.5f) {
    PtWaveGuide guide;
    guide.diffuseFraction = float4(0.0f);
    guide.shadowFraction = float4(diffuseFraction, 0.0f);
    guide.diffuseRadiance = float4(0.0f);
    waveGuides[id] = guide;
  }
}
