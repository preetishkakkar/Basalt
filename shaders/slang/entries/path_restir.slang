// ReSTIR DI's reservoir passes, one thread per pixel, on every GPU backend. A frame runs: the
// backend's wavefront intersect stage at bounce 0 into per-pixel hits; path_restir_initial
// (initial resampling and temporal reuse); path_restir_spatial (spatial reuse, then the final
// reservoir's light as one shadow record per pixel); the backend's wavefront shadow stage, which
// adds the visible light to per-pixel results; then the path tracer, which takes those in place
// of the primary vertex's light sample. The per-pixel passes are pt_restir's, shared with the CPU
// tracer.
import pt_restir;
import pt_wavefront;
import pt_gpu;

ParameterBlock<PtScene<GpuMaps>> scene;

ConstantBuffer<PathUniforms> uniforms;
StructuredBuffer<PtWaveHit> hits;
StructuredBuffer<PtRestirCamera> restirCamera; // the previous camera, history validity
StructuredBuffer<PtRestirSurface> previousSurfaces;
StructuredBuffer<PtReservoir> previousReservoirs;
RWStructuredBuffer<PtRestirSurface> surfaces;
RWStructuredBuffer<PtReservoir> reservoirs;
RWStructuredBuffer<PtReservoir> finals;
RWStructuredBuffer<PtWaveResult> results;
RWStructuredBuffer<PtWaveShadow> shadows;
RWStructuredBuffer<PtWaveGuide> waveGuides;

PtRestir<GpuMaps> restir() {
  PtRestir<GpuMaps> context;
  context.scene = scene;
  context.uniforms = uniforms;
  return context;
}

// Initial resampling at the pixel's primary hit, then temporal reuse from the previous frame's
// final reservoirs.
[shader("compute")]
[numthreads(64, 1, 1)]
void path_restir_initial(uint3 thread: SV_DispatchThreadID) {
  const uint id = thread.x;
  const uint width = uint(uniforms.image.x), height = uint(uniforms.image.y);
  if (id >= width * height) return;
  const uint pixelX = id % width, pixelY = id / width;
  const uint candidates = min(uniforms.estimator.y, kPtRestirMaxCandidates);
  const uint seed = pathSeed(pixelX, pixelY, uniforms.counts.w, uniforms.counts.z);
  const float2 cameraCone = ptCameraCone(uniforms.lens);
  const PtWaveHit waveHit = hits[id];
  PtRestirSurface record = PtRestirSurface.none();
  PtReservoir r = PtReservoir.empty();
  r.count = float(candidates);
  if ((waveHit.flags & 1u) != 0u) {
    float3 origin;
    const float3 direction = ptCameraRay(uniforms, pixelX, pixelY, seed, origin);
    PtHit hit = PtHit(waveHit.t);
    hit.ambiguous = (waveHit.flags >> 1u) & 1u;
    hit.barycentric = float2(waveHit.barycentricX, waveHit.barycentricY);
    hit.instance = waveHit.instance;
    hit.primitive = waveHit.primitive;
    hit.found = 1u;
    const float hitWidth = ptConeWidthOrLevelZero(cameraCone, hit.t);
    const PtSurface surface = ptSurfaceAt(scene.geometry, scene.maps, hit, direction, hitWidth);
    record = PtRestirSurface(surface, hit, scene.geometry.traceInstances[hit.instance].material, direction, hitWidth,
                             uniforms.cameraPosition.xyz, uniforms.cameraForward.xyz);
    const PtBsdf bsdf = PtBsdf(surface, -direction, scene.specularAlbedo);
    const PtRestir<GpuMaps> context = restir();
    const PtReservoir initial =
        context.initial(bsdf, surface.position, surface.geometricNormal, ptConeShadow(cameraCone, hitWidth), seed, candidates);
    r = context.temporal(restirCamera[0], cameraCone, id, uniforms.counts.w, candidates, record, initial, previousSurfaces,
                         previousReservoirs);
  }
  surfaces[id] = record;
  reservoirs[id] = r;
}

// Spatial reuse (when enabled), the final reservoir kept for the next frame, and its light queued
// as the pixel's shadow record: the wavefront shadow stage adds it to results[pixel] (zeroed here)
// and its diffuse part to waveGuides[pixel] when guides are collected.
[shader("compute")]
[numthreads(64, 1, 1)]
void path_restir_spatial(uint3 thread: SV_DispatchThreadID) {
  const uint id = thread.x;
  const uint width = uint(uniforms.image.x), height = uint(uniforms.image.y);
  if (id >= width * height) return;
  const float2 cameraCone = ptCameraCone(uniforms.lens);
  const PtRestir<GpuMaps> context = restir();
  PtReservoir final = reservoirs[id];
  if ((uniforms.estimator.z & 2u) != 0u) final = context.spatial(cameraCone, id, uniforms.counts.w, width, height, surfaces, reservoirs);
  finals[id] = final;
  float3 shadowOrigin, shadowDirection, diffuseFraction;
  float shadowReach;
  float2 shadowCone;
  const float3 contribution = context.shade(cameraCone, surfaces[id], final, pathSeed(id % width, id / width, uniforms.counts.w,
                                                                                      uniforms.counts.z),
                                            shadowOrigin, shadowDirection, shadowReach, shadowCone, diffuseFraction);
  PtWaveShadow work;
  work.originReach = float4(shadowOrigin, ptMaxComponent(contribution) > 0.0f ? shadowReach : 0.0f);
  work.direction = float4(shadowDirection, asfloat(id));
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
