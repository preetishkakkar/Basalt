// The CPU path tracer's side of the shared code: compiled to C++ (slangc -target cpp/hpp) into
// the host's pt library. The host keeps what only it has (its texture copies with their
// filtering, Embree, the AVX2 wide traversal) and this module calls them through __extern_cpp
// functions it declares; everything else, the path loop and ReSTIR DI included, is the same
// source the GPU kernels are compiled from. The exports are the host's entry points.
module pt_cpu;

__exported import pt_restir;
__exported import pt_temporal;
__exported import pt_wide_bvh;
__exported import pt_bvh_build;
__exported import pt_exact_float;
__include pt_cpu_shared;

// What the host implements. `host` is its CpuScene and frame, as an opaque handle.
public __extern_cpp float4 ptHostSampleTexture(uint64_t host, uint slot, float2 uv, uint samplerCode, float lodBase);
public __extern_cpp float ptHostTextureFootprint(uint64_t host, uint slot, float lodBase);
public __extern_cpp float3 ptHostSampleEnvironment(uint64_t host, float2 uv);
public __extern_cpp PtHit ptHostTrace(uint64_t host, float3 origin, float3 direction, float tMax, uint mask, uint seed,
                                      float2 cone, uint anyHit);

// The hit texture table and the environment, as the host's copies.
public struct HostMaps : IPtMaps {
  uint64_t host;

  float4 sampleTexture(uint slot, float2 uv, uint samplerCode, float lodBase) {
    return ptHostSampleTexture(host, slot, uv, samplerCode, lodBase);
  }
  float textureFootprint(uint slot, float lodBase) { return ptHostTextureFootprint(host, slot, lodBase); }
  float3 sampleEnvironment(float2 uv) { return ptHostSampleEnvironment(host, uv); }
};

// The host's own intersectors (Embree, the AVX2 wide BVH).
public struct HostTracer : IPtTracer {
  uint64_t host;

  PtHit trace(float3 origin, float3 direction, float tMax, uint mask, uint seed, float2 cone, bool anyHit) {
    return ptHostTrace(host, origin, direction, tMax, mask, seed, cone, anyHit ? 1u : 0u);
  }
};

// A frame of the CPU tracer: the scene as buffers over the host's arrays, and which intersector
// traces it.
public __extern_cpp struct PtCpuScene {
  __extern_cpp StructuredBuffer<TraceInstance> traceInstances;
  __extern_cpp StructuredBuffer<Material> materials;
  __extern_cpp StructuredBuffer<uint> indices;
  __extern_cpp StructuredBuffer<float> vertices;
  __extern_cpp StructuredBuffer<Light> lights;
  __extern_cpp StructuredBuffer<float> environmentDistribution;
  __extern_cpp StructuredBuffer<float> specularAlbedo;
  __extern_cpp StructuredBuffer<PtEmissiveTriangle> emissiveTriangles;
  // The binary software BVH, traced here when intersector is 0; otherwise the host's.
  __extern_cpp StructuredBuffer<float4> bvhNodes;
  __extern_cpp StructuredBuffer<float4> bvhTriangles;
  __extern_cpp PathUniforms uniforms;
  // The host's CpuScene and frame, for its maps and intersectors.
  __extern_cpp uint64_t host;
  __extern_cpp uint intersector;

  PtScene<HostMaps> scene() {
    PtScene<HostMaps> result;
    result.geometry.traceInstances = traceInstances;
    result.geometry.materials = materials;
    result.geometry.indices = indices;
    result.geometry.vertices = vertices;
    result.maps.host = host;
    result.lights = lights;
    result.environmentDistribution = environmentDistribution;
    result.specularAlbedo = specularAlbedo;
    result.emissiveTriangles = emissiveTriangles;
    return result;
  }

  BinaryBvhTracer<HostMaps> binary() {
    BinaryBvhTracer<HostMaps> tracer;
    tracer.nodes = bvhNodes;
    tracer.triangles = bvhTriangles;
    tracer.geometry = scene().geometry;
    tracer.maps.host = host;
    return tracer;
  }

  HostTracer hostTracer() {
    HostTracer tracer;
    tracer.host = host;
    return tracer;
  }

  PtHit trace(float3 origin, float3 direction, float tMax, uint mask, uint seed, float2 cone, bool anyHit) {
    if (intersector == 0u) return binary().trace(origin, direction, tMax, mask, seed, cone, anyHit);
    return hostTracer().trace(origin, direction, tMax, mask, seed, cone, anyHit);
  }

  PtRestir<HostMaps> restir() {
    PtRestir<HostMaps> context;
    context.scene = scene();
    context.uniforms = uniforms;
    return context;
  }
};

PtPrimary noPrimary() {
  PtPrimary primary;
  primary.available = false;
  primary.surface = {};
  primary.hit = PtHit(0.0f);
  return primary;
}

// One camera path of a pixel; with ReSTIR DI, restirDirect and restirDiffuse are the pixel's
// visible direct light and its diffuse part from ptCpuRestirSpatialShade.
public export __extern_cpp PtPath ptCpuTracePath(PtCpuScene frame, uint pixelX, uint pixelY, uint sampleIndex, float3 restirDirect,
                                                 float3 restirDiffuse) {
  PtGuideRecord guides = PtGuideRecord(ptCollectsGuides(frame.uniforms));
  PtPath path;
  if (frame.intersector == 0u)
    path = ptTracePath<BinaryBvhTracer<HostMaps>, HostMaps, PtGuideRecord, true, true, false>(
        frame.binary(), frame.scene(), frame.uniforms, pixelX, pixelY, sampleIndex, noPrimary(), restirDirect, restirDiffuse, guides);
  else
    path = ptTracePath<HostTracer, HostMaps, PtGuideRecord, true, true, false>(
        frame.hostTracer(), frame.scene(), frame.uniforms, pixelX, pixelY, sampleIndex, noPrimary(), restirDirect, restirDiffuse,
        guides);
  path.guide = guides.guide;
  return path;
}

// ReSTIR DI's first pass for a pixel: its primary hit's record, initial resampling and temporal
// reuse against the previous frame's final reservoirs. A pixel without a surface gets an empty
// reservoir that still counts its candidates.
public export __extern_cpp void ptCpuRestirPrimary(PtCpuScene frame, uint pixelX, uint pixelY, uint sampleIndex, uint candidates,
                                                   PtRestirCamera previous, StructuredBuffer<PtRestirSurface> previousSurfaces,
                                                   StructuredBuffer<PtReservoir> previousReservoirs, out PtRestirSurface record,
                                                   out PtReservoir reservoir) {
  const PathUniforms uniforms = frame.uniforms;
  record = PtRestirSurface.none();
  reservoir = PtReservoir.empty();
  const uint seed = pathSeed(pixelX, pixelY, sampleIndex, uniforms.counts.z);
  const float2 cameraCone = ptCameraCone(uniforms.lens);
  float3 origin;
  const float3 direction = ptCameraRay(uniforms, pixelX, pixelY, seed, origin);
  const PtHit hit = frame.trace(origin, direction, kPtInfinity, uniforms.counts.y, ptTraceSeed(seed, 0u, 0u), cameraCone, false);
  if (hit.found == 0u) {
    reservoir.count = float(candidates);
    return;
  }
  const float hitWidth = ptConeWidthOrLevelZero(cameraCone, hit.t);
  const PtScene<HostMaps> scene = frame.scene();
  const PtSurface surface = ptSurfaceAt(scene.geometry, scene.maps, hit, direction, hitWidth);
  record = PtRestirSurface(surface, hit, frame.traceInstances[hit.instance].material, direction, hitWidth, uniforms.cameraPosition.xyz,
                           uniforms.cameraForward.xyz);
  const PtBsdf bsdf = PtBsdf(surface, -direction, frame.specularAlbedo);
  const PtRestir<HostMaps> restir = frame.restir();
  const PtReservoir initial =
      restir.initial(bsdf, surface.position, surface.geometricNormal, ptConeShadow(cameraCone, hitWidth), seed, candidates);
  const uint pixel = pixelY * uint(uniforms.image.x) + pixelX;
  reservoir = restir.temporal(previous, cameraCone, pixel, sampleIndex, candidates, record, initial, previousSurfaces,
                              previousReservoirs);
}

// ReSTIR DI's second pass for a pixel: spatial reuse (with estimator.z & 2), then the final
// reservoir's light and its shadow ray. direct and directDiffuse are zero when the light is not
// visible.
public export __extern_cpp void ptCpuRestirSpatialShade(PtCpuScene frame, uint pixelX, uint pixelY, uint sampleIndex,
                                                        StructuredBuffer<PtRestirSurface> surfaces,
                                                        StructuredBuffer<PtReservoir> reservoirs, out PtReservoir final,
                                                        out float3 direct, out float3 directDiffuse) {
  const PathUniforms uniforms = frame.uniforms;
  const uint width = uint(uniforms.image.x), height = uint(uniforms.image.y);
  const uint pixel = pixelY * width + pixelX;
  const float2 cameraCone = ptCameraCone(uniforms.lens);
  const PtRestir<HostMaps> restir = frame.restir();
  final = reservoirs[pixel];
  if ((uniforms.estimator.z & 2u) != 0u) final = restir.spatial(cameraCone, pixel, sampleIndex, width, height, surfaces, reservoirs);
  direct = float3(0.0f);
  directDiffuse = float3(0.0f);
  const uint seed = pathSeed(pixelX, pixelY, sampleIndex, uniforms.counts.z);
  float3 shadowOrigin, shadowDirection, diffuseFraction;
  float shadowReach;
  float2 shadowCone;
  const float3 contribution =
      restir.shade(cameraCone, surfaces[pixel], final, seed, shadowOrigin, shadowDirection, shadowReach, shadowCone, diffuseFraction);
  if (!(ptMaxComponent(contribution) > 0.0f)) return;
  const PtHit blocker = frame.trace(shadowOrigin, shadowDirection, shadowReach, uniforms.counts.y, ptTraceSeed(seed, 0u, 1u), shadowCone,
                                    true);
  if (blocker.found != 0u) return;
  direct = contribution;
  directDiffuse = contribution * diffuseFraction;
}

// The alpha test, for the host's intersectors.
public export __extern_cpp bool ptCpuCandidateSolid(PtCpuScene frame, uint instance, uint primitive, float2 barycentric, uint frontFacing,
                                                    uint seed, float3 direction, float coneWidth) {
  const PtScene<HostMaps> scene = frame.scene();
  return ptCandidateSolid(scene.geometry, scene.maps, instance, primitive, barycentric, frontFacing, seed, direction, coneWidth);
}

// The watertight triangle test, for the host's AVX2 wide traversal.
public export __extern_cpp PtRayPrep ptCpuPrepareRay(float3 origin, float3 direction) { return PtRayPrep(origin, direction); }
public export __extern_cpp bool ptCpuIntersectTriangle(PtRayPrep ray, float3 v0, float3 v1, float3 v2, float tMax, out float t,
                                                       out float2 barycentric, inout uint candidateFlags) {
  return ray.intersectTriangle(v0, v1, v2, tMax, t, barycentric, candidateFlags);
}

// Ray-cone levels for the host's texture sampling, bitwise the GPU's.
public export __extern_cpp float ptCpuTextureLevel(float lodBase, float width, float height, float levels) {
  return ptTextureLevel(lodBase, width, height, levels);
}
public export __extern_cpp float ptCpuTextureFootprint(float lodBase, float width, float height) {
  return ptTextureFootprint(lodBase, width, height);
}
