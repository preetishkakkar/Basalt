// The wavefront path tracer: per bounce, an intersect stage (nearest hit per live queue slot), a
// shade stage (escape or emission, the light sample queued as a shadow ray, the BSDF continuation
// queued for the next bounce) and a shadow stage; a resolve adds each batch to the accumulators.
// The transport is the megakernel's (pt_integrator), statement for statement; only the storage
// of a path between bounces differs. Fused shade stages trace their slot's nearest hit
// themselves. Queue slots are reserved per subgroup or, without subgroup arithmetic, one atomic
// per path. The queue ABI is pt_wavefront's.
import pt_integrator;
import pt_wide_bvh;
import pt_wavefront;
import pt_gpu;

// The scene and the traced structure, as the megakernels have them.
ParameterBlock<PtScene<GpuMaps>> scene;
struct TraceStructure {
  StructuredBuffer<float4> bvhNodes;      // binary
  StructuredBuffer<PtWideNode> wideNodes; // BVH4 / BVH8
  StructuredBuffer<float4> bvhTriangles;
  RaytracingAccelerationStructure accelerationStructure;
};
ParameterBlock<TraceStructure> structure;

// This frame's queues.
ConstantBuffer<PathUniforms> uniforms;
RWStructuredBuffer<PtWaveState> statesA;
RWStructuredBuffer<PtWaveState> statesB;
RWStructuredBuffer<uint> counters;
StructuredBuffer<uint4> waveControl; // [0] parity, capacity, batch base, batch count; [1] bounce, samples, flags, first pixel
RWStructuredBuffer<PtWaveResult> results;
RWStructuredBuffer<PtWaveShadow> shadows;
RWStructuredBuffer<PtWaveHit> hits;
RWStructuredBuffer<PtWaveGuide> waveGuides;
RWStructuredBuffer<PtReconstructionSample> guides;
RWStructuredBuffer<uint2> costs;
// ReSTIR DI: the primary vertex's visible direct light and its diffuse part.
StructuredBuffer<PtWaveResult> restirResults;
StructuredBuffer<PtWaveGuide> restirGuides;
[format("rgba32f")] RWTexture2D<float4> accumulation;
[format("rgba32f")] RWTexture2D<float4> albedoAccumulation;
[format("rgba32f")] RWTexture2D<float4> normalAccumulation;
WTexture2D<float4> output;

// Reserves one slot of queue group `group` for the calling lane. A slot >= capacity was not
// reserved and is counted as overflow. With SUBGROUP the active lanes of a subgroup reserve
// together, in lane order: one atomic add and one atomic max per subgroup instead of per path.
// Only the lanes that enqueue may call it.
uint waveReserve<let SUBGROUP : bool>(uint group, uint capacity) {
  if (SUBGROUP) {
    const uint offset = WavePrefixSum(1u);
    const uint total = WaveActiveSum(1u);
    uint base = 0u;
    if (WaveIsFirstLane()) {
      InterlockedAdd(counters[group + kWaveQueueCount], total, base);
      const uint end = min(base + total, capacity);
      if (end > base) InterlockedMax(counters[group], (end + kWaveGroup - 1u) / kWaveGroup);
      if (base + total > capacity) InterlockedAdd(counters[kWaveFrameOverflow], base + total - max(base, capacity));
    }
    return WaveReadLaneFirst(base) + offset;
  }
  uint slot;
  InterlockedAdd(counters[group + kWaveQueueCount], 1u, slot);
  if (slot < capacity) InterlockedMax(counters[group], (slot + kWaveGroup) / kWaveGroup);
  else InterlockedAdd(counters[kWaveFrameOverflow], 1u);
  return slot;
}

// A batch path's pixel, sample and path seed from its batch-local index.
struct WavePath {
  uint local;
  uint pixel;
  uint pixelX, pixelY;
  uint sampleInPixel;
  uint seed;

  __init(uint localIndex) {
    const uint4 control = waveControl[0];
    const uint4 frame = waveControl[1];
    local = localIndex;
    const uint g = control.z + local;
    pixel = g / frame.y;
    sampleInPixel = g % frame.y;
    pixelX = pixel % uint(uniforms.image.x);
    pixelY = pixel / uint(uniforms.image.x);
    seed = pathSeed(pixelX, pixelY, uniforms.counts.w + sampleInPixel, uniforms.counts.z);
  }
};

// The live slots of this bounce's continuation queue: the batch's paths at bounce 0.
uint liveSlots() {
  const uint4 control = waveControl[0];
  const uint parity = control.x & 1u;
  return waveControl[1].x == 0u ? control.w : min(counters[parity * 8u + kWaveQueueCount], control.y);
}

PtWaveState queuedState(uint id) {
  return (waveControl[0].x & 1u) == 0u ? statesA[id] : statesB[id];
}

PtWaveHit compactHit(PtHit hit) {
  PtWaveHit compact;
  compact.t = hit.t;
  compact.barycentricX = hit.barycentric.x;
  compact.barycentricY = hit.barycentric.y;
  compact.instance = hit.instance;
  compact.primitive = hit.primitive;
  compact.flags = (hit.found != 0u ? 1u : 0u) | (hit.ambiguous != 0u ? 2u : 0u);
  return compact;
}

// The traversal work of a path's rays, for the BVH cost views only: the path's own slot,
// integer and exact.
void addCosts(uint local, PtHit hit, bool first) {
  if (uniforms.image.z <= 1.5f) return;
  if (first) costs[local] = uint2(hit.nodeVisits, hit.triangleTests);
  else costs[local] += uint2(hit.nodeVisits, hit.triangleTests);
}

// ---- tracers

BinaryBvhTracer<GpuMaps> binaryTracer() {
  BinaryBvhTracer<GpuMaps> tracer;
  tracer.nodes = structure.bvhNodes;
  tracer.triangles = structure.bvhTriangles;
  tracer.geometry = scene.geometry;
  tracer.maps = scene.maps;
  return tracer;
}

// The wavefront stages' wide traversal sorts its children (pt_wide_bvh.slang says why).
WideBvhTracer<GpuMaps, STACK, true> wideTracer<let STACK : int>() {
  WideBvhTracer<GpuMaps, STACK, true> tracer;
  tracer.nodes = structure.wideNodes;
  tracer.triangles = structure.bvhTriangles;
  tracer.geometry = scene.geometry;
  tracer.maps = scene.maps;
  return tracer;
}

RayQueryTracer<GpuMaps> rayQueryTracer() {
  RayQueryTracer<GpuMaps> tracer;
  tracer.scene = structure.accelerationStructure;
  tracer.geometry = scene.geometry;
  tracer.maps = scene.maps;
  return tracer;
}

// The full ray pipeline's tracers: nearest hits through the hit group with the alpha test, and
// shadow rays that stop at the first solid candidate.
struct WavePayload {
  uint ambiguous;
  float t;
  float2 barycentric;
  uint instance;
  uint primitive;
  uint found;
  uint seed;
  float2 cone;
};

struct RayPipelineTracer : IPtTracer {
  RaytracingAccelerationStructure scene;

  PtHit trace(float3 origin, float3 direction, float tMax, uint mask, uint seed, float2 cone, bool anyHit) {
    WavePayload payload = {};
    payload.seed = seed;
    payload.cone = cone;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0f;
    ray.TMax = tMax;
    TraceRay(scene, kPtRayFlags, mask, 0u, 1u, 0u, ray, payload);
    PtHit hit = PtHit(payload.t);
    hit.ambiguous = payload.ambiguous;
    hit.barycentric = payload.barycentric;
    hit.instance = payload.instance;
    hit.primitive = payload.primitive;
    hit.found = payload.found;
    return hit;
  }
};

struct ShadowPayload {
  uint found;
  uint seed;
  float2 cone;
};

struct ShadowPipelineTracer : IPtTracer {
  RaytracingAccelerationStructure scene;

  PtHit trace(float3 origin, float3 direction, float tMax, uint mask, uint seed, float2 cone, bool anyHit) {
    ShadowPayload payload = {};
    // Ends at the first solid hit with no closest-hit stage: found stays set unless the miss
    // stage clears it.
    payload.found = 1u;
    payload.seed = seed;
    payload.cone = cone;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0f;
    ray.TMax = tMax;
    TraceRay(scene, kPtRayFlags | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, mask, 0u, 1u, 0u,
             ray, payload);
    PtHit hit = PtHit(0.0f);
    hit.found = payload.found;
    return hit;
  }
};

// ---- intersect: the nearest hit of every live slot; at bounce 0 the slots are the batch's paths
// in order and their camera rays are generated here.

void intersect<T : IPtTracer>(T tracer, uint id) {
  if (id >= liveSlots()) return;
  const uint bounce = waveControl[1].x;
  float3 origin, direction;
  uint local = id;
  float2 cone = ptCameraCone(uniforms.lens);
  if (bounce == 0u) {
    const WavePath path = WavePath(id);
    direction = ptCameraRay(uniforms, path.pixelX, path.pixelY, path.seed, origin);
  } else {
    const PtWaveState state = queuedState(id);
    origin = state.origin.xyz;
    direction = state.direction.xyz;
    local = asuint(state.direction.w);
    cone = ptWaveUnpackCone(state.throughput.w);
  }
  const WavePath path = WavePath(local);
  const PtHit hit =
      tracer.trace(origin, direction, kPtInfinity, uniforms.counts.y, ptTraceSeed(path.seed, bounce, 0u), cone, false);
  hits[id] = compactHit(hit);
  addCosts(local, hit, bounce == 0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void path_wavefront_intersect(uint3 id: SV_DispatchThreadID) { intersect(binaryTracer(), id.x); }

[shader("compute")]
[numthreads(64, 1, 1)]
void path_wavefront_intersect_wide(uint3 id: SV_DispatchThreadID) { intersect(wideTracer<kWideStack>(), id.x); }

// The deep traversal stack, for trees whose stack bound exceeds 64.
[shader("compute")]
[numthreads(64, 1, 1)]
void path_wavefront_intersect_wide_deep(uint3 id: SV_DispatchThreadID) { intersect(wideTracer<kWideStackDeep>(), id.x); }

[shader("compute")]
[numthreads(64, 1, 1)]
void path_wavefront_intersect_rt(uint3 id: SV_DispatchThreadID) { intersect(rayQueryTracer(), id.x); }

[shader("raygeneration")]
void path_wavefront_intersect_pipeline() {
  RayPipelineTracer tracer;
  tracer.scene = structure.accelerationStructure;
  intersect(tracer, DispatchRaysIndex().x);
}

[shader("miss")]
void path_wave_pipeline_miss(inout WavePayload payload) { payload.found = 0u; }

[shader("closesthit")]
void path_wave_pipeline_closest(inout WavePayload payload, BuiltInTriangleIntersectionAttributes attributes) {
  payload.t = RayTCurrent();
  payload.barycentric = attributes.barycentrics;
  payload.instance = InstanceIndex();
  payload.primitive = PrimitiveIndex();
  payload.found = 1u;
}

[shader("anyhit")]
void path_wave_pipeline_alpha(inout WavePayload payload, BuiltInTriangleIntersectionAttributes attributes) {
  const uint instance = InstanceIndex();
  if ((scene.geometry.traceInstances[instance].flags & kInstanceBlended) != 0u) payload.ambiguous = 1u;
  const uint front = HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE ? 1u : 0u;
  if (!ptCandidateSolid(scene.geometry, scene.maps, instance, PrimitiveIndex(), attributes.barycentrics, front, payload.seed,
                        WorldRayDirection(), ptConeWidthOrLevelZero(payload.cone, RayTCurrent())))
    IgnoreHit();
}

// ---- shade: one bounce of every live slot.

void shade<let SUBGROUP : bool, let FUSED : bool, T : IPtTracer>(T tracer, uint id) {
  const uint4 control = waveControl[0];
  const uint4 frame = waveControl[1];
  const uint parity = control.x & 1u;
  const uint capacity = control.y;
  const uint bounce = frame.x;
  if (id >= liveSlots()) return;
  PtWaveState state;
  state.origin = float4(uniforms.cameraPosition.xyz, 0.0f);
  state.direction = float4(0.0f, 0.0f, 1.0f, 0.0f);
  state.throughput = float4(1.0f);
  uint local = id;
  if (bounce > 0u) {
    state = queuedState(id);
    local = asuint(state.direction.w);
  }
  const WavePath path = WavePath(local);
  const uint pixel = path.pixel;
  // The path's ray cone: the camera's at bounce 0, else the state's.
  const float2 cone = bounce == 0u ? ptCameraCone(uniforms.lens) : ptWaveUnpackCone(state.throughput.w);
  if (bounce == 0u) {
    float3 cameraOrigin;
    state.direction = float4(ptCameraRay(uniforms, path.pixelX, path.pixelY, path.seed, cameraOrigin), 0.0f);
    state.origin = float4(cameraOrigin, 0.0f);
  }
  const bool collectGuides = uniforms.image.z > 0.5f && uniforms.image.z < 1.5f && path.sampleInPixel + 1u == frame.y;
  const float3 direction = state.direction.xyz;
  float3 diffuseFraction = float3(1.0f);
  if (collectGuides && bounce > 0u) diffuseFraction = waveGuides[pixel].diffuseFraction.xyz;
  float3 guideDiffuse = float3(0.0f);
  float3 radiance = float3(0.0f), albedo = float3(0.0f), normal = float3(0.0f);
  PtWaveHit waveHit;
  if (FUSED) {
    // Fused mode traces the slot's nearest hit here instead of reading the intersect stage's.
    const PtHit traced = tracer.trace(state.origin.xyz, direction, kPtInfinity, uniforms.counts.y,
                                      ptTraceSeed(path.seed, bounce, 0u), cone, false);
    waveHit = compactHit(traced);
    addCosts(local, traced, bounce == 0u);
  } else {
    waveHit = hits[id];
  }
  const uint strategy = uint(uniforms.path.w);
  uint kindCount = 0u;
  if (strategy != 1u && uniforms.distribution.w > 0.5f) ++kindCount;
  if (strategy != 1u && uniforms.sunDirection.w > 0.0f) ++kindCount;
  if (strategy != 1u && uniforms.counts.x > 0u) ++kindCount;
  if (strategy != 1u && uniforms.emissive.x > 0u) ++kindCount;
  const float kindShare = 1.0f / max(float(kindCount), 1.0f);
  PtEmitters emitters;
  emitters.triangles = scene.emissiveTriangles;
  emitters.count = uniforms.emissive.x;
  bool continuing = false;
  if ((waveHit.flags & 1u) == 0u) {
    if (bounce == 0u && uniforms.environment.w > 0.5f)
      albedo = saturate(scene.maps.sampleEnvironment(equirectangularUV(direction)) * uniforms.environment.x);
    if (bounce == 0u && collectGuides) guides[pixel] = PtReconstructionSample.empty();
    if (bounce > 0u || uniforms.environment.w > 0.5f) {
      float environmentWeight = 1.0f, sunWeight = 1.0f;
      if (bounce > 0u && strategy == 2u) {
        environmentWeight = 0.0f;
        sunWeight = 0.0f;
      }
      if (bounce > 0u && strategy == 0u && uniforms.distribution.w > 0.5f)
        environmentWeight = ptPowerHeuristic(state.origin.w,
                                             kindShare * ptEnvironmentPdf(scene.environmentDistribution, uniforms.distribution, direction));
      if (bounce > 0u && strategy == 0u && uniforms.sunDirection.w > 0.0f)
        sunWeight = ptPowerHeuristic(state.origin.w, kindShare * (1.0f / max(uniforms.sunRadiance.w, 1e-12f)));
      float3 escaped = scene.maps.sampleEnvironment(equirectangularUV(direction)) * (uniforms.environment.x * environmentWeight);
      if (uniforms.sunDirection.w > 0.0f && dot(direction, uniforms.sunDirection.xyz) >= uniforms.sunDirection.w)
        escaped += uniforms.sunRadiance.xyz * sunWeight;
      escaped *= state.throughput.xyz;
      if (bounce > 0u) escaped = ptClampContribution(escaped, uniforms.path.z);
      radiance += escaped;
      guideDiffuse += escaped * diffuseFraction;
    }
  } else {
    PtHit hit = PtHit(waveHit.t);
    hit.ambiguous = (waveHit.flags >> 1u) & 1u;
    hit.barycentric = float2(waveHit.barycentricX, waveHit.barycentricY);
    hit.instance = waveHit.instance;
    hit.primitive = waveHit.primitive;
    hit.found = 1u;
    const float hitWidth = ptConeWidthOrLevelZero(cone, hit.t);
    const PtSurface surface = ptSurfaceAt(scene.geometry, scene.maps, hit, direction, hitWidth);
    if (bounce == 0u) {
      albedo = surface.baseColor;
      normal = surface.normal;
      if (collectGuides) guides[pixel] = ptPrimaryGuide(scene.geometry, uniforms, surface, hit, direction);
    } else if (collectGuides && bounce == 1u) {
      guides[pixel].viewDistance.w = hit.t;
    }
    // The megakernel's EMISSIVE weight: 1 at bounce 0, with BSDF sampling only or without
    // emitters; 0 with light sampling only; otherwise the power heuristic.
    float emissiveWeight = strategy == 2u && bounce > 0u ? 0.0f : 1.0f;
    if (bounce > 0u && strategy == 0u && uniforms.emissive.x > 0u)
      emissiveWeight = ptPowerHeuristic(state.origin.w, kindShare * emitters.pdf(scene.geometry, hit, direction));
    const float3 emitted = state.throughput.xyz * surface.emissive * emissiveWeight;
    radiance += emitted;
    guideDiffuse += emitted * diffuseFraction;
    if (bounce < uint(uniforms.path.x)) {
      const PtBsdf bsdf = PtBsdf(surface, -direction, scene.specularAlbedo);
      const float3 rayOrigin = ptOffsetRay(surface.position, surface.geometricNormal);
      const float4 lightRandom = pathRandom4(path.seed, bounce + 1u, 0u);
      const float4 bsdfRandom = pathRandom4(path.seed, bounce + 1u, 1u);
      uint kind = 4u;
      // The light kind, counted down as ptTracePath does.
      uint slot = kindCount > 0u ? min(uint(lightRandom.z * float(kindCount)), kindCount - 1u) : 0u;
      if (kindCount > 0u && strategy != 1u && uniforms.distribution.w > 0.5f) {
        if (slot == 0u) kind = 0u;
        slot--;
      }
      if (kindCount > 0u && strategy != 1u && uniforms.sunDirection.w > 0.0f && kind == 4u) {
        if (slot == 0u) kind = 1u;
        slot--;
      }
      if (kindCount > 0u && strategy != 1u && uniforms.counts.x > 0u && kind == 4u) {
        if (slot == 0u) kind = 2u;
        slot--;
      }
      if (kindCount > 0u && strategy != 1u && uniforms.emissive.x > 0u && kind == 4u) kind = 3u;
      // ReSTIR DI replaces the primary vertex's light sample with its reservoir's visible light.
      if (bounce == 0u && uniforms.estimator.x == 1u) {
        kind = 4u;
        radiance += restirResults[pixel].radiance.xyz;
        if (collectGuides) guideDiffuse += restirGuides[pixel].diffuseRadiance.xyz;
      }
      float3 lightDirection = float3(0.0f, 1.0f, 0.0f), arriving = float3(0.0f);
      float lightPdf = 0.0f, reach = kPtInfinity;
      if (kind == 0u) {
        lightDirection = ptSampleEnvironmentDirection(scene.environmentDistribution, uniforms.distribution, lightRandom.xy, lightPdf);
        lightPdf /= max(float(kindCount), 1.0f);
        arriving = scene.maps.sampleEnvironment(equirectangularUV(lightDirection)) * uniforms.environment.x;
      } else if (kind == 1u) {
        lightDirection = ptSampleCone(uniforms.sunDirection.xyz, uniforms.sunDirection.w, lightRandom.xy);
        lightPdf = (1.0f / max(uniforms.sunRadiance.w, 1e-12f)) / max(float(kindCount), 1.0f);
        arriving = uniforms.sunRadiance.xyz;
      } else if (kind == 2u) {
        const uint pick = min(uint(lightRandom.w * float(uniforms.counts.x)), uniforms.counts.x - 1u);
        float distance;
        arriving = ptPunctualLight(scene.lights[pick], surface.position, lightDirection, distance) *
                   (float(uniforms.counts.x) * float(kindCount));
        reach = distance * 0.999f;
      } else if (kind == 3u) {
        arriving = emitters.sample(scene.geometry, scene.maps, surface.position, float3(lightRandom.w, lightRandom.x, lightRandom.y),
                                   ptConeScattered(cone, hitWidth, bsdf.sampledAlpha(bsdfRandom.z)), lightDirection, reach, lightPdf);
        lightPdf /= max(float(kindCount), 1.0f);
        reach *= 0.9999f;
      }
      const float3 sampledDirection = bsdf.sampleDirection(bsdfRandom.xyz);
      float lightBsdfPdf, sampledPdf;
      const float3 lightValue = bsdf.evaluate(surface.geometricNormal, lightDirection, lightBsdfPdf);
      const float3 sampledValue = bsdf.evaluate(surface.geometricNormal, sampledDirection, sampledPdf);
      if (kind != 4u) {
        float3 contribution = state.throughput.xyz * lightValue * arriving;
        if (kind != 2u) {
          float weight = lightPdf > 0.0f ? 1.0f / lightPdf : 0.0f;
          if (strategy == 0u && lightPdf > 0.0f) weight = ptPowerHeuristic(lightPdf, lightBsdfPdf) / lightPdf;
          contribution *= weight;
        }
        if (ptMaxComponent(contribution) > 0.0f) {
          contribution = ptClampContribution(contribution, uniforms.path.z);
          const uint queued = waveReserve<SUBGROUP>(kWaveShadowGroup, capacity);
          if (queued < capacity) {
            PtWaveShadow work;
            work.originReach = float4(dot(lightDirection, surface.geometricNormal) < 0.0f
                                          ? ptOffsetRay(surface.position, -surface.geometricNormal)
                                          : rayOrigin,
                                      reach);
            work.direction = float4(lightDirection, asfloat(local));
            // contribution.w: the shadow ray's cone; the shadow stage derives the any-hit seed.
            work.contribution = float4(contribution, ptWavePackCone(ptConeShadow(cone, hitWidth)));
            shadows[queued] = work;
            if (collectGuides)
              waveGuides[pixel].shadowFraction =
                  float4(bounce == 0u ? bsdf.diffuseFraction(lightDirection, lightValue) : diffuseFraction, 0.0f);
          }
        }
      }
      continuing = sampledPdf > 0.0f && ptMaxComponent(sampledValue) > 0.0f;
      if (continuing) {
        if (collectGuides && bounce == 0u)
          waveGuides[pixel].diffuseFraction = float4(bsdf.diffuseFraction(sampledDirection, sampledValue), 0.0f);
        state.throughput.xyz *= sampledValue / sampledPdf;
        state.origin = float4(dot(sampledDirection, surface.geometricNormal) < 0.0f
                                  ? ptOffsetRay(surface.position, -surface.geometricNormal)
                                  : rayOrigin,
                              sampledPdf);
        state.direction = float4(sampledDirection, asfloat(local));
        state.throughput.w = ptWavePackCone(ptConeScattered(cone, hitWidth, bsdf.sampledAlpha(bsdfRandom.z)));
        if (bounce + 1u >= uint(uniforms.path.y)) {
          const float survive = min(ptMaxComponent(state.throughput.xyz), 0.95f);
          if (bsdfRandom.w >= survive) continuing = false;
          else state.throughput.xyz /= survive;
        }
      }
    }
  }
  if (bounce == 0u) {
    PtWaveResult result;
    result.radiance = float4(radiance, 0.0f);
    result.albedo = float4(albedo, 0.0f);
    result.normal = float4(normal, 0.0f);
    results[local] = result;
  } else if (any(radiance != float3(0.0f))) {
    results[local].radiance += float4(radiance, 0.0f);
  }
  if (collectGuides) {
    if (bounce == 0u) waveGuides[pixel].diffuseRadiance = float4(guideDiffuse, 0.0f);
    else if (any(guideDiffuse != float3(0.0f))) waveGuides[pixel].diffuseRadiance += float4(guideDiffuse, 0.0f);
  }
  if (continuing) {
    const uint next = 1u - parity;
    const uint slotOut = waveReserve<SUBGROUP>(next * 8u, capacity);
    if (slotOut < capacity) {
      if (next == 0u) statesA[slotOut] = state;
      else statesB[slotOut] = state;
    }
  }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void path_wavefront_shade(uint3 id: SV_DispatchThreadID) { shade<true, false>(binaryTracer(), id.x); }

// For devices without subgroup arithmetic.
[shader("compute")]
[numthreads(64, 1, 1)]
void path_wavefront_shade_atomic(uint3 id: SV_DispatchThreadID) { shade<false, false>(binaryTracer(), id.x); }

[shader("compute")]
[numthreads(64, 1, 1)]
void path_wavefront_fused(uint3 id: SV_DispatchThreadID) { shade<true, true>(binaryTracer(), id.x); }

[shader("compute")]
[numthreads(64, 1, 1)]
void path_wavefront_fused_rt(uint3 id: SV_DispatchThreadID) { shade<true, true>(rayQueryTracer(), id.x); }

[shader("compute")]
[numthreads(64, 1, 1)]
void path_wavefront_fused_wide(uint3 id: SV_DispatchThreadID) { shade<true, true>(wideTracer<kWideStack>(), id.x); }

// The deep traversal stack, for trees whose stack bound exceeds 64.
[shader("compute")]
[numthreads(64, 1, 1)]
void path_wavefront_fused_wide_deep(uint3 id: SV_DispatchThreadID) { shade<true, true>(wideTracer<kWideStackDeep>(), id.x); }

// ---- shadow: occlusion for every queued shadow ray of this bounce; an unoccluded one adds its
// contribution to its path's radiance (and diffuse guide for the guide path).

void shadow<T : IPtTracer>(T tracer, uint id) {
  const uint4 control = waveControl[0];
  const uint4 frame = waveControl[1];
  if (id >= min(counters[kWaveShadowGroup + kWaveQueueCount], control.y)) return;
  const PtWaveShadow work = shadows[id];
  const WavePath path = WavePath(asuint(work.direction.w));
  const PtHit hit = tracer.trace(work.originReach.xyz, work.direction.xyz, work.originReach.w, uniforms.counts.y,
                                 ptTraceSeed(path.seed, frame.x, 1u), ptWaveUnpackCone(work.contribution.w), true);
  if (uniforms.image.z > 1.5f) costs[path.local] += uint2(hit.nodeVisits, hit.triangleTests);
  if (hit.found != 0u) return;
  results[path.local].radiance += float4(work.contribution.xyz, 0.0f);
  if (uniforms.image.z > 0.5f && uniforms.image.z < 1.5f && path.sampleInPixel + 1u == frame.y)
    waveGuides[path.pixel].diffuseRadiance += float4(work.contribution.xyz * waveGuides[path.pixel].shadowFraction.xyz, 0.0f);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void path_wavefront_shadow(uint3 id: SV_DispatchThreadID) { shadow(binaryTracer(), id.x); }

[shader("compute")]
[numthreads(64, 1, 1)]
void path_wavefront_shadow_wide(uint3 id: SV_DispatchThreadID) { shadow(wideTracer<kWideStack>(), id.x); }

[shader("compute")]
[numthreads(64, 1, 1)]
void path_wavefront_shadow_wide_deep(uint3 id: SV_DispatchThreadID) { shadow(wideTracer<kWideStackDeep>(), id.x); }

[shader("compute")]
[numthreads(64, 1, 1)]
void path_wavefront_shadow_rt(uint3 id: SV_DispatchThreadID) { shadow(rayQueryTracer(), id.x); }

[shader("raygeneration")]
void path_wavefront_shadow_pipeline() {
  ShadowPipelineTracer tracer;
  tracer.scene = structure.accelerationStructure;
  shadow(tracer, DispatchRaysIndex().x);
}

[shader("miss")]
void path_wave_shadow_pipeline_miss(inout ShadowPayload payload) { payload.found = 0u; }

[shader("anyhit")]
void path_wave_shadow_pipeline_alpha(inout ShadowPayload payload, BuiltInTriangleIntersectionAttributes attributes) {
  const uint front = HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE ? 1u : 0u;
  if (!ptCandidateSolid(scene.geometry, scene.maps, InstanceIndex(), PrimitiveIndex(), attributes.barycentrics, front, payload.seed,
                        WorldRayDirection(), ptConeWidthOrLevelZero(payload.cone, RayTCurrent())))
    IgnoreHit();
}

// ---- resolve: adds one batch's pixels to the accumulators. Each pixel's S paths are consecutive
// in the batch and are summed in sample order, then added once, exactly as the megakernel does:
// no atomics, and a non-finite path contributes no radiance but still counts. With flag bit 0
// clear (the target was reached) the pass only rewrites the mean, for every pixel.
//
// A workgroup covers kResolveGroup / S whole pixels when S <= kResolveGroup: its threads first
// load the pixels' results into shared memory with consecutive (coalesced) reads, then one
// thread per pixel sums them in sample order. Larger S reads directly.

static const uint kResolveGroup = 128u;
groupshared float4 radianceTile[kResolveGroup];
groupshared float4 albedoTile[kResolveGroup];
groupshared float4 normalTile[kResolveGroup];

// BVH cost views: the path's own traversal work, on the megakernel's fixed scale.
float3 costColour(uint2 cost, float mode) {
  const float value = saturate(mode < 2.5f ? float(cost.x) / 128.0f : float(cost.y) / 64.0f);
  return float3(saturate(value * 3.0f - 1.5f), saturate(1.0f - abs(value * 3.0f - 1.5f)), saturate(1.5f - value * 3.0f));
}

[shader("compute")]
[numthreads(128, 1, 1)]
void path_wavefront_resolve(uint lane: SV_GroupIndex, uint3 groupId: SV_GroupID) {
  const uint group = groupId.x;
  const uint4 control = waveControl[0];
  const uint4 frame = waveControl[1];
  const bool accumulate = (frame.z & 1u) != 0u;
  const uint samples = max(frame.y, 1u);
  const uint batchPixels = accumulate ? control.w / samples : uint(uniforms.image.x) * uint(uniforms.image.y);
  const bool tiled = accumulate && samples <= kResolveGroup;
  const uint pixelsPerGroup = tiled ? kResolveGroup / samples : kResolveGroup;
  const uint firstInGroup = group * pixelsPerGroup;
  // Load phase (uniform for the whole group: no thread returns before the barrier).
  if (tiled) {
    const uint element = lane;
    const uint pixelInGroup = element / samples;
    if (pixelInGroup < pixelsPerGroup && firstInGroup + pixelInGroup < batchPixels) {
      const uint local = (frame.w + firstInGroup + pixelInGroup) * samples + element % samples - control.z;
      const PtWaveResult result = results[local];
      float4 radiance = result.radiance, albedo = result.albedo, normal = result.normal;
      if (uniforms.image.z > 1.5f) {
        radiance = float4(costColour(costs[local], uniforms.image.z), 0.0f);
        albedo = float4(0.0f);
        normal = float4(0.0f);
      }
      radianceTile[element] = radiance;
      albedoTile[element] = albedo;
      normalTile[element] = normal;
    }
  }
  GroupMemoryBarrierWithGroupSync();
  if (lane >= pixelsPerGroup || firstInGroup + lane >= batchPixels) return;
  const uint pixel = (accumulate ? frame.w : 0u) + firstInGroup + lane;
  const uint2 xy = uint2(pixel % uint(uniforms.image.x), pixel / uint(uniforms.image.x));
  if (!accumulate) {
    const float4 total = accumulation[xy];
    output.Store(xy, float4(total.xyz / max(total.w, 1.0f), 1.0f));
    return;
  }
  float3 sum = float3(0.0f), albedoSum = float3(0.0f), normalSum = float3(0.0f);
  float3 lastRadiance = float3(0.0f);
  const uint first = pixel * samples - control.z;
  for (uint s = 0u; s < samples; ++s) {
    float3 radiance, albedo, normal;
    if (tiled) {
      radiance = radianceTile[lane * samples + s].xyz;
      albedo = albedoTile[lane * samples + s].xyz;
      normal = normalTile[lane * samples + s].xyz;
    } else {
      const PtWaveResult result = results[first + s];
      radiance = result.radiance.xyz;
      albedo = result.albedo.xyz;
      normal = result.normal.xyz;
      if (uniforms.image.z > 1.5f) {
        radiance = costColour(costs[first + s], uniforms.image.z);
        albedo = float3(0.0f);
        normal = float3(0.0f);
      }
    }
    lastRadiance = radiance;
    if (!(any(isnan(radiance)) || any(isinf(radiance)))) sum = sum + radiance;
    albedoSum = albedoSum + albedo;
    normalSum = normalSum + normal;
  }
  float4 total = float4(sum, float(samples));
  float4 albedoTotal = float4(albedoSum, 0.0f), normalTotal = float4(normalSum, 0.0f);
  // A restarted sample offset replaces the sums (execution, camera or backend changes).
  if (uniforms.counts.w > 0u) {
    total = total + accumulation[xy];
    albedoTotal = albedoTotal + albedoAccumulation[xy];
    normalTotal = normalTotal + normalAccumulation[xy];
  }
  accumulation[xy] = total;
  albedoAccumulation[xy] = albedoTotal;
  normalAccumulation[xy] = normalTotal;
  if (uniforms.image.z > 0.5f && uniforms.image.z < 1.5f) {
    // The guide path is the pixel's last sample. Only its radiance split is written here: the
    // rest of the record is left as the shade stage wrote it, not read back and rewritten.
    const float3 diffuse = waveGuides[pixel].diffuseRadiance.xyz;
    guides[pixel].diffuse = float4(diffuse, 0.0f);
    guides[pixel].specular = float4(max(lastRadiance - diffuse, float3(0.0f)), 0.0f);
  }
  output.Store(xy, float4(total.xyz / max(total.w, 1.0f), 1.0f));
}
