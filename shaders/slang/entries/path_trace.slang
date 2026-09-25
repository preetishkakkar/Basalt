// The GPU path tracer's megakernels: one thread per pixel, `image.w` samples per dispatch,
// added to float accumulators; this frame's lit image receives the mean. Every entry is the
// same generic pixel function around a different intersector: the binary or quantized wide
// software BVH, or the scene's acceleration structure through ray queries (with the hybrid
// taking its primary surfaces from the rasteriser's G-buffer).
import pt_integrator;
import pt_wide_bvh;
import pt_wavefront;
import pt_gpu;

// The scene, shared by every frame.
ParameterBlock<PtScene<GpuMaps>> scene;

// The software BVHs' nodes and triangles, or the hardware structure.
struct TraceStructure {
  StructuredBuffer<float4> bvhNodes;          // binary
  StructuredBuffer<PtWideNode> wideNodes;     // BVH4 / BVH8
  StructuredBuffer<float4> bvhTriangles;
  RaytracingAccelerationStructure accelerationStructure;
};
ParameterBlock<TraceStructure> structure;

// This frame's.
ConstantBuffer<PathUniforms> uniforms;
RWStructuredBuffer<PtReconstructionSample> reconstructionSamples;
[format("rgba32f")] RWTexture2D<float4> accumulation;
[format("rgba32f")] RWTexture2D<float4> albedoAccumulation;
[format("rgba32f")] RWTexture2D<float4> normalAccumulation;
WTexture2D<float4> output;
// ReSTIR DI: the primary vertex's visible direct light and its diffuse part, per pixel, from
// the reservoir passes.
StructuredBuffer<PtWaveResult> restirResults;
StructuredBuffer<PtWaveGuide> restirGuides;
// The hybrid's G-buffer and the jittered inverse its depth was rendered with.
ConstantBuffer<float4x4> hybridInverseViewProjection;
Texture2D<float4> hybridNormalRoughness;
Texture2D<float4> hybridBaseMetallic;
Texture2D<float4> hybridGeometricDepth;
Texture2D<uint4> hybridIdentity;

// The rasteriser's surface at the pixel, where it can stand in for the traced primary hit.
PtPrimary primaryFromGBuffer(uint2 id) {
  PtPrimary primary;
  const float4 geometric = hybridGeometricDepth[id];
  const float depth = geometric.w;
  const uint4 identity = hybridIdentity[id];
  // Only opaque, non-emissive coverage (class 1) comes from the G-buffer; blended, masked and
  // emissive pixels trace their primary ray.
  primary.available = depth > 0.0f && identity.w == 1u;
  primary.surface.position = float3(0.0f);
  primary.surface.geometricNormal = float3(0.0f, 1.0f, 0.0f);
  primary.surface.normal = primary.surface.geometricNormal;
  primary.surface.baseColor = float3(0.0f);
  primary.surface.emissive = float3(0.0f);
  primary.surface.metallic = 0.0f;
  primary.surface.roughness = 1.0f;
  primary.surface.clearLayers(1.5f, 1.0f);
  // The raster G-buffer has no transmission or clearcoat layer (the rasteriser does not render
  // them) and no barycentrics to sample their textures: such pixels trace their primary ray, as
  // blended coverage does.
  float4 transmission = float4(0.0f, 1.5f, 1.0f, 0.0f);
  if (primary.available) {
    const Material material = scene.geometry.materials[identity.y];
    transmission = material.transmission;
    if (transmission.x > 0.0f || material.clearcoat.x > 0.0f) primary.available = false;
  }
  primary.hit = PtHit(0.0f);
  if (primary.available) {
    const float2 uv = (float2(id) + 0.5f) / uniforms.image.xy;
    const float4 clip = float4(uv.x * 2.0f - 1.0f, (1.0f - uv.y) * 2.0f - 1.0f, depth, 1.0f);
    const float4 world = mul(hybridInverseViewProjection, clip);
    const float4 normal = hybridNormalRoughness[id];
    const float4 base = hybridBaseMetallic[id];
    primary.surface.position = world.xyz / world.w;
    primary.surface.geometricNormal = normalize(geometric.xyz);
    primary.surface.normal = normalize(normal.xyz);
    primary.surface.baseColor = base.xyz;
    primary.surface.emissive = float3(0.0f);
    primary.surface.metallic = base.w;
    primary.surface.roughness = normal.w;
    // The raster faces every primary towards the eye, so the ray enters from the front.
    primary.surface.clearLayers(ptMaterialIor(transmission), 1.0f);
    primary.hit.t = length(primary.surface.position - uniforms.cameraPosition.xyz);
    primary.hit.instance = identity.x;
    primary.hit.primitive = identity.z;
    primary.hit.found = 1u;
  }
  return primary;
}

// The megakernels keep the last sample's guide, each part written to the pixel's record as the
// path forms it.
struct ReconstructionGuides : IPtGuideSink {
  uint pixel;
  bool enabled;

  bool collecting() { return enabled; }
  [mutating] void primary(PtReconstructionSample guide) { reconstructionSamples[pixel] = guide; }
  [mutating] void secondaryDistance(float distance) { reconstructionSamples[pixel].viewDistance.w = distance; }
  [mutating] void radiance(float3 diffuse, float3 specular) {
    reconstructionSamples[pixel].diffuse = float4(diffuse, 0.0f);
    reconstructionSamples[pixel].specular = float4(specular, 0.0f);
  }
};

// One pixel's samples. GBUFFER: the hybrid (primary surfaces from the G-buffer, no albedo or
// normal accumulation, no ReSTIR). COSTS: the tracer counts its work, so the BVH cost views
// (image.z 2 and 3) show it.
void tracePixel<T : IPtTracer, let EMISSIVE : bool, let GBUFFER : bool, let COSTS : bool>(T tracer, uint2 id) {
  if (float(id.x) >= uniforms.image.x || float(id.y) >= uniforms.image.y) return;
  const uint firstSample = uniforms.counts.w;
  // Zero samples only rewrites the mean: the frame's lit image still needs it.
  const uint samples = uint(uniforms.image.w);
  const uint pixel = id.y * uint(uniforms.image.x) + id.x;

  PtPrimary primary;
  if (GBUFFER) {
    primary = primaryFromGBuffer(id);
  } else {
    primary.available = false;
    primary.surface = {};
    primary.hit = PtHit(0.0f);
  }
  // Read only when ReSTIR DI runs: otherwise the buffers are one-element placeholders.
  float3 restirDirect = float3(0.0f), restirDiffuse = float3(0.0f);
  if (!GBUFFER && uniforms.estimator.x == 1u) {
    restirDirect = restirResults[pixel].radiance.xyz;
    restirDiffuse = restirGuides[pixel].diffuseRadiance.xyz;
  }

  float3 sum = float3(0.0f);
  float3 albedoSum = float3(0.0f);
  float3 normalSum = float3(0.0f);
  for (uint s = 0u; s < samples; ++s) {
    // One coherent, genuinely fresh path per reconstruction update: the last. Multi-SPP batches
    // still use every path in the independent raw accumulator. Mode 1 only: modes 2 and 3 are BVH
    // cost views, whose guide binding is one element.
    ReconstructionGuides guides;
    guides.pixel = pixel;
    guides.enabled = ptCollectsGuides(uniforms) && s + 1u == samples;
    PtPath path = ptTracePath<T, GpuMaps, ReconstructionGuides, EMISSIVE, !GBUFFER, GBUFFER>(
        tracer, scene, uniforms, id.x, id.y, firstSample + s, primary, restirDirect, restirDiffuse, guides);
    if (COSTS && uniforms.image.z > 1.5f) {
      const float cost = uniforms.image.z < 2.5f ? float(path.traversalCosts.x) / 128.0f : float(path.traversalCosts.y) / 64.0f;
      // A fixed false-colour scale makes captures comparable between runs: blue is no work,
      // green half the scale, red at or above it.
      const float value = saturate(cost);
      path.radiance = float3(saturate(value * 3.0f - 1.5f), saturate(1.0f - abs(value * 3.0f - 1.5f)), saturate(1.5f - value * 3.0f));
      path.albedo = float3(0.0f);
      path.normal = float3(0.0f);
    }
    // A non-finite path gets no weight rather than poisoning the pixel for good.
    if (!(any(isnan(path.radiance)) || any(isinf(path.radiance)))) sum = sum + path.radiance;
    albedoSum = albedoSum + path.albedo;
    normalSum = normalSum + path.normal;
  }

  float4 total = float4(sum, float(samples));
  float4 albedoTotal = float4(albedoSum, 0.0f);
  float4 normalTotal = float4(normalSum, 0.0f);
  if (firstSample > 0u) {
    total = total + accumulation[id];
    if (!GBUFFER) {
      albedoTotal = albedoTotal + albedoAccumulation[id];
      normalTotal = normalTotal + normalAccumulation[id];
    }
  }
  if (samples > 0u) {
    accumulation[id] = total;
    if (!GBUFFER) {
      albedoAccumulation[id] = albedoTotal;
      normalAccumulation[id] = normalTotal;
    }
  }
  output.Store(id, float4(total.xyz / max(total.w, 1.0f), 1.0f));
}

BinaryBvhTracer<GpuMaps> binaryTracer() {
  BinaryBvhTracer<GpuMaps> tracer;
  tracer.nodes = structure.bvhNodes;
  tracer.triangles = structure.bvhTriangles;
  tracer.geometry = scene.geometry;
  tracer.maps = scene.maps;
  return tracer;
}

WideBvhTracer<GpuMaps, STACK, false> wideTracer<let STACK : int>() {
  WideBvhTracer<GpuMaps, STACK, false> tracer;
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

[shader("compute")]
[numthreads(8, 8, 1)]
void path_trace(uint3 id: SV_DispatchThreadID) { tracePixel<BinaryBvhTracer<GpuMaps>, false, false, true>(binaryTracer(), id.xy); }

[shader("compute")]
[numthreads(8, 8, 1)]
void path_trace_emissive(uint3 id: SV_DispatchThreadID) {
  tracePixel<BinaryBvhTracer<GpuMaps>, true, false, true>(binaryTracer(), id.xy);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void path_trace_wide(uint3 id: SV_DispatchThreadID) {
  tracePixel<WideBvhTracer<GpuMaps, kWideStack, false>, false, false, true>(wideTracer<kWideStack>(), id.xy);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void path_trace_wide_emissive(uint3 id: SV_DispatchThreadID) {
  tracePixel<WideBvhTracer<GpuMaps, kWideStack, false>, true, false, true>(wideTracer<kWideStack>(), id.xy);
}

// The deep traversal stack, for trees whose stack bound exceeds 64.
[shader("compute")]
[numthreads(8, 8, 1)]
void path_trace_wide_deep(uint3 id: SV_DispatchThreadID) {
  tracePixel<WideBvhTracer<GpuMaps, kWideStackDeep, false>, false, false, true>(wideTracer<kWideStackDeep>(), id.xy);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void path_trace_wide_emissive_deep(uint3 id: SV_DispatchThreadID) {
  tracePixel<WideBvhTracer<GpuMaps, kWideStackDeep, false>, true, false, true>(wideTracer<kWideStackDeep>(), id.xy);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void path_trace_rt(uint3 id: SV_DispatchThreadID) { tracePixel<RayQueryTracer<GpuMaps>, true, false, false>(rayQueryTracer(), id.xy); }

[shader("compute")]
[numthreads(8, 8, 1)]
void path_trace_hybrid_rt(uint3 id: SV_DispatchThreadID) {
  tracePixel<RayQueryTracer<GpuMaps>, true, true, false>(rayQueryTracer(), id.xy);
}

// The full ray pipeline: the same pixel function, its rays traced by TraceRay against the
// structure; the any-hit stage runs the alpha test every intersector uses.
struct PipelinePayload {
  uint ambiguous;
  float t;
  float2 barycentric;
  uint instance;
  uint primitive;
  uint found;
  uint seed;
  float2 cone; // the ray cone where the ray left, for alpha tests
};

struct RayPipelineTracer : IPtTracer {
  RaytracingAccelerationStructure scene;

  PtHit trace(float3 origin, float3 direction, float tMax, uint mask, uint seed, float2 cone, bool anyHit) {
    PipelinePayload payload = {};
    payload.seed = seed;
    payload.cone = cone;
    uint flags = kPtRayFlags;
    // An occlusion ray ends at the first solid hit and runs no closest-hit stage: found stays set
    // unless the miss stage clears it.
    if (anyHit) {
      flags = flags | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
      payload.found = 1u;
    }
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0f;
    ray.TMax = tMax;
    TraceRay(scene, flags, mask, 0u, 1u, 0u, ray, payload);
    PtHit hit = PtHit(payload.t);
    hit.ambiguous = payload.ambiguous;
    hit.barycentric = payload.barycentric;
    hit.instance = payload.instance;
    hit.primitive = payload.primitive;
    hit.found = payload.found;
    return hit;
  }
};

[shader("raygeneration")]
void path_trace_pipeline() {
  RayPipelineTracer tracer;
  tracer.scene = structure.accelerationStructure;
  tracePixel<RayPipelineTracer, true, false, false>(tracer, DispatchRaysIndex().xy);
}

[shader("miss")]
void path_pipeline_miss(inout PipelinePayload payload) { payload.found = 0u; }

[shader("closesthit")]
void path_pipeline_closest(inout PipelinePayload payload, BuiltInTriangleIntersectionAttributes attributes) {
  payload.t = RayTCurrent();
  payload.barycentric = attributes.barycentrics;
  payload.instance = InstanceIndex();
  payload.primitive = PrimitiveIndex();
  payload.found = 1u;
}

[shader("anyhit")]
void path_pipeline_alpha(inout PipelinePayload payload, BuiltInTriangleIntersectionAttributes attributes) {
  const uint instance = InstanceIndex();
  if ((scene.geometry.traceInstances[instance].flags & kInstanceBlended) != 0u) payload.ambiguous = 1u;
  const uint front = HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE ? 1u : 0u;
  if (!ptCandidateSolid(scene.geometry, scene.maps, instance, PrimitiveIndex(), attributes.barycentrics, front, payload.seed,
                        WorldRayDirection(), ptConeWidthOrLevelZero(payload.cone, RayTCurrent())))
    IgnoreHit();
}
