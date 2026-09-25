// The path loop: one camera path's radiance, for every backend. The tracer and the maps come in
// as generic parameters, so each backend instantiates the same loop around its own intersector:
// the software BVHs and ray queries on the GPU, the software BVH or the host's intersectors on
// the CPU.
//
// Unidirectional with next-event estimation. Each vertex samples one light, chosen uniformly
// among the kinds the scene has: the environment by its distribution, the analytic sun
// uniformly within its cone, one punctual light chosen uniformly, or an emissive triangle by
// its power. The environment, the sun and the emitters are combined with BSDF sampling by the
// power heuristic, their light pdf including the light-kind choice probability. Each iteration
// of the loop is one vertex: its path ray, then the shadow ray of the light it sampled, traced
// before the next iteration so that nothing about it stays live across the next traversal.
// Random numbers come four at a time (pathRandom4): per bounce, group 0 is the
// light direction (x, y), the kind (z) and the punctual light (w); group 1 the BSDF sample
// (x, y, z) and Russian roulette (w).
module pt_integrator;

__exported import pt_path;
__exported import pt_raycone;
__exported import pt_bvh;
__exported import pt_surface;
__exported import pt_bsdf;
__exported import pt_lights;

// What a path reads besides the intersector.
public struct PtScene<M : IPtMaps> {
  PtGeometry geometry;
  M maps;
  StructuredBuffer<Light> lights;
  StructuredBuffer<float> environmentDistribution;
  StructuredBuffer<float> specularAlbedo;
  StructuredBuffer<PtEmissiveTriangle> emissiveTriangles;
};

// Where a path's reconstruction guide goes as it forms: the first hit's surface, the secondary
// hit's distance, then the diffuse/specular split of the finished path. A sink that keeps the
// value hands it back with the path; a GPU megakernel writes each part to the pixel's record
// when it is known, so the guide is not held across every later trace (a ray pipeline saves
// what the raygen holds across TraceRay to memory, a compute kernel keeps it in registers).
public interface IPtGuideSink {
  // Whether this path's guide is collected at all; the radiance and the random numbers do not
  // depend on it.
  bool collecting();
  // The first hit's guide, or PtReconstructionSample.empty() when the camera ray escapes; its
  // secondary distance, diffuse and specular are still the empty record's.
  [mutating] void primary(PtReconstructionSample guide);
  [mutating] void secondaryDistance(float distance);
  [mutating] void radiance(float3 diffuse, float3 specular);
}

// The guide as a value, for callers that return it with the path.
public struct PtGuideRecord : IPtGuideSink {
  bool enabled;
  PtReconstructionSample guide;

  __init(bool enabled) {
    this.enabled = enabled;
    guide = PtReconstructionSample.empty();
  }
  bool collecting() { return enabled; }
  [mutating] void primary(PtReconstructionSample value) { guide = value; }
  [mutating] void secondaryDistance(float distance) { guide.viewDistance.w = distance; }
  [mutating] void radiance(float3 diffuse, float3 specular) {
    guide.diffuse = float4(diffuse, 0.0f);
    guide.specular = float4(specular, 0.0f);
  }
}

// The reconstruction guide of a camera ray's first hit, before its secondary distance and radiance
// split are known; direction is the camera ray's.
public PtReconstructionSample ptPrimaryGuide(PtGeometry geometry, PathUniforms uniforms, PtSurface surface, PtHit hit,
                                             float3 direction) {
  const uint materialId = geometry.traceInstances[hit.instance].material;
  const uint coverage = uint(geometry.materials[materialId].alpha.y);
  PtReconstructionSample guide = PtReconstructionSample.empty();
  guide.positionDepth = float4(surface.position, dot(surface.position - uniforms.cameraPosition.xyz, uniforms.cameraForward.xyz));
  guide.geometricNormal = float4(surface.geometricNormal, ptGuideLayers(surface.transmission, surface.clearcoat));
  guide.normalRoughness = float4(surface.normal, surface.roughness);
  guide.albedo = float4(surface.baseColor, 0.0f);
  guide.viewDistance = float4(direction, -1.0f);
  guide.identity = uint4(hit.instance, materialId, hit.ambiguous != 0u || coverage == 2u ? 2u : coverage == 1u ? 3u : 1u,
                         hit.primitive);
  return guide;
}

// Whether the uniforms ask for reconstruction guides (image.z 1; 2 and 3 are the cost views).
public bool ptCollectsGuides(PathUniforms uniforms) { return uniforms.image.z > 0.5f && uniforms.image.z < 1.5f; }

// One path's outcome.
public __extern_cpp struct PtPath {
  __extern_cpp float3 radiance;
  // The first hit's base colour and shading normal, the denoiser's guides; for a ray that
  // escapes, the environment's colour clamped to one, and no normal.
  __extern_cpp float3 albedo;
  __extern_cpp float3 normal;
  // The reconstruction's guide, as the CPU export's PtGuideRecord kept it.
  __extern_cpp PtReconstructionSample guide;
  // Software BVH node visits and triangle tests over the path's rays (the cost views).
  __extern_cpp uint2 traversalCosts;
};

// The rasteriser's primary surface, when a hybrid kernel takes it from the G-buffer instead
// of tracing the camera ray.
public struct PtPrimary {
  bool available;
  PtSurface surface;
  PtHit hit;
};

// One camera path of pixel (pixelX, pixelY). EMISSIVE: emissive triangles are sampled as
// lights (else their emission is only hit). RESTIR: with uniforms.estimator.x == 1 the primary
// vertex's direct light is restirDirect (its diffuse part restirDiffuse), from the ReSTIR DI
// reservoir passes, instead of a light sample. GBUFFER: primary replaces the camera ray's hit
// where available. The reconstruction guide, when guides.collecting(), goes to guides; it never
// feeds the radiance or changes random indices.
public PtPath ptTracePath<T : IPtTracer, M : IPtMaps, G : IPtGuideSink, let EMISSIVE : bool, let RESTIR : bool,
                          let GBUFFER : bool>(T tracer, PtScene<M> scene, PathUniforms uniforms, uint pixelX, uint pixelY,
                                              uint sampleIndex, PtPrimary primary, float3 restirDirect, float3 restirDiffuse,
                                              inout G guides) {
  PtPath path;
  path.radiance = float3(0.0f);
  path.albedo = float3(0.0f);
  path.normal = float3(0.0f);
  path.guide = PtReconstructionSample.empty();
  path.traversalCosts = uint2(0u);
  const PtGeometry geometry = scene.geometry;
  PtEmitters emitters;
  emitters.triangles = scene.emissiveTriangles;
  emitters.count = uniforms.emissive.x;

  const uint pathSeedValue = pathSeed(pixelX, pixelY, sampleIndex, uniforms.counts.z);
  float3 nextOrigin;
  float3 nextDirection = ptCameraRay(uniforms, pixelX, pixelY, pathSeedValue, nextOrigin);
  if (GBUFFER && primary.available) nextDirection = normalize(primary.surface.position - nextOrigin);
  float3 throughput = float3(1.0f);
  const bool collectGuides = guides.collecting();
  float3 diffuseFraction = float3(1.0f);
  float3 diffuseRadiance = float3(0.0f);
  float previousPdf = 0.0f;
  // The ray cone: width where the current ray left and spread; level zero keeps the sentinel
  // (ptCameraCone).
  float2 pathCone = ptCameraCone(uniforms.lens);
  bool continuing = true;
  uint bounce = 0u;

  const uint strategy = uint(uniforms.path.w);
  const uint maxBounces = uint(uniforms.path.x);
  uint kindCount = 0u;
  if (strategy != 1u && uniforms.distribution.w > 0.5f) kindCount = kindCount + 1u;
  if (strategy != 1u && uniforms.sunDirection.w > 0.0f) kindCount = kindCount + 1u;
  if (strategy != 1u && uniforms.counts.x > 0u) kindCount = kindCount + 1u;
  if (EMISSIVE && strategy != 1u && uniforms.emissive.x > 0u) kindCount = kindCount + 1u;
  const float kindProbability = 1.0f / max(float(kindCount), 1.0f);

  while (continuing) {
    PtHit hit;
    if (GBUFFER && primary.available && bounce == 0u) {
      hit = primary.hit;
    } else {
      hit = tracer.trace(nextOrigin, nextDirection, kPtInfinity, uniforms.counts.y, ptTraceSeed(pathSeedValue, bounce, 0u), pathCone,
                         false);
      path.traversalCosts = path.traversalCosts + uint2(hit.nodeVisits, hit.triangleTests);
    }

    if (hit.found == 0u) {
      // Escaped. A camera ray sees the environment only when it is drawn; a bounce always does.
      if (collectGuides && bounce == 0u) guides.primary(PtReconstructionSample.empty());
      if (bounce == 0u && uniforms.environment.w > 0.5f)
        path.albedo = saturate(scene.maps.sampleEnvironment(equirectangularUV(nextDirection)) * uniforms.environment.x);
      if (bounce > 0u || uniforms.environment.w > 0.5f) {
        float environmentWeight = 1.0f;
        float sunWeight = 1.0f;
        if (bounce > 0u && strategy == 2u) {
          environmentWeight = 0.0f;
          sunWeight = 0.0f;
        }
        if (bounce > 0u && strategy == 0u && uniforms.distribution.w > 0.5f)
          environmentWeight = ptPowerHeuristic(
              previousPdf, kindProbability * ptEnvironmentPdf(scene.environmentDistribution, uniforms.distribution, nextDirection));
        if (bounce > 0u && strategy == 0u && uniforms.sunDirection.w > 0.0f)
          sunWeight = ptPowerHeuristic(previousPdf, kindProbability * (1.0f / max(uniforms.sunRadiance.w, 1e-12f)));
        float3 escaped =
            scene.maps.sampleEnvironment(equirectangularUV(nextDirection)) * (uniforms.environment.x * environmentWeight);
        if (uniforms.sunDirection.w > 0.0f && dot(nextDirection, uniforms.sunDirection.xyz) >= uniforms.sunDirection.w)
          escaped = escaped + uniforms.sunRadiance.xyz * sunWeight;
        escaped = throughput * escaped;
        if (bounce > 0u) escaped = ptClampContribution(escaped, uniforms.path.z);
        path.radiance = path.radiance + escaped;
        if (collectGuides) diffuseRadiance = diffuseRadiance + escaped * diffuseFraction;
      }
      continuing = false;
    } else {
      const float hitWidth = ptConeWidthOrLevelZero(pathCone, hit.t);
      PtSurface surface;
      if (GBUFFER && primary.available && bounce == 0u) surface = primary.surface;
      else surface = ptSurfaceAt(geometry, scene.maps, hit, nextDirection, hitWidth);
      if (bounce == 0u) {
        path.albedo = surface.baseColor;
        path.normal = surface.normal;
        if (collectGuides) guides.primary(ptPrimaryGuide(geometry, uniforms, surface, hit, nextDirection));
      }
      if (collectGuides && bounce == 1u) guides.secondaryDistance(hit.t);
      if (EMISSIVE) {
        // previousPdf becomes the emission's MIS weight here; it is the BSDF pdf again once the
        // path continues (below).
        if (bounce == 0u || strategy == 1u) previousPdf = 1.0f;
        else if (strategy == 2u) previousPdf = 0.0f;
        else if (bounce > 0u && strategy == 0u && uniforms.emissive.x > 0u)
          previousPdf = ptPowerHeuristic(previousPdf, kindProbability * emitters.pdf(geometry, hit, nextDirection));
        else previousPdf = 1.0f;
        path.radiance = path.radiance + throughput * surface.emissive * previousPdf;
        if (collectGuides) diffuseRadiance = diffuseRadiance + throughput * surface.emissive * diffuseFraction * previousPdf;
      } else {
        path.radiance = path.radiance + throughput * surface.emissive;
        if (collectGuides) diffuseRadiance = diffuseRadiance + throughput * surface.emissive * diffuseFraction;
      }

      if (bounce == maxBounces) {
        continuing = false;
      } else {
        const uint vertex = bounce; // bounce advances below when the path continues
        const uint dimensions = bounce + 1u;
        const PtBsdf bsdf = PtBsdf(surface, -nextDirection, scene.specularAlbedo);
        const float3 origin = ptOffsetRay(surface.position, surface.geometricNormal);

        const float4 lightRandom = pathRandom4(pathSeedValue, dimensions, 0u);
        const float4 bsdfRandom = pathRandom4(pathSeedValue, dimensions, 1u);

        // One light sample: which kind, then a direction, its pdf and what arrives along it.
        uint kind = 4u;
        bool sampleLight = kindCount > 0u;
        if (RESTIR && bounce == 0u && uniforms.estimator.x == 1u) {
          sampleLight = false;
          path.radiance = path.radiance + restirDirect;
          if (collectGuides) diffuseRadiance = diffuseRadiance + restirDiffuse;
        }
        if (sampleLight) {
          // slot counts down through the kinds present, in order environment, sun, punctual,
          // emissive; the first to reach zero is chosen (the later wrap below zero is harmless).
          uint slot = min(uint(lightRandom.z * float(kindCount)), kindCount - 1u);
          if (strategy != 1u && uniforms.distribution.w > 0.5f) {
            if (slot == 0u) kind = 0u;
            slot = slot - 1u;
          }
          if (strategy != 1u && uniforms.sunDirection.w > 0.0f && kind == 4u) {
            if (slot == 0u) kind = 1u;
            slot = slot - 1u;
          }
          if (strategy != 1u && uniforms.counts.x > 0u && kind == 4u) {
            if (slot == 0u) kind = 2u;
            slot = slot - 1u;
          }
          if (EMISSIVE && strategy != 1u && uniforms.emissive.x > 0u && kind == 4u) kind = 3u;
        }
        float3 lightDirection = float3(0.0f, 1.0f, 0.0f);
        float lightPdf = 0.0f; // zero for a delta light
        float reach = kPtInfinity;
        float3 arriving = float3(0.0f);
        if (kind == 0u) {
          lightDirection = ptSampleEnvironmentDirection(scene.environmentDistribution, uniforms.distribution, lightRandom.xy, lightPdf);
          lightPdf = lightPdf * kindProbability;
          arriving = scene.maps.sampleEnvironment(equirectangularUV(lightDirection)) * uniforms.environment.x;
        } else if (kind == 1u) {
          lightDirection = ptSampleCone(uniforms.sunDirection.xyz, uniforms.sunDirection.w, lightRandom.xy);
          lightPdf = (1.0f / max(uniforms.sunRadiance.w, 1e-12f)) * kindProbability;
          arriving = uniforms.sunRadiance.xyz;
        } else if (kind == 2u) {
          const uint pick = min(uint(lightRandom.w * float(uniforms.counts.x)), uniforms.counts.x - 1u);
          float distance;
          arriving = ptPunctualLight(scene.lights[pick], surface.position, lightDirection, distance) *
                     (float(uniforms.counts.x) * float(kindCount));
          reach = distance * 0.999f;
        } else if (EMISSIVE && kind == 3u) {
          // The emitter's texture at the cone a BSDF-sampled path would carry there (the sampled
          // lobe's widened spread), so both MIS strategies read the same texture.
          arriving = emitters.sample(geometry, scene.maps, surface.position, float3(lightRandom.w, lightRandom.x, lightRandom.y),
                                     ptConeScattered(pathCone, hitWidth, bsdf.sampledAlpha(bsdfRandom.z)), lightDirection, reach,
                                     lightPdf);
          lightPdf = lightPdf * kindProbability;
          reach = reach * 0.9999f;
        }

        // The BSDF at the light's direction and at the one it samples for the path, evaluated
        // at one site so the kernel inlines one evaluation.
        const float3 sampledDirection = bsdf.sampleDirection(bsdfRandom.xyz);
        float3 lightValue = float3(0.0f);
        float lightBsdfPdf = 0.0f;
        float3 sampledValue = float3(0.0f);
        float sampledPdf = 0.0f;
        for (uint which = 0u; which < 2u; ++which) {
          const float3 direction = which == 0u ? lightDirection : sampledDirection;
          float pdf;
          const float3 value = bsdf.evaluate(surface.geometricNormal, direction, pdf);
          if (which == 0u) {
            lightValue = value;
            lightBsdfPdf = pdf;
          } else {
            sampledValue = value;
            sampledPdf = pdf;
          }
        }

        bool shadowPending = false;
        float3 shadowContribution = float3(0.0f);
        float3 shadowOrigin = origin;
        float2 shadowCone = pathCone;
        float3 shadowDiffuseFraction = diffuseFraction;
        if (kind != 4u) {
          shadowContribution = throughput * lightValue * arriving;
          if (kind != 2u) {
            float weight = 0.0f;
            if (lightPdf > 0.0f) {
              weight = 1.0f / lightPdf;
              if (strategy == 0u) weight = ptPowerHeuristic(lightPdf, lightBsdfPdf) / lightPdf;
            }
            shadowContribution = shadowContribution * weight;
          }
          if (ptMaxComponent(shadowContribution) > 0.0f) {
            shadowPending = true;
            // A transmitted light sample leaves from the far side of the surface.
            if (dot(lightDirection, surface.geometricNormal) < 0.0f)
              shadowOrigin = ptOffsetRay(surface.position, -surface.geometricNormal);
            shadowCone = ptConeShadow(pathCone, hitWidth);
            shadowContribution = ptClampContribution(shadowContribution, uniforms.path.z);
            if (collectGuides && bounce == 0u) shadowDiffuseFraction = bsdf.diffuseFraction(lightDirection, lightValue);
          }
        }

        continuing = sampledPdf > 0.0f && ptMaxComponent(sampledValue) > 0.0f;
        if (continuing) {
          if (collectGuides && bounce == 0u) diffuseFraction = bsdf.diffuseFraction(sampledDirection, sampledValue);
          throughput = throughput * (sampledValue / sampledPdf);
          previousPdf = sampledPdf;
          nextOrigin = origin;
          if (dot(sampledDirection, surface.geometricNormal) < 0.0f) nextOrigin = ptOffsetRay(surface.position, -surface.geometricNormal);
          nextDirection = sampledDirection;
          pathCone = ptConeScattered(pathCone, hitWidth, bsdf.sampledAlpha(bsdfRandom.z));
          bounce = bounce + 1u;
          if (bounce >= uint(uniforms.path.y)) {
            const float survive = min(ptMaxComponent(throughput), 0.95f);
            if (bsdfRandom.w >= survive) continuing = false;
            else throughput = throughput / survive;
          }
        }

        // The shadow ray, after the path's next ray is set up. Its seed is this vertex's own
        // bounce, as the wavefront shadow stage's: the advanced index would repeat the next
        // vertex's shadow seed when that vertex ends, correlating their alpha decisions.
        if (shadowPending) {
          const PtHit blocker = tracer.trace(shadowOrigin, lightDirection, reach, uniforms.counts.y,
                                             ptTraceSeed(pathSeedValue, vertex, 1u), shadowCone, true);
          path.traversalCosts = path.traversalCosts + uint2(blocker.nodeVisits, blocker.triangleTests);
          if (blocker.found == 0u) {
            path.radiance = path.radiance + shadowContribution;
            if (collectGuides) diffuseRadiance = diffuseRadiance + shadowContribution * shadowDiffuseFraction;
          }
        }
      }
    }
  }
  if (collectGuides) guides.radiance(diffuseRadiance, max(path.radiance - diffuseRadiance, float3(0.0f)));
  return path;
}
