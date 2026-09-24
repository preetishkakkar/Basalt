// The path loop: one camera path, adding its radiance to pathRadiance. Included as
// statements inside a kernel body (MSL) or a function (C++), because msl2spirv traverses an
// acceleration structure only in the entry itself; what differs per backend comes in
// through one macro, expanded exactly once so the kernel inlines one traversal:
//   PT_TRACE(origin, direction, tMax, mask, seed, cone, anyHit, hit)   fills the PtHit hit
// (cone: the ray cone where the ray leaves, for alpha tests)
// In scope: uniforms (PathUniforms), the PT_SCENE_ARGS buffers, lights,
// environmentDistribution, specularAlbedo, emissiveTriangles, the PT_TEXTURE_ARGS and PT_ENVIRONMENT_ARGS,
// pixelX, pixelY, sampleIndex, and float3 pathRadiance, pathAlbedo and pathNormal: the last
// two receive the first hit's base colour and shading normal, the denoiser's guides (for a
// ray that escapes, the environment's colour clamped to one, and no normal).
// Also in scope: pathGuide (PtReconstructionSample), collected when uniforms.image.z is
// enabled. These extra display signals never feed pathRadiance or change random indices.
// With ReSTIR DI (uniforms.estimator.x == 1) the includer defines
// PT_RESTIR_DIRECT and PT_RESTIR_DIFFUSE, the primary vertex's visible direct light from its
// reservoir and that light's diffuse part; they replace the primary vertex's light sample.
//
// Unidirectional with next-event estimation. Each vertex samples one light, chosen
// uniformly among the kinds the scene has: the environment by its distribution, the
// analytic sun uniformly within its cone, or one punctual light chosen uniformly. The
// environment and the sun are combined with BSDF sampling by the power heuristic, their
// light pdf including the light-kind choice probability. Each step of the loop traces either the vertex's shadow
// ray or the path's next ray. Random numbers come four at a time (pathRandom4): per bounce,
// group 0 is the light direction (x, y), the kind (z) and the punctual light (w); group 1
// the BSDF sample (x, y, z) and Russian roulette (w).
{
  const uint pathSeedValue = pathSeed(pixelX, pixelY, sampleIndex, uniforms.counts.z);
  const float4 jitter = pathRandom4(pathSeedValue, 0u, 0u);
  const float ndcX = (float(pixelX) + jitter.x) / uniforms.image.x * 2.0f - 1.0f;
  const float ndcY = 1.0f - (float(pixelY) + jitter.y) / uniforms.image.y * 2.0f;
  float3 nextOrigin = xyz(uniforms.cameraPosition);
  float3 nextDirection = normalize(xyz(uniforms.cameraForward) + xyz(uniforms.cameraRight) * ndcX +
                                   xyz(uniforms.cameraUp) * ndcY);
  ptApplyThinLens(uniforms.lens, xyz(uniforms.cameraForward), xyz(uniforms.cameraRight), xyz(uniforms.cameraUp),
                  float2(jitter.z, jitter.w), nextOrigin, nextDirection);
#ifdef PT_PRIMARY_GBUFFER
  if (primaryAvailable) nextDirection = normalize(primarySurface.position - nextOrigin);
#endif
  float3 throughput = float3(1.0f);
  const bool collectGuides = uniforms.image.z > 0.5f && uniforms.image.z < 1.5f;
  float3 diffuseFraction = float3(1.0f);
  float3 shadowDiffuseFraction = float3(1.0f);
  float3 diffuseRadiance = float3(0.0f);
  float previousPdf = 0.0f;
  // The ray cone: width where the current ray left and spread; level zero keeps
  // the sentinel (ptCameraCone).
  float2 pathCone = ptCameraCone(uniforms.lens);
  float2 shadowCone = pathCone;
  float3 shadowOrigin = float3(0.0f);
  float3 shadowDirection = float3(0.0f, 1.0f, 0.0f);
  float shadowReach = 0.0f;
  float3 shadowContribution = float3(0.0f);
  bool shadowPending = false;
  bool continuing = true;
  uint bounce = 0u;

  const uint maxBounces = uint(uniforms.path.x);
  uint kindCount = 0u;
  if (uint(uniforms.path.w) != 1u && uniforms.distribution.w > 0.5f) kindCount = kindCount + 1u;
  if (uint(uniforms.path.w) != 1u && uniforms.sunDirection.w > 0.0f) kindCount = kindCount + 1u;
  if (uint(uniforms.path.w) != 1u && uniforms.counts.x > 0u) kindCount = kindCount + 1u;
#ifndef PT_DISABLE_EMISSIVE
  if (uint(uniforms.path.w) != 1u && uniforms.emissive.x > 0u) kindCount = kindCount + 1u;
#endif

  for (uint step = 0u; step < 2u * maxBounces + 2u && (continuing || shadowPending); ++step) {
    float3 traceOrigin = nextOrigin;
    float3 traceDirection = nextDirection;
    float traceReach = kPtInfinity;
    uint traceAny = 0u;
    float2 traceCone = pathCone;
    if (shadowPending) {
      traceCone = shadowCone;
      traceOrigin = shadowOrigin;
      traceDirection = shadowDirection;
      traceReach = shadowReach;
      traceAny = 1u;
    }
    PtHit hit;
    hit.ambiguous = 0u;
    hit.t = 0.0f;
    hit.barycentric = float2(0.0f);
    hit.instance = 0u;
    hit.primitive = 0u;
    hit.found = 0u;
    hit.nodeVisits = 0u;
    hit.triangleTests = 0u;
#ifdef PT_PRIMARY_GBUFFER
    if (primaryAvailable && bounce == 0u && step == 0u && !shadowPending) hit = primaryHit;
    else
#endif
      PT_TRACE(traceOrigin, traceDirection, traceReach, uniforms.counts.y,
               ptTraceSeed(pathSeedValue, bounce, traceAny), traceCone, traceAny, hit);

    if (shadowPending) {
      if (hit.found == 0u) pathRadiance = pathRadiance + shadowContribution;
      if (collectGuides && hit.found == 0u)
        diffuseRadiance = diffuseRadiance + shadowContribution * shadowDiffuseFraction;
      shadowPending = false;
    } else if (hit.found == 0u) {
      // Escaped. A camera ray sees the environment only when it is drawn; a bounce always does.
      if (bounce == 0u && uniforms.environment.w > 0.5f)
        pathAlbedo = saturate(ptSampleEnvironment(PT_ENVIRONMENT_ARGS, equirectangularUV(nextDirection)) *
                              uniforms.environment.x);
      if (bounce > 0u || uniforms.environment.w > 0.5f) {
        float environmentWeight = 1.0f;
        float sunWeight = 1.0f;
        if (bounce > 0u && uint(uniforms.path.w) == 2u) {
          environmentWeight = 0.0f;
          sunWeight = 0.0f;
        }
        if (bounce > 0u && uint(uniforms.path.w) == 0u && uniforms.distribution.w > 0.5f)
          environmentWeight = ptPowerHeuristic(
              previousPdf, (1.0f / max(float(kindCount), 1.0f)) * ptEnvironmentPdf(environmentDistribution, uniforms.distribution, nextDirection));
        if (bounce > 0u && uint(uniforms.path.w) == 0u && uniforms.sunDirection.w > 0.0f)
          sunWeight = ptPowerHeuristic(previousPdf, (1.0f / max(float(kindCount), 1.0f)) * (1.0f / max(uniforms.sunRadiance.w, 1e-12f)));
        float3 escaped = ptSampleEnvironment(PT_ENVIRONMENT_ARGS, equirectangularUV(nextDirection)) *
                         (uniforms.environment.x * environmentWeight);
        if (uniforms.sunDirection.w > 0.0f && dot(nextDirection, xyz(uniforms.sunDirection)) >= uniforms.sunDirection.w)
          escaped = escaped + xyz(uniforms.sunRadiance) * sunWeight;
        escaped = throughput * escaped;
        if (bounce > 0u) escaped = ptClampContribution(escaped, uniforms.path.z);
        pathRadiance = pathRadiance + escaped;
        if (collectGuides)
          diffuseRadiance = diffuseRadiance + escaped * diffuseFraction;
      }
      continuing = false;
    } else {
      const float hitWidth = ptConeWidthOrLevelZero(pathCone, hit.t);
#ifdef PT_PRIMARY_GBUFFER
      PtSurface surface;
      if (primaryAvailable && bounce == 0u && step == 0u) surface = primarySurface;
      else surface = ptSurfaceAt(PT_SCENE_ARGS, PT_TEXTURE_ARGS, hit, nextDirection, hitWidth);
#else
      const PtSurface surface = ptSurfaceAt(PT_SCENE_ARGS, PT_TEXTURE_ARGS, hit, nextDirection, hitWidth);
#endif
      if (bounce == 0u) {
        pathAlbedo = surface.baseColor;
        pathNormal = surface.normal;
        if (collectGuides) {
          const uint materialId = traceInstances[hit.instance].material;
          const uint coverage = uint(materials[materialId].alpha.y);
          pathGuide.positionDepth = float4(surface.position,
              dot(surface.position - xyz(uniforms.cameraPosition), xyz(uniforms.cameraForward)));
          pathGuide.geometricNormal = float4(surface.geometricNormal, ptGuideLayers(surface.transmission, surface.clearcoat));
          pathGuide.normalRoughness = float4(surface.normal, surface.roughness);
          pathGuide.albedo = float4(surface.baseColor, 0.0f);
          pathGuide.viewDistance = float4(nextDirection, -1.0f);
          pathGuide.identity = uint4(hit.instance, materialId, hit.ambiguous != 0u || coverage == 2u ? 2u : coverage == 1u ? 3u : 1u,
                                     hit.primitive);
        }
      }
      if (collectGuides && bounce == 1u) pathGuide.viewDistance.w = hit.t;
#ifndef PT_DISABLE_EMISSIVE
      if (bounce == 0u || uint(uniforms.path.w) == 1u) previousPdf = 1.0f;
      else if (uint(uniforms.path.w) == 2u) previousPdf = 0.0f;
      else if (bounce > 0u && uint(uniforms.path.w) == 0u && uniforms.emissive.x > 0u)
        previousPdf = ptPowerHeuristic(previousPdf,
            (1.0f / max(float(kindCount), 1.0f)) * ptEmissivePdf(emissiveTriangles, uniforms.emissive.x, traceInstances, hit, nextDirection));
      else previousPdf = 1.0f;
      pathRadiance = pathRadiance + throughput * surface.emissive * previousPdf;
      if (collectGuides)
        diffuseRadiance = diffuseRadiance + throughput * surface.emissive * diffuseFraction * previousPdf;
#else
      pathRadiance = pathRadiance + throughput * surface.emissive;
      if (collectGuides)
        diffuseRadiance = diffuseRadiance + throughput * surface.emissive * diffuseFraction;
#endif

      if (bounce == maxBounces) {
        continuing = false;
      } else {
        const uint dimensions = bounce + 1u;
        const PtBsdf bsdf = ptMakeBsdf(surface, -nextDirection, specularAlbedo);
        const float3 origin = ptOffsetRay(surface.position, surface.geometricNormal);

        const float4 lightRandom = pathRandom4(pathSeedValue, dimensions, 0u);
        const float4 bsdfRandom = pathRandom4(pathSeedValue, dimensions, 1u);

        // One light sample: which kind, then a direction, its pdf and what arrives along it.
        uint kind = 4u;
        bool sampleLight = kindCount > 0u;
#ifdef PT_RESTIR_DIRECT
        if (bounce == 0u && uniforms.estimator.x == 1u) {
          sampleLight = false;
          pathRadiance = pathRadiance + PT_RESTIR_DIRECT;
          if (collectGuides) diffuseRadiance = diffuseRadiance + PT_RESTIR_DIFFUSE;
        }
#endif
        if (sampleLight) {
          uint slot = min(uint(lightRandom.z * float(kindCount)), kindCount - 1u);
          if (uint(uniforms.path.w) != 1u && uniforms.distribution.w > 0.5f) {
            if (slot == 0u) kind = 0u;
            slot = slot - 1u;
          }
          if (uint(uniforms.path.w) != 1u && uniforms.sunDirection.w > 0.0f && kind == 4u) {
            if (slot == 0u) kind = 1u;
            slot = slot - 1u;
          }
          if (uint(uniforms.path.w) != 1u && uniforms.counts.x > 0u && kind == 4u) {
            if (slot == 0u) kind = 2u;
            slot = slot - 1u;
          }
#ifndef PT_DISABLE_EMISSIVE
          if (uint(uniforms.path.w) != 1u && uniforms.emissive.x > 0u && kind == 4u) kind = 3u;
#endif
        }
        float3 lightDirection = float3(0.0f, 1.0f, 0.0f);
        float lightPdf = 0.0f;  // zero for a delta light
        float reach = kPtInfinity;
        float3 arriving = float3(0.0f);
        if (kind == 0u) {
          lightDirection = ptSampleEnvironmentDirection(environmentDistribution, uniforms.distribution,
                                                        xy(lightRandom), lightPdf);
          lightPdf = lightPdf * (1.0f / max(float(kindCount), 1.0f));
          arriving = ptSampleEnvironment(PT_ENVIRONMENT_ARGS, equirectangularUV(lightDirection)) * uniforms.environment.x;
        } else if (kind == 1u) {
          lightDirection = ptSampleCone(xyz(uniforms.sunDirection), uniforms.sunDirection.w, xy(lightRandom));
          lightPdf = (1.0f / max(uniforms.sunRadiance.w, 1e-12f)) * (1.0f / max(float(kindCount), 1.0f));
          arriving = xyz(uniforms.sunRadiance);
        } else if (kind == 2u) {
          const uint pick = min(uint(lightRandom.w * float(uniforms.counts.x)), uniforms.counts.x - 1u);
          float distance = 0.0f;
          arriving = ptPunctualLight(lights[pick], surface.position, lightDirection, distance) *
                     (float(uniforms.counts.x) * float(kindCount));
          reach = distance * 0.999f;
#ifndef PT_DISABLE_EMISSIVE
        } else if (kind == 3u) {
          arriving = ptSampleEmissiveTriangle(emissiveTriangles, uniforms.emissive.x, traceInstances, materials,
              PT_TEXTURE_ARGS, surface.position, float3(lightRandom.w, lightRandom.x, lightRandom.y),
              // The emitter's texture at the cone a BSDF-sampled path would carry there (the
              // sampled lobe's widened spread), so both MIS strategies read the same texture.
              ptConeScattered(pathCone, hitWidth, ptBsdfSampledAlpha(bsdf, bsdfRandom.z)), lightDirection, reach,
              lightPdf);
          lightPdf = lightPdf * (1.0f / max(float(kindCount), 1.0f));
          reach = reach * 0.9999f;
#endif
        }

        // The BSDF at the light's direction and at the one it samples for the path, evaluated
        // at one site so the kernel inlines one evaluation.
        const float3 sampledDirection = ptBsdfSampleDirection(bsdf, xyz(bsdfRandom));
        float3 lightValue = float3(0.0f);
        float lightBsdfPdf = 0.0f;
        float3 sampledValue = float3(0.0f);
        float sampledPdf = 0.0f;
        for (uint which = 0u; which < 2u; ++which) {
          float3 direction = sampledDirection;
          if (which == 0u) direction = lightDirection;
          float pdf = 0.0f;
          const float3 value = ptBsdfEvaluate(bsdf, surface.geometricNormal, direction, pdf);
          if (which == 0u) {
            lightValue = value;
            lightBsdfPdf = pdf;
          } else {
            sampledValue = value;
            sampledPdf = pdf;
          }
        }

        if (kind != 4u) {
          shadowContribution = throughput * lightValue * arriving;
          if (kind != 2u) {
            float weight = 0.0f;
            if (lightPdf > 0.0f) {
              weight = 1.0f / lightPdf;
              if (uint(uniforms.path.w) == 0u) weight = ptPowerHeuristic(lightPdf, lightBsdfPdf) / lightPdf;
            }
            shadowContribution = shadowContribution * weight;
          }
          if (ptMaxComponent(shadowContribution) > 0.0f) {
            shadowPending = true;
            shadowOrigin = origin;
            // A transmitted light sample leaves from the far side of the surface.
            if (dot(lightDirection, surface.geometricNormal) < 0.0f)
              shadowOrigin = ptOffsetRay(surface.position, -surface.geometricNormal);
            shadowDirection = lightDirection;
            shadowCone = ptConeShadow(pathCone, hitWidth);
            shadowReach = reach;
            shadowContribution = ptClampContribution(shadowContribution, uniforms.path.z);
            if (collectGuides) {
              shadowDiffuseFraction = diffuseFraction;
              if (bounce == 0u) shadowDiffuseFraction = ptDiffuseFraction(bsdf, lightDirection, lightValue);
            }
          }
        }

        continuing = sampledPdf > 0.0f && ptMaxComponent(sampledValue) > 0.0f;
        if (continuing) {
          if (collectGuides && bounce == 0u)
            diffuseFraction = ptDiffuseFraction(bsdf, sampledDirection, sampledValue);
          throughput = throughput * (sampledValue / sampledPdf);
          previousPdf = sampledPdf;
          nextOrigin = origin;
          if (dot(sampledDirection, surface.geometricNormal) < 0.0f)
            nextOrigin = ptOffsetRay(surface.position, -surface.geometricNormal);
          nextDirection = sampledDirection;
          pathCone = ptConeScattered(pathCone, hitWidth, ptBsdfSampledAlpha(bsdf, bsdfRandom.z));
          bounce = bounce + 1u;
          if (bounce >= uint(uniforms.path.y)) {
            const float survive = min(ptMaxComponent(throughput), 0.95f);
            if (bsdfRandom.w >= survive) continuing = false;
            else throughput = throughput / survive;
          }
        }
      }
    }
  }
  if (collectGuides) {
    pathGuide.diffuse = float4(diffuseRadiance, 0.0f);
    pathGuide.specular = float4(max(pathRadiance - diffuseRadiance, float3(0.0f)), 0.0f);
  }
}
