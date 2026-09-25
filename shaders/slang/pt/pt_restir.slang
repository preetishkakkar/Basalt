// ReSTIR DI at the primary vertex (Bitterli et al. 2020), shared by the CPU tracer and the GPU
// reservoir kernels. A light sample y is a light identity (the kind in the top two bits: 0
// environment, 1 sun, 2 punctual, 3 emissive triangle; the light or triangle index below) plus
// the two random numbers that place it on the light: the environment's and the sun's direction
// numbers, or an emissive triangle's barycentric numbers. Re-evaluating a sample at another
// receiver is therefore exact, and its source density is the light sampler's in a
// receiver-independent measure (solid angle of a distant light, area of a triangle, the
// discrete choice of a punctual light), so reuse needs no Jacobian.
module pt_restir;

__exported import pt_integrator;

public static const uint kPtRestirNoLight = 0xFFFFFFFFu;
public static const uint kPtRestirMaxCandidates = 13u;   // groups 2..14 of bounce 0; 15 is the stream
public static const float kPtRestirHistoryLimit = 20.0f; // temporal M is clamped to 20 M_initial
public static const uint kPtRestirSpatialNeighbours = 3u;
public static const float kPtRestirSpatialRadius = 16.0f;

// 32 bytes. light: kind << 30 | index, kPtRestirNoLight when empty; weight: the unbiased
// contribution weight W (0 when empty); target: p_hat of the sample at the owner.
public __extern_cpp struct PtReservoir {
  __extern_cpp uint light;
  __extern_cpp float paramX;
  __extern_cpp float paramY;
  __extern_cpp float weightSum;
  __extern_cpp float count;
  __extern_cpp float weight;
  __extern_cpp float target;
  __extern_cpp uint reserved;

  static PtReservoir empty() {
    PtReservoir r;
    r.light = kPtRestirNoLight;
    r.paramX = 0.0f;
    r.paramY = 0.0f;
    r.weightSum = 0.0f;
    r.count = 0.0f;
    r.weight = 0.0f;
    r.target = 0.0f;
    r.reserved = 0u;
    return r;
  }

  // Streaming weighted selection: adds a sample of resampling weight `weight`, keeping it with
  // probability weight / weightSum. The count is kept by the caller.
  [mutating]
  void stream(uint light, float2 params, float weight, float target, float random) {
    weightSum = weightSum + weight;
    if (weight > 0.0f && random * weightSum < weight) {
      this.light = light;
      paramX = params.x;
      paramY = params.y;
      this.target = target;
    }
  }

  // Adds a contributor's reservoir to a combination at the owner (the generalized balance
  // heuristic over the contributors, targets without visibility). Its sample's resampling
  // weight is m(y) p_hat_owner(y) W, with the MIS weight m(y) = M p_hat_contributor(y) /
  // sum_j M_j p_hat_j(y) over every contributor j (this one included); `balance` is that sum.
  // The count is added whether or not a sample is held.
  [mutating]
  void merge(PtReservoir incoming, float targetAtOwner, float balance, float random) {
    float resampling = 0.0f;
    if (incoming.light != kPtRestirNoLight && balance > 0.0f)
      resampling = incoming.count * incoming.target / balance * targetAtOwner * incoming.weight;
    stream(incoming.light, float2(incoming.paramX, incoming.paramY), resampling, targetAtOwner, random);
    count = count + incoming.count;
  }

  // A finished reservoir: W = weightSum / (normalisation p_hat_owner(y)), or empty with its
  // count kept.
  [mutating]
  void finish(float normalisation) {
    if (light != kPtRestirNoLight && target > 0.0f) {
      weight = weightSum / (normalisation * target);
    } else {
      const float kept = count;
      this = PtReservoir.empty();
      count = kept;
    }
  }
};

// The primary surface a reservoir belongs to, for re-evaluating targets at it and for the
// reuse validity tests. positionDepth.w: linear depth along the view axis, 0 without a
// surface; normalMaterial: shading normal and the material index (bits); view: unit direction
// to the eye and the ray cone's width at the hit (ptConeWidthOrLevelZero); hit: instance,
// primitive and the barycentrics' bits.
public __extern_cpp struct PtRestirSurface {
  __extern_cpp float4 positionDepth;
  __extern_cpp float4 normalMaterial;
  __extern_cpp float4 view;
  __extern_cpp uint4 hit;

  // The record of a primary hit: surface from ptSurfaceAt, the camera ray's direction, the
  // cone's width at the hit and the camera (for the linear depth).
  __init(PtSurface surface, PtHit primary, uint material, float3 rayDirection, float coneWidth, float3 cameraPosition,
                float3 cameraForward) {
    positionDepth = float4(surface.position, max(dot(surface.position - cameraPosition, cameraForward), 1e-6f));
    normalMaterial = float4(surface.normal, asfloat(material));
    view = float4(-rayDirection, coneWidth);
    hit = uint4(primary.instance, primary.primitive, asuint(primary.barycentric.x), asuint(primary.barycentric.y));
  }

  // No surface (a pixel whose primary ray escaped).
  static PtRestirSurface none() {
    PtRestirSurface s;
    s.positionDepth = float4(0.0f);
    s.normalMaterial = float4(0.0f);
    s.view = float4(0.0f);
    s.hit = uint4(0u);
    return s;
  }

  PtHit primaryHit() {
    PtHit result = PtHit(0.0f);
    result.barycentric = float2(asfloat(hit.z), asfloat(hit.w));
    result.instance = hit.x;
    result.primitive = hit.y;
    result.found = positionDepth.w > 0.0f ? 1u : 0u;
    return result;
  }

  // Whether a neighbour's surface may lend its reservoir: the same material, shading normals
  // within 0.9, relative depth within 0.1. depthAtNeighbour: this owner's depth seen as the
  // neighbour's (the reprojected depth for temporal reuse).
  bool similar(PtRestirSurface neighbour, float depthAtNeighbour) {
    if (positionDepth.w <= 0.0f || neighbour.positionDepth.w <= 0.0f) return false;
    if (asuint(normalMaterial.w) != asuint(neighbour.normalMaterial.w)) return false;
    if (dot(normalMaterial.xyz, neighbour.normalMaterial.xyz) < 0.9f) return false;
    return abs(neighbour.positionDepth.w - depthAtNeighbour) <= 0.1f * max(depthAtNeighbour, 1e-6f);
  }
};

// A camera for reprojection (the previous frame's) and the image it covers.
public __extern_cpp struct PtRestirCamera {
  __extern_cpp float4 position;
  __extern_cpp float4 forward;
  __extern_cpp float4 right;
  __extern_cpp float4 up;
  __extern_cpp uint4 image; // width, height, history valid, reuse bits
};

// The random number of decision k of a pixel's reuse at a frame: hash(pixel, frame, 0x5E57),
// with the run's seed mixed in so that seeds differ in their reuse as in their candidates.
public float ptRestirReuseRandom(uint pixel, uint frame, uint seed, uint k) {
  return hashFloat(pcgHash(pixel + pcgHash(frame + pcgHash(seed + 0x5E57u))) + k * 0x9E3779B9u);
}

// The stream's random number for its k-th decision: the reservoir stream group (M + 2 of
// bounce 0) hashed with the decision's index.
public float ptRestirStreamRandom(uint stream, uint k) { return hashFloat(stream + k * 0x9E3779B9u); }
public uint ptRestirStream(uint pathSeedValue, uint candidates) { return asuint(pathRandom4(pathSeedValue, 0u, 2u + candidates).x); }

// A spatial neighbour of a pixel: uniform in a disc of 16 pixels, from the reuse hash; the owner
// itself or a pixel outside the image is returned as width * height (none).
public uint ptRestirNeighbour(uint pixel, uint frame, uint seed, uint k, uint width, uint height) {
  const float angle = 2.0f * kPi * ptRestirReuseRandom(pixel, frame, seed, 2u + 2u * k);
  const float radius = kPtRestirSpatialRadius * sqrt(ptRestirReuseRandom(pixel, frame, seed, 3u + 2u * k));
  const float nx = floor(float(pixel % width) + 0.5f + radius * cos(angle));
  const float ny = floor(float(pixel / width) + 0.5f + radius * sin(angle));
  uint neighbour = width * height;
  if (nx >= 0.0f && ny >= 0.0f && nx < float(width) && ny < float(height)) neighbour = uint(ny) * width + uint(nx);
  if (neighbour == pixel) neighbour = width * height;
  return neighbour;
}

public float ptRestirTarget(float3 integrand) { return max(ptLuminance(integrand), 0.0f); }

// Everything the reservoir passes read: the scene and the uniforms' light and camera fields. The
// per-pixel records come in as any array (the kernels' buffers, read or written elsewhere in the
// same kernel file).
public struct PtRestir<M : IPtMaps> {
  PtScene<M> scene;
  PathUniforms uniforms;

  float kindShare() { return 1.0f / max(float(kindCount()), 1.0f); }

  // The light kinds present, as next-event estimation counts them.
  uint kindCount() {
    const uint strategy = uint(uniforms.path.w);
    uint count = 0u;
    if (strategy != 1u && uniforms.distribution.w > 0.5f) count = count + 1u;
    if (strategy != 1u && uniforms.sunDirection.w > 0.0f) count = count + 1u;
    if (strategy != 1u && uniforms.counts.x > 0u) count = count + 1u;
    if (strategy != 1u && uniforms.emissive.x > 0u) count = count + 1u;
    return count;
  }

  PtEmitters emitters() {
    PtEmitters result;
    result.triangles = scene.emissiveTriangles;
    result.count = uniforms.emissive.x;
    return result;
  }

  // A candidate from four random numbers, chosen as next-event estimation chooses its light: z
  // the kind, w the punctual light or emissive triangle; x, y place it (kept as the sample's
  // parameters). kPtRestirNoLight when the scene has no light.
  uint candidate(float4 u) {
    const uint kinds = kindCount();
    const uint strategy = uint(uniforms.path.w);
    uint light = kPtRestirNoLight;
    if (kinds > 0u) {
      uint slot = min(uint(u.z * float(kinds)), kinds - 1u);
      if (strategy != 1u && uniforms.distribution.w > 0.5f) {
        if (slot == 0u) light = 0u;
        slot = slot - 1u;
      }
      if (strategy != 1u && uniforms.sunDirection.w > 0.0f && light == kPtRestirNoLight) {
        if (slot == 0u) light = 1u << 30u;
        slot = slot - 1u;
      }
      if (strategy != 1u && uniforms.counts.x > 0u && light == kPtRestirNoLight) {
        if (slot == 0u) light = (2u << 30u) | min(uint(u.w * float(uniforms.counts.x)), uniforms.counts.x - 1u);
        slot = slot - 1u;
      }
      if (strategy != 1u && uniforms.emissive.x > 0u && light == kPtRestirNoLight) light = (3u << 30u) | emitters().select(u.w);
    }
    return light;
  }

  // A light sample seen from `position`: returns what arrives in the sample's own measure (the
  // radiance times |cos| / distance^2 for a triangle, the radiance of a distant light, a
  // punctual light's intensity over distance^2), with the unit direction towards it, the shadow
  // ray's extent, its source density in that measure (kind choice included) and, for MIS
  // against BSDF sampling, the solid-angle density next-event estimation would give it (0 for a
  // punctual light). cone: the shadow ray's cone, for emitter textures.
  float3 lightAt(uint light, float2 params, float3 position, float2 cone, out float3 direction, out float reach,
                        out float sourcePdf, out float solidAnglePdf) {
    const float share = kindShare();
    const uint kind = light >> 30u;
    const uint index = light & 0x3FFFFFFFu;
    direction = float3(0.0f, 1.0f, 0.0f);
    reach = kPtInfinity;
    sourcePdf = 0.0f;
    solidAnglePdf = 0.0f;
    float3 arriving = float3(0.0f);
    if (light == kPtRestirNoLight) {
      arriving = float3(0.0f);
    } else if (kind == 0u) {
      float pdf;
      direction = ptSampleEnvironmentDirection(scene.environmentDistribution, uniforms.distribution, params, pdf);
      sourcePdf = pdf * share;
      solidAnglePdf = sourcePdf;
      arriving = scene.maps.sampleEnvironment(equirectangularUV(direction)) * uniforms.environment.x;
    } else if (kind == 1u) {
      direction = ptSampleCone(uniforms.sunDirection.xyz, uniforms.sunDirection.w, params);
      sourcePdf = (1.0f / max(uniforms.sunRadiance.w, 1e-12f)) * share;
      solidAnglePdf = sourcePdf;
      arriving = uniforms.sunRadiance.xyz;
    } else if (kind == 2u) {
      float distance;
      arriving = ptPunctualLight(scene.lights[index], position, direction, distance);
      reach = distance * 0.999f;
      sourcePdf = (1.0f / float(uniforms.counts.x)) * share;
    } else {
      const PtEmissiveTriangle emitter = scene.emissiveTriangles[index];
      // Uniform on the triangle: the square root spreads samples by area.
      const float barycentricX = sqrt(params.x) * (1.0f - params.y);
      const float barycentricY = sqrt(params.x) * params.y;
      direction = emitter.v0Area.xyz + emitter.edge1Probability.xyz * barycentricX + emitter.edge2Cdf.xyz * barycentricY - position;
      reach = length(direction);
      if (reach > 0.0f) direction = direction / reach;
      float cosine = -dot(emitter.normal.xyz, direction);
      const TraceInstance info = scene.geometry.traceInstances[emitter.identity.x];
      if ((info.flags & kInstanceDoubleSided) != 0u) cosine = abs(cosine);
      if (reach > 1e-5f && cosine > 1e-6f) {
        const float area = emitter.v0Area.w;
        const float probability = emitter.edge1Probability.w;
        sourcePdf = probability / area * share;
        solidAnglePdf = probability * reach * reach / (area * cosine) * share;
        float lodBase = kPtLevelZero;
        if (cone.x >= 0.0f)
          lodBase = ptConeLodBase(ptUvCross(emitter.uv01.xy, emitter.uv01.zw, emitter.uv2.xy),
                                  ptExactCross(emitter.edge1Probability.xyz, emitter.edge2Cdf.xyz), direction,
                                  ptConeWidthOrLevelZero(cone, reach));
        const Material material = scene.geometry.materials[info.material];
        arriving = scene.maps.sampleTexture((info.slots >> 16u) & 0xFFu,
                                            emitter.uv01.xy * (1.0f - barycentricX - barycentricY) +
                                                emitter.uv01.zw * barycentricX + emitter.uv2.xy * barycentricY,
                                            (material.texture.y >> 16u) & 0xFFu, lodBase)
                       .xyz *
                   material.emissive.xyz * material.emissive.w * (cosine / (reach * reach));
      }
      reach = reach * 0.9999f;
    }
    return arriving;
  }

  // The integrand f(x, l) L(y) G without visibility at a receiver. bsdfPdf: the BSDF's
  // solid-angle density of the direction, for MIS.
  float3 integrand(PtBsdf bsdf, float3 position, float3 geometricNormal, uint light, float2 params, float2 cone,
                          out float3 direction, out float reach, out float sourcePdf, out float solidAnglePdf,
                          out float bsdfPdf) {
    const float3 arriving = lightAt(light, params, position, cone, direction, reach, sourcePdf, solidAnglePdf);
    bsdfPdf = 0.0f;
    float3 value = float3(0.0f);
    if (ptMaxComponent(arriving) > 0.0f) value = bsdf.evaluate(geometricNormal, direction, bsdfPdf) * arriving;
    return value;
  }

  // Initial resampling: M candidates from the light sampler (groups 2 + i of bounce 0),
  // resampled by p_hat / p. W = weightSum / (M p_hat(y)).
  PtReservoir initial(PtBsdf bsdf, float3 position, float3 geometricNormal, float2 cone, uint pathSeedValue,
                             uint candidates) {
    PtReservoir r = PtReservoir.empty();
    const uint stream = ptRestirStream(pathSeedValue, candidates);
    for (uint i = 0u; i < candidates; ++i) {
      const float4 u = pathRandom4(pathSeedValue, 0u, 2u + i);
      const uint light = candidate(u);
      float3 direction;
      float reach, sourcePdf, solidAnglePdf, bsdfPdf;
      const float target =
          ptRestirTarget(integrand(bsdf, position, geometricNormal, light, u.xy, cone, direction, reach, sourcePdf, solidAnglePdf, bsdfPdf));
      float weight = 0.0f;
      if (sourcePdf > 0.0f) weight = target / sourcePdf;
      r.stream(light, u.xy, weight, target, ptRestirStreamRandom(stream, i));
      r.count = r.count + 1.0f;
    }
    r.finish(r.count); // an empty reservoir keeps its M: it could have held any sample
    return r;
  }

  // The surface and BSDF of a primary surface record, as the path tracer builds them.
  PtBsdf bsdfOf(PtRestirSurface s, out PtSurface surface) {
    surface = ptSurfaceAt(scene.geometry, scene.maps, s.primaryHit(), -s.view.xyz, s.view.w);
    return PtBsdf(surface, s.view.xyz, scene.specularAlbedo);
  }

  // p_hat of a sample at a receiver whose BSDF is given. cameraCone: ptCameraCone(lens).
  float targetAt(PtBsdf bsdf, PtSurface surface, float coneWidth, float2 cameraCone, uint light, float2 params) {
    float3 direction;
    float reach, sourcePdf, solidAnglePdf, bsdfPdf;
    return ptRestirTarget(integrand(bsdf, surface.position, surface.geometricNormal, light, params, ptConeShadow(cameraCone, coneWidth),
                                    direction, reach, sourcePdf, solidAnglePdf, bsdfPdf));
  }

  // p_hat of a sample at another receiver, from its record.
  float targetAtRecord(PtRestirSurface s, float2 cameraCone, uint light, float2 params) {
    if (s.positionDepth.w <= 0.0f || light == kPtRestirNoLight) return 0.0f;
    PtSurface surface;
    const PtBsdf bsdf = bsdfOf(s, surface);
    return targetAt(bsdf, surface, s.view.w, cameraCone, light, params);
  }

  // Temporal reuse: the previous frame's final reservoir at the reprojected pixel, when that
  // pixel's surface is similar, its M clamped to 20 M_initial, combined with the owner's initial
  // reservoir by the generalized balance heuristic (targets without visibility, so "unbiased up
  // to visibility").
  PtReservoir temporal<S : IArray<PtRestirSurface>, R : IArray<PtReservoir>>(
      PtRestirCamera previous, float2 cameraCone, uint pixel, uint frame, uint candidates, PtRestirSurface owner, PtReservoir current,
      S previousSurfaces, R previousReservoirs) {
    if (previous.image.z == 0u || owner.positionDepth.w <= 0.0f) return current;
    const float3 v = owner.positionDepth.xyz - previous.position.xyz;
    const float depth = dot(v, previous.forward.xyz);
    if (depth <= 0.0f) return current;
    const float3 r = previous.right.xyz, up = previous.up.xyz;
    const float x = dot(v, r) / max(depth * dot(r, r), 1e-20f);
    const float y = dot(v, up) / max(depth * dot(up, up), 1e-20f);
    const float px = floor((x * 0.5f + 0.5f) * float(previous.image.x));
    const float py = floor((0.5f - y * 0.5f) * float(previous.image.y));
    if (px < 0.0f || py < 0.0f || px >= float(previous.image.x) || py >= float(previous.image.y)) return current;
    const uint neighbour = uint(py) * previous.image.x + uint(px);
    const PtRestirSurface other = previousSurfaces[neighbour];
    if (!owner.similar(other, depth)) return current;
    PtReservoir history = previousReservoirs[neighbour];
    history.count = min(history.count, kPtRestirHistoryLimit * float(candidates));

    PtSurface surface;
    const PtBsdf bsdf = bsdfOf(owner, surface);
    // Each sample's target at the other contributor; at its own, the target it carries.
    const float historyAtOwner =
        targetAt(bsdf, surface, owner.view.w, cameraCone, history.light, float2(history.paramX, history.paramY));
    const float currentAtHistory = targetAtRecord(other, cameraCone, current.light, float2(current.paramX, current.paramY));
    PtReservoir out = PtReservoir.empty();
    out.merge(current, current.target, current.count * current.target + history.count * currentAtHistory,
              ptRestirReuseRandom(pixel, frame, uniforms.counts.z, 0u));
    out.merge(history, historyAtOwner, current.count * historyAtOwner + history.count * history.target,
              ptRestirReuseRandom(pixel, frame, uniforms.counts.z, 1u));
    out.finish(1.0f);
    return out;
  }

  // Spatial reuse: up to three neighbours within 16 pixels with similar surfaces, combined with
  // the owner's (temporally reused) reservoir by the generalized balance heuristic. Neighbours
  // lend the reservoirs of their own temporal step. Contributor 0 is the owner, 1..3 the
  // accepted neighbours (width * height when rejected).
  PtReservoir spatial<S : IArray<PtRestirSurface>, R : IArray<PtReservoir>>(float2 cameraCone, uint pixel, uint frame, uint width,
                                                                                    uint height, S surfaces, R reservoirs) {
    const PtRestirSurface owner = surfaces[pixel];
    const PtReservoir current = reservoirs[pixel];
    if (owner.positionDepth.w <= 0.0f) return current;
    const uint none = width * height;
    const uint seed = uniforms.counts.z;
    uint n1 = ptRestirNeighbour(pixel, frame, seed, 0u, width, height);
    uint n2 = ptRestirNeighbour(pixel, frame, seed, 1u, width, height);
    uint n3 = ptRestirNeighbour(pixel, frame, seed, 2u, width, height);
    // Both surfaces are seen by this frame's camera, so the owner's depth is the depth to test.
    if (n1 < none && !owner.similar(surfaces[n1], owner.positionDepth.w)) n1 = none;
    if (n2 < none && !owner.similar(surfaces[n2], owner.positionDepth.w)) n2 = none;
    if (n3 < none && !owner.similar(surfaces[n3], owner.positionDepth.w)) n3 = none;
    PtSurface surface;
    const PtBsdf bsdf = bsdfOf(owner, surface);
    PtReservoir out = PtReservoir.empty();
    for (uint i = 0u; i < 4u; ++i) {
      const uint source = i == 0u ? pixel : i == 1u ? n1 : i == 2u ? n2 : n3;
      if (source >= none) continue;
      const PtReservoir incoming = reservoirs[source];
      const float2 params = float2(incoming.paramX, incoming.paramY);
      float atOwner = incoming.target;
      if (i > 0u) atOwner = targetAt(bsdf, surface, owner.view.w, cameraCone, incoming.light, params);
      // sum_j M_j p_hat_j(y) over the contributors; the sample's own carries its target.
      float balance = 0.0f;
      if (incoming.light != kPtRestirNoLight) {
        for (uint j = 0u; j < 4u; ++j) {
          const uint other = j == 0u ? pixel : j == 1u ? n1 : j == 2u ? n2 : n3;
          if (other >= none) continue;
          float target = incoming.target;
          if (j != i && j == 0u) target = atOwner;
          if (j != i && j > 0u) target = targetAtRecord(surfaces[other], cameraCone, incoming.light, params);
          balance = balance + reservoirs[other].count * target;
        }
      }
      out.merge(incoming, atOwner, balance, ptRestirReuseRandom(pixel, frame, seed, 8u + i));
    }
    out.finish(1.0f);
    return out;
  }

  // The primary vertex's direct light from its final reservoir, before visibility: f L G W, times
  // the MIS weight against BSDF sampling that next-event estimation gives a light sample (so
  // emission that BSDF sampling finds keeps its weight), clamped as next-event estimation clamps.
  // Fills the shadow ray (as the path loop builds it) and, for the guides, the diffuse fraction.
  float3 shade(float2 cameraCone, PtRestirSurface owner, PtReservoir r, uint pathSeedValue, out float3 shadowOrigin,
                      out float3 shadowDirection, out float shadowReach, out float2 shadowCone, out float3 diffuseFraction) {
    shadowOrigin = float3(0.0f);
    shadowDirection = float3(0.0f, 1.0f, 0.0f);
    shadowReach = 0.0f;
    shadowCone = cameraCone;
    diffuseFraction = float3(1.0f);
    if (owner.positionDepth.w <= 0.0f || r.light == kPtRestirNoLight || r.weight <= 0.0f) return float3(0.0f);
    PtSurface surface;
    const PtBsdf bsdf = bsdfOf(owner, surface);
    shadowCone = ptConeShadow(cameraCone, owner.view.w);
    // The emitter's texture at the sampled lobe's widened cone, as next-event estimation reads it
    // (the path's BSDF sample at bounce 0 is group 1 of bounce 1).
    const float2 emitterCone = ptConeScattered(cameraCone, owner.view.w, bsdf.sampledAlpha(pathRandom4(pathSeedValue, 1u, 1u).z));
    float3 direction;
    float reach, sourcePdf, solidAnglePdf, bsdfPdf;
    const float3 value = integrand(bsdf, surface.position, surface.geometricNormal, r.light, float2(r.paramX, r.paramY), emitterCone,
                                   direction, reach, sourcePdf, solidAnglePdf, bsdfPdf);
    float weight = r.weight;
    if (uint(uniforms.path.w) == 0u && solidAnglePdf > 0.0f) weight = weight * ptPowerHeuristic(solidAnglePdf, bsdfPdf);
    const float3 contribution = ptClampContribution(value * weight, uniforms.path.z);
    if (!(ptMaxComponent(contribution) > 0.0f)) return float3(0.0f);
    shadowOrigin = ptOffsetRay(surface.position, surface.geometricNormal);
    if (dot(direction, surface.geometricNormal) < 0.0f) shadowOrigin = ptOffsetRay(surface.position, -surface.geometricNormal);
    shadowDirection = direction;
    shadowReach = reach;
    float unused;
    diffuseFraction = bsdf.diffuseFraction(direction, bsdf.evaluate(surface.geometricNormal, direction, unused));
    return contribution;
  }
};
