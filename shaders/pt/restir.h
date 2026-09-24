// ReSTIR DI at the primary vertex (Bitterli et al. 2020). Shared by the CPU
// tracer and the GPU reservoir kernels. A light sample y is a light identity (the kind
// in the top two bits: 0 environment, 1 sun, 2 punctual, 3 emissive triangle; the light or
// triangle index below) plus the two random numbers that place it on the light: the
// environment's and the sun's direction numbers, or an emissive triangle's barycentric
// numbers. Re-evaluating a sample at another receiver is therefore exact, and its source
// density is the V6 light sampler's in a receiver-independent measure (solid angle of a
// distant light, area of a triangle, the discrete choice of a punctual light), so reuse
// needs no Jacobian.
//
// The includer defines PT_TEXTURE_PARAMS/ARGS, PT_ENVIRONMENT_PARAMS/ARGS and
// ptSampleEnvironment, as for lights.h.
#pragma once
#include "path.h"
#include "raycone.h"
#include "surface.h"
#include "bsdf.h"
#include "lights.h"

PT_CONSTANT uint kPtRestirNoLight = 0xFFFFFFFFu;
PT_CONSTANT uint kPtRestirMaxCandidates = 13u;  // groups 2..14 of bounce 0; 15 is the stream
PT_CONSTANT float kPtRestirHistoryLimit = 20.0f;  // temporal M is clamped to 20 M_initial

// 32 bytes. light: kind << 30 | index, kPtRestirNoLight when empty; weight: the unbiased
// contribution weight W (0 when empty); target: p_hat of the sample at the owner.
struct PtReservoir {
  uint light;
  float paramX;
  float paramY;
  float weightSum;
  float count;
  float weight;
  float target;
  uint reserved;
};

// The primary surface a reservoir belongs to, for re-evaluating targets at it and for the
// reuse validity tests. positionDepth.w: linear depth along the view axis, 0 without a
// surface; normalMaterial: shading normal and the material index (bits); view: unit direction
// to the eye and the ray cone's width at the hit (ptConeWidthOrLevelZero); hit: instance,
// primitive and the barycentrics' bits.
struct PtRestirSurface {
  float4 positionDepth;
  float4 normalMaterial;
  float4 view;
  uint4 hit;
};

// What light sampling reads from the uniforms.
struct PtRestirLights {
  float4 environment;
  float4 distribution;
  float4 sunDirection;
  float4 sunRadiance;
  float4 path;
  uint4 counts;
  uint4 emissive;
};

#define PT_RESTIR_LIGHT_PARAMS                                                                    \
  device const Light *lights, device const float *environmentDistribution,                        \
      device const PtEmissiveTriangle *emissiveTriangles, device const TraceInstance *traceInstances, \
      device const Material *materials, PT_TEXTURE_PARAMS, PT_ENVIRONMENT_PARAMS
#define PT_RESTIR_LIGHT_ARGS                                                                      \
  lights, environmentDistribution, emissiveTriangles, traceInstances, materials, PT_TEXTURE_ARGS,    \
      PT_ENVIRONMENT_ARGS

inline PtRestirLights ptRestirLights(float4 environment, float4 distribution, float4 sunDirection, float4 sunRadiance,
                                     float4 path, uint4 counts, uint4 emissive) {
  PtRestirLights l;
  l.environment = environment;
  l.distribution = distribution;
  l.sunDirection = sunDirection;
  l.sunRadiance = sunRadiance;
  l.path = path;
  l.counts = counts;
  l.emissive = emissive;
  return l;
}

inline PtReservoir ptEmptyReservoir() {
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

// The light kinds present, as next-event estimation counts them (integrator.inc).
inline uint ptRestirKindCount(PtRestirLights l) {
  uint count = 0u;
  if (uint(l.path.w) != 1u && l.distribution.w > 0.5f) count = count + 1u;
  if (uint(l.path.w) != 1u && l.sunDirection.w > 0.0f) count = count + 1u;
  if (uint(l.path.w) != 1u && l.counts.x > 0u) count = count + 1u;
  if (uint(l.path.w) != 1u && l.emissive.x > 0u) count = count + 1u;
  return count;
}

// A candidate from four random numbers, chosen as next-event estimation chooses its light:
// z the kind, w the punctual light or emissive triangle; x, y place it (kept as the sample's
// parameters). kPtRestirNoLight when the scene has no light.
inline uint ptRestirCandidate(PtRestirLights l, device const PtEmissiveTriangle *emissiveTriangles, float4 u) {
  const uint kindCount = ptRestirKindCount(l);
  uint light = kPtRestirNoLight;
  if (kindCount > 0u) {
    uint slot = min(uint(u.z * float(kindCount)), kindCount - 1u);
    if (uint(l.path.w) != 1u && l.distribution.w > 0.5f) {
      if (slot == 0u) light = 0u;
      slot = slot - 1u;
    }
    if (uint(l.path.w) != 1u && l.sunDirection.w > 0.0f && light == kPtRestirNoLight) {
      if (slot == 0u) light = 1u << 30u;
      slot = slot - 1u;
    }
    if (uint(l.path.w) != 1u && l.counts.x > 0u && light == kPtRestirNoLight) {
      if (slot == 0u) light = (2u << 30u) | min(uint(u.w * float(l.counts.x)), l.counts.x - 1u);
      slot = slot - 1u;
    }
    if (uint(l.path.w) != 1u && l.emissive.x > 0u && light == kPtRestirNoLight)
      light = (3u << 30u) | ptEmissiveSelect(emissiveTriangles, l.emissive.x, u.w);
  }
  return light;
}

// A light sample seen from `position`: returns what arrives in the sample's own measure (the
// radiance times |cos| / distance^2 for a triangle, the radiance of a distant light, a
// punctual light's intensity over distance^2), with the unit direction towards it, the
// shadow ray's extent, its source density in that measure (kind choice included) and, for
// MIS against BSDF sampling, the solid-angle density next-event estimation would give it
// (0 for a punctual light). cone: the shadow ray's cone, for emitter textures.
inline float3 ptRestirLightAt(PT_RESTIR_LIGHT_PARAMS, PtRestirLights l, uint light, float2 params, float3 position,
                              float2 cone, thread float3 &direction, thread float &reach, thread float &sourcePdf,
                              thread float &solidAnglePdf) {
  const float kindShare = 1.0f / max(float(ptRestirKindCount(l)), 1.0f);
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
    float pdf = 0.0f;
    direction = ptSampleEnvironmentDirection(environmentDistribution, l.distribution, params, pdf);
    sourcePdf = pdf * kindShare;
    solidAnglePdf = sourcePdf;
    arriving = ptSampleEnvironment(PT_ENVIRONMENT_ARGS, equirectangularUV(direction)) * l.environment.x;
  } else if (kind == 1u) {
    direction = ptSampleCone(xyz(l.sunDirection), l.sunDirection.w, params);
    sourcePdf = (1.0f / max(l.sunRadiance.w, 1e-12f)) * kindShare;
    solidAnglePdf = sourcePdf;
    arriving = xyz(l.sunRadiance);
  } else if (kind == 2u) {
    float distance = 0.0f;
    arriving = ptPunctualLight(lights[index], position, direction, distance);
    reach = distance * 0.999f;
    sourcePdf = (1.0f / float(l.counts.x)) * kindShare;
  } else {
    const float barycentricX = sqrt(params.x) * (1.0f - params.y);
    const float barycentricY = sqrt(params.x) * params.y;
    direction = xyz(emissiveTriangles[index].v0Area) + xyz(emissiveTriangles[index].edge1Probability) * barycentricX +
                xyz(emissiveTriangles[index].edge2Cdf) * barycentricY - position;
    reach = length(direction);
    if (reach > 0.0f) direction = direction / reach;
    float cosine = -dot(xyz(emissiveTriangles[index].normal), direction);
    if ((traceInstances[emissiveTriangles[index].identity.x].flags & kInstanceDoubleSided) != 0u) cosine = abs(cosine);
    if (reach > 1e-5f && cosine > 1e-6f) {
      const float area = emissiveTriangles[index].v0Area.w;
      const float probability = emissiveTriangles[index].edge1Probability.w;
      sourcePdf = probability / area * kindShare;
      solidAnglePdf = probability * reach * reach / (area * cosine) * kindShare;
      float lodBase = kPtLevelZero;
      if (cone.x >= 0.0f)
        lodBase = ptConeLodBase(ptUvCross(xy(emissiveTriangles[index].uv01), zw(emissiveTriangles[index].uv01),
                                          xy(emissiveTriangles[index].uv2)),
                                ptExactCross(xyz(emissiveTriangles[index].edge1Probability),
                                             xyz(emissiveTriangles[index].edge2Cdf)),
                                direction, ptConeWidthOrLevelZero(cone, reach));
      const uint material = traceInstances[emissiveTriangles[index].identity.x].material;
      arriving = xyz(ptSampleTexture(PT_TEXTURE_ARGS, (traceInstances[emissiveTriangles[index].identity.x].slots >> 16u) & 0xFFu,
                                     xy(emissiveTriangles[index].uv01) * (1.0f - barycentricX - barycentricY) +
                                         zw(emissiveTriangles[index].uv01) * barycentricX +
                                         xy(emissiveTriangles[index].uv2) * barycentricY,
                                     (materials[material].texture.y >> 16u) & 0xFFu, lodBase)) *
                 xyz(materials[material].emissive) * materials[material].emissive.w * (cosine / (reach * reach));
    }
    reach = reach * 0.9999f;
  }
  return arriving;
}

// The integrand f(x, l) L(y) G without visibility at a receiver, and its target p_hat (the
// luminance). bsdfPdf: the BSDF's solid-angle density of the direction, for MIS.
inline float3 ptRestirIntegrand(PT_RESTIR_LIGHT_PARAMS, PtRestirLights l, thread const PtBsdf &bsdf,
                                float3 position, float3 geometricNormal, uint light, float2 params, float2 cone,
                                thread float3 &direction, thread float &reach, thread float &sourcePdf,
                                thread float &solidAnglePdf, thread float &bsdfPdf) {
  const float3 arriving = ptRestirLightAt(PT_RESTIR_LIGHT_ARGS, l, light, params, position, cone, direction, reach,
                                          sourcePdf, solidAnglePdf);
  bsdfPdf = 0.0f;
  float3 value = float3(0.0f);
  if (ptMaxComponent(arriving) > 0.0f) value = ptBsdfEvaluate(bsdf, geometricNormal, direction, bsdfPdf) * arriving;
  return value;
}

inline float ptRestirTarget(float3 integrand) { return max(ptLuminance(integrand), 0.0f); }

// Streaming weighted selection: adds a sample of resampling weight `weight`, keeping it with
// probability weight / weightSum. The count is kept by the caller.
inline void ptReservoirStream(thread PtReservoir &r, uint light, float2 params, float weight, float target,
                              float random) {
  r.weightSum = r.weightSum + weight;
  if (weight > 0.0f && random * r.weightSum < weight) {
    r.light = light;
    r.paramX = params.x;
    r.paramY = params.y;
    r.target = target;
  }
}

// The stream's random number for its k-th decision: the reservoir stream group (M + 2 of
// bounce 0) hashed with the decision's index.
inline float ptRestirStreamRandom(uint stream, uint k) { return hashFloat(stream + k * 0x9E3779B9u); }
inline uint ptRestirStream(uint pathSeedValue, uint candidates) {
  return as_type<uint>(pathRandom4(pathSeedValue, 0u, 2u + candidates).x);
}

// Adds a contributor's reservoir to a combination at the owner (the generalized
// balance heuristic over the contributors, targets without visibility). Its sample's
// resampling weight is m(y) p_hat_owner(y) W, with the MIS weight
// m(y) = M p_hat_contributor(y) / sum_j M_j p_hat_j(y) over every contributor j (this one
// included); `balance` is that sum. The count is added whether or not a sample is held.
inline void ptReservoirMerge(thread PtReservoir &out, PtReservoir incoming, float targetAtOwner, float balance,
                             float random) {
  float weight = 0.0f;
  if (incoming.light != kPtRestirNoLight && balance > 0.0f)
    weight = incoming.count * incoming.target / balance * targetAtOwner * incoming.weight;
  ptReservoirStream(out, incoming.light, float2(incoming.paramX, incoming.paramY), weight, targetAtOwner, random);
  out.count = out.count + incoming.count;
}

// A finished reservoir: W = weightSum / p_hat_owner(y) (the MIS weights already sum to one), or
// empty with its count kept.
inline void ptReservoirFinish(thread PtReservoir &out, float count) {
  if (out.light != kPtRestirNoLight && out.target > 0.0f) {
    out.weight = out.weightSum / (count * out.target);
  } else {
    const float kept = out.count;
    out = ptEmptyReservoir();
    out.count = kept;
  }
}

// Initial resampling: M candidates from the V6 light sampler (groups 2 + i of bounce 0),
// resampled by p_hat / p. W = weightSum / (M p_hat(y)).
inline PtReservoir ptRestirInitial(PT_RESTIR_LIGHT_PARAMS, PtRestirLights l, thread const PtBsdf &bsdf,
                                   float3 position, float3 geometricNormal, float2 cone, uint pathSeedValue,
                                   uint candidates) {
  PtReservoir r = ptEmptyReservoir();
  const uint stream = ptRestirStream(pathSeedValue, candidates);
  for (uint i = 0u; i < candidates; ++i) {
    const float4 u = pathRandom4(pathSeedValue, 0u, 2u + i);
    const uint light = ptRestirCandidate(l, emissiveTriangles, u);
    float3 direction = float3(0.0f);
    float reach = 0.0f, sourcePdf = 0.0f, solidAnglePdf = 0.0f, bsdfPdf = 0.0f;
    const float target = ptRestirTarget(ptRestirIntegrand(PT_RESTIR_LIGHT_ARGS, l, bsdf, position, geometricNormal, light,
                                                          xy(u), cone, direction, reach, sourcePdf, solidAnglePdf,
                                                          bsdfPdf));
    float weight = 0.0f;
    if (sourcePdf > 0.0f) weight = target / sourcePdf;
    ptReservoirStream(r, light, xy(u), weight, target, ptRestirStreamRandom(stream, i));
    r.count = r.count + 1.0f;
  }
  ptReservoirFinish(r, r.count);  // an empty reservoir keeps its M: it could have held any sample
  return r;
}

// Whether a neighbour's surface may lend its reservoir: the same material, shading
// normals within 0.9, relative depth within 0.1. depthAtNeighbour: the owner's depth seen as
// the neighbour's (the reprojected depth for temporal reuse).
inline bool ptRestirSimilar(PtRestirSurface owner, PtRestirSurface neighbour, float depthAtNeighbour) {
  if (owner.positionDepth.w <= 0.0f || neighbour.positionDepth.w <= 0.0f) return false;
  if (as_type<uint>(owner.normalMaterial.w) != as_type<uint>(neighbour.normalMaterial.w)) return false;
  if (dot(xyz(owner.normalMaterial), xyz(neighbour.normalMaterial)) < 0.9f) return false;
  return abs(neighbour.positionDepth.w - depthAtNeighbour) <= 0.1f * max(depthAtNeighbour, 1e-6f);
}

inline PtHit ptRestirSurfaceHit(PtRestirSurface s) {
  PtHit hit;
  hit.ambiguous = 0u;
  hit.t = 0.0f;
  hit.barycentric = float2(as_type<float>(s.hit.z), as_type<float>(s.hit.w));
  hit.instance = s.hit.x;
  hit.primitive = s.hit.y;
  hit.found = s.positionDepth.w > 0.0f ? 1u : 0u;
  hit.nodeVisits = 0u;
  hit.triangleTests = 0u;
  return hit;
}

// The record of a primary hit: surface from ptSurfaceAt, the camera ray's direction, the
// cone's width at the hit and the camera (for the linear depth).
inline PtRestirSurface ptRestirSurfaceOf(thread const PtSurface &surface, PtHit hit, uint material,
                                         float3 rayDirection, float coneWidth, float3 cameraPosition,
                                         float3 cameraForward) {
  PtRestirSurface s;
  s.positionDepth = float4(surface.position, max(dot(surface.position - cameraPosition, cameraForward), 1e-6f));
  s.normalMaterial = float4(surface.normal, as_type<float>(material));
  s.view = float4(-rayDirection, coneWidth);
  s.hit = uint4(hit.instance, hit.primitive, as_type<uint>(hit.barycentric.x), as_type<uint>(hit.barycentric.y));
  return s;
}

// The random number of decision k of a pixel's reuse at a frame: hash(pixel, frame, 0x5E57),
// with the run's seed mixed in so that seeds differ in their reuse as in their candidates.
inline float ptRestirReuseRandom(uint pixel, uint frame, uint seed, uint k) {
  return hashFloat(pcgHash(pixel + pcgHash(frame + pcgHash(seed + 0x5E57u))) + k * 0x9E3779B9u);
}

#define PT_RESTIR_PASS_PARAMS                                                                     \
  PT_RESTIR_LIGHT_PARAMS, device const uint *indices, device const float *vertices,              \
      device const float *specularAlbedo
#define PT_RESTIR_PASS_ARGS PT_RESTIR_LIGHT_ARGS, indices, vertices, specularAlbedo

// A camera for reprojection (the previous frame's) and the image it covers.
struct PtRestirCamera {
  float4 position;
  float4 forward;
  float4 right;
  float4 up;
  uint4 image;  // width, height, history valid, reuse bits
};

// The camera ray of a pixel's sample, exactly as integrator.inc derives it, lens included.
inline float3 ptRestirCameraRay(float4 cameraPosition, float4 cameraForward, float4 cameraRight, float4 cameraUp,
                                float4 image, float4 lens, uint pixelX, uint pixelY, uint pathSeedValue,
                                thread float3 &origin) {
  const float4 jitter = pathRandom4(pathSeedValue, 0u, 0u);
  const float ndcX = (float(pixelX) + jitter.x) / image.x * 2.0f - 1.0f;
  const float ndcY = 1.0f - (float(pixelY) + jitter.y) / image.y * 2.0f;
  origin = xyz(cameraPosition);
  float3 direction = normalize(xyz(cameraForward) + xyz(cameraRight) * ndcX + xyz(cameraUp) * ndcY);
  ptApplyThinLens(lens, xyz(cameraForward), xyz(cameraRight), xyz(cameraUp), float2(jitter.z, jitter.w), origin,
                  direction);
  return direction;
}

// The surface and BSDF of a primary surface record, as the path tracer builds them.
#define PT_RESTIR_BSDF_ARGS traceInstances, materials, indices, vertices, PT_TEXTURE_ARGS, specularAlbedo
inline PtBsdf ptRestirBsdf(PT_SCENE_PARAMS, PT_TEXTURE_PARAMS, device const float *specularAlbedo, PtRestirSurface s,
                           thread PtSurface &surface) {
  surface = ptSurfaceAt(traceInstances, materials, indices, vertices, PT_TEXTURE_ARGS, ptRestirSurfaceHit(s),
                        -xyz(s.view), s.view.w);
  return ptMakeBsdf(surface, xyz(s.view), specularAlbedo);
}

// p_hat of a sample at a receiver whose BSDF is given. cameraCone: ptCameraCone(lens).
inline float ptRestirTargetAt(PT_RESTIR_LIGHT_PARAMS, PtRestirLights l, thread const PtBsdf &bsdf,
                              thread const PtSurface &surface, float coneWidth, float2 cameraCone, uint light,
                              float2 params) {
  float3 direction = float3(0.0f);
  float reach = 0.0f, sourcePdf = 0.0f, solidAnglePdf = 0.0f, bsdfPdf = 0.0f;
  return ptRestirTarget(ptRestirIntegrand(PT_RESTIR_LIGHT_ARGS, l, bsdf, surface.position, surface.geometricNormal,
                                          light, params, ptConeShadow(cameraCone, coneWidth), direction, reach,
                                          sourcePdf, solidAnglePdf, bsdfPdf));
}

// p_hat of a sample at another receiver, from its record.
inline float ptRestirTargetAtRecord(PT_RESTIR_PASS_PARAMS, PtRestirLights l, PtRestirSurface s, float2 cameraCone,
                                    uint light, float2 params) {
  if (s.positionDepth.w <= 0.0f || light == kPtRestirNoLight) return 0.0f;
  PtSurface surface;
  const PtBsdf bsdf = ptRestirBsdf(PT_RESTIR_BSDF_ARGS, s, surface);
  return ptRestirTargetAt(PT_RESTIR_LIGHT_ARGS, l, bsdf, surface, s.view.w, cameraCone, light, params);
}

// Temporal reuse: the previous frame's final reservoir at the reprojected pixel,
// when that pixel's surface is similar, its M clamped to 20 M_initial, combined with the
// owner's initial reservoir by the generalized balance heuristic (targets without
// visibility, so "unbiased up to visibility").
inline PtReservoir ptRestirTemporal(PT_RESTIR_PASS_PARAMS, PtRestirLights l, PtRestirCamera previous,
                                    float2 cameraCone, uint pixel, uint frame, uint candidates,
                                    PtRestirSurface owner, PtReservoir current,
                                    device const PtRestirSurface *previousSurfaces,
                                    device const PtReservoir *previousReservoirs) {
  if (previous.image.z == 0u || owner.positionDepth.w <= 0.0f) return current;
  const float3 v = xyz(owner.positionDepth) - xyz(previous.position);
  const float depth = dot(v, xyz(previous.forward));
  if (depth <= 0.0f) return current;
  const float3 r = xyz(previous.right), up = xyz(previous.up);
  const float x = dot(v, r) / max(depth * dot(r, r), 1e-20f);
  const float y = dot(v, up) / max(depth * dot(up, up), 1e-20f);
  const float px = floor((x * 0.5f + 0.5f) * float(previous.image.x));
  const float py = floor((0.5f - y * 0.5f) * float(previous.image.y));
  if (px < 0.0f || py < 0.0f || px >= float(previous.image.x) || py >= float(previous.image.y)) return current;
  const uint neighbour = uint(py) * previous.image.x + uint(px);
  const PtRestirSurface other = previousSurfaces[neighbour];
  if (!ptRestirSimilar(owner, other, depth)) return current;
  PtReservoir history = previousReservoirs[neighbour];
  history.count = min(history.count, kPtRestirHistoryLimit * float(candidates));

  PtSurface surface;
  const PtBsdf bsdf = ptRestirBsdf(PT_RESTIR_BSDF_ARGS, owner, surface);
  // Each sample's target at the other contributor; at its own, the target it carries.
  const float historyAtOwner = ptRestirTargetAt(PT_RESTIR_LIGHT_ARGS, l, bsdf, surface, owner.view.w, cameraCone,
                                                history.light, float2(history.paramX, history.paramY));
  const float currentAtHistory = ptRestirTargetAtRecord(PT_RESTIR_PASS_ARGS, l, other, cameraCone, current.light,
                                                        float2(current.paramX, current.paramY));
  PtReservoir out = ptEmptyReservoir();
  ptReservoirMerge(out, current, current.target, current.count * current.target + history.count * currentAtHistory,
                   ptRestirReuseRandom(pixel, frame, l.counts.z, 0u));
  ptReservoirMerge(out, history, historyAtOwner, current.count * historyAtOwner + history.count * history.target,
                   ptRestirReuseRandom(pixel, frame, l.counts.z, 1u));
  ptReservoirFinish(out, 1.0f);
  return out;
}

PT_CONSTANT uint kPtRestirSpatialNeighbours = 3u;
PT_CONSTANT float kPtRestirSpatialRadius = 16.0f;

// A spatial neighbour of a pixel: uniform in a disc of 16 pixels, from the reuse hash; the
// owner itself or a pixel outside the image is returned as width * height (none).
inline uint ptRestirNeighbour(uint pixel, uint frame, uint seed, uint k, uint width, uint height) {
  const float angle = 2.0f * kPi * ptRestirReuseRandom(pixel, frame, seed, 2u + 2u * k);
  const float radius = kPtRestirSpatialRadius * sqrt(ptRestirReuseRandom(pixel, frame, seed, 3u + 2u * k));
  const float nx = floor(float(pixel % width) + 0.5f + radius * cos(angle));
  const float ny = floor(float(pixel / width) + 0.5f + radius * sin(angle));
  uint neighbour = width * height;
  if (nx >= 0.0f && ny >= 0.0f && nx < float(width) && ny < float(height)) neighbour = uint(ny) * width + uint(nx);
  if (neighbour == pixel) neighbour = width * height;
  return neighbour;
}

// Spatial reuse: up to three neighbours within 16 pixels with similar surfaces,
// combined with the owner's (temporally reused) reservoir by the generalized balance
// heuristic. Neighbours lend the reservoirs of their own temporal step. Contributor 0 is the
// owner, 1..3 the accepted neighbours (width * height when rejected).
inline PtReservoir ptRestirSpatial(PT_RESTIR_PASS_PARAMS, PtRestirLights l, float2 cameraCone, uint pixel, uint frame,
                                   uint width, uint height, device const PtRestirSurface *surfaces,
                                   device const PtReservoir *reservoirs) {
  const PtRestirSurface owner = surfaces[pixel];
  const PtReservoir current = reservoirs[pixel];
  if (owner.positionDepth.w <= 0.0f) return current;
  const uint none = width * height;
  uint n1 = ptRestirNeighbour(pixel, frame, l.counts.z, 0u, width, height);
  uint n2 = ptRestirNeighbour(pixel, frame, l.counts.z, 1u, width, height);
  uint n3 = ptRestirNeighbour(pixel, frame, l.counts.z, 2u, width, height);
  // Both surfaces are seen by this frame's camera, so the owner's depth is the depth to test.
  if (n1 < none && !ptRestirSimilar(owner, surfaces[n1], owner.positionDepth.w)) n1 = none;
  if (n2 < none && !ptRestirSimilar(owner, surfaces[n2], owner.positionDepth.w)) n2 = none;
  if (n3 < none && !ptRestirSimilar(owner, surfaces[n3], owner.positionDepth.w)) n3 = none;
  PtSurface surface;
  const PtBsdf bsdf = ptRestirBsdf(PT_RESTIR_BSDF_ARGS, owner, surface);
  PtReservoir out = ptEmptyReservoir();
  for (uint i = 0u; i < 4u; ++i) {
    uint source = pixel;
    if (i == 1u) source = n1;
    if (i == 2u) source = n2;
    if (i == 3u) source = n3;
    if (source >= none) continue;
    const PtReservoir incoming = reservoirs[source];
    const float2 params = float2(incoming.paramX, incoming.paramY);
    float atOwner = incoming.target;
    if (i > 0u) atOwner = ptRestirTargetAt(PT_RESTIR_LIGHT_ARGS, l, bsdf, surface, owner.view.w, cameraCone,
                                           incoming.light, params);
    // sum_j M_j p_hat_j(y) over the contributors; the sample's own carries its target.
    float balance = 0.0f;
    if (incoming.light != kPtRestirNoLight) {
      for (uint j = 0u; j < 4u; ++j) {
        uint other = pixel;
        if (j == 1u) other = n1;
        if (j == 2u) other = n2;
        if (j == 3u) other = n3;
        if (other >= none) continue;
        float target = incoming.target;
        if (j != i && j == 0u) target = atOwner;
        if (j != i && j > 0u)
          target = ptRestirTargetAtRecord(PT_RESTIR_PASS_ARGS, l, surfaces[other], cameraCone, incoming.light, params);
        balance = balance + reservoirs[other].count * target;
      }
    }
    ptReservoirMerge(out, incoming, atOwner, balance, ptRestirReuseRandom(pixel, frame, l.counts.z, 8u + i));
  }
  ptReservoirFinish(out, 1.0f);
  return out;
}

// The primary vertex's direct light from its final reservoir, before visibility: f L G W,
// times the MIS weight against BSDF sampling that next-event estimation gives a light sample
// (so emission that BSDF sampling finds keeps its V6 weight), clamped as next-event
// estimation clamps. Fills the shadow ray (as integrator.inc builds it) and, for the guides,
// the diffuse fraction.
inline float3 ptRestirShade(PT_RESTIR_PASS_PARAMS, PtRestirLights l, float2 cameraCone, PtRestirSurface owner,
                            PtReservoir r, uint pathSeedValue, thread float3 &shadowOrigin,
                            thread float3 &shadowDirection, thread float &shadowReach, thread float2 &shadowCone,
                            thread float3 &diffuseFraction) {
  shadowOrigin = float3(0.0f);
  shadowDirection = float3(0.0f, 1.0f, 0.0f);
  shadowReach = 0.0f;
  shadowCone = cameraCone;
  diffuseFraction = float3(1.0f);
  if (owner.positionDepth.w <= 0.0f || r.light == kPtRestirNoLight || r.weight <= 0.0f) return float3(0.0f);
  PtSurface surface;
  const PtBsdf bsdf = ptRestirBsdf(PT_RESTIR_BSDF_ARGS, owner, surface);
  float3 direction = float3(0.0f);
  float reach = 0.0f, sourcePdf = 0.0f, solidAnglePdf = 0.0f, bsdfPdf = 0.0f;
  shadowCone = ptConeShadow(cameraCone, owner.view.w);
  // The emitter's texture at the sampled lobe's widened cone, as next-event estimation reads
  // it (the path's BSDF sample at bounce 0 is group 1 of bounce 1).
  const float2 emitterCone = ptConeScattered(cameraCone, owner.view.w,
                                             ptBsdfSampledAlpha(bsdf, pathRandom4(pathSeedValue, 1u, 1u).z));
  const float3 integrand = ptRestirIntegrand(PT_RESTIR_LIGHT_ARGS, l, bsdf, surface.position, surface.geometricNormal,
                                             r.light, float2(r.paramX, r.paramY), emitterCone, direction, reach,
                                             sourcePdf, solidAnglePdf, bsdfPdf);
  float weight = r.weight;
  if (uint(l.path.w) == 0u && solidAnglePdf > 0.0f) weight = weight * ptPowerHeuristic(solidAnglePdf, bsdfPdf);
  const float3 contribution = ptClampContribution(integrand * weight, l.path.z);
  if (!(ptMaxComponent(contribution) > 0.0f)) return float3(0.0f);
  shadowOrigin = ptOffsetRay(surface.position, surface.geometricNormal);
  if (dot(direction, surface.geometricNormal) < 0.0f) shadowOrigin = ptOffsetRay(surface.position, -surface.geometricNormal);
  shadowDirection = direction;
  shadowReach = reach;
  float unused = 0.0f;
  diffuseFraction = ptDiffuseFraction(bsdf, direction, ptBsdfEvaluate(bsdf, surface.geometricNormal, direction, unused));
  return contribution;
}
