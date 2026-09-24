// Shared reconstruction mathematics: compiled by both the host tests and MSL kernels.
#pragma once
#include "reconstruction.h"

inline bool ptFiniteColor(float3 v) {
  return !(isnan(v.x) || isnan(v.y) || isnan(v.z) || isinf(v.x) || isinf(v.y) || isinf(v.z));
}
// Display target is RGBA16F. Bound only reconstruction signals (never raw accumulation)
// to that finite range, which also prevents luminance second moments from overflowing.
inline float3 ptSafeColor(float3 v) {
  return ptFiniteColor(v) ? clamp(v, float3(0.0f), float3(65504.0f)) : float3(0.0f);
}
inline float ptTemporalLuminance(float3 v) { return dot(v, float3(0.2126f, 0.7152f, 0.0722f)); }

inline bool ptSameSurface(PtReconstructionSample a, PtReconstructionSample b) {
  return a.identity.z != 0u && a.identity.z != 2u && a.identity.x == b.identity.x &&
         a.identity.y == b.identity.y && a.identity.z == b.identity.z &&
         (a.identity.z != 3u || a.identity.w == b.identity.w) &&
         dot(xyz(a.geometricNormal), xyz(b.geometricNormal)) > 0.95f &&
         dot(xyz(a.normalRoughness), xyz(b.normalRoughness)) > 0.9f;
}

inline float ptPixelFootprint(PtTemporalUniforms u, float depth) {
  return max(1e-5f, depth * 2.0f * max(length(xyz(u.previousCamera.cameraRight)) / float(u.image.x),
                                     length(xyz(u.previousCamera.cameraUp)) / float(u.image.y)));
}

// Previous pixel coordinates use the actual path camera, without a second jitter offset:
// jitter is already represented by the traced world position.
inline float2 ptReproject(PtTemporalUniforms u, float3 position, thread float &depth) {
  const float3 v = position - xyz(u.previousCamera.cameraPosition);
  depth = dot(v, xyz(u.previousCamera.cameraForward));
  const float3 r = xyz(u.previousCamera.cameraRight), up = xyz(u.previousCamera.cameraUp);
  const float x = dot(v, r) / max(depth * dot(r, r), 1e-20f);
  const float y = dot(v, up) / max(depth * dot(up, up), 1e-20f);
  return float2((x * 0.5f + 0.5f) * float(u.image.x) - 0.5f,
                (0.5f - y * 0.5f) * float(u.image.y) - 0.5f);
}

inline PtTemporalHistory ptTemporalPixel(PtTemporalUniforms u, uint x, uint y,
    device const PtReconstructionSample *samples, device const PtReconstructionSample *previousSamples,
    device const PtTemporalHistory *previous) {
  const uint index = y * u.image.x + x;
  const PtReconstructionSample s = samples[index];
  const float3 d = ptSafeColor(xyz(s.diffuse)), p = ptSafeColor(xyz(s.specular));
  const float dl = ptTemporalLuminance(d), pl = ptTemporalLuminance(p);
  PtTemporalHistory result;
  result.diffuse = float4(d, 1.0f);
  result.specular = float4(p, 1.0f);
  result.moments = float4(dl, dl * dl, pl, pl * pl);
  if (s.identity.z == 0u || s.identity.z == 2u) return result;

  float3 dmin = d, dmax = d, pmin = p, pmax = p;
  float neighborD = 0.0f, neighborD2 = 0.0f, neighborP = 0.0f, neighborP2 = 0.0f, count = 0.0f;
  for (int oy = -1; oy <= 1; ++oy) for (int ox = -1; ox <= 1; ++ox) {
    const int nx = int(x) + ox, ny = int(y) + oy;
    if (nx < 0 || ny < 0 || nx >= int(u.image.x) || ny >= int(u.image.y)) continue;
    const PtReconstructionSample n = samples[uint(ny) * u.image.x + uint(nx)];
    if (!ptSameSurface(s, n)) continue;
    const float3 nd = ptSafeColor(xyz(n.diffuse)), np = ptSafeColor(xyz(n.specular));
    dmin = min(dmin, nd); dmax = max(dmax, nd);
    pmin = min(pmin, np); pmax = max(pmax, np);
    const float a = ptTemporalLuminance(nd), b = ptTemporalLuminance(np);
    neighborD = neighborD + a; neighborD2 = neighborD2 + a * a;
    neighborP = neighborP + b; neighborP2 = neighborP2 + b * b;
    count = count + 1.0f;
  }
  const float invCount = 1.0f / max(count, 1.0f);
  const float varD = max(0.0f, neighborD2 * invCount - (neighborD * invCount) * (neighborD * invCount));
  const float varP = max(0.0f, neighborP2 * invCount - (neighborP * invCount) * (neighborP * invCount));
  // Bootstrap variance from compatible neighbors when there is little temporal history.
  result.moments.y = result.moments.y + varD;
  result.moments.w = result.moments.w + varP;
  if (u.image.z != 0u) return result;
  // Depth of field: a lens sample moves a primary hit at depth z by up to
  // A |1 - z / F| across the image plane; beyond one pixel, history at the reprojected pixel
  // belongs to other lens positions and is rejected.
  const float4 lens = u.previousCamera.lens;
  if (lens.x > 0.0f &&
      lens.x * abs(1.0f - s.positionDepth.w / max(lens.y, 1e-6f)) > ptPixelFootprint(u, s.positionDepth.w))
    return result;
  float projectedDepth = 0.0f;
  const float2 pixel = ptReproject(u, xyz(s.positionDepth), projectedDepth);
  if (projectedDepth <= 0.0f || !ptFiniteColor(float3(pixel.x, pixel.y, projectedDepth))) return result;
  const float footprint = ptPixelFootprint(u, projectedDepth);
  float bestDistance = footprint * 2.0f;
  uint best = 0u;
  bool found = false;
  for (int oy = 0; oy <= 1; ++oy) for (int ox = 0; ox <= 1; ++ox) {
    const int px = int(floor(pixel.x)) + ox, py = int(floor(pixel.y)) + oy;
    if (px < 0 || py < 0 || px >= int(u.image.x) || py >= int(u.image.y)) continue;
    const uint at = uint(py) * u.image.x + uint(px);
    const PtReconstructionSample old = previousSamples[at];
    if (!ptSameSurface(s, old)) continue;
    if (abs(old.positionDepth.w - projectedDepth) > max(footprint, projectedDepth * 0.005f)) continue;
    const float3 delta = xyz(old.positionDepth) - xyz(s.positionDepth);
    if (abs(dot(delta, xyz(s.geometricNormal))) > max(1e-5f, footprint * 0.1f)) continue;
    const float distance = length(delta);
    if (distance < bestDistance) { bestDistance = distance; best = at; found = true; }
  }
  if (!found) return result;
  const PtTemporalHistory h = previous[best];
  if (!ptFiniteColor(xyz(h.diffuse)) || !ptFiniteColor(xyz(h.specular))) return result;
  const float dn = min(h.diffuse.w + 1.0f, 32.0f);
  const float da = 1.0f / max(dn, 1.0f);
  result.diffuse = float4(mix(clamp(xyz(h.diffuse), dmin, dmax), d, da), dn);
  result.moments.x = mix(h.moments.x, dl, da);
  result.moments.y = mix(h.moments.y, dl * dl, da);
  const PtReconstructionSample old = previousSamples[best];
  // Primary-surface reprojection is insufficient for moving mirror reflections, and for
  // what transmissive and mirror-coated primaries show (V7 layer flags).
  const bool stableSpecular = s.normalRoughness.w >= 0.15f && s.geometricNormal.w == 0.0f &&
      old.geometricNormal.w == 0.0f &&
      abs(old.normalRoughness.w - s.normalRoughness.w) < 0.05f &&
      dot(xyz(s.viewDistance), xyz(old.viewDistance)) > 0.9995f &&
      ((s.viewDistance.w < 0.0f && old.viewDistance.w < 0.0f) ||
       (s.viewDistance.w > 0.0f && old.viewDistance.w > 0.0f &&
        abs(s.viewDistance.w - old.viewDistance.w) < 0.1f * max(s.viewDistance.w, old.viewDistance.w)));
  if (stableSpecular) {
    const float pn = min(h.specular.w + 1.0f, 8.0f), pa = 1.0f / max(pn, 1.0f);
    result.specular = float4(mix(clamp(xyz(h.specular), pmin, pmax), p, pa), pn);
    result.moments.z = mix(h.moments.z, pl, pa);
    result.moments.w = mix(h.moments.w, pl * pl, pa);
  }
  return result;
}

inline PtTemporalHistory ptAtrousPixel(PtTemporalUniforms u, uint x, uint y,
    device const PtReconstructionSample *samples, device const PtTemporalHistory *input) {
  const uint index = y * u.image.x + x;
  const PtReconstructionSample s = samples[index];
  const PtTemporalHistory center = input[index];
  if (s.identity.z == 0u || s.identity.z == 2u) return center;
  const float dl = ptTemporalLuminance(xyz(center.diffuse)), pl = ptTemporalLuminance(xyz(center.specular));
  const float dv = max(0.0f, center.moments.y - center.moments.x * center.moments.x);
  const float pv = max(0.0f, center.moments.w - center.moments.z * center.moments.z);
  const float ds = 4.0f * sqrt(dv) + 0.05f * dl + 0.001f;
  const float ps = 4.0f * sqrt(pv) + 0.05f * pl + 0.001f;
  const float footprint = ptPixelFootprint(u, s.positionDepth.w) * float(u.image.w);
  float3 dsum = float3(0.0f), psum = float3(0.0f);
  float dw = 0.0f, pw = 0.0f;
  for (int oy = -1; oy <= 1; ++oy) for (int ox = -1; ox <= 1; ++ox) {
    const int nx = int(x) + ox * int(u.image.w), ny = int(y) + oy * int(u.image.w);
    if (nx < 0 || ny < 0 || nx >= int(u.image.x) || ny >= int(u.image.y)) continue;
    const uint at = uint(ny) * u.image.x + uint(nx);
    const PtReconstructionSample n = samples[at];
    if (!ptSameSurface(s, n)) continue;
    const float3 delta = xyz(n.positionDepth) - xyz(s.positionDepth);
    if (abs(dot(delta, xyz(s.geometricNormal))) > max(1e-5f, footprint * 0.2f)) continue;
    const PtTemporalHistory value = input[at];
    float weight = (ox == 0 ? 2.0f : 1.0f) * (oy == 0 ? 2.0f : 1.0f);
    weight = weight * exp(-length(xyz(s.albedo) - xyz(n.albedo)) * 8.0f);
    const float wd = weight * exp(-abs(ptTemporalLuminance(xyz(value.diffuse)) - dl) / ds);
    float wp = weight * exp(-abs(ptTemporalLuminance(xyz(value.specular)) - pl) / ps);
    wp = wp * exp(-abs(s.normalRoughness.w - n.normalRoughness.w) * 32.0f);
    if (s.normalRoughness.w < 0.15f && (ox != 0 || oy != 0)) wp = 0.0f;
    dsum = dsum + xyz(value.diffuse) * wd; dw = dw + wd;
    psum = psum + xyz(value.specular) * wp; pw = pw + wp;
  }
  PtTemporalHistory result = center;
  result.diffuse = float4(dsum / max(dw, 1e-20f), center.diffuse.w);
  result.specular = float4(psum / max(pw, 1e-20f), center.specular.w);
  return result;
}
