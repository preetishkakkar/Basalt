// Light sampling: the environment through its luminance distribution, the analytic sun as
// a cone, and punctual lights with the rasteriser's falloff and spot cones.
#pragma once
#include "path.h"
#include "raycone.h"

// The environment distribution, a piecewise-constant density over the equirect's (u, v),
// built on the host (src/pt/EnvironmentDistribution.cpp) as one float buffer:
//   function          rows x columns   luminance times sin(theta), per cell
//   conditional CDFs  rows x (columns + 1)
//   marginal function rows             each row's integral
//   marginal CDF      rows + 1
// info = (columns, rows, integral, present), from PathUniforms.distribution.

// The largest i in [0, size - 2] with data[offset + i] <= value.
inline uint ptFindInterval(device const float *data, uint offset, uint size, float value) {
  uint low = 0u;
  uint high = size - 1u;
  while (low + 1u < high) {
    const uint middle = (low + high) / 2u;
    if (data[offset + middle] <= value) low = middle;
    else high = middle;
  }
  return low;
}

inline float3 ptEquirectangularDirection(float2 uv) {
  const float theta = uv.y * kPi;
  const float phi = (uv.x - 0.5f) * 2.0f * kPi;
  const float sinTheta = sin(theta);
  return float3(sinTheta * cos(phi), cos(theta), sinTheta * sin(phi));
}

// Solid-angle pdf of a direction under the distribution.
inline float ptEnvironmentPdf(device const float *distribution, float4 info, float3 direction) {
  const uint columns = uint(info.x);
  const uint rows = uint(info.y);
  const float2 uv = equirectangularUV(direction);
  const uint column = min(uint(uv.x * float(columns)), columns - 1u);
  const uint row = min(uint(uv.y * float(rows)), rows - 1u);
  const float sinTheta = sqrt(max(0.0f, 1.0f - direction.y * direction.y));
  float pdf = 0.0f;
  if (sinTheta > 1e-6f && info.z > 0.0f)
    pdf = distribution[row * columns + column] / info.z / (2.0f * kPi * kPi * sinTheta);
  return pdf;
}

inline float3 ptSampleEnvironmentDirection(device const float *distribution, float4 info, float2 xi,
                                           thread float &pdf) {
  const uint columns = uint(info.x);
  const uint rows = uint(info.y);
  const uint conditionalOffset = rows * columns;
  const uint marginalOffset = conditionalOffset + rows * (columns + 1u);
  const uint marginalCdfOffset = marginalOffset + rows;

  const uint row = ptFindInterval(distribution, marginalCdfOffset, rows + 1u, xi.y);
  const float rowLow = distribution[marginalCdfOffset + row];
  const float rowHigh = distribution[marginalCdfOffset + row + 1u];
  float dv = 0.0f;
  if (rowHigh > rowLow) dv = (xi.y - rowLow) / (rowHigh - rowLow);

  const uint rowCdf = conditionalOffset + row * (columns + 1u);
  const uint column = ptFindInterval(distribution, rowCdf, columns + 1u, xi.x);
  const float columnLow = distribution[rowCdf + column];
  const float columnHigh = distribution[rowCdf + column + 1u];
  float du = 0.0f;
  if (columnHigh > columnLow) du = (xi.x - columnLow) / (columnHigh - columnLow);

  const float2 uv = float2((float(column) + du) / float(columns), (float(row) + dv) / float(rows));
  const float3 direction = ptEquirectangularDirection(uv);
  const float sinTheta = sin(uv.y * kPi);
  pdf = 0.0f;
  if (sinTheta > 1e-6f && info.z > 0.0f)
    pdf = distribution[row * columns + column] / info.z / (2.0f * kPi * kPi * sinTheta);
  return direction;
}

// A direction uniform in solid angle within a cone; its pdf is one over the cone's solid angle.
inline float3 ptSampleCone(float3 axis, float cosineMax, float2 xi) {
  const float cosine = 1.0f - xi.x * (1.0f - cosineMax);
  const float sine = sqrt(max(0.0f, 1.0f - cosine * cosine));
  const float angle = 2.0f * kPi * xi.y;
  float3 tangent = float3(0.0f);
  float3 bitangent = float3(0.0f);
  orthonormalBasis(axis, tangent, bitangent);
  return normalize(tangent * (sine * cos(angle)) + bitangent * (sine * sin(angle)) + axis * cosine);
}

// The radiance a punctual light sends to a point, before the BSDF: intensity over distance
// squared inside the range window, times the spot cone, as the rasteriser computes it.
inline float3 ptPunctualLight(Light light, float3 position, thread float3 &direction, thread float &distance) {
  const float3 toLight = xyz(light.position) - position;
  distance = length(toLight);
  direction = float3(0.0f, 1.0f, 0.0f);
  float3 result = float3(0.0f);
  if (distance > 1e-4f) {
    direction = toLight / distance;
    float attenuation = distanceAttenuation(distance, light.position.w);
    if (light.cone.y > 0.5f) {
      const float cosine = dot(normalize(xyz(light.direction)), -direction);
      const float t = saturate((cosine - light.cone.x) / max(light.direction.w - light.cone.x, 1e-4f));
      attenuation = attenuation * t * t;
    }
    result = xyz(light.color) * (light.color.w * attenuation);
  }
  return result;
}

#ifndef PT_DISABLE_EMISSIVE
// The emitter a uniform number selects: the first whose inclusive CDF reaches it, the last
// if none does. A lower-bound binary search over the CDF, O(log n); it returns exactly the
// index a linear walk would.
inline uint ptEmissiveSelect(device const PtEmissiveTriangle *emissiveTriangles, uint count, float xi) {
  uint low = 0u;
  uint high = count > 0u ? count - 1u : 0u;
  while (low < high) {
    const uint middle = (low + high) / 2u;
    if (xi > emissiveTriangles[middle].edge2Cdf.w) low = middle + 1u;
    else high = middle;
  }
  return low;
}

// The list entry of an emitting triangle, or count if it has none. The host builds the list
// in strictly increasing (instance, primitive) order, so this is a binary search, O(log n).
inline uint ptEmissiveFind(device const PtEmissiveTriangle *emissiveTriangles, uint count, uint instance,
                           uint primitive) {
  uint low = 0u;
  uint high = count;
  while (low < high) {
    const uint middle = (low + high) / 2u;
    const uint4 key = emissiveTriangles[middle].identity;
    if (key.x < instance || (key.x == instance && key.y < primitive)) low = middle + 1u;
    else high = middle;
  }
  if (low < count && emissiveTriangles[low].identity.x == instance && emissiveTriangles[low].identity.y == primitive)
    return low;
  return count;
}

inline float3 ptSampleEmissiveTriangle(device const PtEmissiveTriangle *emissiveTriangles, uint count,
    device const TraceInstance *traceInstances, device const Material *materials,
    PT_TEXTURE_PARAMS, float3 position, float3 xi, float2 cone, thread float3 &direction,
    thread float &reach, thread float &pdf) {
  const uint selected = ptEmissiveSelect(emissiveTriangles, count, xi.x);
#define PT_EMISSIVE_BARY_X (sqrt(xi.y) * (1.0f - xi.z))
#define PT_EMISSIVE_BARY_Y (sqrt(xi.y) * xi.z)
  direction = xyz(emissiveTriangles[selected].v0Area) +
              xyz(emissiveTriangles[selected].edge1Probability) * PT_EMISSIVE_BARY_X +
              xyz(emissiveTriangles[selected].edge2Cdf) * PT_EMISSIVE_BARY_Y - position;
  reach = length(direction);
  if (reach > 0.0f) direction = direction / reach;
  pdf = -dot(xyz(emissiveTriangles[selected].normal), direction);
  if ((traceInstances[emissiveTriangles[selected].identity.x].flags & kInstanceDoubleSided) != 0u)
    pdf = abs(pdf);
  if (reach <= 1e-5f || pdf <= 1e-6f) {
    pdf = 0.0f;
    return float3(0.0f);
  }
  pdf = emissiveTriangles[selected].edge1Probability.w * reach * reach /
        (emissiveTriangles[selected].v0Area.w * pdf);
  // The emitter's texture at the shadow ray's cone where it arrives, as a BSDF-sampled ray
  // hitting the same point reads it; level-zero mode skips the arithmetic.
  float lodBase = kPtLevelZero;
  if (cone.x >= 0.0f)
    lodBase = ptConeLodBase(ptUvCross(xy(emissiveTriangles[selected].uv01), zw(emissiveTriangles[selected].uv01),
                                      xy(emissiveTriangles[selected].uv2)),
                            ptExactCross(xyz(emissiveTriangles[selected].edge1Probability),
                                         xyz(emissiveTriangles[selected].edge2Cdf)),
                            direction, ptConeWidthOrLevelZero(cone, reach));
  return xyz(ptSampleTexture(PT_TEXTURE_ARGS,
      (traceInstances[emissiveTriangles[selected].identity.x].slots >> 16u) & 0xFFu,
      xy(emissiveTriangles[selected].uv01) * (1.0f - PT_EMISSIVE_BARY_X - PT_EMISSIVE_BARY_Y) +
      zw(emissiveTriangles[selected].uv01) * PT_EMISSIVE_BARY_X +
      xy(emissiveTriangles[selected].uv2) * PT_EMISSIVE_BARY_Y,
      (materials[traceInstances[emissiveTriangles[selected].identity.x].material].texture.y >> 16u) & 0xFFu,
      lodBase)) *
      xyz(materials[traceInstances[emissiveTriangles[selected].identity.x].material].emissive) *
      materials[traceInstances[emissiveTriangles[selected].identity.x].material].emissive.w;
#undef PT_EMISSIVE_BARY_X
#undef PT_EMISSIVE_BARY_Y
}

inline float ptEmissivePdf(device const PtEmissiveTriangle *emissiveTriangles, uint count,
                           device const TraceInstance *traceInstances, PtHit hit, float3 direction) {
  const uint i = ptEmissiveFind(emissiveTriangles, count, hit.instance, hit.primitive);
  if (i < count) {
#define PT_EMISSIVE_COSINE                                                                       \
  (((traceInstances[hit.instance].flags & kInstanceDoubleSided) != 0u)                            \
       ? abs(-dot(xyz(emissiveTriangles[i].normal), direction))                                  \
       : -dot(xyz(emissiveTriangles[i].normal), direction))
    if (PT_EMISSIVE_COSINE <= 1e-6f || hit.t <= 0.0f) return 0.0f;
    return emissiveTriangles[i].edge1Probability.w * hit.t * hit.t /
           (emissiveTriangles[i].v0Area.w * PT_EMISSIVE_COSINE);
#undef PT_EMISSIVE_COSINE
  }
  return 0.0f;
}
#endif
