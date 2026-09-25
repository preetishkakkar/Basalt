// Light sampling: the environment through its luminance distribution, the analytic sun as a
// cone, punctual lights with the rasteriser's falloff and spot cones, and emissive triangles.
module pt_lights;

import pt_path;
import pt_raycone;

// The environment distribution, a piecewise-constant density over the equirect's (u, v),
// built on the host as one float buffer:
//   function          rows x columns   luminance times sin(theta), per cell
//   conditional CDFs  rows x (columns + 1)
//   marginal function rows             each row's integral
//   marginal CDF      rows + 1
// info = (columns, rows, integral, present), from PathUniforms.distribution.

// The largest i in [0, size - 2] with data[offset + i] <= value.
public uint ptFindInterval(StructuredBuffer<float> data, uint offset, uint size, float value) {
  uint low = 0u;
  uint high = size - 1u;
  while (low + 1u < high) {
    const uint middle = (low + high) / 2u;
    if (data[offset + middle] <= value) low = middle;
    else high = middle;
  }
  return low;
}

public float3 ptEquirectangularDirection(float2 uv) {
  const float theta = uv.y * kPi;
  const float phi = (uv.x - 0.5f) * 2.0f * kPi;
  const float sinTheta = sin(theta);
  return float3(sinTheta * cos(phi), cos(theta), sinTheta * sin(phi));
}

// Solid-angle pdf of a direction under the distribution.
public float ptEnvironmentPdf(StructuredBuffer<float> distribution, float4 info, float3 direction) {
  const uint columns = uint(info.x);
  const uint rows = uint(info.y);
  const float2 uv = equirectangularUV(direction);
  const uint column = min(uint(uv.x * float(columns)), columns - 1u);
  const uint row = min(uint(uv.y * float(rows)), rows - 1u);
  const float sinTheta = sqrt(max(0.0f, 1.0f - direction.y * direction.y));
  float pdf = 0.0f;
  // 2 pi^2 sin(theta): the equirect map's Jacobian from (u, v) to solid angle.
  if (sinTheta > 1e-6f && info.z > 0.0f) pdf = distribution[row * columns + column] / info.z / (2.0f * kPi * kPi * sinTheta);
  return pdf;
}

public float3 ptSampleEnvironmentDirection(StructuredBuffer<float> distribution, float4 info, float2 xi, out float pdf) {
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
  if (sinTheta > 1e-6f && info.z > 0.0f) pdf = distribution[row * columns + column] / info.z / (2.0f * kPi * kPi * sinTheta);
  return direction;
}

// A direction uniform in solid angle within a cone; its pdf is one over the cone's solid angle.
public float3 ptSampleCone(float3 axis, float cosineMax, float2 xi) {
  const float cosine = 1.0f - xi.x * (1.0f - cosineMax);
  const float sine = sqrt(max(0.0f, 1.0f - cosine * cosine));
  const float angle = 2.0f * kPi * xi.y;
  float3 tangent, bitangent;
  orthonormalBasis(axis, tangent, bitangent);
  return normalize(tangent * (sine * cos(angle)) + bitangent * (sine * sin(angle)) + axis * cosine);
}

// The radiance a punctual light sends to a point, before the BSDF: intensity over distance
// squared inside the range window, times the spot cone, as the rasteriser computes it.
public float3 ptPunctualLight(Light light, float3 position, out float3 direction, out float distance) {
  const float3 toLight = light.position.xyz - position;
  distance = length(toLight);
  direction = float3(0.0f, 1.0f, 0.0f);
  float3 result = float3(0.0f);
  if (distance > 1e-4f) {
    direction = toLight / distance;
    float attenuation = distanceAttenuation(distance, light.position.w);
    if (light.cone.y > 0.5f) {
      const float cosine = dot(normalize(light.direction.xyz), -direction);
      const float t = saturate((cosine - light.cone.x) / max(light.direction.w - light.cone.x, 1e-4f));
      attenuation = attenuation * t * t;
    }
    result = light.color.xyz * (light.color.w * attenuation);
  }
  return result;
}

// The scene's emitting triangles, in strictly increasing (instance, primitive) order, with the
// selection CDF.
public struct PtEmitters {
  StructuredBuffer<PtEmissiveTriangle> triangles;
  uint count;

  // The emitter a uniform number selects: the first whose inclusive CDF reaches it, the last
  // if none does. A lower-bound binary search over the CDF, O(log n); it returns exactly the
  // index a linear walk would.
  uint select(float xi) {
    uint low = 0u;
    uint high = count > 0u ? count - 1u : 0u;
    while (low < high) {
      const uint middle = (low + high) / 2u;
      if (xi > triangles[middle].edge2Cdf.w) low = middle + 1u;
      else high = middle;
    }
    return low;
  }

  // The list entry of an emitting triangle, or count if it has none: a binary search, O(log n).
  uint find(uint instance, uint primitive) {
    uint low = 0u;
    uint high = count;
    while (low < high) {
      const uint middle = (low + high) / 2u;
      const uint4 key = triangles[middle].identity;
      if (key.x < instance || (key.x == instance && key.y < primitive)) low = middle + 1u;
      else high = middle;
    }
    if (low < count && triangles[low].identity.x == instance && triangles[low].identity.y == primitive) return low;
    return count;
  }

  // A point on an emitter towards `position`: the radiance it sends, its direction and distance,
  // and its solid-angle pdf (zero when unusable). cone is the one a BSDF-sampled path would
  // carry there, so both MIS strategies read the emitter's texture alike.
  float3 sample<M : IPtMaps>(PtGeometry geometry, M maps, float3 position, float3 xi, float2 cone, out float3 direction,
                                    out float reach, out float pdf) {
    const PtEmissiveTriangle emitter = triangles[select(xi.x)];
    // Uniform on the triangle: the square root spreads samples by area.
    const float baryX = sqrt(xi.y) * (1.0f - xi.z);
    const float baryY = sqrt(xi.y) * xi.z;
    direction = emitter.v0Area.xyz + emitter.edge1Probability.xyz * baryX + emitter.edge2Cdf.xyz * baryY - position;
    reach = length(direction);
    if (reach > 0.0f) direction = direction / reach;
    pdf = -dot(emitter.normal.xyz, direction);
    const TraceInstance info = geometry.traceInstances[emitter.identity.x];
    if ((info.flags & kInstanceDoubleSided) != 0u) pdf = abs(pdf);
    if (reach <= 1e-5f || pdf <= 1e-6f) {
      pdf = 0.0f;
      return float3(0.0f);
    }
    pdf = emitter.edge1Probability.w * reach * reach / (emitter.v0Area.w * pdf);
    // The emitter's texture at the shadow ray's cone where it arrives, as a BSDF-sampled ray
    // hitting the same point reads it; level-zero mode skips the arithmetic.
    float lodBase = kPtLevelZero;
    if (cone.x >= 0.0f)
      lodBase = ptConeLodBase(ptUvCross(emitter.uv01.xy, emitter.uv01.zw, emitter.uv2.xy),
                              ptExactCross(emitter.edge1Probability.xyz, emitter.edge2Cdf.xyz), direction,
                              ptConeWidthOrLevelZero(cone, reach));
    const Material material = geometry.materials[info.material];
    return maps.sampleTexture((info.slots >> 16u) & 0xFFu,
                              emitter.uv01.xy * (1.0f - baryX - baryY) + emitter.uv01.zw * baryX + emitter.uv2.xy * baryY,
                              (material.texture.y >> 16u) & 0xFFu, lodBase).xyz *
           material.emissive.xyz * material.emissive.w;
  }

  // The solid-angle pdf this light sampler has for a path that hit an emitter.
  float pdf(PtGeometry geometry, PtHit hit, float3 direction) {
    const uint i = find(hit.instance, hit.primitive);
    if (i < count) {
      const PtEmissiveTriangle emitter = triangles[i];
      float cosine = -dot(emitter.normal.xyz, direction);
      if ((geometry.traceInstances[hit.instance].flags & kInstanceDoubleSided) != 0u) cosine = abs(cosine);
      if (cosine <= 1e-6f || hit.t <= 0.0f) return 0.0f;
      return emitter.edge1Probability.w * hit.t * hit.t / (emitter.v0Area.w * cosine);
    }
    return 0.0f;
  }
};
