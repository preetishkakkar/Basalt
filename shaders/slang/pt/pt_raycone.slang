// Ray-cone texture level of detail (Akenine-Moller et al. 2021, "Improved shader and texture
// level of detail using ray cones", adapted to a path tracer). The level-of-detail function is
// bitwise identical on the CPU and the GPU, so it uses only correctly rounded operations: a
// polynomial log2 on the float's bits and differences of logarithms instead of divisions.
module pt_raycone;

import pt_path;

// The level-of-detail sentinel for level-zero sampling (texture filter mode 0): sampling then
// reads level 0 exactly.
public static const float kPtLevelZero = -1e30f;

// log2(x) for x > 0 to within 3.1e-6 (clamped below at 1e-30): the exponent from the bits and a
// degree-7 polynomial for the mantissa in [1, 2). Only exactly rounded + and *, so the CPU and
// the GPU produce the same bits.
public float ptLog2(float x) {
  const uint bits = asuint(max(x, 1e-30f));
  const float exponent = float(int((bits >> 23u) & 0xFFu) - 127);
  const float m = asfloat((bits & 0x007FFFFFu) | 0x3F800000u) - 1.0f;
  float q = 0.0202350136f;
  q = q * m - 0.0948813176f;
  q = q * m + 0.213666496f;
  q = q * m - 0.337837152f;
  q = q * m + 0.477224787f;
  q = q * m - 0.721095554f;
  q = q * m + 1.44269078f;
  return exponent + q * m;
}

// dot and cross as explicit products and sums: OpDot and the GLSL Cross instruction are the
// driver's to evaluate (fused or reordered), where separate * and + are exactly rounded.
public float ptExactDot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
public float3 ptExactCross(float3 a, float3 b) {
  return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// A path's cone: x its width where the current ray left, y its spread angle. Filter mode 0
// (level zero) is the sentinel width -1, which every step below preserves.
public float2 ptCameraCone(float4 lens) { return lens.z > 0.5f ? float2(0.0f, lens.w) : float2(-1.0f, 0.0f); }

// The cone's width at distance t along a ray that left with width cone.x and spread angle
// cone.y, |w + beta t|, or -1 (level zero).
public float ptConeWidthOrLevelZero(float2 cone, float t) { return cone.x < 0.0f ? -1.0f : abs(cone.x + cone.y * t); }

// The spread after scattering from a lobe of alpha r^2 (1 for the diffuse lobe): beta + 2 r^2,
// clamped to pi / 2 (no curvature term).
public float ptConeSpreadAfter(float spread, float lobeAlpha) { return min(spread + 2.0f * lobeAlpha, kPi * 0.5f); }

// The cone of the ray scattered at a hit of width `width` from a lobe of alpha r^2.
public float2 ptConeScattered(float2 cone, float width, float lobeAlpha) {
  return cone.x < 0.0f ? cone : float2(width, ptConeSpreadAfter(cone.y, lobeAlpha));
}

// The cone of a shadow ray leaving a hit of width `width`: the path's spread, unwidened (a light
// sample is not a lobe).
public float2 ptConeShadow(float2 cone, float width) { return cone.x < 0.0f ? cone : float2(width, cone.y); }

// The level of detail at a hit without the texture's own size:
//   0.5 log2(uvArea / worldArea) + log2(w / max(|n . d|, 1e-4)),
// from the triangle's UV-space cross product uvCross (the signed parallelogram area in UV
// units), its world-space edge cross product worldCross (whose length is the parallelogram
// area and whose direction is the geometric normal), the unit ray direction and the cone width
// at the hit. Written with squared magnitudes and logarithm differences only, so no sqrt or
// division enters: with |c|^2 = c.c,
//   0.25 log2(uvCross^2) - 0.25 log2(|c|^2) + log2(w) - max(log2|c.d| - 0.5 log2(|c|^2), log2 1e-4).
// A negative width selects level zero (filter mode 0).
public float ptConeLodBase(float uvCross, float3 worldCross, float3 direction, float coneWidth) {
  if (coneWidth < 0.0f) return kPtLevelZero;
  const float worldSquared = ptLog2(ptExactDot(worldCross, worldCross));
  const float cosine = max(ptLog2(abs(ptExactDot(worldCross, direction))) - 0.5f * worldSquared, -13.2877124f);
  return 0.25f * ptLog2(uvCross * uvCross) - 0.25f * worldSquared + ptLog2(coneWidth) - cosine;
}

// The UV-space and world-space cross products of a triangle for ptConeLodBase: object-space
// corners p0..p2 under the instance's object-to-world rows, UV corners t0..t2.
public float ptUvCross(float2 t0, float2 t1, float2 t2) {
  return (t1.x - t0.x) * (t2.y - t0.y) - (t2.x - t0.x) * (t1.y - t0.y);
}
public float3 ptWorldCross(float4 row0, float4 row1, float4 row2, float3 p0, float3 p1, float3 p2) {
  const float3 e1 = p1 - p0, e2 = p2 - p0;
  return ptExactCross(float3(ptExactDot(row0.xyz, e1), ptExactDot(row1.xyz, e1), ptExactDot(row2.xyz, e1)),
                      float3(ptExactDot(row0.xyz, e2), ptExactDot(row1.xyz, e2), ptExactDot(row2.xyz, e2)));
}

// The mip level for a texture of width x height texels with `levels` levels: lodBase +
// 0.5 log2(width height), clamped to [0, levels - 1].
public float ptTextureLevel(float lodBase, float width, float height, float levels) {
  return clamp(lodBase + 0.5f * (ptLog2(width) + ptLog2(height)), 0.0f, max(levels - 1.0f, 0.0f));
}

// The footprint's width in texels, capped at 1 (2 to the unclamped level); 0 at level zero.
// The width of the masked alpha test's stochastic cutoff.
public float ptTextureFootprint(float lodBase, float width, float height) {
  if (lodBase <= kPtLevelZero) return 0.0f;
  return exp2(min(lodBase + 0.5f * (ptLog2(width) + ptLog2(height)), 0.0f));
}
