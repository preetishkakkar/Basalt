// Ray-cone oracle: the shared level-of-detail functions of pt/raycone.h on a
// host-supplied corpus of hits, which the host compares bitwise with the same header compiled
// for the CPU.
#include "shared/prelude.h"
#include "pt/raycone.h"

// One hit: the instance's object-to-world rows; object-space corners p0..p2 (w: the texture's
// width, height and level count); UV corners t0, t1 (uv01) and t2 with the path's cone
// (uv2Cone); the unit ray direction and the hit distance; the lobe alpha and a log2 probe.
struct RayConeCase {
  float4 row0;
  float4 row1;
  float4 row2;
  float4 p0;
  float4 p1;
  float4 p2;
  float4 uv01;
  float4 uv2Cone;
  float4 directionT;
  float4 misc;
};

// lod: the cone width at the hit, lodBase, the clamped level and ptLog2 of the probe.
// cones: the scattered cone and the shadow cone.
struct RayConeResult {
  float4 lod;
  float4 cones;
};

kernel void raycone_oracle(const device RayConeCase* cases [[buffer(0)]],
                           device RayConeResult* results [[buffer(1)]],
                           constant uint4& control [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
  if (id >= control.x) return;
  const RayConeCase c = cases[id];
  const float2 cone = c.uv2Cone.zw;
  const float width = ptConeWidthOrLevelZero(cone, c.directionT.w);
  const float lodBase = ptConeLodBase(ptUvCross(c.uv01.xy, c.uv01.zw, c.uv2Cone.xy),
                                      ptWorldCross(c.row0, c.row1, c.row2, c.p0.xyz, c.p1.xyz, c.p2.xyz),
                                      c.directionT.xyz, width);
  RayConeResult r;
  r.lod = float4(width, lodBase, ptTextureLevel(lodBase, c.p0.w, c.p1.w, c.p2.w), ptLog2(c.misc.y));
  r.cones = float4(ptConeScattered(cone, width, c.misc.x), ptConeShadow(cone, width));
  results[id] = r;
}
