#pragma once
#include "msl_prelude.h"
// Independently authored geometric declarations (MSL section 6.7); dot,
// each lowers per the specification's formula (see docs/NUMERICS.md).
namespace metal {
#define M2V_INTRINSIC(NAME) __attribute__((annotate("msl.math:" NAME)))
#define M2V_GEOMETRIC(T, S) \
  S dot(T x, T y) M2V_INTRINSIC("dot"); \
  S length(T x) M2V_INTRINSIC("length"); \
  T normalize(T x) M2V_INTRINSIC("normalize"); \
  S distance(T x, T y) M2V_INTRINSIC("distance"); \
  S distance_squared(T x, T y) M2V_INTRINSIC("distance_squared"); \
  S length_squared(T x) M2V_INTRINSIC("length_squared"); \
  T faceforward(T n, T i, T nref) M2V_INTRINSIC("faceforward"); \
  T reflect(T i, T n) M2V_INTRINSIC("reflect"); \
  T refract(T i, T n, S eta) M2V_INTRINSIC("refract");
M2V_GEOMETRIC(float2, float)
M2V_GEOMETRIC(float3, float)
M2V_GEOMETRIC(float4, float)
M2V_GEOMETRIC(half2, half)
M2V_GEOMETRIC(half3, half)
M2V_GEOMETRIC(half4, half)
float3 cross(float3 x, float3 y) M2V_INTRINSIC("cross");
half3 cross(half3 x, half3 y) M2V_INTRINSIC("cross");
#undef M2V_GEOMETRIC
#undef M2V_INTRINSIC
}
