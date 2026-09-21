#pragma once
#include "msl_prelude.h"
// Independently authored declarations for the supported Metal math surface
// (MSL chapter 6). Implemented functions lower to typed IR; the rest are
// declared so that sources parse and the compiler reports the missing
// family by name. No Apple headers or code are included; no body here.
namespace metal {
#define M2V_INTRINSIC(NAME) __attribute__((annotate("msl.math:" NAME)))
#define M2V_EXTREMA(T) \
  T min(T x, T y) M2V_INTRINSIC("min"); \
  T max(T x, T y) M2V_INTRINSIC("max"); \
  T clamp(T x, T lower, T upper) M2V_INTRINSIC("clamp");
#define M2V_FLOAT_MATH(T) \
  T fabs(T x) M2V_INTRINSIC("fabs"); \
  T abs(T x) M2V_INTRINSIC("abs"); \
  T sqrt(T x) M2V_INTRINSIC("sqrt"); \
  M2V_EXTREMA(T)
#define M2V_INT_MATH(T) \
  T abs(T x) M2V_INTRINSIC("abs"); \
  M2V_EXTREMA(T)
#define M2V_VECTOR_MATH(T, S) \
  T min(T x, S y) M2V_INTRINSIC("min"); \
  T max(T x, S y) M2V_INTRINSIC("max"); \
  T clamp(T x, S lower, S upper) M2V_INTRINSIC("clamp");
// Declared but not implemented families (V3): each call names its function.
#define M2V_FLOAT_FAMILY(T) \
  T acos(T x) M2V_INTRINSIC("acos"); \
  T acosh(T x) M2V_INTRINSIC("acosh"); \
  T asin(T x) M2V_INTRINSIC("asin"); \
  T asinh(T x) M2V_INTRINSIC("asinh"); \
  T atan(T x) M2V_INTRINSIC("atan"); \
  T atanh(T x) M2V_INTRINSIC("atanh"); \
  T ceil(T x) M2V_INTRINSIC("ceil"); \
  T cos(T x) M2V_INTRINSIC("cos"); \
  T cosh(T x) M2V_INTRINSIC("cosh"); \
  T cospi(T x) M2V_INTRINSIC("cospi"); \
  T exp(T x) M2V_INTRINSIC("exp"); \
  T exp2(T x) M2V_INTRINSIC("exp2"); \
  T exp10(T x) M2V_INTRINSIC("exp10"); \
  T floor(T x) M2V_INTRINSIC("floor"); \
  T fract(T x) M2V_INTRINSIC("fract"); \
  T log(T x) M2V_INTRINSIC("log"); \
  T log2(T x) M2V_INTRINSIC("log2"); \
  T log10(T x) M2V_INTRINSIC("log10"); \
  T rint(T x) M2V_INTRINSIC("rint"); \
  T round(T x) M2V_INTRINSIC("round"); \
  T rsqrt(T x) M2V_INTRINSIC("rsqrt"); \
  T sin(T x) M2V_INTRINSIC("sin"); \
  T sinh(T x) M2V_INTRINSIC("sinh"); \
  T sinpi(T x) M2V_INTRINSIC("sinpi"); \
  T tan(T x) M2V_INTRINSIC("tan"); \
  T tanh(T x) M2V_INTRINSIC("tanh"); \
  T tanpi(T x) M2V_INTRINSIC("tanpi"); \
  T trunc(T x) M2V_INTRINSIC("trunc"); \
  T sign(T x) M2V_INTRINSIC("sign"); \
  T saturate(T x) M2V_INTRINSIC("saturate"); \
  T atan2(T x, T y) M2V_INTRINSIC("atan2"); \
  T copysign(T x, T y) M2V_INTRINSIC("copysign"); \
  T fdim(T x, T y) M2V_INTRINSIC("fdim"); \
  T fmax(T x, T y) M2V_INTRINSIC("fmax"); \
  T fmin(T x, T y) M2V_INTRINSIC("fmin"); \
  T fmod(T x, T y) M2V_INTRINSIC("fmod"); \
  T nextafter(T x, T y) M2V_INTRINSIC("nextafter"); \
  T pow(T x, T y) M2V_INTRINSIC("pow"); \
  T powr(T x, T y) M2V_INTRINSIC("powr"); \
  T divide(T x, T y) M2V_INTRINSIC("divide"); \
  T step(T x, T y) M2V_INTRINSIC("step"); \
  T fma(T x, T y, T z) M2V_INTRINSIC("fma"); \
  T mix(T x, T y, T z) M2V_INTRINSIC("mix"); \
  T smoothstep(T x, T y, T z) M2V_INTRINSIC("smoothstep");
#define M2V_FLOAT_FAMILY_MIXED(T, S, E) \
  T ldexp(T x, S exponent) M2V_INTRINSIC("ldexp"); \
  T frexp(T x, thread S &exponent) M2V_INTRINSIC("frexp"); \
  T modf(T x, thread T &integral) M2V_INTRINSIC("modf"); \
  T sincos(T x, thread T &cosval) M2V_INTRINSIC("sincos"); \
  T mix(T x, T y, E a) M2V_INTRINSIC("mix"); \
  T step(E edge, T x) M2V_INTRINSIC("step"); \
  T smoothstep(E edge0, E edge1, T x) M2V_INTRINSIC("smoothstep");
M2V_FLOAT_MATH(float)
M2V_FLOAT_FAMILY(float)
M2V_FLOAT_MATH(float2)
M2V_FLOAT_FAMILY(float2)
M2V_FLOAT_MATH(float3)
M2V_FLOAT_FAMILY(float3)
M2V_FLOAT_MATH(float4)
M2V_FLOAT_FAMILY(float4)
M2V_INT_MATH(int)
M2V_INT_MATH(int2)
M2V_INT_MATH(int3)
M2V_INT_MATH(int4)
M2V_INT_MATH(uint)
M2V_INT_MATH(uint2)
M2V_INT_MATH(uint3)
M2V_INT_MATH(uint4)
// The same extrema for every other integer width (MSL section 6.2).
#define M2V_INT_MATH_WIDTH(S, U)   M2V_INT_MATH(S) M2V_INT_MATH(S##2) M2V_INT_MATH(S##3) M2V_INT_MATH(S##4)   M2V_INT_MATH(U) M2V_INT_MATH(U##2) M2V_INT_MATH(U##3) M2V_INT_MATH(U##4)   M2V_VECTOR_MATH(S##2, S) M2V_VECTOR_MATH(S##3, S) M2V_VECTOR_MATH(S##4, S)   M2V_VECTOR_MATH(U##2, U) M2V_VECTOR_MATH(U##3, U) M2V_VECTOR_MATH(U##4, U)
M2V_INT_MATH_WIDTH(char, uchar)
M2V_INT_MATH_WIDTH(short, ushort)
M2V_INT_MATH_WIDTH(long, ulong)
#undef M2V_INT_MATH_WIDTH
M2V_VECTOR_MATH(float2, float)
M2V_FLOAT_FAMILY_MIXED(float2, int2, float)
M2V_VECTOR_MATH(float3, float)
M2V_FLOAT_FAMILY_MIXED(float3, int3, float)
M2V_VECTOR_MATH(float4, float)
M2V_FLOAT_FAMILY_MIXED(float4, int4, float)
M2V_FLOAT_FAMILY_MIXED(float, int, float)
// half: the same surface (MSL section 6.5 lists every function for half).
M2V_FLOAT_MATH(half)
M2V_FLOAT_FAMILY(half)
M2V_FLOAT_MATH(half2)
M2V_FLOAT_FAMILY(half2)
M2V_FLOAT_MATH(half3)
M2V_FLOAT_FAMILY(half3)
M2V_FLOAT_MATH(half4)
M2V_FLOAT_FAMILY(half4)
M2V_VECTOR_MATH(half2, half)
M2V_FLOAT_FAMILY_MIXED(half2, int2, half)
M2V_VECTOR_MATH(half3, half)
M2V_FLOAT_FAMILY_MIXED(half3, int3, half)
M2V_VECTOR_MATH(half4, half)
M2V_FLOAT_FAMILY_MIXED(half4, int4, half)
M2V_FLOAT_FAMILY_MIXED(half, int, half)
M2V_VECTOR_MATH(int2, int)
M2V_VECTOR_MATH(int3, int)
M2V_VECTOR_MATH(int4, int)
M2V_VECTOR_MATH(uint2, uint)
M2V_VECTOR_MATH(uint3, uint)
M2V_VECTOR_MATH(uint4, uint)
// fast:: and precise:: name the same functions; V3 records the chosen precision per call.
namespace fast {
#define M2V_FAST(T) M2V_FLOAT_MATH(T) M2V_FLOAT_FAMILY(T)
M2V_FAST(float)
M2V_FAST(float2)
M2V_FAST(float3)
M2V_FAST(float4)
M2V_FAST(half)
M2V_FAST(half2)
M2V_FAST(half3)
M2V_FAST(half4)
#undef M2V_FAST
}
namespace precise {
#define M2V_PRECISE(T) M2V_FLOAT_MATH(T) M2V_FLOAT_FAMILY(T)
M2V_PRECISE(half)
M2V_PRECISE(half2)
M2V_PRECISE(half3)
M2V_PRECISE(half4)
M2V_PRECISE(float)
M2V_PRECISE(float2)
M2V_PRECISE(float3)
M2V_PRECISE(float4)
#undef M2V_PRECISE
}
#undef M2V_FLOAT_FAMILY_MIXED
#undef M2V_FLOAT_FAMILY
#undef M2V_VECTOR_MATH
#undef M2V_INT_MATH
#undef M2V_FLOAT_MATH
#undef M2V_EXTREMA
#undef M2V_INTRINSIC
}
