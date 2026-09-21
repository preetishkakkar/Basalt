#pragma once
#include "msl_prelude.h"
// Independently authored relational declarations (MSL section 6.4): all,
// any and select lower today; classification functions are declared for V3.
namespace metal {
#define M2V_INTRINSIC(NAME) __attribute__((annotate("msl.math:" NAME)))
#define M2V_RELATIONAL(N) \
  bool all(bool##N x) M2V_INTRINSIC("all"); \
  bool any(bool##N x) M2V_INTRINSIC("any");
M2V_RELATIONAL(2)
M2V_RELATIONAL(3)
M2V_RELATIONAL(4)
#define M2V_SELECT(T, B) T select(T a, T b, B c) M2V_INTRINSIC("select");
M2V_SELECT(float, bool) M2V_SELECT(int, bool) M2V_SELECT(uint, bool) M2V_SELECT(bool, bool)
M2V_SELECT(float2, bool2) M2V_SELECT(int2, bool2) M2V_SELECT(uint2, bool2) M2V_SELECT(bool2, bool2)
M2V_SELECT(float3, bool3) M2V_SELECT(int3, bool3) M2V_SELECT(uint3, bool3) M2V_SELECT(bool3, bool3)
M2V_SELECT(float4, bool4) M2V_SELECT(int4, bool4) M2V_SELECT(uint4, bool4) M2V_SELECT(bool4, bool4)
M2V_SELECT(half, bool) M2V_SELECT(half2, bool2) M2V_SELECT(half3, bool3) M2V_SELECT(half4, bool4)
M2V_SELECT(bfloat, bool) M2V_SELECT(bfloat2, bool2) M2V_SELECT(bfloat3, bool3) M2V_SELECT(bfloat4, bool4)
#define M2V_SELECT_WIDTH(S, U)   M2V_SELECT(S, bool) M2V_SELECT(S##2, bool2) M2V_SELECT(S##3, bool3) M2V_SELECT(S##4, bool4)   M2V_SELECT(U, bool) M2V_SELECT(U##2, bool2) M2V_SELECT(U##3, bool3) M2V_SELECT(U##4, bool4)
M2V_SELECT_WIDTH(char, uchar)
M2V_SELECT_WIDTH(short, ushort)
M2V_SELECT_WIDTH(long, ulong)
#undef M2V_SELECT_WIDTH
#define M2V_CLASSIFY(T, B) \
  B isnan(T x) M2V_INTRINSIC("isnan"); \
  B isinf(T x) M2V_INTRINSIC("isinf"); \
  B isfinite(T x) M2V_INTRINSIC("isfinite"); \
  B isnormal(T x) M2V_INTRINSIC("isnormal"); \
  B signbit(T x) M2V_INTRINSIC("signbit"); \
  B isordered(T x, T y) M2V_INTRINSIC("isordered"); \
  B isunordered(T x, T y) M2V_INTRINSIC("isunordered");
M2V_CLASSIFY(float, bool) M2V_CLASSIFY(float2, bool2) M2V_CLASSIFY(float3, bool3) M2V_CLASSIFY(float4, bool4)
M2V_CLASSIFY(half, bool) M2V_CLASSIFY(half2, bool2) M2V_CLASSIFY(half3, bool3) M2V_CLASSIFY(half4, bool4)
M2V_CLASSIFY(bfloat, bool) M2V_CLASSIFY(bfloat2, bool2) M2V_CLASSIFY(bfloat3, bool3) M2V_CLASSIFY(bfloat4, bool4)
template <typename T, typename U> T as_type(U x) M2V_INTRINSIC("as_type");
#undef M2V_CLASSIFY
#undef M2V_SELECT
#undef M2V_RELATIONAL
#undef M2V_INTRINSIC
}
