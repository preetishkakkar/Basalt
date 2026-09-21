#pragma once
#include "msl_prelude.h"
// Independently authored integer function declarations (MSL section 6.2) for
// every integer width; absdiff returns the unsigned type, mul24/mad24 are 32-bit.
namespace metal {
#define M2V_INTRINSIC(NAME) __attribute__((annotate("msl.math:" NAME)))
#define M2V_INTEGER(T, U) \
  U absdiff(T x, T y) M2V_INTRINSIC("absdiff"); \
  T addsat(T x, T y) M2V_INTRINSIC("addsat"); \
  T subsat(T x, T y) M2V_INTRINSIC("subsat"); \
  T clz(T x) M2V_INTRINSIC("clz"); \
  T ctz(T x) M2V_INTRINSIC("ctz"); \
  T hadd(T x, T y) M2V_INTRINSIC("hadd"); \
  T rhadd(T x, T y) M2V_INTRINSIC("rhadd"); \
  T mulhi(T x, T y) M2V_INTRINSIC("mulhi"); \
  T madhi(T x, T y, T z) M2V_INTRINSIC("madhi"); \
  T madsat(T x, T y, T z) M2V_INTRINSIC("madsat"); \
  T popcount(T x) M2V_INTRINSIC("popcount"); \
  T reverse_bits(T x) M2V_INTRINSIC("reverse_bits"); \
  T rotate(T x, T y) M2V_INTRINSIC("rotate"); \
  T extract_bits(T x, uint offset, uint bits) M2V_INTRINSIC("extract_bits"); \
  T insert_bits(T base, T insert, uint offset, uint bits) M2V_INTRINSIC("insert_bits");
#define M2V_INTEGER_WIDTH(S, U) \
  M2V_INTEGER(S, U) M2V_INTEGER(S##2, U##2) M2V_INTEGER(S##3, U##3) M2V_INTEGER(S##4, U##4) \
  M2V_INTEGER(U, U) M2V_INTEGER(U##2, U##2) M2V_INTEGER(U##3, U##3) M2V_INTEGER(U##4, U##4)
M2V_INTEGER_WIDTH(char, uchar)
M2V_INTEGER_WIDTH(short, ushort)
M2V_INTEGER_WIDTH(int, uint)
M2V_INTEGER_WIDTH(long, ulong)
// Metal 4.1 bit interleaving over unsigned pairs (uchar/ushort, ushort/uint, uint/ulong).
#define M2V_INTERLEAVE(N, W) \
  W interleave(N even, N odd) M2V_INTRINSIC("interleave"); \
  W interleave(N##2 v) M2V_INTRINSIC("interleave"); \
  N##2 deinterleave(W v) M2V_INTRINSIC("deinterleave");
M2V_INTERLEAVE(uchar, ushort)
M2V_INTERLEAVE(ushort, uint)
M2V_INTERLEAVE(uint, ulong)
#undef M2V_INTERLEAVE
#define M2V_INTEGER_24(T) \
  T mul24(T x, T y) M2V_INTRINSIC("mul24"); \
  T mad24(T x, T y, T z) M2V_INTRINSIC("mad24");
M2V_INTEGER_24(int)
M2V_INTEGER_24(int2)
M2V_INTEGER_24(int3)
M2V_INTEGER_24(int4)
M2V_INTEGER_24(uint)
M2V_INTEGER_24(uint2)
M2V_INTEGER_24(uint3)
M2V_INTEGER_24(uint4)
#undef M2V_INTEGER_24
#undef M2V_INTEGER_WIDTH
#undef M2V_INTEGER
#undef M2V_INTRINSIC
}
