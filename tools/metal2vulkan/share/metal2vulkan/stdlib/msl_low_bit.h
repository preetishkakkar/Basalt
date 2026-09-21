#pragma once
// Independently authored low-bit numeric formats (Metal 4.1's packed formats; docs/LOW_BIT.md).
// Metal 4.1's spellings live in an SDK this toolchain does not ship (recorded in the registry), so
// the names here are this profile's; the encodings are the Open Compute Project Microscaling (MX)
// specification's: E4M3 (bias 7, no infinities, NaN as 0x7F or 0xFF, largest 448), E5M2 (bias 15,
// IEEE-like with infinities and NaNs, largest 57344), E2M1 (0, 0.5, 1, 1.5, 2, 3, 4, 6), INT4 (two's
// complement -8..7) and INT2 (-2..1). Sub-byte codes pack two (E2M1, INT4) or four (INT2) per byte,
// element 0 in the low bits. Conversions round to nearest even and saturate where the format has no
// infinity; everything is ordinary MSL, compiled like any helper, so every device converts alike.
#include "msl_prelude.h"
#include "msl_math.h"
#include "msl_relational.h"
#include "msl_tensor.h"
#include "msl_tensor_ops.h"
namespace metal {
namespace low_bit {
// A float from sign, unbiased exponent and a 23-bit mantissa field, exact.
inline float __m2v_make_float(uint sign, int exponent, uint mantissa) {
  return as_type<float>((sign << 31) | (uint(exponent + 127) << 23) | mantissa);
}
inline float decode_e4m3(uchar code) {
  uint sign = uint(code) >> 7, exponent = (uint(code) >> 3) & 15u, mantissa = uint(code) & 7u;
  if (exponent == 15u && mantissa == 7u) return as_type<float>(0x7FC00000u | (sign << 31));
  if (exponent == 0u) return (sign ? -1.0f : 1.0f) * float(mantissa) * __m2v_make_float(0u, -9, 0u);
  return __m2v_make_float(sign, int(exponent) - 7, mantissa << 20);
}
inline float decode_e5m2(uchar code) {
  uint sign = uint(code) >> 7, exponent = (uint(code) >> 2) & 31u, mantissa = uint(code) & 3u;
  if (exponent == 31u) return mantissa == 0u ? as_type<float>(0x7F800000u | (sign << 31)) : as_type<float>(0x7FC00000u | (sign << 31));
  if (exponent == 0u) return (sign ? -1.0f : 1.0f) * float(mantissa) * __m2v_make_float(0u, -16, 0u);
  return __m2v_make_float(sign, int(exponent) - 15, mantissa << 21);
}
inline float decode_e2m1(uchar nibble) {
  uint sign = (uint(nibble) >> 3) & 1u, exponent = (uint(nibble) >> 1) & 3u, mantissa = uint(nibble) & 1u;
  float magnitude = exponent == 0u ? 0.5f * float(mantissa) : __m2v_make_float(0u, int(exponent) - 1, mantissa << 22);
  return sign ? -magnitude : magnitude;
}
inline int decode_int4(uchar nibble) {
  int value = int(nibble & 15u);
  return value >= 8 ? value - 16 : value;
}
inline int decode_int2(uchar crumb) {
  int value = int(crumb & 3u);
  return value >= 2 ? value - 4 : value;
}
// Rounds a magnitude to a format with `mantissa_bits`, exponent range [min_exponent, max_exponent]
// and a subnormal step of 2^(min_exponent - mantissa_bits), to nearest even; returns the code below
// the sign bit, or `overflow` when the value rounds beyond the largest finite value.
inline uint __m2v_encode_magnitude(float magnitude, uint mantissa_bits, int min_exponent, int max_exponent, uint overflow) {
  if (magnitude == 0.0f) return 0u;
  uint bits = as_type<uint>(magnitude);
  int exponent = int((bits >> 23) & 255u) - 127;
  uint fraction = bits & 0x7FFFFFu;
  if (exponent < min_exponent) {
    // Subnormal: an integer count of the subnormal step, ties to even.
    float step = __m2v_make_float(0u, min_exponent - int(mantissa_bits), 0u);
    float count = rint(magnitude / step);
    uint quantized = uint(count);
    if (quantized >= (1u << mantissa_bits)) return 1u << mantissa_bits; // Rounded up to the smallest normal.
    return quantized;
  }
  uint dropped = 23u - mantissa_bits;
  uint kept = fraction >> dropped;
  uint remainder = fraction & ((1u << dropped) - 1u);
  uint half = 1u << (dropped - 1u);
  if (remainder > half || (remainder == half && (kept & 1u))) {
    kept += 1u;
    if (kept == (1u << mantissa_bits)) { kept = 0u; exponent += 1; }
  }
  if (exponent > max_exponent) return overflow;
  return (uint(exponent - min_exponent + 1) << mantissa_bits) | kept;
}
inline uchar encode_e4m3(float value) {
  uint sign = as_type<uint>(value) >> 31;
  if (isnan(value)) return uchar((sign << 7) | 0x7Fu);
  float magnitude = abs(value);
  if (isinf(value) || magnitude > 448.0f) return uchar((sign << 7) | 0x7Eu); // Saturates: E4M3 has no infinity.
  uint code = __m2v_encode_magnitude(magnitude, 3u, -6, 8, 0x7Eu);
  if (code == 0x7Fu) code = 0x7Eu; // 1.875 * 2^8 rounded up to the NaN code saturates too.
  return uchar((sign << 7) | code);
}
inline uchar encode_e5m2(float value) {
  uint sign = as_type<uint>(value) >> 31;
  if (isnan(value)) return uchar((sign << 7) | 0x7Fu);
  if (isinf(value)) return uchar((sign << 7) | 0x7Cu);
  return uchar((sign << 7) | __m2v_encode_magnitude(abs(value), 2u, -14, 15, 0x7Cu)); // Overflows to infinity.
}
inline uchar encode_e2m1(float value) {
  uint sign = as_type<uint>(value) >> 31;
  float magnitude = abs(value);
  if (isnan(value)) magnitude = 6.0f; // No NaN: the largest magnitude.
  if (magnitude > 6.0f) magnitude = 6.0f;
  // The eight magnitudes and the midpoints between them; a tie goes to the even code.
  uint code = 0u;
  float values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  for (uint next = 1u; next < 8u; ++next) {
    float midpoint = 0.5f * (values[next - 1u] + values[next]);
    if (magnitude > midpoint || (magnitude == midpoint && (next & 1u) == 0u)) code = next;
  }
  return uchar((sign << 3) | code);
}
inline uchar encode_int4(int value) { return uchar(clamp(value, -8, 7) & 15); }
inline uchar encode_int2(int value) { return uchar(clamp(value, -2, 1) & 3); }
// Element i of a byte tensor holding packed sub-byte codes, low element first.
template <typename Bytes> uchar nibble_at(const thread Bytes &bytes, int i) {
  uchar byte = bytes[i / 2];
  return (i % 2 == 0) ? uchar(byte & 15u) : uchar(byte >> 4);
}
template <typename Bytes> uchar crumb_at(const thread Bytes &bytes, int i) {
  uchar byte = bytes[i / 4];
  return uchar((byte >> uint(2 * (i % 4))) & 3u);
}
} // namespace low_bit
// Blockwise tensors (Metal 4.1's tensor_blockwise; this profile's spelling): a rank-2 tensor of
// low-bit codes, columns x rows with dimension 0 the column, whose elements come in blocks of `Block`
// along dimension 0 sharing one scale. The codes are a rank-1 byte tensor in row-major order (packed
// two or four per byte for the sub-byte formats) and the scales a rank-2 float tensor of extents
// (ceil(columns / Block), rows). Tensors are handles in this profile, so the view is a function of
// its tensors rather than a struct holding them.
enum class low_bit_format { e4m3, e5m2, e2m1, int4, int2 };
namespace low_bit {
template <low_bit_format Format, typename Data> float decode_element(const thread Data &data, int linear) {
  if (Format == low_bit_format::e4m3) return decode_e4m3(data[linear]);
  if (Format == low_bit_format::e5m2) return decode_e5m2(data[linear]);
  if (Format == low_bit_format::e2m1) return decode_e2m1(nibble_at(data, linear));
  if (Format == low_bit_format::int4) return float(decode_int4(nibble_at(data, linear)));
  return float(decode_int2(crumb_at(data, linear)));
}
template <low_bit_format Format, int Block, typename Data, typename Scales>
float blockwise_get(const thread Data &data, const thread Scales &scales, int columns, int x, int y) {
  return decode_element<Format>(data, x + columns * y) * scales[x / Block, y];
}
} // namespace low_bit
} // namespace metal
namespace mpp {
namespace tensor_ops {
// C = A B with A a blockwise low-bit tensor of extents (k, m): each product decodes and scales the
// left element, then accumulates in k order in the result's element type, as matmul2d does.
template <matmul2d_descriptor D, typename Scope, metal::low_bit_format Format, int Block, typename Data, typename Scales, typename RightTensor, typename ResultTensor>
void matmul2d_blockwise(const thread Data &data, const thread Scales &scales, const thread RightTensor &b, const thread ResultTensor &c) {
  int threads = Scope::threads();
  int thread_index = threads == 1 ? 0 : int(metal::__m2v_simd_index() * metal::__m2v_simd_width() + metal::__m2v_simd_lane());
  if (thread_index < threads) {
    for (int linear = thread_index; linear < D.m * D.n; linear += threads) {
      int column = linear % D.n, row = linear / D.n;
      typename ResultTensor::value_type accumulator = 0;
      if (D.result_mode == matmul2d_descriptor::mode::multiply_accumulate) accumulator = c[column, row];
      for (int kk = 0; kk < D.k; ++kk) {
        typename ResultTensor::value_type left = metal::low_bit::blockwise_get<Format, Block>(data, scales, D.k, kk, row);
        typename ResultTensor::value_type right = 0;
        if (D.transpose_right) right = b[kk, column];
        else right = b[column, kk];
        accumulator += left * right;
      }
      c[column, row] = accumulator;
    }
  }
}
} // namespace tensor_ops
} // namespace mpp
