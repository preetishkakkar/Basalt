// The C++ side of the single-source shim: the MSL vector types and functions the shared
// shader headers use, so MSVC compiles the same files msl2spirv does. Only what the
// shared code needs is here; a new MSL construct in shared code means a line here too.
#pragma once
#define BASALT_HOST 1

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace pt {

using uint = std::uint32_t;
using ulong = std::uint64_t;

struct float2 {
  float x = 0.0f, y = 0.0f;
  constexpr float2() = default;
  constexpr explicit float2(float s) : x(s), y(s) {}
  constexpr float2(float a, float b) : x(a), y(b) {}
};

struct float3 {
  float x = 0.0f, y = 0.0f, z = 0.0f;
  constexpr float3() = default;
  constexpr explicit float3(float s) : x(s), y(s), z(s) {}
  constexpr float3(float a, float b, float c) : x(a), y(b), z(c) {}
  constexpr float3(float2 v, float c) : x(v.x), y(v.y), z(c) {}
};

struct uint2 {
  uint x = 0, y = 0;
  constexpr uint2() = default;
  constexpr uint2(uint a, uint b) : x(a), y(b) {}
};

struct uint4 {
  uint x = 0, y = 0, z = 0, w = 0;
  constexpr uint4() = default;
  constexpr explicit uint4(uint s) : x(s), y(s), z(s), w(s) {}
  constexpr uint4(uint a, uint b, uint c, uint d) : x(a), y(b), z(c), w(d) {}
};

// Unsigned arithmetic wraps, as in MSL.
inline uint4 operator+(uint4 a, uint4 b) { return uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
inline uint4 operator*(uint4 a, uint4 b) { return uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
inline uint4 operator^(uint4 a, uint4 b) { return uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w); }
inline uint4 operator>>(uint4 a, uint4 b) { return uint4(a.x >> b.x, a.y >> b.y, a.z >> b.z, a.w >> b.w); }
inline uint4 operator+(uint4 a, uint s) { return a + uint4(s); }
inline uint4 operator*(uint4 a, uint s) { return a * uint4(s); }
inline uint4 operator^(uint4 a, uint s) { return a ^ uint4(s); }
inline uint4 operator>>(uint4 a, uint s) { return a >> uint4(s); }

struct float4 {
  float x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f;
  constexpr float4() = default;
  constexpr explicit float4(float s) : x(s), y(s), z(s), w(s) {}
  constexpr float4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
  constexpr float4(float3 v, float d) : x(v.x), y(v.y), z(v.z), w(d) {}
  constexpr float4(float2 a, float2 b) : x(a.x), y(a.y), z(b.x), w(b.y) {}
  // Value conversion, as MSL's float4(uint4).
  constexpr explicit float4(uint4 v)
      : x(static_cast<float>(v.x)), y(static_cast<float>(v.y)), z(static_cast<float>(v.z)),
        w(static_cast<float>(v.w)) {}
};
static_assert(sizeof(float4) == 16, "float4 must match MSL");

struct int3 {
  int x = 0, y = 0, z = 0;
  constexpr int3() = default;
  constexpr int3(int a, int b, int c) : x(a), y(b), z(c) {}
  // Value conversion toward zero, as MSL's int3(float3).
  constexpr explicit int3(float3 v) : x(static_cast<int>(v.x)), y(static_cast<int>(v.y)), z(static_cast<int>(v.z)) {}
};
static_assert(sizeof(int3) == sizeof(float3), "as_type between int3 and float3 needs equal sizes");
inline int3 operator+(int3 a, int3 b) { return int3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline int3 operator*(int3 a, int3 b) { return int3(a.x * b.x, a.y * b.y, a.z * b.z); }

// Component-wise arithmetic, generated for each vector width.
#define PT_VECTOR_OPS(T, APPLY)                                                                   \
  inline T operator+(T a, T b) { return APPLY(a, b, +); }                                         \
  inline T operator-(T a, T b) { return APPLY(a, b, -); }                                         \
  inline T operator*(T a, T b) { return APPLY(a, b, *); }                                         \
  inline T operator/(T a, T b) { return APPLY(a, b, /); }                                         \
  inline T operator*(T a, float s) { return a * T(s); }                                           \
  inline T operator*(float s, T a) { return T(s) * a; }                                           \
  inline T operator/(T a, float s) { return a / T(s); }                                           \
  inline T operator+(T a, float s) { return a + T(s); }                                           \
  inline T operator-(T a, float s) { return a - T(s); }                                           \
  inline T operator+(float s, T a) { return T(s) + a; }                                           \
  inline T operator-(float s, T a) { return T(s) - a; }                                           \
  inline T &operator+=(T &a, T b) { return a = a + b; }                                           \
  inline T &operator-=(T &a, T b) { return a = a - b; }                                           \
  inline T &operator*=(T &a, T b) { return a = a * b; }                                           \
  inline T &operator/=(T &a, T b) { return a = a / b; }                                           \
  inline T &operator*=(T &a, float s) { return a = a * s; }                                       \
  inline T &operator/=(T &a, float s) { return a = a / s; }

#define PT_APPLY2(a, b, op) float2(a.x op b.x, a.y op b.y)
#define PT_APPLY3(a, b, op) float3(a.x op b.x, a.y op b.y, a.z op b.z)
#define PT_APPLY4(a, b, op) float4(a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w)
PT_VECTOR_OPS(float2, PT_APPLY2)
PT_VECTOR_OPS(float3, PT_APPLY3)
PT_VECTOR_OPS(float4, PT_APPLY4)
#undef PT_VECTOR_OPS

inline float2 operator-(float2 a) { return float2(-a.x, -a.y); }
inline float3 operator-(float3 a) { return float3(-a.x, -a.y, -a.z); }
inline float4 operator-(float4 a) { return float4(-a.x, -a.y, -a.z, -a.w); }

// Scalar functions. Every name that also has a vector form is declared for float here, or
// unqualified calls inside this namespace would find only the vector one.
inline float abs(float x) { return std::fabs(x); }
inline float sqrt(float x) { return std::sqrt(x); }
inline float rsqrt(float x) { return 1.0f / std::sqrt(x); }
inline float pow(float x, float y) { return std::pow(x, y); }
inline float exp(float x) { return std::exp(x); }
inline float exp2(float x) { return std::exp2(x); }
inline float log(float x) { return std::log(x); }
inline float log2(float x) { return std::log2(x); }
inline float sin(float x) { return std::sin(x); }
inline float cos(float x) { return std::cos(x); }
inline float tan(float x) { return std::tan(x); }
inline float asin(float x) { return std::asin(x); }
inline float acos(float x) { return std::acos(x); }
inline float atan(float x) { return std::atan(x); }
inline float atan2(float y, float x) { return std::atan2(y, x); }
inline float floor(float x) { return std::floor(x); }
inline float ceil(float x) { return std::ceil(x); }
inline float fract(float x) { return x - std::floor(x); }
inline float min(float a, float b) { return a < b ? a : b; }
inline float max(float a, float b) { return a > b ? a : b; }
inline uint min(uint a, uint b) { return a < b ? a : b; }
inline uint max(uint a, uint b) { return a > b ? a : b; }
inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
inline float clamp(float x, float lo, float hi) { return min(max(x, lo), hi); }
inline uint clamp(uint x, uint lo, uint hi) { return min(max(x, lo), hi); }
inline float saturate(float x) { return clamp(x, 0.0f, 1.0f); }
inline float mix(float a, float b, float t) { return a + (b - a) * t; }
inline float sign(float x) { return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); }
inline float copysign(float x, float y) { return std::copysign(x, y); }
inline bool isnan(float x) { return std::isnan(x); }
inline bool isinf(float x) { return std::isinf(x); }
inline bool isfinite(float x) { return std::isfinite(x); }
inline uint popcount(uint x) { return static_cast<uint>(std::popcount(x)); }

#define PT_UNARY(name)                                                                            \
  inline float2 name(float2 v) { return float2(name(v.x), name(v.y)); }                           \
  inline float3 name(float3 v) { return float3(name(v.x), name(v.y), name(v.z)); }                \
  inline float4 name(float4 v) { return float4(name(v.x), name(v.y), name(v.z), name(v.w)); }
PT_UNARY(abs)
PT_UNARY(sqrt)
PT_UNARY(rsqrt)
PT_UNARY(exp)
PT_UNARY(exp2)
PT_UNARY(log)
PT_UNARY(sin)
PT_UNARY(cos)
PT_UNARY(floor)
PT_UNARY(ceil)
PT_UNARY(fract)
PT_UNARY(saturate)
PT_UNARY(sign)
#undef PT_UNARY

#define PT_BINARY(name)                                                                           \
  inline float2 name(float2 a, float2 b) { return float2(name(a.x, b.x), name(a.y, b.y)); }       \
  inline float3 name(float3 a, float3 b) {                                                        \
    return float3(name(a.x, b.x), name(a.y, b.y), name(a.z, b.z));                                \
  }                                                                                               \
  inline float4 name(float4 a, float4 b) {                                                        \
    return float4(name(a.x, b.x), name(a.y, b.y), name(a.z, b.z), name(a.w, b.w));                \
  }
PT_BINARY(min)
PT_BINARY(max)
PT_BINARY(pow)
#undef PT_BINARY

inline float2 clamp(float2 v, float2 lo, float2 hi) { return min(max(v, lo), hi); }
inline float3 clamp(float3 v, float3 lo, float3 hi) { return min(max(v, lo), hi); }
inline float4 clamp(float4 v, float4 lo, float4 hi) { return min(max(v, lo), hi); }
inline float2 mix(float2 a, float2 b, float t) { return a + (b - a) * t; }
inline float3 mix(float3 a, float3 b, float t) { return a + (b - a) * t; }
inline float4 mix(float4 a, float4 b, float t) { return a + (b - a) * t; }
inline float3 mix(float3 a, float3 b, float3 t) { return a + (b - a) * t; }

inline float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float dot(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
inline float3 cross(float3 a, float3 b) {
  return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
inline float length(float2 v) { return sqrt(dot(v, v)); }
inline float length(float3 v) { return sqrt(dot(v, v)); }
inline float length(float4 v) { return sqrt(dot(v, v)); }
inline float length_squared(float3 v) { return dot(v, v); }
inline float distance(float3 a, float3 b) { return length(a - b); }
inline float2 normalize(float2 v) { return v * rsqrt(dot(v, v)); }
inline float3 normalize(float3 v) { return v * rsqrt(dot(v, v)); }
inline float4 normalize(float4 v) { return v * rsqrt(dot(v, v)); }
inline float3 reflect(float3 i, float3 n) { return i - n * (2.0f * dot(n, i)); }
// MSL's refract: zero on total internal reflection.
inline float3 refract(float3 i, float3 n, float eta) {
  const float cosine = dot(n, i);
  const float k = 1.0f - eta * eta * (1.0f - cosine * cosine);
  if (k < 0.0f) return float3(0.0f);
  return i * eta - n * (eta * cosine + std::sqrt(k));
}

inline bool any(bool a, bool b, bool c) { return a || b || c; }
inline bool anyNonFinite(float3 v) { return !isfinite(v.x) || !isfinite(v.y) || !isfinite(v.z); }

// Bit casts, as MSL's as_type.
template <class To, class From> inline To as_type(From value) {
  static_assert(sizeof(To) == sizeof(From), "as_type needs equal sizes");
  To result;
  std::memcpy(&result, &value, sizeof(To));
  return result;
}

// Multi-component swizzles, which the shared code spells as calls on both sides.
inline float3 xyz(float4 v) { return float3(v.x, v.y, v.z); }
inline float2 xy(float4 v) { return float2(v.x, v.y); }
inline float2 xy(float3 v) { return float2(v.x, v.y); }
inline float2 zw(float4 v) { return float2(v.z, v.w); }

} // namespace pt
