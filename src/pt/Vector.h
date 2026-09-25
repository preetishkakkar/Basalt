// Host vector math on the Slang C++ prelude's vectors. The CPU tracer's generated code, and the
// shared structs its header declares, use slangcpp::Vector<T, N>; the host's float3 and friends
// derive from those, adding zero initialisation, the mixed constructors, scalar operators and the
// usual functions, so host code reads and writes the shared structs directly.
#pragma once
// The prelude includes these inside its namespace; included first, those includes are no-ops.
#include <float.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244) // the prelude's own narrowing
#endif
#include "slang-cpp-prelude.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif
// The generated exports link into this library, not a DLL.
#undef SLANG_PRELUDE_SHARED_LIB_EXPORT
#define SLANG_PRELUDE_SHARED_LIB_EXPORT

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#ifndef SLANG_PRELUDE_NAMESPACE
#error "the pt library compiles the Slang prelude in a namespace (SLANG_PRELUDE_NAMESPACE)"
#endif

// Scalar operands broadcast; the prelude has the component-wise vector ones. In the prelude's
// namespace, so they are found for its vectors as for the host's. Unsigned arithmetic wraps, as
// in Slang.
namespace SLANG_PRELUDE_NAMESPACE {
#define PT_SCALAR_OPERATOR(op)                                                                    \
  template <class T, int N> inline Vector<T, N> operator op(const Vector<T, N> &a, std::type_identity_t<T> s) { \
    return a op Vector<T, N>(s);                                                                  \
  }                                                                                               \
  template <class T, int N> inline Vector<T, N> operator op(std::type_identity_t<T> s, const Vector<T, N> &a) { \
    return Vector<T, N>(s) op a;                                                                  \
  }                                                                                               \
  template <class T, int N> inline Vector<T, N> &operator op##=(Vector<T, N> &a, const Vector<T, N> &b) { \
    return a = a op b;                                                                            \
  }                                                                                               \
  template <class T, int N> inline Vector<T, N> &operator op##=(Vector<T, N> &a, std::type_identity_t<T> s) { \
    return a = a op Vector<T, N>(s);                                                              \
  }
PT_SCALAR_OPERATOR(+)
PT_SCALAR_OPERATOR(-)
PT_SCALAR_OPERATOR(*)
PT_SCALAR_OPERATOR(/)
#undef PT_SCALAR_OPERATOR
template <int N> inline Vector<uint32_t, N> operator^(const Vector<uint32_t, N> &a, uint32_t s) { return a ^ Vector<uint32_t, N>(s); }
template <int N> inline Vector<uint32_t, N> operator>>(const Vector<uint32_t, N> &a, uint32_t s) { return a >> Vector<uint32_t, N>(s); }

} // namespace SLANG_PRELUDE_NAMESPACE

namespace pt {

using uint = std::uint32_t;
using ulong = std::uint64_t;
template <class T, int N> using Vector = SLANG_PRELUDE_NAMESPACE::Vector<T, N>;

struct uint4;

struct float2 : Vector<float, 2> {
  float2() : Vector(0.0f) {}
  explicit float2(float s) : Vector(s) {}
  float2(float a, float b) : Vector(a, b) {}
  float2(const Vector<float, 2> &v) : Vector(v) {}
};

struct float3 : Vector<float, 3> {
  float3() : Vector(0.0f) {}
  explicit float3(float s) : Vector(s) {}
  float3(float a, float b, float c) : Vector(a, b, c) {}
  float3(const Vector<float, 2> &v, float c) : Vector(v.x, v.y, c) {}
  float3(const Vector<float, 3> &v) : Vector(v) {}
};

struct float4 : Vector<float, 4> {
  float4() : Vector(0.0f) {}
  explicit float4(float s) : Vector(s) {}
  float4(float a, float b, float c, float d) : Vector(a, b, c, d) {}
  float4(const Vector<float, 3> &v, float d) : Vector(v.x, v.y, v.z, d) {}
  float4(const Vector<float, 2> &a, const Vector<float, 2> &b) : Vector(a.x, a.y, b.x, b.y) {}
  float4(const Vector<float, 4> &v) : Vector(v) {}
  // Value conversion, as Slang's float4(uint4).
  explicit float4(const Vector<uint, 4> &v)
      : Vector(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), static_cast<float>(v.w)) {}
};

struct uint4 : Vector<uint, 4> {
  uint4() : Vector(0u) {}
  explicit uint4(uint s) : Vector(s) {}
  uint4(uint a, uint b, uint c, uint d) : Vector(a, b, c, d) {}
  uint4(const Vector<uint, 4> &v) : Vector(v) {}
};

static_assert(sizeof(float2) == 8 && sizeof(float3) == 12 && sizeof(float4) == 16 && sizeof(uint4) == 16,
              "host vectors must have the prelude's layout");

// Scalar functions. Every name that also has a vector form is declared for float here, so
// unqualified calls inside this namespace find both.
inline float abs(float x) { return std::fabs(x); }
inline float sqrt(float x) { return std::sqrt(x); }
inline float rsqrt(float x) { return 1.0f / std::sqrt(x); }
inline float pow(float x, float y) { return std::pow(x, y); }
inline float exp(float x) { return std::exp(x); }
inline float exp2(float x) { return std::exp2(x); }
inline float log(float x) { return std::log(x); }
inline float log2(float x) { return std::log2(x); }
inline float floor(float x) { return std::floor(x); }
inline float ceil(float x) { return std::ceil(x); }
inline float fract(float x) { return x - std::floor(x); }
inline float min(float a, float b) { return a < b ? a : b; }
inline float max(float a, float b) { return a > b ? a : b; }
inline uint min(uint a, uint b) { return a < b ? a : b; }
inline uint max(uint a, uint b) { return a > b ? a : b; }
inline float clamp(float x, float lo, float hi) { return min(max(x, lo), hi); }
inline float saturate(float x) { return clamp(x, 0.0f, 1.0f); }
inline float mix(float a, float b, float t) { return a + (b - a) * t; }
inline bool isfinite(float x) { return std::isfinite(x); }

template <int N, class F> inline Vector<float, N> componentwise(const Vector<float, N> &v, F f) {
  Vector<float, N> result;
  for (int i = 0; i < N; ++i) result[i] = f(v[i]);
  return result;
}
template <int N, class F> inline Vector<float, N> componentwise(const Vector<float, N> &a, const Vector<float, N> &b, F f) {
  Vector<float, N> result;
  for (int i = 0; i < N; ++i) result[i] = f(a[i], b[i]);
  return result;
}
#define PT_UNARY(name)                                                                            \
  template <int N> inline Vector<float, N> name(const Vector<float, N> &v) {                             \
    return componentwise(v, [](float x) { return name(x); });                                     \
  }
PT_UNARY(abs)
PT_UNARY(sqrt)
PT_UNARY(exp)
PT_UNARY(log)
PT_UNARY(floor)
PT_UNARY(fract)
PT_UNARY(saturate)
#undef PT_UNARY
#define PT_BINARY(name)                                                                           \
  template <int N> inline Vector<float, N> name(const Vector<float, N> &a, const Vector<float, N> &b) {  \
    return componentwise(a, b, [](float x, float y) { return name(x, y); });                      \
  }
PT_BINARY(min)
PT_BINARY(max)
PT_BINARY(pow)
#undef PT_BINARY

template <int N> inline Vector<float, N> clamp(const Vector<float, N> &v, const Vector<float, N> &lo, const Vector<float, N> &hi) {
  return min(max(v, lo), hi);
}
template <int N> inline Vector<float, N> mix(const Vector<float, N> &a, const Vector<float, N> &b, float t) {
  return a + (b - a) * t;
}
template <int N> inline float dot(const Vector<float, N> &a, const Vector<float, N> &b) {
  float sum = a[0] * b[0];
  for (int i = 1; i < N; ++i) sum += a[i] * b[i];
  return sum;
}
inline float3 cross(const Vector<float, 3> &a, const Vector<float, 3> &b) {
  return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
template <int N> inline float length(const Vector<float, N> &v) { return std::sqrt(dot(v, v)); }
template <int N> inline float distance(const Vector<float, N> &a, const Vector<float, N> &b) { return length(a - b); }
// As Slang's C++ target computes it, so host and generated code round alike.
template <int N> inline Vector<float, N> normalize(const Vector<float, N> &v) { return v / length(v); }

inline float3 reflect(const Vector<float, 3> &i, const Vector<float, 3> &n) { return i - n * (2.0f * dot(n, i)); }
// Zero on total internal reflection.
inline float3 refract(const Vector<float, 3> &i, const Vector<float, 3> &n, float eta) {
  const float cosine = dot(n, i);
  const float k = 1.0f - eta * eta * (1.0f - cosine * cosine);
  if (k < 0.0f) return float3(0.0f);
  return i * eta - n * (eta * cosine + std::sqrt(k));
}

// Bit casts, as Slang's asuint/asfloat.
template <class To, class From> inline To as_type(const From &value) {
  static_assert(sizeof(To) == sizeof(From), "as_type needs equal sizes");
  To result;
  std::memcpy(&result, &value, sizeof(To));
  return result;
}

// The swizzles host code spells as calls.
inline float3 xyz(const Vector<float, 4> &v) { return float3(v.x, v.y, v.z); }
inline float2 xy(const Vector<float, 4> &v) { return float2(v.x, v.y); }
inline float2 xy(const Vector<float, 3> &v) { return float2(v.x, v.y); }

} // namespace pt
