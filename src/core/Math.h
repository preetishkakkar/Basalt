// Types laid out as MSL lays them out, so structs upload as they stand: float3 is
// 16 bytes, float4x4 is four float4 columns.
#pragma once
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace basalt {

constexpr float kPi = 3.14159265358979323846f;

inline float radians(float degrees) { return degrees * (kPi / 180.0f); }
inline float degrees(float radians) { return radians * (180.0f / kPi); }
inline float saturate(float v) { return std::min(1.0f, std::max(0.0f, v)); }
inline float lerp(float a, float b, float t) { return a + (b - a) * t; }

struct Vec2 {
  float x = 0, y = 0;
  Vec2() = default;
  constexpr Vec2(float x, float y) : x(x), y(y) {}
  explicit constexpr Vec2(float s) : x(s), y(s) {}
};
inline Vec2 operator+(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
inline Vec2 operator-(Vec2 a, Vec2 b) { return {a.x - b.x, a.y - b.y}; }
inline Vec2 operator*(Vec2 a, float s) { return {a.x * s, a.y * s}; }

struct Vec3 {
  float x = 0, y = 0, z = 0;
  Vec3() = default;
  constexpr Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
  explicit constexpr Vec3(float s) : x(s), y(s), z(s) {}
  float &operator[](int i) { return (&x)[i]; }
  const float &operator[](int i) const { return (&x)[i]; }
};
inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }
inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(float s, Vec3 a) { return a * s; }
inline Vec3 operator*(Vec3 a, Vec3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
inline Vec3 operator/(Vec3 a, float s) { return a * (1.0f / s); }
inline Vec3 &operator+=(Vec3 &a, Vec3 b) { a = a + b; return a; }
inline Vec3 &operator-=(Vec3 &a, Vec3 b) { a = a - b; return a; }
inline Vec3 &operator*=(Vec3 &a, float s) { a = a * s; return a; }
inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float length(Vec3 a) { return std::sqrt(dot(a, a)); }
inline Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline Vec3 normalize(Vec3 a) {
  const float l = length(a);
  return l > 0.0f ? a / l : Vec3{0, 0, 0};
}
inline Vec3 minimum(Vec3 a, Vec3 b) { return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)}; }
inline Vec3 maximum(Vec3 a, Vec3 b) { return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}; }

struct Vec4 {
  float x = 0, y = 0, z = 0, w = 0;
  Vec4() = default;
  constexpr Vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
  constexpr Vec4(Vec3 v, float w) : x(v.x), y(v.y), z(v.z), w(w) {}
  explicit constexpr Vec4(float s) : x(s), y(s), z(s), w(s) {}
  Vec3 xyz() const { return {x, y, z}; }
  float &operator[](int i) { return (&x)[i]; }
  const float &operator[](int i) const { return (&x)[i]; }
};
inline Vec4 operator+(Vec4 a, Vec4 b) { return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
inline Vec4 operator-(Vec4 a, Vec4 b) { return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}; }
inline Vec4 operator*(Vec4 a, float s) { return {a.x * s, a.y * s, a.z * s, a.w * s}; }
inline float dot(Vec4 a, Vec4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

// Column-major, like MSL.
struct Mat4 {
  Vec4 columns[4];
  Mat4() : columns{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}} {}
  Mat4(Vec4 c0, Vec4 c1, Vec4 c2, Vec4 c3) : columns{c0, c1, c2, c3} {}
  Vec4 &operator[](int i) { return columns[i]; }
  const Vec4 &operator[](int i) const { return columns[i]; }
  static Mat4 zero() { return {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}; }
  // Rows, for shaders that store a transform as three float4 and apply dot products.
  Vec4 row(int i) const { return {columns[0][i], columns[1][i], columns[2][i], columns[3][i]}; }
};

inline Vec4 operator*(const Mat4 &m, Vec4 v) {
  return m.columns[0] * v.x + m.columns[1] * v.y + m.columns[2] * v.z + m.columns[3] * v.w;
}
inline Mat4 operator*(const Mat4 &a, const Mat4 &b) {
  return {a * b.columns[0], a * b.columns[1], a * b.columns[2], a * b.columns[3]};
}
inline Vec3 transformPoint(const Mat4 &m, Vec3 p) { return (m * Vec4(p, 1.0f)).xyz(); }
inline Vec3 transformDirection(const Mat4 &m, Vec3 d) { return (m * Vec4(d, 0.0f)).xyz(); }

inline Mat4 translation(Vec3 t) {
  Mat4 m;
  m.columns[3] = {t.x, t.y, t.z, 1.0f};
  return m;
}
inline Mat4 scaling(Vec3 s) {
  Mat4 m;
  m.columns[0].x = s.x;
  m.columns[1].y = s.y;
  m.columns[2].z = s.z;
  return m;
}
inline Mat4 transpose(const Mat4 &m) {
  return {m.row(0), m.row(1), m.row(2), m.row(3)};
}

// A rotation from a unit quaternion stored as (x, y, z, w), glTF's order.
inline Mat4 rotation(Vec4 q) {
  const float x = q.x, y = q.y, z = q.z, w = q.w;
  Mat4 m;
  m.columns[0] = {1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w), 0};
  m.columns[1] = {2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w), 0};
  m.columns[2] = {2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y), 0};
  return m;
}

inline Mat4 lookAt(Vec3 eye, Vec3 target, Vec3 up) {
  const Vec3 f = normalize(target - eye);
  const Vec3 s = normalize(cross(f, up));
  const Vec3 u = cross(s, f);
  Mat4 m;
  m.columns[0] = {s.x, u.x, -f.x, 0};
  m.columns[1] = {s.y, u.y, -f.y, 0};
  m.columns[2] = {s.z, u.z, -f.z, 0};
  m.columns[3] = {-dot(s, eye), -dot(u, eye), dot(f, eye), 1};
  return m;
}

// Reverse-Z with an infinite far plane; paired with a GREATER test and a clear of 0.
inline Mat4 perspectiveReverseZ(float fovY, float aspect, float nearPlane) {
  const float t = 1.0f / std::tan(fovY * 0.5f);
  Mat4 m = Mat4::zero();
  m.columns[0].x = t / aspect;
  m.columns[1].y = t;
  m.columns[2].w = -1.0f;
  m.columns[3].z = nearPlane;
  return m;
}

// 0..1 depth for the cascades with a LESS test; view space looks down -Z, so depth is negated.
inline Mat4 orthographic(float left, float right, float bottom, float top, float zNear, float zFar) {
  Mat4 m;
  m.columns[0].x = 2.0f / (right - left);
  m.columns[1].y = 2.0f / (top - bottom);
  m.columns[2].z = -1.0f / (zFar - zNear);
  m.columns[3] = {-(right + left) / (right - left), -(top + bottom) / (top - bottom),
                  -zNear / (zFar - zNear), 1.0f};
  return m;
}

Mat4 inverse(const Mat4 &m);

// Inverse transpose of the upper 3x3, for normals under non-uniform scale.
Mat4 normalMatrix(const Mat4 &model);

struct Aabb {
  Vec3 minimum{1e30f, 1e30f, 1e30f};
  Vec3 maximum{-1e30f, -1e30f, -1e30f};
  bool valid() const { return minimum.x <= maximum.x; }
  void add(Vec3 p) {
    minimum = basalt::minimum(minimum, p);
    maximum = basalt::maximum(maximum, p);
  }
  void add(const Aabb &other) {
    if (!other.valid()) return;
    add(other.minimum);
    add(other.maximum);
  }
  Vec3 center() const { return (minimum + maximum) * 0.5f; }
  Vec3 extent() const { return (maximum - minimum) * 0.5f; }
  float radius() const { return length(extent()); }
  Aabb transformed(const Mat4 &m) const;
};

// Six planes as (normal, distance) with the convention dot(plane.xyz, p) + plane.w >= 0 inside.
struct Frustum {
  Vec4 planes[6];
  static Frustum fromViewProjection(const Mat4 &viewProjection);
  bool intersects(const Aabb &box) const;
};

} // namespace basalt
