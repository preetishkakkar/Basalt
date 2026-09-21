#include "core/Math.h"

namespace basalt {

Mat4 inverse(const Mat4 &m) {
  const float *a = &m.columns[0].x;
  float inv[16];
  inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] +
           a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
  inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] -
           a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
  inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] +
           a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
  inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] -
            a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
  inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] -
           a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
  inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] +
           a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
  inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] -
           a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
  inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] +
            a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
  inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] +
           a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
  inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] -
           a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
  inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] +
            a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
  inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] -
            a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
  inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] -
           a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
  inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] +
           a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
  inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] -
            a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
  inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] +
            a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];
  float determinant = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
  if (determinant == 0.0f) return Mat4{};
  determinant = 1.0f / determinant;
  Mat4 result;
  float *out = &result.columns[0].x;
  for (int i = 0; i < 16; ++i) out[i] = inv[i] * determinant;
  return result;
}

Mat4 normalMatrix(const Mat4 &model) {
  Mat4 upper;
  upper.columns[0] = Vec4(model.columns[0].xyz(), 0.0f);
  upper.columns[1] = Vec4(model.columns[1].xyz(), 0.0f);
  upper.columns[2] = Vec4(model.columns[2].xyz(), 0.0f);
  upper.columns[3] = {0, 0, 0, 1};
  return transpose(inverse(upper));
}

Aabb Aabb::transformed(const Mat4 &m) const {
  if (!valid()) return {};
  // Centre transformed plus the extent through the absolute rotation: tighter and cheaper than eight corners.
  const Vec3 c = transformPoint(m, center());
  const Vec3 e = extent();
  Vec3 half{0, 0, 0};
  for (int row = 0; row < 3; ++row) {
    half[row] = std::abs(m.columns[0][row]) * e.x + std::abs(m.columns[1][row]) * e.y +
                std::abs(m.columns[2][row]) * e.z;
  }
  Aabb box;
  box.minimum = c - half;
  box.maximum = c + half;
  return box;
}

Frustum Frustum::fromViewProjection(const Mat4 &viewProjection) {
  // Gribb-Hartmann: planes from sums and differences of the view-projection rows.
  const Vec4 r0 = viewProjection.row(0), r1 = viewProjection.row(1);
  const Vec4 r2 = viewProjection.row(2), r3 = viewProjection.row(3);
  Frustum f;
  f.planes[0] = r3 + r0; // left
  f.planes[1] = r3 - r0; // right
  f.planes[2] = r3 + r1; // bottom
  f.planes[3] = r3 - r1; // top
  f.planes[4] = r3 - r2; // near under reverse-Z (depth 1 at the near plane)
  f.planes[5] = r2;      // far
  for (Vec4 &plane : f.planes) {
    const float l = length(plane.xyz());
    if (l > 0.0f) plane = plane * (1.0f / l);
  }
  return f;
}

bool Frustum::intersects(const Aabb &box) const {
  if (!box.valid()) return false;
  const Vec3 c = box.center(), e = box.extent();
  for (const Vec4 &plane : planes) {
    const Vec3 n = plane.xyz();
    // The box is outside when its most positive corner still falls behind.
    const float reach = std::abs(n.x) * e.x + std::abs(n.y) * e.y + std::abs(n.z) * e.z;
    if (dot(n, c) + plane.w + reach < 0.0f) return false;
  }
  return true;
}

} // namespace basalt
