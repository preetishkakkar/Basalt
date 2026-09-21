#pragma once
#include "msl_prelude.h"
// Independently authored declarations for tessellation (MSL sections 5.2.2 and
// 2.20). A post-tessellation vertex function runs once per vertex the
// tessellator produced: it takes the patch's control points through a
// patch_control_point<T> stage_in parameter and the tessellator's coordinate
// through [[position_in_patch]]. The compiler recognizes the members by name on
// this owned record; there are no bodies here.
namespace metal {
// The control points of one patch, as many as the entry's [[patch(type, N)]]
// attribute declares. Reading past that count is out of bounds, exactly as it is
// for an array.
template <typename T>
struct patch_control_point {
  // The control point at `index`, below the declared count.
  T operator[](uint index) const;
  // The declared count itself, which is a compile-time property of the entry.
  uint size() const;
};
}
