#pragma once
#include "msl_prelude.h"
// Independently authored declarations for pull-model interpolation (MSL section
// 6.12 and the interpolant type of section 2.18). An interpolant<T, P> is a
// fragment stage_in member that is not interpolated on entry: each method
// samples it at a named location instead. The compiler recognizes the methods by
// name on this owned record; there are no bodies here.
namespace metal {
namespace interpolation {
struct perspective {};
struct no_perspective {};
}
template <typename T, typename P = interpolation::perspective>
struct interpolant {
  // Same value as a T member with center_perspective / center_no_perspective.
  T interpolate_at_center() const;
  // Sampled inside both the pixel and the primitive, as centroid_* would be.
  T interpolate_at_centroid() const;
  // A window-coordinate offset from the pixel's top-left corner, each component
  // in [0.0, 1.0) on a 1/16 pixel grid.
  T interpolate_at_offset(float2 offset) const;
  // The location of the named sample, as sample_* would be under per-sample
  // execution. The position is undefined if the pixel has no such sample.
  T interpolate_at_sample(uint sample) const;
};
}
