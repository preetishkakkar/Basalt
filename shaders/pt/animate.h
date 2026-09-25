// The synthetic animation's vertex warp (animate.metal), shared with its host.
#pragma once

struct AnimateControl {
  uint4 counts; // vertices, reserved, reserved, reserved
  float4 wave;  // amplitude, wavenumber (radians per unit), phase (radians), reserved
};
