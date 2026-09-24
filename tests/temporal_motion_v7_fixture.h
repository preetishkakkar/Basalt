#pragma once
#include "temporal_motion_fixture.h"

// V7 motion corpus: the frozen V3.1 scene and trajectories with V7's layers
// and depth of field. Frozen with its references (tests/data/temporal_v7); do not edit to
// make a failed quality gate pass.
namespace temporal_motion_v7 {
constexpr std::array<const char *, 3> sequences{"glass-translation", "dof-translation", "dof-stop-start"};

// Glass sequences: the brushed-metal sphere becomes smooth closed glass (transmission 1,
// IOR 1.5) and the orange sphere gets a mirror-like clearcoat, so both guide layer flags
// occur. Depth-of-field sequences keep the V3.1 materials.
inline void scene(pt_fixture::Builder &b, pt::uint sequence) {
  using namespace pt;
  temporal_motion::scene(b);
  if (sequence == 0) {
    Material &glass = b.frame.materials[3];
    glass.baseColorFactor = float4(1.0f);
    glass.factors.x = 0.0f;   // dielectric
    glass.factors.y = 0.05f;  // roughness
    glass.transmission = float4(1.0f, 1.5f, 1.0f, 1.0f);
    b.frame.materials[2].clearcoat = float4(1.0f, 0.05f, 1.0f, 0.0f);
  }
}

// The V3.1 translation (0) and stop-start (2) trajectories; depth-of-field sequences focus
// on the look target (about 4.3 units ahead) with an aperture radius of 0.04.
inline void camera(pt_fixture::Builder &b, pt::uint sequence, pt::uint frame) {
  temporal_motion::camera(b, sequence == 2 ? 2u : 0u, frame);
  if (sequence != 0) b.frame.uniforms.lens = pt::float4(0.04f, 4.3f, 0.0f, 0.0f);
}
}
