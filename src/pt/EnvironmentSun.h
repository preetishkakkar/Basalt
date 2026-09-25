// Moves an environment map's sun (or moon) out of the image and into an analytic disc light,
// so every renderer lights with the same sun: the rasteriser as its directional light with
// traced or mapped shadows, the path tracers as the sun they sample and hit. Energy is moved,
// not estimated: whatever radiance leaves the image becomes the disc's irradiance.
#pragma once
#include "pt/Shared.h"

#include <vector>

namespace pt {

struct EnvironmentSun {
  bool found = false;
  float3 direction{0.0f, 1.0f, 0.0f};  // towards the sun; the brightest region even when none was found
  float3 irradiance{0.0f};             // removed from the image: radiance times solid angle, summed
  float angularRadius = 0.0f;          // of a uniform disc with the region's energy-weighted spread
  float share = 0.0f;                  // of the environment's luminance power, before removal
  uint texels = 0;                     // removed region
};

// Finds the brightest compact region of an RGBA equirectangular image (the path tracer's
// mapping, shared/shading.h equirectangularUV), and if it stands out from the sky replaces
// it with the colour around it, returning what was removed. An image without such a region
// is left unchanged, bit for bit.
EnvironmentSun extractEnvironmentSun(std::vector<float> &rgba, uint width, uint height);

// The path tracer's sun uniforms (PathUniforms::sunDirection, sunRadiance) for a disc of this
// irradiance, direction and angular radius; a zero irradiance gives no sun.
void environmentSunUniforms(float3 direction, float3 irradiance, float angularRadius, Vector<float, 4> &sunDirection,
                            Vector<float, 4> &sunRadiance);

} // namespace pt
