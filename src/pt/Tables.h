// Tables the path tracer samples, built on the host once and read by every backend.
#pragma once
#include "pt/Tracing.h"

#include <vector>

namespace pt {

// The environment's luminance distribution, in the layout pt_lights.slang samples:
// up to 1024 x 512 cells, each the mean luminance of its texels times sin(theta).
// `info` receives (columns, rows, integral, present).
void buildEnvironmentDistribution(const float *rgba, uint width, uint height, std::vector<float> &distribution,
                                  float4 &info);

// E(cos view, roughness) for the GGX lobe with Fresnel at one, kAlbedoTableSize squared,
// followed by the closed dielectric interface albedos A (pt_bsdf.slang), all integrated
// with the tracer's own visible-normal sampling.
std::vector<float> buildSpecularAlbedoTable();

} // namespace pt
