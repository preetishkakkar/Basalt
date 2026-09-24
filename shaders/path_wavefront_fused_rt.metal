// Wavefront bounce with the nearest-hit traversal fused into shading.
#define BASALT_WAVE_FUSED 1
#define BASALT_RAY_TRACING 1
#define PATH_WAVEFRONT_SHADE path_wavefront_fused_rt
#include "path_wavefront_body.metal"
