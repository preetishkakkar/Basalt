// Own-BVH wavefront bounce with the nearest-hit traversal fused into shading.
#define BASALT_WAVE_FUSED 1
#define PATH_WAVEFRONT_SHADE path_wavefront_fused
#include "path_wavefront_body.metal"
