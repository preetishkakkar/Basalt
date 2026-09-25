// Wide (BVH4/BVH8) own-BVH wavefront bounce with the nearest-hit traversal fused into shading.
#define BASALT_WAVE_FUSED 1
#define BASALT_WIDE_BVH 1
#define PT_WIDE_ORDER_SORT 1
#define PATH_WAVEFRONT_SHADE path_wavefront_fused_wide
#include "path_wavefront_body.metal"
