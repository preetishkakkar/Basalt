// path_wavefront_fused_wide with the deep traversal stack, for trees whose stack bound exceeds 64.
#define PT_WIDE_BVH_STACK PT_WIDE_BVH_STACK_DEEP
#define BASALT_WAVE_FUSED 1
#define BASALT_WIDE_BVH 1
#define PT_WIDE_ORDER_SORT 1
#define PATH_WAVEFRONT_SHADE path_wavefront_fused_wide_deep
#include "path_wavefront_body.metal"
