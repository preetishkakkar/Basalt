// Queue slots reserved by one atomic per path: for devices without subgroup arithmetic.
#define PATH_WAVEFRONT_SHADE path_wavefront_shade_atomic
#include "path_wavefront_body.metal"
