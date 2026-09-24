// Queue slots reserved once per subgroup; path_wavefront_atomic.metal is the per-path fallback.
#define PT_WAVE_SUBGROUP_ALLOCATION 1
#define PATH_WAVEFRONT_SHADE path_wavefront_shade
#include "path_wavefront_body.metal"
