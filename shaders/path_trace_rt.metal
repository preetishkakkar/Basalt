// The GPU path tracer through ray queries over the rasteriser's TLAS. See path_trace_body.metal.
#define BASALT_RAY_TRACING 1
#define PATH_TRACE_ENTRY path_trace_rt
#include "path_trace_body.metal"
