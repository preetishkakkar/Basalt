// Full Vulkan ray-pipeline path tracer. The shared body owns the same iterative
// integrator as the CPU, inline-query and custom-BVH backends.
#define BASALT_RAY_PIPELINE 1
#define PATH_TRACE_ENTRY path_trace_pipeline
#include "path_trace_body.metal"
