// The GPU path tracer over the software BVH; runs on any Vulkan 1.3 device. See path_trace_body.metal.
#define PATH_TRACE_ENTRY path_trace
#define PT_DISABLE_EMISSIVE 1
#include "path_trace_body.metal"
