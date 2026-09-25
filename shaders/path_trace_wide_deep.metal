// path_trace_wide with the deep traversal stack, for trees whose stack bound exceeds 64.
#define PT_WIDE_BVH_STACK PT_WIDE_BVH_STACK_DEEP
#define BASALT_WIDE_BVH 1
#define PATH_TRACE_ENTRY path_trace_wide_deep
#define PT_DISABLE_EMISSIVE 1
#include "path_trace_body.metal"
