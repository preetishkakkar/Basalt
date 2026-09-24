#define BASALT_RAY_TRACING 1
#define PT_PRIMARY_GBUFFER 1
#define PT_NO_AUX_ACCUMULATION 1
#define PATH_TRACE_ENTRY path_trace_hybrid_rt
#include "path_trace_body.metal"
