// Forward entries with the acceleration structure; the vertex entry is shared with forward.metal.
#define BASALT_RAY_TRACING 1
#define FORWARD_VERTEX forward_vertex_rt
#define FORWARD_FRAGMENT forward_fragment_rt
#include "forward_body.metal"
