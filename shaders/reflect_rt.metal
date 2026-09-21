// Reflection resolve with the acceleration structure; the composite is compiled once from reflect.metal.
#define BASALT_RAY_TRACING 1
#define RESOLVE_REFLECTIONS resolve_reflections_rt
#define COMPOSITE_REFLECTIONS composite_reflections_rt
#include "reflect_body.metal"
