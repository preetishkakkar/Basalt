// Reflection kernels without ray queries; see reflect_body.metal.
#define RESOLVE_REFLECTIONS resolve_reflections
#define COMPOSITE_REFLECTIONS composite_reflections
#include "reflect_body.metal"
