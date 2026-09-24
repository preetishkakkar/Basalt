// The shader headers under shaders/shared and shaders/pt, compiled as C++ inside
// namespace pt: one implementation for the GPU and the CPU.
#pragma once
#include "pt/shim.h"

// Address-space and reference qualifiers have no C++ meaning; defined only around the
// shared headers, so std::thread and friends are untouched.
#define device
#define thread
namespace pt {
#include "shared/prelude.h"
#include "shared/types.h"
#include "shared/random.h"
#include "shared/shading.h"
} // namespace pt
#undef device
#undef thread

namespace pt {
// Russian roulette starts once a path has scattered this many times (PathUniforms.path.y),
// in every backend: the headless CLI, the window's CPU tracer and the GPU tracers.
inline constexpr float kRouletteStartBounce = 3.0f;
static_assert(sizeof(Material) == 128, "Material must match the MSL layout");
static_assert(sizeof(Light) == 64, "Light must match the MSL layout");
static_assert(sizeof(TraceInstance) == 176, "TraceInstance must match the MSL layout");
} // namespace pt
