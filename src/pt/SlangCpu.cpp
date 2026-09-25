// The CPU tracer's shared code: shaders/slang/pt/pt_cpu.slang as slangc writes it in C++,
// compiled here inside namespace pt. It defines the structs pt_cpu.gen.hpp declares, so only
// pt/Vector.h comes first: the prelude, with the exports as plain extern "C" functions.
#include "pt/Vector.h"

#ifdef _MSC_VER
#pragma warning(push)
// The generated code: unreferenced locals and parameters, constant conditions, shadowing.
#pragma warning(disable : 4100 4101 4127 4189 4190 4244 4456 4457 4458 4459 4702)
#endif

namespace pt {
// The prelude's F32_min and F32_max call the CRT's fminf and fmaxf, which MSVC does not inline:
// a call per slab test. SPIR-V leaves FMin and FMax undefined for NaN operands, so the shared code
// cannot depend on NaN handling, and these compile to single minss/maxss instructions. Unqualified
// lookup from the generated code below, compiled in this namespace, finds them before the prelude's.
inline float F32_min(float a, float b) { return a < b ? a : b; }
inline float F32_max(float a, float b) { return a > b ? a : b; }

// In pt, not Shared.h's pt::gen: the structs here are distinct types from the header's with the
// same layout, and the exports and ptHost* callbacks are extern "C", so the two sides meet by
// unmangled name only.
#include "pt_cpu.gen.cpp"
} // namespace pt

#ifdef _MSC_VER
#pragma warning(pop)
#endif
