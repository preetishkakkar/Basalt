// Lets a header compile both as MSL and as C++. The C++ side includes src/pt/Shared.h,
// which defines BASALT_HOST and supplies the MSL types, then includes these headers
// inside namespace pt. Rules for shared code: read components with .x .y .z .w only (the
// xyz()/xy() helpers stand in for swizzles); no ternary between two locals and no
// uninitialised locals; buffers are `device const T*`, textures only through
// ptSampleTexture; constants use PT_CONSTANT. Acceleration structures are traversed only
// in an entry function, which is why the path loop is the textual include integrator.inc.
#pragma once
#ifdef BASALT_HOST
#define PT_CONSTANT static constexpr
#else
#include <metal_stdlib>
using namespace metal;
#define PT_CONSTANT constant

// Multi-component swizzles, spelled as calls so C++ can provide the same names.
inline float3 xyz(float4 v) { return v.xyz; }
inline float2 xy(float4 v) { return v.xy; }
inline float2 xy(float3 v) { return v.xy; }
inline float2 zw(float4 v) { return v.zw; }
#endif
