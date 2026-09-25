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

// threadgroup_broadcast(value, lane, source, slot) is an msl2spirv builtin (msl2spirv's SHARED_MEMORY.md):
// the invocations whose lane equals source store value at *slot, a barrier, every invocation reads
// it, a second barrier. msl2spirv treats the result as workgroup-uniform, so a loop with barriers
// may exit on it; other Metal compilers get the same sequence here.
#ifndef __METAL2VULKAN__
template <typename T> inline T threadgroup_broadcast(T value, uint lane, uint source, threadgroup T *slot) {
  if (lane == source) slot[0] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const T result = slot[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}
#endif
#endif
