#pragma once
#include "msl_prelude.h"
// Independently authored declarations for the MSL graphics functions: fragment
// derivatives and discard_fragment. The compiler lowers them only in fragment
// entries; no body here.
namespace metal {
#define M2V_GRAPHICS(NAME) __attribute__((annotate("msl.math:" NAME)))
#define M2V_DERIVATIVES(T) \
  T dfdx(T x) M2V_GRAPHICS("dfdx"); \
  T dfdy(T x) M2V_GRAPHICS("dfdy"); \
  T fwidth(T x) M2V_GRAPHICS("fwidth");
M2V_DERIVATIVES(float)
M2V_DERIVATIVES(float2)
M2V_DERIVATIVES(float3)
M2V_DERIVATIVES(float4)
M2V_DERIVATIVES(half)
M2V_DERIVATIVES(half2)
M2V_DERIVATIVES(half3)
M2V_DERIVATIVES(half4)
void discard_fragment() M2V_GRAPHICS("discard_fragment");
// Fragment sample queries (MSL Table 6.20): the number of samples of the multisampled
// color attachment, and the normalized offset of a sample index within the pixel.
uint get_num_samples() M2V_GRAPHICS("get_num_samples");
float2 get_sample_position(uint index) M2V_GRAPHICS("get_sample_position");
// Whether this invocation is a helper: one the rasterizer runs only so that a
// neighbour's derivatives are defined, or one this shader demoted with
// discard_fragment. A helper's writes are discarded (MSL 6.10).
bool simd_is_helper_thread() M2V_GRAPHICS("simd_is_helper_thread");
#undef M2V_DERIVATIVES
#undef M2V_GRAPHICS
}
