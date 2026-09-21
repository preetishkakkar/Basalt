#pragma once
#include "msl_prelude.h"
// Independently authored pack/unpack declarations (MSL section 6.16), for V3.
namespace metal {
#define M2V_INTRINSIC(NAME) __attribute__((annotate("msl.math:" NAME)))
uint pack_float_to_unorm4x8(float4 x) M2V_INTRINSIC("pack_float_to_unorm4x8");
uint pack_float_to_snorm4x8(float4 x) M2V_INTRINSIC("pack_float_to_snorm4x8");
uint pack_float_to_unorm2x16(float2 x) M2V_INTRINSIC("pack_float_to_unorm2x16");
uint pack_float_to_snorm2x16(float2 x) M2V_INTRINSIC("pack_float_to_snorm2x16");
uint pack_float_to_unorm10a2(float4 x) M2V_INTRINSIC("pack_float_to_unorm10a2");
uint pack_float_to_snorm10a2(float4 x) M2V_INTRINSIC("pack_float_to_snorm10a2");
float4 unpack_unorm4x8_to_float(uint x) M2V_INTRINSIC("unpack_unorm4x8_to_float");
float4 unpack_snorm4x8_to_float(uint x) M2V_INTRINSIC("unpack_snorm4x8_to_float");
float2 unpack_unorm2x16_to_float(uint x) M2V_INTRINSIC("unpack_unorm2x16_to_float");
float2 unpack_snorm2x16_to_float(uint x) M2V_INTRINSIC("unpack_snorm2x16_to_float");
float4 unpack_unorm10a2_to_float(uint x) M2V_INTRINSIC("unpack_unorm10a2_to_float");
float4 unpack_snorm10a2_to_float(uint x) M2V_INTRINSIC("unpack_snorm10a2_to_float");
#undef M2V_INTRINSIC
}
