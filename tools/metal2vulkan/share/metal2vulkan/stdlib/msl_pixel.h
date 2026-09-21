#pragma once
#include "msl_prelude.h"
// Independently authored pixel data types (MSL section 2.7, table 2.6):
// normalized-integer and packed storage formats whose loads and stores
// convert to and from an arithmetic (ALU) type. The compiler recognizes these
// declarations by their origin in this file and lowers the conversions itself
// (docs/PIXEL_TYPES.md); nothing here has a body. Assignments and conversions
// are the only operations, as in Metal. The member spells the storage size.
namespace metal {
#define M2V_PIXEL(NAME, STORAGE) \
  template <typename T> struct NAME { \
    STORAGE; \
    NAME() = default; \
    NAME(const NAME &) = default; \
    NAME &operator=(const NAME &) = default; \
    NAME(T value); \
    operator T() const; \
    operator T() const device; \
    operator T() const constant; \
    operator T() const threadgroup; \
  };
M2V_PIXEL(r8unorm, uchar v)
M2V_PIXEL(r8snorm, char v)
M2V_PIXEL(r16unorm, ushort v)
M2V_PIXEL(r16snorm, short v)
M2V_PIXEL(rg8unorm, uchar v[2])
M2V_PIXEL(rg8snorm, char v[2])
M2V_PIXEL(rg16unorm, ushort v[2])
M2V_PIXEL(rg16snorm, short v[2])
M2V_PIXEL(rgba8unorm, uchar v[4])
M2V_PIXEL(srgba8unorm, uchar v[4])
M2V_PIXEL(rgba8snorm, char v[4])
M2V_PIXEL(rgba16unorm, ushort v[4])
M2V_PIXEL(rgba16snorm, short v[4])
M2V_PIXEL(rgb10a2, uint v)
M2V_PIXEL(rg11b10f, uint v)
M2V_PIXEL(rgb9e5, uint v)
#undef M2V_PIXEL
}
