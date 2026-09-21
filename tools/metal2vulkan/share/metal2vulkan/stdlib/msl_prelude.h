#pragma once
// Prelude for the owned standard library. The Metal language mode of the
// pinned Clang fork supplies the stage keywords (kernel, vertex, fragment),
// the address-space keywords (device, constant, threadgroup, thread) and the
// __METAL_VERSION__ macros; entry and resource semantics are checked by the
// shader frontend. Vector types remain library typedefs, as in Metal.
using uint = unsigned int;
// Placement construction (Metal 4.1): the compiler constructs into the
// addressed storage itself; these declarations only satisfy overload
// resolution and are never called.
void *operator new(unsigned long, thread void *) noexcept;
void *operator new(unsigned long, device void *) noexcept;
void *operator new(unsigned long, threadgroup void *) noexcept;
namespace metal {
// Integer widths (MSL section 2.1): char/uchar 8, short/ushort 16, int/uint 32,
// long/ulong 64 bits; size_t and ptrdiff_t are the 64-bit sizes of the SPIR64 target.
using uint = unsigned int;
using half = _Float16;  // IEEE binary16 with native arithmetic; literals take the h suffix.
using bfloat = __bf16;  // Brain float (Metal 3.1): computed as float, stored in 16 bits; literals take the bf suffix.
using uchar = unsigned char;
using ushort = unsigned short;
using ulong = unsigned long;
typedef unsigned long size_t;
typedef long ptrdiff_t;
#define M2V_VECTOR_TYPES(T) \
  typedef T T##2 __attribute__((ext_vector_type(2))); \
  typedef T T##3 __attribute__((ext_vector_type(3))); \
  typedef T T##4 __attribute__((ext_vector_type(4)));
M2V_VECTOR_TYPES(bool)
M2V_VECTOR_TYPES(float)
M2V_VECTOR_TYPES(half)
M2V_VECTOR_TYPES(bfloat)
M2V_VECTOR_TYPES(int)
M2V_VECTOR_TYPES(uint)
M2V_VECTOR_TYPES(char)
M2V_VECTOR_TYPES(uchar)
M2V_VECTOR_TYPES(short)
M2V_VECTOR_TYPES(ushort)
M2V_VECTOR_TYPES(long)
M2V_VECTOR_TYPES(ulong)
#undef M2V_VECTOR_TYPES
}
