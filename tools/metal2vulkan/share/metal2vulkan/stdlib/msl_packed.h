#pragma once
#include "msl_prelude.h"
// Independently authored packed vector storage types (MSL section 2.2.3).
// Each packed type stores exactly its lanes with the element's alignment. Values
// convert to and from the aligned vector type; the compiler recognizes these
// declarations by their origin in this file and lowers the operations itself.
// No member has a body here, so nothing is inlined from library code. Member
// functions are overloaded per address space so that, as with Apple's builtin
// packed types, device, constant and threadgroup objects convert and index.
namespace metal {
#define M2V_PACKED(T, N, ...) \
  struct packed_##T##N { \
    T __VA_ARGS__; \
    packed_##T##N() = default; \
    packed_##T##N(const packed_##T##N &) = default; \
    packed_##T##N &operator=(const packed_##T##N &) = default; \
    packed_##T##N(T##N value); \
    operator T##N() const; \
    operator T##N() const device; \
    operator T##N() const constant; \
    operator T##N() const threadgroup; \
    T &operator[](int index); \
    const T &operator[](int index) const; \
    device T &operator[](int index) device; \
    const device T &operator[](int index) const device; \
    const constant T &operator[](int index) const constant; \
    threadgroup T &operator[](int index) threadgroup; \
    const threadgroup T &operator[](int index) const threadgroup; \
  };
M2V_PACKED(float, 2, x, y)
M2V_PACKED(float, 3, x, y, z)
M2V_PACKED(float, 4, x, y, z, w)
M2V_PACKED(int, 2, x, y)
M2V_PACKED(int, 3, x, y, z)
M2V_PACKED(int, 4, x, y, z, w)
M2V_PACKED(uint, 2, x, y)
M2V_PACKED(uint, 3, x, y, z)
M2V_PACKED(uint, 4, x, y, z, w)
// Narrow lanes: 16-bit half, bfloat (Metal 3.1), short and ushort; 8-bit char and uchar.
M2V_PACKED(half, 2, x, y)
M2V_PACKED(half, 3, x, y, z)
M2V_PACKED(half, 4, x, y, z, w)
M2V_PACKED(bfloat, 2, x, y)
M2V_PACKED(bfloat, 3, x, y, z)
M2V_PACKED(bfloat, 4, x, y, z, w)
M2V_PACKED(short, 2, x, y)
M2V_PACKED(short, 3, x, y, z)
M2V_PACKED(short, 4, x, y, z, w)
M2V_PACKED(ushort, 2, x, y)
M2V_PACKED(ushort, 3, x, y, z)
M2V_PACKED(ushort, 4, x, y, z, w)
M2V_PACKED(char, 2, x, y)
M2V_PACKED(char, 3, x, y, z)
M2V_PACKED(char, 4, x, y, z, w)
M2V_PACKED(uchar, 2, x, y)
M2V_PACKED(uchar, 3, x, y, z)
M2V_PACKED(uchar, 4, x, y, z, w)
#undef M2V_PACKED
}
