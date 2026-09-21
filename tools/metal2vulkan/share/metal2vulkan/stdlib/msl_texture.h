#pragma once
#include "msl_prelude.h"
// Independently authored texture and sampler declarations based on MSL 2.9
// and 6.13. The frontend and typed IR enforce supported components, accesses,
// stages and language versions; declarations alone do not imply support.
// See registry/features.json and docs/TEXTURES.md for qualified boundaries.
// The compiler recognizes these declarations by their owned-file origin.
// Constant/device overloads serve handles reached through argument buffers.
namespace metal {
enum class access { sample, read, write, read_write };
// Sampler state (MSL 2.10): the options of a constexpr sampler. The compiler
// reads them from the constexpr declaration and reflects them for the host.
enum class coord { normalized, pixel };
enum class filter { nearest, linear };
enum class min_filter { nearest, linear };
enum class mag_filter { nearest, linear };
enum class s_address { clamp_to_zero, clamp_to_edge, repeat, mirrored_repeat, clamp_to_border };
enum class t_address { clamp_to_zero, clamp_to_edge, repeat, mirrored_repeat, clamp_to_border };
enum class r_address { clamp_to_zero, clamp_to_edge, repeat, mirrored_repeat, clamp_to_border };
enum class address { clamp_to_zero, clamp_to_edge, repeat, mirrored_repeat, clamp_to_border };
enum class mip_filter { none, nearest, linear };
enum class compare_func { none, less, less_equal, greater, greater_equal, equal, not_equal, always, never };
enum class reduction { weighted_average, minimum, maximum };
enum class border_color { transparent_black, opaque_black, opaque_white };
struct max_anisotropy { constexpr explicit max_anisotropy(int value) : value(value) {} int value; };
struct lod_clamp { constexpr lod_clamp(float min, float max) : min(min), max(max) {} float min, max; };
struct sampler {
  sampler() = default;
  sampler() constant = default; // A program-scope handle (MSL 5.2.1, 2v): the host binds it.
  sampler(const sampler &) = default;
  // constexpr sampler s(options...): every option is one of the enumerators or
  // structs above; the compiler rejects duplicates and unsupported states.
  template <typename... Options> constexpr sampler(Options... options) {}
};
// Sampling controls (MSL 6.12.3): an explicit level of detail, a bias added
// to the implicit level, or explicit gradients; the compiler lowers the
// constructor arguments to the matching SPIR-V image operands.
struct level { explicit level(float lod); };
struct bias { explicit bias(float value); };
struct min_lod_clamp { explicit min_lod_clamp(float value); };
struct gradient2d { gradient2d(float2 dPdx, float2 dPdy); };
struct gradient3d { gradient3d(float3 dPdx, float3 dPdy); };
struct gradientcube { gradientcube(float3 dPdx, float3 dPdy); };
enum class component { x, y, z, w }; // The texel component a gather returns.
// The result of a sparse texture read (MSL 6.13): the texel and whether every texel it touched was
// resident. An ordinary struct: the compiler fills both members from the sparse read.
template <typename T> struct sparse_color {
  T value_;
  bool resident_;
  constexpr T value() const { return value_; }
  constexpr bool resident() const { return resident_; }
};
template <typename E, typename L> struct imageblock_slice; // msl_imageblocks.h: a slice write parses, then tile memory is refused.
// Every texture method exists for handles reached directly and through constant or device argument buffers.
#define M2V_TEXTURE_METHOD(declaration) declaration const; declaration const constant; declaration const device;
template <typename T, access A = access::read> struct texture2d_ms {
  typedef T texel_type __attribute__((ext_vector_type(4)));
  texture2d_ms() = default;
  texture2d_ms() constant = default; // A program-scope handle (MSL 5.2.1, 2v): the host binds it.
  texture2d_ms(const texture2d_ms &) = default;
  M2V_TEXTURE_METHOD(texel_type read(uint2 coordinate, uint sample))
  M2V_TEXTURE_METHOD(uint get_width())
  M2V_TEXTURE_METHOD(uint get_height())
  M2V_TEXTURE_METHOD(uint get_num_samples())
};
template <typename T, access A = access::read> struct depth2d_ms {
  depth2d_ms() = default;
  depth2d_ms() constant = default; // A program-scope handle (MSL 5.2.1, 2v): the host binds it.
  depth2d_ms(const depth2d_ms &) = default;
  M2V_TEXTURE_METHOD(T read(uint2 coordinate, uint sample))
  M2V_TEXTURE_METHOD(uint get_width())
  M2V_TEXTURE_METHOD(uint get_height())
  M2V_TEXTURE_METHOD(uint get_num_samples())
};
template <typename T, access A = access::read> struct texture_buffer {
  typedef T texel_type __attribute__((ext_vector_type(4)));
  texture_buffer() = default;
  texture_buffer() constant = default; // A program-scope handle (MSL 5.2.1, 2v): the host binds it.
  texture_buffer(const texture_buffer &) = default;
  M2V_TEXTURE_METHOD(texel_type atomic_load(uint coordinate))
  M2V_TEXTURE_METHOD(void atomic_store(texel_type value, uint coordinate))
  M2V_TEXTURE_METHOD(texel_type atomic_exchange(uint coordinate, texel_type value))
  M2V_TEXTURE_METHOD(bool atomic_compare_exchange_weak(uint coordinate, thread texel_type* expected, texel_type desired))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_add(uint coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_sub(uint coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_min(uint coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_max(uint coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_and(uint coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_or(uint coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_xor(uint coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type read(uint coordinate))
  M2V_TEXTURE_METHOD(void write(texel_type value, uint coordinate))
  M2V_TEXTURE_METHOD(uint get_width())
  M2V_TEXTURE_METHOD(void fence())
};

template <typename T, access A = access::read> struct texture2d_ms_array {
  typedef T texel_type __attribute__((ext_vector_type(4)));
  texture2d_ms_array() = default;
  texture2d_ms_array() constant = default; // A program-scope handle (MSL 5.2.1, 2v): the host binds it.
  texture2d_ms_array(const texture2d_ms_array &) = default;
  M2V_TEXTURE_METHOD(texel_type read(uint2 coordinate, uint layer, uint sample))
  M2V_TEXTURE_METHOD(uint get_width())
  M2V_TEXTURE_METHOD(uint get_height())
  M2V_TEXTURE_METHOD(uint get_array_size())
  M2V_TEXTURE_METHOD(uint get_num_samples())
};
template <typename T, access A = access::read> struct depth2d_ms_array {
  depth2d_ms_array() = default;
  depth2d_ms_array() constant = default; // A program-scope handle (MSL 5.2.1, 2v): the host binds it.
  depth2d_ms_array(const depth2d_ms_array &) = default;
  M2V_TEXTURE_METHOD(T read(uint2 coordinate, uint layer, uint sample))
  M2V_TEXTURE_METHOD(uint get_width())
  M2V_TEXTURE_METHOD(uint get_height())
  M2V_TEXTURE_METHOD(uint get_array_size())
  M2V_TEXTURE_METHOD(uint get_num_samples())
};

template <typename T, access A = access::sample> struct texture2d {
  typedef T texel_type __attribute__((ext_vector_type(4)));
  texture2d() = default;
  texture2d() constant = default; // A program-scope handle (MSL 5.2.1, 2v): the host binds it.
  texture2d(const texture2d &) = default;
  M2V_TEXTURE_METHOD(float calculate_unclamped_lod(sampler s, float2 coordinate))
  M2V_TEXTURE_METHOD(float calculate_clamped_lod(sampler s, float2 coordinate))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float2 coordinate, min_lod_clamp minimum, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float2 coordinate, bias b, min_lod_clamp minimum, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float2 coordinate, gradient2d g, min_lod_clamp minimum, int2 offset = int2(0)))
  // offset is a compile-time constant texel offset in [-8, 7] (SPIR-V ConstOffset).
  texel_type sample(sampler s, float2 coordinate, int2 offset = int2(0)) const;
  texel_type sample(sampler s, float2 coordinate, int2 offset = int2(0)) const constant;
  texel_type sample(sampler s, float2 coordinate, int2 offset = int2(0)) const device;
  texel_type sample(sampler s, float2 coordinate, bias b, int2 offset = int2(0)) const;
  texel_type sample(sampler s, float2 coordinate, bias b, int2 offset = int2(0)) const constant;
  texel_type sample(sampler s, float2 coordinate, bias b, int2 offset = int2(0)) const device;
  texel_type sample(sampler s, float2 coordinate, level l, int2 offset = int2(0)) const;
  texel_type sample(sampler s, float2 coordinate, level l, int2 offset = int2(0)) const constant;
  texel_type sample(sampler s, float2 coordinate, level l, int2 offset = int2(0)) const device;
  texel_type sample(sampler s, float2 coordinate, gradient2d g, int2 offset = int2(0)) const;
  texel_type sample(sampler s, float2 coordinate, gradient2d g, int2 offset = int2(0)) const constant;
  texel_type sample(sampler s, float2 coordinate, gradient2d g, int2 offset = int2(0)) const device;
  M2V_TEXTURE_METHOD(texel_type gather(sampler s, float2 coordinate, int2 offset = int2(0), component c = component::x))
  // Sparse reads (MSL 6.13, docs/TEXTURES.md): the same forms, with residency.
  M2V_TEXTURE_METHOD(sparse_color<texel_type> sparse_sample(sampler s, float2 coordinate, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(sparse_color<texel_type> sparse_sample(sampler s, float2 coordinate, bias b, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(sparse_color<texel_type> sparse_sample(sampler s, float2 coordinate, level l, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(sparse_color<texel_type> sparse_sample(sampler s, float2 coordinate, gradient2d g, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(sparse_color<texel_type> sparse_sample(sampler s, float2 coordinate, min_lod_clamp minimum, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(sparse_color<texel_type> sparse_gather(sampler s, float2 coordinate, int2 offset = int2(0), component c = component::x))
  texel_type read(uint2 coordinate, uint lod = 0) const;
  texel_type read(uint2 coordinate, uint lod = 0) const constant;
  texel_type read(uint2 coordinate, uint lod = 0) const device;
  // fence() (MSL 2.9): orders this thread's writes to the texture before its
  // own later reads of it. Requires access::read_write.
  M2V_TEXTURE_METHOD(void fence())
  // Texture atomics (Metal 3.1): relaxed operations on the first component of one texel of an int or uint
  // read_write texture; values are the 4-vector texel type (results splat the observed component).
  M2V_TEXTURE_METHOD(texel_type atomic_load(uint2 coordinate))
  M2V_TEXTURE_METHOD(void atomic_store(texel_type value, uint2 coordinate))
  M2V_TEXTURE_METHOD(texel_type atomic_exchange(uint2 coordinate, texel_type value))
  M2V_TEXTURE_METHOD(bool atomic_compare_exchange_weak(uint2 coordinate, thread texel_type* expected, texel_type desired))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_add(uint2 coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_sub(uint2 coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_min(uint2 coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_max(uint2 coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_and(uint2 coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_or(uint2 coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_xor(uint2 coordinate, texel_type value))
  void write(texel_type color, uint2 coordinate, uint lod = 0) const;
  void write(texel_type color, uint2 coordinate, uint lod = 0) const constant;
  template <typename E, typename L> void write(imageblock_slice<E, L> slice, ushort2 coordinate) const; // MSL 6.14.3 (docs/TILE_MODEL.md)
  void write(texel_type color, uint2 coordinate, uint lod = 0) const device;
  uint get_width(uint lod = 0) const;
  uint get_width(uint lod = 0) const constant;
  uint get_width(uint lod = 0) const device;
  uint get_height(uint lod = 0) const;
  uint get_height(uint lod = 0) const constant;
  uint get_height(uint lod = 0) const device;
  uint get_num_mip_levels() const;
  uint get_num_mip_levels() const constant;
  uint get_num_mip_levels() const device;
};
// The other texture types (MSL 2.9): 1D textures sample without LOD controls
// or offsets, arrays take the array slice after the coordinate, 3D textures
// take float3/uint3 coordinates and int3 offsets, cube textures take a
// direction (and a face for reads and writes) and no offset.
template <typename T, access A = access::sample> struct texture1d {
  typedef T texel_type __attribute__((ext_vector_type(4)));
  texture1d() = default;
  texture1d() constant = default; // A program-scope handle (MSL 5.2.1, 2v): the host binds it.
  texture1d(const texture1d &) = default;
  // fence() (MSL 2.9): orders this thread's writes to the texture before its
  // own later reads of it. Requires access::read_write.
  M2V_TEXTURE_METHOD(void fence())
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float coordinate))
  M2V_TEXTURE_METHOD(texel_type atomic_load(uint coordinate))
  M2V_TEXTURE_METHOD(void atomic_store(texel_type value, uint coordinate))
  M2V_TEXTURE_METHOD(texel_type atomic_exchange(uint coordinate, texel_type value))
  M2V_TEXTURE_METHOD(bool atomic_compare_exchange_weak(uint coordinate, thread texel_type* expected, texel_type desired))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_add(uint coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_sub(uint coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_min(uint coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_max(uint coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_and(uint coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_or(uint coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_xor(uint coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type read(uint coordinate, uint lod = 0))
  M2V_TEXTURE_METHOD(void write(texel_type color, uint coordinate, uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_width(uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_num_mip_levels())
};
template <typename T, access A = access::sample> struct texture1d_array {
  typedef T texel_type __attribute__((ext_vector_type(4)));
  texture1d_array() = default;
  texture1d_array() constant = default; // A program-scope handle (MSL 5.2.1, 2v): the host binds it.
  texture1d_array(const texture1d_array &) = default;
  // fence() (MSL 2.9): orders this thread's writes to the texture before its
  // own later reads of it. Requires access::read_write.
  M2V_TEXTURE_METHOD(void fence())
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float coordinate, uint array))
  M2V_TEXTURE_METHOD(texel_type atomic_load(uint coordinate, uint array))
  M2V_TEXTURE_METHOD(void atomic_store(texel_type value, uint coordinate, uint array))
  M2V_TEXTURE_METHOD(texel_type atomic_exchange(uint coordinate, uint array, texel_type value))
  M2V_TEXTURE_METHOD(bool atomic_compare_exchange_weak(uint coordinate, uint array, thread texel_type* expected, texel_type desired))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_add(uint coordinate, uint array, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_sub(uint coordinate, uint array, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_min(uint coordinate, uint array, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_max(uint coordinate, uint array, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_and(uint coordinate, uint array, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_or(uint coordinate, uint array, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_xor(uint coordinate, uint array, texel_type value))
  M2V_TEXTURE_METHOD(texel_type read(uint coordinate, uint array, uint lod = 0))
  M2V_TEXTURE_METHOD(void write(texel_type color, uint coordinate, uint array, uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_width(uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_array_size())
  M2V_TEXTURE_METHOD(uint get_num_mip_levels())
};
template <typename T, access A = access::sample> struct texture2d_array {
  typedef T texel_type __attribute__((ext_vector_type(4)));
  texture2d_array() = default;
  texture2d_array() constant = default; // A program-scope handle (MSL 5.2.1, 2v): the host binds it.
  texture2d_array(const texture2d_array &) = default;
  M2V_TEXTURE_METHOD(float calculate_unclamped_lod(sampler s, float2 coordinate))
  M2V_TEXTURE_METHOD(float calculate_clamped_lod(sampler s, float2 coordinate))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float2 coordinate, uint array, min_lod_clamp minimum, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float2 coordinate, uint array, bias b, min_lod_clamp minimum, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float2 coordinate, uint array, gradient2d g, min_lod_clamp minimum, int2 offset = int2(0)))
  // fence() (MSL 2.9): orders this thread's writes to the texture before its
  // own later reads of it. Requires access::read_write.
  M2V_TEXTURE_METHOD(void fence())
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float2 coordinate, uint array, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float2 coordinate, uint array, bias b, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float2 coordinate, uint array, level l, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float2 coordinate, uint array, gradient2d g, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(texel_type gather(sampler s, float2 coordinate, uint array, int2 offset = int2(0), component c = component::x))
  M2V_TEXTURE_METHOD(texel_type atomic_load(uint2 coordinate, uint array))
  M2V_TEXTURE_METHOD(void atomic_store(texel_type value, uint2 coordinate, uint array))
  M2V_TEXTURE_METHOD(texel_type atomic_exchange(uint2 coordinate, uint array, texel_type value))
  M2V_TEXTURE_METHOD(bool atomic_compare_exchange_weak(uint2 coordinate, uint array, thread texel_type* expected, texel_type desired))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_add(uint2 coordinate, uint array, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_sub(uint2 coordinate, uint array, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_min(uint2 coordinate, uint array, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_max(uint2 coordinate, uint array, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_and(uint2 coordinate, uint array, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_or(uint2 coordinate, uint array, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_xor(uint2 coordinate, uint array, texel_type value))
  M2V_TEXTURE_METHOD(texel_type read(uint2 coordinate, uint array, uint lod = 0))
  M2V_TEXTURE_METHOD(void write(texel_type color, uint2 coordinate, uint array, uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_width(uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_height(uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_array_size())
  M2V_TEXTURE_METHOD(uint get_num_mip_levels())
};
template <typename T, access A = access::sample> struct texture3d {
  typedef T texel_type __attribute__((ext_vector_type(4)));
  texture3d() = default;
  texture3d() constant = default; // A program-scope handle (MSL 5.2.1, 2v): the host binds it.
  texture3d(const texture3d &) = default;
  M2V_TEXTURE_METHOD(float calculate_unclamped_lod(sampler s, float3 coordinate))
  M2V_TEXTURE_METHOD(float calculate_clamped_lod(sampler s, float3 coordinate))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float3 coordinate, min_lod_clamp minimum, int3 offset = int3(0)))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float3 coordinate, bias b, min_lod_clamp minimum, int3 offset = int3(0)))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float3 coordinate, gradient3d g, min_lod_clamp minimum, int3 offset = int3(0)))
  // fence() (MSL 2.9): orders this thread's writes to the texture before its
  // own later reads of it. Requires access::read_write.
  M2V_TEXTURE_METHOD(void fence())
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float3 coordinate, int3 offset = int3(0)))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float3 coordinate, bias b, int3 offset = int3(0)))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float3 coordinate, level l, int3 offset = int3(0)))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float3 coordinate, gradient3d g, int3 offset = int3(0)))
  M2V_TEXTURE_METHOD(texel_type atomic_load(uint3 coordinate))
  M2V_TEXTURE_METHOD(void atomic_store(texel_type value, uint3 coordinate))
  M2V_TEXTURE_METHOD(texel_type atomic_exchange(uint3 coordinate, texel_type value))
  M2V_TEXTURE_METHOD(bool atomic_compare_exchange_weak(uint3 coordinate, thread texel_type* expected, texel_type desired))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_add(uint3 coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_sub(uint3 coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_min(uint3 coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_max(uint3 coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_and(uint3 coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_or(uint3 coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_xor(uint3 coordinate, texel_type value))
  M2V_TEXTURE_METHOD(texel_type read(uint3 coordinate, uint lod = 0))
  M2V_TEXTURE_METHOD(void write(texel_type color, uint3 coordinate, uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_width(uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_height(uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_depth(uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_num_mip_levels())
};
template <typename T, access A = access::sample> struct texturecube {
  typedef T texel_type __attribute__((ext_vector_type(4)));
  texturecube() = default;
  texturecube() constant = default; // A program-scope handle (MSL 5.2.1, 2v): the host binds it.
  texturecube(const texturecube &) = default;
  M2V_TEXTURE_METHOD(float calculate_unclamped_lod(sampler s, float3 coordinate))
  M2V_TEXTURE_METHOD(float calculate_clamped_lod(sampler s, float3 coordinate))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float3 coordinate, min_lod_clamp minimum))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float3 coordinate, bias b, min_lod_clamp minimum))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float3 coordinate, gradientcube g, min_lod_clamp minimum))
  // fence() (MSL 2.9): orders this thread's writes to the texture before its
  // own later reads of it. Requires access::read_write.
  M2V_TEXTURE_METHOD(void fence())
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float3 coordinate))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float3 coordinate, bias b))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float3 coordinate, level l))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float3 coordinate, gradientcube g))
  M2V_TEXTURE_METHOD(texel_type gather(sampler s, float3 coordinate, component c = component::x))
  M2V_TEXTURE_METHOD(texel_type atomic_load(uint2 coordinate, uint face))
  M2V_TEXTURE_METHOD(void atomic_store(texel_type value, uint2 coordinate, uint face))
  M2V_TEXTURE_METHOD(texel_type atomic_exchange(uint2 coordinate, uint face, texel_type value))
  M2V_TEXTURE_METHOD(bool atomic_compare_exchange_weak(uint2 coordinate, uint face, thread texel_type* expected, texel_type desired))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_add(uint2 coordinate, uint face, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_sub(uint2 coordinate, uint face, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_min(uint2 coordinate, uint face, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_max(uint2 coordinate, uint face, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_and(uint2 coordinate, uint face, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_or(uint2 coordinate, uint face, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_xor(uint2 coordinate, uint face, texel_type value))
  M2V_TEXTURE_METHOD(texel_type read(uint2 coordinate, uint face, uint lod = 0))
  M2V_TEXTURE_METHOD(void write(texel_type color, uint2 coordinate, uint face, uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_width(uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_height(uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_num_mip_levels())
};
template <typename T, access A = access::sample> struct texturecube_array {
  typedef T texel_type __attribute__((ext_vector_type(4)));
  texturecube_array() = default;
  texturecube_array() constant = default; // A program-scope handle (MSL 5.2.1, 2v): the host binds it.
  texturecube_array(const texturecube_array &) = default;
  M2V_TEXTURE_METHOD(float calculate_unclamped_lod(sampler s, float3 coordinate))
  M2V_TEXTURE_METHOD(float calculate_clamped_lod(sampler s, float3 coordinate))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float3 coordinate, uint array, min_lod_clamp minimum))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float3 coordinate, uint array, bias b, min_lod_clamp minimum))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float3 coordinate, uint array, gradientcube g, min_lod_clamp minimum))
  // fence() (MSL 2.9): orders this thread's writes to the texture before its
  // own later reads of it. Requires access::read_write.
  M2V_TEXTURE_METHOD(void fence())
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float3 coordinate, uint array))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float3 coordinate, uint array, bias b))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float3 coordinate, uint array, level l))
  M2V_TEXTURE_METHOD(texel_type sample(sampler s, float3 coordinate, uint array, gradientcube g))
  M2V_TEXTURE_METHOD(texel_type gather(sampler s, float3 coordinate, uint array, component c = component::x))
  M2V_TEXTURE_METHOD(texel_type atomic_load(uint2 coordinate, uint face, uint array))
  M2V_TEXTURE_METHOD(void atomic_store(texel_type value, uint2 coordinate, uint face, uint array))
  M2V_TEXTURE_METHOD(texel_type atomic_exchange(uint2 coordinate, uint face, uint array, texel_type value))
  M2V_TEXTURE_METHOD(bool atomic_compare_exchange_weak(uint2 coordinate, uint face, uint array, thread texel_type* expected, texel_type desired))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_add(uint2 coordinate, uint face, uint array, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_sub(uint2 coordinate, uint face, uint array, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_min(uint2 coordinate, uint face, uint array, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_max(uint2 coordinate, uint face, uint array, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_and(uint2 coordinate, uint face, uint array, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_or(uint2 coordinate, uint face, uint array, texel_type value))
  M2V_TEXTURE_METHOD(texel_type atomic_fetch_xor(uint2 coordinate, uint face, uint array, texel_type value))
  M2V_TEXTURE_METHOD(texel_type read(uint2 coordinate, uint face, uint array, uint lod = 0))
  M2V_TEXTURE_METHOD(void write(texel_type color, uint2 coordinate, uint face, uint array, uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_width(uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_height(uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_array_size())
  M2V_TEXTURE_METHOD(uint get_num_mip_levels())
};
// Depth textures (MSL 2.9): float texels read as scalars; sample_compare and
// gather_compare compare a reference value through a comparison sampler
// (compare_func); depth textures are sampled (never written).
template <typename T = float, access A = access::sample> struct depth2d {
  typedef T texel_type __attribute__((ext_vector_type(4)));
  depth2d() = default;
  depth2d() constant = default; // A program-scope handle (MSL 5.2.1, 2v): the host binds it.
  depth2d(const depth2d &) = default;
  M2V_TEXTURE_METHOD(float calculate_unclamped_lod(sampler s, float2 coordinate))
  M2V_TEXTURE_METHOD(float calculate_clamped_lod(sampler s, float2 coordinate))
  M2V_TEXTURE_METHOD(T sample(sampler s, float2 coordinate, min_lod_clamp minimum, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample(sampler s, float2 coordinate, bias b, min_lod_clamp minimum, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample(sampler s, float2 coordinate, gradient2d g, min_lod_clamp minimum, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float2 coordinate, float compare_value, min_lod_clamp minimum, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float2 coordinate, float compare_value, bias b, min_lod_clamp minimum, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float2 coordinate, float compare_value, gradient2d g, min_lod_clamp minimum, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(sparse_color<T> sparse_sample(sampler s, float2 coordinate, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(sparse_color<T> sparse_sample_compare(sampler s, float2 coordinate, float compare_value, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample(sampler s, float2 coordinate, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample(sampler s, float2 coordinate, bias b, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample(sampler s, float2 coordinate, level l, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample(sampler s, float2 coordinate, gradient2d g, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float2 coordinate, float compare_value, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float2 coordinate, float compare_value, level l, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float2 coordinate, float compare_value, bias b, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float2 coordinate, float compare_value, gradient2d g, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T read(uint2 coordinate, uint lod = 0))
  M2V_TEXTURE_METHOD(texel_type gather(sampler s, float2 coordinate, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(texel_type gather_compare(sampler s, float2 coordinate, float compare_value, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(uint get_width(uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_height(uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_num_mip_levels())
};
template <typename T = float, access A = access::sample> struct depth2d_array {
  typedef T texel_type __attribute__((ext_vector_type(4)));
  depth2d_array() = default;
  depth2d_array() constant = default; // A program-scope handle (MSL 5.2.1, 2v): the host binds it.
  depth2d_array(const depth2d_array &) = default;
  M2V_TEXTURE_METHOD(float calculate_unclamped_lod(sampler s, float2 coordinate))
  M2V_TEXTURE_METHOD(float calculate_clamped_lod(sampler s, float2 coordinate))
  M2V_TEXTURE_METHOD(T sample(sampler s, float2 coordinate, uint array, min_lod_clamp minimum, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample(sampler s, float2 coordinate, uint array, bias b, min_lod_clamp minimum, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample(sampler s, float2 coordinate, uint array, gradient2d g, min_lod_clamp minimum, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float2 coordinate, uint array, float compare_value, min_lod_clamp minimum, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float2 coordinate, uint array, float compare_value, bias b, min_lod_clamp minimum, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float2 coordinate, uint array, float compare_value, gradient2d g, min_lod_clamp minimum, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample(sampler s, float2 coordinate, uint array, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample(sampler s, float2 coordinate, uint array, level l, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample(sampler s, float2 coordinate, uint array, bias b, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample(sampler s, float2 coordinate, uint array, gradient2d g, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float2 coordinate, uint array, float compare_value, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float2 coordinate, uint array, float compare_value, level l, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float2 coordinate, uint array, float compare_value, bias b, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float2 coordinate, uint array, float compare_value, gradient2d g, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(T read(uint2 coordinate, uint array, uint lod = 0))
  M2V_TEXTURE_METHOD(texel_type gather(sampler s, float2 coordinate, uint array, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(texel_type gather_compare(sampler s, float2 coordinate, uint array, float compare_value, int2 offset = int2(0)))
  M2V_TEXTURE_METHOD(uint get_width(uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_height(uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_array_size())
  M2V_TEXTURE_METHOD(uint get_num_mip_levels())
};
template <typename T = float, access A = access::sample> struct depthcube {
  typedef T texel_type __attribute__((ext_vector_type(4)));
  depthcube() = default;
  depthcube() constant = default; // A program-scope handle (MSL 5.2.1, 2v): the host binds it.
  depthcube(const depthcube &) = default;
  M2V_TEXTURE_METHOD(float calculate_unclamped_lod(sampler s, float3 coordinate))
  M2V_TEXTURE_METHOD(float calculate_clamped_lod(sampler s, float3 coordinate))
  M2V_TEXTURE_METHOD(T sample(sampler s, float3 coordinate, min_lod_clamp minimum))
  M2V_TEXTURE_METHOD(T sample(sampler s, float3 coordinate, bias b, min_lod_clamp minimum))
  M2V_TEXTURE_METHOD(T sample(sampler s, float3 coordinate, gradientcube g, min_lod_clamp minimum))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float3 coordinate, float compare_value, min_lod_clamp minimum))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float3 coordinate, float compare_value, bias b, min_lod_clamp minimum))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float3 coordinate, float compare_value, gradientcube g, min_lod_clamp minimum))
  M2V_TEXTURE_METHOD(T sample(sampler s, float3 coordinate))
  M2V_TEXTURE_METHOD(T sample(sampler s, float3 coordinate, level l))
  M2V_TEXTURE_METHOD(T sample(sampler s, float3 coordinate, bias b))
  M2V_TEXTURE_METHOD(T sample(sampler s, float3 coordinate, gradientcube g))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float3 coordinate, float compare_value))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float3 coordinate, float compare_value, level l))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float3 coordinate, float compare_value, bias b))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float3 coordinate, float compare_value, gradientcube g))
  M2V_TEXTURE_METHOD(texel_type gather(sampler s, float3 coordinate))
  M2V_TEXTURE_METHOD(texel_type gather_compare(sampler s, float3 coordinate, float compare_value))
  M2V_TEXTURE_METHOD(uint get_width(uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_height(uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_num_mip_levels())
};
template <typename T = float, access A = access::sample> struct depthcube_array {
  typedef T texel_type __attribute__((ext_vector_type(4)));
  depthcube_array() = default;
  depthcube_array() constant = default; // A program-scope handle (MSL 5.2.1, 2v): the host binds it.
  depthcube_array(const depthcube_array &) = default;
  M2V_TEXTURE_METHOD(float calculate_unclamped_lod(sampler s, float3 coordinate))
  M2V_TEXTURE_METHOD(float calculate_clamped_lod(sampler s, float3 coordinate))
  M2V_TEXTURE_METHOD(T sample(sampler s, float3 coordinate, uint array, min_lod_clamp minimum))
  M2V_TEXTURE_METHOD(T sample(sampler s, float3 coordinate, uint array, bias b, min_lod_clamp minimum))
  M2V_TEXTURE_METHOD(T sample(sampler s, float3 coordinate, uint array, gradientcube g, min_lod_clamp minimum))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float3 coordinate, uint array, float compare_value, min_lod_clamp minimum))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float3 coordinate, uint array, float compare_value, bias b, min_lod_clamp minimum))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float3 coordinate, uint array, float compare_value, gradientcube g, min_lod_clamp minimum))
  M2V_TEXTURE_METHOD(T sample(sampler s, float3 coordinate, uint array))
  M2V_TEXTURE_METHOD(T sample(sampler s, float3 coordinate, uint array, level l))
  M2V_TEXTURE_METHOD(T sample(sampler s, float3 coordinate, uint array, bias b))
  M2V_TEXTURE_METHOD(T sample(sampler s, float3 coordinate, uint array, gradientcube g))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float3 coordinate, uint array, float compare_value))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float3 coordinate, uint array, float compare_value, level l))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float3 coordinate, uint array, float compare_value, bias b))
  M2V_TEXTURE_METHOD(T sample_compare(sampler s, float3 coordinate, uint array, float compare_value, gradientcube g))
  M2V_TEXTURE_METHOD(texel_type gather(sampler s, float3 coordinate, uint array))
  M2V_TEXTURE_METHOD(texel_type gather_compare(sampler s, float3 coordinate, uint array, float compare_value))
  M2V_TEXTURE_METHOD(uint get_width(uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_height(uint lod = 0))
  M2V_TEXTURE_METHOD(uint get_array_size())
  M2V_TEXTURE_METHOD(uint get_num_mip_levels())
};
#undef M2V_TEXTURE_METHOD
// Null tests report whether the host left a declared-nullable texture unbound.
// Zero sample/read/query results are a Metal2Vulkan extension: MSL 4.1 6.13.18
// leaves member calls on null textures undefined. Writable nulls are rejected.
#define M2V_NULL_TEXTURE(KIND) \
  template <typename T, access A> bool is_null_texture(KIND<T, A> t) __attribute__((annotate("msl.math:is_null_texture")));
M2V_NULL_TEXTURE(texture1d)
M2V_NULL_TEXTURE(texture1d_array)
M2V_NULL_TEXTURE(texture2d)
M2V_NULL_TEXTURE(texture2d_array)
M2V_NULL_TEXTURE(texture3d)
M2V_NULL_TEXTURE(texturecube)
M2V_NULL_TEXTURE(texturecube_array)
M2V_NULL_TEXTURE(depth2d)
M2V_NULL_TEXTURE(depth2d_array)
M2V_NULL_TEXTURE(depthcube)
M2V_NULL_TEXTURE(depthcube_array)
M2V_NULL_TEXTURE(texture_buffer)
M2V_NULL_TEXTURE(texture2d_ms)
M2V_NULL_TEXTURE(depth2d_ms)
M2V_NULL_TEXTURE(texture2d_ms_array)
M2V_NULL_TEXTURE(depth2d_ms_array)
#undef M2V_NULL_TEXTURE
// Fixed arrays of textures or samplers (MSL 2.12.1) and the immutable view a
// helper takes; the compiler resolves indexing and size() itself. Indices and
// sizes are 32-bit in this profile (MSL declares size_t).
template <typename T, size_t N> struct array {
  T __elements_[N]; // An aggregate, as Apple's: array<int, 2>{8, 1} spells tensor strides (msl_tensor.h).
  array() = default;
  array() constant = default; // A program-scope handle (MSL 5.2.1, 2v): the host binds it.
  array(const array &) = default;
  const T &operator[](unsigned int pos) const;
  const constant T &operator[](unsigned int pos) const constant;
  const device T &operator[](unsigned int pos) const device;
  unsigned int size() const;
  unsigned int size() const constant;
  unsigned int size() const device;
};
template <typename T> struct array_ref {
  array_ref() = default;
  array_ref(const array_ref &) = default;
  template <size_t N> array_ref(const array<T, N> &);
  template <size_t N> array_ref(const constant array<T, N> &);
  template <size_t N> array_ref(const device array<T, N> &);
  const T &operator[](unsigned int pos) const;
  unsigned int size() const;
};
}
