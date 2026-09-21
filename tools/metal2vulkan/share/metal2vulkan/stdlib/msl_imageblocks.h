#pragma once
// Independently authored declarations of the MSL imageblock interface (MSL 2.11, 5.6, 6.14), so a
// tile-shading source parses as Apple's compiler parses it and every use is then refused by name:
// Vulkan has no tile memory on this profile (docs/TILE_MODEL.md). Nothing here has a body, and
// threadgroup_imageblock marks its declarations so they are refused wherever they appear.
#define threadgroup_imageblock __attribute__((annotate("msl.tile_memory"))) threadgroup
namespace metal {
struct imageblock_layout_implicit {};
struct imageblock_layout_explicit {};
enum class imageblock_data_rate { color, sample };
template <typename E, typename L> struct imageblock_slice {};
template <typename T, typename L = imageblock_layout_implicit> struct imageblock {
  ushort get_width() const;
  ushort get_height() const;
  ushort get_num_samples() const;
  ushort get_num_colors(ushort2 coord) const;
  T read(ushort2 coord) const;
  T read(ushort2 coord, ushort index, imageblock_data_rate rate) const;
  void write(T data, ushort2 coord);
  void write(T data, ushort2 coord, ushort color_coverage_mask);
  void write(T data, ushort2 coord, ushort index, imageblock_data_rate rate);
  template <typename E> imageblock_slice<E, imageblock_layout_implicit> slice(unsigned index) const;
  template <typename E> imageblock_slice<E, imageblock_layout_implicit> slice(unsigned index, ushort2 size) const;
  threadgroup_imageblock T *data(ushort2 coord) const;
  template <typename E> imageblock_slice<E, imageblock_layout_explicit> slice(const threadgroup_imageblock E &element) const;
  template <typename E> imageblock_slice<E, imageblock_layout_explicit> slice(const threadgroup_imageblock E &element, ushort2 size) const;
};
}
