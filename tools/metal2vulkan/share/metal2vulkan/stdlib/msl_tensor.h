#pragma once
// Independently authored tensor interface (MSL 7, docs/TENSORS.md). A tensor is a rank-1 to rank-4
// view of scalar elements: extents, strides (in elements) and a data pointer. A tensor_handle is
// bound by the host at a [[buffer(n)]] parameter, whose buffer starts with a 64-byte header (rank,
// extents, strides) followed by the data; a tensor_inline is built in the shader over a pointer with
// its extents. The compiler lowers every member below; nothing here has a body.
#include "msl_prelude.h"
#include "msl_texture.h"
namespace metal {
constexpr constant size_t dynamic_extent = ~size_t(0);
template <typename IndexType, size_t... Extents> struct extents {
  using index_type = IndexType;
  static constexpr int rank() { return int(sizeof...(Extents)); }
  extents() thread;
  template <typename... Values> explicit extents(Values... values) thread;
  extents(const thread extents &) thread = default;
  IndexType extent(int r) const thread;
};
template <typename IndexType, size_t Rank, typename Built = extents<IndexType>> struct make_dextents;
template <typename IndexType, size_t Rank, size_t... Pack> struct make_dextents<IndexType, Rank, extents<IndexType, Pack...>> {
  using type = typename make_dextents<IndexType, Rank - 1, extents<IndexType, dynamic_extent, Pack...>>::type;
};
template <typename IndexType, size_t... Pack> struct make_dextents<IndexType, 0, extents<IndexType, Pack...>> {
  using type = extents<IndexType, Pack...>;
};
template <typename IndexType, size_t Rank> using dextents = typename make_dextents<IndexType, Rank>::type;
// The element's value type: without its const and its address space (const device float -> float).
template <typename T> struct __m2v_no_const { using type = T; };
template <typename T> struct __m2v_no_const<const T> { using type = T; };
template <typename T> struct __m2v_no_space { using type = T; };
template <typename T> struct __m2v_no_space<device T> { using type = T; };
template <typename T> struct __m2v_no_space<constant T> { using type = T; };
template <typename T> struct __m2v_no_space<threadgroup T> { using type = T; };
template <typename T> struct __m2v_element_value { using type = typename __m2v_no_space<typename __m2v_no_const<T>::type>::type; };
struct tensor_handle {};
struct tensor_inline {};
struct tensor_offset {}; // The tag a slice carries: its origin is an offset into the tensor it was cut from.
template <typename ElementType, typename Extents, typename Descriptor = tensor_handle, typename... Tags> struct tensor {
  using element_type = ElementType;
  using value_type = typename __m2v_element_value<ElementType>::type;
  using extents_type = Extents;
  using index_type = typename Extents::index_type;
  static constexpr int get_rank() { return Extents::rank(); }
  tensor() thread;
  tensor(const thread tensor &) thread = default;
  // tensor_inline: a view over `data` with these extents; strides are dense with dimension 0 fastest
  // unless given (array<int, rank>{...}, in elements).
  tensor(ElementType *data, const thread Extents &extents) thread;
  tensor(ElementType *data, const thread Extents &extents, const thread array<int, size_t(Extents::rank())> &strides) thread;
  index_type get_extent(int r) const thread;
  index_type get_stride(int r) const thread;
  // t.slice<E...>(i...): the window of extents E... whose origin is element (i...); it keeps the strides.
  template <size_t... SliceExtents, typename... Indices>
  tensor<ElementType, extents<int, SliceExtents...>, Descriptor, tensor_offset> slice(Indices... indices) const thread;
  // t[i, j, ...]: the element at those indices (Metal's multidimensional subscript).
  template <typename... Indices> ElementType &operator[](Indices... indices) const thread;
};
} // namespace metal
