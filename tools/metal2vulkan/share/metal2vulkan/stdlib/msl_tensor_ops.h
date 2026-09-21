#pragma once
// Independently authored TensorOps (Metal Performance Primitives for Metal 4, MSL 7; docs/TENSORS.md).
// Metal's mpp::tensor_ops are a separate framework header this toolchain does not ship, so the
// spellings here follow the Metal 4 documentation and are this profile's; the operations are ordinary
// MSL over tensors, compiled like any helper, so every device runs the same arithmetic: a matmul2d
// accumulates in k order without contraction, and a cooperative tensor holds each thread's share of a
// result in thread storage. Element (x, y) of a rank-2 tensor is t[x, y]: dimension 0 is the column.
#include "msl_prelude.h"
#include "msl_tensor.h"
namespace metal {
// The caller's place among the threadgroup's threads, for the distribution of cooperative elements.
uint __m2v_simd_lane() __attribute__((annotate("msl.simd:__m2v_simd_lane")));
uint __m2v_simd_width() __attribute__((annotate("msl.simd:__m2v_simd_width")));
uint __m2v_simd_index() __attribute__((annotate("msl.simd:__m2v_simd_index")));
} // namespace metal
namespace mpp {
// Execution scopes: the whole operation per thread, or shared by the first N SIMD-groups of the
// threadgroup, counted as 32 lanes each whatever the device's width (docs/TENSORS.md).
template <int N> struct execution_simdgroups { static constexpr int threads() { return 32 * N; } };
struct execution_thread { static constexpr int threads() { return 1; } };
namespace tensor_ops {
struct matmul2d_descriptor {
  enum class mode { multiply, multiply_accumulate };
  int m, n, k;
  bool transpose_left, transpose_right, relaxed_precision;
  mode result_mode;
  constexpr matmul2d_descriptor(int m_, int n_, int k_, bool transpose_left_ = false, bool transpose_right_ = false,
                                bool relaxed_precision_ = false, mode result_mode_ = mode::multiply)
      : m(m_), n(n_), k(k_), transpose_left(transpose_left_), transpose_right(transpose_right_),
        relaxed_precision(relaxed_precision_), result_mode(result_mode_) {}
};
// The distribution of a matmul2d result over the scope's threads: thread t of the scope holds elements
// t, t + threads, t + 2 * threads, ... of the m x n result in row-major order (column fastest).
template <matmul2d_descriptor D, typename Scope> struct matmul2d_layout {
  static constexpr int threads() { return Scope::threads(); }
  static constexpr int columns() { return D.n; }
  static constexpr int rows() { return D.m; }
  // At least two, since a one-element array member is a scalar in this profile (docs/LANGUAGE.md).
  static constexpr int capacity() { return (D.m * D.n + Scope::threads() - 1) / Scope::threads() < 2 ? 2 : (D.m * D.n + Scope::threads() - 1) / Scope::threads(); }
};
} // namespace tensor_ops
} // namespace mpp
namespace metal {
// A cooperative tensor: this thread's share of a distributed rank-2 result. Elements are addressed by
// their index in the share; get_multidimensional_index gives the (column, row) each holds and
// is_valid_element whether the share extends that far; load and store move the share to and from a
// tensor of the result's extents.
template <typename ElementType, typename Extents, typename Layout> struct cooperative_tensor {
  ElementType elements_[Layout::capacity()];
  static constexpr int get_rank() { return 2; }
  static constexpr int get_capacity() { return Layout::capacity(); }
  // Which of the scope's threads this is: every thread is thread 0 of a per-thread scope.
  int thread_index() const { return Layout::threads() == 1 ? 0 : int(__m2v_simd_index() * __m2v_simd_width() + __m2v_simd_lane()); }
  int linear_index(int i) const { return thread_index() + Layout::threads() * i; }
  bool is_valid_element(int i) const {
    return i >= 0 && i < Layout::capacity() && thread_index() < Layout::threads() &&
           linear_index(i) < Layout::columns() * Layout::rows();
  }
  int2 get_multidimensional_index(int i) const {
    int linear = linear_index(i);
    return int2(linear % Layout::columns(), linear / Layout::columns());
  }
  ElementType get(int i) const { return elements_[i]; }
  void set(int i, ElementType value) { elements_[i] = value; }
  template <typename Tensor> void load(const thread Tensor &source) {
    for (int i = 0; i < Layout::capacity(); ++i) {
      if (is_valid_element(i)) {
        int2 index = get_multidimensional_index(i);
        elements_[i] = ElementType(source[index.x, index.y]);
      }
    }
  }
  template <typename Tensor> void store(const thread Tensor &destination) const {
    for (int i = 0; i < Layout::capacity(); ++i) {
      if (is_valid_element(i)) {
        int2 index = get_multidimensional_index(i);
        destination[index.x, index.y] = elements_[i];
      }
    }
  }
};
} // namespace metal
namespace mpp {
namespace tensor_ops {
using metal::cooperative_tensor;
using metal::extents;
using metal::int2;
using metal::__m2v_simd_index;
using metal::__m2v_simd_width;
using metal::__m2v_simd_lane;
using metal::min;
using metal::max;
// C = A * B (or C += A * B in multiply_accumulate mode) for an m x n result over k: A is m x k with
// extents (k, m), or (m, k) when transpose_left; B is k x n with extents (n, k), or (k, n) when
// transpose_right; C has extents (n, m). Products accumulate in k order in the result's element type.
template <matmul2d_descriptor D, typename Scope> struct matmul2d {
  using layout = matmul2d_layout<D, Scope>;
  int reserved_; // The operation carries no state; a struct needs a field in this profile.
  template <typename LeftTensor, typename RightTensor, typename ElementType>
  cooperative_tensor<ElementType, extents<int, D.n, D.m>, layout> get_destination_cooperative_tensor() const {
    cooperative_tensor<ElementType, extents<int, D.n, D.m>, layout> result = {};
    return result;
  }
  template <typename ElementType, typename LeftTensor, typename RightTensor>
  ElementType element(const thread LeftTensor &a, const thread RightTensor &b, int column, int row, ElementType initial) const {
    ElementType accumulator = initial;
    for (int kk = 0; kk < D.k; ++kk) {
      ElementType left = 0, right = 0;
      if (D.transpose_left) left = ElementType(a[row, kk]);
      else left = ElementType(a[kk, row]);
      if (D.transpose_right) right = ElementType(b[kk, column]);
      else right = ElementType(b[column, kk]);
      accumulator += left * right;
    }
    return accumulator;
  }
  // The result into a tensor of extents (n, m): the scope's threads split its elements.
  template <typename LeftTensor, typename RightTensor, typename ResultTensor>
  void run(const thread LeftTensor &a, const thread RightTensor &b, const thread ResultTensor &c) const {
    int thread_index = layout::threads() == 1 ? 0 : int(__m2v_simd_index() * __m2v_simd_width() + __m2v_simd_lane());
    if (thread_index < layout::threads()) {
      for (int linear = thread_index; linear < D.m * D.n; linear += layout::threads()) {
        int column = linear % D.n, row = linear / D.n;
        typename ResultTensor::value_type initial = 0;
        if (D.result_mode == matmul2d_descriptor::mode::multiply_accumulate) initial = c[column, row];
        c[column, row] = element(a, b, column, row, initial);
      }
    }
  }
  // The result into this thread's share of a cooperative tensor.
  template <typename LeftTensor, typename RightTensor, typename ElementType>
  void run(const thread LeftTensor &a, const thread RightTensor &b, thread cooperative_tensor<ElementType, extents<int, D.n, D.m>, layout> &c) const {
    for (int i = 0; i < layout::capacity(); ++i) {
      if (c.is_valid_element(i)) {
        int2 index = c.get_multidimensional_index(i);
        ElementType initial = 0;
        if (D.result_mode == matmul2d_descriptor::mode::multiply_accumulate) initial = c.elements_[i];
        c.elements_[i] = element(a, b, index.x, index.y, initial);
      }
    }
  }
};
} // namespace tensor_ops
} // namespace mpp
namespace mpp {
namespace tensor_ops {
// A 2D convolution over channels-last tensors: activations (width, height, channels), weights
// (kernel_x, kernel_y, channels per group, output channels), output (output width, output height,
// output channels). Output (x, y, oc) sums activation (x * stride - padding + kx, y * stride -
// padding + ky, group channels) times weight (kx, ky, ic, oc) over the kernel and the group's
// channels, reading zero outside the activations; kernels walk ky, kx then ic in that order.
struct convolution2d_descriptor {
  enum class mode { multiply, multiply_accumulate };
  int kernel_width, kernel_height, stride_x, stride_y, padding_x, padding_y, groups;
  mode result_mode;
  constexpr convolution2d_descriptor(int kernel_width_, int kernel_height_, int stride_x_ = 1, int stride_y_ = 1, int padding_x_ = 0,
                                     int padding_y_ = 0, int groups_ = 1, mode result_mode_ = mode::multiply)
      : kernel_width(kernel_width_), kernel_height(kernel_height_), stride_x(stride_x_), stride_y(stride_y_), padding_x(padding_x_),
        padding_y(padding_y_), groups(groups_), result_mode(result_mode_) {}
};
template <convolution2d_descriptor D, typename Scope> struct convolution2d {
  int reserved_; // The operation carries no state; a struct needs a field in this profile.
  template <typename ElementType, typename Activations, typename Weights>
  ElementType element(const thread Activations &a, const thread Weights &w, int x, int y, int oc, int channels_per_group, int outputs_per_group,
                      ElementType initial) const {
    ElementType accumulator = initial;
    int width = a.get_extent(0), height = a.get_extent(1);
    int group = oc / outputs_per_group;
    for (int ic = 0; ic < channels_per_group; ++ic) {
      for (int ky = 0; ky < D.kernel_height; ++ky) {
        for (int kx = 0; kx < D.kernel_width; ++kx) {
          int sx = x * D.stride_x - D.padding_x + kx, sy = y * D.stride_y - D.padding_y + ky;
          if (sx >= 0 && sx < width && sy >= 0 && sy < height)
            accumulator += ElementType(a[sx, sy, group * channels_per_group + ic]) * ElementType(w[kx, ky, ic, oc]);
        }
      }
    }
    return accumulator;
  }
  // The output into a rank-3 tensor: the scope's threads split its elements in row-major order.
  template <typename Activations, typename Weights, typename Output>
  void run(const thread Activations &a, const thread Weights &w, const thread Output &o) const {
    int out_width = o.get_extent(0), out_height = o.get_extent(1), out_channels = o.get_extent(2);
    int channels_per_group = a.get_extent(2) / D.groups, outputs_per_group = out_channels / D.groups;
    int threads = Scope::threads();
    int thread_index = threads == 1 ? 0 : int(__m2v_simd_index() * __m2v_simd_width() + __m2v_simd_lane());
    if (thread_index < threads) {
      for (int linear = thread_index; linear < out_width * out_height * out_channels; linear += threads) {
        int x = linear % out_width, y = (linear / out_width) % out_height, oc = linear / (out_width * out_height);
        typename Output::value_type initial = 0;
        if (D.result_mode == convolution2d_descriptor::mode::multiply_accumulate) initial = o[x, y, oc];
        o[x, y, oc] = element(a, w, x, y, oc, channels_per_group, outputs_per_group, initial);
      }
    }
  }
};
// A reduction along one axis of a rank-2 tensor into a rank-1 tensor: the sum, minimum or maximum
// over the dimension `axis` names, taken in index order so floating sums are the same on every
// device. The result has the other dimension's extent; the scope's threads split its elements.
struct reduce2d_descriptor {
  enum class operation { sum, minimum, maximum };
  int axis;
  operation op;
  constexpr reduce2d_descriptor(int axis_, operation op_ = operation::sum) : axis(axis_), op(op_) {}
};
template <reduce2d_descriptor D, typename Scope> struct reduce2d {
  int reserved_;
  template <typename Input, typename Output>
  void run(const thread Input &in, const thread Output &out) const {
    int kept = D.axis == 0 ? in.get_extent(1) : in.get_extent(0), reduced = D.axis == 0 ? in.get_extent(0) : in.get_extent(1);
    int threads = Scope::threads();
    int thread_index = threads == 1 ? 0 : int(__m2v_simd_index() * __m2v_simd_width() + __m2v_simd_lane());
    if (thread_index < threads) {
      for (int index = thread_index; index < kept; index += threads) {
        typename Output::value_type accumulator = 0;
        if (D.axis == 0) accumulator = in[0, index];
        else accumulator = in[index, 0];
        for (int r = 1; r < reduced; ++r) {
          typename Output::value_type value = 0;
          if (D.axis == 0) value = in[r, index];
          else value = in[index, r];
          if (D.op == reduce2d_descriptor::operation::sum) accumulator += value;
          else if (D.op == reduce2d_descriptor::operation::minimum) accumulator = min(accumulator, value);
          else accumulator = max(accumulator, value);
        }
        out[index] = accumulator;
      }
    }
  }
};
} // namespace tensor_ops
} // namespace mpp
