#pragma once
// Independently authored subset of the MSL SIMD-group matrix interface (MSL 2.4 and 6.8,
// docs/SUBGROUPS.md). A simdgroup_matrix<T, 8, 8> is held cooperatively by the SIMD-group: this
// compiler keeps row r in lane r (the `elements` member is that lane's row), and the functions below
// have no bodies: each is lowered to shuffles, loads and stores by the compiler. The element mapping
// is unspecified by MSL, so thread_elements() is not offered.
namespace metal {
#define M2V_SIMD(NAME) __attribute__((annotate("msl.simd:" NAME)))
template <typename T, int Rows, int Cols> struct simdgroup_matrix {
  T elements[Cols];
  simdgroup_matrix() thread = default;
  simdgroup_matrix(const thread simdgroup_matrix &) thread = default;
  explicit simdgroup_matrix(T value) M2V_SIMD("simdgroup_matrix"); // A diagonal matrix.
};
typedef simdgroup_matrix<float, 8, 8> simdgroup_float8x8;
typedef simdgroup_matrix<half, 8, 8> simdgroup_half8x8;
template <typename T, int Rows, int Cols> simdgroup_matrix<T, Rows, Cols> make_filled_simdgroup_matrix(T value) M2V_SIMD("make_filled_simdgroup_matrix");
template <typename T, int Rows, int Cols>
void simdgroup_load(thread simdgroup_matrix<T, Rows, Cols> &d, const threadgroup T *src, ulong elements_per_row = Cols,
                    ulong2 matrix_origin = ulong2(0), bool transpose_matrix = false) M2V_SIMD("simdgroup_load");
template <typename T, int Rows, int Cols>
void simdgroup_load(thread simdgroup_matrix<T, Rows, Cols> &d, const device T *src, ulong elements_per_row = Cols,
                    ulong2 matrix_origin = ulong2(0), bool transpose_matrix = false) M2V_SIMD("simdgroup_load");
template <typename T, int Rows, int Cols>
void simdgroup_store(const thread simdgroup_matrix<T, Rows, Cols> &a, threadgroup T *dst, ulong elements_per_row = Cols,
                     ulong2 matrix_origin = ulong2(0), bool transpose_matrix = false) M2V_SIMD("simdgroup_store");
template <typename T, int Rows, int Cols>
void simdgroup_store(const thread simdgroup_matrix<T, Rows, Cols> &a, device T *dst, ulong elements_per_row = Cols,
                     ulong2 matrix_origin = ulong2(0), bool transpose_matrix = false) M2V_SIMD("simdgroup_store");
template <typename T, int Rows, int Cols, int K>
void simdgroup_multiply(thread simdgroup_matrix<T, Rows, Cols> &d, const thread simdgroup_matrix<T, Rows, K> &a,
                        const thread simdgroup_matrix<T, K, Cols> &b) M2V_SIMD("simdgroup_multiply");
template <typename T, int Rows, int Cols, int K>
void simdgroup_multiply_accumulate(thread simdgroup_matrix<T, Rows, Cols> &d, const thread simdgroup_matrix<T, Rows, K> &a,
                                   const thread simdgroup_matrix<T, K, Cols> &b,
                                   const thread simdgroup_matrix<T, Rows, Cols> &c) M2V_SIMD("simdgroup_multiply_accumulate");
#undef M2V_SIMD
}
