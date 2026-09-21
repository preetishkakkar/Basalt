#pragma once
// Independently authored subset of the MSL SIMD-group interface (MSL 6.10.2,
// docs/SUBGROUPS.md). Nothing here has a body: the compiler lowers each call to
// a SPIR-V subgroup operation and the entry's reflection names the operation
// classes the device must report.
namespace metal {
#define M2V_SIMD(NAME) __attribute__((annotate("msl.simd:" NAME)))
// A ballot: one bit per lane of the SIMD-group, spelled as a record whose
// methods read it. The compiler represents a simd_vote as its vote_t.
struct simd_vote {
  typedef ulong vote_t;
  simd_vote() thread = default;
  simd_vote(const thread simd_vote &) thread = default;
  explicit simd_vote(vote_t value) M2V_SIMD("simd_vote");
  explicit operator vote_t() const M2V_SIMD("simd_vote::vote_t");
  bool all() const M2V_SIMD("simd_vote::all");
  bool any() const M2V_SIMD("simd_vote::any");
};
// A quad-group ballot: four bits in a ushort (MSL 6.10.3).
struct quad_vote {
  typedef ushort vote_t;
  quad_vote() thread = default;
  quad_vote(const thread quad_vote &) thread = default;
  explicit quad_vote(vote_t value) M2V_SIMD("quad_vote");
  explicit operator vote_t() const M2V_SIMD("quad_vote::vote_t");
  bool all() const M2V_SIMD("quad_vote::all");
  bool any() const M2V_SIMD("quad_vote::any");
};
bool simd_all(bool expr) M2V_SIMD("simd_all");
bool simd_any(bool expr) M2V_SIMD("simd_any");
simd_vote simd_ballot(bool expr) M2V_SIMD("simd_ballot");
simd_vote simd_active_threads_mask() M2V_SIMD("simd_active_threads_mask");
bool simd_is_first() M2V_SIMD("simd_is_first");
template <typename T> T simd_broadcast(T data, ushort broadcast_lane_id) M2V_SIMD("simd_broadcast");
template <typename T> T simd_broadcast_first(T data) M2V_SIMD("simd_broadcast_first");
template <typename T> T simd_shuffle(T data, ushort simd_lane_id) M2V_SIMD("simd_shuffle");
template <typename T> T simd_shuffle_xor(T data, ushort mask) M2V_SIMD("simd_shuffle_xor");
template <typename T> T simd_shuffle_up(T data, ushort delta) M2V_SIMD("simd_shuffle_up");
template <typename T> T simd_shuffle_down(T data, ushort delta) M2V_SIMD("simd_shuffle_down");
template <typename T> T simd_shuffle_rotate_up(T data, ushort delta) M2V_SIMD("simd_shuffle_rotate_up");
template <typename T> T simd_shuffle_rotate_down(T data, ushort delta) M2V_SIMD("simd_shuffle_rotate_down");
template <typename T> T simd_shuffle_and_fill_up(T data, T filling_data, ushort delta) M2V_SIMD("simd_shuffle_and_fill_up");
template <typename T> T simd_shuffle_and_fill_down(T data, T filling_data, ushort delta) M2V_SIMD("simd_shuffle_and_fill_down");
template <typename T> T simd_shuffle_and_fill_up(T data, T filling_data, ushort delta, ushort modulo) M2V_SIMD("simd_shuffle_and_fill_up");
template <typename T> T simd_shuffle_and_fill_down(T data, T filling_data, ushort delta, ushort modulo) M2V_SIMD("simd_shuffle_and_fill_down");
template <typename T> T simd_sum(T data) M2V_SIMD("simd_sum");
template <typename T> T simd_product(T data) M2V_SIMD("simd_product");
template <typename T> T simd_min(T data) M2V_SIMD("simd_min");
template <typename T> T simd_max(T data) M2V_SIMD("simd_max");
template <typename T> T simd_and(T data) M2V_SIMD("simd_and");
template <typename T> T simd_or(T data) M2V_SIMD("simd_or");
template <typename T> T simd_xor(T data) M2V_SIMD("simd_xor");
template <typename T> T simd_prefix_inclusive_sum(T data) M2V_SIMD("simd_prefix_inclusive_sum");
template <typename T> T simd_prefix_inclusive_product(T data) M2V_SIMD("simd_prefix_inclusive_product");
template <typename T> T simd_prefix_exclusive_sum(T data) M2V_SIMD("simd_prefix_exclusive_sum");
template <typename T> T simd_prefix_exclusive_product(T data) M2V_SIMD("simd_prefix_exclusive_product");
bool quad_all(bool expr) M2V_SIMD("quad_all");
bool quad_any(bool expr) M2V_SIMD("quad_any");
quad_vote quad_ballot(bool expr) M2V_SIMD("quad_ballot");
quad_vote quad_active_threads_mask() M2V_SIMD("quad_active_threads_mask");
bool quad_is_first() M2V_SIMD("quad_is_first");
template <typename T> T quad_broadcast(T data, ushort broadcast_lane_id) M2V_SIMD("quad_broadcast");
template <typename T> T quad_broadcast_first(T data) M2V_SIMD("quad_broadcast_first");
template <typename T> T quad_shuffle(T data, ushort quad_lane_id) M2V_SIMD("quad_shuffle");
template <typename T> T quad_shuffle_xor(T data, ushort mask) M2V_SIMD("quad_shuffle_xor");
template <typename T> T quad_shuffle_up(T data, ushort delta) M2V_SIMD("quad_shuffle_up");
template <typename T> T quad_shuffle_down(T data, ushort delta) M2V_SIMD("quad_shuffle_down");
template <typename T> T quad_shuffle_rotate_up(T data, ushort delta) M2V_SIMD("quad_shuffle_rotate_up");
template <typename T> T quad_shuffle_rotate_down(T data, ushort delta) M2V_SIMD("quad_shuffle_rotate_down");
template <typename T> T quad_shuffle_and_fill_up(T data, T filling_data, ushort delta) M2V_SIMD("quad_shuffle_and_fill_up");
template <typename T> T quad_shuffle_and_fill_down(T data, T filling_data, ushort delta) M2V_SIMD("quad_shuffle_and_fill_down");
template <typename T> T quad_sum(T data) M2V_SIMD("quad_sum");
template <typename T> T quad_product(T data) M2V_SIMD("quad_product");
template <typename T> T quad_min(T data) M2V_SIMD("quad_min");
template <typename T> T quad_max(T data) M2V_SIMD("quad_max");
template <typename T> T quad_and(T data) M2V_SIMD("quad_and");
template <typename T> T quad_or(T data) M2V_SIMD("quad_or");
template <typename T> T quad_xor(T data) M2V_SIMD("quad_xor");
template <typename T> T quad_prefix_inclusive_sum(T data) M2V_SIMD("quad_prefix_inclusive_sum");
template <typename T> T quad_prefix_inclusive_product(T data) M2V_SIMD("quad_prefix_inclusive_product");
template <typename T> T quad_prefix_exclusive_sum(T data) M2V_SIMD("quad_prefix_exclusive_sum");
template <typename T> T quad_prefix_exclusive_product(T data) M2V_SIMD("quad_prefix_exclusive_product");
#undef M2V_SIMD
}
