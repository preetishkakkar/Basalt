#pragma once
// Independently authored subset of the MSL indirect command buffer interface (MSL 6.17,
// docs/INDIRECT_COMMANDS.md). A command_buffer is an argument-buffer member the host sizes; a
// compute_command names one of its records, and each method below writes the record's words. The
// pipeline states are inline uints the host fills with pipeline ids. Nothing here has a body.
namespace metal {
struct command_buffer {
  command_buffer() = default;
  command_buffer() constant = default;
  command_buffer() device = default;
  uint size() const;
  uint size() const device;
  uint size() const constant;
};
struct compute_pipeline_state {};
struct render_pipeline_state {};
struct depth_stencil_state {};
enum class primitive_type { point, line, line_strip, triangle, triangle_strip };
enum class cull_mode { none, front, back };
enum class winding { clockwise, counterclockwise };
enum class depth_clip_mode { clip, clamp };
enum class triangle_fill_mode { fill, lines };
#define M2V_ICB(NAME) __attribute__((annotate("msl.icb:" NAME)))
struct compute_command {
  compute_command(const thread compute_command &) thread = default;
  explicit compute_command(command_buffer icb, uint icb_index) thread;
  void set_compute_pipeline_state(compute_pipeline_state pipeline_state) thread M2V_ICB("set_compute_pipeline_state");
  template <typename T> void set_kernel_buffer(device T *buffer, uint index) thread M2V_ICB("set_kernel_buffer");
  template <typename T> void set_kernel_buffer(constant T *buffer, uint index) thread M2V_ICB("set_kernel_buffer");
  template <typename T> void set_kernel_buffer(device T *buffer, size_t stride, uint index) thread M2V_ICB("set_kernel_buffer_stride");
  template <typename T> void set_kernel_buffer(constant T *buffer, size_t stride, uint index) thread M2V_ICB("set_kernel_buffer_stride");
  void set_threadgroup_memory_length(uint length, uint index) thread M2V_ICB("set_threadgroup_memory_length");
  void set_stage_in_region(uint3 origin, uint3 size) thread M2V_ICB("set_stage_in_region");
  void set_imageblock_size(ushort2 size) thread M2V_ICB("set_imageblock_size");
  void set_barrier() thread M2V_ICB("set_barrier");
  void clear_barrier() thread M2V_ICB("clear_barrier");
  void concurrent_dispatch_threadgroups(uint3 threadgroups_per_grid, uint3 threads_per_threadgroup) thread M2V_ICB("concurrent_dispatch_threadgroups");
  void concurrent_dispatch_threads(uint3 threads_per_grid, uint3 threads_per_threadgroup) thread M2V_ICB("concurrent_dispatch_threads");
  void reset() thread M2V_ICB("reset");
  void copy_command(compute_command that) thread M2V_ICB("copy_command");
};
struct render_command {
  render_command(const thread render_command &) thread = default;
  explicit render_command(command_buffer icb, uint icb_index) thread;
  void set_render_pipeline_state(render_pipeline_state pipeline_state) thread M2V_ICB("set_render_pipeline_state");
  template <typename T> void set_vertex_buffer(device T *buffer, uint index) thread M2V_ICB("set_vertex_buffer");
  template <typename T> void set_vertex_buffer(constant T *buffer, uint index) thread M2V_ICB("set_vertex_buffer");
  template <typename T> void set_vertex_buffer(device T *buffer, size_t stride, uint index) thread M2V_ICB("set_vertex_buffer_stride");
  template <typename T> void set_vertex_buffer(constant T *buffer, size_t stride, uint index) thread M2V_ICB("set_vertex_buffer_stride");
  template <typename T> void set_fragment_buffer(device T *buffer, uint index) thread M2V_ICB("set_fragment_buffer");
  template <typename T> void set_fragment_buffer(constant T *buffer, uint index) thread M2V_ICB("set_fragment_buffer");
  void set_cull_mode(cull_mode mode) thread M2V_ICB("set_cull_mode");
  void set_front_facing_winding(winding w) thread M2V_ICB("set_front_facing_winding");
  void set_triangle_fill_mode(triangle_fill_mode mode) thread M2V_ICB("set_triangle_fill_mode");
  void set_depth_bias(float bias, float slope_scale, float clamp) thread M2V_ICB("set_depth_bias");
  void set_depth_clip_mode(depth_clip_mode mode) thread M2V_ICB("set_depth_clip_mode");
  void set_depth_stencil_state(depth_stencil_state state) thread M2V_ICB("set_depth_stencil_state");
  void draw_primitives(primitive_type type, uint vertex_start, uint vertex_count, uint instance_count, uint base_instance = 0) thread M2V_ICB("draw_primitives");
  template <typename T> void draw_indexed_primitives(primitive_type type, uint index_count, const device T *index_buffer, uint instance_count,
                                                     uint base_vertex = 0, uint base_instance = 0) thread M2V_ICB("draw_indexed_primitives");
  template <typename T> void draw_indexed_primitives(primitive_type type, uint index_count, const constant T *index_buffer, uint instance_count,
                                                     uint base_vertex = 0, uint base_instance = 0) thread M2V_ICB("draw_indexed_primitives");
  template <typename T> void draw_patches(uint num_patch_control_points, uint patch_start, uint patch_count, const device uint *patch_index_buffer,
                                          uint instance_count, uint base_instance, const device T *tessellation_factor_buffer, uint instance_stride = 0) thread M2V_ICB("draw_patches");
  void draw_mesh_threadgroups(uint3 threadgroups_per_grid, uint3 threads_per_object_threadgroup, uint3 threads_per_mesh_threadgroup) thread M2V_ICB("draw_mesh_threadgroups");
  void draw_mesh_threads(uint3 threads_per_grid, uint3 threads_per_object_threadgroup, uint3 threads_per_mesh_threadgroup) thread M2V_ICB("draw_mesh_threads");
  template <typename T> void set_object_buffer(device T *buffer, uint index) thread M2V_ICB("set_object_buffer");
  template <typename T> void set_mesh_buffer(device T *buffer, uint index) thread M2V_ICB("set_mesh_buffer");
  void set_object_threadgroup_memory_length(uint length, uint index) thread M2V_ICB("set_object_threadgroup_memory_length");
  void set_barrier() thread M2V_ICB("set_barrier");
  void clear_barrier() thread M2V_ICB("clear_barrier");
  void reset() thread M2V_ICB("reset");
  void copy_command(render_command that) thread M2V_ICB("copy_command");
};
#undef M2V_ICB
}
