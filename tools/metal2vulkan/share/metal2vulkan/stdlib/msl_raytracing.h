#pragma once
#include "msl_prelude.h"
#include "msl_matrix.h"
// Independently authored declarations for ray tracing (MSL section 6.18): the
// ray, the acceleration structure handles, the intersector, the intersection
// query and their results. The compiler recognizes the methods by name on these
// owned records and lowers them to a Vulkan ray query; there are no bodies here.
//
// The records live in namespace metal, where the compiler looks for every owned
// record, and metal::raytracing re-exports them under the names MSL spells.
namespace metal {
// Intersection tags (MSL 6.18.2): what an intersector traverses and reports.
struct triangle_data {};
struct instancing {};
struct world_space_data {};
// Tags the compiler refuses by name (V9 9g, docs/RAY_TRACING.md): motion blur and curves (R09)
// and multi-level instancing (R10) are declared so a source using them is told which feature is
// missing rather than failing on an unknown name.
struct primitive_motion {};
struct instance_motion {};
struct curve_data {};
template <int Levels> struct max_levels {};
// The intersection function buffer tag (MSL 6.19.8, Metal 4): the functions come from a buffer of
// handles the host describes with intersection_function_buffer_arguments rather than from a table,
// and the result says which function ran (function_id, MSL 2.17.4).
struct intersection_function_buffer {};
// What the host passes for an intersection function buffer: the buffer's device address, its size in
// bytes and the stride between handles. Metal spells the first field as a device pointer; here it is
// the 64-bit address itself (the same eight bytes), read through the device-address path.
struct intersection_function_buffer_arguments {
  ulong intersection_function_buffer;
  ulong intersection_function_buffer_size;
  ulong intersection_function_stride;
};
// What an intersection is (MSL 6.18.4), with the values MSL gives them.
enum class intersection_type : uint { none = 0, triangle = 1, bounding_box = 2 };
// What a geometry is, for intersector::assume_geometry_type.
enum class geometry_type : uint { none = 0, triangle = 1, bounding_box = 2, all = 3 }; // all: triangle | bounding_box (no curves here).
// The intersector's options (MSL 6.18.6).
enum class forced_opacity : uint { none = 0, opaque = 1, non_opaque = 2 };
enum class triangle_cull_mode : uint { none = 0, front = 1, back = 2 };
// The traversal settings a query is started with (MSL 6.19.5): the same options an intersector
// sets by its methods, read back through getters, and returned by a query's get_intersection_params().
struct intersection_params {
  void accept_any_intersection(bool accept);
  void force_opacity(forced_opacity opacity);
  void set_triangle_cull_mode(triangle_cull_mode mode);
  void assume_geometry_type(geometry_type type);
  bool should_accept_any_intersection() const;
  forced_opacity get_forced_opacity() const;
  triangle_cull_mode get_triangle_cull_mode() const;
  geometry_type get_geometry_type() const;
};
// A ray (MSL 6.18.1): its origin and direction in the acceleration structure's
// space, and the parametric range along the direction that counts as a hit.
struct ray {
  float3 origin;
  float3 direction;
  float min_distance;
  float max_distance;
  ray() = default;
  ray(float3 o, float3 d, float min_t, float max_t)
      : origin(o), direction(d), min_distance(min_t), max_distance(max_t) {}
};
// Acceleration structures (MSL 6.18.3): handles an entry takes as [[buffer(n)]]
// parameters. A primitive structure holds triangles; an instance structure holds
// transformed references to primitive structures.
struct primitive_acceleration_structure {};
struct instance_acceleration_structure {};
// The result of an intersection (MSL 6.18.5), with the fields the tags admit.
template <typename... Tags> struct intersection_result;
template <> struct intersection_result<triangle_data> {
  intersection_type type;
  float distance;
  uint primitive_id;
  uint geometry_id;
  float2 triangle_barycentric_coord;
  bool triangle_front_facing;
};
template <> struct intersection_result<triangle_data, instancing> {
  intersection_type type;
  float distance;
  uint primitive_id;
  uint geometry_id;
  uint instance_id;
  uint user_instance_id;
  float2 triangle_barycentric_coord;
  bool triangle_front_facing;
};
// With an intersection function buffer: the fields of the plain result, then function_id, the
// index of the function that produced the committed hit (MSL 2.17.4, Metal 4.1).
template <> struct intersection_result<triangle_data, intersection_function_buffer> {
  intersection_type type;
  float distance;
  uint primitive_id;
  uint geometry_id;
  float2 triangle_barycentric_coord;
  bool triangle_front_facing;
  uint function_id;
};
template <> struct intersection_result<triangle_data, instancing, intersection_function_buffer> {
  intersection_type type;
  float distance;
  uint primitive_id;
  uint geometry_id;
  uint instance_id;
  uint user_instance_id;
  float2 triangle_barycentric_coord;
  bool triangle_front_facing;
  uint function_id;
};
// World-space data (MSL 6.18.5): the transforms of the instance the hit lies in, as float4x3
// (four columns of three: the rotation's columns, then the translation).
template <> struct intersection_result<triangle_data, instancing, world_space_data> {
  intersection_type type;
  float distance;
  uint primitive_id;
  uint geometry_id;
  uint instance_id;
  uint user_instance_id;
  float4x3 object_to_world_transform;
  float4x3 world_to_object_transform;
  float2 triangle_barycentric_coord;
  bool triangle_front_facing;
};
// The spellings the compiler refuses by name: each parses and is then refused in the compiler
// with the feature that is missing.
template <> struct intersection_result<triangle_data, primitive_motion> {
  intersection_type type;
  float distance;
  uint primitive_id;
  uint geometry_id;
  float2 triangle_barycentric_coord;
  bool triangle_front_facing;
};
template <> struct intersection_result<triangle_data, instancing, instance_motion> {
  intersection_type type;
  float distance;
  uint primitive_id;
  uint geometry_id;
  uint instance_id;
  uint user_instance_id;
  float2 triangle_barycentric_coord;
  bool triangle_front_facing;
};
template <> struct intersection_result<curve_data> {
  intersection_type type;
  float distance;
  uint primitive_id;
  uint geometry_id;
  float curve_parameter;
};
template <int Levels> struct intersection_result<triangle_data, instancing, max_levels<Levels>> {
  intersection_type type;
  float distance;
  uint primitive_id;
  uint geometry_id;
  uint instance_id;
  uint user_instance_id;
  float2 triangle_barycentric_coord;
  bool triangle_front_facing;
};
// The result of an intersection by reference (MSL 2.17.5, Metal 3.2): what the callback an
// intersect() call takes receives, alive for the callback's duration and read through getters.
template <typename... Tags> struct intersection_result_ref;
template <> struct intersection_result_ref<triangle_data> {
  intersection_type get_type() const;
  float get_distance() const;
  uint get_primitive_id() const;
  uint get_geometry_id() const;
  float2 get_triangle_barycentric_coord() const;
  bool is_triangle_front_facing() const;
};
template <> struct intersection_result_ref<triangle_data, instancing> {
  intersection_type get_type() const;
  float get_distance() const;
  uint get_primitive_id() const;
  uint get_geometry_id() const;
  uint get_instance_id() const;
  uint get_user_instance_id() const;
  float2 get_triangle_barycentric_coord() const;
  bool is_triangle_front_facing() const;
};
template <> struct intersection_result_ref<triangle_data, instancing, world_space_data> {
  intersection_type get_type() const;
  float get_distance() const;
  uint get_primitive_id() const;
  uint get_geometry_id() const;
  uint get_instance_id() const;
  uint get_user_instance_id() const;
  float4x3 get_object_to_world_transform() const;
  float4x3 get_world_to_object_transform() const;
  float2 get_triangle_barycentric_coord() const;
  bool is_triangle_front_facing() const;
};
// An intersection function table (MSL 6.18.8): a resource the entry takes at a buffer index whose
// slots the host fills with the [[intersection(...)]] functions of this source; intersect() runs the
// one a candidate's instance and geometry select. Its tags are the intersector's.
template <typename... Tags> struct intersection_function_table;
template <> struct intersection_function_table<triangle_data> {};
template <> struct intersection_function_table<triangle_data, instancing> {};
// The intersector (MSL 6.18.6): a traversal whose options are set by methods
// and whose intersect() finds the closest hit of a ray, through the intersection
// functions of a table where one is given, with a ray_data payload they share.
template <typename... Tags> struct intersector;
template <> struct intersector<triangle_data> {
  void assume_geometry_type(geometry_type type);
  void accept_any_intersection(bool accept);
  void force_opacity(forced_opacity opacity);
  void set_triangle_cull_mode(triangle_cull_mode mode);
  intersection_result<triangle_data> intersect(ray r, primitive_acceleration_structure structure);
  intersection_result<triangle_data> intersect(ray r, primitive_acceleration_structure structure, intersection_function_table<triangle_data> table);
  template <typename Payload>
  intersection_result<triangle_data> intersect(ray r, primitive_acceleration_structure structure, intersection_function_table<triangle_data> table, thread Payload &payload);
  // The callback forms (MSL 6.19.2, Metal 3.2): the traversal runs, then the callable receives the
  // result by reference and, where a payload was given, the payload the functions left. The callable
  // must have a call operator, which keeps a payload struct from being taken for one. Their empty
  // bodies are never lowered: a template instantiated with a lambda (a type without linkage) has to
  // be defined in the translation unit, and the compiler lowers intersect by name.
  template <typename Callable, typename = decltype(&Callable::operator())>
  void intersect(ray r, primitive_acceleration_structure structure, Callable callback) {}
  template <typename Callable, typename = decltype(&Callable::operator())>
  void intersect(ray r, primitive_acceleration_structure structure, intersection_function_table<triangle_data> table, Callable callback) {}
  template <typename Payload, typename Callable, typename = decltype(&Callable::operator())>
  void intersect(ray r, primitive_acceleration_structure structure, const thread Payload &payload, Callable callback) {}
  template <typename Payload, typename Callable, typename = decltype(&Callable::operator())>
  void intersect(ray r, primitive_acceleration_structure structure, intersection_function_table<triangle_data> table, const thread Payload &payload, Callable callback) {}
};
template <> struct intersector<triangle_data, instancing> {
  void assume_geometry_type(geometry_type type);
  void accept_any_intersection(bool accept);
  void force_opacity(forced_opacity opacity);
  void set_triangle_cull_mode(triangle_cull_mode mode);
  intersection_result<triangle_data, instancing> intersect(ray r, instance_acceleration_structure structure, uint mask);
  intersection_result<triangle_data, instancing> intersect(ray r, instance_acceleration_structure structure, uint mask, intersection_function_table<triangle_data, instancing> table);
  template <typename Payload>
  intersection_result<triangle_data, instancing> intersect(ray r, instance_acceleration_structure structure, uint mask, intersection_function_table<triangle_data, instancing> table, thread Payload &payload);
  template <typename Callable, typename = decltype(&Callable::operator())>
  void intersect(ray r, instance_acceleration_structure structure, uint mask, Callable callback) {}
  template <typename Callable, typename = decltype(&Callable::operator())>
  void intersect(ray r, instance_acceleration_structure structure, uint mask, intersection_function_table<triangle_data, instancing> table, Callable callback) {}
  template <typename Payload, typename Callable, typename = decltype(&Callable::operator())>
  void intersect(ray r, instance_acceleration_structure structure, uint mask, const thread Payload &payload, Callable callback) {}
  template <typename Payload, typename Callable, typename = decltype(&Callable::operator())>
  void intersect(ray r, instance_acceleration_structure structure, uint mask, intersection_function_table<triangle_data, instancing> table, const thread Payload &payload, Callable callback) {}
};
template <> struct intersector<triangle_data, intersection_function_buffer> {
  void assume_geometry_type(geometry_type type);
  void accept_any_intersection(bool accept);
  void force_opacity(forced_opacity opacity);
  void set_triangle_cull_mode(triangle_cull_mode mode);
  intersection_result<triangle_data, intersection_function_buffer> intersect(ray r, primitive_acceleration_structure structure, intersection_function_buffer_arguments ifba);
  template <typename Payload>
  intersection_result<triangle_data, intersection_function_buffer> intersect(ray r, primitive_acceleration_structure structure, intersection_function_buffer_arguments ifba, thread Payload &payload);
};
template <> struct intersector<triangle_data, instancing, intersection_function_buffer> {
  void assume_geometry_type(geometry_type type);
  void accept_any_intersection(bool accept);
  void force_opacity(forced_opacity opacity);
  void set_triangle_cull_mode(triangle_cull_mode mode);
  intersection_result<triangle_data, instancing, intersection_function_buffer> intersect(ray r, instance_acceleration_structure structure, uint mask, intersection_function_buffer_arguments ifba);
  template <typename Payload>
  intersection_result<triangle_data, instancing, intersection_function_buffer> intersect(ray r, instance_acceleration_structure structure, uint mask, intersection_function_buffer_arguments ifba, thread Payload &payload);
};
template <> struct intersector<triangle_data, instancing, world_space_data> {
  void assume_geometry_type(geometry_type type);
  void accept_any_intersection(bool accept);
  void force_opacity(forced_opacity opacity);
  void set_triangle_cull_mode(triangle_cull_mode mode);
  intersection_result<triangle_data, instancing, world_space_data> intersect(ray r, instance_acceleration_structure structure, uint mask);
};
template <> struct intersector<triangle_data, primitive_motion> {
  intersection_result<triangle_data, primitive_motion> intersect(ray r, primitive_acceleration_structure structure, float time);
};
template <> struct intersector<triangle_data, instancing, instance_motion> {
  intersection_result<triangle_data, instancing, instance_motion> intersect(ray r, instance_acceleration_structure structure, uint mask, float time);
};
template <> struct intersector<curve_data> {
  intersection_result<curve_data> intersect(ray r, primitive_acceleration_structure structure);
};
template <int Levels> struct intersector<triangle_data, instancing, max_levels<Levels>> {
  intersection_result<triangle_data, instancing, max_levels<Levels>> intersect(ray r, instance_acceleration_structure structure, uint mask);
};
// The intersection query (MSL 6.18.7): the same traversal stepped by the
// shader, one candidate at a time, committing the ones it accepts.
template <typename... Tags> struct intersection_query;
template <> struct intersection_query<triangle_data> {
  intersection_query() = default;
  intersection_query(ray r, primitive_acceleration_structure structure);
  intersection_query(ray r, primitive_acceleration_structure structure, intersection_params params);
  void reset(ray r, primitive_acceleration_structure structure);
  void reset(ray r, primitive_acceleration_structure structure, intersection_params params);
  intersection_params get_intersection_params() const;
  bool next();
  void abort();
  void commit_triangle_intersection();
  void commit_bounding_box_intersection(float distance);
  intersection_type get_candidate_intersection_type();
  float get_candidate_triangle_distance();
  uint get_candidate_primitive_id();
  uint get_candidate_geometry_id();
  float2 get_candidate_triangle_barycentric_coord();
  bool is_candidate_triangle_front_facing();
  intersection_type get_committed_intersection_type();
  float get_committed_distance();
  uint get_committed_primitive_id();
  uint get_committed_geometry_id();
  float2 get_committed_triangle_barycentric_coord();
  bool is_committed_triangle_front_facing();
  float3 get_world_space_ray_origin();
  float3 get_world_space_ray_direction();
  float3 get_candidate_ray_origin();
  float3 get_candidate_ray_direction();
  float3 get_committed_ray_origin();
  float3 get_committed_ray_direction();
};
template <> struct intersection_query<triangle_data, instancing> {
  intersection_query() = default;
  intersection_query(ray r, instance_acceleration_structure structure, uint mask);
  intersection_query(ray r, instance_acceleration_structure structure, uint mask, intersection_params params);
  void reset(ray r, instance_acceleration_structure structure, uint mask);
  void reset(ray r, instance_acceleration_structure structure, uint mask, intersection_params params);
  intersection_params get_intersection_params() const;
  bool next();
  void abort();
  void commit_triangle_intersection();
  void commit_bounding_box_intersection(float distance);
  intersection_type get_candidate_intersection_type();
  float get_candidate_triangle_distance();
  uint get_candidate_primitive_id();
  uint get_candidate_geometry_id();
  uint get_candidate_instance_id();
  uint get_candidate_user_instance_id();
  float2 get_candidate_triangle_barycentric_coord();
  bool is_candidate_triangle_front_facing();
  intersection_type get_committed_intersection_type();
  float get_committed_distance();
  uint get_committed_primitive_id();
  uint get_committed_geometry_id();
  uint get_committed_instance_id();
  uint get_committed_user_instance_id();
  float2 get_committed_triangle_barycentric_coord();
  bool is_committed_triangle_front_facing();
  float4x3 get_candidate_object_to_world_transform();
  float4x3 get_candidate_world_to_object_transform();
  float4x3 get_committed_object_to_world_transform();
  float4x3 get_committed_world_to_object_transform();
  float3 get_world_space_ray_origin();
  float3 get_world_space_ray_direction();
  float3 get_candidate_ray_origin();
  float3 get_candidate_ray_direction();
  float3 get_committed_ray_origin();
  float3 get_committed_ray_direction();
};
namespace raytracing {
using metal::triangle_data;
using metal::instancing;
using metal::world_space_data;
using metal::primitive_motion;
using metal::instance_motion;
using metal::curve_data;
using metal::max_levels;
using metal::intersection_function_buffer;
using metal::intersection_function_buffer_arguments;
using metal::intersection_type;
using metal::geometry_type;
using metal::forced_opacity;
using metal::triangle_cull_mode;
using metal::ray;
using metal::intersection_params;
using metal::primitive_acceleration_structure;
using metal::instance_acceleration_structure;
using metal::intersection_result;
using metal::intersection_result_ref;
using metal::intersector;
using metal::intersection_function_table;
using metal::intersection_query;
}
}
