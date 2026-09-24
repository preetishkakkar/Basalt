#pragma once
#include "msl_prelude.h"
#include "msl_matrix.h"
#include "msl_raytracing.h"
// Independently authored declarations for the Vulkan ray-tracing pipeline profile
// (docs/RAY_PIPELINES.md). This is an M2V extension, not MSL: Metal has no ray-generation, miss,
// hit or callable entries, so a source that includes <m2v_ray_pipeline> is Vulkan-only and makes
// no Apple source-equivalence claim. An entry is a void function marked
// [[m2v::ray_stage(raygen | miss | closest_hit | any_hit | intersection | callable)]].
//
// Nothing here has a body: the compiler recognizes each declaration by its annotation, only when
// it comes from this owned file, and lowers it to the SPV_KHR_ray_tracing instruction or built-in.
namespace m2v {
#define M2V_RAY(NAME) __attribute__((annotate("msl.raypipe:" NAME)))
// Ray flags: the bits OpTraceRayKHR takes (SPIR-V RayFlags). skip_triangles and skip_aabbs need
// the rayTraversalPrimitiveCulling feature, which a constant flags operand declares.
namespace ray_flags {
constexpr constant uint none = 0u;
constexpr constant uint opaque = 1u;
constexpr constant uint no_opaque = 2u;
constexpr constant uint terminate_on_first_hit = 4u;
constexpr constant uint skip_closest_hit_shader = 8u;
constexpr constant uint cull_back_facing_triangles = 16u;
constexpr constant uint cull_front_facing_triangles = 32u;
constexpr constant uint cull_opaque = 64u;
constexpr constant uint cull_no_opaque = 128u;
constexpr constant uint skip_triangles = 256u;
constexpr constant uint skip_aabbs = 512u;
// Opacity micromaps (VK_EXT_opacity_micromap): the 4-state micromaps this ray meets act as 2-state ones
// (unknown-transparent as transparent, unknown-opaque as opaque). A constant flags operand with this
// bit declares the micromap capability; the host enables the micromap feature.
constexpr constant uint force_opacity_micromap_2_state = 1024u;
}
// The hit kinds a triangle reports (HitKindKHR); an intersection entry reports its own 0..127.
constexpr constant uint hit_kind_front_facing_triangle = 0xFEu;
constexpr constant uint hit_kind_back_facing_triangle = 0xFFu;

// Built-ins. Each is legal in the stages SPV_KHR_ray_tracing lists; any other stage is refused
// at the call, even when the call sits in a helper.
metal::uint3 launch_id() M2V_RAY("launch_id");
metal::uint3 launch_size() M2V_RAY("launch_size");
metal::float3 world_ray_origin() M2V_RAY("world_ray_origin");
metal::float3 world_ray_direction() M2V_RAY("world_ray_direction");
metal::float3 object_ray_origin() M2V_RAY("object_ray_origin");
metal::float3 object_ray_direction() M2V_RAY("object_ray_direction");
float ray_tmin() M2V_RAY("ray_tmin");
// The current end of the ray: in a hit entry the distance of the hit being processed.
float ray_tmax() M2V_RAY("ray_tmax");
uint incoming_ray_flags() M2V_RAY("incoming_ray_flags");
uint hit_kind() M2V_RAY("hit_kind");
// The instance's index in the top-level structure (InstanceId) and the 24-bit value the host gave it.
uint instance_id() M2V_RAY("instance_id");
uint instance_custom_index() M2V_RAY("instance_custom_index");
uint primitive_id() M2V_RAY("primitive_id");
uint geometry_index() M2V_RAY("geometry_index");
// The SIMD-group this invocation runs in until its next shader call (Vulkan repacks invocations at
// every trace and callable): its lane and its width. The simd_* functions of <metal_simdgroup> run in
// ray stages over the same group.
uint thread_index_in_simdgroup() M2V_RAY("thread_index_in_simdgroup");
uint threads_per_simdgroup() M2V_RAY("threads_per_simdgroup");
// Motion blur (VK_NV_ray_tracing_motion_blur): the time a trace_ray_motion gave this ray, 0..1 over the
// motion of the scene's structures. Legal where ray_tmax is.
float current_ray_time() M2V_RAY("current_ray_time");
// Four columns of three: the instance's transform and its inverse.
metal::float4x3 object_to_world() M2V_RAY("object_to_world");
metal::float4x3 world_to_object() M2V_RAY("world_to_object");

// Traces a ray. The payload is any struct lvalue of the caller: it is copied into the trace's
// payload storage before the call and back after it, so the miss or hit entry that runs sees it as
// its [[m2v::payload]] parameter. Legal in raygen, closest_hit and miss entries.
template <typename Payload>
void trace_ray(metal::raytracing::instance_acceleration_structure scene, uint ray_flags, uint cull_mask,
               uint sbt_record_offset, uint sbt_record_stride, uint miss_index,
               metal::raytracing::ray r, thread Payload &payload) M2V_RAY("trace_ray");
template <typename Payload>
void trace_ray(metal::raytracing::primitive_acceleration_structure scene, uint ray_flags, uint cull_mask,
               uint sbt_record_offset, uint sbt_record_stride, uint miss_index,
               metal::raytracing::ray r, thread Payload &payload) M2V_RAY("trace_ray");
// Traces a ray at a time in [0, 1] through a scene built with motion (VK_NV_ray_tracing_motion_blur):
// moving triangles and instances are where they are at that time. The pipeline is created with
// VK_PIPELINE_CREATE_RAY_TRACING_ALLOW_MOTION_BIT_NV and the device enables rayTracingMotionBlur.
template <typename Payload>
void trace_ray_motion(metal::raytracing::instance_acceleration_structure scene, uint ray_flags, uint cull_mask,
                      uint sbt_record_offset, uint sbt_record_stride, uint miss_index,
                      metal::raytracing::ray r, float time, thread Payload &payload) M2V_RAY("trace_ray_motion");
// Runs the callable SBT record `sbt_index` with the data copied in and back out, as trace_ray
// copies a payload. Legal in raygen, closest_hit, miss and callable entries.
template <typename Data>
void execute_callable(uint sbt_index, thread Data &data) M2V_RAY("execute_callable");
// Intersection entries: offers a hit at distance t with a kind in 0..127, and optionally the hit
// attributes a hit entry reads as its [[m2v::hit_attribute]] parameter. Returns whether the hit was
// accepted (an any-hit entry may ignore it; a farther hit than the current one is never accepted).
bool report_intersection(float t, uint kind) M2V_RAY("report_intersection");
template <typename Attributes>
bool report_intersection(float t, uint kind, Attributes attributes) M2V_RAY("report_intersection");
// Any-hit entries: discard this candidate and continue traversal, or accept it and stop traversal.
// Both end the invocation; payload writes made before the call are kept.
void ignore_intersection() M2V_RAY("ignore_intersection");
void terminate_ray() M2V_RAY("terminate_ray");
#undef M2V_RAY
} // namespace m2v
