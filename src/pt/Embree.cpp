#include "pt/Embree.h"

#include "pt/CpuTracer.h"

#include <chrono>
#include <stdexcept>

#if BASALT_WITH_EMBREE
#include <embree4/rtcore.h>
#endif

namespace pt {

#if BASALT_WITH_EMBREE

namespace {

// Handed to Embree with every query and read back in the filter callbacks.
struct QueryContext {
  RTCRayQueryContext context;  // first, so Embree's pointer is ours
  const TraceView *view;
  uint seed;
  float3 direction;  // world space, for the alpha test's level of detail
  float2 cone;       // the ray cone where the ray leaves
  uint ambiguous = 0;
};

void candidateFilter(const RTCFilterFunctionNArguments *args) {
  QueryContext *query = reinterpret_cast<QueryContext *>(args->context);
  for (unsigned i = 0; i < args->N; ++i) {
    if (args->valid[i] == 0) continue;
    const uint instance = RTCHitN_instID(args->hit, args->N, i, 0);
    const uint primitive = RTCHitN_primID(args->hit, args->N, i);
    const float2 barycentric(RTCHitN_u(args->hit, args->N, i), RTCHitN_v(args->hit, args->N, i));
    const float facing = RTCHitN_Ng_x(args->hit, args->N, i) * RTCRayN_dir_x(args->ray, args->N, i) +
                         RTCHitN_Ng_y(args->hit, args->N, i) * RTCRayN_dir_y(args->ray, args->N, i) +
                         RTCHitN_Ng_z(args->hit, args->N, i) * RTCRayN_dir_z(args->ray, args->N, i);
    const PtCpuScene &scene = query->view->scene;
    if ((scene.traceInstances[instance].flags & kInstanceBlended) != 0u) query->ambiguous = 1;
    // Embree reports the object-space normal and ray here: the object-space facing is glTF's
    // front, whatever the instance's determinant.
    if (!ptCandidateSolid(scene, instance, primitive, barycentric, facing < 0.0f ? 1u : 0u, query->seed, query->direction,
                          ptConeWidthOrLevelZero(query->cone, RTCRayN_tfar(args->ray, args->N, i))))
      args->valid[i] = 0;
  }
}

} // namespace

struct EmbreeScene::State {
  RTCDevice device = nullptr;
  RTCScene top = nullptr;
  std::vector<RTCScene> bottoms;
  // Embree reads a whole 16 bytes at the last vertex and index, so both are padded copies.
  std::vector<float> vertices;
  std::vector<uint> indices;
};

bool EmbreeScene::available() { return true; }

EmbreeScene::EmbreeScene(const CpuScene &scene) : state(std::make_unique<State>()) {
  const auto started = std::chrono::steady_clock::now();
  state->device = rtcNewDevice(nullptr);
  if (!state->device) throw std::runtime_error("Embree could not create a device");
  state->vertices = scene.vertices;
  state->vertices.resize(state->vertices.size() + 4, 0.0f);
  state->indices = scene.indices;
  state->indices.resize(state->indices.size() + 4, 0u);
  const std::size_t vertexCount = scene.vertices.size() / kVertexFloats;

  state->top = rtcNewScene(state->device);
  for (uint i = 0; i < scene.instances.size(); ++i) {
    const TraceInstance &instance = scene.instances[i];
    RTCScene bottom = rtcNewScene(state->device);
    RTCGeometry triangles = rtcNewGeometry(state->device, RTC_GEOMETRY_TYPE_TRIANGLE);
    // Indices are relative to the primitive's first vertex, so the vertex buffer starts there.
    rtcSetSharedGeometryBuffer(triangles, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                               state->vertices.data() + static_cast<std::size_t>(instance.vertexOffset) * kVertexFloats, 0,
                               kVertexFloats * sizeof(float), vertexCount - instance.vertexOffset);
    rtcSetSharedGeometryBuffer(triangles, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                               state->indices.data() + instance.firstIndex, 0, 3 * sizeof(uint), scene.triangleCounts[i]);
    // Single-sided geometry needs the filter too: ptCandidateSolid rejects its back faces.
    if ((instance.flags & (kInstanceMasked | kInstanceBlended)) != 0u ||
        (instance.flags & kInstanceDoubleSided) == 0u) {
      rtcSetGeometryIntersectFilterFunction(triangles, candidateFilter);
      rtcSetGeometryOccludedFilterFunction(triangles, candidateFilter);
    }
    rtcCommitGeometry(triangles);
    rtcAttachGeometry(bottom, triangles);
    rtcReleaseGeometry(triangles);
    rtcCommitScene(bottom);
    state->bottoms.push_back(bottom);

    RTCGeometry placed = rtcNewGeometry(state->device, RTC_GEOMETRY_TYPE_INSTANCE);
    rtcSetGeometryInstancedScene(placed, bottom);
    const float rows[12] = {instance.objectToWorld0.x, instance.objectToWorld0.y, instance.objectToWorld0.z,
                            instance.objectToWorld0.w, instance.objectToWorld1.x, instance.objectToWorld1.y,
                            instance.objectToWorld1.z, instance.objectToWorld1.w, instance.objectToWorld2.x,
                            instance.objectToWorld2.y, instance.objectToWorld2.z, instance.objectToWorld2.w};
    rtcSetGeometryTransform(placed, 0, RTC_FORMAT_FLOAT3X4_ROW_MAJOR, rows);
    rtcSetGeometryMask(placed, instance.mask);
    rtcCommitGeometry(placed);
    rtcAttachGeometryByID(state->top, placed, i);  // the geometry id is the instance index, as the hardware's
    rtcReleaseGeometry(placed);
  }
  rtcCommitScene(state->top);
  buildMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

EmbreeScene::~EmbreeScene() {
  if (state->top) rtcReleaseScene(state->top);
  for (RTCScene bottom : state->bottoms) rtcReleaseScene(bottom);
  if (state->device) rtcReleaseDevice(state->device);
}

PtHit EmbreeScene::trace(const TraceView &view, float3 origin, float3 direction, float tMax, uint mask, uint seed,
                         float2 cone, uint anyHit) const {
  PtHit hit;
  hit.ambiguous = 0;
  hit.t = tMax;
  hit.barycentric = float2(0.0f);
  hit.instance = 0u;
  hit.primitive = 0u;
  hit.found = 0u;
  QueryContext query;
  rtcInitRayQueryContext(&query.context);
  query.view = &view;
  query.seed = seed;
  query.direction = direction;
  query.cone = cone;

  if (anyHit != 0u) {
    RTCRay ray{};
    ray.org_x = origin.x;
    ray.org_y = origin.y;
    ray.org_z = origin.z;
    ray.dir_x = direction.x;
    ray.dir_y = direction.y;
    ray.dir_z = direction.z;
    ray.tnear = 0.0f;
    ray.tfar = tMax;
    ray.mask = mask;
    RTCOccludedArguments arguments;
    rtcInitOccludedArguments(&arguments);
    arguments.context = &query.context;
    rtcOccluded1(state->top, &ray, &arguments);
    // Embree marks an occluded ray by setting its far distance to minus infinity.
    if (ray.tfar < 0.0f) hit.found = 1u;
    hit.ambiguous = query.ambiguous;
    return hit;
  }

  RTCRayHit rayHit{};
  rayHit.ray.org_x = origin.x;
  rayHit.ray.org_y = origin.y;
  rayHit.ray.org_z = origin.z;
  rayHit.ray.dir_x = direction.x;
  rayHit.ray.dir_y = direction.y;
  rayHit.ray.dir_z = direction.z;
  rayHit.ray.tnear = 0.0f;
  rayHit.ray.tfar = tMax;
  rayHit.ray.mask = mask;
  rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
  rayHit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
  RTCIntersectArguments arguments;
  rtcInitIntersectArguments(&arguments);
  arguments.context = &query.context;
  rtcIntersect1(state->top, &rayHit, &arguments);
  hit.ambiguous = query.ambiguous;
  if (rayHit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
    hit.found = 1u;
    hit.t = rayHit.ray.tfar;
    // Embree's (u, v) weight the second and third vertices, as the hardware's barycentrics do.
    hit.barycentric = float2(rayHit.hit.u, rayHit.hit.v);
    hit.instance = rayHit.hit.instID[0];
    hit.primitive = rayHit.hit.primID;
  }
  return hit;
}

#else

struct EmbreeScene::State {};
bool EmbreeScene::available() { return false; }
EmbreeScene::EmbreeScene(const CpuScene &) { throw std::runtime_error("this build has no Embree"); }
EmbreeScene::~EmbreeScene() = default;
PtHit EmbreeScene::trace(const TraceView &, float3, float3, float tMax, uint, uint, float2, uint) const {
  PtHit hit;
  hit.ambiguous = 0;
  hit.t = tMax;
  hit.barycentric = float2(0.0f);
  hit.instance = 0u;
  hit.primitive = 0u;
  hit.found = 0u;
  return hit;
}

#endif

} // namespace pt
