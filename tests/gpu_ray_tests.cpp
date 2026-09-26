// Hardware ray-query conformance against an independent double-precision world-space
// triangle oracle. This is a standalone CTest so capability absence is an explicit skip.
#include "core/Log.h"
#include "gpu/Descriptors.h"
#include "gpu/Pipeline.h"
#include "gpu/RayPipeline.h"
#include "gpu/Uploader.h"
#include "platform/Window.h"
#include "pt/Bvh.h"
#include "pt/BvhVariants.h"
#include "pt/Shared.h"
#include "render/RayTracing.h"
#include "render/Renderer.h"
#include "render/GpuBvhBuilder.h"
#include "render/TraceScene.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace basalt;

constexpr std::uint32_t kBaseRayCount = 100'000;
constexpr double kBarycentricTolerance = 3e-4;

struct alignas(16) OracleRay {
  pt::float4 originAndMin;
  pt::float4 directionAndMax;
  pt::uint4 control;
};
static_assert(sizeof(OracleRay) == 48);

struct alignas(16) OracleHit {
  pt::float4 distanceAndBarycentric;
  pt::uint4 identity;
};
static_assert(sizeof(OracleHit) == 32);
struct alignas(16) WavefrontRay { OracleRay ray; pt::uint4 identity; };
static_assert(sizeof(WavefrontRay) == 64);

struct ExpectedHit {
  bool found = false;
  double distance = 0.0;
  double u = 0.0, v = 0.0;
  std::uint32_t instance = 0, primitive = 0;
};

struct WorldTriangle {
  Vec3 vertices[3];
  std::uint32_t instance = 0, primitive = 0;
  std::uint32_t mask = 0, flags = 0, material = 0;
};

// The CPU tracer's view of loose host arrays: a downloaded tree traced on the host with the
// shared tracers.
struct HostView : pt::TraceView {
  HostView(const pt::HostTextures &textures, const std::vector<pt::TraceInstance> &instances,
           const std::vector<pt::Material> &materials, const std::vector<std::uint32_t> &indices, const float *vertices,
           std::size_t vertexFloats, const std::vector<pt::float4> &nodes = {}, const std::vector<pt::float4> &triangles = {})
      : TraceView(textures) {
    scene.traceInstances = pt::buffer(instances);
    scene.materials = pt::buffer(materials);
    scene.indices = pt::buffer(indices);
    scene.vertices = pt::buffer(vertices, vertexFloats);
    scene.bvhNodes = pt::buffer(nodes);
    scene.bvhTriangles = pt::buffer(triangles);
  }
};

struct Fixture {
  Scene scene;
  std::vector<Vertex> vertices;
  std::vector<std::uint32_t> indices;
  std::vector<float> materialAlpha;
  std::vector<float> materialCutoff;
  std::vector<WorldTriangle> triangles;
  Mat4 sharedTransform;
  Mat4 maskedRejectTransform;
  Mat4 maskedAcceptTransform;
  Mat4 blendedTransform;
  Mat4 groundTransform;
  Mat4 degenerateTransform;
  Mat4 oneSidedTransform;
  Mat4 mirroredOneSidedTransform;
  std::uint32_t sharedInstance = 0;
  std::uint32_t maskedRejectInstance = 0;
  std::uint32_t maskedAcceptInstance = 0;
  std::uint32_t blendedInstance = 0;
  std::uint32_t groundInstance = 0;
  std::uint32_t degenerateInstance = 0;
};

Mat4 transform(float yaw, Vec3 scale, Vec3 position) {
  const float half = yaw * 0.5f;
  return translation(position) * rotation({0.0f, std::sin(half), 0.0f, std::cos(half)}) * scaling(scale);
}

std::uint32_t addMaterial(Fixture &fixture, AlphaMode mode, float alpha, float cutoff = 0.5f,
                          bool doubleSided = true) {
  Material material;
  material.name = "oracle material";
  material.alphaMode = mode;
  material.doubleSided = doubleSided;
  material.uniforms.baseColorFactor = {0.6f, 0.7f, 0.8f, alpha};
  material.uniforms.alpha = {cutoff, static_cast<float>(mode), 1.0f, 0.0f};
  fixture.scene.materials.push_back(material);
  fixture.materialAlpha.push_back(alpha);
  fixture.materialCutoff.push_back(cutoff);
  return static_cast<std::uint32_t>(fixture.scene.materials.size() - 1);
}

std::uint32_t addMesh(Fixture &fixture, const std::vector<Vec3> &positions,
                      const std::vector<std::uint32_t> &indices, std::uint32_t material,
                      const Mat4 &model, const char *name) {
  const auto vertexOffset = static_cast<std::int32_t>(fixture.vertices.size());
  const auto firstIndex = static_cast<std::uint32_t>(fixture.indices.size());
  Aabb bounds;
  for (const Vec3 position : positions) {
    Vertex vertex;
    vertex.position = position;
    vertex.normal = {0.0f, 0.0f, 1.0f};
    vertex.tangent = {1.0f, 0.0f, 0.0f, 1.0f};
    vertex.uv0 = {0.5f, 0.5f};
    vertex.uv1 = vertex.uv0;
    fixture.vertices.push_back(vertex);
    bounds.add(position);
  }
  fixture.indices.insert(fixture.indices.end(), indices.begin(), indices.end());
  Primitive primitive;
  primitive.firstIndex = firstIndex;
  primitive.indexCount = static_cast<std::uint32_t>(indices.size());
  primitive.vertexOffset = vertexOffset;
  primitive.material = material;
  primitive.bounds = bounds;
  primitive.worldBounds = bounds.transformed(model);
  primitive.transform = model;
  primitive.name = name;
  fixture.scene.bounds.add(primitive.worldBounds);
  fixture.scene.primitives.push_back(primitive);
  return static_cast<std::uint32_t>(fixture.scene.primitives.size() - 1);
}

void makeFixture(Fixture &fixture) {
  const std::uint32_t opaque = addMaterial(fixture, AlphaMode::Opaque, 1.0f);
  const std::uint32_t maskedReject = addMaterial(fixture, AlphaMode::Mask, 0.3f);
  const std::uint32_t blended = addMaterial(fixture, AlphaMode::Blend, 0.5f);
  const std::uint32_t maskedAccept = addMaterial(fixture, AlphaMode::Mask, 0.8f);
  const std::uint32_t oneSided = addMaterial(fixture, AlphaMode::Opaque, 1.0f, 0.5f, false);

  // The same seeded random-triangle construction used by the CPU BVH/Embree corpus:
  // nonuniform transforms, including a negative scale, and both alpha modes.
  std::mt19937 random(23);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  for (int instance = 0; instance < 5; ++instance) {
    std::vector<Vec3> positions;
    std::vector<std::uint32_t> indices;
    for (std::uint32_t triangle = 0; triangle < 300; ++triangle) {
      const Vec3 centre(unit(random), unit(random), unit(random));
      for (int corner = 0; corner < 3; ++corner) {
        positions.push_back(centre + Vec3(unit(random), unit(random), unit(random)) * 0.15f);
        indices.push_back(triangle * 3 + static_cast<std::uint32_t>(corner));
      }
    }
    // Duplicate-centroid stress: stable Morton ties must retain primitive order. The
    // triangles vary in shape but all have an exact object-space centroid of zero.
    if (instance == 0) {
      for (std::uint32_t triangle = 0; triangle < 32; ++triangle) {
        const float radius = 0.02f + static_cast<float>(triangle) * 0.001f;
        const float angle = static_cast<float>(triangle) * 0.37f;
        const float z = (0.05f + 0.2f * std::abs(std::sin(angle))) * radius;
        const Vec3 a{radius, 0.0f, z};
        const Vec3 b{0.0f, radius, -z};
        const std::uint32_t base = static_cast<std::uint32_t>(positions.size());
        positions.push_back(a);
        positions.push_back(b);
        positions.push_back({-radius, -radius, 0.0f});
        indices.insert(indices.end(), {base, base + 1u, base + 2u});
      }
    }
    // A zero-area primitive stays in each BLAS and must never become a hit.
    const std::uint32_t base = static_cast<std::uint32_t>(positions.size());
    positions.insert(positions.end(), {{0.0f, 0.0f, 0.0f}, {0.2f, 0.0f, 0.0f}, {0.4f, 0.0f, 0.0f}});
    indices.insert(indices.end(), {base, base + 1, base + 2});
    const float sign = instance == 3 ? -1.0f : 1.0f;
    const std::uint32_t material = instance == 2 ? maskedReject
                                     : instance == 3 ? blended
                                     : instance == 4 ? maskedAccept
                                                     : opaque;
    const Mat4 model = transform(0.5f * static_cast<float>(instance),
                                 {sign * (0.5f + 0.2f * instance), 0.65f + 0.1f * instance,
                                  1.1f - 0.1f * instance},
                                 {static_cast<float>(instance) * 0.9f - 1.8f,
                                  0.1f * static_cast<float>(instance), 0.0f});
    addMesh(fixture, positions, indices, material, model, "random soup");
  }

  // Two triangles sharing an edge under a rotated, mirrored, nonuniform transform.
  fixture.sharedTransform = transform(0.35f, {-1.2f, 0.8f, 1.1f}, {5.0f, 0.0f, 0.0f});
  fixture.sharedInstance = addMesh(fixture,
                                   {{-1.0f, -1.0f, 0.0f}, {1.0f, -1.0f, 0.0f},
                                    {1.0f, 1.0f, 0.0f}, {-1.0f, 1.0f, 0.0f}},
                                   {0, 1, 2, 0, 2, 3}, opaque, fixture.sharedTransform,
                                   "shared edge");
  const std::vector<Vec3> probePositions{{-0.75f, -0.75f, 0.0f}, {0.75f, -0.75f, 0.0f},
                                         {0.75f, 0.75f, 0.0f}, {-0.75f, 0.75f, 0.0f}};
  const std::vector<std::uint32_t> probeIndices{0, 1, 2, 0, 2, 3};
  fixture.maskedRejectTransform = transform(-0.2f, {0.8f, 1.3f, 0.9f}, {8.0f, 0.0f, 0.0f});
  fixture.maskedRejectInstance = addMesh(fixture, probePositions, probeIndices, maskedReject,
                                         fixture.maskedRejectTransform, "masked reject probe");
  fixture.maskedAcceptTransform = transform(0.25f, {1.2f, 0.7f, 1.1f}, {10.5f, 0.0f, 0.0f});
  fixture.maskedAcceptInstance = addMesh(fixture, probePositions, probeIndices, maskedAccept,
                                         fixture.maskedAcceptTransform, "masked accept probe");
  fixture.blendedTransform = transform(0.4f, {-0.9f, 1.1f, 0.8f}, {13.0f, 0.0f, 0.0f});
  fixture.blendedInstance = addMesh(fixture, probePositions, probeIndices, blended,
                                    fixture.blendedTransform, "blended probe");
  fixture.groundTransform = transform(-0.3f, {1.0f, 0.8f, 1.2f}, {15.5f, 0.0f, 0.0f});
  fixture.groundInstance = addMesh(fixture, probePositions, probeIndices, opaque,
                                   fixture.groundTransform, "ground-mask probe");
  fixture.scene.groundPrimitive = static_cast<int>(fixture.groundInstance);
  fixture.degenerateTransform = translation({18.0f, 0.0f, 0.0f});
  fixture.degenerateInstance = addMesh(fixture,
                                       {{-0.5f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f},
                                        {0.5f, 0.0f, 0.0f}},
                                       {0, 1, 2}, opaque, fixture.degenerateTransform,
                                       "degenerate");
  fixture.oneSidedTransform = translation({20.0f, 0.0f, 0.0f});
  fixture.mirroredOneSidedTransform = transform(0.0f, {-1.0f, 1.0f, 1.0f}, {23.0f, 0.0f, 0.0f});
  addMesh(fixture, probePositions, probeIndices, oneSided, fixture.oneSidedTransform, "one-sided probe");
  addMesh(fixture, probePositions, probeIndices, oneSided, fixture.mirroredOneSidedTransform,
          "mirrored one-sided probe");
}

std::array<double, 3> subtract(Vec3 a, Vec3 b) {
  return {static_cast<double>(a.x) - static_cast<double>(b.x),
          static_cast<double>(a.y) - static_cast<double>(b.y),
          static_cast<double>(a.z) - static_cast<double>(b.z)};
}

std::array<double, 3> crossDouble(const std::array<double, 3> &a,
                                  const std::array<double, 3> &b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}

double dotDouble(const std::array<double, 3> &a, const std::array<double, 3> &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

bool intersect(const OracleRay &ray, const WorldTriangle &triangle, double limit,
               double &distance, double &u, double &v, bool *frontFacing = nullptr) {
  const auto edge1 = subtract(triangle.vertices[1], triangle.vertices[0]);
  const auto edge2 = subtract(triangle.vertices[2], triangle.vertices[0]);
  const std::array<double, 3> direction{ray.directionAndMax.x, ray.directionAndMax.y,
                                        ray.directionAndMax.z};
  const auto p = crossDouble(direction, edge2);
  const double determinant = dotDouble(edge1, p);
  if (std::abs(determinant) < 1e-14) return false;
  const double inverse = 1.0 / determinant;
  const Vec3 origin(ray.originAndMin.x, ray.originAndMin.y, ray.originAndMin.z);
  const auto fromVertex = subtract(origin, triangle.vertices[0]);
  u = dotDouble(fromVertex, p) * inverse;
  if (u < 0.0 || u > 1.0) return false;
  const auto q = crossDouble(fromVertex, edge1);
  v = dotDouble(direction, q) * inverse;
  if (v < 0.0 || u + v > 1.0) return false;
  const double t = dotDouble(edge2, q) * inverse;
  if (!(t > static_cast<double>(ray.originAndMin.w)) || !(t < limit)) return false;
  distance = t;
  if (frontFacing) *frontFacing = determinant > 0.0;
  return true;
}

bool candidateSolid(const Fixture &fixture, const WorldTriangle &triangle,
                    std::uint32_t seed) {
  const float alpha = fixture.materialAlpha[triangle.material];
  if ((triangle.flags & pt::kInstanceBlended) != 0u)
    return pt::hashFloat(seed ^ pt::pcgHash(triangle.instance * 0x9E3779B9u + triangle.primitive)) < alpha;
  if ((triangle.flags & pt::kInstanceMasked) != 0u)
    return alpha >= fixture.materialCutoff[triangle.material];
  return true;
}

ExpectedHit traceOracle(const Fixture &fixture, const OracleRay &ray) {
  ExpectedHit hit;
  hit.distance = ray.directionAndMax.w;
  for (const WorldTriangle &triangle : fixture.triangles) {
    if ((triangle.mask & ray.control.x) == 0u) continue;
    double distance = 0.0, u = 0.0, v = 0.0;
    bool frontFacing = false;
    if (!intersect(ray, triangle, hit.distance, distance, u, v, &frontFacing)) continue;
    // glTF 2.0: a negative-determinant transform keeps the object-space front, whose world
    // winding is clockwise.
    if ((triangle.flags & pt::kInstanceMirrored) != 0u) frontFacing = !frontFacing;
    if ((triangle.flags & pt::kInstanceDoubleSided) == 0u && !frontFacing) continue;
    if (!candidateSolid(fixture, triangle, ray.control.y)) continue;
    hit.found = true;
    hit.distance = distance;
    hit.u = u;
    hit.v = v;
    hit.instance = triangle.instance;
    hit.primitive = triangle.primitive;
  }
  return hit;
}

void buildWorldTriangles(Fixture &fixture, const TraceScene &trace) {
  for (std::uint32_t instance = 0; instance < trace.instances.size(); ++instance) {
    const pt::TraceInstance &record = trace.instances[instance];
    const Primitive &primitive = fixture.scene.primitives[trace.primitives[instance]];
    for (std::uint32_t triangle = 0; triangle < primitive.indexCount / 3; ++triangle) {
      WorldTriangle world;
      world.instance = instance;
      world.primitive = triangle;
      world.mask = record.mask;
      world.flags = record.flags;
      world.material = record.material;
      for (std::uint32_t corner = 0; corner < 3; ++corner) {
        const std::uint32_t vertex =
            fixture.indices[record.firstIndex + triangle * 3 + corner] + record.vertexOffset;
        world.vertices[corner] = transformPoint(primitive.transform, fixture.vertices[vertex].position);
      }
      fixture.triangles.push_back(world);
    }
  }
}

OracleRay makeRay(Vec3 origin, Vec3 direction, float minimum, float maximum,
                  std::uint32_t mask, std::uint32_t seed) {
  direction = normalize(direction);
  return {{origin.x, origin.y, origin.z, minimum},
          {direction.x, direction.y, direction.z, maximum},
          {mask, seed, 0u, 0u}};
}

std::vector<OracleRay> makeCorpus(const Fixture &fixture) {
  std::vector<OracleRay> rays;
  rays.reserve(kBaseRayCount);
  const Vec3 centre = transformPoint(fixture.sharedTransform, {0.0f, 0.0f, 0.0f});
  const Vec3 edgePoint = transformPoint(fixture.sharedTransform, {0.5f, 0.5f, 0.0f});
  const Vec3 normal = normalize(transformDirection(fixture.sharedTransform, {0.0f, 0.0f, 1.0f}));
  rays.push_back(makeRay(centre + normal * 2.0f, -normal, 0.0f, 10.0f, pt::kRayMaskScene, 1u));
  rays.push_back(makeRay(centre + normal * 2.0f, -normal, 0.0f, 1.999f, pt::kRayMaskScene, 2u));
  rays.push_back(makeRay(centre + normal * 2.0f, -normal, 0.0f, 2.001f, pt::kRayMaskScene, 3u));
  rays.push_back(makeRay(centre + normal * 2.0f, -normal, 2.001f, 10.0f, pt::kRayMaskScene, 4u));
  rays.push_back(makeRay(edgePoint + normal * 2.0f, -normal, 0.0f, 10.0f, pt::kRayMaskScene, 5u));
  rays.push_back(makeRay(centre + normal * 2.0f, -normal, 0.0f, 10.0f, pt::kRayMaskBlended, 6u));
  auto probeRay = [&](const Mat4 &model, std::uint32_t mask, std::uint32_t seed) {
    // Away from the quad diagonal so the alpha seed selects one known primitive.
    const Vec3 point = transformPoint(model, {0.2f, -0.1f, 0.0f});
    const Vec3 probeNormal = normalize(transformDirection(model, {0.0f, 0.0f, 1.0f}));
    return makeRay(point + probeNormal * 2.0f, -probeNormal, 0.0f, 10.0f, mask, seed);
  };
  rays.push_back(probeRay(fixture.degenerateTransform, pt::kRayMaskScene, 7u));
  // From the side where the transformed quad winds counter-clockwise, then from the other:
  // the one-sided quad is hit by the first, the mirrored one (glTF: object-space front) by the
  // second.
  auto sidedRays = [&](const Mat4 &model, std::uint32_t seed) {
    const Vec3 point = transformPoint(model, {0.2f, -0.1f, 0.0f});
    const Vec3 probeNormal = normalize(cross(transformDirection(model, {1.0f, 0.0f, 0.0f}),
                                             transformDirection(model, {0.0f, 1.0f, 0.0f})));
    rays.push_back(makeRay(point + probeNormal * 2.0f, -probeNormal, 0.0f, 10.0f, pt::kRayMaskScene, seed));
    rays.push_back(makeRay(point - probeNormal * 2.0f, probeNormal, 0.0f, 10.0f, pt::kRayMaskScene, seed + 1u));
  };
  sidedRays(fixture.oneSidedTransform, 70u);
  sidedRays(fixture.mirroredOneSidedTransform, 72u);
  rays.push_back(probeRay(fixture.maskedRejectTransform, pt::kRayMaskScene, 8u));
  rays.push_back(probeRay(fixture.maskedAcceptTransform, pt::kRayMaskScene, 9u));
  std::uint32_t blendAcceptSeed = 0u, blendRejectSeed = 0u;
  while (pt::hashFloat(blendAcceptSeed ^
                       pt::pcgHash(fixture.blendedInstance * 0x9E3779B9u)) >= 0.5f)
    ++blendAcceptSeed;
  while (pt::hashFloat(blendRejectSeed ^
                       pt::pcgHash(fixture.blendedInstance * 0x9E3779B9u)) < 0.5f)
    ++blendRejectSeed;
  rays.push_back(probeRay(fixture.blendedTransform, pt::kRayMaskBlended, blendAcceptSeed));
  rays.push_back(probeRay(fixture.blendedTransform, pt::kRayMaskBlended, blendRejectSeed));
  rays.push_back(probeRay(fixture.groundTransform, pt::kRayMaskScene, 10u));
  rays.push_back(probeRay(fixture.groundTransform, pt::kRayMaskGround, 11u));

  std::mt19937 random(5);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  const std::array<std::uint32_t, 5> masks{
      pt::kRayMaskScene, pt::kRayMaskGround, pt::kRayMaskBlended,
      pt::kRayMaskScene | pt::kRayMaskBlended,
      pt::kRayMaskScene | pt::kRayMaskGround | pt::kRayMaskBlended};
  while (rays.size() < kBaseRayCount) {
    const Vec3 origin = Vec3(unit(random), unit(random), unit(random)) * 3.0f;
    const Vec3 target(unit(random) * 1.8f, unit(random) * 0.8f, unit(random) * 0.8f);
    const std::size_t index = rays.size();
    const float minimum = index % 17 == 0 ? 0.2f : 0.0f;
    const float maximum = index % 11 == 0 ? 1.5f + 2.0f * std::abs(unit(random)) : 100.0f;
    rays.push_back(makeRay(origin, target - origin, minimum, maximum,
                           masks[index % masks.size()], random()));
  }
  return rays;
}

pt::Material convertMaterial(const MaterialUniforms &source) {
  auto convert = [](Vec4 value) { return pt::float4(value.x, value.y, value.z, value.w); };
  pt::Material material{};
  material.baseColorFactor = convert(source.baseColorFactor);
  material.emissive = convert(source.emissive);
  material.factors = convert(source.factors);
  material.alpha = convert(source.alpha);
  material.texture = pt::uint4(source.texture[0], source.texture[1], source.texture[2], source.texture[3]);
  material.transmission = convert(source.transmission);
  material.clearcoat = convert(source.clearcoat);
  material.extensionTextures = pt::uint4(source.extensionTextures[0], source.extensionTextures[1],
                                         source.extensionTextures[2], source.extensionTextures[3]);
  return material;
}

double distanceTolerance(double distance) { return std::max(2e-7, 1e-4 * distance); }

const WorldTriangle *findTriangle(const Fixture &fixture, std::uint32_t instance,
                                  std::uint32_t primitive) {
  for (const WorldTriangle &triangle : fixture.triangles)
    if (triangle.instance == instance && triangle.primitive == primitive) return &triangle;
  return nullptr;
}

int compareResults(const Fixture &fixture, const std::vector<OracleRay> &baseRays,
                   const std::vector<ExpectedHit> &expected,
                   const std::vector<OracleHit> &actual) {
  std::uint32_t nearestMismatch = 0, occlusionMismatch = 0, geometricTies = 0;
  for (std::uint32_t index = 0; index < kBaseRayCount; ++index) {
    const ExpectedHit &want = expected[index];
    const OracleHit &got = actual[index];
    const bool found = got.identity.x != 0u;
    bool explained = found == want.found;
    if (explained && found) {
      const double distance = got.distanceAndBarycentric.x;
      const double difference = std::abs(distance - want.distance);
      const bool identity = got.identity.y == want.instance && got.identity.z == want.primitive;
      if (difference > distanceTolerance(want.distance)) {
        explained = false;
      } else if (identity) {
        explained = std::abs(static_cast<double>(got.distanceAndBarycentric.y) - want.u) <=
                        kBarycentricTolerance &&
                    std::abs(static_cast<double>(got.distanceAndBarycentric.z) - want.v) <=
                        kBarycentricTolerance;
      } else {
        const WorldTriangle *alternative = findTriangle(fixture, got.identity.y, got.identity.z);
        double alternativeDistance = 0.0, alternativeU = 0.0, alternativeV = 0.0;
        const bool validAlternative = alternative &&
            (alternative->mask & baseRays[index].control.x) != 0u &&
            candidateSolid(fixture, *alternative, baseRays[index].control.y) &&
            intersect(baseRays[index], *alternative, baseRays[index].directionAndMax.w,
                      alternativeDistance, alternativeU, alternativeV);
        // Either a shared boundary or deliberately coincident geometry may have more
        // than one correct identity. Both triangles must independently hit at the same
        // distance under the tolerance fixed for the corpus.
        const bool tie = validAlternative &&
            std::abs(alternativeDistance - want.distance) <= distanceTolerance(want.distance);
        explained = tie;
        if (tie) ++geometricTies;
      }
    }
    if (!explained) {
      if (nearestMismatch < 10) {
        std::printf("nearest mismatch %u: CPU %u %u/%u t %.9g bary %.9g %.9g; "
                    "GPU %u %u/%u t %.9g bary %.9g %.9g\n",
                    index, want.found ? 1u : 0u, want.instance, want.primitive, want.distance,
                    want.u, want.v, found ? 1u : 0u, got.identity.y, got.identity.z,
                    got.distanceAndBarycentric.x, got.distanceAndBarycentric.y,
                    got.distanceAndBarycentric.z);
      }
      ++nearestMismatch;
    }

    // Occlusion is a boolean contract. With accept-any-intersection, distance and identity
    // are deliberately traversal-order-dependent and are not an observable result.
    const OracleHit &occlusion = actual[kBaseRayCount + index];
    if ((occlusion.identity.x != 0u) != want.found) {
      if (occlusionMismatch < 10)
        std::printf("occlusion mismatch %u: CPU %u, GPU %u\n", index,
                    want.found ? 1u : 0u, occlusion.identity.x != 0u ? 1u : 0u);
      ++occlusionMismatch;
    }
  }
  std::printf("GPU ray oracle: %u nearest + %u occlusion queries, %u classified geometric ties\n",
              kBaseRayCount, kBaseRayCount, geometricTies);
  if (nearestMismatch != 0 || occlusionMismatch != 0) {
    std::fprintf(stderr, "%u nearest and %u occlusion mismatches were unexplained\n",
                 nearestMismatch, occlusionMismatch);
    return 1;
  }
  return 0;
}

struct DownloadedTree {
  std::vector<pt::float4> nodes, triangles;
  std::vector<pt::TraceInstance> instances;
};

DownloadedTree downloadTree(Uploader &uploader, const GpuBvhBuildResult &build) {
  DownloadedTree tree;
  const std::vector<std::uint8_t> nodes = uploader.readBuffer(build.nodes, build.nodes.size);
  const std::vector<std::uint8_t> triangles = uploader.readBuffer(build.triangles, build.triangles.size);
  tree.nodes.resize(nodes.size() / sizeof(pt::float4));
  // Only the published triangles: an empty scene's one-element buffer is never written.
  tree.triangles.resize(static_cast<std::size_t>(build.statistics.triangles) * 3u);
  std::memcpy(tree.nodes.data(), nodes.data(), nodes.size());
  std::memcpy(tree.triangles.data(), triangles.data(), tree.triangles.size() * sizeof(pt::float4));
  tree.instances = build.instances;
  return tree;
}

bool sameBits(const pt::float4 &a, const pt::float4 &b) { return std::memcmp(&a, &b, sizeof(a)) == 0; }

// Two published binary trees are the same tree when, walked from the same roots, every
// child slot matches bitwise (bounds, count, and the data of leaves) and interior children
// match recursively. Interior node numbering is not compared. "" when identical.
std::string compareTrees(const DownloadedTree &a, const DownloadedTree &b) {
  if (a.nodes.size() != b.nodes.size()) return "node arrays differ in size";
  if (a.triangles.size() != b.triangles.size()) return "triangle arrays differ in size";
  for (std::size_t i = 0; i < a.triangles.size(); ++i)
    if (!sameBits(a.triangles[i], b.triangles[i])) return "triangle float4 " + std::to_string(i) + " differs";
  if (a.instances.size() != b.instances.size()) return "instance counts differ";
  struct Pair { pt::uint na, nb; };
  std::vector<Pair> stack{{0u, 0u}};
  for (std::size_t i = 0; i < a.instances.size(); ++i) {
    if (a.instances[i].triangleOffset != b.instances[i].triangleOffset) return "instance triangle offsets differ";
    stack.push_back({a.instances[i].blasRoot, b.instances[i].blasRoot});
  }
  std::size_t visited = 0;
  const std::size_t nodeCount = a.nodes.size() / 4u;
  while (!stack.empty()) {
    const Pair pair = stack.back();
    stack.pop_back();
    if (pair.na >= nodeCount || pair.nb >= nodeCount) return "a child index is outside the node array";
    if (++visited > nodeCount) return "a tree revisits nodes";
    for (pt::uint side = 0; side < 2u; ++side) {
      const pt::float4 &lowA = a.nodes[pair.na * 4u + side * 2u], &highA = a.nodes[pair.na * 4u + side * 2u + 1u];
      const pt::float4 &lowB = b.nodes[pair.nb * 4u + side * 2u], &highB = b.nodes[pair.nb * 4u + side * 2u + 1u];
      const pt::uint countA = pt::as_type<pt::uint>(highA.w), countB = pt::as_type<pt::uint>(highB.w);
      const pt::uint dataA = pt::as_type<pt::uint>(lowA.w), dataB = pt::as_type<pt::uint>(lowB.w);
      if (!sameBits(highA, highB) || countA != countB ||
          std::memcmp(&lowA, &lowB, 3 * sizeof(float)) != 0)
        return "nodes " + std::to_string(pair.na) + " / " + std::to_string(pair.nb) + " differ in child " +
               std::to_string(side) + "'s bounds or count";
      const bool interior = countA == 0u && dataA != 0xFFFFFFFFu;
      if (interior) stack.push_back({dataA, dataB});
      else if (dataA != dataB)
        return "nodes " + std::to_string(pair.na) + " / " + std::to_string(pair.nb) + " differ in leaf data";
    }
  }
  return {};
}

// A scene's geometry read back, and each instance's triangle count: the CPU builder's and split
// clipping's input.
struct HostGeometry {
  std::vector<float> vertices;
  std::vector<std::uint32_t> indices, triangleCounts;
};

HostGeometry readGeometry(Uploader &uploader, const Scene &scene, const TraceScene &trace) {
  HostGeometry geometry;
  geometry.vertices.resize(static_cast<std::size_t>(scene.vertexCount) * pt::kVertexFloats);
  geometry.indices.resize(scene.indexCount);
  std::memcpy(geometry.vertices.data(),
              uploader.readBuffer(scene.vertexBuffer, geometry.vertices.size() * sizeof(float)).data(),
              geometry.vertices.size() * sizeof(float));
  std::memcpy(geometry.indices.data(), uploader.readBuffer(scene.indexBuffer, geometry.indices.size() * 4u).data(),
              geometry.indices.size() * 4u);
  for (const std::uint32_t primitive : trace.primitives)
    geometry.triangleCounts.push_back(scene.primitives[primitive].indexCount / 3u);
  return geometry;
}

// A published binary tree is well formed: walked from its roots, every node is reached once and
// all are reached; an interior child's slot bounds its node's children; every bottom-level leaf
// is one triangle of its instance (or, for the SAH builders, up to eight consecutive ones), each
// once, inside its slot's bounds (with split clipping, a slot within its triangle's bounds: a
// reference's box); every top-level leaf is one instance, each once. "" when well formed.
std::string checkTree(const DownloadedTree &tree, bool clipped = false) {
  const std::size_t nodeCount = tree.nodes.size() / 4u, triangleCount = tree.triangles.size() / 3u;
  const std::size_t instanceCount = tree.instances.size();
  std::vector<std::uint8_t> reached(nodeCount, 0u), triangleSeen(triangleCount, 0u), instanceSeen(instanceCount, 0u);
  // Each instance's triangles: from its offset to the next larger offset (or the end).
  std::vector<pt::uint> offsets;
  for (const pt::TraceInstance &instance : tree.instances) offsets.push_back(instance.triangleOffset);
  std::sort(offsets.begin(), offsets.end());
  auto triangleEnd = [&](pt::uint first) {
    const auto next = std::upper_bound(offsets.begin(), offsets.end(), first);
    return next == offsets.end() ? static_cast<pt::uint>(triangleCount) : *next;
  };
  // Subnormal coordinates (of the point and of the box) compare as zero: devices may flush them
  // in the builders' box arithmetic (Vulkan's default float controls), and every GPU builder's
  // boxes then do.
  auto inside = [](const pt::float4 &low, const pt::float4 &high, float x, float y, float z) {
    auto f = [](float v) { return std::fpclassify(v) == FP_SUBNORMAL ? 0.0f : v; };
    return f(low.x) <= f(x) && f(x) <= f(high.x) && f(low.y) <= f(y) && f(y) <= f(high.y) && f(low.z) <= f(z) &&
           f(z) <= f(high.z);
  };
  struct Entry { pt::uint node; std::int64_t instance; };  // instance -1: the TLAS
  std::vector<Entry> stack{{0u, -1}};
  for (std::size_t i = 0; i < instanceCount; ++i) stack.push_back({tree.instances[i].blasRoot, std::int64_t(i)});
  std::size_t visited = 0;
  while (!stack.empty()) {
    const Entry entry = stack.back();
    stack.pop_back();
    if (entry.node >= nodeCount) return "a child index is outside the node array";
    if (reached[entry.node]++ != 0u) return "node " + std::to_string(entry.node) + " is reached twice";
    ++visited;
    for (pt::uint side = 0; side < 2u; ++side) {
      const pt::float4 &low = tree.nodes[entry.node * 4u + side * 2u], &high = tree.nodes[entry.node * 4u + side * 2u + 1u];
      const pt::uint data = pt::as_type<pt::uint>(low.w), count = pt::as_type<pt::uint>(high.w);
      const std::string where = "node " + std::to_string(entry.node) + " child " + std::to_string(side);
      if (data == 0xFFFFFFFFu) {
        if (count != 0u) return where + " is empty with a count";
        continue;
      }
      if (count == 0u) {
        if (data >= nodeCount) return where + " points outside the node array";
        for (pt::uint inner = 0; inner < 2u; ++inner) {
          const pt::float4 &innerLow = tree.nodes[data * 4u + inner * 2u];
          const pt::float4 &innerHigh = tree.nodes[data * 4u + inner * 2u + 1u];
          if (pt::as_type<pt::uint>(innerLow.w) == 0xFFFFFFFFu) continue;
          if (!inside(low, high, innerLow.x, innerLow.y, innerLow.z) ||
              !inside(low, high, innerHigh.x, innerHigh.y, innerHigh.z))
            return where + " does not bound its node's children";
        }
        stack.push_back({data, entry.instance});
        continue;
      }
      if (count > (entry.instance < 0 ? 1u : 8u))
        return where + " is a leaf of " + std::to_string(count) + " primitives";
      if (entry.instance < 0) {
        if (data >= instanceCount || instanceSeen[data]++ != 0u) return where + " is a missing or repeated instance";
        continue;
      }
      const pt::uint first = tree.instances[std::size_t(entry.instance)].triangleOffset;
      // Clipped, the leaf's box is its references' boxes: within its triangles' box.
      pt::float4 trianglesLow(3.0e38f), trianglesHigh(-3.0e38f);
      for (pt::uint triangle = data; triangle < data + count; ++triangle) {
        if (triangle < first || triangle >= triangleEnd(first) || triangleSeen[triangle]++ != 0u)
          return where + " is a triangle outside its instance or a repeated one";
        if (clipped) {
          for (pt::uint vertex = 0; vertex < 3u; ++vertex) {
            const pt::float4 &v = tree.triangles[triangle * 3u + vertex];
            trianglesLow = pt::float4(std::min(trianglesLow.x, v.x), std::min(trianglesLow.y, v.y),
                                      std::min(trianglesLow.z, v.z), 0.0f);
            trianglesHigh = pt::float4(std::max(trianglesHigh.x, v.x), std::max(trianglesHigh.y, v.y),
                                       std::max(trianglesHigh.z, v.z), 0.0f);
          }
          continue;
        }
        for (pt::uint vertex = 0; vertex < 3u; ++vertex) {
          const pt::float4 &position = tree.triangles[triangle * 3u + vertex];
          if (!inside(low, high, position.x, position.y, position.z)) {
            char values[256];
            std::snprintf(values, sizeof(values), " (%a %a %a outside %a %a %a .. %a %a %a)", position.x, position.y,
                          position.z, low.x, low.y, low.z, high.x, high.y, high.z);
            return where + " does not bound its triangle" + values;
          }
        }
      }
      if (clipped && (!(low.x <= high.x && low.y <= high.y && low.z <= high.z) ||
                      !inside(trianglesLow, trianglesHigh, low.x, low.y, low.z) ||
                      !inside(trianglesLow, trianglesHigh, high.x, high.y, high.z)))
        return where + " is not a box within its triangles' box";
    }
  }
  if (visited != nodeCount) return "only " + std::to_string(visited) + " of " + std::to_string(nodeCount) + " nodes are reached";
  for (std::size_t i = 0; i < triangleCount; ++i)
    if (triangleSeen[i] == 0u) return "triangle " + std::to_string(i) + " is in no leaf";
  for (std::size_t i = 0; i < instanceCount; ++i)
    if (instanceSeen[i] == 0u) return "instance " + std::to_string(i) + " is in no leaf";
  return {};
}

// The parallel builder's radix sort against std::stable_sort on (hi, lo): exact equality,
// over sizes around the 1024-key tile and with many duplicate keys.
void runSortTest(Uploader &uploader, GpuLbvhBuilder &builder) {
  std::mt19937 random(20260924u);
  const std::uint32_t sizes[] = {0u, 1u, 2u, 1000u, 1023u, 1024u, 1025u, 4097u, 300000u};
  const std::uint32_t segmentCounts[] = {1u, 7u, 300u};
  for (const std::uint32_t count : sizes) {
    for (const std::uint32_t segments : segmentCounts) {
      std::vector<std::uint32_t> lo(count), hi(count), values(count);
      for (std::uint32_t i = 0; i < count; ++i) {
        // Few distinct low keys (heavy duplication) mixed with full 30-bit codes.
        lo[i] = (i % 3u == 0u) ? static_cast<std::uint32_t>(random() % 16u) : static_cast<std::uint32_t>(random() & 0x3FFFFFFFu);
        hi[i] = static_cast<std::uint32_t>(random() % segments);
        values[i] = i;
      }
      std::vector<std::uint32_t> order(count);
      for (std::uint32_t i = 0; i < count; ++i) order[i] = i;
      std::stable_sort(order.begin(), order.end(), [&](std::uint32_t x, std::uint32_t y) {
        return hi[x] != hi[y] ? hi[x] < hi[y] : lo[x] < lo[y];
      });
      const std::uint32_t highPasses = segments > 1u ? (static_cast<std::uint32_t>(std::bit_width(segments - 1u)) + 7u) / 8u : 0u;
      std::vector<std::uint32_t> gpuLo = lo, gpuHi = hi, gpuValues = values;
      builder.sortTriples(uploader, gpuLo, gpuHi, gpuValues, highPasses);
      for (std::uint32_t i = 0; i < count; ++i)
        if (gpuValues[i] != order[i] || gpuLo[i] != lo[order[i]] || gpuHi[i] != hi[order[i]])
          throw Error("GPU radix sort differs from std::stable_sort at " + std::to_string(i) + " of " +
                      std::to_string(count) + " keys, " + std::to_string(segments) + " segments");
    }
  }
  std::printf("PASS: GPU radix sort equals std::stable_sort (0 to 300000 keys, 1 to 300 segments)\n");
}

// The GPU collapse of a resident binary tree against pt::buildWideBvh of the same tree: the
// nodes byte for byte, the bottom levels' roots and the traversal stack bound.
GpuWideCollapseResult checkCollapse(Uploader &uploader, GpuBvhCollapser &collapser, const Buffer &resident,
                                    const std::vector<pt::float4> &binaryNodes,
                                    const std::vector<pt::TraceInstance> &instances, std::uint32_t width,
                                    const std::string &what, const pt::BvhStatistics &statistics) {
  pt::Bvh binary;
  binary.nodes = binaryNodes;
  const pt::WideBvh reference = pt::buildWideBvh(binary, instances, width);
  GpuWideCollapseResult gpu = collapser.collapse(
      uploader, resident, static_cast<std::uint32_t>(binaryNodes.size() / 4u), instances, width,
      statistics.topDepth + statistics.bottomDepth);
  const std::string label = "GPU BVH" + std::to_string(width) + " collapse of the " + what;
  if (gpu.nodeCount != reference.nodes.size())
    throw Error(label + " published " + std::to_string(gpu.nodeCount) + " nodes, the host " +
                std::to_string(reference.nodes.size()));
  const std::vector<std::uint8_t> bytes = uploader.readBuffer(gpu.nodes, gpu.nodes.size);
  for (std::size_t n = 0; n < reference.nodes.size(); ++n) {
    std::uint32_t words[40]{}, expected[40]{};
    std::memcpy(words, bytes.data() + n * sizeof(pt::QuantizedWideNode), sizeof(words));
    std::memcpy(expected, &reference.nodes[n], sizeof(expected));
    for (std::uint32_t w = 0; w < 40u; ++w)
      if (words[w] != expected[w]) {
        char why[160];
        std::snprintf(why, sizeof(why), " differs from the host at node %zu word %u: %08x, host %08x", n, w,
                      words[w], expected[w]);
        throw Error(label + why);
      }
  }
  for (std::size_t i = 0; i < instances.size(); ++i)
    if (gpu.instances[i].blasRoot != reference.instances[i].blasRoot)
      throw Error(label + " numbered bottom level " + std::to_string(i) + "'s root differently");
  if (gpu.maximumStack != reference.maximumStack)
    throw Error(label + " bounds the traversal stack by " + std::to_string(gpu.maximumStack) + ", the host by " +
                std::to_string(reference.maximumStack));
  std::printf("PASS: %s publishes buildWideBvh's nodes byte for byte (%u nodes, %u levels, stack %u, "
              "%u dispatches, GPU %.3f ms: gather %.3f, size %.3f, emit %.3f)\n", label.c_str(), gpu.nodeCount,
              gpu.levels, gpu.maximumStack, gpu.dispatches, gpu.gpuMilliseconds, gpu.gatherMilliseconds,
              gpu.sizeMilliseconds, gpu.emitMilliseconds);
  return gpu;
}

// Boxes that stress the collapse's float rule: a plane on y = 0 (exact zeros, flat boxes, equal
// areas), subnormal and signed-zero coordinates, and magnitudes whose areas overflow to
// infinity, over four instances.
void makeCollapseFixture(Fixture &fixture) {
  const std::uint32_t material = addMaterial(fixture, AlphaMode::Opaque, 1.0f);
  std::mt19937 random(4242u);
  std::vector<Vec3> positions;
  std::vector<std::uint32_t> indices;
  auto triangle = [&](Vec3 a, Vec3 b, Vec3 c) {
    const auto base = static_cast<std::uint32_t>(positions.size());
    positions.insert(positions.end(), {a, b, c});
    indices.insert(indices.end(), {base, base + 1u, base + 2u});
  };
  auto flush = [&](const Mat4 &model, const char *name) {
    addMesh(fixture, positions, indices, material, model, name);
    positions.clear();
    indices.clear();
  };
  for (std::uint32_t z = 0; z < 48u; ++z)
    for (std::uint32_t x = 0; x < 48u; ++x) {
      const auto x0 = static_cast<float>(x), z0 = static_cast<float>(z);
      triangle({x0, 0.0f, z0}, {x0 + 1.0f, 0.0f, z0}, {x0, 0.0f, z0 + 1.0f});
    }
  flush(Mat4{}, "collapse plane");
  std::uniform_int_distribution<std::uint32_t> fraction(0u, 0x7FFFFFu);
  auto tiny = [&] {
    std::uint32_t bits = fraction(random) >> (random() % 24u);
    if ((random() & 1u) != 0u) bits |= 0x80000000u;
    return std::bit_cast<float>(bits);
  };
  for (std::uint32_t t = 0; t < 2000u; ++t)
    triangle({tiny(), tiny(), tiny()}, {tiny(), tiny(), tiny()}, {tiny(), tiny(), tiny()});
  flush(Mat4{}, "collapse subnormals");
  std::uniform_real_distribution<float> mantissa(1.0f, 2.0f);
  auto spread = [&] {
    const float value = std::ldexp(mantissa(random), static_cast<int>(random() % 94u) - 30);
    return (random() & 1u) != 0u ? -value : value;
  };
  for (std::uint32_t t = 0; t < 1500u; ++t)
    triangle({spread(), spread(), spread()}, {spread(), spread(), spread()}, {spread(), spread(), spread()});
  const std::vector<Vec3> spreadPositions = positions;
  const std::vector<std::uint32_t> spreadIndices = indices;
  flush(Mat4{}, "collapse magnitudes");
  positions = spreadPositions;
  indices = spreadIndices;
  flush(transform(0.7f, {0.5f, 0.5f, 0.5f}, {5.0f, 0.0f, 0.0f}), "collapse magnitudes, moved");
}

void uploadFixture(Fixture &fixture, Uploader &uploader, const Context &context, const char *name) {
  fixture.scene.vertexCount = static_cast<std::uint32_t>(fixture.vertices.size());
  fixture.scene.indexCount = static_cast<std::uint32_t>(fixture.indices.size());
  fixture.scene.triangleCount = fixture.scene.indexCount / 3;
  fixture.scene.vertexBuffer = uploader.createBuffer(fixture.vertices.data(), fixture.vertices.size() * sizeof(Vertex),
      geometryBufferUsage(context, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT), std::string(name) + ".vertices");
  fixture.scene.indexBuffer = uploader.createBuffer(fixture.indices.data(),
      fixture.indices.size() * sizeof(std::uint32_t),
      geometryBufferUsage(context, VK_BUFFER_USAGE_INDEX_BUFFER_BIT), std::string(name) + ".indices");
}

pt::BvhStatistics buildOnCpu(const Fixture &fixture, TraceScene &trace, pt::Bvh &bvh) {
  std::vector<float> vertices(fixture.vertices.size() * pt::kVertexFloats);
  std::memcpy(vertices.data(), fixture.vertices.data(), fixture.vertices.size() * sizeof(Vertex));
  std::vector<std::uint32_t> triangleCounts;
  triangleCounts.reserve(trace.primitives.size());
  for (const std::uint32_t primitive : trace.primitives)
    triangleCounts.push_back(fixture.scene.primitives[primitive].indexCount / 3);
  return pt::buildBvh(vertices, fixture.indices, trace.instances, triangleCounts, bvh,
                      std::max(1u, std::thread::hardware_concurrency()));
}

// A GPU build's binary nodes, as published.
std::vector<pt::float4> downloadBinaryNodes(Uploader &uploader, const GpuBvhBuildResult &build) {
  std::vector<pt::float4> nodes(static_cast<std::size_t>(build.status.nodes) * 4u);
  const std::vector<std::uint8_t> bytes = uploader.readBuffer(build.nodes, nodes.size() * sizeof(pt::float4));
  std::memcpy(nodes.data(), bytes.data(), bytes.size());
  return nodes;
}

std::vector<std::uint8_t> downloadAll(Uploader &uploader, const Buffer &buffer) {
  return uploader.readBuffer(buffer, buffer.size);
}

// Three meshes over four instances, for the refit test: a plane, a sphere-like soup and a
// strip, two instances of the soup.
void makeRefitFixture(Fixture &fixture) {
  const std::uint32_t material = addMaterial(fixture, AlphaMode::Opaque, 1.0f);
  std::mt19937 random(777u);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  std::vector<Vec3> positions;
  std::vector<std::uint32_t> indices;
  auto triangle = [&](Vec3 a, Vec3 b, Vec3 c) {
    const auto base = static_cast<std::uint32_t>(positions.size());
    positions.insert(positions.end(), {a, b, c});
    indices.insert(indices.end(), {base, base + 1u, base + 2u});
  };
  auto flush = [&](const Mat4 &model, const char *name) {
    addMesh(fixture, positions, indices, material, model, name);
  };
  for (std::uint32_t z = 0; z < 32u; ++z)
    for (std::uint32_t x = 0; x < 32u; ++x) {
      const float x0 = static_cast<float>(x) * 0.25f - 4.0f, z0 = static_cast<float>(z) * 0.25f - 4.0f;
      triangle({x0, -1.0f, z0}, {x0 + 0.25f, -1.0f, z0}, {x0, -1.0f, z0 + 0.25f});
      triangle({x0 + 0.25f, -1.0f, z0}, {x0 + 0.25f, -1.0f, z0 + 0.25f}, {x0, -1.0f, z0 + 0.25f});
    }
  flush(Mat4{}, "refit plane");
  positions.clear();
  indices.clear();
  for (std::uint32_t t = 0; t < 3000u; ++t) {
    const Vec3 centre{unit(random), unit(random), unit(random)};
    const Vec3 a{centre.x + 0.05f * unit(random), centre.y + 0.05f * unit(random), centre.z + 0.05f * unit(random)};
    const Vec3 b{centre.x + 0.05f * unit(random), centre.y + 0.05f * unit(random), centre.z + 0.05f * unit(random)};
    triangle(centre, a, b);
  }
  flush(transform(0.3f, {1.0f, 1.0f, 1.0f}, {-1.5f, 0.0f, 0.0f}), "refit soup");
  flush(transform(-0.8f, {0.7f, 1.2f, 0.7f}, {1.5f, 0.2f, 0.5f}), "refit soup, second");
  positions.clear();
  indices.clear();
  for (std::uint32_t t = 0; t < 400u; ++t) {
    const float x = static_cast<float>(t) * 0.02f - 4.0f;
    triangle({x, 0.0f, -2.0f}, {x + 0.02f, 0.5f, -2.0f}, {x, 1.0f, -2.1f});
  }
  flush(Mat4{}, "refit strip");
}

// The parallel LBVH's updates. A refit of unchanged geometry republishes the build byte for
// byte; a TLAS rebuild after instances move is a fresh build of the moved scene byte for byte;
// after the vertices warp too, the refit tree finds the same closest hits as a fresh build
// (host traversal of both); a refit and a TLAS rebuild back restore the first build exactly.
const char *topologyName(GpuBvhTopology topology) {
  return topology == GpuBvhTopology::Ploc           ? "PLOC" :
         topology == GpuBvhTopology::PlocPlusPlus   ? "PLOC++" :
         topology == GpuBvhTopology::Hploc          ? "H-PLOC" :
         topology == GpuBvhTopology::SinglePassLbvh ? "single-pass LBVH" :
         topology == GpuBvhTopology::BatchedLbvh    ? "batched LBVH" :
         topology == GpuBvhTopology::BinnedSah      ? "binned SAH" :
                                                      "LBVH";
}

void runRefitTest(const Context &context, Uploader &uploader, GpuLbvhBuilder &builder, GpuBvhTopology topology) {
  GpuLbvhBuilder reference(context);  // fresh builds, leaving `builder`'s kept scratch alone
  Fixture fixture;
  makeRefitFixture(fixture);
  uploadFixture(fixture, uploader, context, "refit");
  const std::vector<std::uint32_t> slots(fixture.scene.materials.size(), 0u);
  const TraceScene trace = buildTraceScene(fixture.scene, slots);
  GpuBvhBuildResult built = builder.build(uploader, fixture.scene, trace, true, topology);
  std::vector<std::uint8_t> nodes = downloadAll(uploader, built.nodes);
  const std::vector<std::uint8_t> triangles = downloadAll(uploader, built.triangles);
  GpuBvhUpdateResult same = builder.refit(uploader, fixture.scene, trace, built.nodes, built.triangles);
  if (downloadAll(uploader, built.nodes) != nodes || downloadAll(uploader, built.triangles) != triangles)
    throw Error("a GPU LBVH refit of unchanged geometry is not the build byte for byte");
  // What a TLAS rebuild publishes: a fresh build of the scene, except that the top level is
  // always an LBVH over the instances (a fresh LBVH build's top-level nodes, which come first).
  auto afterTopRebuild = [&](const Fixture &scene, const TraceScene &sceneTrace, GpuBvhBuildResult &fresh) {
    fresh = reference.build(uploader, scene.scene, sceneTrace, false, topology);
    std::vector<std::uint8_t> expected = downloadAll(uploader, fresh.nodes);
    if (topology != GpuBvhTopology::Lbvh) {
      const GpuBvhBuildResult lbvh = reference.build(uploader, scene.scene, sceneTrace);
      const std::vector<std::uint8_t> top = downloadAll(uploader, lbvh.nodes);
      const std::size_t topBytes = std::size_t(lbvh.statistics.topNodes) * 4u * sizeof(pt::float4);
      std::memcpy(expected.data(), top.data(), topBytes);
    }
    return expected;
  };
  if (topology != GpuBvhTopology::Lbvh) {
    // The first scene with its top level rebuilt: the state the updates below return to.
    builder.rebuildTopLevel(uploader, fixture.scene, trace, built.nodes, built.triangles);
    GpuBvhBuildResult fresh;
    nodes = afterTopRebuild(fixture, trace, fresh);
    if (downloadAll(uploader, built.nodes) != nodes)
      throw Error("a GPU TLAS rebuild of an unmoved PLOC build is not its bottom levels under an LBVH top level");
  }

  // Move instances only: the TLAS rebuild is what a fresh build of the moved scene publishes.
  Fixture turned;
  makeRefitFixture(turned);
  turned.scene.primitives[1].transform = transform(1.4f, {1.0f, 1.0f, 1.0f}, {-2.5f, 0.3f, 1.0f});
  turned.scene.primitives[2].transform = transform(0.4f, {0.7f, 1.2f, 0.7f}, {3.0f, 0.2f, -1.5f});
  turned.scene.primitives[3].transform = transform(0.0f, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 4.0f});
  uploadFixture(turned, uploader, context, "refit.turned");
  const TraceScene turnedTrace = buildTraceScene(turned.scene, slots);
  const GpuBvhUpdateResult top = builder.rebuildTopLevel(uploader, turned.scene, turnedTrace, built.nodes,
                                                          built.triangles);
  {
    GpuBvhBuildResult freshTurned;
    if (downloadAll(uploader, built.nodes) != afterTopRebuild(turned, turnedTrace, freshTurned) ||
        downloadAll(uploader, built.triangles) != downloadAll(uploader, freshTurned.triangles))
      throw Error("a GPU LBVH TLAS rebuild is not a fresh build of the moved scene byte for byte");
    for (std::size_t i = 0; i < top.instances.size(); ++i)
      if (std::memcmp(&top.instances[i], &freshTurned.instances[i], sizeof(pt::TraceInstance)) != 0)
        throw Error("a GPU LBVH TLAS rebuild published different instance rows");
  }

  // Warp the vertices (a per-vertex displacement, so boxes change unevenly) and move instances.
  Fixture moved;
  makeRefitFixture(moved);
  for (Vertex &vertex : moved.vertices) {
    Vec3 &p = vertex.position;
    p = {p.x + 0.15f * std::sin(3.0f * p.y), p.y * 1.3f + 0.1f * std::cos(2.0f * p.x), p.z - 0.2f * p.x};
  }
  moved.scene.primitives[1].transform = transform(0.9f, {1.0f, 1.0f, 1.0f}, {-1.0f, 0.4f, 0.3f});
  moved.scene.primitives[2].transform = transform(-0.2f, {0.9f, 0.9f, 1.3f}, {2.0f, -0.3f, 0.0f});
  uploadFixture(moved, uploader, context, "refit.moved");
  const TraceScene movedTrace = buildTraceScene(moved.scene, slots);
  const GpuBvhUpdateResult refitted = builder.refit(uploader, moved.scene, movedTrace, built.nodes, built.triangles);
  std::vector<pt::float4> refitNodes(built.nodes.size / sizeof(pt::float4)), refitTriangles(built.triangles.size / sizeof(pt::float4));
  std::memcpy(refitNodes.data(), downloadAll(uploader, built.nodes).data(), built.nodes.size);
  std::memcpy(refitTriangles.data(), downloadAll(uploader, built.triangles).data(), built.triangles.size);
  const std::vector<pt::TraceInstance> refitInstances = refitted.instances;

  // Back to the first geometry and transforms: a refit (the TLAS keeps the turned topology) and a
  // TLAS rebuild restore the first build.
  builder.refit(uploader, fixture.scene, trace, built.nodes, built.triangles);
  builder.rebuildTopLevel(uploader, fixture.scene, trace, built.nodes, built.triangles);
  if (downloadAll(uploader, built.nodes) != nodes || downloadAll(uploader, built.triangles) != triangles)
    throw Error("a GPU LBVH refit and TLAS rebuild back to the first scene are not the first build byte for byte");

  GpuBvhBuildResult fresh = reference.build(uploader, moved.scene, movedTrace, false, topology);
  std::vector<pt::float4> freshNodes(fresh.nodes.size / sizeof(pt::float4)), freshTriangles(fresh.triangles.size / sizeof(pt::float4));
  std::memcpy(freshNodes.data(), downloadAll(uploader, fresh.nodes).data(), fresh.nodes.size);
  std::memcpy(freshTriangles.data(), downloadAll(uploader, fresh.triangles).data(), fresh.triangles.size);

  std::vector<pt::Material> materials;
  for (const Material &material : moved.scene.materials) materials.push_back(convertMaterial(material.uniforms));
  const pt::HostTextures textures;
  const auto *vertices = reinterpret_cast<const float *>(moved.vertices.data());
  const std::size_t vertexFloats = moved.vertices.size() * sizeof(moved.vertices[0]) / sizeof(float);
  const HostView refitView(textures, refitInstances, materials, moved.indices, vertices, vertexFloats, refitNodes,
                           refitTriangles);
  const HostView freshView(textures, fresh.instances, materials, moved.indices, vertices, vertexFloats, freshNodes,
                           freshTriangles);
  std::mt19937 random(31337u);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  std::uint32_t hits = 0;
  for (std::uint32_t r = 0; r < 20000u; ++r) {
    const pt::float3 origin(unit(random) * 5.0f, unit(random) * 3.0f + 1.0f, unit(random) * 5.0f);
    const pt::float3 target(unit(random) * 2.5f, unit(random) * 1.5f, unit(random) * 2.5f);
    const pt::float3 direction = pt::normalize(target - origin);
    const pt::PtHit a = pt::ptTraceBvh(refitView, origin, direction, 1e30f, 0xFFu, 0u, pt::float2(-1.0f, 0.0f), 0u);
    const pt::PtHit b = pt::ptTraceBvh(freshView, origin, direction, 1e30f, 0xFFu, 0u, pt::float2(-1.0f, 0.0f), 0u);
    // The same closest distance (a different triangle only at an exact tie).
    if (a.found != b.found || (a.found != 0u && a.t != b.t))
      throw Error("the refit GPU LBVH and a fresh build of the moved scene disagree on ray " + std::to_string(r));
    hits += a.found;
  }
  std::printf("PASS: %s: GPU refit republishes an unchanged tree byte for byte, matches a fresh build's hits on "
              "moved geometry (%u of 20000 rays hit) and refits back exactly (GPU %.3f ms: extents %.3f, fit %.3f, "
              "emit %.3f; %u dispatches)\n", topologyName(topology), hits, refitted.gpuMilliseconds, refitted.stageMilliseconds[0],
              refitted.stageMilliseconds[4], refitted.stageMilliseconds[5], refitted.dispatches);
  // In-frame updates: the same work recorded into a command buffer, the rows read from a device
  // buffer, publishes what the synchronous updates do.
  auto recorded = [&](const Fixture &scene, const TraceScene &sceneTrace, bool top) {
    std::vector<pt::TraceInstance> rows = builder.updateRows(scene.scene, sceneTrace);
    Buffer rowBuffer = uploader.createBuffer(rows.data(), rows.size() * sizeof(pt::TraceInstance),
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                             "refit.frame-rows");
    uploader.runImmediate([&](VkCommandBuffer command) {
      if (top) builder.recordRebuildTopLevel(command, scene.scene, rowBuffer, built.nodes, built.triangles, 1u);
      else builder.recordRefit(command, scene.scene, rowBuffer, built.nodes, built.triangles, 1u);
    });
    const std::string error = builder.frameError(1u);
    if (!error.empty()) throw Error("an in-frame update reported: " + error);
  };
  builder.refit(uploader, moved.scene, movedTrace, built.nodes, built.triangles);
  const std::vector<std::uint8_t> movedNodes = downloadAll(uploader, built.nodes);
  const std::vector<std::uint8_t> movedTriangles = downloadAll(uploader, built.triangles);
  builder.refit(uploader, fixture.scene, trace, built.nodes, built.triangles);
  recorded(moved, movedTrace, false);
  if (downloadAll(uploader, built.nodes) != movedNodes || downloadAll(uploader, built.triangles) != movedTriangles)
    throw Error("an in-frame GPU LBVH refit differs from the synchronous refit");
  recorded(fixture, trace, false);
  if (downloadAll(uploader, built.nodes) != nodes || downloadAll(uploader, built.triangles) != triangles)
    throw Error("an in-frame GPU LBVH refit back is not the first build byte for byte");
  recorded(turned, turnedTrace, true);
  {
    GpuBvhBuildResult freshTurned;
    if (downloadAll(uploader, built.nodes) != afterTopRebuild(turned, turnedTrace, freshTurned))
      throw Error("an in-frame GPU LBVH TLAS rebuild is not a fresh build of the moved scene byte for byte");
  }
  std::printf("PASS: %s: in-frame GPU refit and TLAS rebuild (recorded, rows from a device buffer, status read "
              "afterwards) publish what the synchronous updates do\n", topologyName(topology));
  std::printf("PASS: %s: GPU TLAS rebuild is a fresh build of the moved scene byte for byte, under an LBVH top level "
              "(GPU %.3f ms: instances %.3f, morton %.3f, sort %.3f, topology %.3f, fit %.3f, number and emit %.3f; "
              "%u dispatches)\n", topologyName(topology), top.gpuMilliseconds, top.stageMilliseconds[0], top.stageMilliseconds[1], top.stageMilliseconds[2],
              top.stageMilliseconds[3], top.stageMilliseconds[4], top.stageMilliseconds[5], top.dispatches);
  (void)same;
}

// A wide refit: the kept collapse re-emitted after the LBVH refit to moved and warped geometry
// finds the same closest hits as the refit binary tree (host traversal of both), and an
// unchanged re-emit republishes the collapse byte for byte.
void runWideRefitTest(const Context &context, Uploader &uploader, GpuLbvhBuilder &builder, GpuBvhCollapser &collapser,
                      std::uint32_t width, GpuBvhTopology topology) {
  Fixture fixture;
  makeRefitFixture(fixture);
  uploadFixture(fixture, uploader, context, "wide-refit");
  const std::vector<std::uint32_t> slots(fixture.scene.materials.size(), 0u);
  const TraceScene trace = buildTraceScene(fixture.scene, slots);
  GpuBvhBuildResult built = builder.build(uploader, fixture.scene, trace, true, topology);
  GpuWideCollapseResult wide = collapser.collapse(uploader, built.nodes, built.status.nodes, built.instances, width,
                                                  built.statistics.topDepth + built.statistics.bottomDepth, true);
  const std::vector<std::uint8_t> first = downloadAll(uploader, wide.nodes);
  collapser.reemit(uploader, built.nodes, wide.nodes);
  if (downloadAll(uploader, wide.nodes) != first)
    throw Error("an unchanged wide re-emit is not the collapse byte for byte");

  // In frame: the collapse recorded with nothing read back publishes the same nodes (in its
  // larger output) and writes the traced rows' wide roots.
  Buffer frameNodes(context, VkDeviceSize(built.status.nodes) * sizeof(pt::QuantizedWideNode),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO, 0,
                    "wide-refit.frame-nodes");
  collapser.prepareFrames(uploader, built.nodes, built.status.nodes, built.instances, width, frameNodes);
  const VkBufferUsageFlags rowUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  Buffer binaryRows = uploader.createBuffer(built.instances.data(), built.instances.size() * sizeof(pt::TraceInstance),
                                            rowUsage, "wide-refit.binary-rows");
  Buffer tracedRows = uploader.createBuffer(built.instances.data(), built.instances.size() * sizeof(pt::TraceInstance),
                                            rowUsage, "wide-refit.traced-rows");
  auto frameBytes = [&](std::size_t bytes) {
    std::vector<std::uint8_t> all = downloadAll(uploader, frameNodes);
    all.resize(bytes);
    return all;
  };
  uploader.runImmediate([&](VkCommandBuffer command) { collapser.recordCollapse(command, binaryRows, tracedRows, 2u); });
  if (const std::string error = collapser.frameError(2u); !error.empty()) throw Error(error);
  if (frameBytes(first.size()) != first)
    throw Error("an in-frame wide collapse differs from the synchronous collapse");
  {
    std::vector<pt::TraceInstance> rows(built.instances.size());
    std::memcpy(rows.data(), downloadAll(uploader, tracedRows).data(), rows.size() * sizeof(pt::TraceInstance));
    for (std::size_t i = 0; i < rows.size(); ++i)
      if (rows[i].blasRoot != wide.instances[i].blasRoot)
        throw Error("an in-frame wide collapse wrote a different bottom-level root");
  }

  Fixture moved;
  makeRefitFixture(moved);
  for (Vertex &vertex : moved.vertices) {
    Vec3 &p = vertex.position;
    p = {p.x + 0.15f * std::sin(3.0f * p.y), p.y * 1.3f + 0.1f * std::cos(2.0f * p.x), p.z - 0.2f * p.x};
  }
  moved.scene.primitives[1].transform = transform(0.9f, {1.0f, 1.0f, 1.0f}, {-1.0f, 0.4f, 0.3f});
  uploadFixture(moved, uploader, context, "wide-refit.moved");
  const TraceScene movedTrace = buildTraceScene(moved.scene, slots);
  const GpuBvhUpdateResult refitted = builder.refit(uploader, moved.scene, movedTrace, built.nodes, built.triangles);
  const GpuWideCollapseResult reemitted = collapser.reemit(uploader, built.nodes, wide.nodes);
  uploader.runImmediate([&](VkCommandBuffer command) { collapser.recordReemit(command, binaryRows, tracedRows, 3u); });
  if (const std::string error = collapser.frameError(3u); !error.empty()) throw Error(error);
  if (frameBytes(first.size()) != downloadAll(uploader, wide.nodes))
    throw Error("an in-frame wide re-emit differs from the synchronous re-emit");

  std::vector<pt::float4> binaryNodes(built.nodes.size / sizeof(pt::float4)), triangles(built.triangles.size / sizeof(pt::float4));
  std::memcpy(binaryNodes.data(), downloadAll(uploader, built.nodes).data(), built.nodes.size);
  std::memcpy(triangles.data(), downloadAll(uploader, built.triangles).data(), built.triangles.size);
  pt::WideBvh wideTree;
  wideTree.width = width;
  wideTree.nodes.resize(wide.nodes.size / sizeof(pt::QuantizedWideNode));
  std::memcpy(wideTree.nodes.data(), downloadAll(uploader, wide.nodes).data(), wide.nodes.size);
  wideTree.triangles = triangles;
  wideTree.instances = refitted.instances;
  for (std::size_t i = 0; i < wideTree.instances.size(); ++i) wideTree.instances[i].blasRoot = wide.instances[i].blasRoot;

  std::vector<pt::Material> materials;
  for (const Material &material : moved.scene.materials) materials.push_back(convertMaterial(material.uniforms));
  const pt::HostTextures textures;
  const auto *vertices = reinterpret_cast<const float *>(moved.vertices.data());
  const HostView binaryView(textures, refitted.instances, materials, moved.indices, vertices,
                            moved.vertices.size() * sizeof(moved.vertices[0]) / sizeof(float), binaryNodes, triangles);
  std::mt19937 random(4711u);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  std::uint32_t hits = 0;
  for (std::uint32_t r = 0; r < 20000u; ++r) {
    const pt::float3 origin(unit(random) * 5.0f, unit(random) * 3.0f + 1.0f, unit(random) * 5.0f);
    const pt::float3 target(unit(random) * 2.5f, unit(random) * 1.5f, unit(random) * 2.5f);
    const pt::float3 direction = pt::normalize(target - origin);
    const pt::PtHit a = pt::traceWideBvh(wideTree, binaryView, origin, direction, 1e30f, 0xFFu, 0u, pt::float2(-1.0f, 0.0f), 0u);
    const pt::PtHit b = pt::ptTraceBvh(binaryView, origin, direction, 1e30f, 0xFFu, 0u, pt::float2(-1.0f, 0.0f), 0u);
    if (a.found != b.found || (a.found != 0u && a.t != b.t))
      throw Error("the re-emitted BVH" + std::to_string(width) + " and the refit binary tree disagree on ray " +
                  std::to_string(r));
    hits += a.found;
  }
  std::printf("PASS: BVH%u in-frame collapse and re-emit (recorded, nothing read back) publish the synchronous "
              "ones' nodes and bottom-level roots\n", width);
  std::printf("PASS: BVH%u re-emit after a GPU LBVH refit matches the refit tree's hits (%u of 20000 rays hit; "
              "GPU %.3f ms, %u dispatches)\n", width, hits, reemitted.gpuMilliseconds, reemitted.dispatches);
}

int run(bool software, bool gpuBuilder, bool lbvh, GpuBvhTopology lbvhTopology, bool ploc, GpuBvhTopology plocTopology,
        std::uint32_t wide, bool wavefront, bool rayPipeline, float clip, const std::string &traversal) {
  Window window("Basalt GPU ray oracle", 64, 64, false);
  Context context(window, true);
  if (rayPipeline && !context.rayPipelineSupported) {
    std::printf("SKIP: Vulkan ray pipelines are unavailable on %s\n", context.info.name.c_str());
    return 77;
  }
  if (!software && !rayPipeline && !context.rayQuerySupported) {
    std::printf("SKIP: Vulkan ray queries are unavailable on %s\n", context.info.name.c_str());
    return 77;
  }
  if (!context.validationEnabled) {
    std::fprintf(stderr, "the Khronos validation layer is required for the GPU ray oracle\n");
    return 1;
  }

  int result = 0;
  {
    Uploader uploader(context);
    std::unique_ptr<GpuLbvhBuilder> lbvhBuilder;
    if (lbvh || ploc) lbvhBuilder = std::make_unique<GpuLbvhBuilder>(context);
    const GpuBvhTopology topology = ploc ? plocTopology : lbvhTopology;
    const bool plocFused = topology == GpuBvhTopology::PlocPlusPlus;
    const bool sah = topology == GpuBvhTopology::BinnedSah;
    // The builder under test; with --gpu-lbvh, --gpu-apetrei (the single-pass LBVH) and
    // --gpu-batched every build is also the serial builder's tree.
    // With --gpu-ploc every build is well formed, the same bytes when repeated, and publishes
    // the serial builder's triangles (the same Morton order) and instance rows; with --gpu-plocpp
    // (PLOC++'s fused iterations) it is also the PLOC build byte for byte; --gpu-hploc (H-PLOC)
    // takes PLOC's checks. With --gpu-sah (the binned SAH) every build is repeatable and the CPU
    // builder's over the same geometry (read back, subnormals flushed where the device flushes
    // them), byte for byte. With --split-clipping, every build (and the tree it is compared with)
    // is over split clipping's references: the host's, which the GPU's must equal byte for byte
    // (where the device has 64-bit floats); the GPU builds take the GPU's.
    const bool clipping = clip > 0.0f;
    if (clipping) std::printf("Split clipping (factor %g) for every build\n", clip);
    std::unique_ptr<GpuLbvhBuilder> clipper;  // the GPU clipping, when no builder under test has it
    auto buildOnGpu = [&](const Scene &scene, const TraceScene &trace, const char *what) {
      std::optional<pt::BvhReferences> clippedReferences;
      std::optional<GpuBvhReferences> gpuClipped;
      if (clipping) {
        const HostGeometry geometry = readGeometry(uploader, scene, trace);
        clippedReferences = pt::clipReferences(geometry.vertices, geometry.indices, trace.instances,
                                               geometry.triangleCounts, clip, std::thread::hardware_concurrency());
        if (!lbvhBuilder && !clipper) clipper = std::make_unique<GpuLbvhBuilder>(context);
        GpuLbvhBuilder &gpu = lbvhBuilder ? *lbvhBuilder : *clipper;
        if (gpu.clipAvailable()) {
          gpuClipped = gpu.clipReferences(uploader, scene, trace, clip);
          const std::vector<pt::BvhReference> &host = clippedReferences->references;
          const std::vector<std::uint8_t> bytes =
              uploader.readBuffer(gpuClipped->references, std::max<std::size_t>(1, host.size()) * sizeof(pt::BvhReference));
          if (gpuClipped->counts != clippedReferences->counts ||
              (!host.empty() && std::memcmp(bytes.data(), host.data(), host.size() * sizeof(pt::BvhReference)) != 0)) {
            std::size_t first = 0;
            while (first < host.size() && std::memcmp(bytes.data() + first * sizeof(pt::BvhReference), &host[first],
                                                      sizeof(pt::BvhReference)) == 0)
              ++first;
            char detail[512] = "";
            for (std::size_t i = 0; i < host.size() && i < gpuClipped->counts.size(); ++i)
              if (i < clippedReferences->counts.size() && gpuClipped->counts[i] != clippedReferences->counts[i]) {
                std::snprintf(detail, sizeof(detail), "; instance %zu: %u references, the host %u", i,
                              gpuClipped->counts[i], clippedReferences->counts[i]);
                break;
              }
            if (!detail[0] && first < host.size()) {
              pt::BvhReference gpu{};
              std::memcpy(&gpu, bytes.data() + first * sizeof(pt::BvhReference), sizeof(gpu));
              const pt::BvhReference &cpu = host[first];
              std::snprintf(detail, sizeof(detail), "; GPU %a %a %a .. %a %a %a, the host %a %a %a .. %a %a %a", gpu.low.x,
                            gpu.low.y, gpu.low.z, gpu.high.x, gpu.high.y, gpu.high.z, cpu.low.x, cpu.low.y, cpu.low.z,
                            cpu.high.x, cpu.high.y, cpu.high.z);
            }
            throw Error(std::string("the GPU split clipping differs from the host's on the ") + what + " (first at reference " +
                        std::to_string(first) + " of " + std::to_string(host.size()) + detail + ")");
          }
          std::printf("PASS: the GPU split clipping makes the host's %zu references byte for byte on the %s "
                      "(GPU %.3f ms, host %.1f ms)\n", host.size(), what, gpuClipped->gpuMilliseconds,
                      clippedReferences->milliseconds);
        } else {
          gpuClipped = uploadReferences(uploader, *clippedReferences);
        }
      }
      const pt::BvhReferences *hostReferences = clippedReferences ? &*clippedReferences : nullptr;
      const GpuBvhReferences *references = gpuClipped ? &*gpuClipped : nullptr;
      if (sah) {
        GpuBvhBuildResult built = lbvhBuilder->build(uploader, scene, trace, false, topology, references);
        const GpuBvhBuildResult again = lbvhBuilder->build(uploader, scene, trace, false, topology, references);
        // Only the published nodes: the buffer holds as many as an LBVH, more than a SAH tree's.
        const VkDeviceSize published = VkDeviceSize(built.status.nodes) * 4u * sizeof(pt::float4);
        if (again.status.nodes != built.status.nodes ||
            uploader.readBuffer(again.nodes, published) != uploader.readBuffer(built.nodes, published))
          throw Error(std::string("two binned SAH builds of the ") + what + " publish different nodes");
        DownloadedTree tree = downloadTree(uploader, built);
        tree.nodes.resize(std::size_t(built.status.nodes) * 4u);
        const std::string malformed = checkTree(tree, clipping);
        if (!malformed.empty()) throw Error(std::string("binned SAH tree is malformed on the ") + what + ": " + malformed);
        const HostGeometry geometry = readGeometry(uploader, scene, trace);
        // Devices may flush subnormals to signed zero in the builders' arithmetic (Vulkan's
        // default float controls): the CPU builder's tree over the vertices as read, or else as
        // flushed (its triangles then compared flushed too). "" when the GPU build is that tree.
        auto flush = [](float v) { return std::fpclassify(v) == FP_SUBNORMAL ? std::copysign(0.0f, v) : v; };
        pt::BvhStatistics cpuStatistics{};
        auto difference = [&](bool flushed) -> std::string {
          std::vector<float> input = geometry.vertices;
          if (flushed) std::transform(input.begin(), input.end(), input.begin(), flush);
          // The references' boxes, flushed as the device reads them.
          std::optional<pt::BvhReferences> flushedReferences;
          if (flushed && hostReferences) {
            flushedReferences = *hostReferences;
            for (pt::BvhReference &reference : flushedReferences->references) {
              reference.low = pt::float4(flush(reference.low.x), flush(reference.low.y), flush(reference.low.z), reference.low.w);
              reference.high = pt::float4(flush(reference.high.x), flush(reference.high.y), flush(reference.high.z), 0.0f);
            }
          }
          std::vector<pt::TraceInstance> instances = trace.instances;
          pt::Bvh cpu;
          cpuStatistics = pt::buildBvh(input, geometry.indices, instances, geometry.triangleCounts, cpu, 1u,
                                       flushedReferences ? &*flushedReferences : hostReferences);
          if (std::size_t(built.status.nodes) * 4u != cpu.nodes.size())
            return std::to_string(built.status.nodes) + " nodes, the CPU builder " + std::to_string(cpu.nodes.size() / 4u);
          // Flushed, box coordinates compare as flushed: the device flushes in arithmetic (the
          // triangles' boxes, the instances' transforms) but not where it copies or bounds by
          // integer ordering (split clipping's boxes), and its zeros may take either sign. The
          // data and count words are exact.
          const auto *a = reinterpret_cast<const std::uint32_t *>(tree.nodes.data());
          const auto *b = reinterpret_cast<const std::uint32_t *>(cpu.nodes.data());
          for (std::size_t w = 0; w < cpu.nodes.size() * 4u; ++w) {
            const bool box = w % 4u != 3u;
            const bool same = flushed && box ? flush(std::bit_cast<float>(a[w])) == flush(std::bit_cast<float>(b[w]))
                                             : a[w] == b[w];
            if (!same)
              return "node word " + std::to_string(w) + " (node " + std::to_string(w / 16u) + ") " + std::to_string(a[w]) +
                     ", the CPU builder's " + std::to_string(b[w]);
          }
          if (tree.triangles.size() != cpu.triangles.size()) return "a different triangle count";
          for (std::size_t i = 0; i < cpu.triangles.size(); ++i) {
            const pt::float4 &x = tree.triangles[i], &y = cpu.triangles[i];
            const bool same = flushed ? flush(x.x) == y.x && flush(x.y) == y.y && flush(x.z) == y.z &&
                                            std::bit_cast<std::uint32_t>(x.w) == std::bit_cast<std::uint32_t>(y.w)
                                      : std::memcmp(&x, &y, sizeof(x)) == 0;
            if (!same) return "triangle word " + std::to_string(i);
          }
          for (std::size_t i = 0; i < instances.size(); ++i)
            if (std::memcmp(&tree.instances[i], &instances[i], sizeof(pt::TraceInstance)) != 0)
              return "instance row " + std::to_string(i);
          if (built.statistics.topDepth != cpuStatistics.topDepth || built.statistics.bottomDepth != cpuStatistics.bottomDepth)
            return "depth";
          return {};
        };
        const std::string raw = difference(false);
        const bool flushed = !raw.empty();
        if (flushed) {
          if (const std::string after = difference(true); !after.empty())
            throw Error(std::string("binned SAH differs from the CPU builder on the ") + what + ": " + raw +
                        " (with subnormals flushed: " + after + ")");
        }
        std::printf("PASS: GPU binned SAH publishes the CPU builder's nodes, triangles and rows byte for byte on the %s%s "
                    "(depth %u + %u, SAH %.6g vs CPU %.6g, %u dispatches, GPU %.3f ms)\n", what,
                    flushed ? " (with subnormals flushed)" : "", built.statistics.topDepth,
                    built.statistics.bottomDepth, built.statistics.sahCost, cpuStatistics.sahCost, built.status.dispatches,
                    [&] { double sum = 0.0; for (double v : built.stageMilliseconds) sum += v; return sum; }());
        return built;
      }
      if (ploc) {
        GpuBvhBuildResult built = lbvhBuilder->build(uploader, scene, trace, false, topology, references);
        const GpuBvhBuildResult again = lbvhBuilder->build(uploader, scene, trace, false, topology, references);
        const GpuBvhBuildResult serial = buildGpuBvh(context, uploader, scene, trace, references);
        const DownloadedTree tree = downloadTree(uploader, built), serialTree = downloadTree(uploader, serial);
        const std::string malformed = checkTree(tree, clipping);
        if (!malformed.empty()) throw Error(std::string("PLOC tree is malformed on the ") + what + ": " + malformed);
        if (downloadAll(uploader, again.nodes) != downloadAll(uploader, built.nodes))
          throw Error(std::string("two PLOC builds of the ") + what + " publish different nodes");
        if (tree.triangles.size() != serialTree.triangles.size() ||
            std::memcmp(tree.triangles.data(), serialTree.triangles.data(), tree.triangles.size() * sizeof(pt::float4)) != 0)
          throw Error(std::string("PLOC triangles differ from the serial builder's on the ") + what);
        for (std::size_t i = 0; i < tree.instances.size(); ++i)
          if (std::memcmp(&tree.instances[i], &serialTree.instances[i], sizeof(pt::TraceInstance)) != 0)
            throw Error(std::string("PLOC instance rows differ from the serial builder's on the ") + what);
        if (plocFused) {
          const GpuBvhBuildResult plain = lbvhBuilder->build(uploader, scene, trace, false, GpuBvhTopology::Ploc, references);
          if (downloadAll(uploader, plain.nodes) != downloadAll(uploader, built.nodes))
            throw Error(std::string("PLOC++ nodes are not the PLOC build's byte for byte on the ") + what);
        }
        std::printf("PASS: parallel %s builds a well-formed, repeatable tree on the %s (depth %u + %u, SAH %.4g vs "
                    "LBVH %.4g, %u merge iterations, %u dispatches, GPU %.3f ms)\n", topologyName(topology), what,
                    built.statistics.topDepth,
                    built.statistics.bottomDepth, built.statistics.sahCost, serial.statistics.sahCost,
                    built.status.plocIterations, built.status.dispatches,
                    [&] { double sum = 0.0; for (double v : built.stageMilliseconds) sum += v; return sum; }());
        return built;
      }
      if (!lbvh) return buildGpuBvh(context, uploader, scene, trace, references);
      GpuBvhBuildResult parallel = lbvhBuilder->build(uploader, scene, trace, false, topology, references);
      const GpuBvhBuildResult serial = buildGpuBvh(context, uploader, scene, trace, references);
      const DownloadedTree parallelTree = downloadTree(uploader, parallel), serialTree = downloadTree(uploader, serial);
      const std::string difference = compareTrees(parallelTree, serialTree);
      if (!difference.empty())
        throw Error(std::string(topologyName(topology)) + " differs from the serial builder on the " + what + ": " + difference);
      // The serial numbering too: the published nodes are the serial builder's, byte for byte.
      if (parallelTree.nodes.size() != serialTree.nodes.size() ||
          std::memcmp(parallelTree.nodes.data(), serialTree.nodes.data(), parallelTree.nodes.size() * sizeof(pt::float4)) != 0)
        throw Error(std::string(topologyName(topology)) + " nodes are not byte-identical to the serial builder's on the " + what);
      if (parallel.statistics.topDepth != serial.statistics.topDepth ||
          parallel.statistics.bottomDepth != serial.statistics.bottomDepth ||
          std::abs(parallel.statistics.sahCost - serial.statistics.sahCost) > 1e-9 * serial.statistics.sahCost)
        throw Error(std::string(topologyName(topology)) + " depth or cost differs from the serial builder on the " + what);
      std::printf("PASS: %s publishes the serial builder's nodes byte for byte on the %s (depth %u + %u, %u fit levels, "
                  "%u dispatches, GPU %.3f ms)\n", topologyName(topology), what, parallel.statistics.topDepth, parallel.statistics.bottomDepth,
                  parallel.status.fitIterations, parallel.status.dispatches,
                  [&] { double sum = 0.0; for (double v : parallel.stageMilliseconds) sum += v; return sum; }());
      return parallel;
    };
    if (lbvh && !clipping) runSortTest(uploader, *lbvhBuilder);
    if ((lbvh || ploc) && !sah && !clipping && wide == 0u) runRefitTest(context, uploader, *lbvhBuilder, topology);
    if (sah) {
      bool refused = false;
      try {
        lbvhBuilder->build(uploader, Scene{}, TraceScene{}, true, topology);
      } catch (const std::invalid_argument &) {
        refused = true;
      }
      if (!refused) throw Error("a refittable binned SAH build was not refused");
      std::printf("PASS: a refittable binned SAH build is refused\n");
    }
    std::unique_ptr<GpuBvhCollapser> collapser;
    if (wide != 0u) collapser = std::make_unique<GpuBvhCollapser>(context);
    if ((lbvh || ploc) && !sah && !clipping && wide != 0u) runWideRefitTest(context, uploader, *lbvhBuilder, *collapser, wide, topology);
    // The collapse on the adversarial fixture's tree, built by the builder under test.
    if (wide != 0u) {
      Fixture collapseFixture;
      makeCollapseFixture(collapseFixture);
      uploadFixture(collapseFixture, uploader, context, "collapse");
      TraceScene collapseTrace = buildTraceScene(collapseFixture.scene,
                                                 std::vector<std::uint32_t>(collapseFixture.scene.materials.size(), 0u));
      if (gpuBuilder) {
        const GpuBvhBuildResult built = buildOnGpu(collapseFixture.scene, collapseTrace, "collapse fixture");
        checkCollapse(uploader, *collapser, built.nodes, downloadBinaryNodes(uploader, built), built.instances, wide,
                      "GPU-built collapse fixture", built.statistics);
      } else {
        pt::Bvh collapseBvh;
        const pt::BvhStatistics collapseStatistics = buildOnCpu(collapseFixture, collapseTrace, collapseBvh);
        Buffer resident = uploader.createBuffer(collapseBvh.nodes.data(), collapseBvh.nodes.size() * sizeof(pt::float4),
                                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "collapse.bvh.nodes");
        checkCollapse(uploader, *collapser, resident, collapseBvh.nodes, collapseTrace.instances, wide,
                      "CPU-built collapse fixture", collapseStatistics);
      }
    }
    if (gpuBuilder) {
      Scene emptyScene;
      const Vertex dummyVertex{};
      const std::uint32_t dummyIndex = 0u;
      emptyScene.vertexCount = 1;
      emptyScene.indexCount = 1;
      emptyScene.vertexBuffer = uploader.createBuffer(&dummyVertex, sizeof(dummyVertex),
          geometryBufferUsage(context, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT), "empty.vertices");
      emptyScene.indexBuffer = uploader.createBuffer(&dummyIndex, sizeof(dummyIndex),
          geometryBufferUsage(context, VK_BUFFER_USAGE_INDEX_BUFFER_BIT), "empty.indices");
      const TraceScene emptyTrace;
      GpuBvhBuildResult empty = buildOnGpu(emptyScene, emptyTrace, "empty scene");
      if (empty.statistics.instances != 0 || empty.statistics.triangles != 0 ||
          empty.statistics.topNodes != 1 || empty.statistics.topDepth != 1)
        throw Error("GPU LBVH empty-scene publication contract failed");
      const auto bytes = uploader.readBuffer(empty.nodes, sizeof(pt::float4) * 4u);
      pt::float4 root[4]{};
      std::memcpy(root, bytes.data(), bytes.size());
      if (pt::as_type<pt::uint>(root[0].w) != 0xFFFFFFFFu ||
          pt::as_type<pt::uint>(root[2].w) != 0xFFFFFFFFu)
        throw Error("GPU LBVH empty-scene root was not explicitly empty");
      std::printf("PASS: GPU LBVH empty-scene publication\n");

      Fixture stress;
      const std::uint32_t stressMaterial = addMaterial(stress, AlphaMode::Opaque, 1.0f);
      constexpr std::uint32_t stressTriangles = 32u * 1024u;
      std::vector<Vec3> stressPositions;
      std::vector<std::uint32_t> stressIndices;
      stressPositions.reserve(stressTriangles * 3u);
      stressIndices.reserve(stressTriangles * 3u);
      for (std::uint32_t triangle = 0; triangle < stressTriangles; ++triangle) {
        const float x = static_cast<float>(triangle & 255u) * 0.01f;
        const float y = static_cast<float>(triangle >> 8u) * 0.01f;
        const float z = static_cast<float>((triangle * 17u) & 63u) * 0.001f;
        const std::uint32_t base = static_cast<std::uint32_t>(stressPositions.size());
        stressPositions.insert(stressPositions.end(), {{x, y, z}, {x + 0.004f, y, z},
                                                       {x, y + 0.004f, z + 0.0001f}});
        stressIndices.insert(stressIndices.end(), {base, base + 1u, base + 2u});
      }
      addMesh(stress, stressPositions, stressIndices, stressMaterial, Mat4{}, "LBVH stress");
      stress.scene.vertexCount = static_cast<std::uint32_t>(stress.vertices.size());
      stress.scene.indexCount = static_cast<std::uint32_t>(stress.indices.size());
      stress.scene.vertexBuffer = uploader.createBuffer(stress.vertices.data(), stress.vertices.size() * sizeof(Vertex),
          geometryBufferUsage(context, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT), "stress.vertices");
      stress.scene.indexBuffer = uploader.createBuffer(stress.indices.data(),
          stress.indices.size() * sizeof(std::uint32_t),
          geometryBufferUsage(context, VK_BUFFER_USAGE_INDEX_BUFFER_BIT), "stress.indices");
      const TraceScene stressTrace = buildTraceScene(stress.scene, std::vector<std::uint32_t>(1, 0u));
      const GpuBvhBuildResult stressBuild = buildOnGpu(stress.scene, stressTrace, "stress fixture");
      if (wide != 0u)
        checkCollapse(uploader, *collapser, stressBuild.nodes, downloadBinaryNodes(uploader, stressBuild),
                      stressBuild.instances, wide, "GPU-built stress fixture", stressBuild.statistics);
      if ((clipping ? stressBuild.statistics.triangles < stressTriangles
                    : stressBuild.statistics.triangles != stressTriangles) ||
          stressBuild.statistics.topDepth + stressBuild.statistics.bottomDepth > pt::kBvhStack)
        throw Error("GPU LBVH stress fixture failed its count or depth contract");
      std::printf("PASS: GPU LBVH %u-triangle stress fixture (depth %u + %u)\n", stressTriangles,
                  stressBuild.statistics.topDepth, stressBuild.statistics.bottomDepth);

      // Bottom levels around the batched build's one-workgroup limit (256 records), with runs of
      // duplicate centroids: every size the batched sort and climb take, and the neighbours
      // that go through the global kernels, under the same checks as the other fixtures.
      Fixture small;
      const std::uint32_t smallMaterial = addMaterial(small, AlphaMode::Opaque, 1.0f);
      std::mt19937 smallRandom(41);
      std::uniform_real_distribution<float> smallUnit(-1.0f, 1.0f);
      const std::uint32_t smallSizes[] = {1u, 2u, 3u, 7u, 64u, 200u, 255u, 256u, 257u};
      for (std::size_t level = 0; level < std::size(smallSizes); ++level) {
        std::vector<Vec3> positions;
        std::vector<std::uint32_t> indices;
        for (std::uint32_t triangle = 0; triangle < smallSizes[level]; ++triangle) {
          // Every fifth triangle is centred at the origin: equal Morton codes, kept in primitive order.
          const Vec3 centre = triangle % 5u == 0u ? Vec3(0.0f, 0.0f, 0.0f)
                                                  : Vec3(smallUnit(smallRandom), smallUnit(smallRandom), smallUnit(smallRandom));
          const Vec3 a(smallUnit(smallRandom), smallUnit(smallRandom), smallUnit(smallRandom));
          const Vec3 b(smallUnit(smallRandom), smallUnit(smallRandom), smallUnit(smallRandom));
          const std::uint32_t base = static_cast<std::uint32_t>(positions.size());
          positions.insert(positions.end(), {centre + a * 0.1f, centre + b * 0.1f, centre - (a + b) * 0.1f});
          indices.insert(indices.end(), {base, base + 1u, base + 2u});
        }
        addMesh(small, positions, indices, smallMaterial,
                transform(0.3f * static_cast<float>(level), {1.0f, 1.0f, 1.0f}, {2.5f * static_cast<float>(level), 0.0f, 0.0f}),
                "small level");
      }
      small.scene.vertexCount = static_cast<std::uint32_t>(small.vertices.size());
      small.scene.indexCount = static_cast<std::uint32_t>(small.indices.size());
      small.scene.vertexBuffer = uploader.createBuffer(small.vertices.data(), small.vertices.size() * sizeof(Vertex),
          geometryBufferUsage(context, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT), "small.vertices");
      small.scene.indexBuffer = uploader.createBuffer(small.indices.data(), small.indices.size() * sizeof(std::uint32_t),
          geometryBufferUsage(context, VK_BUFFER_USAGE_INDEX_BUFFER_BIT), "small.indices");
      const TraceScene smallTrace = buildTraceScene(small.scene, std::vector<std::uint32_t>(1, 0u));
      buildOnGpu(small.scene, smallTrace, "small-levels fixture");
    }
    Fixture fixture;
    makeFixture(fixture);
    uploadFixture(fixture, uploader, context, "oracle");

    std::vector<std::uint32_t> slots(fixture.scene.materials.size(), 0u);
    TraceScene trace = buildTraceScene(fixture.scene, slots);
    pt::Bvh bvh;
    pt::BvhStatistics cpuStatistics{};
    pt::WideBvh wideBvh;
    std::unique_ptr<GpuBvhBuildResult> gpuBuild;
    if (gpuBuilder) {
      gpuBuild = std::make_unique<GpuBvhBuildResult>(buildOnGpu(fixture.scene, trace, "oracle fixture"));
      trace.instances = gpuBuild->instances;
      if (wide != 0u) {
        // The corpus below traverses the GPU collapse of the GPU-built tree.
        GpuWideCollapseResult collapsed =
            checkCollapse(uploader, *collapser, gpuBuild->nodes, downloadBinaryNodes(uploader, *gpuBuild),
                          gpuBuild->instances, wide, "GPU-built oracle fixture", gpuBuild->statistics);
        gpuBuild->nodes = std::move(collapsed.nodes);
        trace.instances = gpuBuild->instances = collapsed.instances;
      }
    } else if (software) {
      cpuStatistics = buildOnCpu(fixture, trace, bvh);
      if (wide != 0u) {
        Buffer resident = uploader.createBuffer(bvh.nodes.data(), bvh.nodes.size() * sizeof(pt::float4),
                                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "oracle.binary.nodes");
        checkCollapse(uploader, *collapser, resident, bvh.nodes, trace.instances, wide, "CPU-built oracle fixture",
                      cpuStatistics);
        wideBvh = pt::buildWideBvh(bvh, trace.instances, wide);
        trace.instances = wideBvh.instances;
      }
    }
    buildWorldTriangles(fixture, trace);
    std::unique_ptr<SceneAccelerationStructure> acceleration;
    if (!software)
      acceleration = std::make_unique<SceneAccelerationStructure>(context, uploader, fixture.scene, trace);

    std::vector<OracleRay> baseRays = makeCorpus(fixture);
    std::vector<ExpectedHit> expected;
    expected.reserve(baseRays.size());
    for (const OracleRay &ray : baseRays) expected.push_back(traceOracle(fixture, ray));
    // Pin the adversarial fixture expectations so a broken corpus cannot agree with itself.
    if (!expected[0].found || expected[1].found || !expected[2].found || expected[3].found ||
        !expected[4].found || expected[5].found || expected[6].found || !expected[7].found ||
        expected[8].found || expected[9].found || !expected[10].found || expected[11].found ||
        !expected[12].found || !expected[13].found || expected[14].found || expected[15].found ||
        !expected[16].found)
      throw Error("the edge, extent, mask, alpha or degenerate adversarial corpus is malformed");

    std::vector<OracleRay> queries = baseRays;
    queries.reserve(kBaseRayCount * 2u);
    for (OracleRay ray : baseRays) {
      ray.control.z = 1u;
      queries.push_back(ray);
    }

    std::vector<pt::Material> materials;
    for (const Material &material : fixture.scene.materials)
      materials.push_back(convertMaterial(material.uniforms));
    Buffer rayBuffer = uploader.createBuffer(queries.data(), queries.size() * sizeof(OracleRay),
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "oracle.rays");
    Buffer hitBuffer(context, queries.size() * sizeof(OracleHit),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VMA_MEMORY_USAGE_AUTO, 0, "oracle.hits");
    Buffer instanceBuffer = uploader.createBuffer(trace.instances.data(),
                                                   trace.instances.size() * sizeof(pt::TraceInstance),
                                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "oracle.instances");
    Buffer materialBuffer = uploader.createBuffer(materials.data(), materials.size() * sizeof(pt::Material),
                                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "oracle.materials");
    const pt::uint4 control(static_cast<std::uint32_t>(queries.size()),
                            static_cast<std::uint32_t>(queries.size()), 0u, 0u);
    Buffer controlBuffer = uploader.createBuffer(&control, sizeof(control),
                                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, "oracle.control");
    Buffer bvhNodeBuffer, bvhTriangleBuffer;
    if (software) {
      if (gpuBuilder) {
        bvhNodeBuffer = std::move(gpuBuild->nodes);
        bvhTriangleBuffer = std::move(gpuBuild->triangles);
      } else {
        const void *nodes = wide != 0u ? static_cast<const void *>(wideBvh.nodes.data())
                                      : static_cast<const void *>(bvh.nodes.data());
        const std::size_t nodeBytes = wide != 0u ? wideBvh.nodes.size() * sizeof(pt::QuantizedWideNode)
                                                 : bvh.nodes.size() * sizeof(pt::float4);
        bvhNodeBuffer = uploader.createBuffer(nodes, nodeBytes,
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "oracle.bvh.nodes");
        const std::vector<pt::float4> &triangles = wide != 0u ? wideBvh.triangles : bvh.triangles;
        bvhTriangleBuffer = uploader.createBuffer(triangles.data(), triangles.size() * sizeof(pt::float4),
                                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "oracle.bvh.triangles");
      }
    }
    if (software) {
      // The GPU cost kernel against the host walk of the same published tree; the CPU
      // builder's own statistic is a third, independent account of the binary cost.
      const std::size_t nodeStride = wide != 0u ? sizeof(pt::QuantizedWideNode) : 4u * sizeof(pt::float4);
      const auto nodeCount = static_cast<std::uint32_t>(bvhNodeBuffer.size / nodeStride);
      const pt::LayoutCost measured = gpuBvhSahCost(context, uploader, bvhNodeBuffer, nodeCount, trace.instances, wide != 0u);
      pt::LayoutCost reference;
      if (gpuBuilder && wide != 0u) {
        const std::vector<std::uint8_t> bytes = uploader.readBuffer(bvhNodeBuffer, bvhNodeBuffer.size);
        std::vector<pt::QuantizedWideNode> nodes(bytes.size() / sizeof(pt::QuantizedWideNode));
        std::memcpy(nodes.data(), bytes.data(), bytes.size());
        reference = pt::wideLayoutCost(nodes, trace.instances);
      } else if (gpuBuilder) {
        const std::vector<std::uint8_t> bytes = uploader.readBuffer(bvhNodeBuffer, bvhNodeBuffer.size);
        std::vector<pt::float4> nodes(bytes.size() / sizeof(pt::float4));
        std::memcpy(nodes.data(), bytes.data(), bytes.size());
        reference = pt::binaryLayoutCost(nodes, trace.instances);
        if (gpuBuild->statistics.sahCost != measured.bottom)
          throw Error("the GPU builder's published SAH cost is not its tree's");
      } else {
        reference = wide != 0u ? pt::wideLayoutCost(wideBvh.nodes, trace.instances) : pt::binaryLayoutCost(bvh.nodes, trace.instances);
      }
      auto close = [](double a, double b) { return std::abs(a - b) <= 1e-4 * std::max(std::abs(b), 1e-30); };
      if (!close(measured.top, reference.top) || !close(measured.bottom, reference.bottom) ||
          (!gpuBuilder && wide == 0u && !close(measured.bottom, cpuStatistics.sahCost)))
        throw Error("GPU SAH cost " + std::to_string(measured.top) + " + " + std::to_string(measured.bottom) +
                    " differs from the host's " + std::to_string(reference.top) + " + " + std::to_string(reference.bottom));
      std::printf("PASS: GPU SAH cost matches the host (top %.4f, bottom %.4f)\n", measured.top, measured.bottom);
    }
    if (gpuBuilder) {
      const std::vector<std::uint8_t> nodeBytes = uploader.readBuffer(bvhNodeBuffer, bvhNodeBuffer.size);
      const std::vector<std::uint8_t> triangleBytes = uploader.readBuffer(bvhTriangleBuffer, bvhTriangleBuffer.size);
      std::vector<pt::float4> downloadedNodes(nodeBytes.size() / sizeof(pt::float4));
      std::vector<pt::float4> downloadedTriangles(triangleBytes.size() / sizeof(pt::float4));
      std::memcpy(downloadedNodes.data(), nodeBytes.data(), nodeBytes.size());
      std::memcpy(downloadedTriangles.data(), triangleBytes.data(), triangleBytes.size());
      pt::WideBvh downloadedWide;
      if (wide != 0u) {
        downloadedWide.nodes.resize(nodeBytes.size() / sizeof(pt::QuantizedWideNode));
        std::memcpy(downloadedWide.nodes.data(), nodeBytes.data(), nodeBytes.size());
        downloadedWide.triangles = downloadedTriangles;
        downloadedWide.instances = trace.instances;
        downloadedWide.width = wide;
      }
      // Duplicate Morton codes keep primitive order (a triangle per primitive: not when clipped).
      std::uint32_t expectedTie = clipping ? 332u : 300u;
      const std::uint32_t firstCount = fixture.scene.primitives[trace.primitives[0]].indexCount / 3u;
      for (std::uint32_t sorted = 0; !clipping && sorted < firstCount; ++sorted) {
        const std::uint32_t primitive = pt::as_type<pt::uint>(
            downloadedTriangles[(trace.instances[0].triangleOffset + sorted) * 3u].w);
        if (primitive >= 300u && primitive < 332u) {
          if (primitive != expectedTie++)
            throw Error("GPU LBVH duplicate-Morton stable ordering failed");
        }
      }
      if (expectedTie != 332u) throw Error("GPU LBVH duplicate-Morton fixture was not emitted");
      if (!clipping) std::printf("PASS: GPU LBVH stable duplicate-Morton ordering\n");
      pt::HostTextures hostTextures;
      const HostView downloadedView(hostTextures, trace.instances, materials, fixture.indices,
                                    reinterpret_cast<const float *>(fixture.vertices.data()),
                                    fixture.vertices.size() * sizeof(fixture.vertices[0]) / sizeof(float), downloadedNodes,
                                    downloadedTriangles);
      std::vector<OracleHit> downloadedHits(queries.size());
      for (std::size_t i = 0; i < queries.size(); ++i) {
        const OracleRay &ray = queries[i];
        const pt::float3 direction(ray.directionAndMax.x, ray.directionAndMax.y, ray.directionAndMax.z);
        const float minimum = ray.originAndMin.w;
        const pt::float3 origin(ray.originAndMin.x + direction.x * minimum,
                                ray.originAndMin.y + direction.y * minimum,
                                ray.originAndMin.z + direction.z * minimum);
        const pt::PtHit hit = wide != 0u
            ? pt::traceWideBvh(downloadedWide, downloadedView, origin, direction, ray.directionAndMax.w - minimum, ray.control.x,
                               ray.control.y, pt::float2(-1.0f, 0.0f), ray.control.z)
            : pt::ptTraceBvh(downloadedView, origin, direction, ray.directionAndMax.w - minimum, ray.control.x, ray.control.y,
                             pt::float2(-1.0f, 0.0f), ray.control.z);
        OracleHit output{};
        output.distanceAndBarycentric = {ray.directionAndMax.w, 0.0f, 0.0f, 0.0f};
        output.identity = {0u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0u};
        if (hit.found != 0u) {
          output.distanceAndBarycentric = {hit.t + minimum, hit.barycentric.x, hit.barycentric.y, 0.0f};
          output.identity = {1u, hit.instance, hit.primitive, 0u};
        }
        downloadedHits[i] = output;
      }
      if (compareResults(fixture, baseRays, expected, downloadedHits) != 0)
        throw Error("CPU traversal of the downloaded GPU LBVH did not match the independent oracle");
      std::printf("PASS: CPU traversal of downloaded GPU LBVH\n");
    }

    const std::array<std::uint8_t, 4> white{255, 255, 255, 255};
    Image whiteTexture = uploader.createTexture(white.data(), white.size(), 1, 1,
                                                VK_FORMAT_R8G8B8A8_UNORM, 1, "oracle.white");
    std::vector<VkImageView> textureViews(pt::kHitTextureSlots, whiteTexture.view);
    GltfSamplers samplers(context);

    std::unique_ptr<Program> program;
    Pipeline pipeline;
    std::unique_ptr<RayPipeline> tracePipeline;
    if (rayPipeline) {
      const std::vector<RayStageDescription> stages{
          {"pipeline_oracle_generate", VK_SHADER_STAGE_RAYGEN_BIT_KHR},
          {"pipeline_oracle_miss", VK_SHADER_STAGE_MISS_BIT_KHR},
          {"pipeline_oracle_closest", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR},
          {"pipeline_oracle_alpha", VK_SHADER_STAGE_ANY_HIT_BIT_KHR}};
      const std::vector<RayShaderGroup> groups{
          {VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 0u},
          {VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 1u},
          {VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
           VK_SHADER_UNUSED_KHR, 2u, 3u, VK_SHADER_UNUSED_KHR}};
      tracePipeline = std::make_unique<RayPipeline>(
          context, stages, groups, ShaderBindingRecords{0u, {1u}, {2u}});
    } else {
      program = std::make_unique<Program>(context, wavefront ? "wavefront_trace" :
          wide != 0u ? "bvh_wide_oracle" : software ? "bvh_oracle" + traversal : std::string("ray_query_oracle"));
      pipeline = Pipeline(context, *program, software ? "software BVH oracle" : "ray query oracle");
    }
    const ShaderLayout &layout = rayPipeline ? tracePipeline->shaderLayout() : program->shaderLayout();
    std::unique_ptr<Program> enqueueProgram;
    Pipeline enqueuePipeline;
    Buffer queueBuffer, counterBuffer;
    if (wavefront) {
      enqueueProgram = std::make_unique<Program>(context, "wavefront_enqueue");
      enqueuePipeline = Pipeline(context, *enqueueProgram, "wavefront enqueue");
      queueBuffer = Buffer(context, queries.size() * sizeof(WavefrontRay),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, 0,
                           "oracle.wavefront.queue");
      const pt::uint4 counters(0u, 0u, static_cast<std::uint32_t>(queries.size()), 0u);
      counterBuffer = uploader.createBuffer(&counters, sizeof(counters),
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
          "oracle.wavefront.counters");
    }
    DescriptorPool pool(context, wavefront ? 3u : 1u);
    // Every oracle entry declares the same globals; each stage reads only its own.
    const VkDescriptorSet set = layout.allocate(pool);
    DescriptorWriter writer(context, layout, set);
    writer.buffer("rays", rayBuffer).buffer("hits", hitBuffer).buffer("control", controlBuffer)
        .buffer("geometry.traceInstances", instanceBuffer).buffer("geometry.materials", materialBuffer)
        .buffer("geometry.indices", fixture.scene.indexBuffer).buffer("geometry.vertices", fixture.scene.vertexBuffer)
        .textureArray("maps.table", textureViews, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    for (int i = 0; i < 6; ++i) writer.sampler(std::string("maps.") + GltfSamplers::names[i], samplers.handles[i]);
    if (wavefront) writer.buffer("queue", queueBuffer).buffer("counters", counterBuffer);
    if (software)
      writer.buffer(wide != 0u ? "wideNodes" : "bvhNodes", bvhNodeBuffer).buffer("bvhTriangles", bvhTriangleBuffer);
    else
      writer.accelerationStructure("accelerationStructure", acceleration->topLevel);
    writer.apply();

    VkDescriptorSet enqueueSet = VK_NULL_HANDLE;
    if (wavefront) {
      enqueueSet = enqueueProgram->allocate(pool);
      DescriptorWriter(context, *enqueueProgram, enqueueSet)
          .buffer("rays", rayBuffer).buffer("queue", queueBuffer).buffer("counters", counterBuffer)
          .buffer("control", controlBuffer).apply();

      const pt::uint4 overflowInitial(0u, 0u, static_cast<std::uint32_t>(queries.size() / 2u), 0u);
      Buffer overflowCounter = uploader.createBuffer(&overflowInitial, sizeof(overflowInitial),
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
          "oracle.wavefront.overflow-counters");
      const pt::uint4 overflowControl(static_cast<std::uint32_t>(queries.size()),
                                      static_cast<std::uint32_t>(queries.size() / 2u), 0u, 0u);
      Buffer overflowControlBuffer = uploader.createBuffer(&overflowControl, sizeof(overflowControl),
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, "oracle.wavefront.overflow-control");
      const VkDescriptorSet overflowSet = enqueueProgram->allocate(pool);
      DescriptorWriter(context, *enqueueProgram, overflowSet)
          .buffer("rays", rayBuffer).buffer("queue", queueBuffer).buffer("counters", overflowCounter)
          .buffer("control", overflowControlBuffer).apply();
      uploader.runImmediate([&](VkCommandBuffer command) {
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, enqueuePipeline.handle);
        enqueueProgram->bind(command, overflowSet);
        vkCmdDispatch(command, (static_cast<std::uint32_t>(queries.size()) + 63u) / 64u, 1, 1);
      });
      const std::vector<std::uint8_t> overflowBytes = uploader.readBuffer(overflowCounter, sizeof(pt::uint4));
      pt::uint4 overflow{};
      std::memcpy(&overflow, overflowBytes.data(), sizeof(overflow));
      if (overflow.x != queries.size() || overflow.y != queries.size() - queries.size() / 2u)
        throw Error("wavefront queue overflow counters did not report every rejected ray");
      std::printf("PASS: wavefront queue reports %u over-capacity rays\n", overflow.y);
    }

    const auto dispatchStarted = std::chrono::steady_clock::now();
    uploader.runImmediate([&](VkCommandBuffer command) {
      if (wavefront) {
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, enqueuePipeline.handle);
        enqueueProgram->bind(command, enqueueSet);
        vkCmdDispatch(command, (static_cast<std::uint32_t>(queries.size()) + 63u) / 64u, 1, 1);
        VkMemoryBarrier2 queueBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        queueBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        queueBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        queueBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        queueBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        VkDependencyInfo queueDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        queueDependency.memoryBarrierCount = 1;
        queueDependency.pMemoryBarriers = &queueBarrier;
        vkCmdPipelineBarrier2(command, &queueDependency);
      }
      if (rayPipeline) tracePipeline->bind(command);
      else vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
      layout.bind(command, set);
      if (rayPipeline) tracePipeline->traceRays(command, static_cast<std::uint32_t>(queries.size()));
      else vkCmdDispatch(command, (static_cast<std::uint32_t>(queries.size()) + 63u) / 64u, 1, 1);
      VkBufferMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
      barrier.srcStageMask = rayPipeline ? VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR
                                         : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
      barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
      barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
      barrier.buffer = hitBuffer.handle;
      barrier.offset = 0;
      barrier.size = VK_WHOLE_SIZE;
      VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      dependency.bufferMemoryBarrierCount = 1;
      dependency.pBufferMemoryBarriers = &barrier;
      vkCmdPipelineBarrier2(command, &dependency);
    });
    const double dispatchSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - dispatchStarted).count();
    std::printf("%s dispatch: %.2f M queries/s (host-timed dispatch plus wait)\n",
                rayPipeline ? "ray pipeline" : wavefront ? "wavefront binary BVH" : wide == 4u ? "quantized BVH4" : wide == 8u ? "quantized BVH8" :
                sah ? "GPU binned-SAH software-BVH" : ploc ? "parallel-PLOC software-BVH" : lbvh ? "parallel-LBVH software-BVH" : gpuBuilder ? "GPU-built software-BVH" : software ? "software-BVH" : "ray-query",
                static_cast<double>(queries.size()) / dispatchSeconds * 1e-6);
    const std::vector<std::uint8_t> bytes =
        uploader.readBuffer(hitBuffer, queries.size() * sizeof(OracleHit));
    std::vector<OracleHit> hits(queries.size());
    std::memcpy(hits.data(), bytes.data(), bytes.size());
    result = compareResults(fixture, baseRays, expected, hits);
    context.waitIdle();
  }
  if (context.sawValidationError) {
    for (const LogEntry &entry : logEntries())
      if (entry.level != LogEntry::Level::Info && entry.level != LogEntry::Level::Warning)
        std::fprintf(stderr, "%s\n", entry.text.c_str());
    std::fprintf(stderr, "a Vulkan validation error was reported during the GPU ray oracle\n");
    return 1;
  }
  if (result == 0)
    std::printf("PASS: %s corpus on %s; Vulkan validation clean\n",
                rayPipeline ? "ray pipeline" : wavefront ? "wavefront binary BVH" : wide == 4u ? "quantized BVH4" : wide == 8u ? "quantized BVH8" :
                ploc && plocTopology == GpuBvhTopology::BinnedSah ? "GPU binned-SAH software BVH" : ploc ? "parallel-PLOC software BVH" : lbvh ? "parallel-LBVH software BVH" : gpuBuilder ? "GPU-built software BVH" : software ? "software BVH" : "hardware ray-query",
                context.info.name.c_str());
  return result;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "";
    // --gpu-lbvh-wide4/8, --gpu-apetrei-wide8, --gpu-batched-wide8, --gpu-ploc-wide8: the parallel
    // builder's tree collapsed on the GPU.
    const bool singlePass = mode == "--gpu-apetrei" || mode == "--gpu-apetrei-wide8";
    const bool batched = mode == "--gpu-batched" || mode == "--gpu-batched-wide8";
    const bool lbvh = singlePass || batched || mode == "--gpu-lbvh" || mode == "--gpu-lbvh-wide4" || mode == "--gpu-lbvh-wide8";
    const GpuBvhTopology lbvhTopology = singlePass ? GpuBvhTopology::SinglePassLbvh :
                                       batched    ? GpuBvhTopology::BatchedLbvh :
                                                    GpuBvhTopology::Lbvh;
    // --gpu-sah(-wide8): the binned SAH, run through the PLOC path with its own checks.
    const bool plocFused = mode == "--gpu-plocpp";
    const bool hploc = mode == "--gpu-hploc" || mode == "--gpu-hploc-wide8";
    const bool sah = mode == "--gpu-sah" || mode == "--gpu-sah-wide8";
    const bool ploc = plocFused || hploc || sah || mode == "--gpu-ploc" || mode == "--gpu-ploc-wide8";
    const GpuBvhTopology plocTopology = plocFused ? GpuBvhTopology::PlocPlusPlus :
                                        hploc     ? GpuBvhTopology::Hploc :
                                        sah       ? GpuBvhTopology::BinnedSah :
                                                    GpuBvhTopology::Ploc;
    const bool gpuBuilder = lbvh || ploc || mode == "--gpu-builder";
    const std::uint32_t wide = mode == "--wide4" || mode == "--gpu-lbvh-wide4" ? 4u :
                               mode == "--wide8" || mode == "--gpu-lbvh-wide8" || mode == "--gpu-apetrei-wide8" ||
                                       mode == "--gpu-batched-wide8" || mode == "--gpu-hploc-wide8" ||
                                       mode == "--gpu-sah-wide8" ||
                                       mode == "--gpu-ploc-wide8"
                                   ? 8u
                                   : 0u;
    const bool wavefront = argc > 1 && std::string(argv[1]) == "--wavefront";
    const bool rayPipeline = argc > 1 && std::string(argv[1]) == "--pipeline";
    const bool software = gpuBuilder || wide != 0u || wavefront ||
                          (argc > 1 && std::string(argv[1]) == "--software");
    // A second argument --split-clipping: every build over split clipping's references.
    const float clip = argc > 2 && std::string(argv[2]) == "--split-clipping" ? 0.5f : 0.0f;
    // --traversal NAME (a --bvh-traversal): the binary oracle traced by that traversal's kernel.
    std::string traversal;
    if (argc > 3 && std::string(argv[2]) == "--traversal") {
      for (const BvhTraversalInfo &info : kBvhTraversals)
        if (argv[3] == std::string(info.name)) traversal = info.suffix;
      if (traversal.empty() || wide != 0u || wavefront || !software)
        throw Error(std::string("--traversal takes a binary software mode and one of while-while, speculative, "
                                "restart-trail, not ") + argv[3]);
      std::printf("Binary BVH traversal: %s\n", argv[3]);
    }
    // --deep (last): that traversal's deep-stack (kBvhStackDeep) kernel.
    if (argc > 2 && std::string(argv[argc - 1]) == "--deep") {
      if (wide != 0u || wavefront || !software || traversal == "_restart_trail")
        throw Error("--deep takes a binary software mode and a traversal with a stack");
      traversal += "_deep";
      std::printf("Binary BVH traversal stack: %d entries\n", pt::kBvhStackDeep);
    }
    return run(software, gpuBuilder, lbvh, lbvhTopology, ploc, plocTopology, wide, wavefront, rayPipeline, clip,
               traversal);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "GPU ray oracle failed: %s\n", error.what());
    return 1;
  }
}
