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
#include "render/GpuBvhBuilder.h"
#include "render/TraceScene.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
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

struct SamplerOwner {
  VkDevice device = VK_NULL_HANDLE;
  VkSampler sampler = VK_NULL_HANDLE;
  ~SamplerOwner() {
    if (sampler) vkDestroySampler(device, sampler, nullptr);
  }
};

int run(bool software, bool gpuBuilder, std::uint32_t wide, bool wavefront, bool rayPipeline) {
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
      GpuBvhBuildResult empty = buildGpuBvh(context, uploader, emptyScene, emptyTrace);
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
      const GpuBvhBuildResult stressBuild = buildGpuBvh(context, uploader, stress.scene, stressTrace);
      if (stressBuild.statistics.triangles != stressTriangles ||
          stressBuild.statistics.topDepth + stressBuild.statistics.bottomDepth > PT_BVH_STACK)
        throw Error("GPU LBVH stress fixture failed its count or depth contract");
      std::printf("PASS: GPU LBVH %u-triangle stress fixture (depth %u + %u)\n", stressTriangles,
                  stressBuild.statistics.topDepth, stressBuild.statistics.bottomDepth);
    }
    Fixture fixture;
    makeFixture(fixture);
    fixture.scene.vertexCount = static_cast<std::uint32_t>(fixture.vertices.size());
    fixture.scene.indexCount = static_cast<std::uint32_t>(fixture.indices.size());
    fixture.scene.triangleCount = fixture.scene.indexCount / 3;
    fixture.scene.vertexBuffer = uploader.createBuffer(
        fixture.vertices.data(), fixture.vertices.size() * sizeof(Vertex),
        geometryBufferUsage(context, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT), "oracle.vertices");
    fixture.scene.indexBuffer = uploader.createBuffer(
        fixture.indices.data(), fixture.indices.size() * sizeof(std::uint32_t),
        geometryBufferUsage(context, VK_BUFFER_USAGE_INDEX_BUFFER_BIT), "oracle.indices");

    std::vector<std::uint32_t> slots(fixture.scene.materials.size(), 0u);
    TraceScene trace = buildTraceScene(fixture.scene, slots);
    pt::Bvh bvh;
    pt::WideBvh wideBvh;
    std::unique_ptr<GpuBvhBuildResult> gpuBuild;
    if (gpuBuilder) {
      gpuBuild = std::make_unique<GpuBvhBuildResult>(buildGpuBvh(context, uploader, fixture.scene, trace));
      trace.instances = gpuBuild->instances;
    } else if (software) {
      std::vector<float> vertices(fixture.vertices.size() * pt::kVertexFloats);
      std::memcpy(vertices.data(), fixture.vertices.data(), fixture.vertices.size() * sizeof(Vertex));
      std::vector<std::uint32_t> triangleCounts;
      triangleCounts.reserve(trace.primitives.size());
      for (const std::uint32_t primitive : trace.primitives)
        triangleCounts.push_back(fixture.scene.primitives[primitive].indexCount / 3);
      pt::buildBvh(vertices, fixture.indices, trace.instances, triangleCounts, bvh,
                   std::max(1u, std::thread::hardware_concurrency()));
      if (wide != 0u) {
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
    if (gpuBuilder) {
      const std::vector<std::uint8_t> nodeBytes = uploader.readBuffer(bvhNodeBuffer, bvhNodeBuffer.size);
      const std::vector<std::uint8_t> triangleBytes = uploader.readBuffer(bvhTriangleBuffer, bvhTriangleBuffer.size);
      std::vector<pt::float4> downloadedNodes(nodeBytes.size() / sizeof(pt::float4));
      std::vector<pt::float4> downloadedTriangles(triangleBytes.size() / sizeof(pt::float4));
      std::memcpy(downloadedNodes.data(), nodeBytes.data(), nodeBytes.size());
      std::memcpy(downloadedTriangles.data(), triangleBytes.data(), triangleBytes.size());
      std::uint32_t expectedTie = 300u;
      const std::uint32_t firstCount = fixture.scene.primitives[trace.primitives[0]].indexCount / 3u;
      for (std::uint32_t sorted = 0; sorted < firstCount; ++sorted) {
        const std::uint32_t primitive = pt::as_type<pt::uint>(
            downloadedTriangles[(trace.instances[0].triangleOffset + sorted) * 3u].w);
        if (primitive >= 300u && primitive < 332u) {
          if (primitive != expectedTie++)
            throw Error("GPU LBVH duplicate-Morton stable ordering failed");
        }
      }
      if (expectedTie != 332u) throw Error("GPU LBVH duplicate-Morton fixture was not emitted");
      std::printf("PASS: GPU LBVH stable duplicate-Morton ordering\n");
      pt::HostTextures hostTextures;
      std::vector<OracleHit> downloadedHits(queries.size());
      for (std::size_t i = 0; i < queries.size(); ++i) {
        const OracleRay &ray = queries[i];
        const pt::float3 direction(ray.directionAndMax.x, ray.directionAndMax.y, ray.directionAndMax.z);
        const float minimum = ray.originAndMin.w;
        const pt::float3 origin(ray.originAndMin.x + direction.x * minimum,
                                ray.originAndMin.y + direction.y * minimum,
                                ray.originAndMin.z + direction.z * minimum);
        const pt::PtHit hit = pt::ptTraceBvh(
            downloadedNodes.data(), downloadedTriangles.data(), trace.instances.data(), materials.data(),
            fixture.indices.data(), reinterpret_cast<const float *>(fixture.vertices.data()), hostTextures,
            origin, direction, ray.directionAndMax.w - minimum, ray.control.x, ray.control.y, pt::float2(-1.0f, 0.0f), ray.control.z);
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
    SamplerOwner sampler{context.device};
    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.maxLod = 0.0f;
    check(vkCreateSampler(context.device, &samplerInfo, nullptr, &sampler.sampler),
          "vkCreateSampler (ray oracle)");

    std::unique_ptr<Program> program;
    Pipeline pipeline;
    std::unique_ptr<RayPipeline> tracePipeline;
    if (rayPipeline) {
      const std::vector<RayStageDescription> stages{
          {"pipeline_oracle_generate", VK_SHADER_STAGE_RAYGEN_BIT_KHR},
          {"pipeline_oracle_miss", VK_SHADER_STAGE_MISS_BIT_KHR},
          {"pipeline_oracle_closest", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR},
          {"pipeline_oracle_alpha", VK_SHADER_STAGE_ANY_HIT_BIT_KHR}};
      const std::vector<m2v::host::RayShaderGroup> groups{
          {VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 0u},
          {VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 1u},
          {VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
           VK_SHADER_UNUSED_KHR, 2u, 3u, VK_SHADER_UNUSED_KHR}};
      tracePipeline = std::make_unique<RayPipeline>(
          context, stages, groups, m2v::host::ShaderBindingRecord{0u, {}},
          std::vector<m2v::host::ShaderBindingRecord>{{1u, {}}},
          std::vector<m2v::host::ShaderBindingRecord>{{2u, {}}});
    } else {
      program = std::make_unique<Program>(context, wavefront ? "wavefront_trace" :
          wide != 0u ? "bvh_wide_oracle" : software ? "bvh_oracle" : "ray_query_oracle");
      pipeline = Pipeline(context, *program, software ? "software BVH oracle" : "ray query oracle");
    }
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
    const VkDescriptorSet set = pool.allocate(rayPipeline ? tracePipeline->setLayout : program->setLayouts[0]);
    if (rayPipeline) {
      DescriptorWriter(context, tracePipeline->shader(0), set)
          .buffer("rays", rayBuffer).buffer("hits", hitBuffer).buffer("control", controlBuffer)
          .accelerationStructure("scene", acceleration->topLevel).apply();
      DescriptorWriter(context, tracePipeline->shader(3), set)
          .buffer("traceInstances", instanceBuffer).buffer("materials", materialBuffer)
          .buffer("indices", fixture.scene.indexBuffer).buffer("vertices", fixture.scene.vertexBuffer)
          .textureArray("maps", textureViews, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
          .sampler("materialSampler", sampler.sampler).apply();
    } else {
      DescriptorWriter writer(context, program->compute(), set);
      if (wavefront)
        writer.buffer("queue", queueBuffer).buffer("counters", counterBuffer).buffer("hits", hitBuffer);
      else
        writer.buffer("rays", rayBuffer).buffer("hits", hitBuffer);
      writer.buffer("traceInstances", instanceBuffer).buffer("materials", materialBuffer)
          .buffer("indices", fixture.scene.indexBuffer).buffer("vertices", fixture.scene.vertexBuffer)
          .buffer("control", controlBuffer);
      if (software)
        writer.buffer("bvhNodes", bvhNodeBuffer).buffer("bvhTriangles", bvhTriangleBuffer);
      else
        writer.accelerationStructure("scene", acceleration->topLevel);
      writer.textureArray("maps", textureViews, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
          .sampler("materialSampler", sampler.sampler).apply();
    }

    VkDescriptorSet enqueueSet = VK_NULL_HANDLE;
    if (wavefront) {
      enqueueSet = pool.allocate(enqueueProgram->setLayouts[0]);
      DescriptorWriter(context, enqueueProgram->compute(), enqueueSet)
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
      const VkDescriptorSet overflowSet = pool.allocate(enqueueProgram->setLayouts[0]);
      DescriptorWriter(context, enqueueProgram->compute(), overflowSet)
          .buffer("rays", rayBuffer).buffer("queue", queueBuffer).buffer("counters", overflowCounter)
          .buffer("control", overflowControlBuffer).apply();
      uploader.runImmediate([&](VkCommandBuffer command) {
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, enqueuePipeline.handle);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, enqueueProgram->layout,
                                0, 1, &overflowSet, 0, nullptr);
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
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, enqueueProgram->layout,
                                0, 1, &enqueueSet, 0, nullptr);
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
      if (rayPipeline)
        tracePipeline->trace(command, set, static_cast<std::uint32_t>(queries.size()));
      else {
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, program->layout, 0, 1, &set, 0, nullptr);
        vkCmdDispatch(command, (static_cast<std::uint32_t>(queries.size()) + 63u) / 64u, 1, 1);
      }
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
                gpuBuilder ? "GPU-built software-BVH" : software ? "software-BVH" : "ray-query",
                static_cast<double>(queries.size()) / dispatchSeconds * 1e-6);
    const std::vector<std::uint8_t> bytes =
        uploader.readBuffer(hitBuffer, queries.size() * sizeof(OracleHit));
    std::vector<OracleHit> hits(queries.size());
    std::memcpy(hits.data(), bytes.data(), bytes.size());
    result = compareResults(fixture, baseRays, expected, hits);
    context.waitIdle();
  }
  if (context.sawValidationError) {
    std::fprintf(stderr, "a Vulkan validation error was reported during the GPU ray oracle\n");
    return 1;
  }
  if (result == 0)
    std::printf("PASS: %s corpus on %s; Vulkan validation clean\n",
                rayPipeline ? "ray pipeline" : wavefront ? "wavefront binary BVH" : wide == 4u ? "quantized BVH4" : wide == 8u ? "quantized BVH8" :
                gpuBuilder ? "GPU-built software BVH" : software ? "software BVH" : "hardware ray-query",
                context.info.name.c_str());
  return result;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const bool gpuBuilder = argc > 1 && std::string(argv[1]) == "--gpu-builder";
    const std::uint32_t wide = argc > 1 && std::string(argv[1]) == "--wide4" ? 4u :
                               argc > 1 && std::string(argv[1]) == "--wide8" ? 8u : 0u;
    const bool wavefront = argc > 1 && std::string(argv[1]) == "--wavefront";
    const bool rayPipeline = argc > 1 && std::string(argv[1]) == "--pipeline";
    const bool software = gpuBuilder || wide != 0u || wavefront ||
                          (argc > 1 && std::string(argv[1]) == "--software");
    return run(software, gpuBuilder, wide, wavefront, rayPipeline);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "GPU ray oracle failed: %s\n", error.what());
    return 1;
  }
}
