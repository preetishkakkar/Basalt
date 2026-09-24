#pragma once
#include "pt/CpuTracer.h"
#include "pt/Tables.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace pt_fixture {
using namespace pt;
inline WorkerPool &sharedPool() {
  static WorkerPool pool(std::max(1u, std::thread::hardware_concurrency()));
  return pool;
}

// Rows of an affine transform (rotation about y, uniform scale, translation) and its inverse.
struct Affine {
  float4 rows[3];
  float4 inverse[3];
  float4 normal[3];
};

inline Affine affine(float yawRadians, float3 scale, float3 translation) {
  const float c = std::cos(yawRadians), s = std::sin(yawRadians);
  Affine a;
  a.rows[0] = float4(c * scale.x, 0.0f, s * scale.z, translation.x);
  a.rows[1] = float4(0.0f, scale.y, 0.0f, translation.y);
  a.rows[2] = float4(-s * scale.x, 0.0f, c * scale.z, translation.z);
  // The inverse: inverse scale times transposed rotation, then undo translation.
  const float3 r0(c / scale.x, 0.0f, -s / scale.x);
  const float3 r1(0.0f, 1.0f / scale.y, 0.0f);
  const float3 r2(s / scale.z, 0.0f, c / scale.z);
  a.inverse[0] = float4(r0, -dot(r0, translation));
  a.inverse[1] = float4(r1, -dot(r1, translation));
  a.inverse[2] = float4(r2, -dot(r2, translation));
  // Inverse transpose of the upper 3x3.
  a.normal[0] = float4(c / scale.x, 0.0f, s / scale.z, 0.0f);
  a.normal[1] = float4(0.0f, 1.0f / scale.y, 0.0f, 0.0f);
  a.normal[2] = float4(-s / scale.x, 0.0f, c / scale.z, 0.0f);
  return a;
}

inline Affine affine(float yawRadians, float scale, float3 translation) {
  return affine(yawRadians, float3(scale), translation);
}

struct Builder {
  CpuScene scene;
  CpuFrame frame;

  uint material(float3 base, float metallic, float roughness, float alpha = 1.0f, uint alphaMode = 0) {
    Material m{};
    m.baseColorFactor = float4(base, alpha);
    m.emissive = float4(0.0f, 0.0f, 0.0f, 1.0f);
    m.factors = float4(metallic, roughness, 1.0f, 1.0f);
    m.alpha = float4(0.5f, static_cast<float>(alphaMode), 0.0f, 0.0f);
    m.transmission = float4(0.0f, 1.5f, 0.0f, 0.0f);
    m.clearcoat = float4(0.0f, 0.0f, 1.0f, 0.0f);
    m.extensionTextures = uint4(0u, 0u, 0u, 1u);
    frame.materials.push_back(m);
    return static_cast<uint>(frame.materials.size() - 1);
  }

  uint emissiveMaterial(float3 radiance) {
    const uint index = material(float3(0.0f), 0.0f, 1.0f);
    frame.materials[index].emissive = float4(radiance, 1.0f);
    return index;
  }

  void mesh(const std::vector<float3> &positions, const std::vector<float3> &normals, const std::vector<uint> &triangles,
            uint materialIndex, const Affine &transform, uint mask = kRayMaskScene, uint flags = 0) {
    const uint vertexOffset = static_cast<uint>(scene.vertices.size() / kVertexFloats);
    for (std::size_t i = 0; i < positions.size(); ++i) {
      const float3 p = positions[i], n = normals[i];
      const float v[kVertexFloats] = {p.x, p.y, p.z, n.x, n.y, n.z, 0, 0, 0, 1,
                                     p.x, p.z, 0, 0, 1, 1, 1, 1};
      scene.vertices.insert(scene.vertices.end(), v, v + kVertexFloats);
    }
    TraceInstance instance{};
    instance.objectToWorld0 = transform.rows[0];
    instance.objectToWorld1 = transform.rows[1];
    instance.objectToWorld2 = transform.rows[2];
    instance.worldToObject0 = transform.inverse[0];
    instance.worldToObject1 = transform.inverse[1];
    instance.worldToObject2 = transform.inverse[2];
    instance.normalToWorld0 = transform.normal[0];
    instance.normalToWorld1 = transform.normal[1];
    instance.normalToWorld2 = transform.normal[2];
    instance.firstIndex = static_cast<uint>(scene.indices.size());
    instance.vertexOffset = vertexOffset;
    instance.material = materialIndex;
    instance.slots = 1u << 24;  // white maps, flat normal map
    instance.mask = mask;
    instance.flags = flags | kInstanceDoubleSided;
    if (transform.rows[0].x * (transform.rows[1].y * transform.rows[2].z - transform.rows[1].z * transform.rows[2].y) -
            transform.rows[0].y * (transform.rows[1].x * transform.rows[2].z - transform.rows[1].z * transform.rows[2].x) +
            transform.rows[0].z * (transform.rows[1].x * transform.rows[2].y - transform.rows[1].y * transform.rows[2].x) < 0.0f)
      instance.flags |= kInstanceMirrored;
    scene.indices.insert(scene.indices.end(), triangles.begin(), triangles.end());
    scene.instances.push_back(instance);
    scene.triangleCounts.push_back(static_cast<uint>(triangles.size() / 3));
  }

  void sphere(float3 centre, float radius, uint materialIndex, uint segments = 96) {
    std::vector<float3> positions, normals;
    std::vector<uint> triangles;
    const uint rings = segments / 2;
    for (uint r = 0; r <= rings; ++r)
      for (uint s = 0; s <= segments; ++s) {
        const float theta = kPi * static_cast<float>(r) / static_cast<float>(rings);
        const float phi = 2.0f * kPi * static_cast<float>(s) / static_cast<float>(segments);
        const float3 n(std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi));
        positions.push_back(n);
        normals.push_back(n);
      }
    for (uint r = 0; r < rings; ++r)
      for (uint s = 0; s < segments; ++s) {
        const uint a = r * (segments + 1) + s, b = a + segments + 1;
        triangles.insert(triangles.end(), {a, a + 1, b, a + 1, b + 1, b});
      }
    mesh(positions, normals, triangles, materialIndex, affine(0.0f, radius, centre));
  }

  // A square in the plane y = height, facing +y.
  void floor(float halfSize, float height, uint materialIndex) {
    const std::vector<float3> p{{-halfSize, height, -halfSize}, {halfSize, height, -halfSize},
                                {halfSize, height, halfSize}, {-halfSize, height, halfSize}};
    const std::vector<float3> n(4, float3(0.0f, 1.0f, 0.0f));
    mesh(p, n, {0, 2, 1, 0, 3, 2}, materialIndex, affine(0.0f, 1.0f, float3(0.0f)));
  }

  // A square in the plane y = height, facing -y. Builder meshes are double-sided by
  // default; clear that flag after calling this helper for a one-sided emitter.
  void ceiling(float halfSize, float height, uint materialIndex) {
    const std::vector<float3> p{{-halfSize, height, -halfSize}, {halfSize, height, -halfSize},
                                {halfSize, height, halfSize}, {-halfSize, height, halfSize}};
    const std::vector<float3> n(4, float3(0.0f, -1.0f, 0.0f));
    mesh(p, n, {0, 1, 2, 0, 2, 3}, materialIndex, affine(0.0f, 1.0f, float3(0.0f)));
  }

  // A square in the plane z = depth, facing +z (towards a camera on +z).
  void wall(float halfSize, float depth, uint materialIndex, uint flags = 0) {
    const std::vector<float3> p{{-halfSize, -halfSize, depth}, {halfSize, -halfSize, depth},
                                {halfSize, halfSize, depth}, {-halfSize, halfSize, depth}};
    const std::vector<float3> n(4, float3(0.0f, 0.0f, 1.0f));
    mesh(p, n, {0, 1, 2, 0, 2, 3}, materialIndex, affine(0.0f, 1.0f, float3(0.0f)), kRayMaskScene, flags);
  }

  // An environment given by a function of direction, sampled on a small equirect.
  void environment(const std::function<float3(float3)> &radiance, uint width = 256, uint height = 128) {
    scene.environment.width = width;
    scene.environment.height = height;
    scene.environment.texels.assign(static_cast<std::size_t>(width) * height * 4, 1.0f);
    for (uint y = 0; y < height; ++y)
      for (uint x = 0; x < width; ++x) {
        const float2 uv((static_cast<float>(x) + 0.5f) / static_cast<float>(width),
                        (static_cast<float>(y) + 0.5f) / static_cast<float>(height));
        const float3 l = radiance(ptEquirectangularDirection(uv));
        float *t = scene.environment.texels.data() + (static_cast<std::size_t>(y) * width + x) * 4;
        t[0] = l.x;
        t[1] = l.y;
        t[2] = l.z;
      }
    buildEnvironmentDistribution(scene.environment.texels.data(), width, height, scene.distribution,
                                 scene.distributionInfo);
  }

  // A pinhole camera at `eye` looking at `target`, with a vertical field of view.
  void camera(float3 eye, float3 target, float fieldOfView, uint width, uint height) {
    const float3 forward = normalize(target - eye);
    const float3 right = normalize(cross(forward, float3(0.0f, 1.0f, 0.0f)));
    const float3 up = cross(right, forward);
    const float tangent = std::tan(fieldOfView * 0.5f);
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    PathUniforms &u = frame.uniforms;
    u.cameraPosition = float4(eye, 1.0f);
    u.cameraForward = float4(forward, 0.0f);
    u.cameraRight = float4(right * (tangent * aspect), 0.0f);
    u.cameraUp = float4(up * tangent, 0.0f);
    u.image = float4(static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f);
    frame.width = width;
    frame.height = height;
  }

  // The analytic sun: irradiance E at normal incidence from a disc of angular radius.
  void sun(float3 direction, float irradiance, float angularRadius) {
    const float cosine = std::cos(angularRadius);
    const float solidAngle = 2.0f * kPi * (1.0f - cosine);
    frame.uniforms.sunDirection = float4(normalize(direction), cosine);
    frame.uniforms.sunRadiance = float4(float3(irradiance / solidAngle), solidAngle);
  }

  // The rest of the uniforms, then the tables and the BVH.
  void finish(uint bounces = 16, uint strategy = 0, uint seed = 0) {
    PathUniforms &u = frame.uniforms;
    u.path = float4(static_cast<float>(bounces), 4.0f, 0.0f, static_cast<float>(strategy));
    u.environment = float4(1.0f, static_cast<float>(scene.environment.width),
                           static_cast<float>(scene.environment.height), 1.0f);
    u.distribution = scene.distribution.empty() ? float4(0.0f) : scene.distributionInfo;
    if (scene.distribution.empty()) scene.distribution.assign(4, 0.0f);
    u.counts = uint4(static_cast<uint>(frame.lights.size()), 7u, seed, 0u);
    if (frame.materials.empty()) material(float3(1.0f), 0.0f, 1.0f);
    scene.specularAlbedo = buildSpecularAlbedoTable();
    scene.emissiveTriangles = buildEmissiveTriangles(scene.vertices, scene.indices, scene.instances,
                                                      scene.triangleCounts, frame.materials);
    u.emissive = uint4(static_cast<uint>(scene.emissiveTriangles.size()), 0u, 0u, 0u);
    scene.bvhStatistics = buildBvh(scene.vertices, scene.indices, scene.instances, scene.triangleCounts, scene.bvh,
                                   sharedPool().size());
  }

  std::vector<float> render(uint samples) const { return CpuTracer::render(scene, frame, samples, sharedPool()).color; }
  CpuTracer::Result renderAll(uint samples) const { return CpuTracer::render(scene, frame, samples, sharedPool()); }
};


} // namespace pt_fixture
