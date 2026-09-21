#include "scene/Scene.h"

#include <algorithm>

#include "core/Log.h"
#include "gpu/Uploader.h"

#include <array>

namespace basalt {

Scene::~Scene() = default;

void Scene::destroy(const Context &context) {
  for (VkSampler sampler : samplers)
    if (sampler) vkDestroySampler(context.device, sampler, nullptr);
  samplers.clear();
  textures.clear();
  vertexBuffer.reset();
  indexBuffer.reset();
  primitives.clear();
}

VkBufferUsageFlags geometryBufferUsage(const Context &context, VkBufferUsageFlags base) {
  VkBufferUsageFlags usage = base | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (context.rayTracingSupported)
    usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  return usage;
}

void appendGroundPlane(Scene &scene, std::vector<Vertex> &vertices,
                       std::vector<std::uint32_t> &indices) {
  if (!scene.bounds.valid()) return;
  // Wide enough that its horizon stays outside a framed view.
  const float extent = std::max(scene.bounds.radius() * 6.0f, 1.0f);
  const Vec3 centre = scene.bounds.center();
  // A hair below the model, or a flat-bottomed one fights it for depth.
  const float y = scene.bounds.minimum.y - std::max(scene.bounds.radius(), 1e-3f) * 1e-3f;

  Material material;
  material.name = "ground";
  material.uniforms.baseColorFactor = {0.34f, 0.34f, 0.36f, 1.0f};
  material.uniforms.factors = {0.0f, 0.8f, 1.0f, 0.0f};
  scene.materials.push_back(material);
  const auto materialIndex = static_cast<std::uint32_t>(scene.materials.size()) - 1;

  const auto firstVertex = static_cast<std::int32_t>(vertices.size());
  const auto firstIndex = static_cast<std::uint32_t>(indices.size());
  for (int corner = 0; corner < 4; ++corner) {
    const float sx = (corner == 1 || corner == 2) ? 1.0f : -1.0f;
    const float sz = corner >= 2 ? 1.0f : -1.0f;
    Vertex vertex;
    vertex.position = {centre.x + sx * extent, y, centre.z + sz * extent};
    vertex.normal = {0, 1, 0};
    vertex.tangent = {1, 0, 0, 1};
    vertex.uv0 = {sx * extent, sz * extent};
    vertex.uv1 = vertex.uv0;
    vertices.push_back(vertex);
  }
  for (std::uint32_t index : {0u, 2u, 1u, 0u, 3u, 2u}) indices.push_back(index);

  Primitive primitive;
  primitive.firstIndex = firstIndex;
  primitive.indexCount = 6;
  primitive.vertexOffset = firstVertex;
  primitive.material = materialIndex;
  primitive.name = "ground";
  primitive.bounds.add({centre.x - extent, y, centre.z - extent});
  primitive.bounds.add({centre.x + extent, y, centre.z + extent});
  primitive.worldBounds = primitive.bounds;
  scene.groundPrimitive = static_cast<int>(scene.primitives.size());
  scene.primitives.push_back(primitive);
}

std::unique_ptr<Scene> createDefaultScene(const Context &context, Uploader &uploader) {
  auto scene = std::make_unique<Scene>();
  scene->sourcePath = "(built-in cube)";

  // Four vertices per face, so each face has its own normal and tangent.
  struct Face {
    Vec3 normal, tangent, right, up;
  };
  // right x up is the normal for every face, so all six wind the same way.
  const std::array<Face, 6> faces{{
      {{1, 0, 0}, {0, 0, -1}, {0, 0, -1}, {0, 1, 0}},
      {{-1, 0, 0}, {0, 0, 1}, {0, 0, 1}, {0, 1, 0}},
      {{0, 1, 0}, {1, 0, 0}, {1, 0, 0}, {0, 0, -1}},
      {{0, -1, 0}, {1, 0, 0}, {1, 0, 0}, {0, 0, 1}},
      {{0, 0, 1}, {1, 0, 0}, {1, 0, 0}, {0, 1, 0}},
      {{0, 0, -1}, {-1, 0, 0}, {-1, 0, 0}, {0, 1, 0}},
  }};

  std::vector<Vertex> vertices;
  std::vector<std::uint32_t> indices;
  for (const Face &face : faces) {
    const std::uint32_t base = static_cast<std::uint32_t>(vertices.size());
    for (int corner = 0; corner < 4; ++corner) {
      const float u = (corner == 1 || corner == 2) ? 1.0f : 0.0f;
      const float v = (corner >= 2) ? 1.0f : 0.0f;
      Vertex vertex;
      vertex.position = face.normal * 0.5f + face.right * (u - 0.5f) + face.up * (0.5f - v);
      vertex.normal = face.normal;
      vertex.tangent = Vec4(face.tangent, 1.0f);
      vertex.uv0 = {u, v};
      vertex.uv1 = {u, v};
      vertices.push_back(vertex);
      scene->bounds.add(vertex.position);
    }
    // Counter-clockwise from outside, as glTF specifies.
    for (std::uint32_t index : {0u, 2u, 1u, 0u, 3u, 2u}) indices.push_back(base + index);
  }

  Material material;
  material.name = "default";
  material.uniforms.baseColorFactor = {0.62f, 0.64f, 0.67f, 1.0f};
  material.uniforms.factors = {0.0f, 0.45f, 1.0f, 0.0f};
  scene->materials.push_back(material);
  const auto cubeIndexCount = static_cast<std::uint32_t>(indices.size());
  appendGroundPlane(*scene, vertices, indices);

  scene->vertexBuffer = uploader.createBuffer(vertices.data(), vertices.size() * sizeof(Vertex),
                                              geometryBufferUsage(context, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT),
                                              "cube.vertices");
  scene->indexBuffer = uploader.createBuffer(indices.data(), indices.size() * sizeof(std::uint32_t),
                                             geometryBufferUsage(context, VK_BUFFER_USAGE_INDEX_BUFFER_BIT),
                                             "cube.indices");
  scene->vertexCount = static_cast<std::uint32_t>(vertices.size());
  scene->indexCount = static_cast<std::uint32_t>(indices.size());
  scene->triangleCount = scene->indexCount / 3;

  scene->samplerDescriptions.push_back({});

  VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
  VkSampler sampler = VK_NULL_HANDLE;
  check(vkCreateSampler(context.device, &samplerInfo, nullptr, &sampler), "vkCreateSampler (cube)");
  scene->samplers.push_back(sampler);

  Primitive primitive;
  primitive.firstIndex = 0;
  primitive.indexCount = cubeIndexCount; // The ground's indices follow and are its own primitive's.
  primitive.material = 0;
  primitive.bounds = scene->bounds;
  primitive.worldBounds = scene->bounds;
  primitive.name = "cube";
  scene->primitives.push_back(primitive);
  return scene;
}

} // namespace basalt
