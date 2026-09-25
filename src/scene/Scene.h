// The scene as the renderer sees it: one vertex and one index buffer, flat primitives, materials.
#pragma once
#include "core/Math.h"
#include "gpu/Resources.h"

#include <cstdint>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace basalt {

class Uploader;

// Interleaved to match the vertex inputs the Slang vertex entries declare.
struct Vertex {
  Vec3 position;
  Vec3 normal;
  Vec4 tangent; // xyz tangent, w handedness (+1 or -1)
  Vec2 uv0;
  Vec2 uv1;
  Vec4 color{1, 1, 1, 1};
};
static_assert(sizeof(Vertex) == 72, "the vertex layout must match the pipeline's stride");

enum class AlphaMode : std::uint32_t { Opaque = 0, Mask = 1, Blend = 2 };

// Mirrors Material in shaders/slang/basalt/basalt_types.slang.
struct MaterialUniforms {
  Vec4 baseColorFactor{1, 1, 1, 1};
  Vec4 emissive{0, 0, 0, 1};      // rgb factor, w strength
  Vec4 factors{1, 1, 1, 1};       // metallic, roughness, normal scale, occlusion strength
  Vec4 alpha{0.5f, 0, 0, 0};      // cutoff, mode, double sided, unused
  std::array<std::uint32_t, 4> texture{}; // packed UV-set and sampler-mode bytes
  Vec4 transmission{0, 1.5f, 0, 0};       // factor, IOR, volume thickness (0 thin-walled), IOR given
  Vec4 clearcoat{0, 0, 1, 0};             // factor, roughness, normal scale, reserved
  std::array<std::uint32_t, 4> extensionTextures{0, 0, 0, 1};  // slot | UV << 8 | sampler << 16
};
static_assert(sizeof(MaterialUniforms) == 128, "the material layout must match the shader");

// Texture indices into Scene::textures; -1 is the neutral default.
struct Material {
  std::string name;
  MaterialUniforms uniforms;
  int baseColor = -1;
  int metallicRoughness = -1;
  int normal = -1;
  int occlusion = -1;
  int emissive = -1;
  // KHR_materials_transmission / clearcoat textures, placed in the hit table's slots.
  int transmissionTexture = -1, clearcoatTexture = -1, clearcoatRoughnessTexture = -1, clearcoatNormalTexture = -1;
  int sampler = 0;           // Index into Scene::samplers.
  AlphaMode alphaMode = AlphaMode::Opaque;
  bool doubleSided = false;
};

struct Primitive {
  std::uint32_t firstIndex = 0;
  std::uint32_t indexCount = 0;
  std::int32_t vertexOffset = 0;
  std::uint32_t material = 0;
  Aabb bounds;          // In the primitive's own space.
  Aabb worldBounds;     // After the node transform.
  Mat4 transform;       // World transform of the node that holds it.
  std::string name;
};

// Mirrors Instance in shaders/slang/basalt/basalt_common.slang.
struct InstanceRecord {
  Vec4 modelRow0, modelRow1, modelRow2;
  Vec4 normalRow0, normalRow1, normalRow2;
  Vec4 materialAndFlags;
};
static_assert(sizeof(InstanceRecord) == 112, "the instance layout must match the shader");

// Mirrors Light in shaders/slang/basalt/basalt_types.slang.
struct LightRecord {
  Vec4 position{0, 0, 0, 0};
  Vec4 color{1, 1, 1, 0};
  Vec4 direction{0, -1, 0, 1};
  Vec4 cone{1, 0, 0, 0};
};
static_assert(sizeof(LightRecord) == 64, "the light layout must match the shader");

struct SamplerDescription {
  VkFilter magFilter = VK_FILTER_LINEAR;
  VkFilter minFilter = VK_FILTER_LINEAR;
  VkSamplerMipmapMode mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  VkSamplerAddressMode addressU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  VkSamplerAddressMode addressV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  bool operator==(const SamplerDescription &) const = default;
};

class Scene {
public:
  Scene() = default;
  ~Scene();
  Scene(const Scene &) = delete;
  Scene &operator=(const Scene &) = delete;

  std::string sourcePath;
  std::vector<Primitive> primitives;
  std::vector<Material> materials;
  std::vector<Image> textures;
  std::vector<SamplerDescription> samplerDescriptions;
  std::vector<VkSampler> samplers;
  std::vector<LightRecord> lights;

  Buffer vertexBuffer;
  Buffer indexBuffer;
  std::uint32_t vertexCount = 0;
  std::uint32_t indexCount = 0;
  Aabb bounds;

  // Appended at load for the sun to cast onto; not part of `bounds`.
  int groundPrimitive = -1;

  std::uint32_t triangleCount = 0;
  std::size_t textureBytes = 0;

  bool empty() const { return primitives.empty(); }
  void destroy(const Context &context);
};

// Vertex class plus storage, and on a ray tracing device what an acceleration structure build reads.
VkBufferUsageFlags geometryBufferUsage(const Context &context, VkBufferUsageFlags base);

std::unique_ptr<Scene> loadGltf(const Context &context, Uploader &uploader, const std::string &path);

std::unique_ptr<Scene> createDefaultScene(const Context &context, Uploader &uploader);

// Called once the geometry is known, since it is sized from the bounds.
void appendGroundPlane(Scene &scene, std::vector<Vertex> &vertices,
                       std::vector<std::uint32_t> &indices);

} // namespace basalt
