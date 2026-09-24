// glTF loading: nodes flattened, primitives packed into one vertex and one index buffer.
#include "scene/Scene.h"

#include "core/Log.h"
#include "gpu/Uploader.h"

#include <cgltf.h>
#include <stb_image.h>

#include <algorithm>
#include <filesystem>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <unordered_map>

namespace basalt {
namespace {

Mat4 localTransform(const cgltf_node &node) {
  if (node.has_matrix) {
    Mat4 m;
    std::memcpy(&m.columns[0].x, node.matrix, sizeof(float) * 16);
    return m;
  }
  Mat4 result;
  if (node.has_translation)
    result = translation({node.translation[0], node.translation[1], node.translation[2]});
  if (node.has_rotation)
    result = result * rotation({node.rotation[0], node.rotation[1], node.rotation[2], node.rotation[3]});
  if (node.has_scale) result = result * scaling({node.scale[0], node.scale[1], node.scale[2]});
  return result;
}

std::vector<float> readFloats(const cgltf_accessor *accessor, cgltf_size components) {
  if (!accessor) return {};
  const cgltf_size count = cgltf_num_components(accessor->type);
  if (count != components)
    throw Error("a glTF accessor has " + std::to_string(count) + " components where " +
                std::to_string(components) + " were expected");
  std::vector<float> values(accessor->count * components);
  if (cgltf_accessor_unpack_floats(accessor, values.data(), values.size()) == 0)
    throw Error("a glTF accessor could not be read as floats");
  return values;
}

VkSamplerAddressMode addressMode(cgltf_int wrap) {
  switch (wrap) {
  case 33071: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  case 33648: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
  default: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  }
}

std::uint32_t packedSampler(const cgltf_texture_view &view) {
  const cgltf_sampler *sampler = view.texture ? view.texture->sampler : nullptr;
  auto wrap = [](cgltf_int value) {
    if (value == 33071) return 1u; // clamp
    if (value == 33648) return 2u; // mirrored repeat
    return 0u;                     // repeat
  };
  const std::uint32_t u = wrap(sampler ? sampler->wrap_s : 10497);
  const std::uint32_t v = wrap(sampler ? sampler->wrap_t : 10497);
  if (u != v) throw Error("path tracing currently requires matching glTF wrapS and wrapT modes");
  const cgltf_int mag = sampler ? sampler->mag_filter : 9729;
  const std::uint32_t nearest = mag == 9728 ? 1u : 0u;
  return u | (v << 2u) | (nearest << 4u);
}

std::uint32_t textureUv(const cgltf_texture_view &view, const std::string &material) {
  if (!view.texture) return 0u;
  if (view.texcoord < 0 || view.texcoord > 1)
    throw Error("material " + material + " uses unsupported UV set " + std::to_string(view.texcoord));
  return static_cast<std::uint32_t>(view.texcoord);
}

// The minification enumerant encodes the mip filter too.
VkFilter filterOf(cgltf_int filter, bool defaultLinear = true) {
  switch (filter) {
  case 9728: // NEAREST
  case 9984: // NEAREST_MIPMAP_NEAREST
  case 9986: // NEAREST_MIPMAP_LINEAR
    return VK_FILTER_NEAREST;
  case 9729:
  case 9985:
  case 9987:
    return VK_FILTER_LINEAR;
  default:
    return defaultLinear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
  }
}

VkSamplerMipmapMode mipmapModeOf(cgltf_int filter) {
  switch (filter) {
  case 9984: // NEAREST_MIPMAP_NEAREST
  case 9985: // LINEAR_MIPMAP_NEAREST
    return VK_SAMPLER_MIPMAP_MODE_NEAREST;
  default:
    return VK_SAMPLER_MIPMAP_MODE_LINEAR;
  }
}

std::vector<std::uint8_t> imageBytes(const cgltf_image &image, const std::filesystem::path &base) {
  if (image.buffer_view && image.buffer_view->buffer && image.buffer_view->buffer->data) {
    const auto *start = static_cast<const std::uint8_t *>(image.buffer_view->buffer->data) +
                        image.buffer_view->offset;
    return {start, start + image.buffer_view->size};
  }
  if (!image.uri) return {};
  const std::string uri = image.uri;
  if (uri.rfind("data:", 0) == 0) {
    const std::size_t comma = uri.find(',');
    if (comma == std::string::npos) return {};
    void *decoded = nullptr;
    cgltf_options options{};
    const std::string payload = uri.substr(comma + 1);
    // Padding is not data; asking the decoder for it fails.
    std::size_t padding = 0;
    while (padding < 2 && payload.size() > padding && payload[payload.size() - 1 - padding] == '=') ++padding;
    if (payload.size() * 3 / 4 < padding) return {};
    const cgltf_size size = payload.size() * 3 / 4 - padding;
    if (cgltf_load_buffer_base64(&options, size, payload.c_str(), &decoded) != cgltf_result_success)
      return {};
    std::vector<std::uint8_t> bytes(static_cast<std::uint8_t *>(decoded),
                                    static_cast<std::uint8_t *>(decoded) + size);
    free(decoded);
    return bytes;
  }
  std::string path;
  for (std::size_t i = 0; i < uri.size(); ++i) {
    if (uri[i] == '%' && i + 2 < uri.size()) {
      path.push_back(static_cast<char>(std::stoi(uri.substr(i + 1, 2), nullptr, 16)));
      i += 2;
    } else {
      path.push_back(uri[i]);
    }
  }
  std::ifstream file(base / path, std::ios::binary | std::ios::ate);
  if (!file) return {};
  const auto size = static_cast<std::size_t>(file.tellg());
  std::vector<std::uint8_t> bytes(size);
  file.seekg(0);
  file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(size));
  return bytes;
}

// Missing normals come from the triangle, missing tangents from the texture coordinates.
void generateNormals(std::vector<Vertex> &vertices, const std::vector<std::uint32_t> &indices,
                     std::size_t firstVertex) {
  for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
    Vertex &a = vertices[firstVertex + indices[i]];
    Vertex &b = vertices[firstVertex + indices[i + 1]];
    Vertex &c = vertices[firstVertex + indices[i + 2]];
    const Vec3 face = cross(b.position - a.position, c.position - a.position);
    a.normal += face;
    b.normal += face;
    c.normal += face;
  }
  for (std::size_t i = firstVertex; i < vertices.size(); ++i)
    vertices[i].normal = normalize(vertices[i].normal);
}

void generateTangents(std::vector<Vertex> &vertices, const std::vector<std::uint32_t> &indices,
                      std::size_t firstVertex) {
  std::vector<Vec3> tangents(vertices.size() - firstVertex, Vec3{0, 0, 0});
  std::vector<Vec3> bitangents(tangents.size(), Vec3{0, 0, 0});
  for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
    const std::uint32_t ia = indices[i], ib = indices[i + 1], ic = indices[i + 2];
    const Vertex &a = vertices[firstVertex + ia];
    const Vertex &b = vertices[firstVertex + ib];
    const Vertex &c = vertices[firstVertex + ic];
    const Vec3 edge1 = b.position - a.position, edge2 = c.position - a.position;
    const Vec2 delta1 = b.uv0 - a.uv0, delta2 = c.uv0 - a.uv0;
    const float determinant = delta1.x * delta2.y - delta2.x * delta1.y;
    if (std::abs(determinant) < 1e-12f) continue;
    const float r = 1.0f / determinant;
    const Vec3 tangent = (edge1 * delta2.y - edge2 * delta1.y) * r;
    const Vec3 bitangent = (edge2 * delta1.x - edge1 * delta2.x) * r;
    for (std::uint32_t index : {ia, ib, ic}) {
      tangents[index] += tangent;
      bitangents[index] += bitangent;
    }
  }
  for (std::size_t i = 0; i < tangents.size(); ++i) {
    Vertex &vertex = vertices[firstVertex + i];
    const Vec3 n = vertex.normal;
    Vec3 t = tangents[i] - n * dot(n, tangents[i]);
    if (dot(t, t) < 1e-16f) {
      // No usable tangent: any direction in the normal's plane will do.
      const Vec3 axis = std::abs(n.x) < 0.9f ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
      t = normalize(cross(axis, n));
    } else {
      t = normalize(t);
    }
    const float handedness = dot(cross(n, t), bitangents[i]) < 0.0f ? -1.0f : 1.0f;
    vertex.tangent = Vec4(t, handedness);
  }
}

} // namespace

std::unique_ptr<Scene> loadGltf(const Context &context, Uploader &uploader, const std::string &path) {
  cgltf_options options{};
  cgltf_data *data = nullptr;
  if (cgltf_parse_file(&options, path.c_str(), &data) != cgltf_result_success)
    throw Error("cannot parse the glTF file " + path);
  struct Guard {
    cgltf_data *data;
    ~Guard() { cgltf_free(data); }
  } guard{data};

  if (cgltf_load_buffers(&options, data, path.c_str()) != cgltf_result_success)
    throw Error("cannot load the buffers of " + path);
  for (cgltf_size i = 0; i < data->buffer_views_count; ++i)
    if (data->buffer_views[i].has_meshopt_compression) {
      logWarning("{} uses EXT_meshopt_compression, which this loader does not decode; its geometry will be wrong",
                 path);
      break;
    }
  if (cgltf_validate(data) != cgltf_result_success)
    throw Error("the glTF file " + path + " does not validate");

  auto scene = std::make_unique<Scene>();
  scene->sourcePath = path;
  const std::filesystem::path base = std::filesystem::path(path).parent_path();

  // Each image is uploaded once per colour space: base colour and emissive are sRGB.
  std::map<std::pair<const cgltf_image *, bool>, int> uploaded;
  auto textureIndex = [&](const cgltf_texture *texture, bool srgb) -> int {
    if (!texture) return -1;
    if (!texture->image && (texture->has_basisu || texture->has_webp)) {
      logWarning("texture {} is only available as {}, which this loader does not decode",
                 texture->name ? texture->name : "(unnamed)", texture->has_basisu ? "KTX2/Basis" : "WebP");
      return -1;
    }
    if (!texture->image) return -1;
    const auto key = std::make_pair(texture->image, srgb);
    if (const auto found = uploaded.find(key); found != uploaded.end()) return found->second;

    const std::vector<std::uint8_t> bytes = imageBytes(*texture->image, base);
    if (bytes.empty()) {
      logWarning("texture {} has no readable image data", texture->image->uri ? texture->image->uri : "(embedded)");
      return -1;
    }
    int width = 0, height = 0, channels = 0;
    stbi_uc *pixels = stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()), &width,
                                            &height, &channels, 4);
    if (!pixels) {
      logWarning("cannot decode texture {}: {}", texture->image->uri ? texture->image->uri : "(embedded)",
                 stbi_failure_reason() ? stbi_failure_reason() : "unknown format");
      return -1;
    }
    const VkDeviceSize size = static_cast<VkDeviceSize>(width) * height * 4;
    const std::string name = texture->image->name ? texture->image->name
                             : texture->image->uri ? texture->image->uri
                                                   : "image";
    Image image = uploader.createTexture(pixels, size, static_cast<std::uint32_t>(width),
                                         static_cast<std::uint32_t>(height),
                                         srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM,
                                         0, name);
    stbi_image_free(pixels);
    scene->textureBytes += static_cast<std::size_t>(size);
    scene->textures.push_back(std::move(image));
    const int index = static_cast<int>(scene->textures.size()) - 1;
    uploaded.emplace(key, index);
    return index;
  };

  auto samplerIndex = [&](const cgltf_sampler *sampler) -> int {
    SamplerDescription description;
    if (sampler) {
      description.magFilter = filterOf(sampler->mag_filter);
      description.minFilter = filterOf(sampler->min_filter);
      description.mipmapMode = mipmapModeOf(sampler->min_filter);
      description.addressU = addressMode(sampler->wrap_s);
      description.addressV = addressMode(sampler->wrap_t);
    }
    for (std::size_t i = 0; i < scene->samplerDescriptions.size(); ++i)
      if (scene->samplerDescriptions[i] == description) return static_cast<int>(i);
    scene->samplerDescriptions.push_back(description);
    return static_cast<int>(scene->samplerDescriptions.size()) - 1;
  };

  // Index zero is the default sampler.
  scene->samplerDescriptions.push_back({});
  std::unordered_map<const cgltf_material *, std::uint32_t> materialIndices;
  auto addMaterial = [&](const cgltf_material *source) -> std::uint32_t {
    if (const auto found = materialIndices.find(source); found != materialIndices.end())
      return found->second;

    Material material;
    material.name = source && source->name ? source->name : "material";
    // Through the view, so a texture transform is at least reported.
    auto viewTexture = [&](const cgltf_texture_view &view, bool srgb) -> int {
      if (view.has_transform)
        logWarning("material {} uses KHR_texture_transform, which is ignored; its texture coordinates will be off",
                   material.name);
      return textureIndex(view.texture, srgb);
    };
    if (source) {
      if (source->has_pbr_metallic_roughness) {
        const cgltf_pbr_metallic_roughness &pbr = source->pbr_metallic_roughness;
        material.uniforms.baseColorFactor = {pbr.base_color_factor[0], pbr.base_color_factor[1],
                                             pbr.base_color_factor[2], pbr.base_color_factor[3]};
        material.uniforms.factors.x = pbr.metallic_factor;
        material.uniforms.factors.y = pbr.roughness_factor;
        material.baseColor = viewTexture(pbr.base_color_texture, true);
        material.metallicRoughness = viewTexture(pbr.metallic_roughness_texture, false);
        if (pbr.base_color_texture.texture) material.sampler = samplerIndex(pbr.base_color_texture.texture->sampler);
        material.uniforms.texture[0] = textureUv(pbr.base_color_texture, material.name) |
            (textureUv(pbr.metallic_roughness_texture, material.name) << 8u) |
            (textureUv(source->emissive_texture, material.name) << 16u) |
            (textureUv(source->normal_texture, material.name) << 24u);
        material.uniforms.texture[1] = packedSampler(pbr.base_color_texture) |
            (packedSampler(pbr.metallic_roughness_texture) << 8u) |
            (packedSampler(source->emissive_texture) << 16u) |
            (packedSampler(source->normal_texture) << 24u);
      } else if (source->has_pbr_specular_glossiness) {
        // Specular-glossiness, approximated: diffuse as base colour, glossiness as roughness.
        const cgltf_pbr_specular_glossiness &sg = source->pbr_specular_glossiness;
        material.uniforms.baseColorFactor = {sg.diffuse_factor[0], sg.diffuse_factor[1],
                                             sg.diffuse_factor[2], sg.diffuse_factor[3]};
        material.uniforms.factors.x = 0.0f;
        material.uniforms.factors.y = 1.0f - sg.glossiness_factor;
        material.baseColor = viewTexture(sg.diffuse_texture, true);
        logWarning("material {} uses specular-glossiness, approximated as metallic-roughness",
                   material.name);
      }
      material.normal = viewTexture(source->normal_texture, false);
      material.occlusion = viewTexture(source->occlusion_texture, false);
      material.emissive = viewTexture(source->emissive_texture, true);
      material.uniforms.factors.z = source->normal_texture.scale != 0.0f ? source->normal_texture.scale : 1.0f;
      material.uniforms.factors.w = source->occlusion_texture.texture ? source->occlusion_texture.scale : 0.0f;
      if (!source->occlusion_texture.texture) material.uniforms.factors.w = 0.0f;
      const float strength = source->has_emissive_strength
                                 ? source->emissive_strength.emissive_strength
                                 : 1.0f;
      material.uniforms.emissive = {source->emissive_factor[0], source->emissive_factor[1],
                                    source->emissive_factor[2], strength};
      // V7 extensions: read here, rendered by the path tracers only.
      auto extension = [&](const cgltf_texture_view &view, bool srgb, int &texture) -> std::uint32_t {
        texture = viewTexture(view, srgb);
        return (textureUv(view, material.name) << 8u) | (packedSampler(view) << 16u);
      };
      if (source->has_transmission) {
        material.uniforms.transmission.x = source->transmission.transmission_factor;
        material.uniforms.extensionTextures[0] =
            extension(source->transmission.transmission_texture, false, material.transmissionTexture);
      }
      if (source->has_ior) {
        material.uniforms.transmission.y = source->ior.ior;
        material.uniforms.transmission.w = 1.0f;
      }
      if (source->has_volume) {
        material.uniforms.transmission.z = source->volume.thickness_factor;
        if (source->volume.attenuation_distance > 0.0f && source->volume.attenuation_distance < 1e30f)
          logWarning("material {} has KHR_materials_volume attenuation, ignored: there are no participating media",
                     material.name);
      }
      if (source->has_clearcoat) {
        material.uniforms.clearcoat = {source->clearcoat.clearcoat_factor,
                                       source->clearcoat.clearcoat_roughness_factor,
                                       source->clearcoat.clearcoat_normal_texture.texture
                                           ? source->clearcoat.clearcoat_normal_texture.scale : 1.0f, 0.0f};
        material.uniforms.extensionTextures[1] =
            extension(source->clearcoat.clearcoat_texture, false, material.clearcoatTexture);
        material.uniforms.extensionTextures[2] =
            extension(source->clearcoat.clearcoat_roughness_texture, false, material.clearcoatRoughnessTexture);
        material.uniforms.extensionTextures[3] =
            extension(source->clearcoat.clearcoat_normal_texture, false, material.clearcoatNormalTexture);
      }
      material.doubleSided = source->double_sided != 0;
      switch (source->alpha_mode) {
      case cgltf_alpha_mode_mask: material.alphaMode = AlphaMode::Mask; break;
      case cgltf_alpha_mode_blend: material.alphaMode = AlphaMode::Blend; break;
      default: material.alphaMode = AlphaMode::Opaque; break;
      }
      material.uniforms.alpha = {source->alpha_cutoff, static_cast<float>(material.alphaMode),
                                 material.doubleSided ? 1.0f : 0.0f, 0.0f};
      // The shader reads occlusion from uv1; other sets are reported rather than drawn wrong.
      if (source->occlusion_texture.texture && source->occlusion_texture.texcoord != 1 &&
          source->occlusion_texture.texcoord != 0)
        logWarning("material {} reads occlusion from UV set {}, which is not loaded", material.name,
                   source->occlusion_texture.texcoord);
    }
    scene->materials.push_back(material);
    const auto index = static_cast<std::uint32_t>(scene->materials.size()) - 1;
    materialIndices.emplace(source, index);
    return index;
  };

  std::vector<Vertex> vertices;
  std::vector<std::uint32_t> indices;

  auto addPrimitive = [&](const cgltf_primitive &primitive, const Mat4 &transform,
                          const std::string &nodeName) {
    if (primitive.type != cgltf_primitive_type_triangles) {
      logWarning("primitive of {} is not a triangle list and was skipped", nodeName);
      return;
    }
    // cgltf parses Draco but does not decode it; the accessors read as zeros.
    if (primitive.has_draco_mesh_compression) {
      logWarning("primitive of {} is Draco-compressed, which this loader does not decode; skipped", nodeName);
      return;
    }
    const cgltf_accessor *positionAccessor = nullptr;
    const cgltf_accessor *normalAccessor = nullptr;
    const cgltf_accessor *tangentAccessor = nullptr;
    const cgltf_accessor *uv0Accessor = nullptr;
    const cgltf_accessor *uv1Accessor = nullptr;
    const cgltf_accessor *colorAccessor = nullptr;
    for (cgltf_size a = 0; a < primitive.attributes_count; ++a) {
      const cgltf_attribute &attribute = primitive.attributes[a];
      switch (attribute.type) {
      case cgltf_attribute_type_position: positionAccessor = attribute.data; break;
      case cgltf_attribute_type_normal: normalAccessor = attribute.data; break;
      case cgltf_attribute_type_tangent: tangentAccessor = attribute.data; break;
      case cgltf_attribute_type_texcoord:
        if (attribute.index == 0) uv0Accessor = attribute.data;
        else if (attribute.index == 1) uv1Accessor = attribute.data;
        break;
      case cgltf_attribute_type_color:
        if (attribute.index == 0) colorAccessor = attribute.data;
        break;
      default: break;
      }
    }
    if (!positionAccessor) return;

    const std::size_t firstVertex = vertices.size();
    const std::size_t firstIndex = indices.size();
    const std::vector<float> positions = readFloats(positionAccessor, 3);
    const std::vector<float> normals = readFloats(normalAccessor, 3);
    const std::vector<float> tangents = readFloats(tangentAccessor, 4);
    const std::vector<float> uv0 = readFloats(uv0Accessor, 2);
    const std::vector<float> uv1 = readFloats(uv1Accessor, 2);
    const std::size_t count = positionAccessor->count;
    std::vector<float> colors(count * 4u, 1.0f);
    if (colorAccessor) {
      const cgltf_size components = cgltf_num_components(colorAccessor->type);
      if (components != 3 && components != 4) throw Error("COLOR_0 must have three or four components");
      for (std::size_t i = 0; i < count; ++i)
        if (!cgltf_accessor_read_float(colorAccessor, i, colors.data() + i * 4u, 4u))
          throw Error("COLOR_0 could not be read as floats");
    }
    Aabb localBounds;
    for (std::size_t i = 0; i < count; ++i) {
      Vertex vertex;
      vertex.position = {positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]};
      if (!normals.empty()) vertex.normal = {normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]};
      if (!tangents.empty())
        vertex.tangent = {tangents[i * 4], tangents[i * 4 + 1], tangents[i * 4 + 2], tangents[i * 4 + 3]};
      if (!uv0.empty()) vertex.uv0 = {uv0[i * 2], uv0[i * 2 + 1]};
      if (!uv1.empty()) vertex.uv1 = {uv1[i * 2], uv1[i * 2 + 1]};
      else vertex.uv1 = vertex.uv0;
      vertex.color = {colors[i * 4], colors[i * 4 + 1], colors[i * 4 + 2], colors[i * 4 + 3]};
      localBounds.add(vertex.position);
      vertices.push_back(vertex);
    }

    std::vector<std::uint32_t> localIndices;
    if (primitive.indices) {
      localIndices.resize(primitive.indices->count);
      for (cgltf_size i = 0; i < primitive.indices->count; ++i)
        localIndices[i] = static_cast<std::uint32_t>(cgltf_accessor_read_index(primitive.indices, i));
    } else {
      localIndices.resize(count);
      for (std::size_t i = 0; i < count; ++i) localIndices[i] = static_cast<std::uint32_t>(i);
    }

    if (normals.empty()) generateNormals(vertices, localIndices, firstVertex);
    if (tangents.empty()) generateTangents(vertices, localIndices, firstVertex);

    indices.insert(indices.end(), localIndices.begin(), localIndices.end());

    Primitive record;
    record.firstIndex = static_cast<std::uint32_t>(firstIndex);
    record.indexCount = static_cast<std::uint32_t>(localIndices.size());
    record.vertexOffset = static_cast<std::int32_t>(firstVertex);
    record.material = addMaterial(primitive.material);
    record.bounds = localBounds;
    record.transform = transform;
    record.worldBounds = localBounds.transformed(transform);
    record.name = nodeName;
    scene->bounds.add(record.worldBounds);
    scene->primitives.push_back(record);
  };

  // Depth first with accumulated transforms; a file with no scene draws every root.
  std::function<void(const cgltf_node &, const Mat4 &)> visit = [&](const cgltf_node &node,
                                                                   const Mat4 &parent) {
    const Mat4 world = parent * localTransform(node);
    const std::string name = node.name ? node.name : "node";
    if (node.mesh)
      for (cgltf_size p = 0; p < node.mesh->primitives_count; ++p)
        addPrimitive(node.mesh->primitives[p], world, name);

    if (node.light) {
      LightRecord light;
      const Vec3 position = transformPoint(world, {0, 0, 0});
      // A glTF light points down its local negative Z.
      const Vec3 direction = normalize(transformDirection(world, {0, 0, -1}));
      const float range = node.light->range > 0.0f ? node.light->range : 0.0f;
      light.position = Vec4(position, range);
      light.color = {node.light->color[0], node.light->color[1], node.light->color[2],
                     node.light->intensity};
      light.direction = Vec4(direction, std::cos(node.light->spot_inner_cone_angle));
      const bool spot = node.light->type == cgltf_light_type_spot;
      light.cone = {std::cos(node.light->spot_outer_cone_angle), spot ? 1.0f : 0.0f, 0.0f, 0.0f};
      if (node.light->type == cgltf_light_type_directional) {
        logWarning("the directional light {} is not loaded; the sun is controlled by the UI", name);
      } else {
        // Candela to radiance, scaled as the reference viewer does.
        light.color = {light.color.x, light.color.y, light.color.z, node.light->intensity / 683.0f};
        scene->lights.push_back(light);
      }
    }

    for (cgltf_size c = 0; c < node.children_count; ++c) visit(*node.children[c], world);
  };

  const cgltf_scene *active = data->scene ? data->scene : (data->scenes_count ? data->scenes : nullptr);
  if (active) {
    for (cgltf_size n = 0; n < active->nodes_count; ++n) visit(*active->nodes[n], Mat4{});
  } else {
    for (cgltf_size n = 0; n < data->nodes_count; ++n)
      if (!data->nodes[n].parent) visit(data->nodes[n], Mat4{});
  }

  if (vertices.empty() || indices.empty())
    throw Error("the glTF file " + path + " holds no triangle geometry");
  if (scene->materials.empty()) addMaterial(nullptr);
  appendGroundPlane(*scene, vertices, indices);

  scene->vertexBuffer = uploader.createBuffer(vertices.data(), vertices.size() * sizeof(Vertex),
                                              geometryBufferUsage(context, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT),
                                              "scene.vertices");
  scene->indexBuffer = uploader.createBuffer(indices.data(), indices.size() * sizeof(std::uint32_t),
                                             geometryBufferUsage(context, VK_BUFFER_USAGE_INDEX_BUFFER_BIT),
                                             "scene.indices");
  scene->vertexCount = static_cast<std::uint32_t>(vertices.size());
  scene->indexCount = static_cast<std::uint32_t>(indices.size());
  scene->triangleCount = scene->indexCount / 3;

  if (scene->samplerDescriptions.empty()) scene->samplerDescriptions.push_back({});
  for (const SamplerDescription &description : scene->samplerDescriptions) {
    VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    info.magFilter = description.magFilter;
    info.minFilter = description.minFilter;
    info.mipmapMode = description.mipmapMode;
    info.addressModeU = description.addressU;
    info.addressModeV = description.addressV;
    info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.maxLod = VK_LOD_CLAMP_NONE;
    info.anisotropyEnable = context.properties.limits.maxSamplerAnisotropy > 1.0f ? VK_TRUE : VK_FALSE;
    info.maxAnisotropy = std::min(8.0f, context.properties.limits.maxSamplerAnisotropy);
    VkSampler sampler = VK_NULL_HANDLE;
    check(vkCreateSampler(context.device, &info, nullptr, &sampler), "vkCreateSampler");
    scene->samplers.push_back(sampler);
  }

  logInfo("loaded {}: {} primitives, {} triangles, {} materials, {} textures, {} lights",
          std::filesystem::path(path).filename().string(), scene->primitives.size(),
          scene->triangleCount, scene->materials.size(), scene->textures.size(),
          scene->lights.size());
  return scene;
}

} // namespace basalt
