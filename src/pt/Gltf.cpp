#include "pt/Gltf.h"
#include "pt/Tables.h"

#include <cgltf.h>
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace pt {
namespace {

using Matrix = std::array<float, 16>; // column major, as glTF stores it

void hashBytes(std::uint64_t &hash, const void *data, std::size_t size) {
  const auto *bytes = static_cast<const std::uint8_t *>(data);
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ull;
  }
}

template <class T> void hashVector(std::uint64_t &hash, const std::vector<T> &values) {
  if (!values.empty()) hashBytes(hash, values.data(), values.size() * sizeof(T));
}

std::vector<float> floats(const cgltf_accessor *accessor, cgltf_size components) {
  if (!accessor) return {};
  if (cgltf_num_components(accessor->type) != components) throw std::runtime_error("unexpected accessor width");
  std::vector<float> result(accessor->count * components);
  if (!cgltf_accessor_unpack_floats(accessor, result.data(), result.size()))
    throw std::runtime_error("could not unpack a glTF accessor");
  return result;
}

uint packedSampler(const cgltf_texture_view &view) {
  const cgltf_sampler *sampler = view.texture ? view.texture->sampler : nullptr;
  auto wrap = [](cgltf_int value) {
    if (value == 33071) return 1u;
    if (value == 33648) return 2u;
    return 0u;
  };
  const uint u = wrap(sampler ? sampler->wrap_s : 10497);
  const uint v = wrap(sampler ? sampler->wrap_t : 10497);
  if (u != v) throw std::runtime_error("path tracing currently requires matching glTF wrapS and wrapT modes");
  return u | (v << 2u) | ((sampler && sampler->mag_filter == 9728 ? 1u : 0u) << 4u);
}

uint textureUv(const cgltf_texture_view &view) {
  if (!view.texture) return 0u;
  if (view.texcoord < 0 || view.texcoord > 1)
    throw std::runtime_error("headless CPU loader supports only TEXCOORD_0 and TEXCOORD_1");
  return static_cast<uint>(view.texcoord);
}

std::vector<std::uint8_t> imageBytes(const cgltf_image &image, const std::filesystem::path &base) {
  if (image.buffer_view && image.buffer_view->buffer && image.buffer_view->buffer->data) {
    const auto *start = static_cast<const std::uint8_t *>(image.buffer_view->buffer->data) + image.buffer_view->offset;
    return {start, start + image.buffer_view->size};
  }
  if (!image.uri) return {};
  const std::string uri = image.uri;
  if (uri.rfind("data:", 0) == 0) {
    const std::size_t comma = uri.find(',');
    if (comma == std::string::npos) return {};
    const std::string payload = uri.substr(comma + 1);
    std::size_t padding = 0;
    while (padding < 2 && padding < payload.size() && payload[payload.size() - 1 - padding] == '=') ++padding;
    const cgltf_size size = payload.size() * 3 / 4 - padding;
    void *decoded = nullptr;
    cgltf_options options{};
    if (cgltf_load_buffer_base64(&options, size, payload.c_str(), &decoded) != cgltf_result_success) return {};
    std::vector<std::uint8_t> result(static_cast<std::uint8_t *>(decoded),
                                     static_cast<std::uint8_t *>(decoded) + size);
    std::free(decoded);
    return result;
  }
  std::string decodedUri;
  for (std::size_t i = 0; i < uri.size(); ++i) {
    if (uri[i] == '%' && i + 2 < uri.size()) {
      decodedUri.push_back(static_cast<char>(std::stoi(uri.substr(i + 1, 2), nullptr, 16)));
      i += 2;
    } else decodedUri.push_back(uri[i]);
  }
  std::ifstream input(base / decodedUri, std::ios::binary | std::ios::ate);
  if (!input) return {};
  const std::size_t size = static_cast<std::size_t>(input.tellg());
  std::vector<std::uint8_t> result(size);
  input.seekg(0);
  input.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(size));
  return result;
}

float3 point(const Matrix &m, float3 p) {
  return float3(m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
                m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
                m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14]);
}

float3 direction(const Matrix &m, float3 p) {
  return float3(m[0] * p.x + m[4] * p.y + m[8] * p.z,
                m[1] * p.x + m[5] * p.y + m[9] * p.z,
                m[2] * p.x + m[6] * p.y + m[10] * p.z);
}

void setTransforms(TraceInstance &instance, const Matrix &m) {
  const float a = m[0], b = m[4], c = m[8], d = m[1], e = m[5], f = m[9], g = m[2], h = m[6], i = m[10];
  const float determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (!std::isfinite(determinant) || std::abs(determinant) < 1e-12f)
    throw std::runtime_error("glTF instance has a singular transform");
  const float k = 1.0f / determinant;
  const float3 r0((e * i - f * h) * k, (c * h - b * i) * k, (b * f - c * e) * k);
  const float3 r1((f * g - d * i) * k, (a * i - c * g) * k, (c * d - a * f) * k);
  const float3 r2((d * h - e * g) * k, (b * g - a * h) * k, (a * e - b * d) * k);
  const float3 translation(m[12], m[13], m[14]);
  instance.objectToWorld0 = float4(a, b, c, translation.x);
  instance.objectToWorld1 = float4(d, e, f, translation.y);
  instance.objectToWorld2 = float4(g, h, i, translation.z);
  instance.worldToObject0 = float4(r0, -dot(r0, translation));
  instance.worldToObject1 = float4(r1, -dot(r1, translation));
  instance.worldToObject2 = float4(r2, -dot(r2, translation));
  instance.normalToWorld0 = float4(r0.x, r1.x, r2.x, 0.0f);
  instance.normalToWorld1 = float4(r0.y, r1.y, r2.y, 0.0f);
  instance.normalToWorld2 = float4(r0.z, r1.z, r2.z, 0.0f);
  if (determinant < 0.0f) instance.flags |= kInstanceMirrored;
}

} // namespace

LoadedGltf loadGltf(const std::string &path, unsigned buildThreads) {
  cgltf_options options{};
  cgltf_data *raw = nullptr;
  if (cgltf_parse_file(&options, path.c_str(), &raw) != cgltf_result_success)
    throw std::runtime_error("cannot parse glTF: " + path);
  struct Guard { cgltf_data *data; ~Guard() { cgltf_free(data); } } guard{raw};
  if (cgltf_load_buffers(&options, raw, path.c_str()) != cgltf_result_success)
    throw std::runtime_error("cannot load glTF buffers: " + path);
  if (cgltf_validate(raw) != cgltf_result_success) throw std::runtime_error("glTF validation failed: " + path);

  LoadedGltf loaded;
  loaded.boundsMin = float3(std::numeric_limits<float>::max());
  loaded.boundsMax = float3(-std::numeric_limits<float>::max());
  const std::filesystem::path base = std::filesystem::path(path).parent_path();
  uint nextSlot = 2;
  std::map<std::pair<const cgltf_image *, bool>, uint> textureSlots;
  auto textureSlot = [&](const cgltf_texture_view &view, bool srgb, uint fallback) {
    if (!view.texture || !view.texture->image) return fallback;
    const auto key = std::make_pair(view.texture->image, srgb);
    if (const auto found = textureSlots.find(key); found != textureSlots.end()) return found->second;
    if (nextSlot >= kHitTextureSlots) throw std::runtime_error("glTF needs more than 118 decoded texture slots");
    const auto bytes = imageBytes(*view.texture->image, base);
    int width = 0, height = 0, channels = 0;
    stbi_uc *pixels = bytes.empty() ? nullptr : stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()),
                                                                     &width, &height, &channels, 4);
    if (!pixels) throw std::runtime_error("cannot decode a glTF texture");
    HostTexture texture;
    texture.width = static_cast<uint>(width);
    texture.height = static_cast<uint>(height);
    texture.srgb = srgb;
    texture.texels.resize(static_cast<std::size_t>(width) * height);
    std::memcpy(texture.texels.data(), pixels, texture.texels.size() * sizeof(std::uint32_t));
    stbi_image_free(pixels);
    buildMipChain(texture);  // ray-cone filtering
    const uint slot = nextSlot++;
    loaded.scene.textures.slots[slot] = static_cast<uint>(loaded.scene.textures.textures.size());
    loaded.scene.textures.textures.push_back(std::move(texture));
    textureSlots.emplace(key, slot);
    return slot;
  };

  std::unordered_map<const cgltf_material *, std::pair<uint, uint>> materialIndices;
  auto material = [&](const cgltf_material *source) {
    if (const auto found = materialIndices.find(source); found != materialIndices.end()) return found->second;
    Material result{};
    result.baseColorFactor = float4(1.0f);
    result.emissive = float4(0.0f, 0.0f, 0.0f, 1.0f);
    result.factors = float4(1.0f);
    result.alpha = float4(0.5f, 0.0f, 0.0f, 0.0f);
    result.transmission = float4(0.0f, 1.5f, 0.0f, 0.0f);
    result.clearcoat = float4(0.0f, 0.0f, 1.0f, 0.0f);
    result.extensionTextures = uint4(0u, 0u, 0u, 1u);
    uint baseSlot = 0, metallicSlot = 0, emissiveSlot = 0, normalSlot = 1;
    if (source) {
      if (!source->has_pbr_metallic_roughness)
        throw std::runtime_error("headless CPU loader requires metallic-roughness materials");
      const auto &pbr = source->pbr_metallic_roughness;
      result.baseColorFactor = float4(pbr.base_color_factor[0], pbr.base_color_factor[1],
                                      pbr.base_color_factor[2], pbr.base_color_factor[3]);
      result.factors.x = pbr.metallic_factor;
      result.factors.y = pbr.roughness_factor;
      baseSlot = textureSlot(pbr.base_color_texture, true, 0);
      metallicSlot = textureSlot(pbr.metallic_roughness_texture, false, 0);
      normalSlot = textureSlot(source->normal_texture, false, 1);
      emissiveSlot = textureSlot(source->emissive_texture, true, 0);
      result.factors.z = source->normal_texture.texture ? source->normal_texture.scale : 1.0f;
      const float strength = source->has_emissive_strength ? source->emissive_strength.emissive_strength : 1.0f;
      result.emissive = float4(source->emissive_factor[0], source->emissive_factor[1],
                               source->emissive_factor[2], strength);
      result.texture = uint4(textureUv(pbr.base_color_texture) |
                                 (textureUv(pbr.metallic_roughness_texture) << 8u) |
                                 (textureUv(source->emissive_texture) << 16u) |
                                 (textureUv(source->normal_texture) << 24u),
                             packedSampler(pbr.base_color_texture) |
                                 (packedSampler(pbr.metallic_roughness_texture) << 8u) |
                                 (packedSampler(source->emissive_texture) << 16u) |
                                 (packedSampler(source->normal_texture) << 24u),
                             0u, 0u);
      uint mode = 0;
      if (source->alpha_mode == cgltf_alpha_mode_mask) mode = 1;
      if (source->alpha_mode == cgltf_alpha_mode_blend) mode = 2;
      result.alpha = float4(source->alpha_cutoff, static_cast<float>(mode), source->double_sided ? 1.0f : 0.0f, 0.0f);
      // Transmission, IOR, volume thickness and clearcoat extensions.
      auto extension = [&](const cgltf_texture_view &view, uint fallback) {
        return textureSlot(view, false, fallback) | (textureUv(view) << 8u) | (packedSampler(view) << 16u);
      };
      if (source->has_transmission) {
        result.transmission.x = source->transmission.transmission_factor;
        result.extensionTextures.x = extension(source->transmission.transmission_texture, 0);
      }
      if (source->has_ior) result.transmission = float4(result.transmission.x, source->ior.ior, result.transmission.z, 1.0f);
      if (source->has_volume) result.transmission.z = source->volume.thickness_factor;
      if (source->has_clearcoat) {
        result.clearcoat = float4(source->clearcoat.clearcoat_factor, source->clearcoat.clearcoat_roughness_factor,
                                  source->clearcoat.clearcoat_normal_texture.texture
                                      ? source->clearcoat.clearcoat_normal_texture.scale : 1.0f, 0.0f);
        result.extensionTextures.y = extension(source->clearcoat.clearcoat_texture, 0);
        result.extensionTextures.z = extension(source->clearcoat.clearcoat_roughness_texture, 0);
        result.extensionTextures.w = extension(source->clearcoat.clearcoat_normal_texture, 1);
      }
    }
    const uint index = static_cast<uint>(loaded.frame.materials.size());
    loaded.frame.materials.push_back(result);
    const std::pair<uint, uint> value{index, baseSlot | (metallicSlot << 8u) |
                                            (emissiveSlot << 16u) | (normalSlot << 24u)};
    materialIndices.emplace(source, value);
    return value;
  };

  std::function<void(const cgltf_node &)> visit = [&](const cgltf_node &node) {
    Matrix transform{};
    cgltf_node_transform_world(&node, transform.data());
    if (node.mesh) for (cgltf_size p = 0; p < node.mesh->primitives_count; ++p) {
      const cgltf_primitive &primitive = node.mesh->primitives[p];
      if (primitive.type != cgltf_primitive_type_triangles || primitive.has_draco_mesh_compression) continue;
      const cgltf_accessor *positions = nullptr, *normals = nullptr, *tangents = nullptr, *uv0 = nullptr, *uv1 = nullptr,
                           *colors = nullptr;
      for (cgltf_size a = 0; a < primitive.attributes_count; ++a) {
        const auto &attribute = primitive.attributes[a];
        if (attribute.type == cgltf_attribute_type_position) positions = attribute.data;
        else if (attribute.type == cgltf_attribute_type_normal) normals = attribute.data;
        else if (attribute.type == cgltf_attribute_type_tangent) tangents = attribute.data;
        else if (attribute.type == cgltf_attribute_type_texcoord && attribute.index == 0) uv0 = attribute.data;
        else if (attribute.type == cgltf_attribute_type_texcoord && attribute.index == 1) uv1 = attribute.data;
        else if (attribute.type == cgltf_attribute_type_color && attribute.index == 0) colors = attribute.data;
      }
      if (!positions) continue;
      const auto pos = floats(positions, 3), nor = floats(normals, 3), tan = floats(tangents, 4);
      const auto tex0 = floats(uv0, 2), tex1 = floats(uv1, 2);
      std::vector<float> color(positions->count * 4u, 1.0f);
      if (colors) {
        const cgltf_size components = cgltf_num_components(colors->type);
        if (components != 3 && components != 4)
          throw std::runtime_error("COLOR_0 must have three or four components");
        for (cgltf_size v = 0; v < positions->count; ++v)
          if (!cgltf_accessor_read_float(colors, v, color.data() + v * 4u, 4u))
            throw std::runtime_error("could not unpack COLOR_0");
      }
      const uint firstVertex = static_cast<uint>(loaded.scene.vertices.size() / kVertexFloats);
      for (cgltf_size v = 0; v < positions->count; ++v) {
        const float values[kVertexFloats] = {
          pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2],
          nor.empty() ? 0.0f : nor[v * 3], nor.empty() ? 0.0f : nor[v * 3 + 1], nor.empty() ? 0.0f : nor[v * 3 + 2],
          tan.empty() ? 1.0f : tan[v * 4], tan.empty() ? 0.0f : tan[v * 4 + 1],
          tan.empty() ? 0.0f : tan[v * 4 + 2], tan.empty() ? 1.0f : tan[v * 4 + 3],
          tex0.empty() ? 0.0f : tex0[v * 2], tex0.empty() ? 0.0f : tex0[v * 2 + 1],
          tex1.empty() ? (tex0.empty() ? 0.0f : tex0[v * 2]) : tex1[v * 2],
          tex1.empty() ? (tex0.empty() ? 0.0f : tex0[v * 2 + 1]) : tex1[v * 2 + 1],
          color[v * 4], color[v * 4 + 1], color[v * 4 + 2], color[v * 4 + 3]};
        loaded.scene.vertices.insert(loaded.scene.vertices.end(), values, values + kVertexFloats);
        const float3 world = point(transform, float3(values[0], values[1], values[2]));
        loaded.boundsMin = min(loaded.boundsMin, world);
        loaded.boundsMax = max(loaded.boundsMax, world);
      }
      const uint firstIndex = static_cast<uint>(loaded.scene.indices.size());
      if (primitive.indices) for (cgltf_size n = 0; n < primitive.indices->count; ++n)
        loaded.scene.indices.push_back(static_cast<uint>(cgltf_accessor_read_index(primitive.indices, n)));
      else for (uint n = 0; n < positions->count; ++n) loaded.scene.indices.push_back(n);
      const uint count = static_cast<uint>(loaded.scene.indices.size()) - firstIndex;
      if (count < 3 || count % 3 != 0) throw std::runtime_error("glTF triangle index count is invalid");
      if (nor.empty()) {
        for (uint n = 0; n < count; n += 3) {
          float *a = loaded.scene.vertices.data() + (firstVertex + loaded.scene.indices[firstIndex + n]) * kVertexFloats;
          float *b = loaded.scene.vertices.data() + (firstVertex + loaded.scene.indices[firstIndex + n + 1]) * kVertexFloats;
          float *c = loaded.scene.vertices.data() + (firstVertex + loaded.scene.indices[firstIndex + n + 2]) * kVertexFloats;
          const float3 face = cross(float3(b[0] - a[0], b[1] - a[1], b[2] - a[2]),
                                    float3(c[0] - a[0], c[1] - a[1], c[2] - a[2]));
          for (float *vertex : {a, b, c}) {
            vertex[3] += face.x; vertex[4] += face.y; vertex[5] += face.z;
          }
        }
        for (uint v = 0; v < positions->count; ++v) {
          float *vertex = loaded.scene.vertices.data() + (firstVertex + v) * kVertexFloats;
          float3 normal(vertex[3], vertex[4], vertex[5]);
          normal = length(normal) > 1e-12f ? normalize(normal) : float3(0.0f, 1.0f, 0.0f);
          vertex[3] = normal.x; vertex[4] = normal.y; vertex[5] = normal.z;
        }
      }
      if (tan.empty()) {
        std::vector<float3> tangent(positions->count, float3(0.0f));
        std::vector<float3> bitangent(positions->count, float3(0.0f));
        for (uint n = 0; n < count; n += 3) {
          const uint ia = loaded.scene.indices[firstIndex + n];
          const uint ib = loaded.scene.indices[firstIndex + n + 1];
          const uint ic = loaded.scene.indices[firstIndex + n + 2];
          const float *a = loaded.scene.vertices.data() + (firstVertex + ia) * kVertexFloats;
          const float *b = loaded.scene.vertices.data() + (firstVertex + ib) * kVertexFloats;
          const float *c = loaded.scene.vertices.data() + (firstVertex + ic) * kVertexFloats;
          const float3 edge1(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
          const float3 edge2(c[0] - a[0], c[1] - a[1], c[2] - a[2]);
          const float du1 = b[10] - a[10], dv1 = b[11] - a[11];
          const float du2 = c[10] - a[10], dv2 = c[11] - a[11];
          const float determinant = du1 * dv2 - du2 * dv1;
          if (std::abs(determinant) < 1e-12f) continue;
          const float inverse = 1.0f / determinant;
          const float3 t = (edge1 * dv2 - edge2 * dv1) * inverse;
          const float3 bt = (edge2 * du1 - edge1 * du2) * inverse;
          for (uint index : {ia, ib, ic}) { tangent[index] += t; bitangent[index] += bt; }
        }
        for (uint v = 0; v < positions->count; ++v) {
          float *vertex = loaded.scene.vertices.data() + (firstVertex + v) * kVertexFloats;
          const float3 normal(vertex[3], vertex[4], vertex[5]);
          float3 t = tangent[v] - normal * dot(normal, tangent[v]);
          if (length(t) <= 1e-8f) {
            const float3 axis = std::abs(normal.x) < 0.9f ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
            t = normalize(cross(axis, normal));
          } else t = normalize(t);
          vertex[6] = t.x; vertex[7] = t.y; vertex[8] = t.z;
          vertex[9] = dot(cross(normal, t), bitangent[v]) < 0.0f ? -1.0f : 1.0f;
        }
      }
      const auto [materialIndex, slots] = material(primitive.material);
      TraceInstance instance{};
      setTransforms(instance, transform);
      instance.firstIndex = firstIndex;
      instance.vertexOffset = firstVertex;
      instance.material = materialIndex;
      instance.slots = slots;
      const uint mode = static_cast<uint>(loaded.frame.materials[materialIndex].alpha.y);
      instance.mask = mode == 2 ? kRayMaskBlended : kRayMaskScene;
      instance.flags = (instance.flags & kInstanceMirrored) | (mode == 1 ? kInstanceMasked : 0u) |
                       (mode == 2 ? kInstanceBlended : 0u) |
                       (loaded.frame.materials[materialIndex].alpha.z > 0.5f ? kInstanceDoubleSided : 0u);
      loaded.scene.instances.push_back(instance);
      loaded.scene.triangleCounts.push_back(count / 3);
    }
    if (node.light && node.light->type != cgltf_light_type_directional) {
      Light light{};
      const float3 location = point(transform, float3(0.0f));
      const float3 axis = normalize(direction(transform, float3(0.0f, 0.0f, -1.0f)));
      light.position = float4(location, node.light->range > 0.0f ? node.light->range : 0.0f);
      light.color = float4(node.light->color[0], node.light->color[1], node.light->color[2],
                           node.light->intensity / 683.0f);
      light.direction = float4(axis, std::cos(node.light->spot_inner_cone_angle));
      light.cone = float4(std::cos(node.light->spot_outer_cone_angle),
                          node.light->type == cgltf_light_type_spot ? 1.0f : 0.0f, 0.0f, 0.0f);
      loaded.frame.lights.push_back(light);
    }
    for (cgltf_size c = 0; c < node.children_count; ++c) visit(*node.children[c]);
  };
  const cgltf_scene *active = raw->scene ? raw->scene : (raw->scenes_count ? raw->scenes : nullptr);
  if (active) for (cgltf_size n = 0; n < active->nodes_count; ++n) visit(*active->nodes[n]);
  else for (cgltf_size n = 0; n < raw->nodes_count; ++n) if (!raw->nodes[n].parent) visit(raw->nodes[n]);
  if (loaded.scene.instances.empty()) throw std::runtime_error("glTF contains no supported triangle geometry");

  loaded.scene.environment.width = 1;
  loaded.scene.environment.height = 1;
  loaded.scene.environment.texels = {0.2f, 0.2f, 0.2f, 1.0f};
  buildEnvironmentDistribution(loaded.scene.environment.texels.data(), 1, 1,
                               loaded.scene.distribution, loaded.scene.distributionInfo);
  loaded.scene.specularAlbedo = buildSpecularAlbedoTable();
  loaded.scene.emissiveTriangles = buildEmissiveTriangles(loaded.scene.vertices, loaded.scene.indices,
      loaded.scene.instances, loaded.scene.triangleCounts, loaded.frame.materials);
  const unsigned threads = buildThreads ? buildThreads : std::max(1u, std::thread::hardware_concurrency());
  loaded.scene.bvhStatistics = buildBvh(loaded.scene.vertices, loaded.scene.indices, loaded.scene.instances,
                                        loaded.scene.triangleCounts, loaded.scene.bvh, threads);
  loaded.contentHash = 14695981039346656037ull;
  hashVector(loaded.contentHash, loaded.scene.vertices);
  hashVector(loaded.contentHash, loaded.scene.indices);
  hashVector(loaded.contentHash, loaded.scene.instances);
  hashVector(loaded.contentHash, loaded.frame.materials);
  hashVector(loaded.contentHash, loaded.frame.lights);
  for (const HostTexture &texture : loaded.scene.textures.textures) {
    hashBytes(loaded.contentHash, &texture.width, sizeof(texture.width));
    hashBytes(loaded.contentHash, &texture.height, sizeof(texture.height));
    hashVector(loaded.contentHash, texture.texels);
  }
  return loaded;
}

void loadEnvironment(CpuScene &scene, const std::string &path) {
  int width = 0, height = 0, channels = 0;
  float *pixels = stbi_loadf(path.c_str(), &width, &height, &channels, 4);
  if (!pixels) throw std::runtime_error("cannot decode HDR environment: " + path);
  if (width <= 0 || height <= 0) {
    stbi_image_free(pixels);
    throw std::runtime_error("HDR environment has invalid dimensions: " + path);
  }
  const std::size_t count = static_cast<std::size_t>(width) * height * 4;
  scene.environment.width = static_cast<uint>(width);
  scene.environment.height = static_cast<uint>(height);
  scene.environment.texels.assign(pixels, pixels + count);
  stbi_image_free(pixels);
  for (float value : scene.environment.texels)
    if (!std::isfinite(value)) throw std::runtime_error("HDR environment contains non-finite values: " + path);
  buildEnvironmentDistribution(scene.environment.texels.data(), scene.environment.width,
                               scene.environment.height, scene.distribution, scene.distributionInfo);
}

} // namespace pt
