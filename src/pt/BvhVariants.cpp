#include "pt/BvhVariants.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace pt {
namespace {
struct Child { float3 low, high; uint data = kBvhEmpty, count = 0; };

Child binaryChild(const Bvh &bvh, uint node, uint child) {
  if (static_cast<std::size_t>(node) * 4u + child * 2u + 1u >= bvh.nodes.size())
    throw std::runtime_error("wide BVH conversion found an invalid binary child");
  const float4 low = bvh.nodes[node * 4u + child * 2u];
  const float4 high = bvh.nodes[node * 4u + child * 2u + 1u];
  return {xyz(low), xyz(high), as_type<uint>(low.w), as_type<uint>(high.w)};
}
float area(const Child &c) {
  const float3 d = c.high - c.low;
  return std::max(0.0f, 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x));
}
float component(float3 v, uint a) { return a == 0u ? v.x : a == 1u ? v.y : v.z; }
uint packBounds(float low, float high, float origin, float scale) {
  if (!(scale > 0.0f)) return 0u;
  uint lo = static_cast<uint>(std::clamp(std::floor((low - origin) / scale), 0.0f, 65535.0f));
  uint hi = static_cast<uint>(std::clamp(std::ceil((high - origin) / scale), 0.0f, 65535.0f));
  // Leave an extra quantization bin on either side. This keeps the decoded box
  // conservative even when the shader's multiply/add rounds differently from the host.
  if (lo > 0u) --lo;
  if (hi < 65535u) ++hi;
  return lo | (hi << 16u);
}

uint maximumTraversalStack(const WideBvh &bvh) {
  struct Entry { uint value; bool bottom; };
  if (bvh.nodes.empty()) return 0u;
  std::vector<Entry> stack{{0u, false}};
  uint maximum = 1u;
  std::size_t operations = 0;
  while (!stack.empty()) {
    if (++operations > bvh.nodes.size() * 16u + bvh.instances.size() * 2u + 1u)
      throw std::runtime_error("wide BVH conversion produced a cycle");
    const Entry entry = stack.back(); stack.pop_back();
    if ((entry.value & kBvhLeafTag) != 0u) {
      if (!entry.bottom) {
        const uint instance = entry.value & ~kBvhLeafTag;
        if (instance >= bvh.instances.size())
          throw std::runtime_error("wide BVH contains an invalid instance leaf");
        stack.push_back({bvh.instances[instance].blasRoot, true});
      }
    } else {
      if (entry.value >= bvh.nodes.size())
        throw std::runtime_error("wide BVH contains an invalid node reference");
      const QuantizedWideNode &node = bvh.nodes[entry.value];
      const uint count = as_type<uint>(node.origin.w);
      if (count > bvh.width) throw std::runtime_error("wide BVH node exceeds its declared width");
      for (uint child = 0u; child < count; ++child)
        stack.push_back({node.children[child].w, entry.bottom});
    }
    maximum = std::max(maximum, static_cast<uint>(stack.size()));
  }
  return maximum;
}
} // namespace

bool cpuAvx2Available() {
#ifdef _MSC_VER
  int registers[4]{};
  __cpuid(registers, 1);
  if ((registers[2] & (1 << 27)) == 0 || (registers[2] & (1 << 28)) == 0) return false;
  if ((_xgetbv(0) & 6u) != 6u) return false;
  __cpuidex(registers, 7, 0);
  return (registers[1] & (1 << 5)) != 0;
#else
  return false;
#endif
}

WideBvh buildWideBvh(const Bvh &binary, const std::vector<TraceInstance> &source, uint width) {
  if (width != 4u && width != 8u) throw std::runtime_error("wide BVH width must be four or eight");
  const auto started = std::chrono::steady_clock::now();
  WideBvh result;
  result.width = width; result.triangles = binary.triangles; result.instances = source;
  std::function<uint(uint, bool)> emit = [&](uint binaryNode, bool bottom) -> uint {
    const uint output = static_cast<uint>(result.nodes.size());
    result.nodes.emplace_back();
    std::vector<Child> frontier;
    for (uint child = 0; child < 2u; ++child) {
      const Child c = binaryChild(binary, binaryNode, child);
      if (c.data != kBvhEmpty) frontier.push_back(c);
    }
    while (frontier.size() < width) {
      std::size_t chosen = frontier.size(); float largest = -1.0f;
      for (std::size_t i = 0; i < frontier.size(); ++i)
        if (frontier[i].count == 0u && area(frontier[i]) > largest) {
          chosen = i; largest = area(frontier[i]);
        }
      if (chosen == frontier.size()) break;
      const uint node = frontier[chosen].data;
      frontier.erase(frontier.begin() + static_cast<std::ptrdiff_t>(chosen));
      for (uint child = 0; child < 2u; ++child) {
        const Child c = binaryChild(binary, node, child);
        if (c.data != kBvhEmpty) frontier.push_back(c);
      }
    }
    float3 low(std::numeric_limits<float>::infinity());
    float3 high(-std::numeric_limits<float>::infinity());
    for (const Child &c : frontier) { low = min(low, c.low); high = max(high, c.high); }
    if (frontier.empty()) {
      QuantizedWideNode node{};
      node.origin.w = as_type<float>(0u);
      result.nodes[output] = node;
      return output;
    }
    low.x = std::nextafter(low.x, -std::numeric_limits<float>::infinity());
    low.y = std::nextafter(low.y, -std::numeric_limits<float>::infinity());
    low.z = std::nextafter(low.z, -std::numeric_limits<float>::infinity());
    high.x = std::nextafter(high.x, std::numeric_limits<float>::infinity());
    high.y = std::nextafter(high.y, std::numeric_limits<float>::infinity());
    high.z = std::nextafter(high.z, std::numeric_limits<float>::infinity());
    float3 scale = (high - low) * (1.0f / 65535.0f);
    scale.x = std::nextafter(scale.x, std::numeric_limits<float>::infinity());
    scale.y = std::nextafter(scale.y, std::numeric_limits<float>::infinity());
    scale.z = std::nextafter(scale.z, std::numeric_limits<float>::infinity());
    QuantizedWideNode node{};
    node.origin = float4(low, as_type<float>(static_cast<uint>(frontier.size())));
    node.scale = float4(scale, 0.0f);
    for (std::size_t slot = 0; slot < frontier.size(); ++slot) {
      const Child &c = frontier[slot];
      uint data = c.data;
      if (c.count == 0u) data = emit(c.data, bottom);
      else if (bottom) data = kBvhLeafTag | (c.count << kBvhCountShift) | c.data;
      else data = kBvhLeafTag | c.data;
      node.children[slot] = uint4(packBounds(c.low.x, c.high.x, low.x, scale.x),
                                  packBounds(c.low.y, c.high.y, low.y, scale.y),
                                  packBounds(c.low.z, c.high.z, low.z, scale.z), data);
    }
    result.nodes[output] = node;
    return output;
  };
  if (!binary.nodes.empty()) emit(0u, false);
  for (TraceInstance &instance : result.instances) instance.blasRoot = emit(instance.blasRoot, true);
  result.maximumStack = maximumTraversalStack(result);
  if (result.maximumStack > 64u)
    throw std::runtime_error("wide BVH requires more than the GPU traversal stack's 64 entries");
  result.milliseconds = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  return result;
}

PtHit traceWideBvh(const WideBvh &bvh, const Material *materials, const uint *indices,
                   const float *vertices, const HostTextures &textures, float3 origin,
                   float3 direction, float tMax, uint mask, uint seed, float2 cone, uint anyHit) {
  PtHit hit{}; hit.t = tMax;
  PtRayPrep ray{}; float3 prepOrigin = origin, prepDirection = direction;
  bool prepare = true, done = false; uint stack[128]{};
  uint depth = bvh.nodes.empty() ? 0u : 1u, bottomBase = 0u, inBottom = 0u, instance = 0u;
  while (depth > 0u && !done) {
    if (inBottom != 0u && depth <= bottomBase) {
      inBottom = 0u; prepOrigin = origin; prepDirection = direction; prepare = true;
    }
    if (prepare) { ray = ptPrepareRay(prepOrigin, prepDirection); prepare = false; }
    const uint entry = stack[--depth];
    if ((entry & kBvhLeafTag) != 0u) {
      if (inBottom == 0u) {
        instance = entry & ~kBvhLeafTag;
        const TraceInstance &row = bvh.instances[instance];
        if ((row.mask & mask) != 0u) {
          prepOrigin = float3(dot(row.worldToObject0, float4(origin, 1.0f)),
                              dot(row.worldToObject1, float4(origin, 1.0f)),
                              dot(row.worldToObject2, float4(origin, 1.0f)));
          prepDirection = float3(dot(xyz(row.worldToObject0), direction),
                                 dot(xyz(row.worldToObject1), direction),
                                 dot(xyz(row.worldToObject2), direction));
          prepare = true; inBottom = 1u; bottomBase = depth; stack[depth++] = row.blasRoot;
        }
      } else {
        const uint first = entry & kBvhFirstMask;
        const uint count = (entry & ~kBvhLeafTag) >> kBvhCountShift;
        for (uint i = 0; i < count && !done; ++i) {
          const uint triangle = (first + i) * 3u; const float4 v0 = bvh.triangles[triangle];
          float distance = 0.0f; float2 barycentric(0.0f);
          if (ptIntersectTriangle(ray, xyz(v0), xyz(bvh.triangles[triangle + 1u]),
              xyz(bvh.triangles[triangle + 2u]), hit.t, distance, barycentric, hit.ambiguous)) {
            const uint primitive = as_type<uint>(v0.w);
            if ((bvh.instances[instance].flags & kInstanceBlended) != 0u) hit.ambiguous |= 1u;
            if (ptCandidateSolid(bvh.instances.data(), materials, indices, vertices, textures,
                instance, primitive, barycentric, (hit.ambiguous >> 1u) & 1u, seed, direction,
                ptConeWidthOrLevelZero(cone, distance))) {
              hit.ambiguous &= 1u; hit.t = distance; hit.barycentric = barycentric;
              hit.instance = instance; hit.primitive = primitive; hit.found = 1u;
              if (anyHit != 0u) done = true;
            }
          }
        }
      }
      continue;
    }
    const QuantizedWideNode &node = bvh.nodes[entry];
    const uint childCount = as_type<uint>(node.origin.w);
    float distances[8]{}; uint entries[8]{}; uint found = 0u;
    for (uint child = 0; child < childCount; ++child) {
      const uint4 p = node.children[child]; const uint words[3]{p.x, p.y, p.z};
      float3 low, high;
      for (uint axis = 0; axis < 3u; ++axis) {
        const float base = component(xyz(node.origin), axis), step = component(xyz(node.scale), axis);
        const float l = base + step * float(words[axis] & 0xFFFFu);
        const float h = base + step * float(words[axis] >> 16u);
        if (axis == 0u) { low.x = l; high.x = h; }
        else if (axis == 1u) { low.y = l; high.y = h; }
        else { low.z = l; high.z = h; }
      }
      const float3 a = (low - ray.origin) * ray.inverse, z = (high - ray.origin) * ray.inverse;
      const float3 nearV = min(a, z), farV = max(a, z);
      const float nearD = std::max(std::max(nearV.x, nearV.y), std::max(nearV.z, 0.0f));
      const float farD = std::min(std::min(farV.x, farV.y), std::min(farV.z, hit.t)) * 1.0000004f;
      if (nearD <= farD) { distances[found] = nearD; entries[found++] = p.w; }
    }
    for (uint i = 1u; i < found; ++i) {
      const float d = distances[i]; const uint e = entries[i]; uint j = i;
      while (j > 0u && distances[j - 1u] < d) {
        distances[j] = distances[j - 1u]; entries[j] = entries[j - 1u]; --j;
      }
      distances[j] = d; entries[j] = e;
    }
    for (uint i = 0u; i < found; ++i) {
      if (depth >= 128u) throw std::runtime_error("wide BVH traversal stack overflow");
      stack[depth++] = entries[i];
    }
  }
  hit.ambiguous &= 1u;  // drop the per-candidate facing scratch bit
  return hit;
}
} // namespace pt
