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
      const uint count = as_type<uint>(node.origin.w) & kWideSlotMask;
      if (count > 8u) throw std::runtime_error("wide BVH node exceeds eight slots");
      for (uint child = 0u; child < count; ++child)
        if (node.children[child].w != kBvhEmpty) stack.push_back({node.children[child].w, entry.bottom});
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
    // BVH8: octant slots (bvh_layout.h), greedily giving each slot s the child a ray of octant s
    // meets first, the smallest dot(child centre - node centre, direction of s); the cheapest
    // remaining (child, slot) pair first, lower child then lower slot on a tie.
    std::array<int, 8> childOfSlot;
    childOfSlot.fill(-1);
    const bool octantSlots = width == 8u;
    if (!octantSlots) {
      // BVH4 nodes stay compact: children in collapse order in slots 0..count-1.
      for (std::size_t i = 0; i < frontier.size(); ++i) childOfSlot[i] = static_cast<int>(i);
    } else {
      const float3 centre = (low + high) * 0.5f;
      std::vector<bool> placed(frontier.size(), false);
      for (std::size_t assigned = 0; assigned < frontier.size(); ++assigned) {
        float best = std::numeric_limits<float>::infinity();
        std::size_t bestChild = 0, bestSlot = 0;
        for (std::size_t i = 0; i < frontier.size(); ++i) {
          if (placed[i]) continue;
          const float3 offset = (frontier[i].low + frontier[i].high) * 0.5f - centre;
          for (std::size_t s = 0; s < 8u; ++s) {
            if (childOfSlot[s] >= 0) continue;
            const float cost = ((s & 1u) ? -offset.x : offset.x) + ((s & 2u) ? -offset.y : offset.y) +
                               ((s & 4u) ? -offset.z : offset.z);
            if (cost < best) { best = cost; bestChild = i; bestSlot = s; }
          }
        }
        placed[bestChild] = true;
        childOfSlot[bestSlot] = static_cast<int>(bestChild);
      }
    }
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
    node.origin = float4(low, as_type<float>(octantSlots ? (8u | kWideOctantOrdered) : static_cast<uint>(frontier.size())));
    node.scale = float4(scale, 0.0f);
    for (std::size_t slot = 0; slot < (octantSlots ? 8u : frontier.size()); ++slot) {
      if (childOfSlot[slot] < 0) {
        node.children[slot] = uint4(0u, 0u, 0u, kBvhEmpty);
        continue;
      }
      const Child &c = frontier[static_cast<std::size_t>(childOfSlot[slot])];
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
  if (result.maximumStack > PT_WIDE_BVH_STACK_DEEP)
    throw std::runtime_error("wide BVH requires more than the GPU traversal stack's " +
                             std::to_string(PT_WIDE_BVH_STACK_DEEP) + " entries");
  result.milliseconds = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  return result;
}

namespace {
float layoutArea(float3 low, float3 high) {
  if (!(low.x <= high.x && low.y <= high.y && low.z <= high.z)) return 0.0f;
  const float3 d = high - low;
  return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}

double binaryTreeCost(const std::vector<float4> &nodes, uint root) {
  const std::size_t nodeCount = nodes.size() / 4u;
  if (root >= nodeCount) throw std::runtime_error("BVH cost: a root is outside the node array");
  auto child = [&](uint node, uint side) {
    const float4 a = nodes[node * 4u + side * 2u], b = nodes[node * 4u + side * 2u + 1u];
    return Child{xyz(a), xyz(b), as_type<uint>(a.w), as_type<uint>(b.w)};
  };
  Child children[2] = {child(root, 0), child(root, 1)};
  float3 low(3.0e38f), high(-3.0e38f);
  uint present = 0;
  for (const Child &c : children)
    if (!(c.data == kBvhEmpty && c.count == 0u)) { low = min(low, c.low); high = max(high, c.high); ++present; }
  const float rootArea = std::max(layoutArea(low, high), 1e-30f);
  double cost = present == 2u ? 1.0 : 0.0;
  std::vector<uint> stack{root};
  std::size_t visited = 0;
  while (!stack.empty()) {
    if (++visited > nodeCount) throw std::runtime_error("BVH cost: the tree has a cycle");
    const uint node = stack.back();
    stack.pop_back();
    for (uint side = 0; side < 2u; ++side) {
      const Child c = child(node, side);
      if (c.data == kBvhEmpty && c.count == 0u) continue;
      const double relative = static_cast<double>(layoutArea(c.low, c.high) / rootArea);
      cost += c.count == 0u ? relative : relative * static_cast<double>(c.count);
      if (c.count == 0u) {
        if (c.data >= nodeCount) throw std::runtime_error("BVH cost: a child is outside the node array");
        stack.push_back(c.data);
      }
    }
  }
  return cost;
}

void decodeWide(const QuantizedWideNode &node, uint child, float3 &low, float3 &high) {
  const uint4 &packed = node.children[child];
  const float3 origin = xyz(node.origin), scale = xyz(node.scale);
  low = origin + scale * float3(static_cast<float>(packed.x & 0xFFFFu), static_cast<float>(packed.y & 0xFFFFu),
                                static_cast<float>(packed.z & 0xFFFFu));
  high = origin + scale * float3(static_cast<float>(packed.x >> 16u), static_cast<float>(packed.y >> 16u),
                                 static_cast<float>(packed.z >> 16u));
}

double wideTreeCost(const std::vector<QuantizedWideNode> &nodes, uint root, bool bottom) {
  if (root >= nodes.size()) throw std::runtime_error("wide BVH cost: a root is outside the node array");
  const QuantizedWideNode &top = nodes[root];
  const uint rootSlots = as_type<uint>(top.origin.w) & kWideSlotMask;
  float3 low(3.0e38f), high(-3.0e38f);
  uint rootChildren = 0, onlyChild = kBvhEmpty;
  for (uint child = 0; child < rootSlots; ++child) {
    if (top.children[child].w == kBvhEmpty) continue;
    float3 a, b;
    decodeWide(top, child, a, b);
    low = min(low, a);
    high = max(high, b);
    ++rootChildren;
    onlyChild = top.children[child].w;
  }
  const float rootArea = std::max(layoutArea(low, high), 1e-30f);
  const bool interiorRoot = rootChildren > 1u || (rootChildren == 1u && (onlyChild & kBvhLeafTag) == 0u);
  double cost = interiorRoot ? 1.0 : 0.0;
  std::vector<uint> stack{root};
  std::size_t visited = 0;
  while (!stack.empty()) {
    if (++visited > nodes.size()) throw std::runtime_error("wide BVH cost: the tree has a cycle");
    const QuantizedWideNode &node = nodes[stack.back()];
    stack.pop_back();
    const uint children = as_type<uint>(node.origin.w) & kWideSlotMask;
    for (uint child = 0; child < children; ++child) {
      const uint data = node.children[child].w;
      if (data == kBvhEmpty) continue;
      float3 a, b;
      decodeWide(node, child, a, b);
      const double relative = static_cast<double>(layoutArea(a, b) / rootArea);
      if ((data & kBvhLeafTag) == 0u) {
        cost += relative;
        stack.push_back(data);
      } else {
        cost += relative * static_cast<double>(bottom ? (data & ~kBvhLeafTag) >> kBvhCountShift : 1u);
      }
    }
  }
  return cost;
}
} // namespace

LayoutCost binaryLayoutCost(const std::vector<float4> &nodes, const std::vector<TraceInstance> &instances) {
  LayoutCost cost;
  if (nodes.size() < 4u) return cost;
  cost.top = binaryTreeCost(nodes, 0u);
  for (const TraceInstance &instance : instances) cost.bottom += binaryTreeCost(nodes, instance.blasRoot);
  return cost;
}

LayoutCost wideLayoutCost(const std::vector<QuantizedWideNode> &nodes, const std::vector<TraceInstance> &instances) {
  LayoutCost cost;
  if (nodes.empty()) return cost;
  cost.top = wideTreeCost(nodes, 0u, false);
  for (const TraceInstance &instance : instances) cost.bottom += wideTreeCost(nodes, instance.blasRoot, true);
  return cost;
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
    const uint childCount = as_type<uint>(node.origin.w) & kWideSlotMask;
    float distances[8]{}; uint entries[8]{}; uint found = 0u;
    for (uint child = 0; child < childCount; ++child) {
      const uint4 p = node.children[child]; const uint words[3]{p.x, p.y, p.z};
      if (p.w == kBvhEmpty) continue;
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
