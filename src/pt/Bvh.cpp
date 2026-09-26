#include "pt/Bvh.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <functional>
#include <stdexcept>
#include <thread>

namespace pt {
namespace {

constexpr float kHuge = 3.0e38f;
constexpr uint kBins = 16;
constexpr uint kMaxBottomLeaf = 8;  // the stack entry holds a leaf's count in four bits

struct Box {
  float3 low{kHuge, kHuge, kHuge};
  float3 high{-kHuge, -kHuge, -kHuge};
  void grow(float3 p) {
    low = min(low, p);
    high = max(high, p);
  }
  void grow(const Box &b) {
    low = min(low, b.low);
    high = max(high, b.high);
  }
  bool valid() const { return low.x <= high.x; }
  float area() const {
    if (!valid()) return 0.0f;
    const float3 d = high - low;
    return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
  }
};

float axisOf(float3 v, int axis) { return axis == 0 ? v.x : (axis == 1 ? v.y : v.z); }

struct BuildNode {
  Box box;
  uint left = 0, right = 0;  // children, for an interior node
  uint first = 0, count = 0; // range of `order`, for a leaf (count > 0)
};

struct Tree {
  std::vector<BuildNode> nodes;  // root at 0
  std::vector<uint> order;       // primitives in leaf order
  uint depth = 0;
  double cost = 0.0;
  uint interiorCount() const {
    uint n = 0;
    for (const BuildNode &node : nodes) n += node.count == 0 ? 1u : 0u;
    return n;
  }
  uint emittedCount() const { return std::max(1u, interiorCount()); }
};

// Binned SAH over primitive boxes, with a traversal step and a primitive test costing the same.
Tree buildTree(const std::vector<Box> &boxes, uint maxLeaf) {
  Tree tree;
  const uint n = static_cast<uint>(boxes.size());
  tree.order.resize(n);
  std::vector<float3> centres(n);
  for (uint i = 0; i < n; ++i) {
    tree.order[i] = i;
    centres[i] = (boxes[i].low + boxes[i].high) * 0.5f;
  }
  BuildNode root;
  root.first = 0;
  root.count = n;
  tree.nodes.push_back(root);
  struct Pending {
    uint node;
    uint depth;
  };
  std::vector<Pending> stack{{0, 1}};
  while (!stack.empty()) {
    const Pending pending = stack.back();
    stack.pop_back();
    tree.depth = std::max(tree.depth, pending.depth);
    BuildNode &node = tree.nodes[pending.node];
    const uint first = node.first, count = node.count;
    Box box, centreBox;
    for (uint i = first; i < first + count; ++i) {
      box.grow(boxes[tree.order[i]]);
      centreBox.grow(centres[tree.order[i]]);
    }
    node.box = box;
    if (count <= 1) continue;

    int bestAxis = -1;
    uint bestBin = 0;
    float bestCost = kHuge;
    for (int axis = 0; axis < 3; ++axis) {
      const float low = axisOf(centreBox.low, axis), extent = axisOf(centreBox.high, axis) - low;
      if (!(extent > 0.0f)) continue;
      const float scale = static_cast<float>(kBins) / extent;
      Box binBoxes[kBins];
      uint binCounts[kBins]{};
      for (uint i = first; i < first + count; ++i) {
        const uint p = tree.order[i];
        const uint bin = std::min(kBins - 1, static_cast<uint>((axisOf(centres[p], axis) - low) * scale));
        binBoxes[bin].grow(boxes[p]);
        ++binCounts[bin];
      }
      float rightAreas[kBins]{};
      uint rightCounts[kBins]{};
      Box right;
      uint rightCount = 0;
      for (uint bin = kBins - 1; bin > 0; --bin) {
        right.grow(binBoxes[bin]);
        rightCount += binCounts[bin];
        rightAreas[bin - 1] = right.area();
        rightCounts[bin - 1] = rightCount;
      }
      Box left;
      uint leftCount = 0;
      for (uint bin = 0; bin + 1 < kBins; ++bin) {
        left.grow(binBoxes[bin]);
        leftCount += binCounts[bin];
        if (leftCount == 0 || rightCounts[bin] == 0) continue;
        const float cost = left.area() * static_cast<float>(leftCount) + rightAreas[bin] * static_cast<float>(rightCounts[bin]);
        if (cost < bestCost) {
          bestCost = cost;
          bestAxis = axis;
          bestBin = bin;
        }
      }
    }

    const float parentArea = std::max(box.area(), 1e-30f);
    const float splitCost = 1.0f + bestCost / parentArea;
    if (count <= maxLeaf && (bestAxis < 0 || static_cast<float>(count) <= splitCost)) continue;

    // A stable partition, and halving in order, so the GPU binned SAH builder (bvh_sah.slang)
    // can make the same tree.
    uint middle = first;
    if (bestAxis >= 0) {
      const float low = axisOf(centreBox.low, bestAxis);
      const float scale = static_cast<float>(kBins) / (axisOf(centreBox.high, bestAxis) - low);
      middle = static_cast<uint>(
          std::stable_partition(tree.order.begin() + first, tree.order.begin() + first + count,
                                [&](uint p) {
                                  return std::min(kBins - 1, static_cast<uint>((axisOf(centres[p], bestAxis) - low) *
                                                                               scale)) <= bestBin;
                                }) -
          tree.order.begin());
    }
    // No boundary separates anything only when every centre coincides (or every cost overflows):
    // then there is no order to split by, and the range is halved as it stands.
    if (middle == first || middle == first + count) middle = first + count / 2;

    BuildNode leftChild, rightChild;
    leftChild.first = first;
    leftChild.count = middle - first;
    rightChild.first = middle;
    rightChild.count = first + count - middle;
    const uint leftIndex = static_cast<uint>(tree.nodes.size());
    tree.nodes.push_back(leftChild);
    tree.nodes.push_back(rightChild);
    BuildNode &parent = tree.nodes[pending.node];  // the push may have moved it
    parent.left = leftIndex;
    parent.right = leftIndex + 1;
    parent.count = 0;
    stack.push_back({leftIndex, pending.depth + 1});
    stack.push_back({leftIndex + 1, pending.depth + 1});
  }

  // SAH cost of the finished tree, relative to the root's area.
  const float rootArea = std::max(tree.nodes[0].box.area(), 1e-30f);
  double cost = 0.0;
  for (const BuildNode &node : tree.nodes)
    cost += static_cast<double>(node.box.area() / rootArea) * (node.count == 0 ? 1.0 : static_cast<double>(node.count));
  tree.cost = cost;
  return tree;
}

float4 withBits(float3 v, uint bits) { return float4(v, as_type<float>(bits)); }

// Writes a tree's interior nodes as traversal nodes, in preorder, at out[nodeBase...].
// A leaf child's data is leafData(first, count); an interior child's is its node index.
template <class LeafData>
void emitTree(const Tree &tree, uint nodeBase, std::vector<float4> &out, LeafData leafData) {
  const BuildNode &root = tree.nodes[0];
  auto child = [&](const BuildNode &node, uint index, float4 *slots) {
    slots[0] = withBits(node.box.low, node.count > 0 ? leafData(node.first, node.count) : index);
    slots[1] = withBits(node.box.high, node.count);
  };
  if (root.count > 0 || tree.nodes.size() == 1) {
    // A single leaf still needs a node to hold it; the other child is empty.
    float4 slots[4];
    if (root.count > 0) child(root, 0, slots);
    else {
      slots[0] = withBits(float3(kHuge), kBvhEmpty);
      slots[1] = withBits(float3(-kHuge), 0);
    }
    slots[2] = withBits(float3(kHuge), kBvhEmpty);
    slots[3] = withBits(float3(-kHuge), 0);
    out.insert(out.end(), slots, slots + 4);
    return;
  }
  // Preorder numbering of interior nodes, so the root is the first.
  std::vector<uint> numbering(tree.nodes.size(), 0);
  std::vector<uint> preorder;
  std::vector<uint> stack{0};
  while (!stack.empty()) {
    const uint index = stack.back();
    stack.pop_back();
    const BuildNode &node = tree.nodes[index];
    if (node.count > 0) continue;
    numbering[index] = nodeBase + static_cast<uint>(preorder.size());
    preorder.push_back(index);
    stack.push_back(node.right);
    stack.push_back(node.left);
  }
  for (const uint index : preorder) {
    const BuildNode &node = tree.nodes[index];
    float4 slots[4];
    child(tree.nodes[node.left], numbering[node.left], slots);
    child(tree.nodes[node.right], numbering[node.right], slots + 2);
    out.insert(out.end(), slots, slots + 4);
  }
}

// Early split clipping's arithmetic, which bvh_clip.slang repeats operation for operation so the
// GPU makes the same references: floats compared by their ordered bits (no flush, one order for
// the signed zeros), the clipping in double precision with explicit comparisons, and the rounding
// to float outwards by stepping the bits.
float lesser(float a, float b) { return ptOrderedBits(a) <= ptOrderedBits(b) ? a : b; }
float greater(float a, float b) { return ptOrderedBits(a) >= ptOrderedBits(b) ? a : b; }
double lesserDouble(double a, double b) { return a < b ? a : b; }
double greaterDouble(double a, double b) { return a > b ? a : b; }

// The next float towards minus (or plus) infinity; from either zero, the smallest subnormal.
float nextDown(float f) {
  const uint bits = as_type<uint>(f);
  if ((bits & 0x7FFFFFFFu) == 0u) return as_type<float>(0x80000001u);
  return as_type<float>((bits & 0x80000000u) != 0u ? bits + 1u : bits - 1u);
}
float nextUp(float f) {
  const uint bits = as_type<uint>(f);
  if ((bits & 0x7FFFFFFFu) == 0u) return as_type<float>(0x00000001u);
  return as_type<float>((bits & 0x80000000u) != 0u ? bits - 1u : bits + 1u);
}
float roundDown(double d) {
  const float f = static_cast<float>(d);
  return static_cast<double>(f) > d ? nextDown(f) : f;
}
float roundUp(double d) {
  const float f = static_cast<float>(d);
  return static_cast<double>(f) < d ? nextUp(f) : f;
}

struct ClipBox {
  float low[3], high[3];
};

// A box's surface area in double precision.
double clipArea(const ClipBox &box) {
  const double dx = static_cast<double>(box.high[0]) - box.low[0], dy = static_cast<double>(box.high[1]) - box.low[1],
               dz = static_cast<double>(box.high[2]) - box.low[2];
  return 2.0 * (dx * dy + dy * dz + dz * dx);
}

ClipBox triangleBox(const float3 corner[3]) {
  ClipBox box{{corner[0].x, corner[0].y, corner[0].z}, {corner[0].x, corner[0].y, corner[0].z}};
  for (int k = 1; k < 3; ++k) {
    const float c[3] = {corner[k].x, corner[k].y, corner[k].z};
    for (int a = 0; a < 3; ++a) {
      box.low[a] = lesser(box.low[a], c[a]);
      box.high[a] = greater(box.high[a], c[a]);
    }
  }
  return box;
}

// A split: the plane where coordinate `axis` is c, and the side kept.
struct ClipPlane {
  uint axis;
  float c;
  bool low;
};

constexpr uint kClipDepth = 6;     // kMaxClipReferences = 1 << kClipDepth
constexpr uint kClipBlock = 256;   // the threshold's sum: blocks of a bottom level's triangles
constexpr uint kClipPoints = 3u + kClipDepth;

// The polygon's part on one side of a plane, Sutherland-Hodgman; points on the plane belong to
// both sides. Returns the new point count.
uint clipPolygon(const double (*in)[3], uint n, const ClipPlane &plane, double (*out)[3]) {
  uint made = 0;
  const double c = plane.c;
  for (uint i = 0; i < n; ++i) {
    const double *a = in[i], *b = in[i + 1u == n ? 0u : i + 1u];
    const double da = plane.low ? a[plane.axis] - c : c - a[plane.axis];
    const double db = plane.low ? b[plane.axis] - c : c - b[plane.axis];
    if (da <= 0.0) {
      for (int k = 0; k < 3; ++k) out[made][k] = a[k];
      ++made;
    }
    if ((da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0)) {
      const double t = da / (da - db);
      for (int k = 0; k < 3; ++k) out[made][k] = a[k] + t * (b[k] - a[k]);
      out[made][plane.axis] = c;
      ++made;
    }
  }
  return made;
}

// A triangle's references, depth first, the low half before the high: each part is the triangle
// clipped by the planes on its path (replayed from the triangle, the same operations as clipping
// the parent's part), its box padded by `pad` (above the double arithmetic's error), rounded
// outwards, kept within the triangle's box and exactly on the last plane. A part splits while
// area * count > factor * sum (its bottom level's triangle count and box area sum) and it is
// shallower than kClipDepth, at the middle of its box's longest axis.
template <class Emit>
void clipTriangle(const float3 corner[3], uint t, double count, double factorSum, Emit emit) {
  const ClipBox triangle = triangleBox(corner);
  double scale = 0.0;
  for (int a = 0; a < 3; ++a)
    scale = greaterDouble(scale, greaterDouble(std::fabs(static_cast<double>(triangle.low[a])),
                                               std::fabs(static_cast<double>(triangle.high[a]))));
  const double pad = scale * 0x1p-40;
  ClipPlane path[kClipDepth]{};
  bool pendingHigh[kClipDepth]{};
  uint depth = 0;
  for (;;) {
    ClipBox box = triangle;
    bool empty = false;
    if (depth > 0) {
      double points[2][kClipPoints][3];
      uint n = 3;
      for (int k = 0; k < 3; ++k) {
        points[0][k][0] = corner[k].x;
        points[0][k][1] = corner[k].y;
        points[0][k][2] = corner[k].z;
      }
      for (uint j = 0; j < depth && n > 0; ++j) n = clipPolygon(points[j & 1u], n, path[j], points[(j + 1u) & 1u]);
      empty = n == 0;
      if (!empty) {
        const double(*polygon)[3] = points[depth & 1u];
        for (int a = 0; a < 3; ++a) {
          double low = polygon[0][a], high = polygon[0][a];
          for (uint i = 1; i < n; ++i) {
            low = lesserDouble(low, polygon[i][a]);
            high = greaterDouble(high, polygon[i][a]);
          }
          box.low[a] = greater(roundDown(low - pad), triangle.low[a]);
          box.high[a] = lesser(roundUp(high + pad), triangle.high[a]);
        }
        const ClipPlane &last = path[depth - 1u];
        if (last.low) box.high[last.axis] = lesser(box.high[last.axis], last.c);
        else box.low[last.axis] = greater(box.low[last.axis], last.c);
      }
    }
    if (!empty) {
      const double d[3] = {static_cast<double>(box.high[0]) - box.low[0], static_cast<double>(box.high[1]) - box.low[1],
                           static_cast<double>(box.high[2]) - box.low[2]};
      if (depth < kClipDepth && clipArea(box) * count > factorSum && greaterDouble(d[0], greaterDouble(d[1], d[2])) > 0.0) {
        const uint axis = d[0] >= d[1] && d[0] >= d[2] ? 0u : (d[1] >= d[2] ? 1u : 2u);
        path[depth] = {axis, static_cast<float>((static_cast<double>(box.low[axis]) + box.high[axis]) * 0.5), true};
        pendingHigh[depth] = true;
        ++depth;
        continue;
      }
      BvhReference reference;
      reference.low = float4(box.low[0], box.low[1], box.low[2], as_type<float>(t));
      reference.high = float4(box.high[0], box.high[1], box.high[2], 0.0f);
      emit(reference);
    }
    // Next: the high half of the deepest split whose high half is still to come.
    while (depth > 0 && !pendingHigh[depth - 1u]) --depth;
    if (depth == 0) break;
    pendingHigh[depth - 1u] = false;
    path[depth - 1u].low = false;
  }
}

} // namespace

BvhReferences clipReferences(const std::vector<float> &vertices, const std::vector<uint> &indices,
                             const std::vector<TraceInstance> &instances, const std::vector<uint> &triangleCounts,
                             float factor, unsigned threads) {
  const auto started = std::chrono::steady_clock::now();
  auto corners = [&](const TraceInstance &instance, uint t, float3 out[3]) {
    for (uint k = 0; k < 3; ++k) {
      const float *v = vertices.data() +
                       static_cast<std::size_t>(indices[instance.firstIndex + t * 3 + k] + instance.vertexOffset) * kVertexFloats;
      out[k] = float3(v[0], v[1], v[2]);
    }
  };
  // Each bottom level's triangle box area sum: blocks of kClipBlock summed in order, then the
  // blocks' sums in order (the GPU's order).
  const std::size_t instanceCount = instances.size();
  std::vector<double> factorSum(instanceCount, 0.0);
  for (std::size_t i = 0; i < instanceCount; ++i) {
    double sum = 0.0;
    for (uint first = 0; first < triangleCounts[i]; first += kClipBlock) {
      double block = 0.0;
      for (uint t = first; t < std::min(triangleCounts[i], first + kClipBlock); ++t) {
        float3 corner[3];
        corners(instances[i], t, corner);
        block += clipArea(triangleBox(corner));
      }
      sum += block;
    }
    factorSum[i] = static_cast<double>(factor) * sum;
  }

  // Chunks of a bottom level's triangles across threads, joined in order.
  constexpr uint kChunk = 2048;
  struct Chunk {
    std::size_t instance;
    uint first, last;
    std::vector<BvhReference> references;
  };
  std::vector<Chunk> chunks;
  for (std::size_t i = 0; i < instanceCount; ++i)
    for (uint first = 0; first < triangleCounts[i]; first += kChunk)
      chunks.push_back({i, first, std::min(triangleCounts[i], first + kChunk), {}});
  std::atomic<std::size_t> next{0};
  auto work = [&] {
    for (std::size_t c; (c = next.fetch_add(1)) < chunks.size();) {
      Chunk &chunk = chunks[c];
      for (uint t = chunk.first; t < chunk.last; ++t) {
        float3 corner[3];
        corners(instances[chunk.instance], t, corner);
        clipTriangle(corner, t, static_cast<double>(triangleCounts[chunk.instance]), factorSum[chunk.instance],
                     [&](const BvhReference &reference) { chunk.references.push_back(reference); });
      }
    }
  };
  std::vector<std::thread> pool;
  const unsigned helpers = static_cast<unsigned>(std::max<std::size_t>(1, std::min<std::size_t>(threads, chunks.size()))) - 1;
  for (unsigned t = 0; t < helpers; ++t) pool.emplace_back(work);
  work();
  for (std::thread &thread : pool) thread.join();

  BvhReferences result;
  result.counts.assign(instanceCount, 0u);
  std::size_t total = 0;
  for (const Chunk &chunk : chunks) total += chunk.references.size();
  result.references.reserve(total);
  for (const Chunk &chunk : chunks) {
    result.counts[chunk.instance] += static_cast<uint>(chunk.references.size());
    result.references.insert(result.references.end(), chunk.references.begin(), chunk.references.end());
  }
  result.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  return result;
}

BvhStatistics buildBvh(const std::vector<float> &vertices, const std::vector<uint> &indices,
                       std::vector<TraceInstance> &instances, const std::vector<uint> &triangleCounts, Bvh &bvh,
                       unsigned threads, const BvhReferences *references) {
  const auto started = std::chrono::steady_clock::now();
  const uint instanceCount = static_cast<uint>(instances.size());
  BvhStatistics statistics;
  statistics.instances = instanceCount;

  auto position = [&](uint index) {
    const float *v = vertices.data() + static_cast<std::size_t>(index) * kVertexFloats;
    return float3(v[0], v[1], v[2]);
  };
  auto corner = [&](const TraceInstance &instance, uint t, uint k) {
    return position(indices[instance.firstIndex + t * 3 + k] + instance.vertexOffset);
  };

  // Each bottom level's first reference; a leaf's record is a reference, or a triangle.
  std::vector<std::size_t> referenceBase(instanceCount + 1u, 0u);
  if (references) {
    if (references->counts.size() != instanceCount)
      throw std::runtime_error("BVH references do not match the instances");
    for (uint i = 0; i < instanceCount; ++i) referenceBase[i + 1u] = referenceBase[i] + references->counts[i];
    if (referenceBase[instanceCount] != references->references.size())
      throw std::runtime_error("BVH reference counts do not match the references");
  }
  auto primitiveOf = [&](uint i, uint record) {
    return references ? as_type<uint>(references->references[referenceBase[i] + record].low.w) : record;
  };

  // Bottom levels, one per instance, across threads.
  std::vector<Tree> bottom(instanceCount);
  std::atomic<uint> next{0};
  auto buildBottom = [&] {
    for (uint i; (i = next.fetch_add(1)) < instanceCount;) {
      const TraceInstance &instance = instances[i];
      std::vector<Box> boxes(references ? references->counts[i] : triangleCounts[i]);
      for (uint r = 0; r < boxes.size(); ++r) {
        if (references) {
          const BvhReference &reference = references->references[referenceBase[i] + r];
          boxes[r].low = float3(reference.low.x, reference.low.y, reference.low.z);
          boxes[r].high = float3(reference.high.x, reference.high.y, reference.high.z);
        } else {
          for (uint k = 0; k < 3; ++k) boxes[r].grow(corner(instance, r, k));
        }
      }
      bottom[i] = buildTree(boxes, kMaxBottomLeaf);
    }
  };
  std::vector<std::thread> pool;
  const unsigned helpers = std::max(1u, std::min(threads, instanceCount)) - 1;
  for (unsigned t = 0; t < helpers; ++t) pool.emplace_back(buildBottom);
  buildBottom();
  for (std::thread &thread : pool) thread.join();

  // The top level over each instance's world bounds: its bottom root's corners, transformed.
  std::vector<Box> worldBoxes(instanceCount);
  for (uint i = 0; i < instanceCount; ++i) {
    const Box &local = bottom[i].nodes[0].box;
    const TraceInstance &instance = instances[i];
    for (uint c = 0; c < 8; ++c) {
      const float3 p((c & 1) ? local.high.x : local.low.x, (c & 2) ? local.high.y : local.low.y,
                     (c & 4) ? local.high.z : local.low.z);
      worldBoxes[i].grow(applyRows(instance.objectToWorld0, instance.objectToWorld1, instance.objectToWorld2, p));
    }
  }
  const Tree top = buildTree(worldBoxes, 1);

  // Traversal keeps the top-level siblings on the same private stack while it walks a
  // bottom level. Reject a scene whose constructed trees cannot fit; dropping a child
  // when the stack fills would turn a capacity problem into a plausible wrong image.
  for (uint i = 0; i < instanceCount; ++i) {
    const uint required = top.depth + bottom[i].depth;
    if (required > kBvhStack)
      throw std::runtime_error("software BVH needs a traversal stack of " + std::to_string(required) +
                               " entries; this build supports " + std::to_string(kBvhStack));
  }

  // Layout: the top level at node 0, then each bottom level; triangles in leaf order.
  bvh.nodes.clear();
  bvh.triangles.clear();
  emitTree(top, 0, bvh.nodes, [&](uint first, uint) { return top.order[first]; });
  statistics.topNodes = static_cast<uint>(bvh.nodes.size() / 4);
  statistics.topDepth = top.depth;
  uint triangleOffset = 0;
  for (uint i = 0; i < instanceCount; ++i) {
    TraceInstance &instance = instances[i];
    const uint nodeBase = static_cast<uint>(bvh.nodes.size() / 4);
    instance.blasRoot = nodeBase;
    instance.triangleOffset = triangleOffset;
    const Tree &tree = bottom[i];
    emitTree(tree, nodeBase, bvh.nodes, [&](uint first, uint) { return triangleOffset + first; });
    for (const uint record : tree.order) {
      const uint t = primitiveOf(i, record);
      bvh.triangles.push_back(withBits(corner(instance, t, 0), t));
      bvh.triangles.push_back(float4(corner(instance, t, 1), 0.0f));
      bvh.triangles.push_back(float4(corner(instance, t, 2), 0.0f));
    }
    triangleOffset += static_cast<uint>(tree.order.size());
    statistics.bottomNodes += tree.emittedCount();
    statistics.bottomDepth = std::max(statistics.bottomDepth, tree.depth);
    statistics.sahCost += tree.cost;
  }
  statistics.triangles = triangleOffset;
  statistics.milliseconds =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  return statistics;
}

double refitBvh(const std::vector<float> &vertices, const std::vector<uint> &indices,
                const std::vector<TraceInstance> &instances, Bvh &bvh) {
  const auto started = std::chrono::steady_clock::now();
  if (bvh.nodes.size() % 4u != 0u || bvh.triangles.size() % 3u != 0u)
    throw std::runtime_error("cannot refit a malformed BVH layout");

  auto position = [&](uint index) {
    const std::size_t base = static_cast<std::size_t>(index) * kVertexFloats;
    if (base + 2u >= vertices.size()) throw std::runtime_error("BVH refit vertex index is out of range");
    return float3(vertices[base], vertices[base + 1u], vertices[base + 2u]);
  };
  // Leaf order is stable and v0.w retains the source primitive identity.
  for (uint instanceIndex = 0; instanceIndex < instances.size(); ++instanceIndex) {
    const TraceInstance &instance = instances[instanceIndex];
    const uint end = instanceIndex + 1u < instances.size()
                         ? instances[instanceIndex + 1u].triangleOffset
                         : static_cast<uint>(bvh.triangles.size() / 3u);
    if (end < instance.triangleOffset) throw std::runtime_error("BVH refit triangle range is invalid");
    for (uint ordered = instance.triangleOffset; ordered < end; ++ordered) {
      const uint primitive = as_type<uint>(bvh.triangles[ordered * 3u].w);
      const std::size_t first = static_cast<std::size_t>(instance.firstIndex) + primitive * 3u;
      if (first + 2u >= indices.size()) throw std::runtime_error("BVH refit index is out of range");
      for (uint corner = 0; corner < 3u; ++corner) {
        const float3 p = position(indices[first + corner] + instance.vertexOffset);
        const float oldW = bvh.triangles[ordered * 3u + corner].w;
        bvh.triangles[ordered * 3u + corner] = float4(p, oldW);
      }
    }
  }

  std::vector<Box> bottomBounds(instances.size());
  std::function<Box(uint, bool)> refitNode = [&](uint nodeIndex, bool bottom) -> Box {
    if (static_cast<std::size_t>(nodeIndex) * 4u + 3u >= bvh.nodes.size())
      throw std::runtime_error("BVH refit node index is out of range");
    Box parent;
    for (uint child = 0; child < 2u; ++child) {
      float4 &lowSlot = bvh.nodes[nodeIndex * 4u + child * 2u];
      float4 &highSlot = bvh.nodes[nodeIndex * 4u + child * 2u + 1u];
      const uint data = as_type<uint>(lowSlot.w), count = as_type<uint>(highSlot.w);
      if (data == kBvhEmpty) continue;
      Box bounds;
      if (count == 0u) {
        bounds = refitNode(data, bottom);
      } else if (bottom) {
        if (static_cast<std::size_t>(data + count) * 3u > bvh.triangles.size())
          throw std::runtime_error("BVH refit leaf range is out of bounds");
        for (uint i = 0; i < count; ++i)
          for (uint corner = 0; corner < 3u; ++corner)
            bounds.grow(xyz(bvh.triangles[(data + i) * 3u + corner]));
      } else {
        if (data >= instances.size()) throw std::runtime_error("BVH refit instance is out of range");
        const Box &local = bottomBounds[data];
        const TraceInstance &instance = instances[data];
        for (uint corner = 0; corner < 8u; ++corner) {
          const float3 p((corner & 1u) ? local.high.x : local.low.x,
                         (corner & 2u) ? local.high.y : local.low.y,
                         (corner & 4u) ? local.high.z : local.low.z);
          bounds.grow(applyRows(instance.objectToWorld0, instance.objectToWorld1,
                                instance.objectToWorld2, p));
        }
      }
      lowSlot = float4(bounds.low, as_type<float>(data));
      highSlot = float4(bounds.high, as_type<float>(count));
      parent.grow(bounds);
    }
    return parent;
  };

  for (uint instance = 0; instance < instances.size(); ++instance)
    bottomBounds[instance] = refitNode(instances[instance].blasRoot, true);
  if (!bvh.nodes.empty()) refitNode(0u, false);
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

std::vector<PtEmissiveTriangle> buildEmissiveTriangles(const std::vector<float> &vertices,
    const std::vector<uint> &indices, const std::vector<TraceInstance> &instances,
    const std::vector<uint> &triangleCounts, const std::vector<Material> &materials) {
  std::vector<PtEmissiveTriangle> result;
  double total = 0.0;
  for (uint instanceIndex = 0; instanceIndex < instances.size(); ++instanceIndex) {
    const TraceInstance &instance = instances[instanceIndex];
    const Material &material = materials.at(instance.material);
    const float3 radiance = xyz(material.emissive) * material.emissive.w;
    const float power = ptLuminance(radiance);
    if (!(power > 0.0f)) continue;
    for (uint primitive = 0; primitive < triangleCounts[instanceIndex]; ++primitive) {
      float3 world[3];
      for (uint corner = 0; corner < 3; ++corner) {
        const uint vertex = indices[instance.firstIndex + primitive * 3u + corner] + instance.vertexOffset;
        const float *p = vertices.data() + static_cast<std::size_t>(vertex) * kVertexFloats;
        world[corner] = applyRows(instance.objectToWorld0, instance.objectToWorld1, instance.objectToWorld2,
                                  float3(p[0], p[1], p[2]));
      }
      const float3 edge1 = world[1] - world[0], edge2 = world[2] - world[0];
      const float3 crossed = cross(edge1, edge2);
      const float area = 0.5f * length(crossed);
      if (!(area > 0.0f) || !std::isfinite(area)) continue;
      const double weight = static_cast<double>(area) * power *
                            ((instance.flags & kInstanceDoubleSided) != 0u ? 2.0 : 1.0);
      total += weight;
      PtEmissiveTriangle triangle{};
      triangle.v0Area = float4(world[0], area);
      triangle.edge1Probability = float4(edge1, static_cast<float>(weight));
      triangle.edge2Cdf = float4(edge2, static_cast<float>(total));
      triangle.normal = float4(normalize(crossed), 0.0f);
      const uint uvOffset = ((material.texture.x >> 16u) & 0xFFu) == 1u ? 12u : 10u;
      float2 uv[3];
      for (uint corner = 0; corner < 3; ++corner) {
        const uint vertex = indices[instance.firstIndex + primitive * 3u + corner] + instance.vertexOffset;
        const float *source = vertices.data() + static_cast<std::size_t>(vertex) * kVertexFloats;
        uv[corner] = float2(source[uvOffset], source[uvOffset + 1u]);
      }
      triangle.uv01 = float4(uv[0], uv[1]);
      triangle.uv2 = float4(uv[2].x, uv[2].y, 0.0f, 0.0f);
      triangle.identity = uint4(instanceIndex, primitive, 0u, 0u);
      result.push_back(triangle);
    }
  }
  if (total > 0.0)
    for (PtEmissiveTriangle &triangle : result) {
      triangle.edge1Probability.w = static_cast<float>(triangle.edge1Probability.w / total);
      triangle.edge2Cdf.w = static_cast<float>(triangle.edge2Cdf.w / total);
    }
  if (!result.empty()) result.back().edge2Cdf.w = 1.0f;
  return result;
}

} // namespace pt
