// SAH cost of GPU-resident trees in the traversal layouts, one thread per node. Each tree
// occupies a contiguous node range; its cost is 1 for an interior root plus, over every
// node, each child's area relative to the root times one (interior) or its count (leaf).
// That is buildTree's definition in src/pt/Bvh.cpp; the host sums per tree in double.
#include "shared/prelude.h"
#include "pt/bvh_layout.h"

constant float kCostEmptyArea = 1e-30f;

// Surface area as Box::area computes it: zero for an inverted (empty) box.
inline float costArea(float3 low, float3 high) {
  if (!(low.x <= high.x && low.y <= high.y && low.z <= high.z)) return 0.0f;
  const float3 d = high - low;
  return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}

// trees[t] = (first node, node count, 0, 0), sorted by first node; returns the tree of node.
inline uint costTree(const device uint4 *trees, uint treeCount, uint node) {
  uint low = 0u, high = treeCount;
  while (high - low > 1u) {
    const uint middle = (low + high) >> 1u;
    if (trees[middle].x <= node) low = middle;
    else high = middle;
  }
  return low;
}

inline bool costBinaryChild(const device float4 *nodes, uint node, uint side, thread float3 &low,
                            thread float3 &high, thread uint &count) {
  const float4 a = nodes[node * 4u + side * 2u];
  const float4 b = nodes[node * 4u + side * 2u + 1u];
  count = as_type<uint>(b.w);
  low = xyz(a);
  high = xyz(b);
  return !(as_type<uint>(a.w) == kBvhEmpty && count == 0u);
}

// contributions[node] = (sum over the node's children, root term: 1 for an interior root).
kernel void bvh_cost(const device float4 *nodes [[buffer(0)]],
                     const device uint4 *trees [[buffer(1)]],
                     device float2 *contributions [[buffer(2)]],
                     constant uint4 &control [[buffer(3)]],  // x node count, y tree count
                     uint id [[thread_position_in_grid]]) {
  if (id >= control.x) return;
  const uint root = trees[costTree(trees, control.y, id)].x;
  float3 rootLow = float3(3.0e38f), rootHigh = float3(-3.0e38f);
  uint present = 0u;
  for (uint side = 0u; side < 2u; ++side) {
    float3 low = float3(0.0f), high = float3(0.0f);
    uint count = 0u;
    if (costBinaryChild(nodes, root, side, low, high, count)) {
      rootLow = min(rootLow, low);
      rootHigh = max(rootHigh, high);
      present += 1u;
    }
  }
  const float rootArea = max(costArea(rootLow, rootHigh), kCostEmptyArea);
  float sum = 0.0f;
  for (uint side = 0u; side < 2u; ++side) {
    float3 low = float3(0.0f), high = float3(0.0f);
    uint count = 0u;
    if (!costBinaryChild(nodes, id, side, low, high, count)) continue;
    const float relative = costArea(low, high) / rootArea;
    if (count == 0u) sum = sum + relative;
    else sum = sum + relative * float(count);
  }
  // A root holding one leaf and an empty slot is the single-leaf tree: no interior root.
  float rootTerm = 0.0f;
  if (id == root && present == 2u) rootTerm = 1.0f;
  contributions[id] = float2(sum, rootTerm);
}

inline void costWideChild(PtWideNode node, uint child, thread float3 &low, thread float3 &high) {
  const uint4 packed = node.children[child];
  const uint3 words = uint3(packed.x, packed.y, packed.z);
  low = xyz(node.origin) + xyz(node.scale) * float3(words & uint3(0xFFFFu));
  high = xyz(node.origin) + xyz(node.scale) * float3(words >> uint3(16u));
}

inline uint costWideCount(uint data, bool bottom) {
  if ((data & kBvhLeafTag) == 0u) return 0u;
  if (!bottom) return 1u;
  return (data & ~kBvhLeafTag) >> kBvhCountShift;
}

// The same definition over quantized wide nodes, with the decoded (conservative) child
// boxes. trees[0] is the TLAS; the others are bottom levels.
kernel void bvh_wide_cost(const device PtWideNode *nodes [[buffer(0)]],
                          const device uint4 *trees [[buffer(1)]],
                          device float2 *contributions [[buffer(2)]],
                          constant uint4 &control [[buffer(3)]],  // x node count, y tree count
                          uint id [[thread_position_in_grid]]) {
  if (id >= control.x) return;
  const uint tree = costTree(trees, control.y, id);
  const bool bottom = tree != 0u;
  const uint root = trees[tree].x;
  const PtWideNode rootNode = nodes[root];
  const uint rootSlots = as_type<uint>(rootNode.origin.w) & kWideSlotMask;
  float3 rootLow = float3(3.0e38f), rootHigh = float3(-3.0e38f);
  uint rootChildren = 0u, onlyChild = kBvhEmpty;
  for (uint child = 0u; child < rootSlots; ++child) {
    if (rootNode.children[child].w == kBvhEmpty) continue;
    float3 low = float3(0.0f), high = float3(0.0f);
    costWideChild(rootNode, child, low, high);
    rootLow = min(rootLow, low);
    rootHigh = max(rootHigh, high);
    rootChildren += 1u;
    onlyChild = rootNode.children[child].w;
  }
  const float rootArea = max(costArea(rootLow, rootHigh), kCostEmptyArea);
  const PtWideNode node = nodes[id];
  const uint children = as_type<uint>(node.origin.w) & kWideSlotMask;
  float sum = 0.0f;
  for (uint child = 0u; child < children; ++child) {
    if (node.children[child].w == kBvhEmpty) continue;
    float3 low = float3(0.0f), high = float3(0.0f);
    costWideChild(node, child, low, high);
    const float relative = costArea(low, high) / rootArea;
    const uint count = costWideCount(node.children[child].w, bottom);
    if (count == 0u) sum = sum + relative;
    else sum = sum + relative * float(count);
  }
  // A root with a single leaf child is the single-leaf tree: no interior root.
  float rootTerm = 0.0f;
  if (id == root && (rootChildren > 1u || (rootChildren == 1u && (onlyChild & kBvhLeafTag) == 0u)))
    rootTerm = 1.0f;
  contributions[id] = float2(sum, rootTerm);
}
