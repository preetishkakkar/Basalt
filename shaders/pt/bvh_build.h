// Stable ABI for the GPU LBVH builder. Shared with its Windows host.
#pragma once
#include "path.h"

struct BvhBuildDescriptor {
  uint4 geometry; // first index, vertex offset, triangle count, instance ID
  uint4 output;   // node base, triangle base, reserved, reserved
};

struct BvhBuildControl {
  uint4 sizes;    // instance count, total triangles, scratch records, node capacity
  uint4 geometry; // index count, vertex count, traversal stack capacity, layout version
};

struct BvhBuildRecord {
  uint4 identity; // primitive/instance ID, Morton code, reserved, reserved
  float4 low;
  float4 high;
};

struct BvhBuildStatus {
  uint4 result; // error, TLAS depth, maximum BLAS depth, maximum builder stack use
  uint4 counts; // nodes written, triangles written, instances built, stable radix passes
};

// Published by every GPU builder after the serial one (the serial builder's status is
// widened into it on the host). The costs are filled on the host by the cost kernel.
struct BvhBuildStatus2 {
  uint4 result; // error, TLAS depth, maximum BLAS depth, maximum builder stack use (0 if none)
  uint4 counts; // nodes written, triangles written, instances built, sort passes
  uint4 extra;  // root node (always 0), dispatches, fit iterations used, reserved
  float4 cost;  // SAH cost of the TLAS, SAH cost summed over the BLASes, reserved, reserved
};

// The parallel builders' scene description: one segment per bottom level (instance order),
// then the TLAS. A segment's records are contiguous; after the (segment, Morton) sort its
// sorted positions are the same range.
struct BvhSegment {
  uint4 range; // first record, record count, first output node, first triangle (0 for the TLAS)
  uint4 flags; // x: 1 for the TLAS, y: instance (bottom levels), z: internal nodes max(1, count - 1),
               // w: first internal node (global internal index)
};

struct BvhLbvhControl {
  uint4 sizes;    // records, segments, internal nodes, triangles
  uint4 geometry; // index count, vertex count, traversal stack capacity, layout version
  uint4 range;    // instance count, first record of the TLAS, fit iteration (per dispatch), reserved
  uint4 part;     // the part built: first record (Morton), first internal node, internal nodes,
                  // triangles (the whole scene: 0, 0, internal nodes, triangles; the TLAS alone:
                  // its first record and node, its nodes, 0)
};

struct BvhSortControl {
  uint4 sizes; // key count, tiles (1024 keys each), reserved, reserved
  uint4 pass;  // pass index, reserved, digit shift, key word (0 low, 1 high)
};

// The wide collapse (bvh_collapse.metal). A task is one wide node: the binary node it starts
// from, and whether it belongs to a bottom level.
struct BvhCollapseControl {
  uint4 sizes; // task capacity, binary nodes, width (4 or 8), instances
  uint4 level; // tree level (per dispatch), levels dispatched (a push below them fails), reserved, reserved
};

struct BvhCollapseStatus {
  uint4 result; // error, levels used, wide nodes, maximum traversal stack
};

PT_CONSTANT uint kBvhBuildLayoutVersion = 2u;
PT_CONSTANT uint kBvhBuildErrorNone = 0u;
PT_CONSTANT uint kBvhBuildErrorCapacity = 1u;
PT_CONSTANT uint kBvhBuildErrorGeometry = 2u;
PT_CONSTANT uint kBvhBuildErrorNonFinite = 3u;
PT_CONSTANT uint kBvhBuildErrorDepth = 4u;

// Shared by every GPU builder, so they place, order and bound primitives identically.
inline float3 buildPosition(const device float *vertices, uint index) {
  const uint base = index * kVertexFloats;
  return float3(vertices[base], vertices[base + 1u], vertices[base + 2u]);
}

inline uint expandMorton(uint value) {
  value &= 0x000003ffu;
  value = (value | (value << 16u)) & 0x030000FFu;
  value = (value | (value << 8u)) & 0x0300F00Fu;
  value = (value | (value << 4u)) & 0x030C30C3u;
  value = (value | (value << 2u)) & 0x09249249u;
  return value;
}

inline uint morton3(float3 point, float3 low, float3 high) {
  const float3 extent = high - low;
  float3 unit = float3(0.5f);
  if (extent.x > 0.0f) unit.x = (point.x - low.x) / extent.x;
  if (extent.y > 0.0f) unit.y = (point.y - low.y) / extent.y;
  if (extent.z > 0.0f) unit.z = (point.z - low.z) / extent.z;
  unit = clamp(unit, float3(0.0f), float3(0.999999f));
  const uint x = uint(unit.x * 1024.0f);
  const uint y = uint(unit.y * 1024.0f);
  const uint z = uint(unit.z * 1024.0f);
  return expandMorton(x) | (expandMorton(y) << 1u) | (expandMorton(z) << 2u);
}

inline float3 transformPoint(TraceInstance instance, float3 point) {
  const float4 p = float4(point.x, point.y, point.z, 1.0f);
  return float3(dot(instance.objectToWorld0, p), dot(instance.objectToWorld1, p),
                dot(instance.objectToWorld2, p));
}

// A uint whose unsigned order is the float's order, for float min/max through integer
// atomics: negative floats reverse, positive ones move above them.
inline uint ptOrderedBits(float f) {
  const uint u = as_type<uint>(f);
  return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
}
inline float ptOrderedFloat(uint u) {
  return as_type<float>((u & 0x80000000u) != 0u ? (u & 0x7FFFFFFFu) : ~u);
}
