// The GPU BVH builders' ABI and the helpers every builder shares, so they place, order and bound
// primitives identically, and the vertex animation's control record. Shared with their host code.
module pt_bvh_build;

__exported import pt_path;

public __extern_cpp struct BvhBuildDescriptor {
  __extern_cpp uint4 geometry; // first index, vertex offset, triangle count, instance ID
  __extern_cpp uint4 output;   // node base, triangle base, reserved, reserved
};

public __extern_cpp struct BvhBuildControl {
  __extern_cpp uint4 sizes;    // instance count, total triangles, scratch records, node capacity
  __extern_cpp uint4 geometry; // index count, vertex count, traversal stack capacity, layout version
};

public __extern_cpp struct BvhBuildRecord {
  __extern_cpp uint4 identity; // primitive/instance ID, Morton code, reserved, reserved
  __extern_cpp float4 low;
  __extern_cpp float4 high;
};

public __extern_cpp struct BvhBuildStatus {
  __extern_cpp uint4 result; // error, TLAS depth, maximum BLAS depth, maximum builder stack use
  __extern_cpp uint4 counts; // nodes written, triangles written, instances built, stable radix passes
};

// Published by every GPU builder after the serial one (the serial builder's status is widened
// into it on the host). The builders leave cost zero: the host sums the cost kernel's per-node
// contributions instead.
public __extern_cpp struct BvhBuildStatus2 {
  __extern_cpp uint4 result; // error, TLAS depth, maximum BLAS depth, maximum builder stack use (0 if none)
  __extern_cpp uint4 counts; // nodes written, triangles written, instances built, sort passes
  __extern_cpp uint4 extra;  // root node (always 0), reserved, fit iterations used, reserved
  __extern_cpp float4 cost;  // SAH cost of the TLAS, SAH cost summed over the BLASes, reserved, reserved
};

// The parallel builders' scene description: one segment per bottom level (instance order), then
// the TLAS. A segment's records are contiguous; after the (segment, Morton) sort its sorted
// positions are the same range.
public __extern_cpp struct BvhSegment {
  __extern_cpp uint4 range; // first record, record count, first output node, first triangle (0 for the TLAS)
  __extern_cpp uint4 flags; // x: 1 for the TLAS, y: instance (bottom levels), z: internal nodes max(1, count - 1),
                            // w: first internal node (global internal index)
};

public __extern_cpp struct BvhLbvhControl {
  __extern_cpp uint4 sizes;    // records, segments, internal nodes, triangles
  __extern_cpp uint4 geometry; // index count, vertex count, traversal stack capacity, layout version
  __extern_cpp uint4 range;    // instance count, first record of the TLAS, fit iteration (per dispatch), reserved
  __extern_cpp uint4 part;     // the part built: first record (Morton), first internal node, internal nodes,
                               // triangles (the whole scene: 0, 0, internal nodes, triangles; the TLAS alone:
                               // its first record and node, its nodes, 0)
};

public __extern_cpp struct BvhSortControl {
  __extern_cpp uint4 sizes; // key count, tiles (1024 keys each), reserved, reserved
  __extern_cpp uint4 pass;  // pass index, reserved, digit shift, key word (0 low, 1 high)
};

// The wide collapse. A task is one wide node: the binary node it starts from, and whether it
// belongs to a bottom level.
public __extern_cpp struct BvhCollapseControl {
  __extern_cpp uint4 sizes; // task capacity, binary nodes, width (4 or 8), instances
  __extern_cpp uint4 level; // tree level (per dispatch), levels dispatched (a push below them fails), reserved, reserved
};

public __extern_cpp struct BvhCollapseStatus {
  __extern_cpp uint4 result; // error, levels used, wide nodes, maximum traversal stack
};

// The vertex animation (--animate) that dynamic BVH updates are measured with.
public __extern_cpp struct AnimateControl {
  __extern_cpp uint4 counts; // vertices, reserved, reserved, reserved
  __extern_cpp float4 wave;  // amplitude, wavenumber (radians per unit), phase (radians), reserved
};

public static const uint kBvhBuildLayoutVersion = 2u;
public static const uint kBvhBuildErrorNone = 0u;
public static const uint kBvhBuildErrorCapacity = 1u;
public static const uint kBvhBuildErrorGeometry = 2u;
public static const uint kBvhBuildErrorNonFinite = 3u;
public static const uint kBvhBuildErrorDepth = 4u;

// The leading zero bits of x, 32 for zero.
public uint leadingZeros(uint x) { return x == 0u ? 32u : 31u - firstbithigh(x); }

public float3 buildPosition<V : IArray<float>>(V vertices, uint index) {
  const uint base = index * kVertexFloats;
  return float3(vertices[base], vertices[base + 1u], vertices[base + 2u]);
}

// Spreads the low 10 bits of value to every third bit (bit i to bit 3i).
public uint expandMorton(uint value) {
  value &= 0x000003ffu;
  value = (value | (value << 16u)) & 0x030000FFu;
  value = (value | (value << 8u)) & 0x0300F00Fu;
  value = (value | (value << 4u)) & 0x030C30C3u;
  value = (value | (value << 2u)) & 0x09249249u;
  return value;
}

public uint morton3(float3 point, float3 low, float3 high) {
  const float3 extent = high - low;
  float3 unit = float3(0.5f);
  if (extent.x > 0.0f) unit.x = (point.x - low.x) / extent.x;
  if (extent.y > 0.0f) unit.y = (point.y - low.y) / extent.y;
  if (extent.z > 0.0f) unit.z = (point.z - low.z) / extent.z;
  // Below 1 so the 10-bit cell index never reaches 1024.
  unit = clamp(unit, float3(0.0f), float3(0.999999f));
  const uint x = uint(unit.x * 1024.0f);
  const uint y = uint(unit.y * 1024.0f);
  const uint z = uint(unit.z * 1024.0f);
  return expandMorton(x) | (expandMorton(y) << 1u) | (expandMorton(z) << 2u);
}

public float3 transformPoint(TraceInstance instance, float3 point) {
  return float3(rowTimesPoint(instance.objectToWorld0, point), rowTimesPoint(instance.objectToWorld1, point),
                rowTimesPoint(instance.objectToWorld2, point));
}

// A row times (point, 1) as separate products and sums, left to right, as the CPU builder
// rounds them. Not dot(): -fp-mode precise leaves OpDot without NoContraction, and a device that
// fuses it (Intel's does) lands an ulp away from the CPU's tree.
float rowTimesPoint(float4 row, float3 point) { return row.x * point.x + row.y * point.y + row.z * point.z + row.w; }

// A uint whose unsigned order is the float's order, for float min/max through integer atomics:
// negative floats reverse, positive ones move above them.
public uint ptOrderedBits(float f) {
  const uint u = asuint(f);
  return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
}
public float ptOrderedFloat(uint u) { return asfloat((u & 0x80000000u) != 0u ? (u & 0x7FFFFFFFu) : ~u); }
