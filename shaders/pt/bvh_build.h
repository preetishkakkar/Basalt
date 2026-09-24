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

PT_CONSTANT uint kBvhBuildLayoutVersion = 1u;
PT_CONSTANT uint kBvhBuildErrorNone = 0u;
PT_CONSTANT uint kBvhBuildErrorCapacity = 1u;
PT_CONSTANT uint kBvhBuildErrorGeometry = 2u;
PT_CONSTANT uint kBvhBuildErrorNonFinite = 3u;
PT_CONSTANT uint kBvhBuildErrorDepth = 4u;

