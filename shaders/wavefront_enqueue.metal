#include "shared/prelude.h"

struct OracleRay { float4 originAndMin; float4 directionAndMax; uint4 control; };
struct WavefrontRay { OracleRay ray; uint4 identity; };

kernel void wavefront_enqueue(const device OracleRay* rays [[buffer(0)]],
                              device WavefrontRay* queue [[buffer(1)]],
                              device atomic_uint* counters [[buffer(2)]],
                              constant uint4& control [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
  if (id >= control.x) return;
  const OracleRay ray = rays[id];
  if (!(ray.directionAndMax.w > ray.originAndMin.w)) return;
  const uint slot = atomic_fetch_add_explicit(&counters[0], 1u, memory_order_relaxed);
  if (slot < control.y) {
    queue[slot].ray = ray;
    queue[slot].identity = uint4(id, 0u, 0u, 0u);
  } else {
    atomic_fetch_add_explicit(&counters[1], 1u, memory_order_relaxed);
  }
}
