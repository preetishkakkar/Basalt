// Synthetic animation for measuring dynamic BVH updates (--animate): each vertex's position is
// its rest position displaced by a travelling wave, from a copy of the scene's vertices at rest.
// Only positions move; the other attributes keep their rest values.
#include "shared/prelude.h"
#include "pt/bvh_build.h"
#include "pt/animate.h"

kernel void animate_vertices(const device float *rest [[buffer(0)]],
                             device float *vertices [[buffer(1)]],
                             constant AnimateControl &control [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
  if (id >= control.counts.x) return;
  const float3 p = buildPosition(rest, id);
  const float amplitude = control.wave.x, k = control.wave.y, phase = control.wave.z;
  const uint base = id * kVertexFloats;
  vertices[base] = p.x + amplitude * sin(k * p.y + phase);
  vertices[base + 1u] = p.y + amplitude * sin(k * p.z + 1.3f * phase);
  vertices[base + 2u] = p.z + amplitude * cos(k * p.x + phase);
}
