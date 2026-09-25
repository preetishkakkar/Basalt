// Synthetic animation for measuring dynamic BVH updates (--animate): each vertex's position is
// its rest position displaced by a travelling wave, from a copy of the scene's vertices at rest.
// Only positions move; the other attributes keep their rest values.
import pt_bvh_build;

StructuredBuffer<float> rest;
RWStructuredBuffer<float> vertices;
ConstantBuffer<AnimateControl> control;

[shader("compute")]
[numthreads(64, 1, 1)]
void animate_vertices(uint3 thread: SV_DispatchThreadID) {
  const uint id = thread.x;
  if (id >= control.counts.x) return;
  const uint base = id * kVertexFloats;
  const float3 p = float3(rest[base], rest[base + 1u], rest[base + 2u]);
  const float amplitude = control.wave.x, k = control.wave.y, phase = control.wave.z;
  vertices[base] = p.x + amplitude * sin(k * p.y + phase);
  vertices[base + 1u] = p.y + amplitude * sin(k * p.z + 1.3f * phase);
  vertices[base + 2u] = p.z + amplitude * cos(k * p.x + phase);
}
