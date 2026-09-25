// Clustered light culling: screen tiles by exponential depth slices, one thread per cell
// recording which lights reach it in a fixed run per cell, so no atomics. A zero-range
// light reaches every cell.
import basalt_common;

struct ClusterUniforms {
  float4x4 view;
  float4 grid;       // x tiles across, y tiles down, z depth slices, w lights a cell can hold
  float4 projection; // x view width per unit depth, y view height per unit depth, z near, w far
  float4 counts;     // x lights in the scene, y cells in the grid, zw unused
};

StructuredBuffer<Light> lights;
// A count per cell, then each cell's fixed run of indices.
RWStructuredBuffer<uint> clusters;
ConstantBuffer<ClusterUniforms> cluster;

[shader("compute")]
[numthreads(64, 1, 1)]
void cull_lights(uint3 thread: SV_DispatchThreadID) {
  const uint id = thread.x;
  const uint cellCount = uint(cluster.counts.y);
  if (id >= cellCount) return;

  const uint tilesX = uint(cluster.grid.x);
  const uint tilesY = uint(cluster.grid.y);
  const uint capacity = uint(cluster.grid.w);
  const uint tileX = id % tilesX;
  const uint tileY = (id / tilesX) % tilesY;
  const uint slice = id / (tilesX * tilesY);

  // The cell's view-space box from its eight corners. Rows count from the top of the image,
  // as a fragment reads its position; the negative viewport puts the top at +y in clip space.
  const float2 lowNdc =
      float2(float(tileX) / cluster.grid.x * 2.0f - 1.0f, 1.0f - float(tileY + 1u) / cluster.grid.y * 2.0f);
  const float2 highNdc =
      float2(float(tileX + 1u) / cluster.grid.x * 2.0f - 1.0f, 1.0f - float(tileY) / cluster.grid.y * 2.0f);
  const float nearPlane = cluster.projection.z;
  const float ratio = max(cluster.projection.w / nearPlane, 1.0f + 1e-4f);
  const float nearDepth = nearPlane * pow(ratio, float(slice) / cluster.grid.z);
  // The last slice gets a little room past the far plane, and not much: a huge box keeps
  // lights by index rather than nearness.
  const float farDepth = slice + 1u >= uint(cluster.grid.z) ? cluster.projection.w * 4.0f
                                                            : nearPlane * pow(ratio, float(slice + 1u) / cluster.grid.z);

  float3 lowest = float3(1e30f);
  float3 highest = float3(-1e30f);
  for (uint corner = 0u; corner < 8u; ++corner) {
    const float depth = (corner & 4u) != 0u ? farDepth : nearDepth;
    const float x = (corner & 1u) != 0u ? highNdc.x : lowNdc.x;
    const float y = (corner & 2u) != 0u ? highNdc.y : lowNdc.y;
    const float3 point = float3(x * depth * cluster.projection.x, y * depth * cluster.projection.y, -depth);
    lowest = min(lowest, point);
    highest = max(highest, point);
  }

  const uint lightCount = uint(cluster.counts.x);
  const uint base = cellCount + id * capacity;
  uint found = 0u;
  for (uint i = 0u; i < lightCount && found < capacity; ++i) {
    const Light light = lights[i];
    const float4 centre = mul(cluster.view, float4(light.position.xyz, 1.0f));
    const float range = light.position.w;
    const float3 nearest = clamp(centre.xyz, lowest, highest);
    const float3 offset = centre.xyz - nearest;
    // Range zero is glTF for no falloff.
    const bool reaches = range <= 0.0f || dot(offset, offset) <= range * range;
    if (reaches) {
      clusters[base + found] = i;
      found += 1u;
    }
  }
  clusters[id] = found;
}
