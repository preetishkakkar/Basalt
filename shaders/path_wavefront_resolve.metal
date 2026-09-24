#include "shared/prelude.h"
#include "pt/path.h"
#include "pt/reconstruction.h"
#include "pt/wavefront.h"

// Adds one batch's pixels to the accumulators. Each pixel's S paths are consecutive in the
// batch and are summed in sample order, then added once, exactly as the megakernel does:
// no atomics, and a non-finite path contributes no radiance but still counts. With flag
// bit 0 clear (the target was reached) the pass only rewrites the mean, for every pixel.
//
// A workgroup covers kResolveGroup / S whole pixels when S <= kResolveGroup: its threads
// first load the pixels' results into threadgroup memory with consecutive (coalesced)
// reads, then one thread per pixel sums them in sample order. Larger S reads directly.
PT_CONSTANT uint kResolveGroup = 128u;

inline float3 ptWaveCostColour(uint2 cost, float mode) {
  // BVH cost views: the path's own traversal work, on the megakernel's fixed scale.
  const float value = saturate(mode < 2.5f ? float(cost.x) / 128.0f : float(cost.y) / 64.0f);
  return float3(saturate(value * 3.0f - 1.5f), saturate(1.0f - abs(value * 3.0f - 1.5f)), saturate(1.5f - value * 3.0f));
}

kernel void path_wavefront_resolve(constant PathUniforms& uniforms [[buffer(0)]],
                                   const device PtWaveResult* results [[buffer(1)]],
                                   device PtReconstructionSample* guides [[buffer(2)]],
                                   const device uint4* waveControl [[buffer(3)]],
                                   const device uint2* costs [[buffer(4)]],
                                   const device PtWaveGuide* waveGuides [[buffer(5)]],
                                   texture2d<float, access::read_write> accumulation [[texture(0)]],
                                   texture2d<float, access::write> output [[texture(1)]],
                                   texture2d<float, access::read_write> albedoAccumulation [[texture(2)]],
                                   texture2d<float, access::read_write> normalAccumulation [[texture(3)]],
                                   uint lane [[thread_index_in_threadgroup]],
                                   uint group [[threadgroup_position_in_grid]]) {
  threadgroup float4 radianceTile[128];
  threadgroup float4 albedoTile[128];
  threadgroup float4 normalTile[128];
  const uint4 control = waveControl[0];
  const uint4 frame = waveControl[1];
  const bool accumulate = (frame.z & 1u) != 0u;
  const uint samples = max(frame.y, 1u);
  const uint batchPixels = accumulate ? control.w / samples : uint(uniforms.image.x) * uint(uniforms.image.y);
  const bool tiled = accumulate && samples <= kResolveGroup;
  const uint pixelsPerGroup = tiled ? kResolveGroup / samples : kResolveGroup;
  const uint firstInGroup = group * pixelsPerGroup;
  // Load phase (uniform for the whole group: no thread returns before the barrier).
  if (tiled) {
    const uint element = lane;
    const uint pixelInGroup = element / samples;
    if (pixelInGroup < pixelsPerGroup && firstInGroup + pixelInGroup < batchPixels) {
      const uint local = (frame.w + firstInGroup + pixelInGroup) * samples + element % samples - control.z;
      const PtWaveResult result = results[local];
      float4 radiance = result.radiance, albedo = result.albedo, normal = result.normal;
      if (uniforms.image.z > 1.5f) {
        radiance = float4(ptWaveCostColour(costs[local], uniforms.image.z), 0.0f);
        albedo = float4(0.0f);
        normal = float4(0.0f);
      }
      radianceTile[element] = radiance;
      albedoTile[element] = albedo;
      normalTile[element] = normal;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane >= pixelsPerGroup || firstInGroup + lane >= batchPixels) return;
  const uint pixel = (accumulate ? frame.w : 0u) + firstInGroup + lane;
  const uint2 xy = uint2(pixel % uint(uniforms.image.x), pixel / uint(uniforms.image.x));
  if (!accumulate) {
    const float4 total = accumulation.read(xy);
    output.write(float4(total.xyz / max(total.w, 1.0f), 1.0f), xy);
    return;
  }
  float3 sum = float3(0.0f), albedoSum = float3(0.0f), normalSum = float3(0.0f);
  float3 lastRadiance = float3(0.0f);
  const uint first = pixel * samples - control.z;
  for (uint s = 0u; s < samples; ++s) {
    float3 radiance = float3(0.0f), albedo = float3(0.0f), normal = float3(0.0f);
    if (tiled) {
      radiance = radianceTile[lane * samples + s].xyz;
      albedo = albedoTile[lane * samples + s].xyz;
      normal = normalTile[lane * samples + s].xyz;
    } else {
      const PtWaveResult result = results[first + s];
      radiance = result.radiance.xyz;
      albedo = result.albedo.xyz;
      normal = result.normal.xyz;
      if (uniforms.image.z > 1.5f) {
        radiance = ptWaveCostColour(costs[first + s], uniforms.image.z);
        albedo = float3(0.0f);
        normal = float3(0.0f);
      }
    }
    lastRadiance = radiance;
    if (!(isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z) || isinf(radiance.x) || isinf(radiance.y) ||
          isinf(radiance.z)))
      sum = sum + radiance;
    albedoSum = albedoSum + albedo;
    normalSum = normalSum + normal;
  }
  float4 total = float4(sum, float(samples));
  float4 albedoTotal = float4(albedoSum, 0.0f), normalTotal = float4(normalSum, 0.0f);
  // A restarted sample offset replaces the sums (execution, camera or backend changes).
  if (uniforms.counts.w > 0u) {
    total = total + accumulation.read(xy);
    albedoTotal = albedoTotal + albedoAccumulation.read(xy);
    normalTotal = normalTotal + normalAccumulation.read(xy);
  }
  accumulation.write(total, xy);
  albedoAccumulation.write(albedoTotal, xy);
  normalAccumulation.write(normalTotal, xy);
  if (uniforms.image.z > 0.5f && uniforms.image.z < 1.5f) {
    // The guide path is the pixel's last sample.
    PtReconstructionSample guide = guides[pixel];
    const float3 diffuse = waveGuides[pixel].diffuseRadiance.xyz;
    guide.diffuse = float4(diffuse, 0.0f);
    guide.specular = float4(max(lastRadiance - diffuse, float3(0.0f)), 0.0f);
    guides[pixel] = guide;
  }
  output.write(float4(total.xyz / max(total.w, 1.0f), 1.0f), xy);
}
