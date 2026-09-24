#include "shared/prelude.h"
#include "pt/path.h"
#include "pt/reconstruction.h"

kernel void path_export_guides(const device PtReconstructionSample* samples [[buffer(0)]],
                               constant PathUniforms& uniforms [[buffer(1)]],
                               texture2d<float, access::write> albedoAccumulation [[texture(0)]],
                               texture2d<float, access::write> normalAccumulation [[texture(1)]],
                               uint2 id [[thread_position_in_grid]]) {
  if (float(id.x) >= uniforms.image.x || float(id.y) >= uniforms.image.y) return;
  const uint index = id.y * uint(uniforms.image.x) + id.x;
  const float count = float(uniforms.counts.w + uint(uniforms.image.w));
  const PtReconstructionSample guide = samples[index];
  albedoAccumulation.write(float4(guide.albedo.xyz * count, 0.0f), id);
  normalAccumulation.write(float4(guide.normalRoughness.xyz * count, 0.0f), id);
}
