#include "shared/prelude.h"
#include "pt/path.h"
#include "pt/temporal.h"

kernel void path_temporal(constant PtTemporalUniforms &uniforms [[buffer(0)]],
    const device PtReconstructionSample *samples [[buffer(1)]],
    const device PtReconstructionSample *previousSamples [[buffer(2)]],
    const device PtTemporalHistory *previous [[buffer(3)]],
    device PtTemporalHistory *output [[buffer(4)]], uint2 id [[thread_position_in_grid]]) {
  if (id.x >= uniforms.image.x || id.y >= uniforms.image.y) return;
  output[id.y * uniforms.image.x + id.x] = ptTemporalPixel(uniforms, id.x, id.y, samples, previousSamples, previous);
}

kernel void path_atrous(constant PtTemporalUniforms &uniforms [[buffer(0)]],
    const device PtReconstructionSample *samples [[buffer(1)]],
    const device PtTemporalHistory *input [[buffer(2)]],
    device PtTemporalHistory *output [[buffer(3)]], uint2 id [[thread_position_in_grid]]) {
  if (id.x >= uniforms.image.x || id.y >= uniforms.image.y) return;
  output[id.y * uniforms.image.x + id.x] = ptAtrousPixel(uniforms, id.x, id.y, samples, input);
}

kernel void path_reconstruct(constant PtTemporalUniforms &uniforms [[buffer(0)]],
    const device PtTemporalHistory *filtered [[buffer(1)]],
    texture2d<float, access::read_write> output [[texture(0)]], uint2 id [[thread_position_in_grid]]) {
  if (id.x >= uniforms.outputExtent.x || id.y >= uniforms.outputExtent.y) return;
  const uint gx = min(id.x * uniforms.image.x / uniforms.outputExtent.x, uniforms.image.x - 1u);
  const uint gy = min(id.y * uniforms.image.y / uniforms.outputExtent.y, uniforms.image.y - 1u);
  const PtTemporalHistory h = filtered[gy * uniforms.image.x + gx];
  const float3 reconstructed = ptSafeColor(xyz(h.diffuse) + xyz(h.specular));
  const float3 raw = ptSafeColor(xyz(output.read(id)));
  // Reconstruction bias vanishes as stationary raw samples converge; at 256 SPP
  // the display is exactly raw. This never changes the raw accumulator itself.
  const float rawWeight = saturate((uniforms.control.x - 32.0f) / 224.0f);
  output.write(float4(mix(reconstructed, raw, rawWeight), 1.0f), id);
}
