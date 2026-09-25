// The path tracer's temporal reconstruction: reprojected history per pixel, an a-trous pass,
// and the blend of the reconstruction into the displayed mean (pt_temporal has the maths).
import pt_temporal;

ConstantBuffer<PtTemporalUniforms> uniforms;
StructuredBuffer<PtReconstructionSample> samples;
StructuredBuffer<PtReconstructionSample> previousSamples;
StructuredBuffer<PtTemporalHistory> previous;
StructuredBuffer<PtTemporalHistory> input;    // the a-trous pass's
StructuredBuffer<PtTemporalHistory> filtered; // the reconstruction's
RWStructuredBuffer<PtTemporalHistory> history;
// The frame's lit image, half-float RGBA: read for the raw mean, written blended.
[format("rgba16f")] RWTexture2D<float4> output;

[shader("compute")]
[numthreads(8, 8, 1)]
void path_temporal(uint3 id: SV_DispatchThreadID) {
  if (id.x >= uniforms.image.x || id.y >= uniforms.image.y) return;
  history[id.y * uniforms.image.x + id.x] = ptTemporalPixel(uniforms, id.x, id.y, samples, previousSamples, previous);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void path_atrous(uint3 id: SV_DispatchThreadID) {
  if (id.x >= uniforms.image.x || id.y >= uniforms.image.y) return;
  history[id.y * uniforms.image.x + id.x] = ptAtrousPixel(uniforms, id.x, id.y, samples, input);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void path_reconstruct(uint3 thread: SV_DispatchThreadID) {
  const uint2 id = thread.xy;
  if (id.x >= uniforms.outputExtent.x || id.y >= uniforms.outputExtent.y) return;
  const uint gx = min(id.x * uniforms.image.x / uniforms.outputExtent.x, uniforms.image.x - 1u);
  const uint gy = min(id.y * uniforms.image.y / uniforms.outputExtent.y, uniforms.image.y - 1u);
  const PtTemporalHistory h = filtered[gy * uniforms.image.x + gx];
  const float3 reconstructed = ptSafeColor(h.diffuse.xyz + h.specular.xyz);
  const float3 raw = ptSafeColor(output[id].xyz);
  // Reconstruction bias vanishes as stationary raw samples converge; at 256 SPP the display is
  // exactly raw. This never changes the raw accumulator itself.
  const float rawWeight = saturate((uniforms.control.x - 32.0f) / 224.0f);
  output[id] = float4(lerp(reconstructed, raw, rawWeight), 1.0f);
}
