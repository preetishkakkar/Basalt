// The reconstruction's guides as the denoiser's albedo and normal accumulators, scaled to the
// sample count so their means are the guides.
import pt_path;

StructuredBuffer<PtReconstructionSample> samples;
ConstantBuffer<PathUniforms> uniforms;
WTexture2D<float4> albedoAccumulation;
WTexture2D<float4> normalAccumulation;

[shader("compute")]
[numthreads(8, 8, 1)]
void path_export_guides(uint3 thread: SV_DispatchThreadID) {
  const uint2 id = thread.xy;
  if (float(id.x) >= uniforms.image.x || float(id.y) >= uniforms.image.y) return;
  const uint index = id.y * uint(uniforms.image.x) + id.x;
  const float count = float(uniforms.counts.w + uint(uniforms.image.w));
  const PtReconstructionSample guide = samples[index];
  albedoAccumulation.Store(id, float4(guide.albedo.xyz * count, 0.0f));
  normalAccumulation.Store(id, float4(guide.normalRoughness.xyz * count, 0.0f));
}
