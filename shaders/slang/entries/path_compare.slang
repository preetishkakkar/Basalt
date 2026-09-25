// The hybrid comparison views: the traced image, the raster one, their difference, a split,
// or the samples per pixel as a false colour.

Texture2D<float4> raster;
Texture2D<float4> traced;
WTexture2D<float4> output;
// x view (0 traced, 1 raster, 2 difference, 3 split, 4 samples), y samples, zw image size.
ConstantBuffer<uint4> control;

[shader("compute")]
[numthreads(8, 8, 1)]
void path_compare(uint3 thread: SV_DispatchThreadID) {
  const uint2 id = thread.xy;
  if (id.x >= control.z || id.y >= control.w) return;
  const float3 a = raster[id].xyz;
  const float3 b = traced[id].xyz;
  float3 shown = b;
  if (control.x == 1u) shown = a;
  else if (control.x == 2u) shown = abs(b - a) * 8.0f;
  else if (control.x == 3u) {
    shown = id.x < control.z / 2u ? a : b;
    if (id.x + 1u >= control.z / 2u && id.x <= control.z / 2u + 1u) shown = float3(1.0f);
  } else if (control.x == 4u) {
    const float value = saturate(log2(float(max(control.y, 1u))) / 12.0f);
    shown = float3(saturate(value * 3.0f - 1.5f), saturate(1.0f - abs(value * 3.0f - 1.5f)), saturate(1.5f - value * 3.0f));
  }
  output.Store(id, float4(shown, 1.0f));
}
