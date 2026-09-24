#include "shared/prelude.h"

kernel void path_compare(texture2d<float> raster [[texture(0)]],
                         texture2d<float> traced [[texture(1)]],
                         texture2d<float, access::write> output [[texture(2)]],
                         constant uint4& control [[buffer(0)]],
                         uint2 id [[thread_position_in_grid]]) {
  if (id.x >= control.z || id.y >= control.w) return;
  const float3 a = raster.read(id).xyz;
  const float3 b = traced.read(id).xyz;
  float3 shown = b;
  if (control.x == 1u) shown = a;
  else if (control.x == 2u) shown = abs(b - a) * 8.0f;
  else if (control.x == 3u) {
    if (id.x < control.z / 2u) shown = a;
    else shown = b;
    if (id.x + 1u >= control.z / 2u && id.x <= control.z / 2u + 1u) shown = float3(1.0f);
  } else if (control.x == 4u) {
    const float value = saturate(log2(float(max(control.y, 1u))) / 12.0f);
    shown = float3(saturate(value * 3.0f - 1.5f),
                   saturate(1.0f - abs(value * 3.0f - 1.5f)),
                   saturate(1.5f - value * 3.0f));
  }
  output.write(float4(shown, 1.0f), id);
}
