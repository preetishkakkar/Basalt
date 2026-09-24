// glTF sampler selection shared by the GPU intersectors: level zero (V6), or the ray-cone
// level of detail, trilinear between mip levels.
#pragma once
#include "raycone.h"

constexpr sampler ptRepeatLinear(coord::normalized, s_address::repeat, t_address::repeat, filter::linear, mip_filter::linear);
constexpr sampler ptClampLinear(coord::normalized, s_address::clamp_to_edge, t_address::clamp_to_edge, filter::linear, mip_filter::linear);
constexpr sampler ptMirrorLinear(coord::normalized, s_address::mirrored_repeat, t_address::mirrored_repeat, filter::linear, mip_filter::linear);
constexpr sampler ptRepeatNearest(coord::normalized, s_address::repeat, t_address::repeat, filter::nearest, mip_filter::linear);
constexpr sampler ptClampNearest(coord::normalized, s_address::clamp_to_edge, t_address::clamp_to_edge, filter::nearest, mip_filter::linear);
constexpr sampler ptMirrorNearest(coord::normalized, s_address::mirrored_repeat, t_address::mirrored_repeat, filter::nearest, mip_filter::linear);
float ptTextureFootprintOf(PT_TEXTURE_PARAMS, uint slot, float lodBase) {
  return ptTextureFootprint(lodBase, float(maps[slot].get_width()), float(maps[slot].get_height()));
}
// lodBase from ptConeLodBase; kPtLevelZero samples level 0 exactly.
float4 ptSampleTexture(PT_TEXTURE_PARAMS, uint slot, float2 uv, uint samplerCode, float lodBase) {
  float lod = 0.0f;
  if (lodBase > kPtLevelZero)
    lod = ptTextureLevel(lodBase, float(maps[slot].get_width()), float(maps[slot].get_height()),
                         float(maps[slot].get_num_mip_levels()));
  if ((samplerCode & 16u) != 0u) {
    if ((samplerCode & 3u) == 1u) return maps[slot].sample(ptClampNearest, uv, level(lod));
    if ((samplerCode & 3u) == 2u) return maps[slot].sample(ptMirrorNearest, uv, level(lod));
    return maps[slot].sample(ptRepeatNearest, uv, level(lod));
  }
  if ((samplerCode & 3u) == 1u) return maps[slot].sample(ptClampLinear, uv, level(lod));
  if ((samplerCode & 3u) == 2u) return maps[slot].sample(ptMirrorLinear, uv, level(lod));
  return maps[slot].sample(ptRepeatLinear, uv, level(lod));
}
