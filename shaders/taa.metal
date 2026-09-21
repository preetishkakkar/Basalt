// Temporal antialiasing. Each frame is rendered with a sub-pixel jitter and blended with
// the last: reprojected and clamped to the neighbourhood while moving, averaged exactly
// while still so it converges rather than settling at a floor.
#include "common.metal"

struct TemporalUniforms {
  float4x4 inverseViewProjection;   // The jittered transform the depth was rendered with.
  float4x4 viewProjection;          // This frame, with the jitter taken out.
  float4x4 previousViewProjection;  // The frame before, with the jitter taken out.
  float4 target;                    // xy size, zw texel size
  float4 parameters;                // x frames accumulated, y feedback, z unused, w mode (0 off, 1 temporal)
  float4 camera;                    // xyz the eye, w how far away the sky is treated as being
};

static float3 finiteOr(float3 value, float3 fallback) {
  // Not a blend: mix() is x + (y - x) * a, and that is NaN whenever x is.
  float3 result = value;
  if (any(isnan(value)) || any(isinf(value))) result = fallback;
  return result;
}

kernel void resolve_temporal(texture2d<float> currentColor [[texture(0)]],
                             texture2d<float> history [[texture(1)]],
                             texture2d<float> depthBuffer [[texture(2)]],
                             texture2d<float, access::write> resolved [[texture(3)]],
                             // w holds the nearest surface's depth, blended ones included.
                             texture2d<float> surfaceWeight [[texture(4)]],
                             constant TemporalUniforms& taa [[buffer(0)]],
                             sampler clampSampler [[sampler(0)]],
                             sampler pointSampler [[sampler(1)]],
                             uint2 id [[thread_position_in_grid]]) {
  if (float(id.x) >= taa.target.x || float(id.y) >= taa.target.y) return;
  const float2 uv = (float2(float(id.x), float(id.y)) + 0.5f) * taa.target.zw;

  float3 current = currentColor.sample(pointSampler, uv, level(0.0f)).rgb;
  // A non-finite frame gets no weight and never enters the history.
  const bool spoiledFrame = any(isnan(current)) || any(isinf(current));
  const float spoiled = spoiledFrame ? 1.0f : 0.0f;
  if (spoiledFrame) current = float3(0.0f);

  float3 outColour = current;
  if (taa.parameters.w > 0.5f) {
    const float accumulated = taa.parameters.x;
    if (accumulated > 0.5f) {
      // Still: exact average, unclamped so it can converge.
      const float3 previous = finiteOr(history.sample(pointSampler, uv, level(0.0f)).rgb, current);
      outColour = mix(previous, current, mix(1.0f / (accumulated + 1.0f), 0.0f, spoiled));
    } else {
      // Moving: reproject through the un-jittered transforms, so only geometry motion remains.
      float depth = depthBuffer.sample(pointSampler, uv, level(0.0f)).r;
      // A blended surface's depth rides in weight.w; nearer reads larger under reverse Z.
      const float nearest = surfaceWeight.sample(pointSampler, uv, level(0.0f)).w;
      if (nearest > depth * 1.001f) depth = nearest;
      // The sky has no depth: reproject it as a far direction, so only the camera's turn matters.
      float3 world = float3(0.0f);
      if (depth > 0.0f) {
        world = reconstructWorld(float2(uv.x, 1.0f - uv.y), depth, taa.inverseViewProjection);
      } else {
        const float3 onNearPlane =
            reconstructWorld(float2(uv.x, 1.0f - uv.y), 1.0f, taa.inverseViewProjection);
        world = taa.camera.xyz + normalize(onNearPlane - taa.camera.xyz) * taa.camera.w;
      }
      {
        const float4 nowClip = taa.viewProjection * float4(world, 1.0f);
        const float4 thenClip = taa.previousViewProjection * float4(world, 1.0f);
        if (nowClip.w > 0.0f && thenClip.w > 0.0f) {
          const float2 nowNdc = nowClip.xy / nowClip.w;
          const float2 thenNdc = thenClip.xy / thenClip.w;
          const float2 nowUV = float2(nowNdc.x * 0.5f + 0.5f, 0.5f - nowNdc.y * 0.5f);
          const float2 thenUV = float2(thenNdc.x * 0.5f + 0.5f, 0.5f - thenNdc.y * 0.5f);
          const float2 historyUV = uv - (nowUV - thenUV);
          const bool onScreen = historyUV.x >= 0.0f && historyUV.x <= 1.0f && historyUV.y >= 0.0f &&
                                historyUV.y <= 1.0f;
          if (onScreen) {
            // Clamp the history to the current neighbourhood, or a revealed surface drags old colour.
            float3 lowest = current;
            float3 highest = current;
            for (int y = -1; y <= 1; ++y) {
              for (int x = -1; x <= 1; ++x) {
                const float2 tap = uv + float2(float(x), float(y)) * taa.target.zw;
                // A non-finite tap would poison the bounds.
                const float3 neighbour =
                    finiteOr(currentColor.sample(pointSampler, tap, level(0.0f)).rgb, current);
                lowest = min(lowest, neighbour);
                highest = max(highest, neighbour);
              }
            }
            const float3 sampled =
                finiteOr(history.sample(clampSampler, historyUV, level(0.0f)).rgb, current);
            const float3 previous = clamp(sampled, lowest, highest);
            // A spoiled frame gets no weight.
            outColour = mix(previous, current, mix(taa.parameters.y, 0.0f, spoiled));
          }
        }
      }
    }
  }
  resolved.write(float4(outColour, 1.0f), id);
}
