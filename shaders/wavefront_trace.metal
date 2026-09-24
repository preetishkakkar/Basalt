#include "shared/prelude.h"
#include "pt/path.h"
#define PT_TEXTURE_PARAMS array<texture2d<float>, kHitTextureSlots> maps, sampler materialSampler
#define PT_TEXTURE_ARGS maps, materialSampler
#include "pt/texture_msl.h"
#include "pt/bvh.h"

struct OracleRay { float4 originAndMin; float4 directionAndMax; uint4 control; };
struct WavefrontRay { OracleRay ray; uint4 identity; };
struct OracleHit { float4 distanceAndBarycentric; uint4 identity; };

kernel void wavefront_trace(const device WavefrontRay* queue [[buffer(0)]],
                            const device uint* counters [[buffer(1)]],
                            device OracleHit* hits [[buffer(2)]],
                            const device TraceInstance* traceInstances [[buffer(3)]],
                            const device Material* materials [[buffer(4)]],
                            const device uint* indices [[buffer(5)]],
                            const device float* vertices [[buffer(6)]],
                            constant uint4& control [[buffer(7)]],
                            const device float4* bvhNodes [[buffer(8)]],
                            const device float4* bvhTriangles [[buffer(9)]],
                            array<texture2d<float>, kHitTextureSlots> maps [[texture(7)]],
                            sampler materialSampler [[sampler(0)]],
                            uint id [[thread_position_in_grid]]) {
  if (id >= min(counters[0], control.y)) return;
  const WavefrontRay queued = queue[id];
  const OracleRay input = queued.ray;
  OracleHit output;
  output.distanceAndBarycentric = float4(input.directionAndMax.w, 0.0f, 0.0f, 0.0f);
  output.identity = uint4(0u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0u);
  const float minimum = input.originAndMin.w;
  const float3 origin = input.originAndMin.xyz + input.directionAndMax.xyz * minimum;
  const PtHit hit = ptTraceBvh(bvhNodes, bvhTriangles, PT_SCENE_ARGS, PT_TEXTURE_ARGS,
      origin, input.directionAndMax.xyz, input.directionAndMax.w - minimum,
      input.control.x, input.control.y, float2(-1.0f, 0.0f), input.control.z);
  if (hit.found != 0u) {
    output.distanceAndBarycentric = float4(hit.t + minimum, hit.barycentric, 0.0f);
    output.identity = uint4(1u, hit.instance, hit.primitive, 0u);
  }
  hits[queued.identity.x] = output;
}
