// The deep traversal stack: tests may trace any tree the conversion accepts.
#define PT_WIDE_BVH_STACK PT_WIDE_BVH_STACK_DEEP
#include "shared/prelude.h"
#include "pt/path.h"
#define PT_TEXTURE_PARAMS array<texture2d<float>, kHitTextureSlots> maps, sampler materialSampler
#define PT_TEXTURE_ARGS maps, materialSampler
#include "pt/texture_msl.h"
#include "pt/wide_bvh.h"

struct OracleRay { float4 originAndMin; float4 directionAndMax; uint4 control; };
struct OracleHit { float4 distanceAndBarycentric; uint4 identity; };

kernel void bvh_wide_oracle(const device OracleRay* rays [[buffer(0)]],
                            device OracleHit* hits [[buffer(1)]],
                            const device TraceInstance* traceInstances [[buffer(2)]],
                            const device Material* materials [[buffer(3)]],
                            const device uint* indices [[buffer(4)]],
                            const device float* vertices [[buffer(5)]],
                            constant uint4& control [[buffer(6)]],
                            const device PtWideNode* bvhNodes [[buffer(7)]],
                            const device float4* bvhTriangles [[buffer(8)]],
                            array<texture2d<float>, kHitTextureSlots> maps [[texture(7)]],
                            sampler materialSampler [[sampler(0)]],
                            uint id [[thread_position_in_grid]]) {
  if (id >= control.x) return;
  const OracleRay input = rays[id];
  OracleHit output;
  output.distanceAndBarycentric = float4(input.directionAndMax.w, 0.0f, 0.0f, 0.0f);
  output.identity = uint4(0u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0u);
  const float minimum = input.originAndMin.w;
  const float maximum = input.directionAndMax.w;
  const float3 origin = input.originAndMin.xyz + input.directionAndMax.xyz * minimum;
  PtHit hit = ptTraceWideBvh(bvhNodes, bvhTriangles, PT_SCENE_ARGS, PT_TEXTURE_ARGS,
      origin, input.directionAndMax.xyz, maximum - minimum,
      input.control.x, input.control.y, float2(-1.0f, 0.0f), input.control.z);
  if (hit.found != 0u) {
    output.distanceAndBarycentric = float4(hit.t + minimum, hit.barycentric, 0.0f);
    output.identity = uint4(1u, hit.instance, hit.primitive, 0u);
  }
  hits[id] = output;
}
