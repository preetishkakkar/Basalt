// Shadow pass, depth only; each cascade's rows are bound at an offset, so one pipeline draws all four.
#include "common.metal"

struct ShadowVertexInput {
  float3 position [[attribute(0)]];
  float2 uv0      [[attribute(1)]];
};

struct ShadowVaryings {
  float4 position [[position]];
  float2 uv0      [[user(locn0)]];
  uint   material [[user(locn1)]] [[flat]];
};

vertex ShadowVaryings shadow_vertex(ShadowVertexInput input [[stage_in]],
                                    uint instanceIndex [[instance_id]],
                                    const device Instance* instances [[buffer(0)]],
                                    const device float4* cascadeRows [[buffer(1)]]) {
  const Instance instance = instances[instanceIndex];
  const float3 world = applyRows(instance.modelRow0, instance.modelRow1, instance.modelRow2,
                                 input.position);
  const float4 point = float4(world, 1.0f);

  ShadowVaryings out;
  out.position = float4(dot(cascadeRows[0], point), dot(cascadeRows[1], point),
                        dot(cascadeRows[2], point), dot(cascadeRows[3], point));
  out.uv0 = input.uv0;
  out.material = uint(instance.materialAndFlags.x);
  return out;
}

// Masked materials cast their cut-out; opaque ones use the pipeline without a fragment stage.
fragment float4 shadow_fragment(ShadowVaryings input [[stage_in]],
                                const device Material* materials [[buffer(0)]],
                                texture2d<float> baseColorMap [[texture(0)]],
                                sampler materialSampler [[sampler(0)]]) {
  const Material material = materials[input.material];
  if (material.alpha.y > 0.5f) {
    const float alpha = baseColorMap.sample(materialSampler, input.uv0).a * material.baseColorFactor.a;
    if (alpha < material.alpha.x) discard_fragment();
  }
  return float4(0.0f, 0.0f, 0.0f, 1.0f);
}
