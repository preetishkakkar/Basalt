// Shadow pass, depth only; one pipeline draws all four cascades, each draw pushing its
// cascade's view-projection rows.
import basalt_common;

// What every draw of the scene reads.
struct ShadowScene {
  StructuredBuffer<Instance> instances;
  StructuredBuffer<Material> materials;
};

// What changes with the material: the cut-out of a masked one.
struct ShadowMaterial {
  Texture2D<float4> baseColorMap;
  SamplerState materialSampler;
};

ParameterBlock<ShadowScene> scene;
ParameterBlock<ShadowMaterial> material;

struct Cascade {
  float4 rows[4]; // the cascade's view-projection, by rows
};

struct ShadowVertexInput {
  float3 position : POSITION;
  float2 uv0 : TEXCOORD0;
  float2 uv1 : TEXCOORD1;
  float4 color : COLOR;
};

struct ShadowVaryings {
  float4 position : SV_Position;
  float2 uv0 : TEXCOORD0;
  float2 uv1 : TEXCOORD1;
  float4 color : COLOR;
  nointerpolation uint material : MATERIAL;
};

// The draw's first instance selects its record, so the index includes the base instance.
[shader("vertex")]
ShadowVaryings shadow_vertex(ShadowVertexInput input, uint instanceIndex: SV_VulkanInstanceID, uniform Cascade cascade) {
  const Instance instance = scene.instances[instanceIndex];
  const float3 world = applyRows(instance.modelRow0, instance.modelRow1, instance.modelRow2, input.position);
  const float4 point = float4(world, 1.0f);

  ShadowVaryings out;
  out.position = float4(dot(cascade.rows[0], point), dot(cascade.rows[1], point), dot(cascade.rows[2], point),
                        dot(cascade.rows[3], point));
  out.uv0 = input.uv0;
  out.uv1 = input.uv1;
  out.color = input.color;
  out.material = uint(instance.materialAndFlags.x);
  return out;
}

// Masked materials cast their cut-out; opaque ones use the pipeline without a fragment stage.
// Depth only: the entry writes no colour, as the pass has no colour attachment.
[shader("fragment")]
void shadow_fragment(ShadowVaryings input) {
  const Material surface = scene.materials[input.material];
  if (surface.alpha.y > 0.5f) {
    float2 uv = input.uv0;
    if ((surface.texture.x & 0xFFu) == 1u) uv = input.uv1;
    const float alpha = material.baseColorMap.Sample(material.materialSampler, uv).a * surface.baseColorFactor.a * input.color.a;
    if (alpha < surface.alpha.x) discard;
  }
}
