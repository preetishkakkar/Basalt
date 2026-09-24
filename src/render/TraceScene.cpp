#include "render/TraceScene.h"

namespace basalt {
namespace {

pt::float4 toFloat4(const Vec4 &v) { return pt::float4(v.x, v.y, v.z, v.w); }

} // namespace

TraceScene buildTraceScene(const Scene &scene, const std::vector<std::uint32_t> &materialSlots) {
  TraceScene trace;
  for (std::uint32_t i = 0; i < scene.primitives.size(); ++i) {
    const Primitive &primitive = scene.primitives[i];
    if (primitive.indexCount < 3) continue;
    const Material &material = scene.materials[primitive.material];
    const Mat4 &model = primitive.transform;
    const Mat4 worldToObject = inverse(model);
    const Mat4 normal = normalMatrix(model);

    pt::TraceInstance instance{};
    instance.objectToWorld0 = toFloat4(model.row(0));
    instance.objectToWorld1 = toFloat4(model.row(1));
    instance.objectToWorld2 = toFloat4(model.row(2));
    instance.worldToObject0 = toFloat4(worldToObject.row(0));
    instance.worldToObject1 = toFloat4(worldToObject.row(1));
    instance.worldToObject2 = toFloat4(worldToObject.row(2));
    instance.normalToWorld0 = toFloat4(normal.row(0));
    instance.normalToWorld1 = toFloat4(normal.row(1));
    instance.normalToWorld2 = toFloat4(normal.row(2));
    instance.firstIndex = primitive.firstIndex;
    instance.vertexOffset = static_cast<std::uint32_t>(primitive.vertexOffset);
    instance.material = primitive.material;
    instance.slots = primitive.material < materialSlots.size() ? materialSlots[primitive.material] : 0;
    const bool blended = material.alphaMode == AlphaMode::Blend;
    instance.mask = blended ? pt::kRayMaskBlended
                    : static_cast<int>(i) == scene.groundPrimitive ? pt::kRayMaskGround
                                                                    : pt::kRayMaskScene;
    instance.flags = (material.alphaMode == AlphaMode::Mask ? pt::kInstanceMasked : 0u) |
                     (blended ? pt::kInstanceBlended : 0u) |
                     (material.doubleSided ? pt::kInstanceDoubleSided : 0u);
    const Vec4 x = model.row(0), y = model.row(1), z = model.row(2);
    const float determinant = x.x * (y.y * z.z - y.z * z.y) - x.y * (y.x * z.z - y.z * z.x) +
                              x.z * (y.x * z.y - y.y * z.x);
    if (determinant < 0.0f) instance.flags |= pt::kInstanceMirrored;
    trace.primitives.push_back(i);
    trace.instances.push_back(instance);
  }
  return trace;
}

} // namespace basalt
