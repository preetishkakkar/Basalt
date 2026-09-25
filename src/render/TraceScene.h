// The per-instance table every ray reads, in the order the acceleration structures use.
#pragma once
#include "pt/Shared.h"
#include "scene/Scene.h"

#include <cstdint>
#include <vector>

namespace basalt {

struct TraceScene {
  std::vector<std::uint32_t> primitives;    // The scene primitive behind each instance.
  std::vector<pt::TraceInstance> instances;
};

// Every primitive with a triangle becomes one instance. Blended ones answer only to
// kRayMaskBlended, so the rasteriser's rays, which never ask for it, pass through them.
TraceScene buildTraceScene(const Scene &scene, const std::vector<std::uint32_t> &materialSlots);

// An instance row's object-to-world, world-to-object and normal matrices and its mirrored flag
// for the model matrix; its other fields are kept.
void setTraceTransform(pt::TraceInstance &instance, const Mat4 &model);

} // namespace basalt
