// Intel Embree as the CPU path tracer's alternative intersector, built the way the
// hardware structure is: one Embree scene per instance, instanced into a top scene with its
// transform and mask. Masked and blended candidates go through ptCandidateSolid, the test
// every backend uses, from Embree's filter callbacks.
#pragma once
#include "pt/Tracing.h"

#include <memory>

namespace pt {

struct CpuScene;
struct CpuFrame;

class EmbreeScene {
public:
  // Whether Embree was fetched and compiled in.
  static bool available();
  explicit EmbreeScene(const CpuScene &scene);
  ~EmbreeScene();
  EmbreeScene(const EmbreeScene &) = delete;
  EmbreeScene &operator=(const EmbreeScene &) = delete;

  // As ptTraceBvh: the closest hit, or with anyHit whether anything solid is in the way. The
  // alpha test reads view's instances, materials, geometry and textures.
  PtHit trace(const TraceView &view, float3 origin, float3 direction, float tMax, uint mask, uint seed, float2 cone,
              uint anyHit) const;
  double buildMilliseconds = 0.0;

private:
  struct State;
  std::unique_ptr<State> state;
};

} // namespace pt
