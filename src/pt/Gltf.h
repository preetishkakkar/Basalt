// GPU-independent glTF loading into the canonical CPU path-tracing scene.
#pragma once
#include "pt/CpuTracer.h"

#include <cstdint>
#include <string>

namespace pt {

struct LoadedGltf {
  CpuScene scene;
  CpuFrame frame;
  float3 boundsMin{0.0f};
  float3 boundsMax{0.0f};
  std::uint64_t contentHash = 0;
};

// Loads triangle geometry, base metallic-roughness materials, decoded level-zero
// textures, punctual lights and instances without creating a Vulkan object.
LoadedGltf loadGltf(const std::string &path, unsigned buildThreads = 0);

// Replaces the default neutral environment with a linear floating-point HDR map.
void loadEnvironment(CpuScene &scene, const std::string &path);

} // namespace pt
