// The environment: sky cube and prefiltered chain baked by ibl.metal, irradiance as SH coefficients projected on the host.
#pragma once
#include "core/Math.h"
#include "gpu/Descriptors.h"
#include "gpu/Pipeline.h"
#include "gpu/Uploader.h"

#include <array>
#include <memory>
#include <string>

namespace basalt {

class Environment {
public:
  Environment(const Context &context, Uploader &uploader);
  ~Environment();

  // Loads an .hdr, or the procedural sky when empty or unreadable; rebakes either way.
  void load(const std::string &path);
  void setProceduralSky(Vec3 sunDirection, float turbidity, float intensity);

  const Image &cube() const { return environmentCube; }
  // Premultiplied so shIrradiance() returns radiance.
  const std::array<Vec4, 9> &irradianceCoefficients() const { return irradiance; }
  const Image &prefiltered() const { return prefilteredCube; }
  float prefilteredMipCount() const { return static_cast<float>(prefilteredCube.description.mipLevels); }
  const std::string &name() const { return sourceName; }
  bool procedural() const { return isProcedural; }
  Vec3 brightestDirection() const { return brightest; }
  VkSampler linearSampler() const { return sampler; }

private:
  void bake();
  void projectIrradiance(const std::vector<float> &texels, std::uint32_t width, std::uint32_t height);
  void uploadEquirectangular(const std::vector<float> &texels, std::uint32_t width,
                             std::uint32_t height, const std::string &name);
  std::vector<float> proceduralSky(std::uint32_t width, std::uint32_t height, Vec3 sunDirection,
                                   float turbidity, float intensity) const;

  const Context &context;
  Uploader &uploader;

  Image equirectangular;
  Image environmentCube;
  Image prefilteredCube;
  std::array<Vec4, 9> irradiance{};
  VkSampler sampler = VK_NULL_HANDLE;

  std::unique_ptr<Program> equirectProgram, prefilterProgram;
  Pipeline equirectPipeline, prefilterPipeline;
  std::unique_ptr<DescriptorPool> pool;
  std::vector<VkImageView> temporaryViews;
  std::vector<Buffer> temporaryBuffers;

  std::string sourceName = "procedural sky";
  bool isProcedural = true;
  Vec3 skySun{0.3f, 0.6f, 0.4f};
  Vec3 brightest{0.0f, 1.0f, 0.0f};
  float skyTurbidity = 3.0f;
  float skyIntensity = 1.0f;
};

} // namespace basalt
