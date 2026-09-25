// The environment: sky cube and prefiltered chain baked by ibl.slang, irradiance as SH coefficients projected on the host.
#pragma once
#include "core/Math.h"
#include "gpu/Descriptors.h"
#include "gpu/Pipeline.h"
#include "gpu/Uploader.h"
#include "pt/EnvironmentSun.h"

#include <array>
#include <memory>
#include <string>

namespace basalt {

class Environment {
public:
  Environment(const Context &context, Uploader &uploader);
  ~Environment();

  // Loads an .hdr, or the procedural sky when empty or unreadable; rebakes either way. With
  // extractSun an .hdr's sun leaves every image baked from it (sky, IBL, path-traced
  // environment) for sun(), which the renderers then light with analytically.
  void load(const std::string &path, bool extractSun = true);
  void setProceduralSky(Vec3 sunDirection, float turbidity, float intensity);

  const Image &cube() const { return environmentCube; }
  // Premultiplied so shIrradiance() returns radiance.
  const std::array<Vec4, 9> &irradianceCoefficients() const { return irradiance; }
  const Image &prefiltered() const { return prefilteredCube; }
  float prefilteredMipCount() const { return static_cast<float>(prefilteredCube.description.mipLevels); }
  const std::string &name() const { return sourceName; }
  bool procedural() const { return isProcedural; }
  // What the path tracer reads: the equirect it shades escaped rays with, and that image's
  // luminance distribution (pt_lights.slang) with its (columns, rows, integral, present).
  // The procedural sky's has no sun disc painted in: the tracer's analytic sun is the disc.
  Image &traceImage() { return isProcedural ? traceEquirectangular : equirectangular; }
  const std::vector<float> &traceDistribution() const { return distribution; }
  const std::array<float, 4> &traceDistributionInfo() const { return distributionInfo; }
  // Changes whenever the environment's images do.
  std::uint32_t version() const { return generation; }
  // An .hdr's extracted sun; found is false for the procedural sky and for an image without one.
  const pt::EnvironmentSun &sun() const { return extracted; }
  bool sunExtractionEnabled() const { return extraction; }
  Vec3 brightestDirection() const { return brightest; }
  VkSampler linearSampler() const { return sampler; }

private:
  void bake();
  void projectIrradiance(const std::vector<float> &texels, std::uint32_t width, std::uint32_t height);
  void uploadEquirectangular(const std::vector<float> &texels, std::uint32_t width,
                             std::uint32_t height, const std::string &name);
  void buildTraceDistribution(const std::vector<float> &texels, std::uint32_t width, std::uint32_t height);
  std::vector<float> proceduralSky(std::uint32_t width, std::uint32_t height, bool sunDisc, Vec3 sunDirection,
                                   float turbidity, float intensity) const;

  const Context &context;
  Uploader &uploader;

  Image equirectangular;
  Image traceEquirectangular;  // the procedural sky without its disc
  std::vector<float> distribution;
  std::array<float, 4> distributionInfo{};
  std::uint32_t generation = 0;
  Image environmentCube;
  Image prefilteredCube;
  std::array<Vec4, 9> irradiance{};
  VkSampler sampler = VK_NULL_HANDLE;

  std::unique_ptr<Program> equirectProgram, prefilterProgram;
  Pipeline equirectPipeline, prefilterPipeline;
  std::unique_ptr<DescriptorPool> pool;
  std::vector<VkImageView> temporaryViews;

  std::string sourceName = "procedural sky";
  bool isProcedural = true;
  Vec3 skySun{0.3f, 0.6f, 0.4f};
  Vec3 brightest{0.0f, 1.0f, 0.0f};
  pt::EnvironmentSun extracted;
  bool extraction = true;
  float skyTurbidity = 3.0f;
  float skyIntensity = 1.0f;
};

} // namespace basalt
