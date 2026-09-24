#include "render/Environment.h"

#include "pt/Tables.h"

#include "core/Log.h"

#include <stb_image.h>

#include <algorithm>
#include <cmath>
#include <filesystem>

namespace basalt {
namespace {

constexpr std::uint32_t kCubeSize = 512;
constexpr std::uint32_t kPrefilteredSize = 128;
constexpr std::uint32_t kPrefilteredMips = 6;
constexpr std::uint32_t kPrefilterSamples = 128;

// Parameters for one bake dispatch, matching BakeUniforms in shaders/ibl.metal.
struct BakeUniforms {
  Vec4 parameters;
};

} // namespace

Environment::Environment(const Context &ctx, Uploader &up) : context(ctx), uploader(up) {
  equirectProgram = std::make_unique<Program>(ctx, "equirect_to_cube");
  prefilterProgram = std::make_unique<Program>(ctx, "prefilter_specular");
  equirectPipeline = Pipeline(ctx, *equirectProgram, "equirect_to_cube");
  prefilterPipeline = Pipeline(ctx, *prefilterProgram, "prefilter_specular");
  pool = std::make_unique<DescriptorPool>(ctx, 64);

  VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  // Longitude wraps; the cubes ignore the mode.
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
  check(vkCreateSampler(ctx.device, &samplerInfo, nullptr, &sampler), "vkCreateSampler (environment)");

  const VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  ImageDescription description;
  description.format = VK_FORMAT_R16G16B16A16_SFLOAT;
  description.usage = usage;
  description.arrayLayers = 6;
  description.cube = true;

  description.width = description.height = kCubeSize;
  description.mipLevels = mipLevelsFor(kCubeSize, kCubeSize);
  description.name = "environment.cube";
  environmentCube = Image(ctx, description);

  description.width = description.height = kPrefilteredSize;
  description.mipLevels = kPrefilteredMips;
  description.name = "environment.prefiltered";
  prefilteredCube = Image(ctx, description);

  setProceduralSky(skySun, skyTurbidity, skyIntensity);
}

Environment::~Environment() {
  for (VkImageView view : temporaryViews) vkDestroyImageView(context.device, view, nullptr);
  if (sampler) vkDestroySampler(context.device, sampler, nullptr);
}

std::vector<float> Environment::proceduralSky(std::uint32_t width, std::uint32_t height, bool sunDisc,
                                              Vec3 sunDirection, float turbidity,
                                              float intensity) const {
  std::vector<float> texels(static_cast<std::size_t>(width) * height * 4, 0.0f);
  const Vec3 sun = normalize(sunDirection);
  const Vec3 zenithColor{0.12f, 0.26f, 0.58f};
  const Vec3 horizonColor{0.62f, 0.70f, 0.82f};
  const Vec3 groundColor{0.15f, 0.13f, 0.11f};
  const Vec3 sunColor{1.0f, 0.92f, 0.78f};
  const float haze = std::clamp(turbidity, 1.0f, 10.0f);

  for (std::uint32_t y = 0; y < height; ++y) {
    const float theta = (static_cast<float>(y) + 0.5f) / static_cast<float>(height) * kPi;
    for (std::uint32_t x = 0; x < width; ++x) {
      const float phi = ((static_cast<float>(x) + 0.5f) / static_cast<float>(width) - 0.5f) * 2.0f * kPi;
      const Vec3 direction{std::sin(theta) * std::cos(phi), std::cos(theta),
                           std::sin(theta) * std::sin(phi)};

      Vec3 colour;
      if (direction.y >= 0.0f) {
        const float t = std::pow(1.0f - direction.y, 2.0f + haze * 0.25f);
        colour = zenithColor * (1.0f - t) + horizonColor * t;
        const float cosine = std::max(0.0f, dot(direction, sun));
        const float disc = sunDisc ? std::pow(cosine, 6000.0f) * 400.0f : 0.0f;
        const float aureole = std::pow(cosine, 8.0f / haze) * 0.5f;
        colour += sunColor * (disc + aureole);
      } else {
        const float t = std::pow(1.0f + direction.y, 3.0f);
        colour = groundColor * (0.6f + 0.4f * t) + horizonColor * (0.12f * t);
      }
      colour *= intensity;

      const std::size_t index = (static_cast<std::size_t>(y) * width + x) * 4;
      texels[index + 0] = colour.x;
      texels[index + 1] = colour.y;
      texels[index + 2] = colour.z;
      texels[index + 3] = 1.0f;
    }
  }
  return texels;
}

void Environment::uploadEquirectangular(const std::vector<float> &texels, std::uint32_t width,
                                        std::uint32_t height, const std::string &name) {
  context.waitIdle();
  ++generation;
  projectIrradiance(texels, width, height);
  equirectangular = uploader.createTexture(texels.data(), texels.size() * sizeof(float), width,
                                           height, VK_FORMAT_R32G32B32A32_SFLOAT, 1, name);
}

void Environment::projectIrradiance(const std::vector<float> &texels, std::uint32_t width,
                                    std::uint32_t height) {
  // SH projection weighted by solid angle, convolved with the clamped cosine (bands pi,
  // 2pi/3, pi/4, over pi for radiance); basis constants are squared in so the shader only
  // evaluates shapes. Every texel counts: a sun a few texels wide holds much of the irradiance.
  std::array<std::array<double, 3>, 9> sum{};
  for (std::uint32_t y = 0; y < height; ++y) {
    const float theta = (static_cast<float>(y) + 0.5f) / static_cast<float>(height) * kPi;
    const float sinTheta = std::sin(theta), cosTheta = std::cos(theta);
    const float solidAngle = (2.0f * kPi / static_cast<float>(width)) * (kPi / static_cast<float>(height)) *
                             sinTheta;
    for (std::uint32_t x = 0; x < width; ++x) {
      const float phi = ((static_cast<float>(x) + 0.5f) / static_cast<float>(width) - 0.5f) * 2.0f * kPi;
      const Vec3 d{sinTheta * std::cos(phi), cosTheta, sinTheta * std::sin(phi)};
      const std::size_t index = (static_cast<std::size_t>(y) * width + x) * 4;
      // Clamped as the cube is, so diffuse and specular see the same sun; one non-finite
      // texel would poison all nine coefficients.
      const Vec3 radiance{std::min(texels[index], 60000.0f), std::min(texels[index + 1], 60000.0f),
                          std::min(texels[index + 2], 60000.0f)};
      if (!std::isfinite(radiance.x) || !std::isfinite(radiance.y) || !std::isfinite(radiance.z))
        continue;
      const Vec3 weighted = radiance * solidAngle;
      const float shape[9] = {1.0f, d.y, d.z, d.x, d.x * d.y, d.y * d.z, 3.0f * d.z * d.z - 1.0f,
                              d.x * d.z, d.x * d.x - d.y * d.y};
      for (int i = 0; i < 9; ++i) {
        std::array<double, 3> &c = sum[static_cast<std::size_t>(i)];
        c[0] += static_cast<double>(weighted.x) * shape[i];
        c[1] += static_cast<double>(weighted.y) * shape[i];
        c[2] += static_cast<double>(weighted.z) * shape[i];
      }
    }
  }
  const float basis[9] = {0.282095f, 0.488603f, 0.488603f, 0.488603f, 1.092548f,
                          1.092548f, 0.315392f, 1.092548f, 0.546274f};
  const float band[9] = {1.0f, 2.0f / 3.0f, 2.0f / 3.0f, 2.0f / 3.0f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f};
  for (int i = 0; i < 9; ++i) {
    const std::array<double, 3> &c = sum[static_cast<std::size_t>(i)];
    const double k = static_cast<double>(basis[i]) * basis[i] * band[i];
    irradiance[static_cast<std::size_t>(i)] =
        Vec4(Vec3{static_cast<float>(c[0] * k), static_cast<float>(c[1] * k), static_cast<float>(c[2] * k)},
             0.0f);
  }
}

void Environment::setProceduralSky(Vec3 sunDirection, float turbidity, float intensity) {
  skySun = sunDirection;
  skyTurbidity = turbidity;
  skyIntensity = intensity;
  isProcedural = true;
  sourceName = "procedural sky";
  uploadEquirectangular(proceduralSky(1024, 512, true, sunDirection, turbidity, intensity), 1024, 512,
                        "environment.procedural");
  const std::vector<float> traced = proceduralSky(1024, 512, false, sunDirection, turbidity, intensity);
  traceEquirectangular = uploader.createTexture(traced.data(), traced.size() * sizeof(float), 1024, 512,
                                                VK_FORMAT_R32G32B32A32_SFLOAT, 1, "environment.procedural.traced");
  buildTraceDistribution(traced, 1024, 512);
  bake();
}

void Environment::load(const std::string &path, bool extractSun) {
  extracted = pt::EnvironmentSun{};
  if (path.empty()) {
    setProceduralSky(skySun, skyTurbidity, skyIntensity);
    return;
  }
  extraction = extractSun;
  int width = 0, height = 0, channels = 0;
  float *pixels = stbi_loadf(path.c_str(), &width, &height, &channels, 4);
  if (!pixels) {
    logWarning("cannot read the environment {}: {}", path,
               stbi_failure_reason() ? stbi_failure_reason() : "unsupported format");
    setProceduralSky(skySun, skyTurbidity, skyIntensity);
    return;
  }
  const std::size_t count = static_cast<std::size_t>(width) * height * 4;
  std::vector<float> texels(pixels, pixels + count);
  stbi_image_free(pixels);

  // The sun leaves the image before anything is baked from it, so neither the IBL nor the
  // path-traced sky counts it a second time. Without extraction the brightest region still
  // gives the direction the interface offers to point the sun at.
  std::vector<float> probe;
  if (!extractSun) probe = texels;
  const pt::EnvironmentSun found = pt::extractEnvironmentSun(extractSun ? texels : probe,
                                                             static_cast<std::uint32_t>(width),
                                                             static_cast<std::uint32_t>(height));
  if (extractSun) extracted = found;
  brightest = normalize(Vec3{found.direction.x, found.direction.y, found.direction.z});
  uploadEquirectangular(texels, static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height),
                        "environment.source");
  isProcedural = false;
  traceEquirectangular = Image();
  buildTraceDistribution(texels, static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height));
  sourceName = std::filesystem::path(path).filename().string();
  bake();
  logInfo("environment: {} ({} by {})", sourceName, width, height);
  if (extracted.found)
    logInfo("environment sun extracted: {} texels, {:.2f} deg radius, irradiance {:.4g}, {:.1f}% of the light",
            extracted.texels, extracted.angularRadius * 180.0f / kPi, pt::ptLuminance(extracted.irradiance),
            extracted.share * 100.0f);
  else if (extractSun)
    logInfo("environment sun: none stands out from the sky; the image lights the scene alone");
}

void Environment::buildTraceDistribution(const std::vector<float> &texels, std::uint32_t width,
                                         std::uint32_t height) {
  pt::float4 info;
  pt::buildEnvironmentDistribution(texels.data(), width, height, distribution, info);
  distributionInfo = {info.x, info.y, info.z, info.w};
}

void Environment::bake() {
  context.waitIdle();
  for (VkImageView view : temporaryViews) vkDestroyImageView(context.device, view, nullptr);
  temporaryViews.clear();
  temporaryBuffers.clear();
  pool->reset();

  // A cube a kernel writes is bound as a six-layer 2D array view.
  auto storageView = [&](const Image &image, std::uint32_t mip) {
    VkImageView view = image.createView(mip, 1, 0, 6, VK_IMAGE_VIEW_TYPE_2D_ARRAY);
    temporaryViews.push_back(view);
    return view;
  };
  auto parameterBuffer = [&](const BakeUniforms &uniforms) -> const Buffer & {
    Buffer buffer(context, sizeof(BakeUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                  VMA_MEMORY_USAGE_AUTO,
                  VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                      VMA_ALLOCATION_CREATE_MAPPED_BIT,
                  "bake.parameters");
    buffer.write(&uniforms, sizeof(uniforms));
    temporaryBuffers.push_back(std::move(buffer));
    return temporaryBuffers.back();
  };

  const VkDescriptorSet equirectSet = pool->allocate(equirectProgram->setLayouts[0]);
  const BakeUniforms equirectUniforms{{static_cast<float>(kCubeSize), 0, 0, 0}};
  DescriptorWriter(context, equirectProgram->compute(), equirectSet)
      .storageTexture("destination", storageView(environmentCube, 0))
      .texture("equirectangular", equirectangular)
      .buffer("bake", parameterBuffer(equirectUniforms))
      .sampler("linearSampler", sampler)
      .apply();

  std::vector<VkDescriptorSet> prefilterSets(kPrefilteredMips);
  for (std::uint32_t mip = 0; mip < kPrefilteredMips; ++mip) {
    const float roughness = static_cast<float>(mip) / static_cast<float>(kPrefilteredMips - 1);
    const BakeUniforms uniforms{{static_cast<float>(kPrefilteredSize >> mip), roughness,
                                 static_cast<float>(kPrefilterSamples),
                                 static_cast<float>(environmentCube.description.mipLevels)}};
    prefilterSets[mip] = pool->allocate(prefilterProgram->setLayouts[0]);
    DescriptorWriter(context, prefilterProgram->compute(), prefilterSets[mip])
        .storageTexture("destination", storageView(prefilteredCube, mip))
        .texture("source", environmentCube.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        .buffer("bake", parameterBuffer(uniforms))
        .sampler("linearSampler", sampler)
        .apply();
  }

  uploader.runImmediate([&](VkCommandBuffer command) {
    auto dispatchCube = [&](const Pipeline &pipeline, const Program &program, VkDescriptorSet set,
                            std::uint32_t size) {
      vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
      vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, program.layout, 0, 1, &set, 0,
                              nullptr);
      vkCmdDispatch(command, std::max(1u, size / 8u), std::max(1u, size / 8u), 6);
    };

    transitionImage(command, environmentCube, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_WRITE_BIT);
    dispatchCube(equirectPipeline, *equirectProgram, equirectSet, kCubeSize);

    // Mips for the prefilter to pick by sample solid angle.
    transitionImage(command, environmentCube, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, 0, 1);
    std::int32_t size = static_cast<std::int32_t>(kCubeSize);
    for (std::uint32_t mip = 1; mip < environmentCube.description.mipLevels; ++mip) {
      VkImageMemoryBarrier2 toDestination{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
      toDestination.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      toDestination.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
      toDestination.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
      toDestination.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
      toDestination.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
      toDestination.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      toDestination.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      toDestination.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      toDestination.image = environmentCube.handle;
      toDestination.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 6};
      VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      dependency.imageMemoryBarrierCount = 1;
      dependency.pImageMemoryBarriers = &toDestination;
      vkCmdPipelineBarrier2(command, &dependency);

      const std::int32_t next = std::max(1, size / 2);
      VkImageBlit blit{};
      blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip - 1, 0, 6};
      blit.srcOffsets[1] = {size, size, 1};
      blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 6};
      blit.dstOffsets[1] = {next, next, 1};
      vkCmdBlitImage(command, environmentCube.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     environmentCube.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                     VK_FILTER_LINEAR);

      VkImageMemoryBarrier2 toSource = toDestination;
      toSource.srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
      toSource.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
      toSource.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
      toSource.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
      toSource.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      toSource.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      dependency.pImageMemoryBarriers = &toSource;
      vkCmdPipelineBarrier2(command, &dependency);
      size = next;
    }
    environmentCube.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    transitionImage(command, environmentCube, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    transitionImage(command, prefilteredCube, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_WRITE_BIT);

    for (std::uint32_t mip = 0; mip < kPrefilteredMips; ++mip)
      dispatchCube(prefilterPipeline, *prefilterProgram, prefilterSets[mip], kPrefilteredSize >> mip);

    transitionImage(command, prefilteredCube, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  });
}

} // namespace basalt
