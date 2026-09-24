#include "render/Renderer.h"
#include "pt/WavefrontPlan.h"
#include "pt/ImageFile.h"

#include "core/Log.h"
#include "pt/Tables.h"
#include "render/GpuBvhBuilder.h"

#include <algorithm>
#include <chrono>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <thread>

namespace basalt {
namespace {

constexpr std::uint32_t kShadowResolution = 2048;
constexpr std::uint32_t kCascadeCount = 4;
constexpr std::uint32_t kHitTextureSlots = 120; // Matches kHitTextureSlots in shaders/common.metal.
constexpr VkFormat kHdrFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
// Sixty-four pixel tiles by twenty-four depth slices. A cell holds at most this many
// lights; past that they are dropped by index, not by nearness.
constexpr std::uint32_t kClusterTileSize = 64;
constexpr std::uint32_t kClusterSlices = 24;
constexpr std::uint32_t kClusterCapacity = 64;

// glTF 2.0: a node transform with a negative determinant keeps the object-space front face,
// which is then clockwise in world space.
VkFrontFace frontFaceOf(const Mat4 &transform) {
  const Vec4 &x = transform[0], &y = transform[1], &z = transform[2];
  const float determinant = x.x * (y.y * z.z - y.z * z.y) - x.y * (y.x * z.z - y.z * z.x) +
                            x.z * (y.x * z.y - y.y * z.x);
  return determinant < 0.0f ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
}

// Mirrors FrameUniforms in shaders/common.metal.
struct FrameUniforms {
  Mat4 viewProjection;
  Mat4 view;
  Mat4 inverseViewProjection;
  Vec4 cameraPosition;
  Vec4 sunDirection;
  Vec4 sunColor;
  Vec4 environment;
  Vec4 shadowParameters;
  Vec4 cascadeSplits;
  Vec4 viewportAndLights;
  Vec4 rays;
  Vec4 occlusion;
  Vec4 options;
  Vec4 clusters;
  Vec4 sh[9];
};
static_assert(sizeof(FrameUniforms) == 512, "the frame uniforms must match the shader");

// Mirrors ClusterUniforms in shaders/cluster.metal.
struct ClusterUniforms {
  Mat4 view;
  Vec4 grid;
  Vec4 projection;
  Vec4 counts;
};
static_assert(sizeof(ClusterUniforms) == 112, "the cluster uniforms must match the shader");

// Mirrors PostUniforms in shaders/post.metal.
struct PostUniforms {
  Vec4 parameters;
  Vec4 target;
  Vec4 effects;
  Vec4 antialias;
};

// Mirrors BloomUniforms in shaders/bloom.metal.
struct BloomUniforms {
  Vec4 parameters;
  Vec4 source;
};

// Mirrors ReflectionUniforms in shaders/reflect_body.metal.
struct ReflectionUniforms {
  Mat4 viewProjection;
  Mat4 inverseViewProjection;
  Vec4 camera;
  Vec4 march;
  Vec4 sun;
  Vec4 sunColor;
  Vec4 target;
  Vec4 environment;
  Vec4 debug;
  Vec4 sampling;
  Vec4 sh[9];
};
static_assert(sizeof(ReflectionUniforms) == 400, "the reflection uniforms must match the shader");

// Mirrors TemporalUniforms in shaders/taa.metal.
struct TemporalUniforms {
  Mat4 inverseViewProjection;
  Mat4 viewProjection;
  Mat4 previousViewProjection;
  Vec4 target;
  Vec4 parameters;
  Vec4 camera;
};
static_assert(sizeof(TemporalUniforms) == 240, "the temporal uniforms must match the shader");

// Halton (2,3) jitter, centred; eight offsets are enough to resolve an edge.
Vec2 haltonJitter(std::uint32_t index) {
  auto halton = [](std::uint32_t i, std::uint32_t base) {
    float result = 0.0f, fraction = 1.0f / static_cast<float>(base);
    while (i > 0) {
      result += static_cast<float>(i % base) * fraction;
      i /= base;
      fraction /= static_cast<float>(base);
    }
    return result;
  };
  const std::uint32_t n = index % 8 + 1;
  return {halton(n, 2) - 0.5f, halton(n, 3) - 0.5f};
}

// FNV-1a over bytes, for the key that decides whether frames may accumulate.
std::uint64_t hashBytes(const void *data, std::size_t size, std::uint64_t seed) {
  const auto *bytes = static_cast<const std::uint8_t *>(data);
  std::uint64_t hash = seed ^ 14695981039346656037ull;
  for (std::size_t i = 0; i < size; ++i) hash = (hash ^ bytes[i]) * 1099511628211ull;
  return hash;
}

VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) {
  if (alignment == 0) return value;
  return (value + alignment - 1) / alignment * alignment;
}

float halfToFloat(std::uint16_t half) {
  const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000u) << 16;
  const std::uint32_t exponent = (half >> 10) & 0x1Fu;
  std::uint32_t mantissa = half & 0x3FFu;
  std::uint32_t bits = 0;
  if (exponent == 0x1Fu) {
    bits = sign | 0x7F800000u | (mantissa << 13);
  } else if (exponent != 0) {
    bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  } else if (mantissa != 0) {
    // Subnormal: shift the mantissa up until the implicit bit appears.
    std::uint32_t shifted = 113u;
    while ((mantissa & 0x400u) == 0) {
      mantissa <<= 1;
      --shifted;
    }
    bits = sign | (shifted << 23) | ((mantissa & 0x3FFu) << 13);
  } else {
    bits = sign;
  }
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// Shadow alpha testing also needs both UV sets and vertex colour.
std::vector<VertexAttributeLayout> vertexLayout(bool shadowPass) {
  if (shadowPass)
    return {{0, 0, offsetof(Vertex, position)}, {1, 0, offsetof(Vertex, uv0)},
            {2, 0, offsetof(Vertex, uv1)}, {3, 0, offsetof(Vertex, color)}};
  return {{0, 0, offsetof(Vertex, position)},
          {1, 0, offsetof(Vertex, normal)},
          {2, 0, offsetof(Vertex, tangent)},
          {3, 0, offsetof(Vertex, uv0)},
          {4, 0, offsetof(Vertex, uv1)},
          {5, 0, offsetof(Vertex, color)}};
}

// Enters GENERAL with mip zero written, leaves SHADER_READ_ONLY.
void generateMips(VkCommandBuffer command, Image &image) {
  const std::uint32_t levels = image.description.mipLevels;
  auto barrier = [&](std::uint32_t mip, VkImageLayout from, VkImageLayout to,
                     VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                     VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = srcStage;
    b.srcAccessMask = srcAccess;
    b.dstStageMask = dstStage;
    b.dstAccessMask = dstAccess;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image.handle;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1};
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(command, &dependency);
  };

  barrier(0, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
  std::int32_t w = static_cast<std::int32_t>(image.description.width);
  std::int32_t h = static_cast<std::int32_t>(image.description.height);
  for (std::uint32_t mip = 1; mip < levels; ++mip) {
    // Undefined discards the level; the composite read it last frame.
    barrier(mip, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    const std::int32_t nw = std::max(1, w / 2), nh = std::max(1, h / 2);
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip - 1, 0, 1};
    blit.srcOffsets[1] = {w, h, 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1};
    blit.dstOffsets[1] = {nw, nh, 1};
    vkCmdBlitImage(command, image.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image.handle,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
    barrier(mip, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT);
    w = nw;
    h = nh;
  }
  image.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  transitionImage(command, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

} // namespace

Renderer::Renderer(Context &ctx, Swapchain &chain, Uploader &up)
    : context(ctx), swapchain(chain), uploader(up) {
  depthFormat = ctx.selectFormat({VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT},
                                 VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
  cascadeStride = alignUp(4 * sizeof(Vec4), ctx.properties.limits.minStorageBufferOffsetAlignment);

  pool = std::make_unique<DescriptorPool>(ctx, 1024);
  // The traced pair binds the acceleration structure; a device without ray queries never loads it.
  const bool traced = ctx.rayTracingSupported;
  forwardProgram = std::make_unique<Program>(ctx, "forward_vertex",
                                             traced ? "forward_fragment_rt" : "forward_fragment");
  shadowProgram = std::make_unique<Program>(ctx, "shadow_vertex", "shadow_fragment");
  skyProgram = std::make_unique<Program>(ctx, "sky_vertex", "sky_fragment");
  postProgram = std::make_unique<Program>(ctx, "post_vertex", "post_fragment");
  bloomDownProgram = std::make_unique<Program>(ctx, "bloom_downsample");
  bloomUpProgram = std::make_unique<Program>(ctx, "bloom_upsample");
  resolveProgram = std::make_unique<Program>(ctx, traced ? "resolve_reflections_rt" : "resolve_reflections");
  compositeProgram = std::make_unique<Program>(ctx, "composite_reflections");
  temporalProgram = std::make_unique<Program>(ctx, "resolve_temporal");
  clusterProgram = std::make_unique<Program>(ctx, "cull_lights");
  pathTraceProgram = std::make_unique<Program>(ctx, "path_trace");
  pathTraceEmissiveProgram = std::make_unique<Program>(ctx, "path_trace_emissive");
  pathTraceWideProgram = std::make_unique<Program>(ctx, "path_trace_wide");
  pathTraceWideEmissiveProgram = std::make_unique<Program>(ctx, "path_trace_wide_emissive");
  pathCompareProgram = std::make_unique<Program>(ctx, "path_compare");
  pathGuideExportProgram = std::make_unique<Program>(ctx, "path_export_guides");
  pathWaveShadeProgram = std::make_unique<Program>(ctx, "path_wavefront_shade");
  pathWaveShadeAtomicProgram = std::make_unique<Program>(ctx, "path_wavefront_shade_atomic");
  if (ctx.rayQuerySupported) pathWaveFusedRtProgram = std::make_unique<Program>(ctx, "path_wavefront_fused_rt");
  pathWaveFusedProgram = std::make_unique<Program>(ctx, "path_wavefront_fused");
  pathWaveFusedWideProgram = std::make_unique<Program>(ctx, "path_wavefront_fused_wide");
  // The subgroup variant needs every property its reflection names, in compute.
  waveSubgroupAllocationSupported = true;
  for (const std::string &property : pathWaveShadeProgram->compute().reflection.requiredProperties)
    if (!m2v::host::supportsProperty(ctx.physical, property)) waveSubgroupAllocationSupported = false;
  pathWaveResolveProgram = std::make_unique<Program>(ctx, "path_wavefront_resolve");
  pathWaveIntersectProgram = std::make_unique<Program>(ctx, "path_wavefront_intersect");
  pathWaveIntersectWideProgram = std::make_unique<Program>(ctx, "path_wavefront_intersect_wide");
  pathWaveShadowProgram = std::make_unique<Program>(ctx, "path_wavefront_shadow");
  pathWaveShadowWideProgram = std::make_unique<Program>(ctx, "path_wavefront_shadow_wide");
  pathRestirInitialProgram = std::make_unique<Program>(ctx, "path_restir_initial");
  pathRestirSpatialProgram = std::make_unique<Program>(ctx, "path_restir_spatial");
  if (traced) {
    pathTraceRtProgram = std::make_unique<Program>(ctx, "path_trace_rt");
    pathTraceHybridProgram = std::make_unique<Program>(ctx, "path_trace_hybrid_rt");
    pathWaveIntersectRtProgram = std::make_unique<Program>(ctx, "path_wavefront_intersect_rt");
    pathWaveShadowRtProgram = std::make_unique<Program>(ctx, "path_wavefront_shadow_rt");
  }

  createStaticResources();
  createPipelines();
  environmentState = std::make_unique<Environment>(ctx, up);

  if (ctx.properties.limits.timestampComputeAndGraphics) {
    VkQueryPoolCreateInfo queryInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryInfo.queryCount = kFramesInFlight * 2;
    check(vkCreateQueryPool(ctx.device, &queryInfo, nullptr, &timestampPool), "vkCreateQueryPool");
    vkResetQueryPool(ctx.device, timestampPool, 0, kFramesInFlight * 2);
  }

  const VkExtent2D extent = chain.extent();
  createTargets(extent.width, extent.height);
  setScene(createDefaultScene(ctx, up));
  // The environment baked its own default sun; the sky and the shadows must agree from frame one.
  rebakeProceduralSky();
}

Renderer::~Renderer() {
  context.waitIdle();
  acceleration.reset();
  if (timestampPool) vkDestroyQueryPool(context.device, timestampPool, nullptr);
  for (VkImageView view : bloomMipViews) vkDestroyImageView(context.device, view, nullptr);
  if (reflectionMipZero) vkDestroyImageView(context.device, reflectionMipZero, nullptr);
  for (VkImageView view : shadowLayerViews)
    if (view) vkDestroyImageView(context.device, view, nullptr);
  if (defaultSampler) vkDestroySampler(context.device, defaultSampler, nullptr);
  if (clampSampler) vkDestroySampler(context.device, clampSampler, nullptr);
  if (pointSampler) vkDestroySampler(context.device, pointSampler, nullptr);
  if (shadowSampler) vkDestroySampler(context.device, shadowSampler, nullptr);
  if (activeScene) activeScene->destroy(context);
}

VkFormat Renderer::swapchainFormat() const { return swapchain.format(); }

float Renderer::sceneScale() const {
  return activeScene && activeScene->bounds.valid() ? std::max(activeScene->bounds.radius(), 1e-3f) : 1.0f;
}

Vec3 Renderer::sunDirection() const {
  return normalize({std::cos(settings.sunElevation) * std::sin(settings.sunAzimuth),
                    std::sin(settings.sunElevation),
                    std::cos(settings.sunElevation) * std::cos(settings.sunAzimuth)});
}

void Renderer::rebakeProceduralSky() {
  if (!environmentState->procedural()) return;
  environmentState->setProceduralSky(sunDirection(), settings.skyTurbidity, settings.skyIntensity);
  rebuildSceneResources();
}

void Renderer::createStaticResources() {
  // Neutral textures for materials without a map, so shaders never branch.
  const std::uint32_t white = 0xFFFFFFFFu;
  const std::uint32_t flat = 0xFFFF8080u; // (0.5, 0.5, 1, 1): a normal pointing straight out.
  const std::uint32_t black = 0xFF000000u;
  whiteTexture = uploader.createTexture(&white, 4, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, 1, "white");
  normalTexture = uploader.createTexture(&flat, 4, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, 1, "flat.normal");
  blackTexture = uploader.createTexture(&black, 4, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, 1, "black");

  VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
  check(vkCreateSampler(context.device, &samplerInfo, nullptr, &defaultSampler), "vkCreateSampler");

  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  check(vkCreateSampler(context.device, &samplerInfo, nullptr, &clampSampler), "vkCreateSampler");

  // Nearest, for G-buffer reads.
  samplerInfo.magFilter = VK_FILTER_NEAREST;
  samplerInfo.minFilter = VK_FILTER_NEAREST;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  check(vkCreateSampler(context.device, &samplerInfo, nullptr, &pointSampler), "vkCreateSampler");

  // Comparison sampler: the hardware depth test plus linear filtering turns the 3x3 taps into a soft edge.
  VkSamplerCreateInfo shadowInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  shadowInfo.magFilter = VK_FILTER_LINEAR;
  shadowInfo.minFilter = VK_FILTER_LINEAR;
  shadowInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  shadowInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  shadowInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  shadowInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  shadowInfo.compareEnable = VK_TRUE;
  shadowInfo.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
  shadowInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
  check(vkCreateSampler(context.device, &shadowInfo, nullptr, &shadowSampler), "vkCreateSampler");

  for (FrameResources &frame : frames) {
    const VmaAllocationCreateFlags hostFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                               VMA_ALLOCATION_CREATE_MAPPED_BIT;
    frame.frameUniforms = Buffer(context, sizeof(FrameUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                 VMA_MEMORY_USAGE_AUTO, hostFlags, "frame.uniforms");
    frame.postUniforms = Buffer(context, sizeof(PostUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                VMA_MEMORY_USAGE_AUTO, hostFlags, "post.uniforms");
    frame.reflectionUniforms = Buffer(context, sizeof(ReflectionUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                      VMA_MEMORY_USAGE_AUTO, hostFlags, "reflection.uniforms");
    frame.temporalUniforms = Buffer(context, sizeof(TemporalUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                    VMA_MEMORY_USAGE_AUTO, hostFlags, "temporal.uniforms");
    frame.clusterUniforms = Buffer(context, sizeof(ClusterUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                   VMA_MEMORY_USAGE_AUTO, hostFlags, "cluster.uniforms");
    frame.pathUniforms = Buffer(context, sizeof(pt::PathUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                VMA_MEMORY_USAGE_AUTO, hostFlags, "path.uniforms");
    frame.hybridInverseViewProjection = Buffer(context, sizeof(Mat4), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                VMA_MEMORY_USAGE_AUTO, hostFlags, "hybrid.inverse-view-projection");
    frame.pathComparisonUniforms = Buffer(context, sizeof(std::uint32_t) * 4,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, hostFlags, "path.comparison.uniforms");
    frame.cascadePacked = Buffer(context, kCascadeCount * 4 * sizeof(Vec4),
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                                 hostFlags, "cascades.packed");
    frame.cascadeStrided = Buffer(context, kCascadeCount * cascadeStride,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                                  hostFlags, "cascades.strided");
  }
}

void Renderer::createPipelines() {
  const std::vector<VertexBinding> geometryBinding{{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX}};
  const std::vector<VkFormat> forwardTargets{kHdrFormat, kHdrFormat, kHdrFormat,
      kHdrFormat, kHdrFormat, kHdrFormat, VK_FORMAT_R32G32B32A32_UINT};

  GraphicsPipelineDescription forward;
  forward.program = forwardProgram.get();
  forward.colorFormats = forwardTargets;
  forward.depthFormat = depthFormat;
  forward.bindings = geometryBinding;
  forward.attributes = vertexLayout(false);
  forward.cullMode = VK_CULL_MODE_BACK_BIT;
  // Counter-clockwise front faces, as glTF specifies; checked against the debug normal view.
  // Set per draw: a mirrored node's front is clockwise in world space (frontFaceOf).
  forward.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  forward.dynamicFrontFace = true;
  forward.depthCompare = VK_COMPARE_OP_GREATER_OR_EQUAL;
  forward.name = "forward.opaque";
  forwardPipeline = Pipeline(context, forward);
  forward.cullMode = VK_CULL_MODE_NONE;
  forward.name = "forward.opaque.two-sided";
  forwardTwoSidedPipeline = Pipeline(context, forward);
  forward.cullMode = VK_CULL_MODE_BACK_BIT;

  // Blended: depth test without write; only the colour attachment blends, the G-buffer replaces.
  forward.blend = true;
  forward.depthWrite = false;
  forward.cullMode = VK_CULL_MODE_NONE;
  forward.name = "forward.blended";
  forwardBlendPipeline = Pipeline(context, forward);

  VkPhysicalDeviceFeatures features{};
  vkGetPhysicalDeviceFeatures(context.physical, &features);
  if (features.fillModeNonSolid) {
    forward.blend = false;
    forward.depthWrite = true;
    forward.polygonMode = VK_POLYGON_MODE_LINE;
    forward.name = "forward.wireframe";
    forwardWirePipeline = Pipeline(context, forward);
  }

  GraphicsPipelineDescription shadow;
  shadow.program = shadowProgram.get();
  // The masked entry returns a colour because the bundled compiler has no fragment-void
  // profile. A throwaway target keeps the depth-only pass validation-clean.
  shadow.colorFormats = {kHdrFormat};
  shadow.depthFormat = depthFormat;
  shadow.bindings = geometryBinding;
  shadow.attributes = vertexLayout(true);
  // Front faces culled so the map stores the caster's far side, moving self-shadowing error to
  // the unlit side. Ordinary viewport here, so front faces stay counter-clockwise (clockwise
  // for mirrored nodes, set per draw).
  shadow.cullMode = VK_CULL_MODE_FRONT_BIT;
  shadow.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  shadow.dynamicFrontFace = true;
  shadow.depthCompare = VK_COMPARE_OP_LESS_OR_EQUAL;
  shadow.depthBias = true;
  // Opaque casters need no fragment stage at all; only alpha-masked ones do.
  shadow.depthOnly = true;
  shadow.name = "shadow.opaque";
  shadowPipeline = Pipeline(context, shadow);
  shadow.depthOnly = false;
  shadow.name = "shadow.masked";
  shadowMaskedPipeline = Pipeline(context, shadow);
  // A double-sided caster has no side to cull.
  shadow.cullMode = VK_CULL_MODE_NONE;
  shadow.depthOnly = true;
  shadow.name = "shadow.opaque.two-sided";
  shadowTwoSidedPipeline = Pipeline(context, shadow);
  shadow.depthOnly = false;
  shadow.name = "shadow.masked.two-sided";
  shadowMaskedTwoSidedPipeline = Pipeline(context, shadow);

  GraphicsPipelineDescription sky;
  sky.program = skyProgram.get();
  sky.colorFormats = forwardTargets;
  sky.depthFormat = depthFormat;
  sky.cullMode = VK_CULL_MODE_NONE;
  sky.depthWrite = false;
  sky.depthCompare = VK_COMPARE_OP_GREATER_OR_EQUAL;
  sky.name = "sky";
  skyPipeline = Pipeline(context, sky);

  GraphicsPipelineDescription post;
  post.program = postProgram.get();
  post.colorFormats = {swapchain.format()};
  post.depthFormat = VK_FORMAT_UNDEFINED;
  post.cullMode = VK_CULL_MODE_NONE;
  post.depthTest = false;
  post.depthWrite = false;
  post.name = "post";
  postPipeline = Pipeline(context, post);

  bloomDownPipeline = Pipeline(context, *bloomDownProgram, "bloom.downsample");
  bloomUpPipeline = Pipeline(context, *bloomUpProgram, "bloom.upsample");
  resolvePipeline = Pipeline(context, *resolveProgram, "reflections.resolve");
  compositePipeline = Pipeline(context, *compositeProgram, "reflections.composite");
  temporalPipeline = Pipeline(context, *temporalProgram, "temporal.resolve");
  clusterPipeline = Pipeline(context, *clusterProgram, "lights.cull");
  pathTracePipeline = Pipeline(context, *pathTraceProgram, "path.trace.software");
  pathTraceEmissivePipeline = Pipeline(context, *pathTraceEmissiveProgram, "path.trace.software.emissive");
  pathTraceWidePipeline = Pipeline(context, *pathTraceWideProgram, "path.trace.software.wide");
  pathTraceWideEmissivePipeline = Pipeline(context, *pathTraceWideEmissiveProgram,
                                            "path.trace.software.wide.emissive");
  if (pathTraceRtProgram) pathTraceRtPipeline = Pipeline(context, *pathTraceRtProgram, "path.trace.rt");
  if (pathTraceHybridProgram)
    pathTraceHybridPipeline = Pipeline(context, *pathTraceHybridProgram, "path.trace.hybrid.rt");
  pathComparePipeline = Pipeline(context, *pathCompareProgram, "path.compare");
  pathGuideExportPipeline = Pipeline(context, *pathGuideExportProgram, "path.guide.export");
  if (waveSubgroupAllocationSupported)
    pathWaveShadePipeline = Pipeline(context, *pathWaveShadeProgram, "path.wavefront.shade");
  pathWaveShadeAtomicPipeline = Pipeline(context, *pathWaveShadeAtomicProgram, "path.wavefront.shade.atomic");
  if (pathWaveFusedRtProgram)
    pathWaveFusedRtPipeline = Pipeline(context, *pathWaveFusedRtProgram, "path.wavefront.fused.rt");
  pathWaveFusedPipeline = Pipeline(context, *pathWaveFusedProgram, "path.wavefront.fused");
  pathWaveFusedWidePipeline = Pipeline(context, *pathWaveFusedWideProgram, "path.wavefront.fused.wide");
  pathWaveResolvePipeline = Pipeline(context, *pathWaveResolveProgram, "path.wavefront.resolve");
  pathWaveIntersectPipeline = Pipeline(context, *pathWaveIntersectProgram, "path.wavefront.intersect");
  pathWaveIntersectWidePipeline = Pipeline(context, *pathWaveIntersectWideProgram, "path.wavefront.intersect.wide");
  pathWaveShadowPipeline = Pipeline(context, *pathWaveShadowProgram, "path.wavefront.shadow");
  pathWaveShadowWidePipeline = Pipeline(context, *pathWaveShadowWideProgram, "path.wavefront.shadow.wide");
  pathRestirInitialPipeline = Pipeline(context, *pathRestirInitialProgram, "path.restir.initial");
  pathRestirSpatialPipeline = Pipeline(context, *pathRestirSpatialProgram, "path.restir.spatial");
  if (pathWaveIntersectRtProgram) pathWaveIntersectRtPipeline = Pipeline(context, *pathWaveIntersectRtProgram, "path.wavefront.intersect.rt");
  if (pathWaveShadowRtProgram) pathWaveShadowRtPipeline = Pipeline(context, *pathWaveShadowRtProgram, "path.wavefront.shadow.rt");
  if (context.rayPipelineSupported) try {
    const std::vector<RayStageDescription> stages{
        {"path_trace_pipeline", VK_SHADER_STAGE_RAYGEN_BIT_KHR},
        {"path_pipeline_miss", VK_SHADER_STAGE_MISS_BIT_KHR},
        {"path_pipeline_closest", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR},
        {"path_pipeline_alpha", VK_SHADER_STAGE_ANY_HIT_BIT_KHR}};
    const std::vector<m2v::host::RayShaderGroup> groups{
        {VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 0u},
        {VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 1u},
        {VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
         VK_SHADER_UNUSED_KHR, 2u, 3u, VK_SHADER_UNUSED_KHR}};
    pathRayPipeline = std::make_unique<RayPipeline>(
        context, stages, groups, m2v::host::ShaderBindingRecord{0u, {}},
        std::vector<m2v::host::ShaderBindingRecord>{{1u, {}}},
        std::vector<m2v::host::ShaderBindingRecord>{{2u, {}}});

    const std::vector<RayStageDescription> intersectStages{
        {"path_wavefront_intersect_pipeline", VK_SHADER_STAGE_RAYGEN_BIT_KHR},
        {"path_wave_pipeline_miss", VK_SHADER_STAGE_MISS_BIT_KHR},
        {"path_wave_pipeline_closest", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR},
        {"path_wave_pipeline_alpha", VK_SHADER_STAGE_ANY_HIT_BIT_KHR}};
    const std::vector<RayStageDescription> shadowStages{
        {"path_wavefront_shadow_pipeline", VK_SHADER_STAGE_RAYGEN_BIT_KHR},
        {"path_wave_shadow_pipeline_miss", VK_SHADER_STAGE_MISS_BIT_KHR},
        {"path_wave_shadow_pipeline_closest", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR},
        {"path_wave_shadow_pipeline_alpha", VK_SHADER_STAGE_ANY_HIT_BIT_KHR}};
    pathWaveIntersectRayPipeline = std::make_unique<RayPipeline>(
        context, intersectStages, groups, m2v::host::ShaderBindingRecord{0u, {}},
        std::vector<m2v::host::ShaderBindingRecord>{{1u, {}}},
        std::vector<m2v::host::ShaderBindingRecord>{{2u, {}}});
    pathWaveShadowRayPipeline = std::make_unique<RayPipeline>(
        context, shadowStages, groups, m2v::host::ShaderBindingRecord{0u, {}},
        std::vector<m2v::host::ShaderBindingRecord>{{1u, {}}},
        std::vector<m2v::host::ShaderBindingRecord>{{2u, {}}});
  } catch (const std::exception &failure) {
    // The other renderers stay usable; this one reports itself unavailable.
    logError("the full ray pipeline renderer is unavailable: {}", failure.what());
    pathRayPipeline.reset();
    pathWaveIntersectRayPipeline.reset();
    pathWaveShadowRayPipeline.reset();
  }
}

void Renderer::createTargets(std::uint32_t newWidth, std::uint32_t newHeight) {
  if (newWidth == 0 || newHeight == 0) return;
  context.waitIdle();
  width = newWidth;
  height = newHeight;

  for (VkImageView view : bloomMipViews) vkDestroyImageView(context.device, view, nullptr);
  bloomMipViews.clear();
  if (reflectionMipZero) vkDestroyImageView(context.device, reflectionMipZero, nullptr);
  reflectionMipZero = VK_NULL_HANDLE;
  for (VkImageView &view : shadowLayerViews) {
    if (view) vkDestroyImageView(context.device, view, nullptr);
    view = VK_NULL_HANDLE;
  }

  ImageDescription colorDescription;
  colorDescription.format = kHdrFormat;
  colorDescription.width = width;
  colorDescription.height = height;
  colorDescription.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  colorDescription.name = "hdr.color";
  hdrColor = Image(context, colorDescription);
  colorDescription.name = "gbuffer.normal";
  normalRoughness = Image(context, colorDescription);
  colorDescription.name = "gbuffer.reflection";
  reflectionWeight = Image(context, colorDescription);
  colorDescription.name = "gbuffer.base-metallic";
  gbufferBaseMetallic = Image(context, colorDescription);
  colorDescription.name = "gbuffer.geometric-coverage";
  gbufferGeometricCoverage = Image(context, colorDescription);
  colorDescription.name = "gbuffer.emissive";
  gbufferEmissive = Image(context, colorDescription);
  colorDescription.format = VK_FORMAT_R32G32B32A32_UINT;
  colorDescription.name = "gbuffer.identity";
  gbufferIdentity = Image(context, colorDescription);
  colorDescription.format = kHdrFormat;
  colorDescription.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  colorDescription.name = "path.comparison.raster";
  pathComparisonRaster = Image(context, colorDescription);

  ImageDescription reflectionDescription;
  reflectionDescription.format = kHdrFormat;
  reflectionDescription.width = width;
  reflectionDescription.height = height;
  reflectionMipCount = std::min(mipLevelsFor(width, height), 6u);
  reflectionDescription.mipLevels = reflectionMipCount;
  reflectionDescription.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  reflectionDescription.name = "reflection";
  reflection = Image(context, reflectionDescription);
  reflectionMipZero = reflection.createView(0, 1, 0, 1, VK_IMAGE_VIEW_TYPE_2D);

  ImageDescription litDescription;
  litDescription.format = kHdrFormat;
  litDescription.width = width;
  litDescription.height = height;
  litDescription.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  clusterGrid = {std::max(1u, (width + kClusterTileSize - 1) / kClusterTileSize),
                 std::max(1u, (height + kClusterTileSize - 1) / kClusterTileSize), kClusterSlices};
  clusterCount = clusterGrid[0] * clusterGrid[1] * clusterGrid[2];
  lightClusters = Buffer(context, sizeof(std::uint32_t) * clusterCount * (1 + kClusterCapacity),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, 0, "lights.clusters");

  litDescription.name = "lit.current";
  litCurrent = Image(context, litDescription);
  for (std::uint32_t i = 0; i < kFramesInFlight; ++i) {
    litDescription.name = "lit.resolved." + std::to_string(i);
    litColor[i] = Image(context, litDescription);
  }
  accumulatedFrames = 0;
  hasPreviousView = false;
  for (Buffer &staging : cpuStaging) staging = Buffer();
  cpuUploaded.fill(0);
  pathAccumulation = Image();
  pathAlbedoAccumulation = Image();
  pathNormalAccumulation = Image();
  pathReconstruction.reset();
  pathReconstructionSamples = Buffer();
  pathWaveStatesA=Buffer();pathWaveStatesB=Buffer();pathWaveResults=Buffer();pathWaveShadows=Buffer();pathWaveHits=Buffer();
  pathWaveCounters=Buffer();pathWaveControl=Buffer();pathWaveCapacity=0;
  pathSetsGeneration = ~0u;

  ImageDescription depthDescription;
  depthDescription.format = depthFormat;
  depthDescription.width = width;
  depthDescription.height = height;
  depthDescription.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  depthDescription.name = "depth";
  depthBuffer = Image(context, depthDescription);

  if (!shadowMap) {
    ImageDescription shadowDescription;
    shadowDescription.format = depthFormat;
    shadowDescription.width = kShadowResolution;
    shadowDescription.height = kShadowResolution;
    shadowDescription.arrayLayers = kCascadeCount;
    shadowDescription.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    shadowDescription.name = "shadow.cascades";
    shadowMap = Image(context, shadowDescription);
    shadowDescription.format = kHdrFormat;
    shadowDescription.arrayLayers = 1;
    shadowDescription.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    shadowDescription.name = "shadow.dummy-color";
    shadowDummyColor = Image(context, shadowDescription);
  }
  for (std::uint32_t layer = 0; layer < kCascadeCount; ++layer)
    shadowLayerViews[layer] = shadowMap.createView(0, 1, layer, 1, VK_IMAGE_VIEW_TYPE_2D);

  // Half resolution, halving until a level would be under one compute group.
  const std::uint32_t bloomWidth = std::max(8u, width / 2);
  const std::uint32_t bloomHeight = std::max(8u, height / 2);
  bloomMipCount = 1;
  for (std::uint32_t w = bloomWidth, h = bloomHeight; w > 16 && h > 16 && bloomMipCount < 7;) {
    w /= 2;
    h /= 2;
    ++bloomMipCount;
  }

  ImageDescription bloomDescription;
  bloomDescription.format = kHdrFormat;
  bloomDescription.width = bloomWidth;
  bloomDescription.height = bloomHeight;
  bloomDescription.mipLevels = bloomMipCount;
  bloomDescription.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
  bloomDescription.name = "bloom.chain";
  bloomChain = Image(context, bloomDescription);
  for (std::uint32_t mip = 0; mip < bloomMipCount; ++mip)
    bloomMipViews.push_back(bloomChain.createView(mip, 1, 0, 1, VK_IMAGE_VIEW_TYPE_2D));

  rebuildSceneResources();
}

void Renderer::setScene(std::unique_ptr<Scene> next) {
  if (!next) return;
  context.waitIdle();

  // Built for the incoming scene before the old one is released, so a scene that cannot be
  // traced is refused without losing what is on screen.
  std::vector<VkImageView> nextViews;
  std::vector<std::uint32_t> nextSlots;
  std::unique_ptr<SceneAccelerationStructure> nextAcceleration;
  std::vector<Image *> nextImages;
  buildHitTextureTable(*next, nextViews, nextSlots, nextImages);
  TraceScene nextTrace = buildTraceScene(*next, nextSlots);
  if (context.accelerationStructureSupported)
    nextAcceleration = std::make_unique<SceneAccelerationStructure>(context, uploader, *next, nextTrace);

  acceleration.reset();
  if (activeScene) activeScene->destroy(context);
  activeScene = std::move(next);
  // Nothing to reproject a history through after a reframe.
  hasPreviousView = false;
  hitTextureViews = std::move(nextViews);
  hitTextureImages = std::move(nextImages);
  ++sceneVersion;
  materialSlots = std::move(nextSlots);
  acceleration = std::move(nextAcceleration);
  traceScene = std::move(nextTrace);
  softwareBvhNodes.reset();
  softwareBvhTriangles.reset();
  softwareBvhStatistics = {};
  softwareBvhVersion = ~0u;
  softwareBvhBuilder = -1;
  softwareBvhScratchBytes = softwareBvhOutputBytes = 0;
  softwareBvhRadixPasses = softwareBvhMaximumStack = 0;
  pathEmissiveBuffer.reset();
  pathEmissiveMetadata.clear();
  pathEmissiveVersion = ~0u;
  {
    // A storage buffer may not be empty.
    std::vector<pt::TraceInstance> rows = traceScene.instances;
    if (rows.empty()) rows.push_back({});
    traceInstanceBuffer = uploader.createBuffer(rows.data(), rows.size() * sizeof(pt::TraceInstance),
                                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "scene.trace.instances");
  }

  std::vector<MaterialUniforms> materials;
  materials.reserve(activeScene->materials.size());
  for (const Material &material : activeScene->materials) materials.push_back(material.uniforms);
  if (materials.empty()) materials.push_back({});
  materialBuffer = uploader.createBuffer(materials.data(), materials.size() * sizeof(MaterialUniforms),
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "scene.materials");

  instanceRecords.clear();
  instanceRecords.reserve(activeScene->primitives.size());
  for (const Primitive &primitive : activeScene->primitives) {
    const Mat4 &model = primitive.transform;
    const Mat4 normal = normalMatrix(model);
    InstanceRecord record;
    record.modelRow0 = model.row(0);
    record.modelRow1 = model.row(1);
    record.modelRow2 = model.row(2);
    record.normalRow0 = normal.row(0);
    record.normalRow1 = normal.row(1);
    record.normalRow2 = normal.row(2);
    record.materialAndFlags = {static_cast<float>(primitive.material), 1.0f, 0.0f, 0.0f};
    instanceRecords.push_back(record);
  }
  if (instanceRecords.empty()) instanceRecords.push_back({});
  instanceBuffer = uploader.createBuffer(instanceRecords.data(),
                                         instanceRecords.size() * sizeof(InstanceRecord),
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "scene.instances");

  buildLights();

  rebuildSceneResources();
}

void Renderer::buildLights() {
  std::vector<LightRecord> lights = activeScene ? activeScene->lights : std::vector<LightRecord>{};

  // Scattered on a fixed hash, so a count always gives the same arrangement.
  const int requested = std::max(0, settings.testLights);
  if (requested > 0 && activeScene && activeScene->bounds.valid()) {
    const Vec3 centre = activeScene->bounds.center();
    const Vec3 extent = activeScene->bounds.extent() * 0.5f;
    const float radius = std::max(activeScene->bounds.radius(), 1e-3f);
    auto hashed = [](std::uint32_t value) {
      value ^= value >> 16;
      value *= 0x7feb352du;
      value ^= value >> 15;
      value *= 0x846ca68bu;
      value ^= value >> 16;
      return static_cast<float>(value) * 2.3283064365386963e-10f;
    };
    for (int i = 0; i < requested; ++i) {
      const std::uint32_t seed = static_cast<std::uint32_t>(i) * 2654435761u + 1u;
      LightRecord light;
      const Vec3 offset{hashed(seed) * 2.0f - 1.0f, hashed(seed + 7u) * 2.0f - 1.0f,
                        hashed(seed + 13u) * 2.0f - 1.0f};
      const Vec3 position{centre.x + offset.x * extent.x, centre.y + std::abs(offset.y) * extent.y,
                          centre.z + offset.z * extent.z};
      // Room-sized, so the grid has something to cull.
      const float range = radius * 0.15f;
      light.position = Vec4(position, range);
      const float hue = hashed(seed + 21u) * 6.0f;
      const Vec3 colour{std::clamp(std::abs(hue - 3.0f) - 1.0f, 0.0f, 1.0f),
                        std::clamp(2.0f - std::abs(hue - 2.0f), 0.0f, 1.0f),
                        std::clamp(2.0f - std::abs(hue - 4.0f), 0.0f, 1.0f)};
      // Dimmed as more are added, so a crowd does not blow out.
      light.color = Vec4(colour, range * range * 0.32f /
                                     std::sqrt(static_cast<float>(requested)));
      // Every third one a spot, so the cone path runs.
      const bool spot = (i % 3) == 2;
      const Vec3 aim = normalize(Vec3{offset.x * 0.4f, -1.0f, offset.z * 0.4f});
      light.direction = Vec4(aim, std::cos(radians(18.0f)));
      light.cone = {std::cos(radians(30.0f)), spot ? 1.0f : 0.0f, 0.0f, 0.0f};
      lights.push_back(light);
    }
  }

  lightCount = static_cast<std::uint32_t>(lights.size());
  hostLights = lights;
  if (lights.empty()) lights.push_back({}); // A storage buffer may not be empty.
  context.waitIdle();
  lightBuffer = uploader.createBuffer(lights.data(), lights.size() * sizeof(LightRecord),
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "scene.lights");
  builtTestLights = requested;
}

void Renderer::buildHitTextureTable(Scene &scene, std::vector<VkImageView> &views,
                                    std::vector<std::uint32_t> &slots, std::vector<Image *> &images) {
  // Slot zero is white for missing maps; a scene with more textures than slots shades the rest white and says so.
  views.assign(kHitTextureSlots, whiteTexture.view);
  images.assign(kHitTextureSlots, &whiteTexture);
  // Slot one is the flat normal; white would tilt every hit.
  views[1] = normalTexture.view;
  images[1] = &normalTexture;
  slots.assign(std::max<std::size_t>(scene.materials.size(), 1), 0);
  std::map<int, std::uint32_t> slotOf;
  std::uint32_t nextSlot = 2;
  bool overflowed = false;
  auto slotFor = [&](int texture) -> std::uint32_t {
    if (texture < 0 || static_cast<std::size_t>(texture) >= scene.textures.size()) return 0;
    const auto found = slotOf.find(texture);
    if (found != slotOf.end()) return found->second;
    if (nextSlot >= kHitTextureSlots) {
      overflowed = true;
      return 0;
    }
    views[nextSlot] = scene.textures[static_cast<std::size_t>(texture)].view;
    images[nextSlot] = &scene.textures[static_cast<std::size_t>(texture)];
    slotOf.emplace(texture, nextSlot);
    return nextSlot++;
  };
  for (std::size_t m = 0; m < scene.materials.size(); ++m) {
    const Material &material = scene.materials[m];
    std::uint32_t normalSlot = material.normal < 0 ? 1 : slotFor(material.normal);
    if (normalSlot == 0) normalSlot = 1;
    slots[m] = slotFor(material.baseColor) | (slotFor(material.metallicRoughness) << 8) |
               (slotFor(material.emissive) << 16) | (normalSlot << 24);
    // V7 extension textures carry their slots in the material (low byte); 0 is white and 1
    // the flat normal, the defaults when a texture is absent.
    std::array<std::uint32_t, 4> &extension = scene.materials[m].uniforms.extensionTextures;
    const int textures[4] = {material.transmissionTexture, material.clearcoatTexture,
                             material.clearcoatRoughnessTexture, material.clearcoatNormalTexture};
    for (std::size_t k = 0; k < 4; ++k) {
      std::uint32_t slot = slotFor(textures[k]);
      if (k == 3 && slot == 0) slot = 1;
      extension[k] = (extension[k] & ~0xFFu) | slot;
    }
  }
  if (overflowed)
    logWarning("more than {} textures reachable by rays: some reflections and traced cut-outs use white",
               kHitTextureSlots - 2);
}

void Renderer::applyGroundMaterial() {
  if (!activeScene || activeScene->groundPrimitive < 0) return;
  const std::uint32_t index =
      activeScene->primitives[static_cast<std::size_t>(activeScene->groundPrimitive)].material;
  MaterialUniforms &uniforms = activeScene->materials[index].uniforms;
  const float metallic = std::clamp(settings.groundMetallic, 0.0f, 1.0f);
  const float roughness = std::clamp(settings.groundRoughness, 0.02f, 1.0f);
  const Vec4 colour{std::clamp(settings.groundColor.x, 0.0f, 1.0f), std::clamp(settings.groundColor.y, 0.0f, 1.0f),
                    std::clamp(settings.groundColor.z, 0.0f, 1.0f), 1.0f};
  if (uniforms.factors.x == metallic && uniforms.factors.y == roughness &&
      uniforms.baseColorFactor.x == colour.x && uniforms.baseColorFactor.y == colour.y &&
      uniforms.baseColorFactor.z == colour.z)
    return;
  uniforms.factors.x = metallic;
  uniforms.factors.y = roughness;
  uniforms.baseColorFactor = colour;

  context.waitIdle();
  std::vector<MaterialUniforms> materials;
  materials.reserve(activeScene->materials.size());
  for (const Material &material : activeScene->materials) materials.push_back(material.uniforms);
  materialBuffer = uploader.createBuffer(materials.data(), materials.size() * sizeof(MaterialUniforms),
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "scene.materials");
  rebuildSceneResources();
}

void Renderer::rebuildSceneResources() {
  if (!activeScene || !hdrColor || !shadowMap) return;
  context.waitIdle();
  pool->reset();
  ++resourceGeneration;

  const bool traced = rayTracingAvailable();
  const std::size_t materialCount = std::max<std::size_t>(activeScene->materials.size(), 1);
  auto textureOr = [&](int index, const Image &fallback) -> const Image & {
    if (index >= 0 && static_cast<std::size_t>(index) < activeScene->textures.size())
      return activeScene->textures[static_cast<std::size_t>(index)];
    return fallback;
  };
  auto samplerOr = [&](int index) {
    if (index >= 0 && static_cast<std::size_t>(index) < activeScene->samplers.size())
      return activeScene->samplers[static_cast<std::size_t>(index)];
    return defaultSampler;
  };

  shadowMaterialSets.assign(materialCount, VK_NULL_HANDLE);
  for (std::size_t m = 0; m < materialCount; ++m) {
    const Material &material = activeScene->materials[std::min(m, activeScene->materials.size() - 1)];
    shadowMaterialSets[m] = pool->allocate(shadowProgram->setLayouts[1]);
    DescriptorWriter(context, shadowProgram->fragment(), shadowMaterialSets[m])
        .buffer("materials", materialBuffer)
        .texture("baseColorMap", textureOr(material.baseColor, whiteTexture))
        .sampler("materialSampler", samplerOr(material.sampler))
        .apply();
  }

  for (FrameResources &frame : frames) {
    frame.forwardVertexSet = pool->allocate(forwardProgram->setLayouts[0]);
    DescriptorWriter(context, forwardProgram->vertex(), frame.forwardVertexSet)
        .buffer("frame", frame.frameUniforms)
        .buffer("instances", instanceBuffer)
        .apply();

    for (std::uint32_t cascade = 0; cascade < kCascadeCount; ++cascade) {
      frame.shadowVertexSets[cascade] = pool->allocate(shadowProgram->setLayouts[0]);
      DescriptorWriter(context, shadowProgram->vertex(), frame.shadowVertexSets[cascade])
          .buffer("instances", instanceBuffer)
          .buffer("cascadeRows", frame.cascadeStrided, cascade * cascadeStride, 4 * sizeof(Vec4))
          .apply();
    }

    frame.forwardMaterialSets.assign(materialCount, VK_NULL_HANDLE);
    for (std::size_t m = 0; m < materialCount; ++m) {
      const Material &material = activeScene->materials[std::min(m, activeScene->materials.size() - 1)];
      frame.forwardMaterialSets[m] = pool->allocate(forwardProgram->setLayouts[1]);
      DescriptorWriter writer(context, forwardProgram->fragment(), frame.forwardMaterialSets[m]);
      writer.buffer("frame", frame.frameUniforms)
          .buffer("materials", materialBuffer)
          .buffer("cascadeRows", frame.cascadePacked)
          .buffer("lights", lightBuffer)
          .texture("baseColorMap", textureOr(material.baseColor, whiteTexture))
          .texture("metallicRoughnessMap", textureOr(material.metallicRoughness, whiteTexture))
          .texture("normalMap", textureOr(material.normal, normalTexture))
          .texture("occlusionMap", textureOr(material.occlusion, whiteTexture))
          .texture("emissiveMap", textureOr(material.emissive, whiteTexture))
          .texture("prefilteredCube", environmentState->prefiltered())
          .texture("shadowMap", shadowMap.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
          .sampler("materialSampler", samplerOr(material.sampler))
          .sampler("clampSampler", clampSampler)
          .sampler("shadowSampler", shadowSampler);
      if (context.rayTracingSupported) {
        if (!acceleration)
          throw Error("the ray tracing forward entry needs an acceleration structure for the scene");
        writer.accelerationStructure("scene", acceleration->topLevel)
            .buffer("traceInstances", traceInstanceBuffer)
            .buffer("indices", activeScene->indexBuffer)
            .buffer("vertices", activeScene->vertexBuffer)
            .textureArray("maps", hitTextureViews, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            .sampler("tableSampler", defaultSampler);
      }
      writer.buffer("lightClusters", lightClusters);
      writer.apply();
    }

    frame.skyVertexSet = pool->allocate(skyProgram->setLayouts[0]);
    frame.skySet = pool->allocate(skyProgram->setLayouts[1]);
    DescriptorWriter(context, skyProgram->fragment(), frame.skySet)
        .buffer("frame", frame.frameUniforms)
        .texture("environmentCube", environmentState->cube())
        .sampler("clampSampler", clampSampler)
        .apply();

    frame.resolveSet = pool->allocate(resolveProgram->setLayouts[0]);
    DescriptorWriter resolveWriter(context, resolveProgram->compute(), frame.resolveSet);
    resolveWriter.texture("depthBuffer", depthBuffer.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        .texture("normalRoughness", normalRoughness)
        .texture("reflectionWeight", reflectionWeight)
        .texture("sceneColor", hdrColor)
        .texture("prefilteredCube", environmentState->prefiltered())
        .storageTexture("reflection", reflectionMipZero)
        .buffer("uniforms", frame.reflectionUniforms)
        .sampler("clampSampler", clampSampler)
        .sampler("pointSampler", pointSampler)
        .sampler("materialSampler", defaultSampler);
    if (traced) {
      resolveWriter.textureArray("maps", hitTextureViews, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
          .accelerationStructure("scene", acceleration->topLevel)
          .buffer("traceInstances", traceInstanceBuffer)
          .buffer("materials", materialBuffer)
          .buffer("indices", activeScene->indexBuffer)
          .buffer("vertices", activeScene->vertexBuffer)
          .buffer("lights", lightBuffer);
    }
    resolveWriter.apply();

    const std::size_t slot = static_cast<std::size_t>(&frame - frames.data());
    const Image &lit = litColor[slot];
    const Image &history = litColor[(slot + 1) % kFramesInFlight];

    frame.compositeSet = pool->allocate(compositeProgram->setLayouts[0]);
    DescriptorWriter(context, compositeProgram->compute(), frame.compositeSet)
        .texture("sceneColor", hdrColor)
        .texture("reflectionWeight", reflectionWeight)
        .texture("normalRoughness", normalRoughness)
        .texture("reflection", reflection)
        .storageTexture("lit", litCurrent.view)
        .buffer("uniforms", frame.reflectionUniforms)
        .sampler("clampSampler", clampSampler)
        .sampler("pointSampler", pointSampler)
        .apply();

    frame.pathComparisonSet = pool->allocate(pathCompareProgram->setLayouts[0]);
    DescriptorWriter(context, pathCompareProgram->compute(), frame.pathComparisonSet)
        .texture("raster", pathComparisonRaster)
        .texture("traced", lit)
        .storageTexture("output", litCurrent.view)
        .buffer("control", frame.pathComparisonUniforms)
        .apply();

    frame.clusterSet = pool->allocate(clusterProgram->setLayouts[0]);
    DescriptorWriter(context, clusterProgram->compute(), frame.clusterSet)
        .buffer("lights", lightBuffer)
        .buffer("clusters", lightClusters)
        .buffer("cluster", frame.clusterUniforms)
        .apply();

    frame.temporalSet = pool->allocate(temporalProgram->setLayouts[0]);
    DescriptorWriter(context, temporalProgram->compute(), frame.temporalSet)
        .texture("currentColor", litCurrent)
        .texture("history", history.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        .texture("depthBuffer", depthBuffer.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        .texture("surfaceWeight", reflectionWeight)
        .storageTexture("resolved", lit.view)
        .buffer("taa", frame.temporalUniforms)
        .sampler("clampSampler", clampSampler)
        .sampler("pointSampler", pointSampler)
        .apply();

    frame.postVertexSet = pool->allocate(postProgram->setLayouts[0]);
    frame.postSet = pool->allocate(postProgram->setLayouts[1]);
    DescriptorWriter(context, postProgram->fragment(), frame.postSet)
        .buffer("post", frame.postUniforms)
        .texture("hdrColor", lit)
        .texture("bloom", bloomChain.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        .sampler("clampSampler", clampSampler)
        .apply();
  }

  // One set and parameter buffer per level and direction: a dispatch has no push constants.
  builtBloom = {settings.bloomThreshold, settings.bloomKnee, settings.bloomRadius};
  bloomUniformBuffers.clear();
  bloomDownSets.assign(bloomMipCount, VK_NULL_HANDLE);
  bloomUpSets.assign(bloomMipCount, VK_NULL_HANDLE);
  bloomUniformBuffers.reserve(bloomMipCount * 2);

  auto bloomBuffer = [&](const BloomUniforms &uniforms) -> const Buffer & {
    Buffer buffer(context, sizeof(BloomUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                  VMA_MEMORY_USAGE_AUTO,
                  VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                      VMA_ALLOCATION_CREATE_MAPPED_BIT,
                  "bloom.parameters");
    buffer.write(&uniforms, sizeof(uniforms));
    bloomUniformBuffers.push_back(std::move(buffer));
    return bloomUniformBuffers.back();
  };

  for (std::uint32_t mip = 0; mip < bloomMipCount; ++mip) {
    const float destinationWidth = static_cast<float>(std::max(1u, bloomChain.description.width >> mip));
    const float destinationHeight = static_cast<float>(std::max(1u, bloomChain.description.height >> mip));
    const float sourceWidth = mip == 0 ? static_cast<float>(width)
                                       : static_cast<float>(std::max(1u, bloomChain.description.width >> (mip - 1)));
    const float sourceHeight = mip == 0 ? static_cast<float>(height)
                                        : static_cast<float>(std::max(1u, bloomChain.description.height >> (mip - 1)));
    BloomUniforms downUniforms;
    // Only the first level thresholds; a negative threshold means none.
    downUniforms.parameters = {destinationWidth, destinationHeight,
                               mip == 0 ? settings.bloomThreshold : -1.0f, 0.0f};
    downUniforms.source = {sourceWidth, sourceHeight, mip == 0 ? 0.0f : static_cast<float>(mip - 1),
                           settings.bloomKnee};

    const Buffer &downBuffer = bloomBuffer(downUniforms);
    if (mip == 0) {
      // The first level reads the lit image, which alternates with the frame.
      for (std::size_t f = 0; f < frames.size(); ++f) {
        frames[f].bloomFirstSet = pool->allocate(bloomDownProgram->setLayouts[0]);
        DescriptorWriter(context, bloomDownProgram->compute(), frames[f].bloomFirstSet)
            .storageTexture("destination", bloomMipViews[0])
            .texture("source", litColor[f].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            .buffer("bloom", downBuffer)
            .sampler("clampSampler", clampSampler)
            .apply();
      }
    } else {
      bloomDownSets[mip] = pool->allocate(bloomDownProgram->setLayouts[0]);
      DescriptorWriter(context, bloomDownProgram->compute(), bloomDownSets[mip])
          .storageTexture("destination", bloomMipViews[mip])
          .texture("source", bloomChain.view, VK_IMAGE_LAYOUT_GENERAL)
          .buffer("bloom", downBuffer)
          .sampler("clampSampler", clampSampler)
          .apply();
    }

    if (mip + 1 < bloomMipCount) {
      BloomUniforms upUniforms;
      upUniforms.parameters = {destinationWidth, destinationHeight, -1.0f, settings.bloomRadius};
      upUniforms.source = {static_cast<float>(std::max(1u, bloomChain.description.width >> (mip + 1))),
                           static_cast<float>(std::max(1u, bloomChain.description.height >> (mip + 1))),
                           static_cast<float>(mip + 1), 0.0f};
      bloomUpSets[mip] = pool->allocate(bloomUpProgram->setLayouts[0]);
      DescriptorWriter(context, bloomUpProgram->compute(), bloomUpSets[mip])
          .storageTexture("destination", bloomMipViews[mip])
          .texture("source", bloomChain.view, VK_IMAGE_LAYOUT_GENERAL)
          .buffer("bloom", bloomBuffer(upUniforms))
          .sampler("clampSampler", clampSampler)
          .apply();
    }
  }
}

void Renderer::resize(std::uint32_t newWidth, std::uint32_t newHeight) {
  if (newWidth == width && newHeight == height) return;
  createTargets(newWidth, newHeight);
}

void Renderer::computeCascades(const Camera &camera) {
  const float aspect = static_cast<float>(width) / static_cast<float>(std::max(1u, height));
  const Mat4 view = camera.viewMatrix();
  const Vec3 light = sunDirection();

  // A near distance of its own: the camera's would collapse the first cascades into centimetres.
  const float nearPlane = std::max(camera.nearPlane, camera.sceneRadius * 0.05f);
  // Kept above the split near plane, or the splits run backwards.
  const float far = std::max(settings.shadowDistance > 0.0f
                                 ? settings.shadowDistance
                                 : std::max(camera.sceneRadius * 6.0f, camera.distance * 2.0f),
                             nearPlane * 1.01f);

  // Practical split scheme: a blend of logarithmic and uniform.
  std::array<float, kCascadeCount> splits{};
  for (std::uint32_t i = 0; i < kCascadeCount; ++i) {
    const float fraction = static_cast<float>(i + 1) / static_cast<float>(kCascadeCount);
    const float logarithmic = nearPlane * std::pow(far / nearPlane, fraction);
    const float uniform = nearPlane + (far - nearPlane) * fraction;
    splits[i] = lerp(uniform, logarithmic, settings.cascadeSplitLambda);
  }
  cascadeSplits = splits;

  const Mat4 inverseView = inverse(view);
  const float tanHalf = std::tan(camera.fieldOfView * 0.5f);
  float previous = nearPlane;
  for (std::uint32_t i = 0; i < kCascadeCount; ++i) {
    const float sliceNear = previous;
    const float sliceFar = splits[i];
    previous = sliceFar;

    Vec3 corners[8];
    int index = 0;
    for (float depth : {sliceNear, sliceFar}) {
      const float halfHeight = tanHalf * depth;
      const float halfWidth = halfHeight * aspect;
      for (float sy : {-1.0f, 1.0f})
        for (float sx : {-1.0f, 1.0f})
          corners[index++] = transformPoint(inverseView,
                                            {sx * halfWidth, sy * halfHeight, -depth});
    }

    // A sphere, whose radius does not change as the camera turns, so edges do not swim.
    Vec3 centre{0, 0, 0};
    for (const Vec3 &corner : corners) centre += corner;
    centre = centre / 8.0f;
    float radius = 0.0f;
    for (const Vec3 &corner : corners) radius = std::max(radius, length(corner - centre));
    radius = std::ceil(radius * 16.0f) / 16.0f;

    const float extent = radius * 2.0f;
    const float texelsPerUnit = static_cast<float>(kShadowResolution) / extent;

    // Centre snapped to whole shadow texels in the light's frame. `light` points at the sun,
    // so the shadow camera sits that way and looks back.
    const Vec3 up = std::abs(light.y) > 0.99f ? Vec3{0, 0, 1} : Vec3{0, 1, 0};
    const Mat4 lightView = lookAt(centre + light * (radius * 2.0f), centre, up);
    Vec3 lightSpaceCentre = transformPoint(lightView, centre);
    lightSpaceCentre.x = std::floor(lightSpaceCentre.x * texelsPerUnit) / texelsPerUnit;
    lightSpaceCentre.y = std::floor(lightSpaceCentre.y * texelsPerUnit) / texelsPerUnit;
    const Vec3 snapped = transformPoint(inverse(lightView), lightSpaceCentre);

    const Mat4 snappedView = lookAt(snapped + light * (radius * 2.0f), snapped, up);
    // Reaches well behind the slice, for casters outside it.
    const Mat4 projection = orthographic(-radius, radius, -radius, radius, 0.0f,
                                         radius * 4.0f + camera.sceneRadius * 2.0f);
    cascadeMatrices[i] = projection * snappedView;
  }
}

std::uint64_t Renderer::settingsKey() const {
  // Field by field rather than the struct's bytes, so padding never counts.
  std::uint64_t key = 0;
  auto add = [&](const auto &value) { key = hashBytes(&value, sizeof(value), key); };
  const RenderSettings &s = settings;
  add(s.sunAzimuth); add(s.sunElevation); add(s.sunColor); add(s.sunIntensity); add(s.sunAngularRadius);
  add(s.iblIntensity); add(s.skyTurbidity); add(s.skyIntensity); add(s.drawSky); add(s.groundPlane);
  add(s.groundRoughness); add(s.groundMetallic); add(s.groundColor);
  add(s.shadowMode); add(s.shadowSamples); add(s.shadowsEnabled); add(s.shadowDepthBias);
  add(s.shadowNormalBias); add(s.shadowSoftness); add(s.cascadeSplitLambda); add(s.shadowDistance);
  add(s.freezeCascades);
  add(s.occlusionMode); add(s.occlusionSamples); add(s.occlusionRadius);
  add(s.reflectionMode); add(s.reflectionDistance); add(s.reflectionThickness); add(s.reflectionSteps);
  add(s.accumulate); add(s.debugView); add(s.wireframe); add(s.frustumCulling);
  add(s.antialiasing); add(s.temporalFeedback); add(s.testLights); add(s.lightShadows);
  add(s.clusteredLights); add(s.renderer); add(s.pathBounces); add(s.pathSeed);
  add(s.pathAperture); add(s.pathFocusDistance); add(s.pathTextureFilter);
  add(s.pathDiEstimator); add(s.pathRestirReuse); add(s.pathRestirCandidates);
  add(s.pathStrategy); add(s.pathClamp); add(s.pathTargetSamples); add(s.cpuIntersector); add(s.pathDenoise);
  add(s.pathSamplesPerFrame); add(s.pathTemporal); add(s.pathBvhDiagnostic); add(s.pathBvhBuilder);
  add(s.pathBvhWidth);
  add(s.pathExecution);
  return key;
}

void Renderer::updateFrameData(const Camera &camera, float deltaSeconds) {
  elapsedSeconds += deltaSeconds;
  ++frameCounter;
  FrameResources &frame = frames[frameIndex];
  // A history from another renderer is nothing to reproject.
  if (settings.renderer != lastRenderer) {
    hasPreviousView = false;
    lastRenderer = settings.renderer;
  }

  const float aspect = static_cast<float>(width) / static_cast<float>(std::max(1u, height));
  const Mat4 view = camera.viewMatrix();
  Mat4 projection = camera.projectionMatrix(aspect);
  const Mat4 steadyViewProjection = projection * view;

  // The jitter is a clip-space shift proportional to w. Motion vectors and the accumulation
  // key use the un-jittered transform, so a still view stays still.
  // A path tracer jitters within the pixel itself.
  const bool temporal = settings.antialiasing == 2 && settings.debugView == 0 && settings.renderer == 0;
  if (temporal) {
    const Vec2 jitter = haltonJitter(frameCounter);
    projection.columns[2].x = -jitter.x * 2.0f / static_cast<float>(width);
    projection.columns[2].y = -jitter.y * 2.0f / static_cast<float>(height);
  }
  const Mat4 viewProjection = projection * view;
  const Mat4 inverseViewProjection = inverse(viewProjection);
  frame.hybridInverseViewProjection.write(&inverseViewProjection, sizeof(inverseViewProjection));
  const float scale = sceneScale();
  const bool traced = rayTracingAvailable();

  if (!settings.freezeCascades || !hasFrozenCascades) {
    computeCascades(camera);
    frozenFrustum = Frustum::fromViewProjection(viewProjection);
    hasFrozenCascades = true;
  }

  FrameUniforms uniforms;
  uniforms.viewProjection = viewProjection;
  uniforms.view = view;
  uniforms.inverseViewProjection = inverseViewProjection;
  uniforms.cameraPosition = Vec4(camera.position(), settings.exposure);
  uniforms.sunDirection = Vec4(sunDirection(), 0.0f);
  uniforms.sunColor = Vec4(settings.sunColor, settings.sunIntensity);
  const float rayMask = settings.groundPlane ? 3.0f : 1.0f; // Bit one is the ground plane.
  uniforms.environment = {settings.iblIntensity, environmentState->prefilteredMipCount(), rayMask,
                          elapsedSeconds};
  uniforms.shadowParameters = {settings.shadowDepthBias,
                               settings.shadowNormalBias, 1.0f / static_cast<float>(kShadowResolution),
                               settings.shadowSoftness};
  uniforms.cascadeSplits = {cascadeSplits[0], cascadeSplits[1], cascadeSplits[2], cascadeSplits[3]};
  const float visibleLights = static_cast<float>(lightCount);
  uniforms.viewportAndLights = {static_cast<float>(width), static_cast<float>(height), visibleLights,
                                static_cast<float>(settings.debugView)};
  const bool tracedShadows = traced && settings.shadowsEnabled && settings.shadowMode == 1;
  const bool tracedOcclusion = traced && settings.occlusionMode == 1;

  // Accumulate only while settings, view, targets and resources hold; never a debug view.
  std::uint64_t key = settingsKey();
  key = hashBytes(&steadyViewProjection, sizeof(steadyViewProjection), key);
  const std::uint32_t extentAndGeneration[3] = {width, height, resourceGeneration};
  key = hashBytes(extentAndGeneration, sizeof(extentAndGeneration), key);
  // A path tracer always accumulates while the view holds; that is how it converges.
  const bool accumulating = (settings.accumulate || settings.renderer != 0) && settings.debugView == 0;
  if (accumulating && key == accumulationKey) accumulatedFrames = std::min(accumulatedFrames + 1, 4096u);
  else accumulatedFrames = 0;
  accumulationKey = key;
  stats.accumulatedFrames = accumulatedFrames;
  stats.samples = accumulating ? accumulatedFrames + 1 : 1;
  // Noise moves and reflections sample the lobe only once a frame has accumulated: a moving
  // view is deterministic, and the first still frame seeds the average.
  const bool refining = accumulating && accumulatedFrames > 0;
  const float noiseFrame = refining ? static_cast<float>(frameCounter % 4096) : 0.0f;

  // Zero none, one the cascades, two a ray.
  const float shadowKind = !settings.shadowsEnabled ? 0.0f : (tracedShadows ? 2.0f : 1.0f);
  uniforms.rays = {shadowKind, static_cast<float>(std::max(1, settings.shadowSamples)),
                   settings.sunAngularRadius, tracedOcclusion ? 1.0f : 0.0f};
  uniforms.occlusion = {settings.occlusionRadius > 0.0f ? settings.occlusionRadius : scale * 0.05f,
                        static_cast<float>(std::max(1, settings.occlusionSamples)), noiseFrame, scale};
  // The grid reaches as far as the scene does from this camera; a slice stretched to cover
  // what lies beyond would hold lights it should not.
  float sceneViewFar = camera.nearPlane * 4.0f;
  if (activeScene) {
    for (const Primitive &primitive : activeScene->primitives) {
      if (!primitive.worldBounds.valid()) continue;
      const Vec3 low = primitive.worldBounds.minimum, high = primitive.worldBounds.maximum;
      for (int corner = 0; corner < 8; ++corner) {
        const Vec3 point{(corner & 1) ? high.x : low.x, (corner & 2) ? high.y : low.y,
                         (corner & 4) ? high.z : low.z};
        sceneViewFar = std::max(sceneViewFar, -transformPoint(view, point).z);
      }
    }
  }
  const float clusterFar = std::max(camera.nearPlane * 4.0f, sceneViewFar * 1.05f);
  const bool clustering = settings.clusteredLights && lightCount > 0;
  uniforms.options = {traced && settings.lightShadows ? 1.0f : 0.0f, camera.nearPlane, clusterFar,
                      clustering ? 1.0f : 0.0f};
  uniforms.clusters = {static_cast<float>(clusterGrid[0]), static_cast<float>(clusterGrid[1]),
                       static_cast<float>(clusterGrid[2]), static_cast<float>(kClusterCapacity)};

  // View extent per unit depth, for the culling pass.
  const float tangent = std::tan(camera.fieldOfView * 0.5f);
  ClusterUniforms clusterBlock;
  clusterBlock.view = view;
  clusterBlock.grid = uniforms.clusters;
  clusterBlock.projection = {tangent * aspect, tangent, camera.nearPlane, clusterFar};
  clusterBlock.counts = {static_cast<float>(lightCount), static_cast<float>(clusterCount), 0.0f, 0.0f};
  frame.clusterUniforms.write(&clusterBlock, sizeof(clusterBlock));
  const std::array<Vec4, 9> &harmonics = environmentState->irradianceCoefficients();
  for (std::size_t i = 0; i < 9; ++i) uniforms.sh[i] = harmonics[i];
  frame.frameUniforms.write(&uniforms, sizeof(uniforms));

  // The cascade transforms, as the rows the shaders apply with dot products.
  std::array<Vec4, kCascadeCount * 4> packed{};
  for (std::uint32_t cascade = 0; cascade < kCascadeCount; ++cascade) {
    for (int row = 0; row < 4; ++row) packed[cascade * 4 + row] = cascadeMatrices[cascade].row(row);
    std::array<Vec4, 4> rows{cascadeMatrices[cascade].row(0), cascadeMatrices[cascade].row(1),
                             cascadeMatrices[cascade].row(2), cascadeMatrices[cascade].row(3)};
    frame.cascadeStrided.write(rows.data(), sizeof(rows), cascade * cascadeStride);
  }
  frame.cascadePacked.write(packed.data(), sizeof(packed));

  int reflectionMode = settings.reflectionMode;
  if (reflectionMode == 2 && !traced) reflectionMode = 1;
  ReflectionUniforms reflect;
  reflect.viewProjection = viewProjection;
  reflect.inverseViewProjection = inverseViewProjection;
  reflect.camera = Vec4(camera.position(), static_cast<float>(reflectionMode));
  reflect.march = {settings.reflectionDistance > 0.0f ? settings.reflectionDistance : scale * 2.0f,
                   settings.reflectionThickness > 0.0f ? settings.reflectionThickness : scale * 0.05f,
                   static_cast<float>(std::max(8, settings.reflectionSteps)), camera.nearPlane};
  reflect.sun = Vec4(sunDirection(), settings.sunIntensity);
  reflect.sunColor = Vec4(settings.sunColor, settings.iblIntensity);
  reflect.target = {static_cast<float>(width), static_cast<float>(height), 1.0f / static_cast<float>(width),
                    1.0f / static_cast<float>(height)};
  reflect.environment = {environmentState->prefilteredMipCount(), static_cast<float>(reflectionMipCount),
                         scale, static_cast<float>(frameCounter)};
  reflect.debug = {static_cast<float>(settings.debugView), rayMask, static_cast<float>(accumulatedFrames),
                   visibleLights};
  reflect.sampling = {refining && reflectionMode == 2 ? 1.0f : 0.0f, noiseFrame, 0.0f, 0.0f};
  for (std::size_t i = 0; i < 9; ++i) reflect.sh[i] = harmonics[i];
  frame.reflectionUniforms.write(&reflect, sizeof(reflect));

  // Debug views pass through post untouched.
  const bool debugging = settings.debugView != 0;
  PostUniforms post;
  post.parameters = {debugging ? 1.0f : settings.exposure,
                     debugging ? 0.0f : settings.bloomStrength,
                     debugging ? 2.0f : static_cast<float>(settings.tonemap), settings.whitePoint};
  post.target = {static_cast<float>(width), static_cast<float>(height), 1.0f / static_cast<float>(width),
                 1.0f / static_cast<float>(height)};
  post.effects = {debugging ? 0.0f : settings.vignette, debugging ? 0.0f : settings.grain,
                  debugging ? 0.0f : settings.sharpen, elapsedSeconds};
  post.antialias = {settings.antialiasing == 1 && !debugging ? 1.0f : 0.0f, 0.0312f, 0.75f, 0.0f};
  frame.postUniforms.write(&post, sizeof(post));

  // Passes through until there is a previous frame to reproject from.
  TemporalUniforms temporalBlock;
  temporalBlock.inverseViewProjection = inverseViewProjection;
  temporalBlock.viewProjection = steadyViewProjection;
  temporalBlock.previousViewProjection = hasPreviousView ? previousViewProjection : steadyViewProjection;
  temporalBlock.target = {static_cast<float>(width), static_cast<float>(height),
                          1.0f / static_cast<float>(width), 1.0f / static_cast<float>(height)};
  temporalBlock.parameters = {static_cast<float>(accumulatedFrames),
                              std::clamp(settings.temporalFeedback, 0.02f, 1.0f), 0.0f,
                              temporal && hasPreviousView ? 1.0f : 0.0f};
  temporalBlock.camera = Vec4(camera.position(), std::max(scale, camera.nearPlane) * 4096.0f);
  frame.temporalUniforms.write(&temporalBlock, sizeof(temporalBlock));
  previousViewProjection = steadyViewProjection;
  hasPreviousView = true;

  visibleOpaque.clear();
  visibleBlended.clear();
  shadowCasters.clear();
  if (activeScene) {
    const Vec3 eye = camera.position();
    for (std::uint32_t i = 0; i < activeScene->primitives.size(); ++i) {
      if (!settings.groundPlane && static_cast<int>(i) == activeScene->groundPrimitive) continue;
      const Primitive &primitive = activeScene->primitives[i];
      const Material &material = activeScene->materials[primitive.material];
      // Casters are not frustum-culled: one behind the camera still casts into view.
      if (static_cast<int>(i) != activeScene->groundPrimitive) shadowCasters.push_back(i);
      if (settings.frustumCulling && !frozenFrustum.intersects(primitive.worldBounds)) continue;
      if (material.alphaMode == AlphaMode::Blend) visibleBlended.push_back(i);
      else visibleOpaque.push_back(i);
    }
    std::sort(visibleBlended.begin(), visibleBlended.end(), [&](std::uint32_t a, std::uint32_t b) {
      const float da = length(activeScene->primitives[a].worldBounds.center() - eye);
      const float db = length(activeScene->primitives[b].worldBounds.center() - eye);
      return da > db;
    });
  }

  stats.visiblePrimitives = static_cast<std::uint32_t>(visibleOpaque.size() + visibleBlended.size());
  stats.triangles = 0;
  for (std::uint32_t index : visibleOpaque) stats.triangles += activeScene->primitives[index].indexCount / 3;
  for (std::uint32_t index : visibleBlended) stats.triangles += activeScene->primitives[index].indexCount / 3;
}

void Renderer::recordShadowPass(VkCommandBuffer command) {
  FrameResources &frame = frames[frameIndex];
  stats.shadowDrawCalls = 0;
  // With traced shadows the cascades are not read, so they are not drawn.
  const bool drawCascades = settings.shadowsEnabled && activeScene &&
                            !(rayTracingAvailable() && settings.shadowMode == 1);

  transitionImage(command, shadowMap, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
  transitionImage(command, shadowDummyColor, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

  for (std::uint32_t cascade = 0; cascade < kCascadeCount; ++cascade) {
    VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAttachment.imageView = shadowLayerViews[cascade];
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
    rendering.renderArea = {{0, 0}, {kShadowResolution, kShadowResolution}};
    rendering.layerCount = 1;
    VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colorAttachment.imageView = shadowDummyColor.view;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &colorAttachment;
    rendering.pDepthAttachment = &depthAttachment;
    vkCmdBeginRendering(command, &rendering);

    if (drawCascades) {
      const VkViewport viewport{0.0f, 0.0f, static_cast<float>(kShadowResolution),
                                static_cast<float>(kShadowResolution), 0.0f, 1.0f};
      const VkRect2D scissor{{0, 0}, {kShadowResolution, kShadowResolution}};
      vkCmdSetViewport(command, 0, 1, &viewport);
      vkCmdSetScissor(command, 0, 1, &scissor);
      vkCmdSetDepthBias(command, 1.25f, 0.0f, 2.0f);
      vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowProgram->layout, 0, 1,
                              &frame.shadowVertexSets[cascade], 0, nullptr);

      const VkDeviceSize offset = 0;
      vkCmdBindVertexBuffers(command, 0, 1, &activeScene->vertexBuffer.handle, &offset);
      vkCmdBindIndexBuffer(command, activeScene->indexBuffer.handle, 0, VK_INDEX_TYPE_UINT32);

      // Opaque casters first, then masked; the pipeline changes only when sidedness does.
      std::uint32_t boundMaterial = UINT32_MAX;
      VkPipeline boundPipeline = VK_NULL_HANDLE;
      VkFrontFace boundFace = VK_FRONT_FACE_MAX_ENUM;
      for (int pass = 0; pass < 2; ++pass) {
        const bool masked = pass == 1;
        for (std::uint32_t index : shadowCasters) {
          const Primitive &primitive = activeScene->primitives[index];
          const Material &material = activeScene->materials[primitive.material];
          if (material.alphaMode == AlphaMode::Blend) continue; // Transparent surfaces cast nothing.
          if ((material.alphaMode == AlphaMode::Mask) != masked) continue;
          const VkPipeline wanted =
              masked ? (material.doubleSided ? shadowMaskedTwoSidedPipeline.handle : shadowMaskedPipeline.handle)
                     : (material.doubleSided ? shadowTwoSidedPipeline.handle : shadowPipeline.handle);
          if (wanted != boundPipeline) {
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, wanted);
            boundPipeline = wanted;
            boundMaterial = UINT32_MAX;
          }
          if (masked && primitive.material != boundMaterial) {
            vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowProgram->layout,
                                    1, 1, &shadowMaterialSets[primitive.material], 0, nullptr);
            boundMaterial = primitive.material;
          }
          if (const VkFrontFace face = frontFaceOf(primitive.transform); face != boundFace) {
            vkCmdSetFrontFace(command, face);
            boundFace = face;
          }
          vkCmdDrawIndexed(command, primitive.indexCount, 1, primitive.firstIndex,
                           primitive.vertexOffset, index);
          ++stats.shadowDrawCalls;
        }
      }
    }
    vkCmdEndRendering(command);
  }

  transitionImage(command, shadowMap, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

void Renderer::recordForwardPass(VkCommandBuffer command) {
  FrameResources &frame = frames[frameIndex];
  stats.drawCalls = 0;

  for (Image *target : {&hdrColor, &normalRoughness, &reflectionWeight, &gbufferBaseMetallic,
                        &gbufferGeometricCoverage, &gbufferEmissive, &gbufferIdentity})
    transitionImage(command, *target, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
  transitionImage(command, depthBuffer, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

  std::array<VkRenderingAttachmentInfo, 7> colorAttachments{};
  const Image *targets[7]{&hdrColor, &normalRoughness, &reflectionWeight, &gbufferBaseMetallic,
                          &gbufferGeometricCoverage, &gbufferEmissive, &gbufferIdentity};
  for (int i = 0; i < 7; ++i) {
    colorAttachments[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachments[i].imageView = targets[i]->view;
    colorAttachments[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachments[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachments[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachments[i].clearValue.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
  }

  VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  depthAttachment.imageView = depthBuffer.view;
  depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
  depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  depthAttachment.clearValue.depthStencil = {0.0f, 0}; // Reverse-Z clears to the far plane.

  VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
  rendering.renderArea = {{0, 0}, {width, height}};
  rendering.layerCount = 1;
  rendering.colorAttachmentCount = static_cast<std::uint32_t>(colorAttachments.size());
  rendering.pColorAttachments = colorAttachments.data();
  rendering.pDepthAttachment = &depthAttachment;
  vkCmdBeginRendering(command, &rendering);

  const VkViewport viewport{0.0f, static_cast<float>(height), static_cast<float>(width),
                            -static_cast<float>(height), 0.0f, 1.0f};
  const VkRect2D scissor{{0, 0}, {width, height}};
  vkCmdSetViewport(command, 0, 1, &viewport);
  vkCmdSetScissor(command, 0, 1, &scissor);

  auto drawSky = [&] {
    if (!settings.drawSky || settings.wireframe) return;
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, skyPipeline.handle);
    const VkDescriptorSet sets[2]{frame.skyVertexSet, frame.skySet};
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, skyProgram->layout, 0, 2, sets,
                            0, nullptr);
    vkCmdDraw(command, 3, 1, 0, 0);
  };

  if (activeScene && !activeScene->primitives.empty()) {
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(command, 0, 1, &activeScene->vertexBuffer.handle, &offset);
    vkCmdBindIndexBuffer(command, activeScene->indexBuffer.handle, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, forwardProgram->layout, 0, 1,
                            &frame.forwardVertexSet, 0, nullptr);

    // One pipeline per sidedness, rebound only where they alternate.
    auto drawList = [&](const std::vector<std::uint32_t> &list, const Pipeline &oneSided,
                        const Pipeline &twoSided) {
      if (list.empty() || !oneSided || !twoSided) return;
      std::uint32_t boundMaterial = UINT32_MAX;
      VkPipeline boundPipeline = VK_NULL_HANDLE;
      VkFrontFace boundFace = VK_FRONT_FACE_MAX_ENUM;  // the sky's pipeline in between resets it
      for (std::uint32_t index : list) {
        const Primitive &primitive = activeScene->primitives[index];
        const VkPipeline wanted =
            activeScene->materials[primitive.material].doubleSided ? twoSided.handle : oneSided.handle;
        if (wanted != boundPipeline) {
          vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, wanted);
          boundPipeline = wanted;
        }
        if (primitive.material != boundMaterial) {
          vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, forwardProgram->layout, 1,
                                  1, &frame.forwardMaterialSets[primitive.material], 0, nullptr);
          boundMaterial = primitive.material;
        }
        if (const VkFrontFace face = frontFaceOf(primitive.transform); face != boundFace) {
          vkCmdSetFrontFace(command, face);
          boundFace = face;
        }
        vkCmdDrawIndexed(command, primitive.indexCount, 1, primitive.firstIndex,
                         primitive.vertexOffset, index);
        ++stats.drawCalls;
      }
    };

    // Wireframe shows every edge regardless of side, so it is its own pair.
    const bool wire = settings.wireframe && forwardWirePipeline;
    drawList(visibleOpaque, wire ? forwardWirePipeline : forwardPipeline,
             wire ? forwardWirePipeline : forwardTwoSidedPipeline);

    // The sky fills the far plane before blended surfaces draw over it.
    drawSky();

    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, forwardProgram->layout, 0, 1,
                            &frame.forwardVertexSet, 0, nullptr);
    // Blended geometry already draws both faces.
    drawList(visibleBlended, wire ? forwardWirePipeline : forwardBlendPipeline,
             wire ? forwardWirePipeline : forwardBlendPipeline);
  } else {
    drawSky();
  }

  vkCmdEndRendering(command);
}

void Renderer::recordLightCulling(VkCommandBuffer command) {
  if (!settings.clusteredLights || lightCount == 0) return;
  FrameResources &frame = frames[frameIndex];

  // The forward pass of the frame before read this buffer.
  VkMemoryBarrier2 before{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  before.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
  before.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
  before.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  before.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.memoryBarrierCount = 1;
  dependency.pMemoryBarriers = &before;
  vkCmdPipelineBarrier2(command, &dependency);

  vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, clusterPipeline.handle);
  vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, clusterProgram->layout, 0, 1,
                          &frame.clusterSet, 0, nullptr);
  vkCmdDispatch(command, (clusterCount + 63) / 64, 1, 1);

  VkMemoryBarrier2 after{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  after.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  after.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
  after.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
  after.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
  dependency.pMemoryBarriers = &after;
  vkCmdPipelineBarrier2(command, &dependency);
}

void Renderer::recordReflections(VkCommandBuffer command) {
  FrameResources &frame = frames[frameIndex];

  for (Image *target : {&hdrColor, &normalRoughness, &reflectionWeight})
    transitionImage(command, *target, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  transitionImage(command, depthBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  transitionImage(command, reflection, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_WRITE_BIT);

  const std::uint32_t groupsX = (width + 7) / 8, groupsY = (height + 7) / 8;
  vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, resolvePipeline.handle);
  vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, resolveProgram->layout, 0, 1,
                          &frame.resolveSet, 0, nullptr);
  vkCmdDispatch(command, groupsX, groupsY, 1);

  generateMips(command, reflection);

  Image &lit = litColor[frameIndex];
  Image &history = litColor[(frameIndex + 1) % kFramesInFlight];
  // Never written on the first frame; the pass is told to pass through, but the layout must be the one the descriptor named.
  if (history.layout == VK_IMAGE_LAYOUT_UNDEFINED)
    transitionImage(command, history, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

  transitionImage(command, litCurrent, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_WRITE_BIT);
  vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, compositePipeline.handle);
  vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, compositeProgram->layout, 0, 1,
                          &frame.compositeSet, 0, nullptr);
  vkCmdDispatch(command, groupsX, groupsY, 1);
  transitionImage(command, litCurrent, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

  transitionImage(command, lit, VK_IMAGE_LAYOUT_GENERAL,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_WRITE_BIT);
  vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, temporalPipeline.handle);
  vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, temporalProgram->layout, 0, 1,
                          &frame.temporalSet, 0, nullptr);
  vkCmdDispatch(command, groupsX, groupsY, 1);
  transitionImage(command, lit, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

void Renderer::recordBloom(VkCommandBuffer command) {
  if (bloomMipCount == 0) return;

  transitionImage(command, bloomChain, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT);

  auto dispatchMip = [&](std::uint32_t mip) {
    const std::uint32_t mipWidth = std::max(1u, bloomChain.description.width >> mip);
    const std::uint32_t mipHeight = std::max(1u, bloomChain.description.height >> mip);
    vkCmdDispatch(command, (mipWidth + 7) / 8, (mipHeight + 7) / 8, 1);
  };

  vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, bloomDownPipeline.handle);
  for (std::uint32_t mip = 0; mip < bloomMipCount; ++mip) {
    const VkDescriptorSet set = mip == 0 ? frames[frameIndex].bloomFirstSet : bloomDownSets[mip];
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, bloomDownProgram->layout, 0, 1,
                            &set, 0, nullptr);
    dispatchMip(mip);
    computeImageBarrier(command, bloomChain.handle, VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 1);
  }

  vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, bloomUpPipeline.handle);
  for (std::uint32_t mip = bloomMipCount - 1; mip-- > 0;) {
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, bloomUpProgram->layout, 0, 1,
                            &bloomUpSets[mip], 0, nullptr);
    dispatchMip(mip);
    computeImageBarrier(command, bloomChain.handle, VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 1);
  }

  transitionImage(command, bloomChain, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

void Renderer::recordPost(VkCommandBuffer command, std::uint32_t imageIndex) {
  FrameResources &frame = frames[frameIndex];

  VkImageMemoryBarrier2 toAttachment{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  // Same stage the acquire semaphore is waited at, or the transition could run before the image is released.
  toAttachment.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  toAttachment.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  toAttachment.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
  toAttachment.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  toAttachment.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  toAttachment.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toAttachment.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toAttachment.image = swapchain.image(imageIndex);
  toAttachment.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.imageMemoryBarrierCount = 1;
  dependency.pImageMemoryBarriers = &toAttachment;
  vkCmdPipelineBarrier2(command, &dependency);

  VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  colorAttachment.imageView = swapchain.view(imageIndex);
  colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

  VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
  rendering.renderArea = {{0, 0}, swapchain.extent()};
  rendering.layerCount = 1;
  rendering.colorAttachmentCount = 1;
  rendering.pColorAttachments = &colorAttachment;
  vkCmdBeginRendering(command, &rendering);

  const VkExtent2D extent = swapchain.extent();
  const VkViewport viewport{0.0f, 0.0f, static_cast<float>(extent.width),
                            static_cast<float>(extent.height), 0.0f, 1.0f};
  const VkRect2D scissor{{0, 0}, extent};
  vkCmdSetViewport(command, 0, 1, &viewport);
  vkCmdSetScissor(command, 0, 1, &scissor);

  vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, postPipeline.handle);
  const VkDescriptorSet sets[2]{frame.postVertexSet, frame.postSet};
  vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, postProgram->layout, 0, 2, sets,
                          0, nullptr);
  vkCmdDraw(command, 3, 1, 0, 0);
  // The render pass stays open: the interface is drawn into it by the caller.
}

void Renderer::recordLinearCapture(VkCommandBuffer command) {
  // The CPU tracer's own float mean is the better source; writeLinearCapture takes it.
  if (settings.renderer == 1 && cpuTracer) return;
  // The GPU tracer's float accumulator, likewise: its w holds the sample count.
  linearCaptureFloat = settings.renderer >= 2 && pathAccumulation;
  Image &source = linearCaptureFloat ? pathAccumulation : litColor[frameIndex];
  const VkDeviceSize planeBytes =
      static_cast<VkDeviceSize>(width) * height * 4 * (linearCaptureFloat ? sizeof(float) : sizeof(std::uint16_t));
  const VkDeviceSize bytes = planeBytes * (linearCaptureFloat ? 3u : 1u);
  if (!linearCaptureBuffer || linearCaptureBuffer.size < bytes)
    linearCaptureBuffer = Buffer(context, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO,
                                 VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                 "capture.linear");
  const std::array<Image *, 3> gpuSources{&pathAccumulation, &pathAlbedoAccumulation, &pathNormalAccumulation};
  const std::size_t planeCount = linearCaptureFloat ? gpuSources.size() : 1u;
  for (std::size_t plane = 0; plane < planeCount; ++plane) {
    Image &image = linearCaptureFloat ? *gpuSources[plane] : source;
    transitionImage(command, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                        context.rayTracingShaderStage(),
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferImageCopy region{};
    region.bufferOffset = planeBytes * plane;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width, height, 1};
    vkCmdCopyImageToBuffer(command, image.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           linearCaptureBuffer.handle, 1, &region);
    transitionImage(command, image,
                    linearCaptureFloat ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                        context.rayTracingShaderStage(),
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
  }
  linearCaptureExtent = {width, height};
}

bool Renderer::writeLinearCapture(const std::string &path) {
  std::uint32_t w = 0, h = 0;
  std::vector<float> rows;
  // Float RGB rows from the bottom up, written as PFM or OpenEXR by the path's extension.
  if (settings.renderer == 1 && cpuTracer) {
    pt::CpuTracer::Image image;
    if (!cpuTracer->latest(image, 0, true)) return false;
    w = image.width;
    h = image.height;
    // Denoised only when denoising was asked for; a reference is otherwise the raw mean.
    const std::vector<float> &source = image.mean; // PFM is always raw; display filtering is independent.
    rows.resize(static_cast<std::size_t>(w) * h * 3);
    for (std::uint32_t y = 0; y < h; ++y)
      for (std::uint32_t x = 0; x < w; ++x)
        for (std::uint32_t c = 0; c < 3; ++c)
          rows[(static_cast<std::size_t>(h - 1 - y) * w + x) * 3 + c] =
              source[(static_cast<std::size_t>(y) * w + x) * 4 + c];
  } else {
    if (!linearCaptureBuffer || linearCaptureExtent.width == 0) return false;
    context.waitIdle();
    w = linearCaptureExtent.width;
    h = linearCaptureExtent.height;
    rows.resize(static_cast<std::size_t>(w) * h * 3);
    if (linearCaptureFloat) {
      // Sums with the sample count in w: the mean is their quotient.
      const auto *texels = static_cast<const float *>(linearCaptureBuffer.mapped);
      for (std::uint32_t y = 0; y < h; ++y)
        for (std::uint32_t x = 0; x < w; ++x) {
          const float *t = texels + (static_cast<std::size_t>(y) * w + x) * 4;
          const float count = std::max(t[3], 1.0f);
          for (std::uint32_t c = 0; c < 3; ++c) rows[(static_cast<std::size_t>(h - 1 - y) * w + x) * 3 + c] = t[c] / count;
        }
    } else {
      const auto *texels = static_cast<const std::uint16_t *>(linearCaptureBuffer.mapped);
      for (std::uint32_t y = 0; y < h; ++y)
        for (std::uint32_t x = 0; x < w; ++x)
          for (std::uint32_t c = 0; c < 3; ++c)
            rows[(static_cast<std::size_t>(h - 1 - y) * w + x) * 3 + c] =
                halfToFloat(texels[(static_cast<std::size_t>(y) * w + x) * 4 + c]);
    }
  }
  return pt::writeLinearImage(path, rows, w, h, "raw-linear-rgb");
}

void Renderer::recordHybridSnapshot(VkCommandBuffer command) {
  Image &lit = litColor[frameIndex];
  transitionImage(command, lit, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
  transitionImage(command, pathComparisonRaster, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
  VkImageCopy copy{};
  copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copy.extent = {width, height, 1};
  vkCmdCopyImage(command, lit.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 pathComparisonRaster.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
  transitionImage(command, pathComparisonRaster, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  transitionImage(command, lit, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

void Renderer::recordPathComparison(VkCommandBuffer command) {
  if (settings.pathComparison == 0) return;
  FrameResources &frame = frames[frameIndex];
  const std::uint32_t control[4] = {static_cast<std::uint32_t>(settings.pathComparison),
                                    gpuSamples, width, height};
  frame.pathComparisonUniforms.write(control, sizeof(control));

  Image &lit = litColor[frameIndex];
  transitionImage(command, litCurrent, VK_IMAGE_LAYOUT_GENERAL,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
  vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pathComparePipeline.handle);
  vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pathCompareProgram->layout,
                          0, 1, &frame.pathComparisonSet, 0, nullptr);
  vkCmdDispatch(command, (width + 7) / 8, (height + 7) / 8, 1);
  transitionImage(command, litCurrent, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
  transitionImage(command, lit, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
  VkImageCopy copy{};
  copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copy.extent = {width, height, 1};
  vkCmdCopyImage(command, litCurrent.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 lit.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
  transitionImage(command, lit, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

bool Renderer::writeGuideCapture(const std::string &path, int plane) {
  // The GPU tracers' first-hit albedo (plane 1) and normal (plane 2) sums, as means.
  if (settings.renderer < 2 || !linearCaptureFloat || !linearCaptureBuffer ||
      linearCaptureExtent.width == 0 || plane < 1 || plane > 2) return false;
  context.waitIdle();
  const std::uint32_t w = linearCaptureExtent.width, h = linearCaptureExtent.height;
  const std::size_t pixels = static_cast<std::size_t>(w) * h;
  const auto *planes = static_cast<const float *>(linearCaptureBuffer.mapped);
  const float *counts = planes;  // the radiance plane's w holds the sample count
  const float *guide = planes + pixels * 4 * static_cast<std::size_t>(plane);
  std::vector<float> rows(pixels * 3);
  for (std::uint32_t y = 0; y < h; ++y)
    for (std::uint32_t x = 0; x < w; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * w + x;
      const float count = std::max(counts[i * 4 + 3], 1.0f);
      for (std::uint32_t c = 0; c < 3; ++c)
        rows[(static_cast<std::size_t>(h - 1 - y) * w + x) * 3 + c] = guide[i * 4 + c] / count;
    }
  return pt::writeLinearImage(path, rows, w, h, plane == 1 ? "first-hit-albedo-mean" : "first-hit-normal-mean");
}

bool Renderer::writeDenoisedCapture(const std::string &path) {
  if (settings.renderer < 2 || !linearCaptureFloat || !linearCaptureBuffer ||
      linearCaptureExtent.width == 0 || !pt::Denoiser::available()) return false;
  context.waitIdle();
  const std::uint32_t w = linearCaptureExtent.width, h = linearCaptureExtent.height;
  const std::size_t pixels = static_cast<std::size_t>(w) * h;
  const auto *planes = static_cast<const float *>(linearCaptureBuffer.mapped);
  const std::size_t stride = pixels * 4;
  std::vector<float> color(pixels * 3), albedo(pixels * 3), normal(pixels * 3);
  for (std::size_t i = 0; i < pixels; ++i) {
    const float count = std::max(planes[i * 4 + 3], 1.0f);
    for (std::size_t c = 0; c < 3; ++c) {
      color[i * 3 + c] = planes[i * 4 + c] / count;
      albedo[i * 3 + c] = planes[stride + i * 4 + c] / count;
      normal[i * 3 + c] = planes[stride * 2 + i * 4 + c] / count;
    }
  }
  pt::Denoiser denoiser;
  if (!denoiser.error().empty()) return false;
  const std::vector<float> filtered = denoiser.denoise(color, albedo, normal, w, h);
  if (filtered.empty()) return false;
  std::vector<float> rows(filtered.size());
  for (std::uint32_t y = 0; y < h; ++y)
    std::copy_n(filtered.data() + static_cast<std::size_t>(y) * w * 3, static_cast<std::size_t>(w) * 3,
                rows.data() + static_cast<std::size_t>(h - 1 - y) * w * 3);
  return pt::writeLinearImage(path, rows, w, h, "denoised-linear-rgb");
}

void Renderer::render(VkCommandBuffer command, std::uint32_t imageIndex, const Camera &camera,
                      float deltaSeconds) {
  frameIndex = swapchain.frameIndex();

  // Bloom parameters are baked into the sets' buffers, so a slider rebuilds them; safe here, before any recording.
  if (builtBloom[0] != settings.bloomThreshold || builtBloom[1] != settings.bloomKnee ||
      builtBloom[2] != settings.bloomRadius)
    rebuildSceneResources();
  // Same for the light count.
  if (builtTestLights != std::max(0, settings.testLights)) {
    buildLights();
    rebuildSceneResources();
  }

  if (timestampPool) {
    const std::uint32_t first = frameIndex * 2;
    std::uint64_t timestamps[2]{};
    if (vkGetQueryPoolResults(context.device, timestampPool, first, 2, sizeof(timestamps), timestamps,
                              sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
      const double period = context.properties.limits.timestampPeriod;
      stats.gpuMilliseconds = static_cast<float>(
          static_cast<double>(timestamps[1] - timestamps[0]) * period * 1e-6);
      if (freshSampleFrame[frameIndex]) stats.freshSampleGpuMilliseconds = stats.gpuMilliseconds;
    }
    vkCmdResetQueryPool(command, timestampPool, first, 2);
    vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, timestampPool, first);
  }
  freshSampleFrame[frameIndex] = false;

  updateFrameData(camera, deltaSeconds);
  const bool reconstruct = settings.pathTemporal && settings.renderer != 0 &&
                           !(settings.renderer == 3 && settings.pathBvhDiagnostic != 0);
  if ((!reconstruct && pathReconstruction) || reconstructionBackend != settings.renderer) {
    context.waitIdle();
    pathReconstruction.reset();
    pathSetsGeneration = ~0u;
  }
  reconstructionBackend = settings.renderer;
  if (reconstruct && settings.renderer >= 2 && !pathReconstruction) {
    pathReconstruction = std::make_unique<PathReconstruction>(context, width, height);
    pathSetsGeneration = ~0u;
  }
  stats.previewScale = 1;
  stats.pathsPerSecond = 0.0; // only a current CPU publication supplies this statistic
  stats.activeRenderer = settings.renderer;
  stats.unavailableReason.clear();
  stats.activeWavefront = false;
  if (settings.renderer == 1) {
    recordCpuTrace(command, camera);
  } else {
    // Leaving the CPU tracer frees its threads and its copy of the scene.
    if (cpuTracer) {
      cpuTracer.reset();
      cpuScene.reset();
      cpuImage = {};
      cpuUploaded.fill(0);
    }
    if (settings.renderer == 4 && pathTraceHybridPipeline && rayTracingAvailable()) {
      recordLightCulling(command);
      recordShadowPass(command);
      recordForwardPass(command);
      if (settings.pathComparison != 0) {
        recordReflections(command);
        recordHybridSnapshot(command);
      }
      recordGpuTrace(command, camera);
      recordPathComparison(command);
    } else if ((settings.renderer == 5 && pathRayPipeline && rayPipelineAvailable()) ||
        (settings.renderer == 2 && pathTraceRtPipeline && rayTracingAvailable()) ||
        (settings.renderer == 3 && pathTracePipeline)) {
      recordGpuTrace(command, camera);
    } else {
      // The rasteriser, or a requested tracer that cannot run here: the frame is rasterised,
      // and only the second case says so.
      stats.activeRenderer = 0;
      if (settings.renderer != 0) stats.unavailableReason = "the selected renderer is unavailable on this device";
      recordLightCulling(command);
      recordShadowPass(command);
      recordForwardPass(command);
      recordReflections(command);
    }
  }
  stats.reconstructionMilliseconds = pathReconstruction ? pathReconstruction->filterMilliseconds() : 0.0f;
  stats.guideUploadMilliseconds = pathReconstruction ? pathReconstruction->uploadMilliseconds() : 0.0f;
  stats.guideTransferMilliseconds = pathReconstruction ? pathReconstruction->transferMilliseconds() : 0.0f;
  stats.reconstructionBytes = pathReconstruction ? pathReconstruction->bytes() : 0;
  recordBloom(command);
  recordPost(command, imageIndex);

  if (timestampPool)
    vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, timestampPool,
                         frameIndex * 2 + 1);
}

pt::PathUniforms Renderer::makePathUniforms(const Camera &camera) {
  // The rasteriser's own view and projection, as a pinhole: forward, and right and up scaled
  // to the image's half extent at unit distance.
  const Mat4 cameraToWorld = inverse(camera.viewMatrix());
  const Vec3 right = normalize(Vec3{cameraToWorld.columns[0].x, cameraToWorld.columns[0].y, cameraToWorld.columns[0].z});
  const Vec3 up = normalize(Vec3{cameraToWorld.columns[1].x, cameraToWorld.columns[1].y, cameraToWorld.columns[1].z});
  const Vec3 back = normalize(Vec3{cameraToWorld.columns[2].x, cameraToWorld.columns[2].y, cameraToWorld.columns[2].z});
  const float tangent = std::tan(camera.fieldOfView * 0.5f);
  const float aspect = static_cast<float>(width) / static_cast<float>(std::max(1u, height));
  const Vec3 eye = camera.position();
  auto vector = [](Vec3 v, float w) { return pt::float4(v.x, v.y, v.z, w); };

  pt::PathUniforms u{};
  u.cameraPosition = vector(eye, 1.0f);
  u.cameraForward = vector(-back, 0.0f);
  u.cameraRight = vector(right * (tangent * aspect), 0.0f);
  u.cameraUp = vector(up * tangent, 0.0f);
  const float guideMode = settings.renderer == 3 && settings.pathBvhDiagnostic != 0
                              ? static_cast<float>(settings.pathBvhDiagnostic + 1)
                              : settings.pathTemporal || settings.renderer == 4 ? 1.0f : 0.0f;
  u.image = {static_cast<float>(width), static_cast<float>(height), guideMode, 1.0f};
  u.path = {static_cast<float>(std::max(1, settings.pathBounces)), pt::kRouletteStartBounce, std::max(0.0f, settings.pathClamp),
            static_cast<float>(std::clamp(settings.pathStrategy, 0, 2))};
  const Image &environment = environmentState->traceImage();
  u.environment = {settings.iblIntensity, static_cast<float>(environment.description.width),
                   static_cast<float>(environment.description.height), settings.drawSky ? 1.0f : 0.0f};
  const std::array<float, 4> &distribution = environmentState->traceDistributionInfo();
  u.distribution = {distribution[0], distribution[1], distribution[2], distribution[3]};
  // With an .hdr the environment's own sun is the sun; the procedural sky's disc is analytic.
  if (environmentState->procedural()) {
    const double radius = std::max(static_cast<double>(settings.sunAngularRadius), 1e-4);
    const double solidAngle = 2.0 * 3.14159265358979323846 * (1.0 - std::cos(radius));
    const Vec3 radiance = settings.sunColor * static_cast<float>(settings.sunIntensity / solidAngle);
    u.sunDirection = vector(sunDirection(), static_cast<float>(std::cos(radius)));
    u.sunRadiance = vector(radiance, static_cast<float>(solidAngle));
  }
  const std::uint32_t mask = pt::kRayMaskScene | (settings.groundPlane ? pt::kRayMaskGround : 0u) | pt::kRayMaskBlended;
  u.counts = {static_cast<std::uint32_t>(hostLights.size()), mask, settings.pathSeed, 0u};
  u.emissive = {static_cast<std::uint32_t>(pathEmissiveMetadata.size()), 0u, 0u, 0u};
  // Thin lens: the focus defaults to the orbit target's depth. The pixel spread angle
  // is the angle one pixel subtends at the image centre (ray cones).
  // Hybrid rasterises primaries through a pinhole, so it never gets a lens.
  const float focus = settings.pathFocusDistance > 0.0f ? settings.pathFocusDistance : std::max(camera.distance, 1e-3f);
  u.lens = {settings.renderer == 4 ? 0.0f : std::max(0.0f, settings.pathAperture), focus,
            settings.pathTextureFilter == 1 ? 1.0f : 0.0f,
            std::atan(2.0f * tangent / static_cast<float>(std::max(1u, height)))};
  // ReSTIR DI: pure path tracers only; hybrid and raster never select it.
  const bool restir = settings.pathDiEstimator == 1 && settings.renderer != 0 && settings.renderer != 4;
  u.estimator = {restir ? 1u : 0u,
                 static_cast<std::uint32_t>(std::clamp(settings.pathRestirCandidates, 1,
                                                       static_cast<int>(pt::kPtRestirMaxCandidates))),
                 static_cast<std::uint32_t>(std::clamp(settings.pathRestirReuse, 0, 3)), 0u};
  return u;
}

void Renderer::buildCpuScene() {
  const auto started = std::chrono::steady_clock::now();
  context.waitIdle();
  auto scene = std::make_shared<pt::CpuScene>();

  // Geometry, read back from the buffers the rasteriser draws.
  const std::vector<std::uint8_t> vertexBytes =
      uploader.readBuffer(activeScene->vertexBuffer, static_cast<VkDeviceSize>(activeScene->vertexCount) * sizeof(Vertex));
  scene->vertices.resize(vertexBytes.size() / sizeof(float));
  std::memcpy(scene->vertices.data(), vertexBytes.data(), vertexBytes.size());
  const std::vector<std::uint8_t> indexBytes =
      uploader.readBuffer(activeScene->indexBuffer, static_cast<VkDeviceSize>(activeScene->indexCount) * sizeof(std::uint32_t));
  scene->indices.resize(indexBytes.size() / sizeof(std::uint32_t));
  std::memcpy(scene->indices.data(), indexBytes.data(), indexBytes.size());
  scene->instances = traceScene.instances;
  for (const std::uint32_t primitive : traceScene.primitives)
    scene->triangleCounts.push_back(activeScene->primitives[primitive].indexCount / 3);

  // The hit table's images at level zero, each read once however many slots share it.
  std::map<const Image *, std::uint32_t> textureOf;
  scene->textures.textures.clear();
  for (std::uint32_t slot = 0; slot < hitTextureImages.size() && slot < pt::kHitTextureSlots; ++slot) {
    Image *image = hitTextureImages[slot];
    auto found = textureOf.find(image);
    if (found == textureOf.end()) {
      pt::HostTexture texture;
      texture.width = image->description.width;
      texture.height = image->description.height;
      texture.srgb = image->description.format == VK_FORMAT_R8G8B8A8_SRGB;
      const std::vector<std::uint8_t> bytes = uploader.readImage(*image, 4);
      texture.texels.resize(bytes.size() / 4);
      std::memcpy(texture.texels.data(), bytes.data(), bytes.size());
      pt::buildMipChain(texture);  // ray-cone filtering
      found = textureOf.emplace(image, static_cast<std::uint32_t>(scene->textures.textures.size())).first;
      scene->textures.textures.push_back(std::move(texture));
    }
    scene->textures.slots[slot] = found->second;
  }

  Image &environment = environmentState->traceImage();
  const std::vector<std::uint8_t> environmentBytes = uploader.readImage(environment, 16);
  scene->environment.width = environment.description.width;
  scene->environment.height = environment.description.height;
  scene->environment.texels.resize(environmentBytes.size() / sizeof(float));
  std::memcpy(scene->environment.texels.data(), environmentBytes.data(), environmentBytes.size());
  scene->distribution = environmentState->traceDistribution();
  if (scene->distribution.empty()) scene->distribution.assign(4, 0.0f);

  if (specularAlbedo.empty()) specularAlbedo = pt::buildSpecularAlbedoTable();
  scene->specularAlbedo = specularAlbedo;
  std::vector<pt::Material> pathMaterials;
  pathMaterials.reserve(activeScene->materials.size());
  for (const Material &source : activeScene->materials) {
    pt::Material material{};
    std::memcpy(&material, &source.uniforms, sizeof(material));
    pathMaterials.push_back(material);
  }
  scene->emissiveTriangles = pt::buildEmissiveTriangles(scene->vertices, scene->indices, scene->instances,
                                                         scene->triangleCounts, pathMaterials);
  pathEmissiveMetadata = scene->emissiveTriangles;
  pathEmissiveVersion = sceneVersion;
  scene->bvhStatistics = pt::buildBvh(scene->vertices, scene->indices, scene->instances, scene->triangleCounts,
                                      scene->bvh, cpuTracer->threads());
  if (settings.cpuIntersector == 1 && pt::EmbreeScene::available()) {
    scene->embree = std::make_shared<pt::EmbreeScene>(*scene);
    logInfo("Embree scene built in {:.0f} ms", scene->embree->buildMilliseconds);
  }
  if (settings.cpuIntersector == 2 && pt::cpuAvx2Available()) {
    scene->wideAvx2 = std::make_shared<pt::WideBvh>(pt::buildWideBvh(scene->bvh, scene->instances, 8u));
    logInfo("AVX2 BVH8: {} quantized nodes, converted in {:.2f} ms",
            scene->wideAvx2->nodes.size(), scene->wideAvx2->milliseconds);
  }
  cpuScene = std::move(scene);
  const pt::BvhStatistics &bvh = cpuScene->bvhStatistics;
  logInfo("CPU path tracer: {} instances, {} triangles, {} + {} BVH nodes (depth {} + {}, SAH {:.1f}) in {:.0f} ms; "
          "scene ready in {:.0f} ms",
          bvh.instances, bvh.triangles, bvh.topNodes, bvh.bottomNodes, bvh.topDepth, bvh.bottomDepth, bvh.sahCost,
          bvh.milliseconds,
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
}

void Renderer::ensureEmissiveTriangles() {
  if (!activeScene) return;
  if (pathEmissiveVersion != sceneVersion) {
    const std::vector<std::uint8_t> vertexBytes = uploader.readBuffer(
        activeScene->vertexBuffer, static_cast<VkDeviceSize>(activeScene->vertexCount) * sizeof(Vertex));
    const std::vector<std::uint8_t> indexBytes = uploader.readBuffer(
        activeScene->indexBuffer, static_cast<VkDeviceSize>(activeScene->indexCount) * sizeof(std::uint32_t));
    std::vector<float> vertices(vertexBytes.size() / sizeof(float));
    std::vector<std::uint32_t> indices(indexBytes.size() / sizeof(std::uint32_t));
    std::memcpy(vertices.data(), vertexBytes.data(), vertexBytes.size());
    std::memcpy(indices.data(), indexBytes.data(), indexBytes.size());
    std::vector<std::uint32_t> triangleCounts;
    for (const std::uint32_t primitive : traceScene.primitives)
      triangleCounts.push_back(activeScene->primitives[primitive].indexCount / 3u);
    std::vector<pt::Material> materials;
    for (const Material &source : activeScene->materials) {
      pt::Material material{};
      std::memcpy(&material, &source.uniforms, sizeof(material));
      materials.push_back(material);
    }
    pathEmissiveMetadata = pt::buildEmissiveTriangles(vertices, indices, traceScene.instances,
                                                       triangleCounts, materials);
    pathEmissiveVersion = sceneVersion;
  }
  if (!pathEmissiveBuffer) {
    std::vector<pt::PtEmissiveTriangle> rows = pathEmissiveMetadata;
    if (rows.empty()) rows.push_back({});
    pathEmissiveBuffer = uploader.createBuffer(rows.data(), rows.size() * sizeof(pt::PtEmissiveTriangle),
                                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "path.emissive.triangles");
    pathSetsGeneration = ~0u;
  }
}

void Renderer::buildSoftwareBvh() {
  if (!activeScene || (softwareBvhVersion == sceneVersion && softwareBvhBuilder == settings.pathBvhBuilder &&
                       softwareBvhWidth == settings.pathBvhWidth)) return;
  const auto started = std::chrono::steady_clock::now();
  context.waitIdle();

  if (settings.pathBvhBuilder == 1) {
    if (settings.pathBvhWidth != 0)
      throw std::runtime_error("quantized wide BVHs currently require the CPU SAH builder");
    GpuBvhBuildResult built = buildGpuBvh(context, uploader, *activeScene, traceScene);
    softwareBvhNodes = std::move(built.nodes);
    softwareBvhTriangles = std::move(built.triangles);
    softwareBvhStatistics = built.statistics;
    softwareBvhScratchBytes = built.scratchBytes;
    softwareBvhOutputBytes = built.outputBytes;
    softwareBvhRadixPasses = built.radixPasses;
    softwareBvhMaximumStack = built.maximumBuilderStack;
    std::vector<pt::TraceInstance> rows = std::move(built.instances);
    if (rows.empty()) rows.push_back({});
    // Own buffer: the raster and hardware sets keep naming the shared instance table.
    softwareTraceInstanceBuffer = uploader.createBuffer(rows.data(), rows.size() * sizeof(pt::TraceInstance),
                                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "path.bvh.instances");
    softwareBvhVersion = sceneVersion;
    softwareBvhBuilder = settings.pathBvhBuilder;
    softwareBvhWidth = settings.pathBvhWidth;
    pathSetsGeneration = ~0u;
    logInfo("GPU LBVH: {} instances, {} triangles, {} + {} nodes (depth {} + {}), {} radix passes, "
            "stack {}, {:.1f} MiB scratch, {:.1f} MiB output, built in {:.1f} ms",
            softwareBvhStatistics.instances, softwareBvhStatistics.triangles,
            softwareBvhStatistics.topNodes, softwareBvhStatistics.bottomNodes,
            softwareBvhStatistics.topDepth, softwareBvhStatistics.bottomDepth,
            softwareBvhRadixPasses, softwareBvhMaximumStack,
            static_cast<double>(softwareBvhScratchBytes) / (1024.0 * 1024.0),
            static_cast<double>(softwareBvhOutputBytes) / (1024.0 * 1024.0),
            softwareBvhStatistics.milliseconds);
    return;
  }

  const std::vector<std::uint8_t> vertexBytes = uploader.readBuffer(
      activeScene->vertexBuffer, static_cast<VkDeviceSize>(activeScene->vertexCount) * sizeof(Vertex));
  std::vector<float> vertices(vertexBytes.size() / sizeof(float));
  std::memcpy(vertices.data(), vertexBytes.data(), vertexBytes.size());
  const std::vector<std::uint8_t> indexBytes = uploader.readBuffer(
      activeScene->indexBuffer, static_cast<VkDeviceSize>(activeScene->indexCount) * sizeof(std::uint32_t));
  std::vector<std::uint32_t> indices(indexBytes.size() / sizeof(std::uint32_t));
  std::memcpy(indices.data(), indexBytes.data(), indexBytes.size());

  std::vector<pt::TraceInstance> instances = traceScene.instances;
  std::vector<std::uint32_t> triangleCounts;
  triangleCounts.reserve(traceScene.primitives.size());
  for (const std::uint32_t primitive : traceScene.primitives)
    triangleCounts.push_back(activeScene->primitives[primitive].indexCount / 3);

  pt::Bvh bvh;
  const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
  softwareBvhStatistics = pt::buildBvh(vertices, indices, instances, triangleCounts, bvh, threads);
  std::vector<pt::QuantizedWideNode> wideNodes;
  if (settings.pathBvhWidth != 0) {
    pt::WideBvh wide = pt::buildWideBvh(bvh, instances, settings.pathBvhWidth == 1 ? 4u : 8u);
    wideNodes = std::move(wide.nodes);
    bvh.triangles = std::move(wide.triangles);
    instances = std::move(wide.instances);
  }
  const pt::float4 empty{};
  const void *nodeData = settings.pathBvhWidth != 0
      ? (wideNodes.empty() ? static_cast<const void *>(&empty) : static_cast<const void *>(wideNodes.data()))
      : (bvh.nodes.empty() ? static_cast<const void *>(&empty) : static_cast<const void *>(bvh.nodes.data()));
  const void *triangleData = bvh.triangles.empty() ? static_cast<const void *>(&empty) : bvh.triangles.data();
  const VkDeviceSize nodeBytes = settings.pathBvhWidth != 0
      ? std::max<std::size_t>(1, wideNodes.size()) * sizeof(pt::QuantizedWideNode)
      : std::max<std::size_t>(1, bvh.nodes.size()) * sizeof(pt::float4);
  const VkDeviceSize triangleBytes = std::max<std::size_t>(1, bvh.triangles.size()) * sizeof(pt::float4);
  softwareBvhNodes = uploader.createBuffer(nodeData, nodeBytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, "path.bvh.nodes");
  softwareBvhTriangles = uploader.createBuffer(triangleData, triangleBytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, "path.bvh.triangles");

  // The builder's rows carry its BLAS roots and triangle offsets. They get their own
  // buffer: replacing the shared table would free it under the raster descriptor sets.
  std::vector<pt::TraceInstance> rows = std::move(instances);
  if (rows.empty()) rows.push_back({});
  softwareTraceInstanceBuffer = uploader.createBuffer(rows.data(), rows.size() * sizeof(pt::TraceInstance),
                                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "path.bvh.instances");
  softwareBvhVersion = sceneVersion;
  softwareBvhBuilder = settings.pathBvhBuilder;
  softwareBvhWidth = settings.pathBvhWidth;
  softwareBvhScratchBytes = softwareBvhOutputBytes = 0;
  softwareBvhRadixPasses = softwareBvhMaximumStack = 0;
  pathSetsGeneration = ~0u;
  logInfo("GPU software BVH: {} instances, {} triangles, {} + {} nodes (depth {} + {}, SAH {:.1f}) "
          "built in {:.0f} ms; uploaded in {:.0f} ms total",
          softwareBvhStatistics.instances, softwareBvhStatistics.triangles,
          softwareBvhStatistics.topNodes, softwareBvhStatistics.bottomNodes,
          softwareBvhStatistics.topDepth, softwareBvhStatistics.bottomDepth,
          softwareBvhStatistics.sahCost, softwareBvhStatistics.milliseconds,
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
}

void Renderer::recordGpuTrace(VkCommandBuffer command, const Camera &camera) {
  FrameResources &frame = frames[frameIndex];
  if (!activeScene) return;
  ensureEmissiveTriangles();
  const bool hybrid = settings.renderer == 4;
  const bool software = settings.renderer == 3;
  const bool pipelineBackend = settings.renderer == 5;
  if (software) buildSoftwareBvh();
  const bool emissiveSoftware = software && !pathEmissiveMetadata.empty();
  const bool wideSoftware = software && settings.pathBvhWidth != 0;
  const bool wavefront = settings.pathExecution == 1 && !hybrid;
  stats.activeWavefront = wavefront;
  Program *program = hybrid ? pathTraceHybridProgram.get() :
                     wideSoftware && emissiveSoftware ? pathTraceWideEmissiveProgram.get() :
                     wideSoftware ? pathTraceWideProgram.get() :
                     emissiveSoftware ? pathTraceEmissiveProgram.get() :
                     software ? pathTraceProgram.get() : pipelineBackend ? nullptr : pathTraceRtProgram.get();
  Pipeline *pipeline = hybrid ? &pathTraceHybridPipeline :
                       wideSoftware && emissiveSoftware ? &pathTraceWideEmissivePipeline :
                       wideSoftware ? &pathTraceWidePipeline :
                       emissiveSoftware ? &pathTraceEmissivePipeline :
                       software ? &pathTracePipeline : pipelineBackend ? nullptr : &pathTraceRtPipeline;
  if (wavefront && settings.pathWaveAllocation == 1 && !waveSubgroupAllocationSupported)
    throw std::runtime_error("subgroup queue allocation needs subgroup arithmetic and ballot in compute");
  const int backendKey = (wavefront ? 100 + (waveSubgroupAllocationActive() ? 1000 : 0) + (waveFusionActive() ? 2000 : 0) : 0) + (pipelineBackend ? 50 : hybrid ? 5 : wideSoftware ? 10 + settings.pathBvhWidth * 2 + (emissiveSoftware ? 1 : 0) :
                         emissiveSoftware ? 4 : settings.renderer);
  if (pipelineBackend ? !pathRayPipeline : (!program || !*pipeline)) return;

  if (hybrid) {
    for (Image *image : {&normalRoughness, &gbufferBaseMetallic, &gbufferGeometricCoverage,
                         &gbufferIdentity})
      transitionImage(command, *image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                      VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,  // hybrid is a compute ray-query entry
                      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  }

  // Float accumulators, made the first time and at each resize; they live in GENERAL.
  if (!pathAccumulation) {
    ImageDescription description;
    description.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    description.width = width;
    description.height = height;
    description.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    description.name = "path.accumulation";
    pathAccumulation = Image(context, description);
    description.name = "path.albedo";
    pathAlbedoAccumulation = Image(context, description);
    description.name = "path.normal";
    pathNormalAccumulation = Image(context, description);
    for (Image *image : {&pathAccumulation, &pathAlbedoAccumulation, &pathNormalAccumulation})
      transitionImage(command, *image, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                          context.rayTracingShaderStage(),
                      VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    pathSetsGeneration = ~0u;
    gpuSamples = 0;
  }
  if (pathDistributionVersion != environmentState->version()) {
    std::vector<float> distribution = environmentState->traceDistribution();
    if (distribution.empty()) distribution.assign(4, 0.0f);
    context.waitIdle();
    pathDistribution = uploader.createBuffer(distribution.data(), distribution.size() * sizeof(float),
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "path.distribution");
    pathDistributionVersion = environmentState->version();
    pathSetsGeneration = ~0u;
  }
  if (!pathAlbedoTable) {
    if (specularAlbedo.empty()) specularAlbedo = pt::buildSpecularAlbedoTable();
    pathAlbedoTable = uploader.createBuffer(specularAlbedo.data(), specularAlbedo.size() * sizeof(float),
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "path.albedo.table");
  }
  // The sets name this frame's lit image; the pool is reset whenever the scene's resources are.
  // Wavefront only indexes this binding while temporal guide collection is enabled, in
  // which case PathReconstruction supplies the full-sized buffer.  Keep its dormant
  // binding small so switching from the megakernel cannot replace an in-flight buffer.
  const std::size_t guideCount = hybrid && !pathReconstruction
                                     ? static_cast<std::size_t>(width) * height : 1u;
  const VkDeviceSize guideBytes = guideCount * sizeof(pt::PtReconstructionSample);
  if (!pathReconstructionSamples || pathReconstructionSamples.size < guideBytes) {
    // Descriptor sets from submitted frames can still name the old dummy buffer.  This
    // growth is rare (currently only entry into hybrid mode), so drain them before VMA
    // destroys the allocation during move-assignment.
    if (pathReconstructionSamples) context.waitIdle();
    // Valid dormant binding when guide collection is disabled.
    pathReconstructionSamples = Buffer(context, guideBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, 0, "path.guides");
    pathSetsGeneration = ~0u;
  }
  if (wavefront) {
    // Queues hold one batch of whole pixels (pt/WavefrontPlan.h); the allocation follows the
    // configured samples per frame, and a smaller final frame uses whole-pixel sub-batches.
    const std::uint32_t planSamples = static_cast<std::uint32_t>(std::max(1, settings.pathSamplesPerFrame));
    const std::uint64_t pixels = static_cast<std::uint64_t>(width) * height;
    const std::uint64_t guidePixels = pathReconstruction ? pixels : 0u;
    pt::WavefrontLimits limits;
    limits.maxStorageBufferRange = context.properties.limits.maxStorageBufferRange;
    limits.maxWorkGroupCountX = context.properties.limits.maxComputeWorkGroupCount[0];
    limits.maxRayDispatchInvocations =
        pipelineBackend ? context.rayPipelineProperties.maxRayDispatchInvocationCount : 0u;
    const pt::WavefrontPlan plan = pt::planWavefront(pixels, planSamples,
        static_cast<std::uint64_t>(std::max(0, settings.pathWaveCapacity)), guidePixels, limits);
    if (!plan.error.empty()) throw std::runtime_error("wavefront queues: " + plan.error);
    if (pathWaveCapacity != plan.capacity || pathWaveGuidePixels != std::max<std::uint64_t>(guidePixels, 1u)) {
      // Submitted frames' descriptor sets name the old queues: drain before replacing them.
      if (pathWaveCapacity) context.waitIdle();
      const VkDeviceSize paths = plan.capacity;
      const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      pathWaveStatesA = Buffer(context, paths * 48, storage, VMA_MEMORY_USAGE_AUTO, 0, "path.wave.states.a");
      pathWaveStatesB = Buffer(context, paths * 48, storage, VMA_MEMORY_USAGE_AUTO, 0, "path.wave.states.b");
      pathWaveHits = Buffer(context, paths * 24, storage, VMA_MEMORY_USAGE_AUTO, 0, "path.wave.hits");
      pathWaveResults = Buffer(context, paths * 48, storage, VMA_MEMORY_USAGE_AUTO, 0, "path.wave.results");
      pathWaveShadows = Buffer(context, paths * 48, storage, VMA_MEMORY_USAGE_AUTO, 0, "path.wave.shadows");
      pathWaveCosts = Buffer(context, paths * 8, storage, VMA_MEMORY_USAGE_AUTO, 0, "path.wave.costs");
      pathWaveGuidePixels = std::max<std::uint64_t>(guidePixels, 1u);
      pathWaveGuides = Buffer(context, pathWaveGuidePixels * 48, storage, VMA_MEMORY_USAGE_AUTO, 0, "path.wave.guides");
      // The counters are also the indirect dispatch and trace arguments.
      VkBufferUsageFlags counterUsage = storage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
      if (context.accelerationStructureSupported) counterUsage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
      pathWaveCounters = Buffer(context, 32 * sizeof(std::uint32_t), counterUsage, VMA_MEMORY_USAGE_AUTO, 0,
                                "path.wave.counters");
      pathWaveControl = Buffer(context, 8 * sizeof(std::uint32_t), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VMA_MEMORY_USAGE_AUTO, 0, "path.wave.control");
      pathWaveCapacity = plan.capacity;
      pathSetsGeneration = ~0u;
      stats.wavefrontQueueBytes = plan.bytes;
      logInfo("wavefront queues: {} paths per batch ({} per frame, limited by {}), {:.1f} MiB", plan.capacity,
              plan.batches, plan.limitedBy, static_cast<double>(plan.bytes) / (1024.0 * 1024.0));
    }
  }
  const bool restir = restirActive();
  if (restir && settings.pathSamplesPerFrame > 1)
    throw std::runtime_error("ReSTIR DI reuses one path per pixel per frame; samples per frame must be 1");
  ensureRestirResources(restir, settings.pathTemporal, settings.pathBvhDiagnostic != 0);
  const Buffer &instanceRows = software ? softwareTraceInstanceBuffer : traceInstanceBuffer;
  const bool fused = wavefront && waveFusionActive();
  Program *fusedProgram = !software ? pathWaveFusedRtProgram.get()
                          : wideSoftware ? pathWaveFusedWideProgram.get() : pathWaveFusedProgram.get();
  Program *shadeProgram = fused ? fusedProgram :
                          waveSubgroupAllocationActive() ? pathWaveShadeProgram.get() : pathWaveShadeAtomicProgram.get();
  if (pathSetsGeneration != resourceGeneration || pathSetsEnvironment != environmentState->version() ||
      pathSetsBackend != backendKey) {
    for (std::size_t f = 0; f < frames.size(); ++f) {
      if (restir) {
        Program *intersectProgram = pipelineBackend ? nullptr : software ? (wideSoftware ? pathWaveIntersectWideProgram.get() : pathWaveIntersectProgram.get()) : pathWaveIntersectRtProgram.get();
        Program *waveShadowProgram = pipelineBackend ? nullptr : software ? (wideSoftware ? pathWaveShadowWideProgram.get() : pathWaveShadowProgram.get()) : pathWaveShadowRtProgram.get();
        if (pipelineBackend ? (!pathWaveIntersectRayPipeline || !pathWaveShadowRayPipeline)
                            : (!intersectProgram || !waveShadowProgram))
          throw std::runtime_error("ReSTIR DI needs the backend's wavefront intersector");
        frames[f].restirIntersectSet = pool->allocate(pipelineBackend ? pathWaveIntersectRayPipeline->setLayout : intersectProgram->setLayouts[0]);
        DescriptorWriter iw(context, pipelineBackend ? pathWaveIntersectRayPipeline->shader(0) : intersectProgram->compute(), frames[f].restirIntersectSet);
        iw.buffer("uniforms", frames[f].pathUniforms).buffer("traceInstances", instanceRows)
          .buffer("materials", materialBuffer).buffer("indices", activeScene->indexBuffer)
          .buffer("vertices", activeScene->vertexBuffer).buffer("statesA", restirShadows)
          .buffer("statesB", restirShadows).buffer("counters", restirCounters)
          .buffer("waveControl", restirControl).buffer("costs", restirCosts).buffer("hits", restirShadows);
        if (software) iw.buffer("bvhNodes", softwareBvhNodes).buffer("bvhTriangles", softwareBvhTriangles);
        else iw.accelerationStructure("scene", acceleration->topLevel);
        iw.textureArray("maps", hitTextureViews, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
          .sampler("materialSampler", defaultSampler).apply();
        frames[f].restirShadowSet = pool->allocate(pipelineBackend ? pathWaveShadowRayPipeline->setLayout : waveShadowProgram->setLayouts[0]);
        DescriptorWriter sw(context, pipelineBackend ? pathWaveShadowRayPipeline->shader(0) : waveShadowProgram->compute(), frames[f].restirShadowSet);
        sw.buffer("uniforms", frames[f].pathUniforms).buffer("traceInstances", instanceRows)
          .buffer("materials", materialBuffer).buffer("indices", activeScene->indexBuffer)
          .buffer("vertices", activeScene->vertexBuffer).buffer("counters", restirCounters)
          .buffer("waveControl", restirControl).buffer("results", restirResults).buffer("shadows", restirShadows)
          .buffer("costs", restirCosts).buffer("waveGuides", restirGuides);
        if (software) sw.buffer("bvhNodes", softwareBvhNodes).buffer("bvhTriangles", softwareBvhTriangles);
        else sw.accelerationStructure("scene", acceleration->topLevel);
        sw.textureArray("maps", hitTextureViews, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
          .sampler("materialSampler", defaultSampler).apply();
        for (std::uint32_t parity = 0; parity < 2; ++parity) {
          frames[f].restirInitialSet[parity] = pool->allocate(pathRestirInitialProgram->setLayouts[0]);
          DescriptorWriter(context, pathRestirInitialProgram->compute(), frames[f].restirInitialSet[parity])
            .buffer("uniforms", frames[f].pathUniforms).buffer("traceInstances", instanceRows)
            .buffer("materials", materialBuffer).buffer("indices", activeScene->indexBuffer)
            .buffer("vertices", activeScene->vertexBuffer).buffer("lights", lightBuffer)
            .buffer("environmentDistribution", pathDistribution).buffer("specularAlbedo", pathAlbedoTable)
            .buffer("emissiveTriangles", pathEmissiveBuffer).buffer("hits", restirShadows)
            .buffer("restirCamera", restirCamera).buffer("previousSurfaces", restirSurfaces[1 - parity])
            .buffer("previousReservoirs", restirFinals[1 - parity]).buffer("surfaces", restirSurfaces[parity])
            .buffer("reservoirs", restirReservoirs)
            .texture("environmentMap", environmentState->traceImage())
            .textureArray("maps", hitTextureViews, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            .sampler("environmentSampler", environmentState->linearSampler()).sampler("materialSampler", defaultSampler)
            .apply();
          frames[f].restirSpatialSet[parity] = pool->allocate(pathRestirSpatialProgram->setLayouts[0]);
          DescriptorWriter(context, pathRestirSpatialProgram->compute(), frames[f].restirSpatialSet[parity])
            .buffer("uniforms", frames[f].pathUniforms).buffer("traceInstances", instanceRows)
            .buffer("materials", materialBuffer).buffer("indices", activeScene->indexBuffer)
            .buffer("vertices", activeScene->vertexBuffer).buffer("lights", lightBuffer)
            .buffer("environmentDistribution", pathDistribution).buffer("specularAlbedo", pathAlbedoTable)
            .buffer("emissiveTriangles", pathEmissiveBuffer).buffer("results", restirResults)
            .buffer("shadows", restirShadows).buffer("waveGuides", restirGuides)
            .buffer("surfaces", restirSurfaces[parity]).buffer("reservoirs", restirReservoirs)
            .buffer("finals", restirFinals[parity])
            .texture("environmentMap", environmentState->traceImage())
            .textureArray("maps", hitTextureViews, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            .sampler("environmentSampler", environmentState->linearSampler()).sampler("materialSampler", defaultSampler)
            .apply();
        }
      }
      if (wavefront) {
        Program *intersectProgram = pipelineBackend ? nullptr : software ? (wideSoftware ? pathWaveIntersectWideProgram.get() : pathWaveIntersectProgram.get()) : pathWaveIntersectRtProgram.get();
        Program *waveShadowProgram = pipelineBackend ? nullptr : software ? (wideSoftware ? pathWaveShadowWideProgram.get() : pathWaveShadowProgram.get()) : pathWaveShadowRtProgram.get();
        if (pipelineBackend ? (!pathWaveIntersectRayPipeline || !pathWaveShadowRayPipeline)
                            : (!intersectProgram || !waveShadowProgram))
          throw std::runtime_error("wavefront intersector is unavailable");
        frames[f].pathWaveIntersectSet=pool->allocate(pipelineBackend ? pathWaveIntersectRayPipeline->setLayout : intersectProgram->setLayouts[0]);
        DescriptorWriter iw(context,pipelineBackend ? pathWaveIntersectRayPipeline->shader(0) : intersectProgram->compute(),frames[f].pathWaveIntersectSet);
        iw.buffer("uniforms",frames[f].pathUniforms).buffer("traceInstances",instanceRows)
          .buffer("materials",materialBuffer).buffer("indices",activeScene->indexBuffer)
          .buffer("vertices",activeScene->vertexBuffer).buffer("statesA",pathWaveStatesA)
          .buffer("statesB",pathWaveStatesB).buffer("counters",pathWaveCounters)
          .buffer("waveControl",pathWaveControl).buffer("costs",pathWaveCosts).buffer("hits",pathWaveHits);
        if(software)iw.buffer("bvhNodes",softwareBvhNodes).buffer("bvhTriangles",softwareBvhTriangles);
        else iw.accelerationStructure("scene",acceleration->topLevel);
        iw.textureArray("maps",hitTextureViews,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
          .sampler("materialSampler",defaultSampler).apply();
        frames[f].pathWaveShadeSet=pool->allocate(shadeProgram->setLayouts[0]);
        DescriptorWriter(context,shadeProgram->compute(),frames[f].pathWaveShadeSet)
          .buffer("uniforms",frames[f].pathUniforms).buffer("traceInstances",instanceRows)
          .buffer("materials",materialBuffer).buffer("indices",activeScene->indexBuffer)
          .buffer("vertices",activeScene->vertexBuffer).buffer("lights",lightBuffer)
          .buffer("environmentDistribution",pathDistribution).buffer("specularAlbedo",pathAlbedoTable)
          .buffer("guides",pathReconstruction?pathReconstruction->samples():pathReconstructionSamples)
          .buffer("emissiveTriangles",pathEmissiveBuffer).buffer("statesA",pathWaveStatesA)
          .buffer("statesB",pathWaveStatesB).buffer("counters",pathWaveCounters)
          .buffer("waveControl",pathWaveControl).buffer("results",pathWaveResults)
          .buffer("shadows",pathWaveShadows).buffer("hits",pathWaveHits).buffer("waveGuides",pathWaveGuides)
          .buffer("restirResults",restirResults).buffer("restirGuides",restirGuides)
          .apply();
        if (fused) {
          DescriptorWriter fw(context,shadeProgram->compute(),frames[f].pathWaveShadeSet);
          if(software)fw.buffer("bvhNodes",softwareBvhNodes).buffer("bvhTriangles",softwareBvhTriangles);
          else fw.accelerationStructure("scene",acceleration->topLevel);
          fw.buffer("costs",pathWaveCosts).apply();
        }
        DescriptorWriter(context,shadeProgram->compute(),frames[f].pathWaveShadeSet)
          .texture("environmentMap",environmentState->traceImage())
          .textureArray("maps",hitTextureViews,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
          .sampler("environmentSampler",environmentState->linearSampler()).sampler("materialSampler",defaultSampler).apply();
        frames[f].pathWaveShadowSet=pool->allocate(pipelineBackend ? pathWaveShadowRayPipeline->setLayout : waveShadowProgram->setLayouts[0]);
        DescriptorWriter sw(context,pipelineBackend ? pathWaveShadowRayPipeline->shader(0) : waveShadowProgram->compute(),frames[f].pathWaveShadowSet);
        sw.buffer("uniforms",frames[f].pathUniforms).buffer("traceInstances",instanceRows)
          .buffer("materials",materialBuffer).buffer("indices",activeScene->indexBuffer)
          .buffer("vertices",activeScene->vertexBuffer).buffer("counters",pathWaveCounters)
          .buffer("waveControl",pathWaveControl).buffer("results",pathWaveResults).buffer("shadows",pathWaveShadows)
          .buffer("costs",pathWaveCosts).buffer("waveGuides",pathWaveGuides);
        if(software)sw.buffer("bvhNodes",softwareBvhNodes).buffer("bvhTriangles",softwareBvhTriangles);
        else sw.accelerationStructure("scene",acceleration->topLevel);
        sw.textureArray("maps",hitTextureViews,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
          .sampler("materialSampler",defaultSampler).apply();
        frames[f].pathWaveResolveSet=pool->allocate(pathWaveResolveProgram->setLayouts[0]);
        DescriptorWriter(context,pathWaveResolveProgram->compute(),frames[f].pathWaveResolveSet)
          .buffer("uniforms",frames[f].pathUniforms).buffer("results",pathWaveResults)
          .buffer("guides",pathReconstruction?pathReconstruction->samples():pathReconstructionSamples)
          .buffer("waveControl",pathWaveControl).buffer("costs",pathWaveCosts).buffer("waveGuides",pathWaveGuides)
          .storageTexture("accumulation",pathAccumulation.view).storageTexture("output",litColor[f].view)
          .storageTexture("albedoAccumulation",pathAlbedoAccumulation.view)
          .storageTexture("normalAccumulation",pathNormalAccumulation.view).apply();
        continue;
      }
      const Shader &pathShader = pipelineBackend ? pathRayPipeline->shader(0) : program->compute();
      VkDescriptorSet &pathSet = pipelineBackend ? frames[f].pathPipelineSet : frames[f].pathSet;
      pathSet = pool->allocate(pipelineBackend ? pathRayPipeline->setLayout : program->setLayouts[0]);
      DescriptorWriter writer(context, pathShader, pathSet);
      writer
          .buffer("uniforms", frames[f].pathUniforms)
          .buffer("traceInstances", instanceRows)
          .buffer("materials", materialBuffer)
          .buffer("indices", activeScene->indexBuffer)
          .buffer("vertices", activeScene->vertexBuffer)
          .buffer("lights", lightBuffer)
          .buffer("environmentDistribution", pathDistribution)
          .buffer("specularAlbedo", pathAlbedoTable)
          .buffer("emissiveTriangles", pathEmissiveBuffer)
          .buffer("reconstructionSamples", pathReconstruction ? pathReconstruction->samples() : pathReconstructionSamples);
      if (hybrid)
        writer.buffer("hybridInverseViewProjection", frames[f].hybridInverseViewProjection);
      else
        writer.buffer("restirResults", restirResults).buffer("restirGuides", restirGuides);
      if (software)
        writer.buffer("bvhNodes", softwareBvhNodes).buffer("bvhTriangles", softwareBvhTriangles);
      else
        writer.accelerationStructure("scene", acceleration->topLevel);
      writer
          .texture("environmentMap", environmentState->traceImage())
          .storageTexture("accumulation", pathAccumulation.view)
          .storageTexture("output", litColor[f].view)
          .textureArray("maps", hitTextureViews, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      if (!hybrid)
        writer.storageTexture("albedoAccumulation", pathAlbedoAccumulation.view)
              .storageTexture("normalAccumulation", pathNormalAccumulation.view);
      if (hybrid)
        writer.texture("hybridNormalRoughness", normalRoughness)
              .texture("hybridBaseMetallic", gbufferBaseMetallic)
              .texture("hybridGeometricDepth", gbufferGeometricCoverage)
              .texture("hybridIdentity", gbufferIdentity);
      writer
          .sampler("environmentSampler", environmentState->linearSampler())
          .sampler("materialSampler", defaultSampler)
          .apply();
      if (hybrid) {
        frames[f].pathGuideExportSet = pool->allocate(pathGuideExportProgram->setLayouts[0]);
        DescriptorWriter(context, pathGuideExportProgram->compute(), frames[f].pathGuideExportSet)
            .buffer("samples", pathReconstruction ? pathReconstruction->samples() : pathReconstructionSamples)
            .buffer("uniforms", frames[f].pathUniforms)
            .storageTexture("albedoAccumulation", pathAlbedoAccumulation.view)
            .storageTexture("normalAccumulation", pathNormalAccumulation.view)
            .apply();
      }
    }
    pathSetsGeneration = resourceGeneration;
    pathSetsEnvironment = environmentState->version();
    pathSetsBackend = backendKey;
  }

  // Samples: the accumulation key restarts them; the target stops them, and a frame past it
  // only rewrites the mean into its own lit image.
  if (accumulatedFrames == 0) gpuSamples = 0;
  const std::uint32_t perFrame = static_cast<std::uint32_t>(std::max(1, settings.pathSamplesPerFrame));
  const std::uint32_t target = static_cast<std::uint32_t>(std::max(0, settings.pathTargetSamples));
  std::uint32_t thisFrame = perFrame;
  if (target > 0) thisFrame = gpuSamples >= target ? 0u : std::min(perFrame, target - gpuSamples);
  pt::PathUniforms uniforms = makePathUniforms(camera);
  uniforms.counts.w = gpuSamples;
  uniforms.image.w = static_cast<float>(thisFrame);
  frame.pathUniforms.write(&uniforms, sizeof(uniforms));
  gpuSamples += thisFrame;
  freshSampleFrame[frameIndex] = thisFrame > 0;
  stats.samples = gpuSamples;

  // Stages this frame's tracing runs in. Pipeline wavefront keeps its init, shade and
  // resolve stages in compute, so it uses both.
  const VkPipelineStageFlags2 traceStages =
      pipelineBackend ? (wavefront ? VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                   : VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR)
                      : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  // The stage that writes the lit image last: the resolve in every wavefront mode.
  const VkPipelineStageFlags2 litWriter = pipelineBackend && !wavefront ? VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR
                                                                         : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  if (!profiler) profiler = std::make_unique<GpuProfiler>(context);
  profiler->begin(command, frameIndex);
  profiler->setFresh(thisFrame > 0);
  stats.traceGpuMilliseconds = static_cast<float>(profiler->latest().traceMilliseconds);
  const bool profiling = settings.pathProfile;
  profiler->mark(command, GpuStage::Setup);
  // The accumulators were written by the last frame's dispatch. The transfer stage is a
  // destination too: this frame's vkCmdUpdateBuffer of the wavefront and ReSTIR control
  // words must follow the last frame's reads of them.
  VkMemoryBarrier2 previous{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  previous.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                          context.rayTracingShaderStage() |
                          VK_PIPELINE_STAGE_2_COPY_BIT;
  previous.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                           VK_ACCESS_2_TRANSFER_READ_BIT;
  previous.dstStageMask = traceStages | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  previous.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                           VK_ACCESS_2_TRANSFER_WRITE_BIT;
  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.memoryBarrierCount = 1;
  dependency.pMemoryBarriers = &previous;
  vkCmdPipelineBarrier2(command, &dependency);

  Image &lit = litColor[frameIndex];
  transitionImage(command, lit, VK_IMAGE_LAYOUT_GENERAL,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                      context.rayTracingShaderStage(),
                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, traceStages, VK_ACCESS_2_SHADER_WRITE_BIT);
  if (restir && thisFrame > 0u) recordRestir(command, frame, software, wideSoftware, pipelineBackend, uniforms);
  if (wavefront) {
    recordWavefront(command, frame, thisFrame, software, wideSoftware, pipelineBackend, profiling);
  } else if (pipelineBackend) {
    pathRayPipeline->trace(command, frame.pathPipelineSet, width, height);
  } else {
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->handle);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, program->layout, 0, 1, &frame.pathSet, 0,
                            nullptr);
    vkCmdDispatch(command, (width + 7) / 8, (height + 7) / 8, 1);
  }
  if (hybrid) {
    VkMemoryBarrier2 guideBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    guideBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    guideBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    guideBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    guideBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    VkDependencyInfo guideDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    guideDependency.memoryBarrierCount = 1;
    guideDependency.pMemoryBarriers = &guideBarrier;
    vkCmdPipelineBarrier2(command, &guideDependency);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pathGuideExportPipeline.handle);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pathGuideExportProgram->layout,
                            0, 1, &frame.pathGuideExportSet, 0, nullptr);
    vkCmdDispatch(command, (width + 7) / 8, (height + 7) / 8, 1);
  }
  transitionImage(command, lit, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, litWriter,
                  VK_ACCESS_2_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  // Ends the trace interval: waits for the trace's last writes (the lit transition above).
  profiler->mark(command, wavefront && profiling ? GpuStage::Resolve : GpuStage::Trace);
  profiler->end(command);
  if (pathReconstruction) pathReconstruction->record(command, frameIndex, lit, uniforms,
      hashBytes(&resourceGeneration, sizeof(resourceGeneration), settingsKey()), gpuSamples, thisFrame > 0,
      sceneScale() * 0.5f);
}

bool Renderer::restirActive() const {
  return settings.pathDiEstimator == 1 && (settings.renderer == 2 || settings.renderer == 3 || settings.renderer == 5);
}

void Renderer::ensureRestirResources(bool active, bool guides, bool costs) {
  const std::uint64_t pixels = active ? static_cast<std::uint64_t>(width) * height : 1u;
  const std::uint64_t guidePixels = active && guides ? pixels : 1u, costPixels = active && costs ? pixels : 1u;
  if (restirPixels == pixels && restirGuidePixels == guidePixels && restirCostPixels == costPixels) return;
  // Submitted frames' descriptor sets name the old buffers: drain before replacing them.
  if (restirPixels) context.waitIdle();
  const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  auto perPixel = [&](VkDeviceSize bytes, const char *name) {
    return Buffer(context, pixels * bytes, storage, VMA_MEMORY_USAGE_AUTO, 0, name);
  };
  restirSurfaces[0] = perPixel(sizeof(pt::PtRestirSurface), "path.restir.surfaces.0");
  restirSurfaces[1] = perPixel(sizeof(pt::PtRestirSurface), "path.restir.surfaces.1");
  restirReservoirs = perPixel(sizeof(pt::PtReservoir), "path.restir.reservoirs");
  restirFinals[0] = perPixel(sizeof(pt::PtReservoir), "path.restir.finals.0");
  restirFinals[1] = perPixel(sizeof(pt::PtReservoir), "path.restir.finals.1");
  restirShadows = perPixel(48, "path.restir.shadows");  // also the primary hits (24 bytes each)
  restirResults = perPixel(48, "path.restir.results");
  restirGuides = Buffer(context, guidePixels * 48, storage, VMA_MEMORY_USAGE_AUTO, 0, "path.restir.guides");
  restirCosts = Buffer(context, costPixels * 8, storage, VMA_MEMORY_USAGE_AUTO, 0, "path.restir.costs");
  VkBufferUsageFlags counterUsage = storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  if (context.accelerationStructureSupported) counterUsage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  restirCounters = Buffer(context, 32 * sizeof(std::uint32_t), counterUsage, VMA_MEMORY_USAGE_AUTO, 0, "path.restir.counters");
  restirControl = Buffer(context, 8 * sizeof(std::uint32_t), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VMA_MEMORY_USAGE_AUTO, 0, "path.restir.control");
  restirCamera = Buffer(context, sizeof(pt::PtRestirCamera), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VMA_MEMORY_USAGE_AUTO, 0, "path.restir.camera");
  restirPixels = pixels;
  restirGuidePixels = guidePixels;
  restirCostPixels = costPixels;
  restirHistory = false;
  pathSetsGeneration = ~0u;
  if (active) {
    const double bytes = static_cast<double>(pixels) * 320.0 + static_cast<double>(guidePixels) * 48.0 +
                         static_cast<double>(costPixels) * 8.0;
    logInfo("ReSTIR DI buffers: {} pixels, {:.1f} MiB ({} bytes per pixel)", pixels, bytes / (1024.0 * 1024.0),
            320 + (guides ? 48 : 0) + (costs ? 8 : 0));
  }
}

// A ReSTIR DI frame: the backend's wavefront intersect stage at bounce 0 into
// per-pixel hits, initial resampling with temporal reuse, spatial reuse with the final
// reservoir's shadow record, and the backend's shadow stage adding the visible light to the
// per-pixel results the path trace reads. History follows the reconstruction's reset rules.
void Renderer::recordRestir(VkCommandBuffer command, FrameResources &frame, bool software, bool wideSoftware,
                            bool pipelineBackend, const pt::PathUniforms &uniforms) {
  const std::uint32_t pixels = width * height;
  const std::uint32_t groups = (pixels + pt::kWavefrontGroup - 1u) / pt::kWavefrontGroup;
  const std::uint32_t environmentVersion = environmentState->version();
  std::uint64_t key = hashBytes(&resourceGeneration, sizeof(resourceGeneration), settingsKey());
  key = hashBytes(&environmentVersion, sizeof(environmentVersion), key);
  bool valid = restirHistory && key == restirHistoryKey && (uniforms.estimator.z & 1u) != 0u;
  if (valid) {
    const pt::PathUniforms &p = restirPreviousUniforms;
    const float cut = sceneScale() * 0.5f;
    if (pt::dot(pt::xyz(uniforms.cameraForward), pt::xyz(p.cameraForward)) < 0.85f ||
        pt::length(pt::xyz(uniforms.cameraPosition) - pt::xyz(p.cameraPosition)) > cut ||
        std::abs(pt::length(pt::xyz(uniforms.cameraRight)) - pt::length(pt::xyz(p.cameraRight))) > 1e-6f ||
        std::abs(pt::length(pt::xyz(uniforms.cameraUp)) - pt::length(pt::xyz(p.cameraUp))) > 1e-6f ||
        uniforms.lens.x != p.lens.x || uniforms.lens.y != p.lens.y)
      valid = false;
  }
  pt::PtRestirCamera camera{};
  camera.position = restirPreviousUniforms.cameraPosition;
  camera.forward = restirPreviousUniforms.cameraForward;
  camera.right = restirPreviousUniforms.cameraRight;
  camera.up = restirPreviousUniforms.cameraUp;
  camera.image = pt::uint4(width, height, valid ? 1u : 0u, uniforms.estimator.z);
  // The shadow group (uint 16) holds this frame's shadow records: one per pixel.
  std::array<std::uint32_t, 32> counters{};
  counters[16] = groups; counters[17] = 1u; counters[18] = 1u;
  counters[19] = pixels; counters[20] = 1u; counters[21] = 1u;
  const std::array<std::uint32_t, 8> control{0u, pixels, 0u, pixels, 0u, 1u, 1u, 0u};
  vkCmdUpdateBuffer(command, restirCamera.handle, 0, sizeof(camera), &camera);
  vkCmdUpdateBuffer(command, restirCounters.handle, 0, sizeof(counters), counters.data());
  vkCmdUpdateBuffer(command, restirControl.handle, 0, sizeof(control), control.data());

  const VkPipelineStageFlags2 shaderStages =
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | (pipelineBackend ? context.rayTracingShaderStage() : VkPipelineStageFlags2{0});
  auto barrier = [&]() {
    VkMemoryBarrier2 b{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    b.srcStageMask = shaderStages | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b.dstStageMask = shaderStages;
    b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo d{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    d.memoryBarrierCount = 1;
    d.pMemoryBarriers = &b;
    vkCmdPipelineBarrier2(command, &d);
  };
  auto dispatch = [&](const Pipeline &pipeline, const Program &program, VkDescriptorSet set) {
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, program.layout, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(command, groups, 1, 1);
  };
  barrier();
  if (pipelineBackend) {
    pathWaveIntersectRayPipeline->trace(command, frame.restirIntersectSet, pixels);
  } else {
    const Program &program = software ? (wideSoftware ? *pathWaveIntersectWideProgram : *pathWaveIntersectProgram)
                                      : *pathWaveIntersectRtProgram;
    const Pipeline &pipeline = software ? (wideSoftware ? pathWaveIntersectWidePipeline : pathWaveIntersectPipeline)
                                        : pathWaveIntersectRtPipeline;
    dispatch(pipeline, program, frame.restirIntersectSet);
  }
  barrier();
  dispatch(pathRestirInitialPipeline, *pathRestirInitialProgram, frame.restirInitialSet[restirParity]);
  barrier();
  dispatch(pathRestirSpatialPipeline, *pathRestirSpatialProgram, frame.restirSpatialSet[restirParity]);
  barrier();
  if (pipelineBackend) {
    pathWaveShadowRayPipeline->trace(command, frame.restirShadowSet, pixels);
  } else {
    const Program &program = software ? (wideSoftware ? *pathWaveShadowWideProgram : *pathWaveShadowProgram)
                                      : *pathWaveShadowRtProgram;
    const Pipeline &pipeline = software ? (wideSoftware ? pathWaveShadowWidePipeline : pathWaveShadowPipeline)
                                        : pathWaveShadowRtPipeline;
    dispatch(pipeline, program, frame.restirShadowSet);
  }
  barrier();
  restirParity ^= 1u;
  restirHistory = true;
  restirHistoryKey = key;
  restirPreviousUniforms = uniforms;
}

// Wavefront execution: the frame's pixels x S paths in whole-pixel batches. Per batch: for
// every bounce an intersect, shade and shadow stage over GPU-resident queue counts, then one
// resolve that adds the batch's pixels to the accumulators. Every stage boundary is a full
// compute/ray/transfer/indirect barrier.
void Renderer::recordWavefront(VkCommandBuffer command, FrameResources &frame, std::uint32_t thisFrame,
                               bool software, bool wideSoftware, bool pipelineBackend, bool profiling) {
  Program *intersectProgram = pipelineBackend ? nullptr : software ? (wideSoftware ? pathWaveIntersectWideProgram.get() : pathWaveIntersectProgram.get()) : pathWaveIntersectRtProgram.get();
  Program *shadowProgram = pipelineBackend ? nullptr : software ? (wideSoftware ? pathWaveShadowWideProgram.get() : pathWaveShadowProgram.get()) : pathWaveShadowRtProgram.get();
  Pipeline *intersectPipeline = pipelineBackend ? nullptr : software ? (wideSoftware ? &pathWaveIntersectWidePipeline : &pathWaveIntersectPipeline) : &pathWaveIntersectRtPipeline;
  Pipeline *shadowPipeline = pipelineBackend ? nullptr : software ? (wideSoftware ? &pathWaveShadowWidePipeline : &pathWaveShadowPipeline) : &pathWaveShadowRtPipeline;
  // Stages this backend's wavefront runs in: ray tracing only for pipeline launches.
  const VkPipelineStageFlags2 shaderStages =
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | (pipelineBackend ? context.rayTracingShaderStage() : VkPipelineStageFlags2{0});
  auto barrier = [&]() {
    VkMemoryBarrier2 b{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    b.srcStageMask = shaderStages | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b.dstStageMask = shaderStages | VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                      VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    VkDependencyInfo d{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    d.memoryBarrierCount = 1;
    d.pMemoryBarriers = &b;
    vkCmdPipelineBarrier2(command, &d);
  };
  const bool indirectTrace = pipelineBackend && context.rayPipelineIndirectSupported;
  const VkDeviceAddress counterAddress = indirectTrace ? pathWaveCounters.deviceAddress() : 0u;
  // Counter group g (0, 1: continuation parity; 2: shadow) at uint 8g: dispatch x, y, z, then
  // count, 1, 1 (the indirect trace dimensions), overflow, reserved.
  auto intersect = [&](std::uint32_t parity) {
    if (pipelineBackend) {
      if (indirectTrace) pathWaveIntersectRayPipeline->traceIndirect(command, frame.pathWaveIntersectSet, counterAddress + parity * 32u + 12u);
      else pathWaveIntersectRayPipeline->trace(command, frame.pathWaveIntersectSet, pathWaveCapacity);
    } else {
      vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, intersectPipeline->handle);
      vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, intersectProgram->layout, 0, 1, &frame.pathWaveIntersectSet, 0, nullptr);
      vkCmdDispatchIndirect(command, pathWaveCounters.handle, parity * 32u);
    }
  };
  const bool fused = waveFusionActive();
  const bool ownBvh = settings.renderer == 3, wideBvh = ownBvh && settings.pathBvhWidth != 0;
  const Pipeline &fusedPipeline = !ownBvh ? pathWaveFusedRtPipeline : wideBvh ? pathWaveFusedWidePipeline : pathWaveFusedPipeline;
  const Program *fusedProgram = !ownBvh ? pathWaveFusedRtProgram.get() : wideBvh ? pathWaveFusedWideProgram.get()
                                                                          : pathWaveFusedProgram.get();
  auto shade = [&](std::uint32_t parity) {
    const bool subgroup = waveSubgroupAllocationActive();
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                      fused ? fusedPipeline.handle :
                      subgroup ? pathWaveShadePipeline.handle : pathWaveShadeAtomicPipeline.handle);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            (fused ? fusedProgram : subgroup ? pathWaveShadeProgram.get() : pathWaveShadeAtomicProgram.get())->layout,
                            0, 1, &frame.pathWaveShadeSet, 0, nullptr);
    vkCmdDispatchIndirect(command, pathWaveCounters.handle, parity * 32u);
  };
  auto shadow = [&]() {
    if (pipelineBackend) {
      if (indirectTrace) pathWaveShadowRayPipeline->traceIndirect(command, frame.pathWaveShadowSet, counterAddress + 64u + 12u);
      else pathWaveShadowRayPipeline->trace(command, frame.pathWaveShadowSet, pathWaveCapacity);
    } else {
      vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, shadowPipeline->handle);
      vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, shadowProgram->layout, 0, 1, &frame.pathWaveShadowSet, 0, nullptr);
      vkCmdDispatchIndirect(command, pathWaveCounters.handle, 64u);
    }
  };
  // Groups of 128 threads; a tiled group covers 128 / S whole pixels (path_wavefront_resolve).
  auto resolve = [&](std::uint32_t pixels, std::uint32_t samples) {
    const std::uint32_t pixelsPerGroup = samples > 0u && samples <= 128u ? 128u / samples : 128u;
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pathWaveResolvePipeline.handle);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pathWaveResolveProgram->layout, 0, 1, &frame.pathWaveResolveSet, 0, nullptr);
    vkCmdDispatch(command, (pixels + pixelsPerGroup - 1u) / pixelsPerGroup, 1, 1);
  };
  const std::uint32_t pixels = width * height;
  if (thisFrame == 0u) {
    // Target reached: rewrite this frame's lit image from the accumulator.
    const std::array<std::uint32_t, 8> control{0u, pathWaveCapacity, 0u, 0u, 0u, 1u, 0u, 0u};
    vkCmdUpdateBuffer(command, pathWaveControl.handle, 0, sizeof(control), control.data());
    barrier();
    resolve(pixels, 0u);
    barrier();
    if (profiling) profiler->mark(command, GpuStage::Resolve);
    return;
  }
  const std::uint32_t batchPaths = pt::wavefrontBatchPaths(pathWaveCapacity, thisFrame);
  if (batchPaths == 0u) throw std::runtime_error("wavefront batch cannot hold one pixel's samples");
  const std::uint64_t totalPaths = static_cast<std::uint64_t>(pixels) * thisFrame;
  const std::uint32_t bounces = static_cast<std::uint32_t>(std::max(1, settings.pathBounces));
  // Queue reservations are bounded by the capacity; a test-only limit below it provokes the
  // overflow path, which must be reported, never silently absorbed.
  const std::uint32_t queueCapacity = settings.pathWaveQueueLimit > 0
      ? std::min(pathWaveCapacity, static_cast<std::uint32_t>(settings.pathWaveQueueLimit)) : pathWaveCapacity;
  for (std::uint64_t base = 0; base < totalPaths; base += batchPaths) {
    const std::uint32_t count = static_cast<std::uint32_t>(std::min<std::uint64_t>(batchPaths, totalPaths - base));
    const std::uint32_t firstPixel = static_cast<std::uint32_t>(base / thisFrame);
    const std::array<std::uint32_t, 32> counters{(count + pt::kWavefrontGroup - 1u) / pt::kWavefrontGroup, 1u, 1u, count, 1u, 1u, 0u, 0u,
                                                 0u, 1u, 1u, 0u, 1u, 1u, 0u, 0u,
                                                 0u, 1u, 1u, 0u, 1u, 1u, 0u, 0u,
                                                 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    std::array<std::uint32_t, 8> control{0u, queueCapacity, static_cast<std::uint32_t>(base), count,
                                         0u, thisFrame, 1u, firstPixel};
    // The frame's overflow total (uint 24) is cleared by the first batch only.
    vkCmdUpdateBuffer(command, pathWaveCounters.handle, 0, base == 0 ? 32u * 4u : 24u * 4u, counters.data());
    vkCmdUpdateBuffer(command, pathWaveControl.handle, 0, sizeof(control), control.data());
    barrier();
    if (profiling) profiler->mark(command, GpuStage::Setup);
    std::uint32_t parity = 0u;
    for (std::uint32_t bounce = 0; bounce <= bounces; ++bounce) {
      const std::uint32_t next = 1u - parity;
      if (bounce > 0u) {
        // The next continuation queue and the shadow queue start empty; the shadow queue
        // follows group 1, so parity 0's reset is one write and parity 1's two.
        const std::array<std::uint32_t, 8> empty{0u, 1u, 1u, 0u, 1u, 1u, 0u, 0u};
        vkCmdUpdateBuffer(command, pathWaveCounters.handle, next * 32u, sizeof(empty), empty.data());
        vkCmdUpdateBuffer(command, pathWaveCounters.handle, 64u, sizeof(empty), empty.data());
        control[0] = parity;
        control[4] = bounce;
        vkCmdUpdateBuffer(command, pathWaveControl.handle, 0, sizeof(control), control.data());
        barrier();
        if (profiling) profiler->mark(command, GpuStage::Setup);
      }
      if (!fused) {
        intersect(parity);
        barrier();
        if (profiling) profiler->mark(command, GpuStage::Intersect);
      }
      shade(parity);
      barrier();
      if (profiling) {
        profiler->mark(command, GpuStage::Shade);
        profiler->copyCounter(command, pathWaveCounters.handle, next * 32u + 12u, 0);
        profiler->copyCounter(command, pathWaveCounters.handle, 64u + 12u, 1);
      }
      shadow();
      barrier();
      if (profiling) profiler->mark(command, GpuStage::Shadow);
      parity = next;
    }
    resolve(count / thisFrame, thisFrame);
    barrier();
    if (profiling) profiler->mark(command, GpuStage::Resolve);
  }
  profiler->watchOverflow(command, pathWaveCounters.handle, 24u * 4u);
}

const GpuProfiler *Renderer::finishProfiling() {
  if (!profiler) return nullptr;
  profiler->finish();
  stats.traceGpuMilliseconds = static_cast<float>(profiler->latest().traceMilliseconds);
  return profiler.get();
}

void Renderer::recordCpuTrace(VkCommandBuffer command, const Camera &camera) {
  if (!cpuTracer) cpuTracer = std::make_unique<pt::CpuTracer>(static_cast<unsigned>(std::max(0, settings.cpuThreads)));
  if (!activeScene) return;
  const std::array<std::uint32_t, 3> sceneKey{sceneVersion, environmentState->version(),
                                               static_cast<std::uint32_t>(settings.cpuIntersector)};
  if (!cpuScene || sceneKey != cpuSceneKey) {
    buildCpuScene();
    cpuSceneKey = sceneKey;
    cpuTracer->setScene(cpuScene);
    cpuStartedKey = 0;
  }
  // Anything the accumulation key sees restarts the tracer: the view, settings, targets, resources.
  if (cpuStartedKey == 0 || accumulationKey != cpuStartedKey) {
    pt::CpuFrame frame;
    frame.uniforms = makePathUniforms(camera);
    frame.materials.resize(activeScene->materials.size());
    static_assert(sizeof(pt::Material) == sizeof(MaterialUniforms), "materials must share a layout");
    static_assert(sizeof(pt::Light) == sizeof(LightRecord), "lights must share a layout");
    for (std::size_t m = 0; m < activeScene->materials.size(); ++m)
      std::memcpy(&frame.materials[m], &activeScene->materials[m].uniforms, sizeof(pt::Material));
    frame.lights.resize(hostLights.size());
    if (!hostLights.empty()) std::memcpy(frame.lights.data(), hostLights.data(), hostLights.size() * sizeof(pt::Light));
    frame.width = width;
    frame.height = height;
    frame.targetSamples = static_cast<std::uint32_t>(std::max(0, settings.pathTargetSamples));
    frame.intersector = settings.cpuIntersector == 1 && cpuScene->embree ? 1u :
                        settings.cpuIntersector == 2 && cpuScene->wideAvx2 ? 2u : 0u;
    frame.denoise = settings.pathDenoise && !settings.pathTemporal;
    cpuTracer->start(std::move(frame));
    cpuImage = {}; // old display may remain visible, but it cannot supply current SPP/guides/capture
    cpuUploaded.fill(0);
    cpuStartedKey = accumulationKey;
  }

  pt::CpuTracer::Image image;
  const bool published = cpuTracer->latest(image, cpuImage.version, false);
  freshSampleFrame[frameIndex] = published;
  if (published) cpuImage = std::move(image);
  if (published && settings.pathTemporal && !cpuImage.guides.empty()) {
    const auto guideWidth = static_cast<std::uint32_t>(cpuImage.uniforms.image.x);
    const auto guideHeight = static_cast<std::uint32_t>(cpuImage.uniforms.image.y);
    if (!pathReconstruction || !pathReconstruction->fits(guideWidth, guideHeight)) {
      context.waitIdle();
      pathReconstruction.reset(); // release the old extent before allocating the new history
      pathReconstruction = std::make_unique<PathReconstruction>(context, guideWidth, guideHeight);
    }
  }
  stats.samples = cpuImage.samples;
  stats.previewScale = cpuImage.previewScale;
  stats.pathsPerSecond = cpuTracer->samplesPerSecond();

  // This frame's lit image gets the latest mean; each of the frames in flight has its own.
  Image &lit = litColor[frameIndex];
  const bool fits = cpuImage.version != 0 && cpuImage.width == width && cpuImage.height == height;
  if (fits && (cpuUploaded[frameIndex] != cpuImage.version || pathReconstruction)) {
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(width) * height * 4 * sizeof(std::uint16_t);
    Buffer &staging = cpuStaging[frameIndex];
    if (!staging || staging.size < bytes)
      staging = Buffer(context, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO,
                       VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                       "cpu.upload");
    staging.write(cpuImage.half.data(), static_cast<std::size_t>(bytes));
    transitionImage(command, lit, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(command, staging.handle, lit.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    transitionImage(command, lit, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    cpuUploaded[frameIndex] = cpuImage.version;
  } else if (lit.layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    // Nothing traced yet into an image nothing has written: black, in the layout bloom reads.
    transitionImage(command, lit, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    const VkClearColorValue black{{0.0f, 0.0f, 0.0f, 1.0f}};
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(command, lit.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
    transitionImage(command, lit, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_CLEAR_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  }
  if (pathReconstruction && fits && !cpuImage.guides.empty()) {
    if (published) pathReconstruction->upload(command, frameIndex, cpuImage.guides);
    pathReconstruction->record(command, frameIndex, lit, cpuImage.uniforms,
        hashBytes(&resourceGeneration, sizeof(resourceGeneration), settingsKey()), cpuImage.samples, published,
        sceneScale() * 0.5f);
  }
}

} // namespace basalt
