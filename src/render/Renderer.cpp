#include "render/Renderer.h"

#include "core/Log.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <map>

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

// The shadow entry declares only position and uv0, at the same offsets.
std::vector<VertexAttributeLayout> vertexLayout(bool shadowPass) {
  if (shadowPass)
    return {{0, 0, offsetof(Vertex, position)}, {1, 0, offsetof(Vertex, uv0)}};
  return {{0, 0, offsetof(Vertex, position)},
          {1, 0, offsetof(Vertex, normal)},
          {2, 0, offsetof(Vertex, tangent)},
          {3, 0, offsetof(Vertex, uv0)},
          {4, 0, offsetof(Vertex, uv1)}};
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
  const std::vector<VkFormat> forwardTargets{kHdrFormat, kHdrFormat, kHdrFormat};

  GraphicsPipelineDescription forward;
  forward.program = forwardProgram.get();
  forward.colorFormats = forwardTargets;
  forward.depthFormat = depthFormat;
  forward.bindings = geometryBinding;
  forward.attributes = vertexLayout(false);
  forward.cullMode = VK_CULL_MODE_BACK_BIT;
  // Counter-clockwise front faces, as glTF specifies; checked against the debug normal view.
  forward.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
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
  shadow.depthFormat = depthFormat;
  shadow.bindings = geometryBinding;
  shadow.attributes = vertexLayout(true);
  // Front faces culled so the map stores the caster's far side, moving self-shadowing error to
  // the unlit side. Ordinary viewport here, so front faces stay counter-clockwise.
  shadow.cullMode = VK_CULL_MODE_FRONT_BIT;
  shadow.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
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
  litDescription.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
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
  buildHitTextureTable(*next, nextViews, nextSlots);
  if (context.rayTracingSupported)
    nextAcceleration = std::make_unique<SceneAccelerationStructure>(context, uploader, *next, nextSlots);

  acceleration.reset();
  if (activeScene) activeScene->destroy(context);
  activeScene = std::move(next);
  // Nothing to reproject a history through after a reframe.
  hasPreviousView = false;
  hitTextureViews = std::move(nextViews);
  materialSlots = std::move(nextSlots);
  acceleration = std::move(nextAcceleration);

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
  if (lights.empty()) lights.push_back({}); // A storage buffer may not be empty.
  context.waitIdle();
  lightBuffer = uploader.createBuffer(lights.data(), lights.size() * sizeof(LightRecord),
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "scene.lights");
  builtTestLights = requested;
}

void Renderer::buildHitTextureTable(const Scene &scene, std::vector<VkImageView> &views,
                                    std::vector<std::uint32_t> &slots) const {
  // Slot zero is white for missing maps; a scene with more textures than slots shades the rest white and says so.
  views.assign(kHitTextureSlots, whiteTexture.view);
  // Slot one is the flat normal; white would tilt every hit.
  views[1] = normalTexture.view;
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
    slotOf.emplace(texture, nextSlot);
    return nextSlot++;
  };
  for (std::size_t m = 0; m < scene.materials.size(); ++m) {
    const Material &material = scene.materials[m];
    std::uint32_t normalSlot = material.normal < 0 ? 1 : slotFor(material.normal);
    if (normalSlot == 0) normalSlot = 1;
    slots[m] = slotFor(material.baseColor) | (slotFor(material.metallicRoughness) << 8) |
               (slotFor(material.emissive) << 16) | (normalSlot << 24);
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
            .buffer("primitives", acceleration->primitiveInfo)
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
          .buffer("primitives", acceleration->primitiveInfo)
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
  add(s.clusteredLights);
  return key;
}

void Renderer::updateFrameData(const Camera &camera, float deltaSeconds) {
  elapsedSeconds += deltaSeconds;
  ++frameCounter;
  FrameResources &frame = frames[frameIndex];

  const float aspect = static_cast<float>(width) / static_cast<float>(std::max(1u, height));
  const Mat4 view = camera.viewMatrix();
  Mat4 projection = camera.projectionMatrix(aspect);
  const Mat4 steadyViewProjection = projection * view;

  // The jitter is a clip-space shift proportional to w. Motion vectors and the accumulation
  // key use the un-jittered transform, so a still view stays still.
  const bool temporal = settings.antialiasing == 2 && settings.debugView == 0;
  if (temporal) {
    const Vec2 jitter = haltonJitter(frameCounter);
    projection.columns[2].x = -jitter.x * 2.0f / static_cast<float>(width);
    projection.columns[2].y = -jitter.y * 2.0f / static_cast<float>(height);
  }
  const Mat4 viewProjection = projection * view;
  const Mat4 inverseViewProjection = inverse(viewProjection);
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
  const bool accumulating = settings.accumulate && settings.debugView == 0;
  if (accumulating && key == accumulationKey) accumulatedFrames = std::min(accumulatedFrames + 1, 4096u);
  else accumulatedFrames = 0;
  accumulationKey = key;
  stats.accumulatedFrames = accumulatedFrames;
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

  for (Image *target : {&hdrColor, &normalRoughness, &reflectionWeight})
    transitionImage(command, *target, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
  transitionImage(command, depthBuffer, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

  std::array<VkRenderingAttachmentInfo, 3> colorAttachments{};
  const Image *targets[3]{&hdrColor, &normalRoughness, &reflectionWeight};
  for (int i = 0; i < 3; ++i) {
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
    }
    vkCmdResetQueryPool(command, timestampPool, first, 2);
    vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, timestampPool, first);
  }

  updateFrameData(camera, deltaSeconds);
  recordLightCulling(command);
  recordShadowPass(command);
  recordForwardPass(command);
  recordReflections(command);
  recordBloom(command);
  recordPost(command, imageIndex);

  if (timestampPool)
    vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, timestampPool,
                         frameIndex * 2 + 1);
}

} // namespace basalt
