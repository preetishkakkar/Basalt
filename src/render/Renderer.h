// The frame: shadows, forward PBR into an HDR target and a small G-buffer, sky, reflection
// resolve, temporal pass, bloom, post.
#pragma once
#include "gpu/Descriptors.h"
#include "gpu/Pipeline.h"
#include "gpu/Swapchain.h"
#include "gpu/Uploader.h"
#include "render/Environment.h"
#include "render/RayTracing.h"
#include "scene/Camera.h"
#include "scene/Scene.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace basalt {

struct RenderSettings {
  // Sun.
  float sunAzimuth = radians(40.0f);
  float sunElevation = radians(35.0f);
  Vec3 sunColor{1.0f, 0.96f, 0.90f};
  float sunIntensity = 4.0f;
  float sunAngularRadius = radians(0.6f); // Soft traced shadows widen with it.

  // Environment.
  float iblIntensity = 1.0f;
  float skyTurbidity = 3.0f;
  float skyIntensity = 1.0f;
  bool drawSky = true;
  bool groundPlane = true;
  // Point lights scattered through the scene, since almost no glTF file carries any.
  int testLights = 0;
  // Traced shadows for punctual lights; the rasterised path lights through walls.
  bool lightShadows = true;
  bool clusteredLights = true;
  float groundRoughness = 0.8f;
  float groundMetallic = 0.0f;
  Vec3 groundColor{0.34f, 0.34f, 0.36f};

  int shadowMode = 0; // 0 cascades, 1 traced
  int shadowSamples = 4;
  bool shadowsEnabled = true;
  float shadowDepthBias = 0.0015f;
  float shadowNormalBias = 0.02f;
  float shadowSoftness = 1.0f;
  float cascadeSplitLambda = 0.85f;
  float shadowDistance = 0.0f; // Zero means the whole scene.
  bool freezeCascades = false;

  int occlusionMode = 0; // 0 texture, 1 traced
  int occlusionSamples = 4;
  float occlusionRadius = 0.0f; // Zero means a fraction of the scene's radius.

  int reflectionMode = 1; // 0 environment, 1 screen space, 2 traced
  float reflectionDistance = 0.0f; // Zero means twice the scene's radius.
  float reflectionThickness = 0.0f; // Zero means a fraction of the scene's radius.
  int reflectionSteps = 48;

  // Average frames while nothing changes, so stochastic terms converge.
  bool accumulate = true;

  // Post.
  float exposure = 1.0f;
  int tonemap = 0; // 0 ACES, 1 Reinhard
  float whitePoint = 4.0f;
  float bloomStrength = 0.04f;
  float bloomThreshold = 1.2f;
  float bloomKnee = 0.6f;
  float bloomRadius = 1.0f;
  float vignette = 0.35f;
  float grain = 0.0f;
  float sharpen = 0.0f;
  // Temporal jitters the projection and blends with the last frame.
  int antialiasing = 2;
  float temporalFeedback = 0.1f; // How much of a moving frame is the new one.

  // Debug.
  int debugView = 0;
  bool wireframe = false;
  bool frustumCulling = true;
  bool vsync = true;
};

struct FrameStatistics {
  std::uint32_t drawCalls = 0;
  std::uint32_t shadowDrawCalls = 0;
  std::uint32_t visiblePrimitives = 0;
  std::uint32_t triangles = 0;
  std::uint32_t accumulatedFrames = 0;
  float gpuMilliseconds = 0.0f;
};

class Renderer {
public:
  Renderer(Context &context, Swapchain &swapchain, Uploader &uploader);
  ~Renderer();
  Renderer(const Renderer &) = delete;
  Renderer &operator=(const Renderer &) = delete;

  void setScene(std::unique_ptr<Scene> scene);
  Scene *scene() { return activeScene.get(); }
  Environment &environment() { return *environmentState; }
  bool rayTracingAvailable() const { return context.rayTracingSupported && acceleration != nullptr; }

  void resize(std::uint32_t width, std::uint32_t height);

  // Leaves the render pass open for the interface.
  void render(VkCommandBuffer command, std::uint32_t imageIndex, const Camera &camera,
              float deltaSeconds);

  RenderSettings settings;
  const FrameStatistics &statistics() const { return stats; }
  Vec3 sunDirection() const;
  void rebakeProceduralSky();
  // Every material set holds the environment cubes and must be rewritten.
  void environmentChanged() { rebuildSceneResources(); }
  // Pushes the ground settings into the plane's material.
  void applyGroundMaterial();
  VkFormat swapchainFormat() const;

private:
  void createTargets(std::uint32_t width, std::uint32_t height);
  void createPipelines();
  void createStaticResources();
  void rebuildSceneResources();
  void updateFrameData(const Camera &camera, float deltaSeconds);
  void computeCascades(const Camera &camera);
  void recordShadowPass(VkCommandBuffer command);
  void recordForwardPass(VkCommandBuffer command);
  void recordLightCulling(VkCommandBuffer command);
  void recordReflections(VkCommandBuffer command);
  void recordBloom(VkCommandBuffer command);
  void recordPost(VkCommandBuffer command, std::uint32_t imageIndex);
  float sceneScale() const;
  std::uint64_t settingsKey() const;
  void buildLights();
  void buildHitTextureTable(const Scene &scene, std::vector<VkImageView> &views,
                            std::vector<std::uint32_t> &slots) const;

  Context &context;
  Swapchain &swapchain;
  Uploader &uploader;

  std::unique_ptr<Scene> activeScene;
  std::unique_ptr<Environment> environmentState;
  std::unique_ptr<SceneAccelerationStructure> acceleration;
  // Hit texture table, in the slots the primitive table packs.
  std::vector<VkImageView> hitTextureViews;
  std::vector<std::uint32_t> materialSlots;

  // Targets.
  std::uint32_t width = 0, height = 0;
  Image hdrColor, normalRoughness, reflectionWeight, reflection;
  // The composite writes litCurrent; the temporal pass resolves it against the other
  // litColor into this frame's, which bloom and post read.
  Image litCurrent;
  Image litColor[kFramesInFlight];
  Image depthBuffer, shadowMap, bloomChain;
  std::vector<VkImageView> bloomMipViews;
  std::uint32_t bloomMipCount = 0;
  std::uint32_t reflectionMipCount = 0;
  VkImageView reflectionMipZero = VK_NULL_HANDLE;
  VkImageView shadowLayerViews[4]{};
  VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

  // Programs and pipelines.
  std::unique_ptr<Program> forwardProgram, shadowProgram, skyProgram, postProgram;
  std::unique_ptr<Program> bloomDownProgram, bloomUpProgram, resolveProgram, compositeProgram;
  std::unique_ptr<Program> temporalProgram, clusterProgram;
  Pipeline forwardPipeline, forwardBlendPipeline, forwardWirePipeline;
  Pipeline shadowPipeline, shadowMaskedPipeline;
  // The same without back-face culling, for materials glTF marks double-sided.
  Pipeline forwardTwoSidedPipeline, shadowTwoSidedPipeline, shadowMaskedTwoSidedPipeline;
  Pipeline skyPipeline, postPipeline, bloomDownPipeline, bloomUpPipeline;
  Pipeline resolvePipeline, compositePipeline, temporalPipeline, clusterPipeline;

  // Per-frame data, one copy per frame in flight.
  struct FrameResources {
    Buffer frameUniforms;
    Buffer postUniforms;
    Buffer reflectionUniforms;
    Buffer temporalUniforms;
    Buffer clusterUniforms;
    // Cascade rows packed for the forward pass and strided for the shadow pass, which binds one at a time.
    Buffer cascadePacked;
    Buffer cascadeStrided;
    VkDescriptorSet forwardVertexSet = VK_NULL_HANDLE;
    VkDescriptorSet skySet = VK_NULL_HANDLE;
    VkDescriptorSet skyVertexSet = VK_NULL_HANDLE;
    VkDescriptorSet postSet = VK_NULL_HANDLE;
    VkDescriptorSet postVertexSet = VK_NULL_HANDLE;
    VkDescriptorSet resolveSet = VK_NULL_HANDLE;
    VkDescriptorSet compositeSet = VK_NULL_HANDLE;
    VkDescriptorSet temporalSet = VK_NULL_HANDLE;
    VkDescriptorSet clusterSet = VK_NULL_HANDLE;
    VkDescriptorSet bloomFirstSet = VK_NULL_HANDLE; // Bloom's first level reads this frame's lit image.
    std::array<VkDescriptorSet, 4> shadowVertexSets{};
    std::vector<VkDescriptorSet> forwardMaterialSets;
  };
  std::array<FrameResources, kFramesInFlight> frames;
  std::vector<VkDescriptorSet> shadowMaterialSets;
  std::vector<VkDescriptorSet> bloomDownSets, bloomUpSets;
  // The bloom parameters the current sets were built with: threshold, knee, radius.
  std::array<float, 3> builtBloom{-1.0f, -1.0f, -1.0f};
  std::vector<Buffer> bloomUniformBuffers;

  // Static for the scene's life; instance records are indexed by the draw's first instance.
  Buffer materialBuffer, lightBuffer, instanceBuffer;
  // Counts then fixed runs, in one buffer: a fragment stage is only promised four storage buffers.
  Buffer lightClusters;
  std::array<std::uint32_t, 3> clusterGrid{1, 1, 1};
  std::uint32_t clusterCount = 1;
  std::uint32_t lightCount = 0;
  int builtTestLights = -1;
  VkDeviceSize cascadeStride = 0;

  // Neutral textures bound where a material has none.
  Image whiteTexture, normalTexture, blackTexture;
  VkSampler defaultSampler = VK_NULL_HANDLE;
  VkSampler clampSampler = VK_NULL_HANDLE;
  VkSampler pointSampler = VK_NULL_HANDLE;
  VkSampler shadowSampler = VK_NULL_HANDLE;

  std::unique_ptr<DescriptorPool> pool;
  VkQueryPool timestampPool = VK_NULL_HANDLE;

  std::vector<std::uint32_t> visibleOpaque, visibleBlended, shadowCasters;
  std::vector<InstanceRecord> instanceRecords;
  std::array<Mat4, 4> cascadeMatrices;
  std::array<float, 4> cascadeSplits{};
  Frustum frozenFrustum;
  bool hasFrozenCascades = false;
  // Un-jittered, for the motion vectors.
  Mat4 previousViewProjection;
  bool hasPreviousView = false;

  FrameStatistics stats;
  float elapsedSeconds = 0.0f;
  std::uint32_t frameIndex = 0;
  std::uint32_t frameCounter = 0;
  std::uint32_t accumulatedFrames = 0;
  std::uint64_t accumulationKey = 0;
  std::uint32_t resourceGeneration = 0;
};

} // namespace basalt
