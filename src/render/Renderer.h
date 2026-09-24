// The frame: shadows, forward PBR into an HDR target and a small G-buffer, sky, reflection
// resolve, temporal pass, bloom, post.
#pragma once
#include "gpu/Descriptors.h"
#include "gpu/Pipeline.h"
#include "gpu/RayPipeline.h"
#include "gpu/Swapchain.h"
#include "gpu/Uploader.h"
#include "pt/CpuTracer.h"
#include "render/Environment.h"
#include "render/RayTracing.h"
#include "render/GpuProfiler.h"
#include "render/PathReconstruction.h"
#include "scene/Camera.h"
#include "scene/Scene.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace basalt {

struct RenderSettings {
  float sunAzimuth = radians(40.0f);
  float sunElevation = radians(35.0f);
  Vec3 sunColor{1.0f, 0.96f, 0.90f};
  float sunIntensity = 4.0f;
  float sunAngularRadius = radians(0.6f); // Soft traced shadows widen with it.

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

  // Path tracing. The renderer: 0 the rasteriser, 1 the CPU path tracer, 2 the GPU path
  // tracer with ray queries, 3 the GPU path tracer with its own BVH, 4 hybrid raster primary + traced continuation,
  // 5 the GPU path tracer using a full Vulkan ray pipeline.
  int renderer = 0;
  int pathBounces = 8;
  std::uint32_t pathSeed = 0;
  int pathStrategy = 0;       // 0 multiple importance sampling, 1 BSDF sampling only, 2 light sampling only
  float pathClamp = 0.0f;     // largest contribution a path may add after the first hit; 0 is none
  int pathTargetSamples = 0;  // stop accumulating there; 0 never stops
  int pathSamplesPerFrame = 1;  // GPU path tracers: paths per pixel per frame
  int cpuThreads = 0;         // 0 is every hardware thread but one
  int cpuIntersector = 0;     // 0 binary software BVH, 1 Intel Embree, 2 quantized BVH8 + AVX2
  bool pathDenoise = false;   // Intel Open Image Denoise over the image on screen
  bool pathTemporal = false;  // traced temporal/a-trous display reconstruction; raw capture stays raw
  int pathBvhDiagnostic = 0;  // software GPU only: 0 shaded, 1 node visits, 2 triangle tests
  int pathBvhBuilder = 0;     // software GPU only: 0 CPU SAH, 1 GPU LBVH
  int pathBvhWidth = 0;       // software GPU only: 0 binary, 1 quantized BVH4, 2 quantized BVH8
  int pathExecution = 0;      // GPU tracer: 0 megakernel (iterative raygen for the ray pipeline), 1 wavefront queues
  int pathWaveCapacity = 0;   // wavefront: paths per batch; 0 automatic (pt::kWavefrontDefaultPaths within device limits)
  int pathWaveAllocation = 0; // wavefront queue slots: 0 automatic (per path), 1 one atomic per subgroup, 2 one atomic per path
  int pathWaveQueueLimit = 0; // tests only: queue capacity below the batch, to provoke reported overflow
  int pathWaveFusion = 0;     // wavefront, ray queries and own BVH: 0 automatic (fused), 1 fused, 2 separate
  int pathComparison = 0;     // hybrid: 0 traced, 1 raster, 2 difference, 3 split, 4 SPP
  bool pathProfile = false;   // GPU tracers: per-stage timestamps and queue counters
  float pathAperture = 0.0f;       // thin-lens radius in world units; 0 is the pinhole camera
  float pathFocusDistance = 0.0f;  // along the view axis; 0 focuses on the orbit target
  int pathTextureFilter = 0;       // 0 level zero, 1 ray cones
  int pathDiEstimator = 0;         // direct light at the primary vertex: 0 NEE, 1 ReSTIR DI (one path per frame)
  int pathRestirReuse = 3;         // ReSTIR reuse bits: 1 temporal, 2 spatial
  int pathRestirCandidates = 8;    // ReSTIR initial candidates M, 1 to 13

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
  // Samples averaged into the image on screen: accumulated frames for the rasteriser,
  // samples per pixel for a path tracer.
  std::uint32_t samples = 0;
  double pathsPerSecond = 0.0;  // path tracers only
  float gpuMilliseconds = 0.0f;
  float freshSampleGpuMilliseconds = 0.0f; // last frame that actually received fresh path samples
  float reconstructionMilliseconds = 0.0f, guideUploadMilliseconds = 0.0f;
  float guideTransferMilliseconds = 0.0f;
  std::uint64_t reconstructionBytes = 0;
  std::uint32_t previewScale = 1;
  std::string unavailableReason;  // why the requested renderer did not record this frame
  // What actually recorded the last frame, which can differ from the settings when the
  // requested tracer cannot run on this device: captures record and check this.
  int activeRenderer = 0;
  bool activeWavefront = false;
  std::uint64_t wavefrontQueueBytes = 0;  // current wavefront queue allocation
  float traceGpuMilliseconds = 0.0f;       // GPU tracing alone, the latest collected frame
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
  bool rayPipelineAvailable() const {
    return context.rayPipelineSupported && acceleration != nullptr && pathRayPipeline != nullptr;
  }

  void resize(std::uint32_t width, std::uint32_t height);

  // Leaves the render pass open for the interface.
  void render(VkCommandBuffer command, std::uint32_t imageIndex, const Camera &camera,
              float deltaSeconds);

  RenderSettings settings;
  const FrameStatistics &statistics() const { return stats; }
  bool waveSubgroupAllocationSupportedOnDevice() const { return waveSubgroupAllocationSupported; }
  // Whether this frame's wavefront fuses nearest-hit traversal into shading: ray queries and
  // the own BVH (faster in every measured case, V6.1 results), never with subgroup allocation,
  // which only the split stages implement.
  bool waveFusionActive() const {
    if (settings.pathExecution != 1 || settings.pathWaveFusion == 2 || waveSubgroupAllocationActive()) return false;
    if (settings.renderer == 2) return bool(pathWaveFusedRtPipeline);
    return settings.renderer == 3;
  }
  // The wavefront queue allocation in use. Automatic is per-path atomics: subgroup
  // aggregation measured no gain (V6.1 results) and stays an explicit option.
  bool waveSubgroupAllocationActive() const {
    return settings.pathWaveAllocation == 1 && waveSubgroupAllocationSupported;
  }
  // GPU tracer timing: waits for the device, then returns the collected profile.
  const GpuProfiler *finishProfiling();
  Vec3 sunDirection() const;
  void rebakeProceduralSky();
  // Every material set holds the environment cubes and must be rewritten.
  void environmentChanged() { rebuildSceneResources(); }
  void applyGroundMaterial();
  // Copies this frame's linear image, before bloom and tone mapping, for writeLinearCapture.
  // Recorded after the frame's rendering has ended.
  void recordLinearCapture(VkCommandBuffer command);
  bool writeLinearCapture(const std::string &path);
  bool writeDenoisedCapture(const std::string &path);
  // GPU tracers only: plane 1 is the first-hit albedo mean, plane 2 the first-hit normal mean.
  bool writeGuideCapture(const std::string &path, int plane);
  // The active software BVH, when one has been built; and the CPU tracer's thread count.
  const pt::BvhStatistics *cpuBvh() const {
    // Only for the renderers that use one: driver-built hardware structures have none here.
    if (settings.renderer == 1) return cpuScene ? &cpuScene->bvhStatistics : nullptr;
    if (settings.renderer == 3) return softwareBvhVersion == sceneVersion ? &softwareBvhStatistics : nullptr;
    return nullptr;
  }
  unsigned cpuThreadCount() const { return cpuTracer ? cpuTracer->threads() : 0u; }
  std::string denoiserStatus() const { return cpuTracer ? cpuTracer->denoiserStatus() : std::string(); }
  std::uint32_t denoisedSamples() const { return cpuImage.denoisedSamples; }
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
  void buildHitTextureTable(Scene &scene, std::vector<VkImageView> &views, std::vector<std::uint32_t> &slots,
                            std::vector<Image *> &images);
  // The CPU path tracer: the host scene is read back when it is chosen and dropped when not.
  void recordCpuTrace(VkCommandBuffer command, const Camera &camera);
  // The GPU path tracer: accumulates into float images and writes this frame's lit image.
  void recordGpuTrace(VkCommandBuffer command, const Camera &camera);
  void recordHybridSnapshot(VkCommandBuffer command);
  void recordPathComparison(VkCommandBuffer command);
  void buildCpuScene();
  void buildSoftwareBvh();
  void ensureEmissiveTriangles();
  pt::PathUniforms makePathUniforms(const Camera &camera);
  struct FrameResources;
  void recordWavefront(VkCommandBuffer command, FrameResources &frame, std::uint32_t thisFrame, bool software,
                       bool wideSoftware, bool pipelineBackend, bool profiling);
  // ReSTIR DI on a GPU path tracer: per-pixel buffers sized for the image when
  // active (one element otherwise, so the path tracers' bindings stay valid), and a frame's
  // reservoir passes ahead of the path trace.
  bool restirActive() const;
  void ensureRestirResources(bool active, bool guides, bool costs);
  void recordRestir(VkCommandBuffer command, FrameResources &frame, bool software, bool wideSoftware,
                    bool pipelineBackend, const pt::PathUniforms &uniforms);

  Context &context;
  Swapchain &swapchain;
  Uploader &uploader;

  std::unique_ptr<Scene> activeScene;
  std::unique_ptr<Environment> environmentState;
  std::unique_ptr<SceneAccelerationStructure> acceleration;
  // Hit texture table, in the slots the instance table packs, and the images behind them.
  std::vector<VkImageView> hitTextureViews;
  std::vector<Image *> hitTextureImages;
  std::vector<std::uint32_t> materialSlots;
  // One row per acceleration structure instance, read by every ray.
  TraceScene traceScene;
  Buffer traceInstanceBuffer;

  std::uint32_t width = 0, height = 0;
  Image hdrColor, normalRoughness, reflectionWeight;
  Image gbufferBaseMetallic, gbufferGeometricCoverage, gbufferEmissive, gbufferIdentity;
  Image pathComparisonRaster;
  Image reflection;
  // The composite writes litCurrent; the temporal pass resolves it against the other
  // litColor into this frame's, which bloom and post read.
  Image litCurrent;
  Image litColor[kFramesInFlight];
  Image depthBuffer, shadowMap, shadowDummyColor, bloomChain;
  std::vector<VkImageView> bloomMipViews;
  std::uint32_t bloomMipCount = 0;
  std::uint32_t reflectionMipCount = 0;
  VkImageView reflectionMipZero = VK_NULL_HANDLE;
  VkImageView shadowLayerViews[4]{};
  VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

  std::unique_ptr<Program> forwardProgram, shadowProgram, skyProgram, postProgram;
  std::unique_ptr<Program> bloomDownProgram, bloomUpProgram, resolveProgram, compositeProgram;
  std::unique_ptr<Program> temporalProgram, clusterProgram;
  std::unique_ptr<Program> pathTraceProgram;    // software BVH, on every Vulkan device
  std::unique_ptr<Program> pathTraceEmissiveProgram;
  std::unique_ptr<Program> pathTraceWideProgram, pathTraceWideEmissiveProgram;
  std::unique_ptr<Program> pathTraceRtProgram, pathTraceHybridProgram;  // only on a ray tracing device
  std::unique_ptr<RayPipeline> pathRayPipeline;
  std::unique_ptr<RayPipeline> pathWaveIntersectRayPipeline, pathWaveShadowRayPipeline;
  std::unique_ptr<Program> pathCompareProgram, pathGuideExportProgram;
  std::unique_ptr<Program> pathWaveShadeProgram, pathWaveShadeAtomicProgram, pathWaveResolveProgram;
  // Intersect fused into shade: ray queries, own binary BVH, own BVH4/BVH8.
  std::unique_ptr<Program> pathWaveFusedRtProgram, pathWaveFusedProgram, pathWaveFusedWideProgram;
  bool waveSubgroupAllocationSupported = false;
  std::unique_ptr<Program> pathWaveIntersectProgram, pathWaveIntersectWideProgram, pathWaveIntersectRtProgram;
  std::unique_ptr<Program> pathWaveShadowProgram, pathWaveShadowWideProgram, pathWaveShadowRtProgram;
  std::unique_ptr<Program> pathRestirInitialProgram, pathRestirSpatialProgram;
  Pipeline forwardPipeline, forwardBlendPipeline, forwardWirePipeline;
  Pipeline shadowPipeline, shadowMaskedPipeline;
  // The same without back-face culling, for materials glTF marks double-sided.
  Pipeline forwardTwoSidedPipeline, shadowTwoSidedPipeline, shadowMaskedTwoSidedPipeline;
  Pipeline skyPipeline, postPipeline, bloomDownPipeline, bloomUpPipeline;
  Pipeline resolvePipeline, compositePipeline, temporalPipeline, clusterPipeline;
  Pipeline pathTracePipeline, pathTraceEmissivePipeline, pathTraceRtPipeline, pathTraceHybridPipeline;
  Pipeline pathTraceWidePipeline, pathTraceWideEmissivePipeline;
  Pipeline pathComparePipeline, pathGuideExportPipeline;
  Pipeline pathWaveShadePipeline, pathWaveShadeAtomicPipeline, pathWaveResolvePipeline, pathWaveFusedRtPipeline;
  Pipeline pathWaveFusedPipeline, pathWaveFusedWidePipeline;
  Pipeline pathWaveIntersectPipeline, pathWaveIntersectWidePipeline, pathWaveIntersectRtPipeline;
  Pipeline pathWaveShadowPipeline, pathWaveShadowWidePipeline, pathWaveShadowRtPipeline;
  Pipeline pathRestirInitialPipeline, pathRestirSpatialPipeline;

  // Per-frame data, one copy per frame in flight.
  struct FrameResources {
    Buffer frameUniforms;
    Buffer postUniforms;
    Buffer reflectionUniforms;
    Buffer temporalUniforms;
    Buffer clusterUniforms;
    Buffer pathUniforms;
    Buffer hybridInverseViewProjection;
    Buffer pathComparisonUniforms;
    VkDescriptorSet pathSet = VK_NULL_HANDLE;
    VkDescriptorSet pathPipelineSet = VK_NULL_HANDLE;
    VkDescriptorSet pathGuideExportSet = VK_NULL_HANDLE;
    VkDescriptorSet pathComparisonSet = VK_NULL_HANDLE;
    VkDescriptorSet pathWaveShadeSet = VK_NULL_HANDLE;
    VkDescriptorSet pathWaveIntersectSet = VK_NULL_HANDLE, pathWaveShadowSet = VK_NULL_HANDLE;
    VkDescriptorSet pathWaveResolveSet = VK_NULL_HANDLE;
    // ReSTIR DI: the backend's intersect and shadow stages over the per-pixel buffers, and
    // the reservoir passes per surface/final-reservoir parity.
    VkDescriptorSet restirIntersectSet = VK_NULL_HANDLE, restirShadowSet = VK_NULL_HANDLE;
    VkDescriptorSet restirInitialSet[2]{}, restirSpatialSet[2]{};
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
  std::vector<LightRecord> hostLights;
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
  std::array<bool, kFramesInFlight> freshSampleFrame{};

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
  Buffer linearCaptureBuffer;
  bool linearCaptureFloat = false;  // the copy is the GPU tracer's float accumulator
  VkExtent2D linearCaptureExtent{};
  float elapsedSeconds = 0.0f;
  std::uint32_t frameIndex = 0;
  std::uint32_t frameCounter = 0;
  std::uint32_t accumulatedFrames = 0;
  std::uint64_t accumulationKey = 0;
  std::uint32_t resourceGeneration = 0;
  std::uint32_t sceneVersion = 0;  // counts setScene calls
  int lastRenderer = 0;

  std::unique_ptr<pt::CpuTracer> cpuTracer;
  std::shared_ptr<pt::CpuScene> cpuScene;
  std::array<std::uint32_t, 3> cpuSceneKey{};  // scene version, environment version, intersector
  std::uint64_t cpuStartedKey = 0;             // the accumulation key the tracer is working on
  pt::CpuTracer::Image cpuImage;               // the latest image, as RGBA16F for display
  std::array<Buffer, kFramesInFlight> cpuStaging;
  std::array<std::uint64_t, kFramesInFlight> cpuUploaded{};
  std::vector<float> specularAlbedo;           // built once, the first time a tracer needs it

  // The GPU path tracer's state: float accumulators (radiance with the sample count in w,
  // first-hit albedo and normal), made the first time it runs and at each resize.
  Image pathAccumulation, pathAlbedoAccumulation, pathNormalAccumulation;
  Buffer pathDistribution, pathAlbedoTable, pathEmissiveBuffer;
  Buffer pathWaveStatesA, pathWaveStatesB, pathWaveResults, pathWaveShadows, pathWaveHits;
  // ReSTIR DI per-pixel state: primary hits, surfaces and final reservoirs alternating by
  // parity (this frame's and the previous frame's), the temporally reused reservoirs, one
  // shadow record, visible result and guide per pixel, and the stage control words.
  // The hits share the shadow records' buffer (never live at the same time); the guides
  // and traversal costs are one element unless reconstruction or a cost view reads them.
  Buffer restirSurfaces[2], restirReservoirs, restirFinals[2], restirShadows, restirResults;
  Buffer restirGuides, restirCosts, restirCounters, restirControl, restirCamera;
  std::uint64_t restirPixels = 0, restirGuidePixels = 0, restirCostPixels = 0;
  std::uint32_t restirParity = 0;
  std::uint64_t restirHistoryKey = 0;
  bool restirHistory = false;
  pt::PathUniforms restirPreviousUniforms{};
  Buffer pathWaveCounters, pathWaveControl, pathWaveCosts, pathWaveGuides;
  std::uint32_t pathWaveCapacity = 0;       // paths per queue array (pt/WavefrontPlan.h)
  std::uint64_t pathWaveGuidePixels = 0;
  std::vector<pt::PtEmissiveTriangle> pathEmissiveMetadata;
  std::uint32_t pathEmissiveVersion = ~0u;
  Buffer softwareBvhNodes, softwareBvhTriangles;
  Buffer softwareTraceInstanceBuffer;  // the builder's instance rows, read only by own-BVH sets
  pt::BvhStatistics softwareBvhStatistics{};
  std::uint32_t softwareBvhVersion = ~0u;
  int softwareBvhBuilder = -1;
  int softwareBvhWidth = -1;
  VkDeviceSize softwareBvhScratchBytes = 0, softwareBvhOutputBytes = 0;
  std::uint32_t softwareBvhRadixPasses = 0, softwareBvhMaximumStack = 0;
  Buffer pathReconstructionSamples;
  std::unique_ptr<PathReconstruction> pathReconstruction;
  std::unique_ptr<GpuProfiler> profiler;  // GPU tracers: trace timing, stage marks on request
  int reconstructionBackend = 0;
  std::uint32_t pathDistributionVersion = ~0u;
  std::uint32_t pathSetsGeneration = ~0u, pathSetsEnvironment = ~0u;
  int pathSetsBackend = -1;
  std::uint32_t gpuSamples = 0;
};

} // namespace basalt
