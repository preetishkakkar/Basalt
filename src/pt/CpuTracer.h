// The CPU path tracer: the shared path loop (pt_integrator.slang, generated as C++) run per
// pixel on worker threads, over the software BVH, Embree or the AVX2 wide BVH, accumulating
// progressively, denoised on request. Independent of Vulkan.
#pragma once
#include "pt/Bvh.h"
#include "pt/BvhVariants.h"
#include "pt/Denoiser.h"
#include "pt/Embree.h"
#include "pt/EnvironmentSun.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace pt {

// What a scene contributes, rebuilt when the geometry or the environment changes.
struct CpuScene {
  std::vector<float> vertices;          // kVertexFloats per vertex
  std::vector<uint> indices;
  std::vector<TraceInstance> instances;
  std::vector<uint> triangleCounts;     // per instance
  HostTextures textures;
  HostEnvironment environment;
  std::vector<float> distribution;      // the environment distribution
  float4 distributionInfo;              // columns, rows, integral, present
  EnvironmentSun environmentSun;        // what loadEnvironment moved out of the image
  std::vector<float> specularAlbedo;    // buildSpecularAlbedoTable()
  std::vector<PtEmissiveTriangle> emissiveTriangles;
  Bvh bvh;
  BvhStatistics bvhStatistics;
  std::shared_ptr<EmbreeScene> embree;  // when the frame asks for Embree
  std::shared_ptr<WideBvh> wideAvx2;    // optional quantized BVH8 for the AVX2 traversal
};

// What changes every restart.
struct CpuFrame {
  PathUniforms uniforms{};
  std::vector<Material> materials;
  std::vector<Light> lights;
  uint width = 0;
  uint height = 0;
  uint targetSamples = 0;  // stop there; zero runs until the next restart
  uint intersector = 0;    // 0 binary software BVH, 1 Embree, 2 quantized BVH8 + AVX2
  bool denoise = false;
};

// A CPU frame as the generated code sees it: the scene's and the frame's arrays, the uniforms
// and the frame's intersector.
class FrameView : public TraceView {
public:
  FrameView(const CpuScene &scene, const CpuFrame &frame);
};

// ReSTIR DI state, per pixel, carried from frame to frame: this frame's primary
// surfaces and reservoirs after temporal reuse, the previous frame's surfaces and final
// reservoirs, and the visible direct light the path tracer adds at the primary vertex.
struct RestirState {
  std::vector<PtRestirSurface> surfaces, previousSurfaces;
  std::vector<PtReservoir> reservoirs, previousReservoirs;
  std::vector<float3> direct, directDiffuse;
  PtRestirCamera previous{};  // image.z: the history is valid
};

// One path's radiance, and the first hit's base colour and normal for the denoiser.
struct PathSample {
  float3 radiance;
  float3 albedo;
  float3 normal;
  PtReconstructionSample guide;
};

// Runs fn(i) for every i below a count across its threads and the calling one.
class WorkerPool {
public:
  explicit WorkerPool(unsigned threads);
  ~WorkerPool();
  WorkerPool(const WorkerPool &) = delete;
  WorkerPool &operator=(const WorkerPool &) = delete;
  void parallelFor(uint count, const std::function<void(uint)> &fn);
  unsigned size() const { return static_cast<unsigned>(workers.size()) + 1; }

private:
  void work();
  std::vector<std::thread> workers;
  std::mutex mutex;
  std::condition_variable startCondition, finishedCondition;
  const std::function<void(uint)> *task = nullptr;
  uint taskCount = 0;
  std::atomic<uint> next{0};
  std::uint64_t epoch = 0;
  unsigned finishedWorkers = 0;
  bool quit = false;
};

class CpuTracer {
public:
  // Zero threads uses every hardware thread but one.
  explicit CpuTracer(unsigned threads = 0);
  ~CpuTracer();
  CpuTracer(const CpuTracer &) = delete;
  CpuTracer &operator=(const CpuTracer &) = delete;

  void setScene(std::shared_ptr<const CpuScene> scene);
  // Restarts accumulation with this frame; anything in flight is abandoned.
  void start(CpuFrame frame);

  struct Image {
    std::vector<float> mean;           // RGBA, the running mean
    std::vector<float> denoised;       // RGBA, the latest denoised mean; empty without denoising
    std::vector<std::uint16_t> half;   // what to show, RGBA16F: denoised when there is one
    uint width = 0, height = 0;
    uint samples = 0;                  // samples per pixel; zero for the preview
    uint denoisedSamples = 0;          // the samples the denoised image was made from
    uint previewScale = 1;             // 4 for the explicitly labelled reduced-resolution preview
    std::uint64_t version = 0;
    std::uint64_t generation = 0;      // source camera/scene generation, never a display frame
    PathUniforms uniforms{};          // camera that actually traced this publication
    std::vector<PtReconstructionSample> guides; // fresh paths at uniforms.image.xy (preview or full resolution)
  };
  // Copies the latest image when it is newer than `version`; false when there is nothing new.
  // Without `withFloats`, only the half-float copy the display needs.
  bool latest(Image &image, std::uint64_t version, bool withFloats = true) const;
  double samplesPerSecond() const { return rate.load(); }
  unsigned threads() const { return pool.size(); }
  // The denoiser's device once it has run, or why it could not.
  std::string denoiserStatus() const;

  // restirDirect, restirDiffuse: with ReSTIR DI, the pixel's visible direct light at the
  // primary vertex (restirFrame) and its diffuse part.
  static PathSample tracePixel(const CpuScene &scene, const CpuFrame &frame, uint pixelX, uint pixelY,
                               uint sampleIndex, float3 restirDirect = float3(0.0f),
                               float3 restirDiffuse = float3(0.0f));
  static PathSample tracePixel(const TraceView &view, uint pixelX, uint pixelY, uint sampleIndex,
                               float3 restirDirect = float3(0.0f), float3 restirDiffuse = float3(0.0f));

  // One ReSTIR DI frame for sample `sampleIndex` of every pixel: primary hits, initial
  // resampling, temporal and spatial reuse (per uniforms.estimator) and the shadow rays,
  // leaving the direct light in state.direct and the state ready for the next frame.
  static void restirFrame(const CpuScene &scene, const CpuFrame &frame, uint sampleIndex, RestirState &state,
                          WorkerPool &pool);

  // Synchronous: the mean of `samples` paths per pixel, RGB, rows from the top.
  struct Result {
    std::vector<float> color, albedo, normal;
  };
  // With ReSTIR DI each sample is a frame, reusing the previous sample's reservoirs.
  static Result render(const CpuScene &scene, const CpuFrame &frame, uint samples, WorkerPool &pool);

private:
  void coordinate();
  void publish(const std::vector<float> &sums, uint samples, const CpuFrame &frame, std::uint64_t generation,
               const std::vector<PtReconstructionSample> &guides = {}, const PathUniforms *guideCamera = nullptr);

  WorkerPool pool;
  std::thread coordinator;
  mutable std::mutex mutex;
  std::condition_variable wake;
  std::shared_ptr<const CpuScene> scene;
  std::shared_ptr<const CpuFrame> frame;
  std::atomic<std::uint64_t> generation{0};
  bool quit = false;
  Image published;
  std::atomic<double> rate{0.0};
  // Used only on the coordinator thread.
  std::unique_ptr<Denoiser> denoiser;
  std::vector<float> lastDenoised;
  uint lastDenoisedSamples = 0;
  std::string denoiserState;
};

std::uint16_t floatToHalf(float value);

} // namespace pt
