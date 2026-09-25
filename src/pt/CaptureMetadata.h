// The JSON sidecar written beside every raw capture, by the window application and the
// headless CPU CLI alike: one schema ("basalt-capture/1"), so captures from different
// backends can be compared field by field. Values describe what actually ran, not what
// was requested; a capture whose active backend differs from its request fails instead.
// Independent of Vulkan.
#pragma once
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace pt {

struct CaptureMetadata {
  // Source and toolchain.
  std::string revision;
  bool dirtyAtConfigure = false;
  std::string shaderCompiler;  // empty when no shader was compiled (CPU-only build)

  // Scene: the file as given, a hash of its bytes, and the environment ("" is the default sky).
  std::string scene;
  std::uint64_t sceneFileHash = 0;
  std::uint64_t sceneContentHash = 0;  // loader-level hash when available, else 0
  std::string environment;

  // Device: "CPU" for the host tracer. Feature flags are what the device was created with.
  std::string device;
  std::uint32_t driverVersion = 0, apiVersion = 0;
  bool accelerationStructures = false, rayQuery = false, rayPipeline = false;

  // Backend, as it ran.
  std::string renderer;     // cpu, gpu-ray-query, gpu-own-bvh, gpu-ray-pipeline, hybrid-ray-query, raster
  std::string intersector;  // own-bvh, embree, quantized-bvh8-avx2, driver-built-hardware-as, software-bvh ...
  std::string builder;      // cpu-binned-sah, gpu-serial-lbvh, gpu-parallel-lbvh, gpu-parallel-ploc, embree, driver, not-applicable
  std::string bvhLayout;    // binary-float, quantized-bvh4, quantized-bvh8, not-applicable
  std::string bvhUpdate;    // on-change, rebuild-every-frame, refit, not-applicable
  std::string animation = "off";  // --animate: off, instances, vertices, both
  std::string wideStack = "not-applicable";  // --wide-stack: auto, deep, shallow
  std::string execution;    // megakernel, wavefront, iterative-raygen, wavefront-ray-pipeline, cpu-tiles
  unsigned cpuThreads = 0;

  // Integrator.
  std::uint32_t width = 0, height = 0;
  std::uint32_t spp = 0, targetSpp = 0, samplesPerDispatch = 0;
  std::uint32_t bounces = 0;
  float rouletteStart = 0.0f;
  std::string strategy;     // mis, bsdf-only, light-only
  std::uint32_t seed = 0;
  float fireflyClamp = 0.0f;  // 0 is off
  std::string textureFilter = "level-zero bilinear";
  float aperture = 0.0f;       // thin-lens radius, world units; 0 is a pinhole
  float focusDistance = 0.0f;  // along the view axis
  std::string diEstimator = "nee";  // direct light at the primary vertex: nee or restir
  std::string restirReuse = "none";  // none, temporal, spatial or both
  std::uint32_t restirCandidates = 0;
  std::string restirBias = "none";   // none (NEE), unbiased (initial RIS only) or visibility-only

  // Display paths, which never change the raw output.
  std::string reconstruction = "off";
  std::string denoise = "off";  // off, or the OIDN device used for a separate output

  // Outputs written by this capture (path, kind), e.g. ("x.pfm", "raw-linear-rgb").
  std::vector<std::pair<std::string, std::string>> outputs;

  // Extra numeric facts: memory and timing, (name, value); names are snake_case.
  std::vector<std::pair<std::string, double>> measurements;
  // Per-frame series, (name, values), e.g. fresh-frame trace times for benchmarks.
  std::vector<std::pair<std::string, std::vector<double>>> series;
  // Extra string facts, (name, value).
  std::vector<std::pair<std::string, std::string>> notes;
};

// FNV-1a over a file's bytes; 0 for an empty path. Throws when the file cannot be read.
std::uint64_t hashFile(const std::string &path);

// Writes the sidecar; returns false if the file could not be written.
bool writeCaptureMetadata(const std::string &path, const CaptureMetadata &metadata);

} // namespace pt
