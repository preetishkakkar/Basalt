#include "pt/Gltf.h"
#include "pt/Denoiser.h"
#include "pt/CaptureMetadata.h"
#include "pt/ImageFile.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using pt::float3;
using pt::float4;
using pt::uint;
using pt::CaptureMetadata;
using pt::hashFile;
using pt::kRouletteStartBounce;
using pt::writeCaptureMetadata;

// rgb rows from the top; PFM or OpenEXR by the path's extension (pt/ImageFile.h).
void writeImage(const std::filesystem::path &path, const std::vector<float> &rgb, uint width, uint height,
                const std::string &content) {
  for (float value : rgb) if (!std::isfinite(value)) throw std::runtime_error("render contains non-finite pixels");
  std::vector<float> rows(rgb.size());
  for (uint y = 0; y < height; ++y)
    std::copy_n(rgb.data() + static_cast<std::size_t>(y) * width * 3, static_cast<std::size_t>(width) * 3,
                rows.data() + static_cast<std::size_t>(height - 1 - y) * width * 3);
  if (!pt::writeLinearImage(path.string(), rows, width, height, content))
    throw std::runtime_error("could not write " + path.string());
}

std::string outputKind(const std::string &path, const std::string &content) {
  return content + (pt::isExrPath(path) ? "-exr" : "-pfm");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
      std::cerr << "usage: basalt-pt-cli scene.gltf --output image.pfm|image.exr [--spp N] [--width N] [--height N] "
                   "[--bounces N] [--seed N] [--threads N] [--environment file.hdr] [--environment-sun extract|keep] "
                   "[--intersector own|embree|avx2] [--denoised-output image.pfm] "
                   "[--aperture R] [--focus-distance D] [--texture-filter level0|raycone] "
                   "[--di-estimator nee|restir] [--restir-reuse none|temporal|spatial|both] "
                   "[--restir-candidates N]\n";
      return argc < 2 ? 2 : 0;
    }
    std::string scenePath = argv[1], outputPath = "reference.pfm", intersector = "own";
    std::string environmentPath, denoisedPath;
    uint spp = 64, width = 640, height = 360, bounces = 8, seed = 0, threads = 0;
    float aperture = 0.0f, focusDistance = 0.0f;
    bool rayCones = false, extractSun = true;
    int restir = 0, restirReuse = -1;
    uint restirCandidates = 0;
    // Whole decimal numbers only: "-1" and "12abc" are errors, not 4294967295 and 12.
    auto number = [&](int &i, const char *option) {
      if (++i >= argc) throw std::runtime_error(std::string(option) + " requires a value");
      const std::string text = argv[i];
      if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos || text.size() > 10)
        throw std::runtime_error(std::string(option) + " needs a non-negative integer, not '" + text + "'");
      const unsigned long long value = std::stoull(text);
      if (value > std::numeric_limits<uint>::max()) throw std::runtime_error(std::string(option) + " is too large");
      return static_cast<uint>(value);
    };
    for (int i = 2; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--output") {
        if (++i >= argc) throw std::runtime_error("--output requires a path");
        outputPath = argv[i];
      } else if (option == "--environment") {
        if (++i >= argc) throw std::runtime_error("--environment requires a path");
        environmentPath = argv[i];
      } else if (option == "--environment-sun") {
        if (++i >= argc) throw std::runtime_error("--environment-sun requires extract or keep");
        const std::string name = argv[i];
        if (name != "extract" && name != "keep") throw std::runtime_error("--environment-sun requires extract or keep");
        extractSun = name == "extract";
      } else if (option == "--denoised-output") {
        if (++i >= argc) throw std::runtime_error("--denoised-output requires a path");
        denoisedPath = argv[i];
      } else if (option == "--spp") spp = number(i, "--spp");
      else if (option == "--width") width = number(i, "--width");
      else if (option == "--height") height = number(i, "--height");
      else if (option == "--bounces") bounces = number(i, "--bounces");
      else if (option == "--seed") seed = number(i, "--seed");
      else if (option == "--threads") threads = number(i, "--threads");
      else if (option == "--aperture" || option == "--focus-distance") {
        if (++i >= argc) throw std::runtime_error(option + " requires a value");
        char *end = nullptr;
        const double value = std::strtod(argv[i], &end);
        if (end == argv[i] || *end != '\0' || !(value >= 0.0) || value > 1e6)
          throw std::runtime_error(option + " needs a non-negative number");
        (option == "--aperture" ? aperture : focusDistance) = static_cast<float>(value);
      }
      else if (option == "--texture-filter") {
        if (++i >= argc) throw std::runtime_error("--texture-filter requires level0 or raycone");
        const std::string name = argv[i];
        if (name != "level0" && name != "raycone") throw std::runtime_error("--texture-filter requires level0 or raycone");
        rayCones = name == "raycone";
      }
      else if (option == "--di-estimator") {
        if (++i >= argc) throw std::runtime_error("--di-estimator requires nee or restir");
        const std::string name = argv[i];
        if (name != "nee" && name != "restir") throw std::runtime_error("--di-estimator requires nee or restir");
        restir = name == "restir" ? 1 : 0;
      }
      else if (option == "--restir-reuse") {
        if (++i >= argc) throw std::runtime_error("--restir-reuse requires none, temporal, spatial or both");
        const std::string name = argv[i];
        restirReuse = name == "none" ? 0 : name == "temporal" ? 1 : name == "spatial" ? 2 : name == "both" ? 3 : -1;
        if (restirReuse < 0) throw std::runtime_error("--restir-reuse requires none, temporal, spatial or both");
      }
      else if (option == "--restir-candidates") {
        restirCandidates = number(i, "--restir-candidates");
        if (restirCandidates < 1 || restirCandidates > 13) throw std::runtime_error("--restir-candidates must be 1 to 13");
      }
      else if (option == "--intersector") {
        if (++i >= argc) throw std::runtime_error("--intersector requires own, embree or avx2");
        intersector = argv[i];
      }
      else throw std::runtime_error("unknown option " + option);
    }
    if (!spp || !width || !height || !bounces) throw std::runtime_error("SPP, dimensions and bounces must be positive");
    if (intersector != "own" && intersector != "embree" && intersector != "avx2")
      throw std::runtime_error("intersector must be own, embree or avx2");

    const auto loadStart = std::chrono::steady_clock::now();
    pt::LoadedGltf loaded = pt::loadGltf(scenePath, threads);
    if (!environmentPath.empty()) pt::loadEnvironment(loaded.scene, environmentPath, extractSun);
    if (intersector == "embree") {
      if (!pt::EmbreeScene::available()) throw std::runtime_error("this build has no Embree");
      loaded.scene.embree = std::make_shared<pt::EmbreeScene>(loaded.scene);
      loaded.frame.intersector = 1;
    } else if (intersector == "avx2") {
      if (!pt::cpuAvx2Available()) throw std::runtime_error("this CPU has no AVX2 support");
      loaded.scene.wideAvx2 = std::make_shared<pt::WideBvh>(
          pt::buildWideBvh(loaded.scene.bvh, loaded.scene.instances, 8u));
      loaded.frame.intersector = 2;
    }
    const double loadSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - loadStart).count();

    const float3 centre = (loaded.boundsMin + loaded.boundsMax) * 0.5f;
    const float radius = std::max(pt::length(loaded.boundsMax - loaded.boundsMin) * 0.5f, 0.1f);
    const float3 eye = centre + float3(0.0f, radius * 0.35f, radius * 2.5f);
    const float3 forward = pt::normalize(centre - eye);
    const float3 right = pt::normalize(pt::cross(forward, float3(0.0f, 1.0f, 0.0f)));
    const float3 up = pt::cross(right, forward);
    const float tangent = std::tan(0.8f * 0.5f);
    loaded.frame.width = width;
    loaded.frame.height = height;
    auto &u = loaded.frame.uniforms;
    u.cameraPosition = float4(eye, 1.0f);
    u.cameraForward = float4(forward, 0.0f);
    u.cameraRight = float4(right * (tangent * static_cast<float>(width) / height), 0.0f);
    u.cameraUp = float4(up * tangent, 0.0f);
    u.image = float4(static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f);
    u.path = float4(static_cast<float>(bounces), kRouletteStartBounce, 0.0f, 0.0f);
    u.environment = float4(1.0f, 1.0f, 1.0f, 1.0f);
    u.distribution = loaded.scene.distributionInfo;
    const pt::EnvironmentSun &sun = loaded.scene.environmentSun;
    pt::environmentSunUniforms(sun.direction, sun.irradiance, sun.angularRadius, u.sunDirection, u.sunRadiance);
    u.counts = pt::uint4(static_cast<uint>(loaded.frame.lights.size()), 7u, seed, 0u);
    u.emissive = pt::uint4(static_cast<uint>(loaded.scene.emissiveTriangles.size()), 0u, 0u, 0u);
    // Thin lens: the focus defaults to the scene centre's depth.
    const float focus = focusDistance > 0.0f ? focusDistance : std::max(pt::dot(centre - eye, forward), 1e-3f);
    u.lens = float4(aperture, focus, rayCones ? 1.0f : 0.0f, std::atan(2.0f * tangent / static_cast<float>(height)));
    if (!restir && (restirReuse >= 0 || restirCandidates > 0))
      throw std::runtime_error("--restir-reuse and --restir-candidates require --di-estimator restir");
    // ReSTIR DI: every sample is a frame reusing the previous one's reservoirs.
    u.estimator = pt::uint4(static_cast<uint>(restir), restirCandidates > 0 ? restirCandidates : 8u,
                            static_cast<uint>(restirReuse >= 0 ? restirReuse : 3), 0u);

    pt::WorkerPool pool(threads ? threads : std::max(1u, std::thread::hardware_concurrency()));
    const auto renderStart = std::chrono::steady_clock::now();
    pt::CpuTracer::Result result = pt::CpuTracer::render(loaded.scene, loaded.frame, spp, pool);
    const double renderSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - renderStart).count();
    writeImage(outputPath, result.color, width, height, "raw-linear-rgb");
    double denoiseSeconds = 0.0;
    if (!denoisedPath.empty()) {
      if (!pt::Denoiser::available()) throw std::runtime_error("this build has no Open Image Denoise");
      const auto denoiseStart = std::chrono::steady_clock::now();
      pt::Denoiser oidn(pt::Denoiser::Device::Cpu);
      if (!oidn.error().empty()) throw std::runtime_error(oidn.error());
      auto filtered = oidn.denoise(result.color, result.albedo, result.normal, width, height);
      if (filtered.empty()) throw std::runtime_error(oidn.error());
      denoiseSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - denoiseStart).count();
      writeImage(denoisedPath, filtered, width, height, "denoised-linear-rgb");
    }
    CaptureMetadata m;
    m.revision = BASALT_GIT_REVISION;
    m.dirtyAtConfigure = BASALT_GIT_DIRTY != 0;
    m.scene = scenePath;
    m.sceneFileHash = hashFile(scenePath);
    m.sceneContentHash = loaded.contentHash;
    m.environment = environmentPath;
    m.device = "CPU";
    m.shaderCompiler = "slang-" BASALT_SLANG_VERSION;  // the path tracer is pt_cpu.slang as C++
    m.renderer = "cpu";
    m.intersector = intersector == "embree" ? "embree" : intersector == "avx2" ? "quantized-bvh8-avx2" : "own-bvh";
    m.builder = intersector == "embree" ? "embree" : "cpu-binned-sah";
    m.bvhLayout = intersector == "embree" ? "embree" : intersector == "avx2" ? "quantized-bvh8" : "binary-float";
    m.execution = "cpu-tiles";
    m.cpuThreads = pool.size();
    m.width = width;
    m.height = height;
    m.spp = spp;
    m.targetSpp = spp;
    m.bounces = bounces;
    m.rouletteStart = kRouletteStartBounce;
    m.strategy = "mis";
    m.seed = seed;
    m.textureFilter = rayCones ? "ray-cone trilinear" : "level-zero bilinear";
    if (restir) {
      static const char *reuse[] = {"none", "temporal", "spatial", "both"};
      m.diEstimator = "restir";
      m.restirReuse = reuse[u.estimator.z];
      m.restirCandidates = u.estimator.y;
      m.restirBias = u.estimator.z == 0 ? "unbiased" : "visibility-only";
    }
    m.aperture = aperture;
    m.focusDistance = u.lens.y;
    m.denoise = denoisedPath.empty() ? "off" : "oidn-cpu (separate output)";
    m.outputs.push_back({outputPath, outputKind(outputPath, "raw-linear-rgb")});
    if (!denoisedPath.empty()) m.outputs.push_back({denoisedPath, outputKind(denoisedPath, "denoised-linear-rgb")});
    m.measurements = {{"load_seconds", loadSeconds}, {"render_seconds", renderSeconds},
                      {"denoise_seconds", denoiseSeconds}};
    m.notes = {{"camera_eye", std::to_string(eye.x) + " " + std::to_string(eye.y) + " " + std::to_string(eye.z)}};
    if (!environmentPath.empty()) {
      m.notes.push_back({"environment_sun", !extractSun ? "kept in the image" : sun.found ? "extracted" : "none found"});
      if (sun.found) {
        m.measurements.push_back({"environment_sun_irradiance", pt::ptLuminance(sun.irradiance)});
        m.measurements.push_back({"environment_sun_radius_degrees", sun.angularRadius * 180.0 / 3.14159265358979323846});
        m.measurements.push_back({"environment_sun_share", sun.share});
      }
    }
    if (!writeCaptureMetadata(outputPath + ".json", m))
      throw std::runtime_error("could not write " + outputPath + ".json");
    std::cout << "wrote " << outputPath << " (" << width << 'x' << height << ", " << spp << " spp, "
              << renderSeconds << " s; scene/BVH " << loadSeconds << " s)\n";
    if (!denoisedPath.empty()) std::cout << "wrote denoised " << denoisedPath << " (raw output unchanged)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
