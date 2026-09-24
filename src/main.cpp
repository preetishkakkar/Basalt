// Window, interface and frame loop.
#include "core/Log.h"
#include "gpu/Context.h"
#include "gpu/Swapchain.h"
#include "gpu/Uploader.h"
#include "platform/Window.h"
#include "render/Renderer.h"
#include "render/UiPass.h"
#include "scene/Camera.h"
#include "pt/CaptureMetadata.h"
#include "pt/ImageFile.h"
#include "pt/WavefrontPlan.h"

#include <imgui.h>
#include <imgui_impl_win32.h>

#include <commdlg.h>
#include <shellapi.h>

#include <stb_image_write.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <format>
#include <iomanip>
#include <memory>
#include <string>

namespace basalt {
namespace {

// Renderer names shared by --renderer and scripted --at-frame switches; -2 is unknown.
int rendererFromName(const std::string &name) {
  return name == "raster" ? 0 : name == "cpu" ? 1 : (name == "gpu" || name == "gpu-rt") ? 2 :
         (name == "gpu-bvh" || name == "gpu-software") ? 3 : name == "hybrid" ? 4 :
         (name == "gpu-pipeline" || name == "ray-pipeline") ? 5 : -2;
}

// A lifecycle transition applied before recording frame `frame` (--at-frame).
struct ScheduledAction {
  int frame = 0;
  std::string key, value;
};

std::string openFileDialog(HWND owner, const wchar_t *filter, const wchar_t *title) {
  wchar_t path[MAX_PATH] = L"";
  OPENFILENAMEW dialog{};
  dialog.lStructSize = sizeof(dialog);
  dialog.hwndOwner = owner;
  dialog.lpstrFilter = filter;
  dialog.lpstrFile = path;
  dialog.nMaxFile = MAX_PATH;
  dialog.lpstrTitle = title;
  dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
  if (!GetOpenFileNameW(&dialog)) return {};
  const std::wstring wide(path);
  return std::filesystem::path(wide).string();
}

void applyDarkStyle() {
  ImGui::StyleColorsDark();
  ImGuiStyle &style = ImGui::GetStyle();
  style.WindowRounding = 6.0f;
  style.FrameRounding = 4.0f;
  style.GrabRounding = 4.0f;
  style.WindowBorderSize = 1.0f;
  style.FrameBorderSize = 0.0f;
  style.WindowPadding = ImVec2(10, 10);
  style.ItemSpacing = ImVec2(8, 6);
  ImVec4 *colors = style.Colors;
  colors[ImGuiCol_WindowBg] = ImVec4(0.07f, 0.07f, 0.08f, 0.94f);
  colors[ImGuiCol_TitleBgActive] = ImVec4(0.16f, 0.17f, 0.20f, 1.00f);
  colors[ImGuiCol_Header] = ImVec4(0.20f, 0.22f, 0.27f, 1.00f);
  colors[ImGuiCol_HeaderHovered] = ImVec4(0.28f, 0.31f, 0.38f, 1.00f);
  colors[ImGuiCol_Button] = ImVec4(0.20f, 0.22f, 0.27f, 1.00f);
  colors[ImGuiCol_ButtonHovered] = ImVec4(0.28f, 0.31f, 0.38f, 1.00f);
  colors[ImGuiCol_FrameBg] = ImVec4(0.13f, 0.14f, 0.17f, 1.00f);
  colors[ImGuiCol_SliderGrab] = ImVec4(0.40f, 0.55f, 0.85f, 1.00f);
  colors[ImGuiCol_CheckMark] = ImVec4(0.45f, 0.62f, 0.95f, 1.00f);
}

// A white floor shows shadows but drowns a reflection; a metal mirror reflects but has
// no diffuse for a shadow to darken; a dark gloss does both.
struct GroundPreset {
  Vec3 colour;
  float roughness;
  float metallic;
};
constexpr GroundPreset kGroundPresets[] = {
    {{0.34f, 0.34f, 0.36f}, 0.8f, 0.0f},   // Matte
    {{0.18f, 0.18f, 0.19f}, 0.08f, 0.0f},  // Polished
    {{0.03f, 0.03f, 0.035f}, 0.05f, 0.0f}, // Glossy dark
    {{0.95f, 0.95f, 0.95f}, 0.02f, 1.0f},  // Mirror
};
constexpr int kGroundPresetCount = IM_ARRAYSIZE(kGroundPresets);
// The presets in order, then "Custom" for anything the sliders make.
constexpr const char *groundPresetNames[] = {"Matte", "Polished", "Glossy dark", "Mirror", "Custom"};
static_assert(IM_ARRAYSIZE(groundPresetNames) == kGroundPresetCount + 1);

int groundPresetIndex(const RenderSettings &s) {
  for (int i = 0; i < kGroundPresetCount; ++i) {
    const GroundPreset &p = kGroundPresets[i];
    if (s.groundRoughness == p.roughness && s.groundMetallic == p.metallic && s.groundColor.x == p.colour.x &&
        s.groundColor.y == p.colour.y && s.groundColor.z == p.colour.z)
      return i;
  }
  return kGroundPresetCount;
}

} // namespace

// Decided before the window and device exist.
struct StartupOptions {
  std::uint32_t width = 1600, height = 900;
  bool validation = true;  // benchmarks turn the layer off; correctness runs keep it
};

struct Application {
  explicit Application(const StartupOptions &options)
      : window{"Basalt: Metal shaders on Vulkan", options.width, options.height},
        context{window, options.validation} {}
  Window window;
  Context context;
  Swapchain swapchain{context, window.width(), window.height(), true};
  Uploader uploader{context};
  Renderer renderer{context, swapchain, uploader};
  std::unique_ptr<UiPass> ui;
  Camera camera;

  bool showDemo = false;
  bool showLog = false;
  bool showUi = true;
  bool f1Held = false, f5Held = false, f12Held = false;
  float frameMilliseconds = 0.0f;
  float frameHistory[128]{};
  int frameCursor = 0;
  std::string status;

  void loadScene(const std::string &path) {
    try {
      renderer.setScene(loadGltf(context, uploader, path));
      renderer.applyGroundMaterial();
      camera.frame(renderer.scene()->bounds);
      status = "loaded " + std::filesystem::path(path).filename().string();
    } catch (const std::exception &failure) {
      if (failOnLoadError) throw;
      logError("{}", failure.what());
      status = failure.what();
    }
  }

  // Capture mode: render N frames, or until N samples have accumulated, write the last one, exit.
  std::string capturePath;
  std::string linearPath;  // The raw linear image as PFM, beside or instead of the PNG.
  std::string denoisedLinearPath; // Optional GPU OIDN result; raw accumulation stays separate.
  std::string albedoPath, normalPath; // Optional GPU first-hit guide means.
  bool captureFailed = false;          // a requested output could not be written: exit nonzero
  bool failOnLoadError = false;        // capture mode never renders a substitute scene
  int captureFrame = -1;
  int captureSamples = 0;
  int pathExecutionSwitchFrame = -1; // Scripted capture regression for in-flight backend changes.
  std::vector<ScheduledAction> scheduledActions; // --at-frame lifecycle transitions
  // Turns the camera every frame, so a capture exercises reprojection.
  float spinPerFrame = 0.0f;
  bool fixedSun = false;  // --sun: keep it where it was put
  bool extractEnvironmentSun = true;  // --environment-sun: an .hdr's sun becomes the analytic one
  // The size the copy was taken at; the swapchain may have been rebuilt since.
  VkExtent2D captureExtent{};
  bool exitAfterCapture = false;
  float viewDistanceScale = 1.0f;
  Buffer captureBuffer;

  ~Application() { context.waitIdle(); }

  void drawInterface();
  void requestScreenshot();
  void run(const std::string &initialScene, const std::string &initialEnvironment);
  void captureSwapchain(VkCommandBuffer command, std::uint32_t imageIndex);
  void writeCapture();
  void applyScheduledAction(const ScheduledAction &action);
  bool writeCaptureMetadata(const std::string &path, int framesRendered, float delta);
  std::string initialScenePath, initialEnvironmentPath;
};

// Describes the frame that was captured: the backend that actually recorded it.
bool Application::writeCaptureMetadata(const std::string &path, int framesRendered, float delta) {
  const RenderSettings &s = renderer.settings;
  const FrameStatistics &stats = renderer.statistics();
  const int active = stats.activeRenderer;
  pt::CaptureMetadata m;
  m.revision = BASALT_GIT_REVISION;
  m.dirtyAtConfigure = BASALT_GIT_DIRTY != 0;
  m.shaderCompiler = "msl2spirv-" BASALT_MSL2SPIRV_VERSION;
  m.scene = initialScenePath;
  try {
    m.sceneFileHash = pt::hashFile(initialScenePath);
  } catch (const std::exception &failure) {
    logError("{}", failure.what());
    return false;
  }
  m.environment = initialEnvironmentPath;
  if (!initialEnvironmentPath.empty() && !renderer.environment().procedural()) {
    const pt::EnvironmentSun &sun = renderer.environment().sun();
    m.notes.push_back({"environment_sun", !extractEnvironmentSun ? "kept in the image" : sun.found ? "extracted" : "none found"});
  }
  m.device = context.properties.deviceName;
  m.driverVersion = context.properties.driverVersion;
  m.apiVersion = context.properties.apiVersion;
  m.accelerationStructures = context.accelerationStructureSupported;
  m.rayQuery = context.rayQuerySupported;
  m.rayPipeline = context.rayPipelineSupported;
  const bool wavefront = stats.activeWavefront;
  m.renderer = active == 5 ? "gpu-ray-pipeline" : active == 4 ? "hybrid-ray-query" : active == 3 ? "gpu-own-bvh" :
               active == 2 ? "gpu-ray-query" : active == 1 ? "cpu" : "raster";
  m.intersector = active == 1 ? (s.cpuIntersector == 1 ? "embree" : s.cpuIntersector == 2 ? "quantized-bvh8-avx2" : "own-bvh") :
                  active == 3 ? "software-bvh-compute" : active == 5 ? "driver-built-hardware-as" :
                  active == 4 ? "raster-primary-hardware-ray-query" : active == 2 ? "hardware-ray-query" : "not-applicable";
  m.builder = active == 1 ? (s.cpuIntersector == 1 ? "embree" : "cpu-binned-sah") :
              active == 3 ? (s.pathBvhBuilder == 1 ? "gpu-lbvh" : "cpu-binned-sah") :
              active == 0 ? "not-applicable" : "driver";
  m.bvhLayout = active == 3 ? (s.pathBvhWidth == 1 ? "quantized-bvh4" : s.pathBvhWidth == 2 ? "quantized-bvh8" : "binary-float") :
                active == 1 ? (s.cpuIntersector == 2 ? "quantized-bvh8" : s.cpuIntersector == 1 ? "embree" : "binary-float") :
                "not-applicable";
  m.execution = active == 1 ? "cpu-tiles" : active == 0 ? "not-applicable" : active == 4 ? "hybrid-megakernel" :
                active == 5 ? (wavefront ? "wavefront-ray-pipeline" : "iterative-raygen") :
                wavefront ? "wavefront" : "megakernel";
  m.cpuThreads = active == 1 ? renderer.cpuThreadCount() : 0u;
  m.width = swapchain.extent().width;
  m.height = swapchain.extent().height;
  m.spp = stats.samples;
  m.targetSpp = static_cast<std::uint32_t>(std::max(0, s.pathTargetSamples));
  m.samplesPerDispatch = active >= 2 ? static_cast<std::uint32_t>(std::max(1, s.pathSamplesPerFrame)) : 0u;
  m.bounces = static_cast<std::uint32_t>(std::max(1, s.pathBounces));
  m.rouletteStart = pt::kRouletteStartBounce;
  m.strategy = s.pathStrategy == 1 ? "bsdf-only" : s.pathStrategy == 2 ? "light-only" : "mis";
  m.seed = s.pathSeed;
  m.fireflyClamp = std::max(0.0f, s.pathClamp);
  m.textureFilter = s.pathTextureFilter == 1 ? "ray-cone trilinear" : "level-zero bilinear";
  if (s.pathDiEstimator == 1 && s.renderer != 0 && s.renderer != 4) {
    static const char *reuse[] = {"none", "temporal", "spatial", "both"};
    m.diEstimator = "restir";
    m.restirReuse = reuse[std::clamp(s.pathRestirReuse, 0, 3)];
    m.restirCandidates = static_cast<std::uint32_t>(std::clamp(s.pathRestirCandidates, 1, 13));
    m.restirBias = s.pathRestirReuse == 0 ? "unbiased" : "visibility-only";
  }
  m.aperture = std::max(0.0f, s.pathAperture);
  m.focusDistance = s.pathFocusDistance > 0.0f ? s.pathFocusDistance : std::max(camera.distance, 1e-3f);
  const bool reconstructed = s.pathTemporal && active != 0 && !(active == 3 && s.pathBvhDiagnostic != 0);
  m.reconstruction = reconstructed ? "temporal-atrous" : "off";
  m.denoise = !denoisedLinearPath.empty() ? "oidn-automatic-device (separate output)" :
              active == 1 && s.pathDenoise && !s.pathTemporal ? "oidn (display only)" : "off";
  // The file format follows the extension: .exr is OpenEXR, anything else PFM (pt/ImageFile.h).
  auto kind = [](const std::string &path, const char *content) {
    return std::string(content) + (pt::isExrPath(path) ? "-exr" : "-pfm");
  };
  if (!linearPath.empty()) m.outputs.push_back({linearPath, kind(linearPath, "raw-linear-rgb")});
  if (!denoisedLinearPath.empty()) m.outputs.push_back({denoisedLinearPath, kind(denoisedLinearPath, "denoised-linear-rgb")});
  if (!albedoPath.empty()) m.outputs.push_back({albedoPath, kind(albedoPath, "first-hit-albedo-mean")});
  if (!normalPath.empty()) m.outputs.push_back({normalPath, kind(normalPath, "first-hit-normal-mean")});
  if (!capturePath.empty()) m.outputs.push_back({capturePath, "display-png"});
  if (active == 3) m.notes.push_back({"bvh_diagnostic", s.pathBvhDiagnostic == 1 ? "node-visits" :
                                                         s.pathBvhDiagnostic == 2 ? "triangle-tests" : "off"});
  if (wavefront) {
    m.notes.push_back({"wavefront_allocation", renderer.waveSubgroupAllocationActive() ? "subgroup" : "atomic"});
    m.notes.push_back({"wavefront_stages", renderer.waveFusionActive() ? "fused-intersect-shade" : "split"});
    m.measurements.push_back({"wavefront_capacity_request", std::max(0, s.pathWaveCapacity)});
  }
  if (active == 4) m.notes.push_back({"hybrid_comparison", s.pathComparison == 1 ? "rasterised" :
                                                            s.pathComparison == 2 ? "absolute-difference-x8" :
                                                            s.pathComparison == 3 ? "split" :
                                                            s.pathComparison == 4 ? "samples-per-pixel" : "traced"});
  m.measurements = {
      {"frames_rendered", framesRendered},
      {"frame_gpu_ms", stats.gpuMilliseconds},
      {"fresh_sample_frame_gpu_ms", stats.freshSampleGpuMilliseconds},
      {"frame_wall_delta_ms", delta * 1000.0f},
      {"cpu_paths_per_second", stats.pathsPerSecond},
      {"reconstruction_gpu_ms", stats.reconstructionMilliseconds},
      {"guide_host_copy_ms", stats.guideUploadMilliseconds},
      {"guide_transfer_gpu_ms", stats.guideTransferMilliseconds},
      {"reconstruction_buffer_bytes", static_cast<double>(stats.reconstructionBytes)},
      {"preview_scale", stats.previewScale},
      {"wavefront_queue_bytes", static_cast<double>(stats.wavefrontQueueBytes)},
      {"validation_enabled", context.validationEnabled ? 1.0 : 0.0},
  };
  if (const pt::EnvironmentSun &sun = renderer.environment().sun(); sun.found) {
    m.measurements.push_back({"environment_sun_irradiance", pt::ptLuminance(sun.irradiance)});
    m.measurements.push_back({"environment_sun_radius_degrees", sun.angularRadius * 180.0 / 3.14159265358979323846});
    m.measurements.push_back({"environment_sun_share", sun.share});
  }
  if (const GpuProfiler *profile = active >= 2 ? renderer.finishProfiling() : nullptr) {
    m.series.push_back({"fresh_trace_gpu_ms", profile->traceSeries()});
    const GpuProfile &latest = profile->latest();
    m.measurements.push_back({"trace_gpu_ms", latest.traceMilliseconds});
    m.measurements.push_back({"wavefront_refused_reservations", static_cast<double>(profile->overflowTotal())});
    if (profile->overflowTotal() > 0) {
      logError("wavefront queues refused {} reservations: paths were lost", profile->overflowTotal());
      captureFailed = true;
    }
    if (s.pathProfile && latest.valid) {
      for (std::size_t i = 0; i < latest.stageMilliseconds.size(); ++i)
        m.measurements.push_back({std::string("stage_ms_") + gpuStageName(static_cast<GpuStage>(i)),
                                  latest.stageMilliseconds[i]});
      m.measurements.push_back({"profiled_dispatches", latest.dispatches});
      m.measurements.push_back({"profile_truncated", latest.truncated ? 1.0 : 0.0});
      m.series.push_back({"live_paths_by_bounce", std::vector<double>(latest.liveByBounce.begin(), latest.liveByBounce.end())});
      m.series.push_back({"shadow_rays_by_bounce", std::vector<double>(latest.shadowByBounce.begin(), latest.shadowByBounce.end())});
    }
  }
  return pt::writeCaptureMetadata(path, m);
}

// Scripted lifecycle transitions for tests: every change goes through the same settings
// the interface edits, with frames still in flight.
void Application::applyScheduledAction(const ScheduledAction &action) {
  RenderSettings &s = renderer.settings;
  const std::string &v = action.value;
  if (action.key == "renderer") {
    const int kind = rendererFromName(v);
    if (kind == -2) throw std::runtime_error("unknown scheduled renderer " + v);
    if ((kind == 2 || kind == 4) && !renderer.rayTracingAvailable())
      throw std::runtime_error("the scheduled renderer requires Vulkan ray queries on the selected device");
    if (kind == 5 && !renderer.rayPipelineAvailable())
      throw std::runtime_error("the scheduled renderer requires VK_KHR_ray_tracing_pipeline on the selected device");
    s.renderer = kind;
  } else if (action.key == "execution") {
    if (v != "megakernel" && v != "iterative" && v != "wavefront")
      throw std::runtime_error("unknown scheduled execution " + v);
    s.pathExecution = v == "wavefront" ? 1 : 0;
  } else if (action.key == "temporal") {
    s.pathTemporal = v == "on";
  } else if (action.key == "builder") {
    s.pathBvhBuilder = v == "gpu" ? 1 : 0;
    if (s.pathBvhBuilder == 1 && s.pathBvhWidth != 0)
      throw std::runtime_error("the GPU LBVH builder produces the binary layout only");
  } else if (action.key == "layout") {
    s.pathBvhWidth = v == "bvh4" ? 1 : v == "bvh8" ? 2 : 0;
    if (s.pathBvhWidth != 0) s.pathBvhBuilder = 0;  // as the interface does
  } else if (action.key == "spf") {
    s.pathSamplesPerFrame = std::max(1, std::atoi(v.c_str()));
  } else if (action.key == "yaw") {
    camera.yaw += radians(static_cast<float>(std::atof(v.c_str())));  // a camera cut or step
  } else if (action.key == "scene") {
    loadScene(v);
  } else if (action.key == "resize") {
    unsigned w = 0, h = 0;
    if (std::sscanf(v.c_str(), "%ux%u", &w, &h) != 2 || w == 0 || h == 0)
      throw std::runtime_error("scheduled resize needs WIDTHxHEIGHT, not " + v);
    RECT rect{0, 0, static_cast<LONG>(w), static_cast<LONG>(h)};
    AdjustWindowRectEx(&rect, static_cast<DWORD>(GetWindowLongW(window.handle(), GWL_STYLE)), FALSE,
                       static_cast<DWORD>(GetWindowLongW(window.handle(), GWL_EXSTYLE)));
    SetWindowPos(window.handle(), nullptr, 0, 0, rect.right - rect.left, rect.bottom - rect.top,
                 SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
  } else {
    throw std::runtime_error("unknown scheduled action " + action.key);
  }
  logInfo("frame {}: applied {}={}", action.frame, action.key, v);
}

// Copies the presented image before it goes to the presentation engine.
void Application::captureSwapchain(VkCommandBuffer command, std::uint32_t imageIndex) {
  const VkExtent2D extent = swapchain.extent();
  captureExtent = extent;
  const VkDeviceSize bytes = static_cast<VkDeviceSize>(extent.width) * extent.height * 4;
  if (!captureBuffer || captureBuffer.size < bytes) {
    captureBuffer = Buffer(context, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO,
                           VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                               VMA_ALLOCATION_CREATE_MAPPED_BIT,
                           "capture");
  }

  VkImageMemoryBarrier2 toSource{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  toSource.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  toSource.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
  toSource.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  toSource.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
  toSource.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  toSource.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  toSource.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toSource.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toSource.image = swapchain.image(imageIndex);
  toSource.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.imageMemoryBarrierCount = 1;
  dependency.pImageMemoryBarriers = &toSource;
  vkCmdPipelineBarrier2(command, &dependency);

  VkBufferImageCopy region{};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = {extent.width, extent.height, 1};
  vkCmdCopyImageToBuffer(command, swapchain.image(imageIndex), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         captureBuffer.handle, 1, &region);

  VkImageMemoryBarrier2 back = toSource;
  back.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  back.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
  back.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  back.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
  back.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  back.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  dependency.pImageMemoryBarriers = &back;
  vkCmdPipelineBarrier2(command, &dependency);
}

void Application::requestScreenshot() {
  // A scripted capture that has not happened yet keeps its name and its exit.
  if (exitAfterCapture && captureFrame >= 0) return;
  const auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
  std::string stamp;
  try {
    stamp = std::format("{:%Y%m%d-%H%M%S}", std::chrono::zoned_time{std::chrono::current_zone(), now});
  } catch (const std::exception &) {
    stamp = std::format("{:%Y%m%d-%H%M%S}", now); // No time zone database: UTC will do.
  }
  capturePath = "basalt-" + stamp + ".png";
  for (int n = 2; std::filesystem::exists(capturePath) && n < 100; ++n)
    capturePath = "basalt-" + stamp + "-" + std::to_string(n) + ".png";
  captureFrame = -2; // The next frame rendered.
  exitAfterCapture = false;
}

void Application::writeCapture() {
  if (!captureBuffer || capturePath.empty()) return;
  context.waitIdle();
  const VkExtent2D extent = captureExtent;
  const std::size_t count = static_cast<std::size_t>(extent.width) * extent.height;
  std::vector<std::uint8_t> rgba(count * 4);
  const auto *source = static_cast<const std::uint8_t *>(captureBuffer.mapped);
  const bool swizzle = swapchain.format() == VK_FORMAT_B8G8R8A8_SRGB ||
                       swapchain.format() == VK_FORMAT_B8G8R8A8_UNORM;
  for (std::size_t i = 0; i < count; ++i) {
    rgba[i * 4 + 0] = source[i * 4 + (swizzle ? 2 : 0)];
    rgba[i * 4 + 1] = source[i * 4 + 1];
    rgba[i * 4 + 2] = source[i * 4 + (swizzle ? 0 : 2)];
    rgba[i * 4 + 3] = 255;
  }
  if (stbi_write_png(capturePath.c_str(), static_cast<int>(extent.width),
                     static_cast<int>(extent.height), 4, rgba.data(),
                     static_cast<int>(extent.width) * 4))
    logInfo("wrote {} ({} by {}), {:.2f} ms CPU, {:.2f} ms GPU, {} lights", capturePath, extent.width,
            extent.height, frameMilliseconds, renderer.statistics().gpuMilliseconds,
            renderer.settings.testLights);
  else {
    logError("could not write {}", capturePath);
    captureFailed = true;
  }
}

// The sun an .hdr gave up when it was loaded (Environment::sun) becomes the renderers' sun:
// its direction (unless keepDirection), colour, irradiance and angular radius. An image
// without one leaves no sun, so it lights the scene alone in every renderer.
static void adoptEnvironmentSun(Renderer &renderer, bool keepDirection) {
  RenderSettings &settings = renderer.settings;
  const Environment &environment = renderer.environment();
  if (!keepDirection) {
    const Vec3 direction = environment.brightestDirection();
    settings.sunAzimuth = std::atan2(direction.x, direction.z);
    settings.sunElevation = std::asin(std::clamp(direction.y, -1.0f, 1.0f));
  }
  const pt::EnvironmentSun &sun = environment.sun();
  const float peak = std::max(sun.irradiance.x, std::max(sun.irradiance.y, sun.irradiance.z));
  if (!sun.found || !(peak > 0.0f)) {
    settings.sunIntensity = 0.0f;
    return;
  }
  settings.sunColor = {sun.irradiance.x / peak, sun.irradiance.y / peak, sun.irradiance.z / peak};
  settings.sunIntensity = peak;
  settings.sunAngularRadius = sun.angularRadius;
}

// The procedural sky's own sun, for when an .hdr's (or its absence) is left behind.
static void restoreDefaultSun(RenderSettings &settings) {
  const RenderSettings defaults;
  settings.sunColor = defaults.sunColor;
  settings.sunIntensity = defaults.sunIntensity;
  settings.sunAngularRadius = defaults.sunAngularRadius;
}

void Application::drawInterface() {
  const ImGuiViewport *viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + 12, viewport->WorkPos.y + 12),
                          ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(360, 720), ImGuiCond_FirstUseEver);
  ImGui::Begin("Basalt");

  RenderSettings &settings = renderer.settings;
  const FrameStatistics &stats = renderer.statistics();

  const char *rendererNames[] = {"Rasteriser", "Path traced (CPU)",
                                 "Path traced (GPU, inline ray queries)",
                                 "Path traced (GPU, own BVH compute)",
                                 "Hybrid (raster primary, inline ray queries)",
                                 "Path traced (GPU, full ray pipeline)"};
  if (ImGui::BeginCombo("Renderer", rendererNames[std::clamp(settings.renderer, 0, 5)])) {
    for (int choice = 0; choice < 6; ++choice) {
      const bool available = choice == 5 ? renderer.rayPipelineAvailable() :
                             (choice != 2 && choice != 4) || renderer.rayTracingAvailable();
      ImGui::BeginDisabled(!available);
      if (ImGui::Selectable(rendererNames[choice], settings.renderer == choice) && available)
        settings.renderer = choice;
      ImGui::EndDisabled();
      if (!available && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip(choice == 5 ? "Needs VK_KHR_ray_tracing_pipeline with acceleration structures on this device"
                                      : "Needs VK_KHR_ray_query with acceleration structures on this device");
    }
    ImGui::EndCombo();
  }
  if (!stats.unavailableReason.empty())
    ImGui::TextColored(ImVec4(1, 0.4f, 0.3f, 1), "Rasterising instead: %s", stats.unavailableReason.c_str());
  if (settings.renderer != 0 && ImGui::CollapsingHeader("Path tracing", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Text("%u samples per pixel", stats.samples);
    if (stats.previewScale > 1) ImGui::TextDisabled("CPU preview: 1/%u resolution per axis; no full-resolution SPP yet",
                                                  stats.previewScale);
    if (stats.pathsPerSecond > 0.0) {
      ImGui::SameLine();
      ImGui::TextDisabled("%.2f M paths/s", stats.pathsPerSecond * 1e-6);
    }
    ImGui::SliderInt("Bounces", &settings.pathBounces, 1, 32);
    ImGui::BeginDisabled(settings.renderer == 3 && settings.pathBvhDiagnostic != 0);
    ImGui::Checkbox("Temporal reconstruction", &settings.pathTemporal);
    ImGui::EndDisabled();
    if (settings.pathTemporal)
      ImGui::TextDisabled("Filter %.2f ms; CPU guide copy %.2f ms; %.1f MiB history/buffers",
          stats.reconstructionMilliseconds, stats.guideUploadMilliseconds,
          static_cast<double>(stats.reconstructionBytes) / (1024.0 * 1024.0));
    if (settings.renderer >= 2) {
      // ReSTIR DI reuses one path per pixel per frame.
      if (settings.pathDiEstimator == 1 && settings.renderer != 4) settings.pathSamplesPerFrame = 1;
      ImGui::BeginDisabled(settings.pathDiEstimator == 1 && settings.renderer != 4);
      ImGui::SliderInt("Samples per frame", &settings.pathSamplesPerFrame, 1, 64);
      ImGui::EndDisabled();
    }
    if (settings.renderer == 2 || settings.renderer == 3)
      ImGui::Combo("Execution", &settings.pathExecution, "Megakernel (default)\0Wavefront queues\0");
    if (settings.renderer == 5)
      ImGui::Combo("Execution", &settings.pathExecution,
                   "Iterative ray generation (default)\0Wavefront queues (ray pipeline)\0");
    if (settings.pathExecution == 1 && (settings.renderer == 2 || settings.renderer == 3 || settings.renderer == 5) &&
        ImGui::TreeNode("Wavefront settings")) {
      bool automatic = settings.pathWaveCapacity == 0;
      if (ImGui::Checkbox("Automatic batch capacity", &automatic))
        settings.pathWaveCapacity = automatic ? 0 : static_cast<int>(pt::kWavefrontDefaultPaths);
      if (!automatic)
        ImGui::SliderInt("Paths per batch", &settings.pathWaveCapacity, 65536, 1 << 24, "%d",
                         ImGuiSliderFlags_Logarithmic);
      ImGui::TextDisabled("%.1f MiB of queues", static_cast<double>(stats.wavefrontQueueBytes) / (1024.0 * 1024.0));
      const char *allocations[] = {"Automatic", "One atomic per subgroup", "One atomic per path"};
      if (ImGui::BeginCombo("Queue allocation", allocations[std::clamp(settings.pathWaveAllocation, 0, 2)])) {
        for (int choice = 0; choice < 3; ++choice) {
          const bool available = choice != 1 || renderer.waveSubgroupAllocationSupportedOnDevice();
          ImGui::BeginDisabled(!available);
          if (ImGui::Selectable(allocations[choice], settings.pathWaveAllocation == choice) && available)
            settings.pathWaveAllocation = choice;
          ImGui::EndDisabled();
          if (!available && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("Needs subgroup arithmetic and ballot in compute shaders");
        }
        ImGui::EndCombo();
      }
      ImGui::TreePop();
    }
    if (settings.renderer == 3) {
      const char *builders[] = {"CPU binned SAH", "GPU stable LBVH (binary only)"};
      if (ImGui::BeginCombo("BVH builder", builders[std::clamp(settings.pathBvhBuilder, 0, 1)])) {
        for (int choice = 0; choice < 2; ++choice) {
          // The GPU builder emits the binary layout only; wide layouts need CPU SAH.
          const bool available = choice == 0 || settings.pathBvhWidth == 0;
          ImGui::BeginDisabled(!available);
          if (ImGui::Selectable(builders[choice], settings.pathBvhBuilder == choice) && available)
            settings.pathBvhBuilder = choice;
          ImGui::EndDisabled();
          if (!available && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("The GPU LBVH builder produces binary nodes; choose the binary layout first");
        }
        ImGui::EndCombo();
      }
      if (ImGui::Combo("BVH layout", &settings.pathBvhWidth,
                       "Binary float bounds\0BVH4 quantized bounds\0BVH8 quantized bounds\0") &&
          settings.pathBvhWidth != 0)
        settings.pathBvhBuilder = 0;
      if (settings.pathBvhWidth != 0)
        ImGui::TextDisabled("Wide layouts currently use the CPU SAH builder");
      if (ImGui::Combo("BVH diagnostic", &settings.pathBvhDiagnostic,
                       "Shaded\0Node visits\0Triangle tests\0") && settings.pathBvhDiagnostic != 0)
        settings.pathTemporal = false;
      if (settings.pathBvhDiagnostic == 1) ImGui::TextDisabled("False colour: red at 128+ node visits per path");
      if (settings.pathBvhDiagnostic == 2) ImGui::TextDisabled("False colour: red at 64+ triangle tests per path");
    }
    if (settings.renderer == 4)
      ImGui::Combo("Comparison", &settings.pathComparison,
                   "Traced\0Rasterised\0Difference x8\0Split raster/traced\0Samples per pixel\0");
    ImGui::SliderInt("Stop at", &settings.pathTargetSamples, 0, 4096, "%d samples (0 = never)");
    ImGui::BeginDisabled(settings.renderer == 4);
    ImGui::SliderFloat("Aperture radius", &settings.pathAperture, 0.0f, 0.5f, "%.4f (0 = pinhole)",
                       ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Focus distance", &settings.pathFocusDistance, 0.0f, 100.0f, "%.2f (0 = orbit target)",
                       ImGuiSliderFlags_Logarithmic);
    ImGui::EndDisabled();
    if (settings.renderer == 4 && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
      ImGui::SetTooltip("Depth of field needs traced primary rays; hybrid rasterises them");
    ImGui::Combo("Texture filtering", &settings.pathTextureFilter, "Level zero (V6)\0Ray cones\0");
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Ray cones pick each texture's mip level from the path's footprint: less aliasing\n"
                        "far away and after rough bounces. Level zero reads full resolution, as V6 did.");
    ImGui::BeginDisabled(settings.renderer == 4);
    ImGui::Combo("Direct light", &settings.pathDiEstimator, "Next-event estimation\0ReSTIR DI\0");
    if (settings.renderer != 4 && ImGui::IsItemHovered())
      ImGui::SetTooltip("ReSTIR DI resamples many light candidates at the first hit and reuses them\n"
                        "across frames and neighbours. One path per pixel per frame; reuse is\n"
                        "unbiased up to visibility.");
    if (settings.pathDiEstimator == 1) {
      bool temporal = (settings.pathRestirReuse & 1) != 0, spatial = (settings.pathRestirReuse & 2) != 0;
      ImGui::Checkbox("Temporal reuse", &temporal);
      ImGui::SameLine();
      ImGui::Checkbox("Spatial reuse", &spatial);
      settings.pathRestirReuse = (temporal ? 1 : 0) | (spatial ? 2 : 0);
      ImGui::SliderInt("Candidates", &settings.pathRestirCandidates, 1, 13);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Initial light candidates M per pixel and frame");
    }
    ImGui::EndDisabled();
    if (settings.renderer == 4 && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
      ImGui::SetTooltip("ReSTIR DI needs traced primary rays; hybrid rasterises them");
    ImGui::Combo("Sampling", &settings.pathStrategy, "Multiple importance\0BSDF only\0Lights only\0");
    ImGui::SliderFloat("Firefly clamp", &settings.pathClamp, 0.0f, 100.0f, "%.1f (0 = off)",
                       ImGuiSliderFlags_Logarithmic);
    if (settings.renderer == 1) {
      const char *intersectors[] = {"Own binary BVH", "Intel Embree", "Quantized BVH8 (AVX2)"};
      if (ImGui::BeginCombo("Intersector", intersectors[std::clamp(settings.cpuIntersector, 0, 2)])) {
        for (int choice = 0; choice < 3; ++choice) {
          const bool available = choice == 0 || (choice == 1 && pt::EmbreeScene::available()) ||
                                 (choice == 2 && pt::cpuAvx2Available());
          ImGui::BeginDisabled(!available);
          if (ImGui::Selectable(intersectors[choice], settings.cpuIntersector == choice) && available)
            settings.cpuIntersector = choice;
          ImGui::EndDisabled();
        }
        ImGui::EndCombo();
      }
    }
    ImGui::BeginDisabled(!pt::Denoiser::available() || settings.renderer != 1 || settings.pathTemporal);
    ImGui::Checkbox("Denoise (Open Image Denoise)", &settings.pathDenoise);
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
      ImGui::SetTooltip("%s", !pt::Denoiser::available() ? "This build has no Open Image Denoise" :
                              settings.renderer != 1 ? "Interactive OIDN is CPU-renderer only; GPU tracers denoise "
                                                       "captures with --denoised-pfm. Raw output is never denoised." :
                              settings.pathTemporal ? "Temporal reconstruction replaces OIDN on screen" :
                                                      "Display only: raw accumulation and PFM stay undenoised");
    if (settings.renderer == 1 && settings.pathDenoise && !settings.pathTemporal) {
      const std::string denoiserState = renderer.denoiserStatus();
      ImGui::SameLine();
      ImGui::TextDisabled("%s, %u samples", denoiserState.empty() ? "starting" : denoiserState.c_str(), renderer.denoisedSamples());
    }
    if (const pt::BvhStatistics *bvh = renderer.cpuBvh()) {
      if (settings.renderer == 3)
        ImGui::TextWrapped("GPU BVH (%s): %u instances, %u triangles, %u + %u nodes, built %s in %.0f ms",
                           settings.pathBvhWidth == 1 ? "quantized BVH4" : settings.pathBvhWidth == 2 ? "quantized BVH8" : "binary",
                           bvh->instances, bvh->triangles, bvh->topNodes, bvh->bottomNodes,
                           settings.pathBvhBuilder == 1 ? "by GPU LBVH" : "on CPU (SAH)", bvh->milliseconds);
      else
        ImGui::TextWrapped("BVH: %u instances, %u triangles, %u + %u nodes, built in %.0f ms on %u threads",
                           bvh->instances, bvh->triangles, bvh->topNodes, bvh->bottomNodes, bvh->milliseconds,
                           renderer.cpuThreadCount());
    }
    ImGui::TextDisabled(renderer.environment().procedural() ? "Sun: analytic, the sky's own disc left out"
                        : settings.sunIntensity > 0.0f ? "Sun: analytic, moved out of the environment"
                                                       : "Sun: none; the environment lights alone");
  }

  if (ImGui::CollapsingHeader("Frame", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Text("%s", context.info.name.c_str());
    ImGui::Text("%.2f ms CPU (%.0f fps)", frameMilliseconds,
                frameMilliseconds > 0.0f ? 1000.0f / frameMilliseconds : 0.0f);
    if (stats.gpuMilliseconds > 0.0f) ImGui::Text("%.2f ms GPU", stats.gpuMilliseconds);
    ImGui::Checkbox("Accumulate while still", &settings.accumulate);
    if (settings.accumulate) {
      ImGui::SameLine();
      ImGui::TextDisabled("%u frames", stats.accumulatedFrames);
    }
    ImGui::PlotLines("##frames", frameHistory, IM_ARRAYSIZE(frameHistory), frameCursor, nullptr,
                     0.0f, 33.0f, ImVec2(0, 48));
    ImGui::Text("%u draws, %u shadow draws", stats.drawCalls, stats.shadowDrawCalls);
    ImGui::Text("%u primitives visible, %u triangles", stats.visiblePrimitives, stats.triangles);
    if (context.sawValidationError)
      ImGui::TextColored(ImVec4(1, 0.4f, 0.3f, 1), "a validation error was reported: see the log");
  }

  if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (Scene *scene = renderer.scene()) {
      ImGui::TextWrapped("%s", std::filesystem::path(scene->sourcePath).filename().string().c_str());
      ImGui::Text("%zu primitives, %u triangles", scene->primitives.size(), scene->triangleCount);
      ImGui::Text("%zu materials, %zu textures (%.1f MB)", scene->materials.size(),
                  scene->textures.size(), static_cast<double>(scene->textureBytes) / (1024 * 1024));
      ImGui::Text("%zu punctual lights", scene->lights.size());
    }
    if (ImGui::Button("Open glTF...", ImVec2(-1, 0))) {
      const std::string path = openFileDialog(
          window.handle(), L"glTF\0*.gltf;*.glb\0All files\0*.*\0\0", L"Open a glTF model");
      if (!path.empty()) loadScene(path);
    }
    ImGui::SliderInt("Test lights", &settings.testLights, 0, 256);
    if (settings.testLights > 0) {
      ImGui::BeginDisabled(!renderer.rayTracingAvailable());
      ImGui::Checkbox("Lights cast shadows", &settings.lightShadows);
      ImGui::EndDisabled();
      ImGui::Checkbox("Cull lights into a grid", &settings.clusteredLights);
    }
    if (ImGui::Button("Reset to the built-in cube", ImVec2(-1, 0))) {
      renderer.setScene(createDefaultScene(context, uploader));
      renderer.applyGroundMaterial();
      camera.frame(renderer.scene()->bounds);
    }
    if (!status.empty()) ImGui::TextWrapped("%s", status.c_str());
  }

  if (ImGui::CollapsingHeader("Sun and sky", ImGuiTreeNodeFlags_DefaultOpen)) {
    bool sunMoved = false;
    float azimuth = degrees(settings.sunAzimuth), elevation = degrees(settings.sunElevation);
    sunMoved |= ImGui::SliderFloat("Azimuth", &azimuth, -180.0f, 180.0f, "%.0f deg");
    sunMoved |= ImGui::SliderFloat("Elevation", &elevation, -10.0f, 89.0f, "%.0f deg");
    settings.sunAzimuth = radians(azimuth);
    settings.sunElevation = radians(elevation);
    ImGui::ColorEdit3("Sun colour", &settings.sunColor.x);
    ImGui::SliderFloat("Sun intensity", &settings.sunIntensity, 0.0f, 20.0f);
    ImGui::SliderFloat("IBL intensity", &settings.iblIntensity, 0.0f, 4.0f);
    ImGui::Checkbox("Draw the sky", &settings.drawSky);
    ImGui::SameLine();
    ImGui::Checkbox("Ground", &settings.groundPlane);
    if (settings.groundPlane) {
      int preset = groundPresetIndex(settings);
      bool groundChanged = false;
      if (ImGui::Combo("Ground look", &preset, groundPresetNames, IM_ARRAYSIZE(groundPresetNames))) {
        if (preset < kGroundPresetCount) {
          const GroundPreset &chosen = kGroundPresets[preset];
          settings.groundColor = chosen.colour;
          settings.groundRoughness = chosen.roughness;
          settings.groundMetallic = chosen.metallic;
          groundChanged = true;
        }
      }
      groundChanged |= ImGui::ColorEdit3("Ground colour", &settings.groundColor.x);
      groundChanged |= ImGui::SliderFloat("Ground roughness", &settings.groundRoughness, 0.02f, 1.0f);
      groundChanged |= ImGui::SliderFloat("Ground metallic", &settings.groundMetallic, 0.0f, 1.0f);
      if (groundChanged) renderer.applyGroundMaterial();
    }

    Environment &environment = renderer.environment();
    ImGui::TextWrapped("Environment: %s", environment.name().c_str());
    if (!environment.procedural()) {
      const pt::EnvironmentSun &sun = environment.sun();
      if (sun.found)
        ImGui::TextDisabled("Its sun: %.2f deg across, %.1f%% of its light, now the sun above",
                            degrees(sun.angularRadius) * 2.0f, sun.share * 100.0f);
      else
        ImGui::TextDisabled(environment.sunExtractionEnabled() ? "No sun stands out in it: it lights alone"
                                                               : "Its sun stays in the image (--environment-sun keep)");
    }
    bool skyChanged = false;
    if (environment.procedural()) {
      skyChanged |= ImGui::SliderFloat("Haze", &settings.skyTurbidity, 1.0f, 10.0f);
      skyChanged |= ImGui::SliderFloat("Sky brightness", &settings.skyIntensity, 0.0f, 4.0f);
    }
    if (ImGui::Button("Open an HDR environment...", ImVec2(-1, 0))) {
      const std::string path = openFileDialog(window.handle(), L"Radiance HDR\0*.hdr\0All files\0*.*\0\0",
                                              L"Open an environment map");
      if (!path.empty()) {
        environment.load(path, environment.sunExtractionEnabled());
        renderer.environmentChanged();
        adoptEnvironmentSun(renderer, false);
      }
    }
    if (!environment.procedural() && ImGui::Button("Reset the sun to the environment's", ImVec2(-1, 0)))
      adoptEnvironmentSun(renderer, false);
    if (ImGui::Button("Back to the procedural sky", ImVec2(-1, 0))) {
      if (!environment.procedural()) {
        environment.load("");
        renderer.environmentChanged();
        restoreDefaultSun(settings);
      }
      skyChanged = true;
    }
    if (skyChanged || (sunMoved && environment.procedural())) renderer.rebakeProceduralSky();
  }

  const bool traced = renderer.rayTracingAvailable();
  auto tracedNote = [&] {
    if (!traced)
      ImGui::TextDisabled(context.rayTracingSupported ? "no acceleration structure for this scene"
                                                      : "ray queries are not available on this device");
  };

  if (ImGui::CollapsingHeader("Shadows")) {
    ImGui::Checkbox("Enabled", &settings.shadowsEnabled);
    ImGui::BeginDisabled(!traced);
    ImGui::Combo("Shadow method", &settings.shadowMode, "Cascaded shadow maps\0Ray traced\0");
    ImGui::EndDisabled();
    tracedNote();
    if (traced && settings.shadowMode == 1) {
      ImGui::SliderInt("Rays per pixel", &settings.shadowSamples, 1, 16);
      float radius = degrees(settings.sunAngularRadius);
      ImGui::SliderFloat("Sun angular radius", &radius, 0.0f, 5.0f, "%.2f deg");
      settings.sunAngularRadius = radians(radius);
    } else {
      ImGui::SliderFloat("Depth bias", &settings.shadowDepthBias, 0.0f, 0.01f, "%.4f");
      ImGui::SliderFloat("Normal bias", &settings.shadowNormalBias, 0.0f, 0.2f, "%.3f");
      ImGui::SliderFloat("Softness", &settings.shadowSoftness, 0.0f, 4.0f);
      ImGui::SliderFloat("Cascade split blend", &settings.cascadeSplitLambda, 0.0f, 1.0f);
      ImGui::SliderFloat("Distance", &settings.shadowDistance, 0.0f, 500.0f, "%.0f (0 = automatic)");
      ImGui::Checkbox("Freeze the cascades", &settings.freezeCascades);
    }
  }

  if (ImGui::CollapsingHeader("Ambient occlusion")) {
    ImGui::BeginDisabled(!traced);
    ImGui::Combo("Occlusion method", &settings.occlusionMode,
                 "Material map\0Ray traced contact\0Ray traced sky visibility\0");
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
      ImGui::SetTooltip("Contact: darkens creases, fading over a short radius.\n"
                        "Sky visibility: the share of the sky each point sees, as the path tracers light it;\n"
                        "matches them under an environment without a strong sun.");
    tracedNote();
    if (traced && (settings.occlusionMode == 1 || settings.occlusionMode == 2)) {
      ImGui::SliderInt("Rays per pixel##ao", &settings.occlusionSamples, 1, 16);
      const bool sky = settings.occlusionMode == 2;
      const float automatic = sky ? renderer.occlusionReach() : renderer.sceneScale() * 0.05f;
      char format[64];
      std::snprintf(format, sizeof(format), "%%.2f (0 = automatic, %.2f)", automatic);
      ImGui::SliderFloat(sky ? "Reach" : "Radius", &settings.occlusionRadius, 0.0f,
                         std::max(10.0f, automatic * 2.0f), format, ImGuiSliderFlags_Logarithmic);
    }
  }

  if (ImGui::CollapsingHeader("Reflections", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (traced) {
      ImGui::Combo("Reflection method", &settings.reflectionMode,
                   "Environment only\0Screen space\0Ray traced\0");
    } else {
      int mode = std::min(settings.reflectionMode, 1);
      if (ImGui::Combo("Reflection method", &mode, "Environment only\0Screen space\0"))
        settings.reflectionMode = mode;
      tracedNote();
    }
    if (settings.reflectionMode == 1) {
      ImGui::SliderInt("March steps", &settings.reflectionSteps, 8, 128);
      ImGui::SliderFloat("March distance", &settings.reflectionDistance, 0.0f, 100.0f,
                         "%.1f (0 = automatic)", ImGuiSliderFlags_Logarithmic);
      ImGui::SliderFloat("Surface thickness", &settings.reflectionThickness, 0.0f, 2.0f,
                         "%.3f (0 = automatic)", ImGuiSliderFlags_Logarithmic);
    } else if (settings.reflectionMode == 2) {
      ImGui::SliderFloat("Ray distance", &settings.reflectionDistance, 0.0f, 100.0f,
                         "%.1f (0 = automatic)", ImGuiSliderFlags_Logarithmic);
    }
  }

  if (ImGui::CollapsingHeader("Post processing")) {
    ImGui::SliderFloat("Exposure", &settings.exposure, 0.05f, 8.0f, "%.2f",
                       ImGuiSliderFlags_Logarithmic);
    ImGui::Combo("Tone map", &settings.tonemap, "ACES\0Reinhard\0");
    if (settings.tonemap == 1) ImGui::SliderFloat("White point", &settings.whitePoint, 1.0f, 16.0f);
    ImGui::SliderFloat("Bloom", &settings.bloomStrength, 0.0f, 0.5f, "%.3f");
    ImGui::SliderFloat("Bloom threshold", &settings.bloomThreshold, 0.0f, 8.0f);
    ImGui::SliderFloat("Vignette", &settings.vignette, 0.0f, 2.0f);
    ImGui::SliderFloat("Grain", &settings.grain, 0.0f, 0.2f, "%.3f");
    ImGui::SliderFloat("Sharpen", &settings.sharpen, 0.0f, 2.0f);
    ImGui::Combo("Anti-aliasing", &settings.antialiasing, "None\0FXAA\0Temporal\0");
    if (settings.antialiasing == 2)
      ImGui::SliderFloat("Temporal feedback", &settings.temporalFeedback, 0.02f, 1.0f, "%.2f");
  }

  if (ImGui::CollapsingHeader("Debug")) {
    ImGui::Combo("View", &settings.debugView,
                 "Shaded\0Base colour\0Normal\0Metallic\0Roughness\0Occlusion\0Shadow\0Cascades\0"
                 "Emissive\0Texture coordinates\0Reflection weight\0Reflection\0"
                 "Reflection confidence\0Geometric normal\0Coverage class\0Surface identity\0");
    ImGui::Checkbox("Wireframe", &settings.wireframe);
    ImGui::Checkbox("Frustum culling", &settings.frustumCulling);
    if (ImGui::Checkbox("Vertical sync", &settings.vsync))
      swapchain.recreate(window.width(), window.height(), settings.vsync);
    ImGui::Checkbox("Log", &showLog);
    ImGui::Checkbox("ImGui demo", &showDemo);
    if (ImGui::Button("Save a screenshot (F12)", ImVec2(-1, 0))) requestScreenshot();
    ImGui::TextDisabled("F1 hides the interface");
  }

  if (ImGui::CollapsingHeader("Camera")) {
    ImGui::Text("Orbit: left drag. Pan: middle drag. Zoom: wheel.");
    ImGui::Text("Fly: hold right mouse, then WASD, Q and E.");
    if (ImGui::Button("Frame the scene", ImVec2(-1, 0)) && renderer.scene())
      camera.frame(renderer.scene()->bounds);
    ImGui::SliderFloat("Field of view", &camera.fieldOfView, radians(20.0f), radians(110.0f),
                       "%.2f rad");
    ImGui::SliderFloat("Fly speed", &camera.flySpeed, 0.01f, 100.0f, "%.2f",
                       ImGuiSliderFlags_Logarithmic);
  }

  ImGui::Separator();
  ImGui::TextDisabled("Shaders: Metal Shading Language, compiled by msl2spirv");
  ImGui::End();

  if (showLog) {
    ImGui::SetNextWindowSize(ImVec2(760, 260), ImGuiCond_FirstUseEver);
    ImGui::Begin("Log", &showLog);
    for (const LogEntry &entry : logEntries()) {
      const ImVec4 colour = entry.level == LogEntry::Level::ErrorLevel ? ImVec4(1, 0.4f, 0.35f, 1)
                            : entry.level == LogEntry::Level::Warning  ? ImVec4(1, 0.85f, 0.4f, 1)
                                                                       : ImVec4(0.8f, 0.8f, 0.8f, 1);
      ImGui::TextColored(colour, "%s", entry.text.c_str());
    }
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0f) ImGui::SetScrollHereY(1.0f);
    ImGui::End();
  }
  if (showDemo) ImGui::ShowDemoWindow(&showDemo);
}

void Application::run(const std::string &initialScene, const std::string &initialEnvironment) {
  initialScenePath = initialScene;
  initialEnvironmentPath = initialEnvironment;
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  io.IniFilename = nullptr;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  applyDarkStyle();
  ImGui_ImplWin32_Init(window.handle());
  ui = std::make_unique<UiPass>(context, uploader, swapchain.format());

  if (!initialEnvironment.empty()) {
    renderer.environment().load(initialEnvironment, extractEnvironmentSun);
    adoptEnvironmentSun(renderer, fixedSun);
  }
  // The procedural sky was baked for the default sun; a sun given on the command line needs its own.
  if (fixedSun) renderer.rebakeProceduralSky();
  if (!initialScene.empty()) loadScene(initialScene);
  else renderer.applyGroundMaterial();
  if (renderer.scene()) {
    const float yaw = camera.yaw, pitch = camera.pitch;
    camera.frame(renderer.scene()->bounds);
    camera.yaw = yaw;
    camera.pitch = pitch;
    camera.distance *= viewDistanceScale;
  }

  auto previous = std::chrono::high_resolution_clock::now();
  int framesRendered = 0;
  while (window.pumpMessages()) {
    const auto now = std::chrono::high_resolution_clock::now();
    const float delta = std::chrono::duration<float>(now - previous).count();
    previous = now;
    frameMilliseconds = frameMilliseconds * 0.9f + delta * 1000.0f * 0.1f;
    frameHistory[frameCursor] = frameMilliseconds;
    frameCursor = (frameCursor + 1) % IM_ARRAYSIZE(frameHistory);

    if (window.minimized()) {
      window.endFrame();
      window.waitForMessage();
      continue;
    }
    if (window.consumeResized()) {
      swapchain.recreate(window.width(), window.height(), renderer.settings.vsync);
      if (swapchain.extent().width > 0 && swapchain.extent().height > 0)
        renderer.resize(swapchain.extent().width, swapchain.extent().height);
    }

    // F1 hides the interface, F12 saves a screenshot; both on the key's edge.
    const bool f1 = window.input().keyPressed(VK_F1), f12 = window.input().keyPressed(VK_F12);
    // F5 switches between the rasteriser and the path tracer.
    const bool f5 = window.input().keyPressed(VK_F5);
    if (f5 && !f5Held) renderer.settings.renderer = renderer.settings.renderer == 0 ? 1 : 0;
    f5Held = f5;
    if (f1 && !f1Held) showUi = !showUi;
    if (f12 && !f12Held) requestScreenshot();
    f1Held = f1;
    f12Held = f12;

    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    if (showUi) drawInterface();
    ImGui::Render();

    const bool overUi = io.WantCaptureMouse;
    if (!overUi && window.input().mouseDown[1] && !window.mouseCaptured()) window.captureMouse(true);
    if (!window.input().mouseDown[1] && window.mouseCaptured()) window.captureMouse(false);
    camera.setFlying(window.mouseCaptured());
    camera.update(window.input(), delta, overUi, window.mouseCaptured());
    if (spinPerFrame != 0.0f) camera.yaw += spinPerFrame;
    for (const ScheduledAction &action : scheduledActions)
      if (action.frame == framesRendered) applyScheduledAction(action);
    if (framesRendered == pathExecutionSwitchFrame) {
      renderer.settings.pathExecution = renderer.settings.pathExecution == 0 ? 1 : 0;
      logInfo("switched path execution to {} at frame {}",
              renderer.settings.pathExecution == 1 ? "wavefront" : "megakernel", framesRendered);
    }

    VkCommandBuffer command = VK_NULL_HANDLE;
    std::uint32_t imageIndex = 0;
    if (!swapchain.beginFrame(command, imageIndex)) {
      swapchain.recreate(window.width(), window.height(), renderer.settings.vsync);
      renderer.resize(swapchain.extent().width, swapchain.extent().height);
      window.endFrame();
      continue;
    }

    renderer.render(command, imageIndex, camera, delta);
    ui->record(command, ImGui::GetDrawData(), swapchain.frameIndex());
    vkCmdEndRendering(command);

    if (captureFrame == -2) captureFrame = framesRendered;
    const bool capturing = captureSamples > 0 ? renderer.statistics().samples >= static_cast<std::uint32_t>(captureSamples)
                                              : captureFrame >= 0 && framesRendered == captureFrame;
    if (capturing && !capturePath.empty()) captureSwapchain(command, imageIndex);
    if (capturing && (!linearPath.empty() || !denoisedLinearPath.empty() || !albedoPath.empty() ||
                      !normalPath.empty()))
      renderer.recordLinearCapture(command);

    VkImageMemoryBarrier2 toPresent{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    toPresent.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    toPresent.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    toPresent.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    toPresent.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    toPresent.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toPresent.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toPresent.image = swapchain.image(imageIndex);
    toPresent.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &toPresent;
    vkCmdPipelineBarrier2(command, &dependency);

    if (!swapchain.endFrame(command, imageIndex)) {
      swapchain.recreate(window.width(), window.height(), renderer.settings.vsync);
      renderer.resize(swapchain.extent().width, swapchain.extent().height);
    }
    window.endFrame();
    ++framesRendered;
    if (capturing) {
      if (!capturePath.empty()) writeCapture();
      if (!linearPath.empty()) {
        if (renderer.writeLinearCapture(linearPath))
          logInfo("wrote {} ({} samples)", linearPath, renderer.statistics().samples);
        else {
          logError("could not write {}", linearPath);
          captureFailed = true;
        }
      }
      if (!denoisedLinearPath.empty()) {
        if (renderer.writeDenoisedCapture(denoisedLinearPath))
          logInfo("wrote denoised {} (raw accumulation unchanged)", denoisedLinearPath);
        else {
          logError("could not write denoised {}", denoisedLinearPath);
          captureFailed = true;
        }
      }
      for (const auto &[guidePath, plane] : {std::pair{albedoPath, 1}, std::pair{normalPath, 2}}) {
        if (guidePath.empty()) continue;
        if (renderer.writeGuideCapture(guidePath, plane))
          logInfo("wrote {} guide {}", plane == 1 ? "albedo" : "normal", guidePath);
        else {
          logError("could not write guide {}", guidePath);
          captureFailed = true;
        }
      }
      if (renderer.statistics().activeRenderer != renderer.settings.renderer) {
        // A capture is evidence for the backend it names; never label a substitute.
        logError("the requested renderer did not run: the captured frame came from renderer {}",
                 renderer.statistics().activeRenderer);
        captureFailed = true;
      }
      const std::string metadataTarget = !linearPath.empty() ? linearPath : !denoisedLinearPath.empty()
                                             ? denoisedLinearPath : !albedoPath.empty() ? albedoPath : normalPath;
      if (!metadataTarget.empty() && !writeCaptureMetadata(metadataTarget + ".json", framesRendered, delta)) {
        logError("could not write {}.json", metadataTarget);
        captureFailed = true;
      }
      status = "wrote " + (!capturePath.empty() ? capturePath : !linearPath.empty() ? linearPath : denoisedLinearPath);
      captureFrame = -1;
      captureSamples = 0;
      if (exitAfterCapture) break;
    }
  }

  context.waitIdle();
  ui.reset();
  ImGui_ImplWin32_Shutdown();
  ImGui::DestroyContext();
}

} // namespace basalt

constexpr const char *kUsage =
    "basalt [model.gltf|model.glb] [environment.hdr] [options]\n"
    "\n"
    "Environment: --environment-sun extract|keep (an .hdr's sun becomes the analytic sun, or stays in the image)\n"
    "Capture: --screenshot FILE.png  --pfm FILE (.pfm or .exr)  --denoised-pfm FILE  --albedo-pfm FILE\n"
    "         --normal-pfm FILE  --frame N  --spp N  --log FILE  --size WxH  --view YAW PITCH D  --spin DEG\n"
    "         --no-ui  --no-vsync  --no-validation  --device NAME\n"
    "Renderer: --renderer raster|cpu|gpu|gpu-bvh|gpu-pipeline|hybrid\n"
    "Path tracing: --bounces N  --strategy 0|1|2  --seed N  --spf N  --threads N  --intersector own|embree|avx2\n"
    "         --denoise  --path-execution megakernel|wavefront  --bvh-builder cpu|gpu  --bvh-width binary|bvh4|bvh8\n"
    "         --bvh-cost off|nodes|triangles  --path-temporal  --texture-filter level0|raycone\n"
    "         --aperture R  --focus-distance D  --di-estimator nee|restir  --restir-reuse none|temporal|spatial|both\n"
    "         --restir-candidates N  --hybrid-comparison traced|raster|difference|split|spp  --gpu-profile\n"
    "         --wavefront-capacity N  --wavefront-allocation auto|subgroup|atomic  --wavefront-fusion auto|on|off\n"
    "         --wavefront-queue-limit N (tests only)  --at-frame N KEY=VALUE  --switch-path-execution-frame N\n"
    "Rasteriser: --shadows 0|1|2  --ao 0|1|2  --ao-radius R  --reflections 0|1|2  --aa 0|1|2  --taa-feedback F  --sun AZ EL\n"
    "         --ground R M  --ground-color R G B  --ground-preset N  --lights N  --light-shadows 0|1\n"
    "         --clustered 0|1  --debug N  --wireframe\n"
    "\n"
    "Output files are PFM or OpenEXR by extension. --help or -h prints this.\n";

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR commandLine, int) {
  // First, so a bad argument is reported.
  basalt::openLogFile("basalt.log");
  std::string scene, environment, screenshot, linear, denoisedLinear, albedoLinear, normalLinear;
  int captureFrame = -1;
  int captureSamples = 0;
  int rendererKind = -1;
  int bounces = -1;
  int threads = -1;
  int samplesPerFrame = -1;
  int strategy = -1;
  int intersector = -1;
  bool denoise = false;
  long long seed = -1;
  int debugView = 0;
  int shadowMode = -1, occlusionMode = -1, reflectionMode = -1;
  float occlusionRadius = -1.0f;
  float groundRoughness = -1.0f, groundMetallic = 0.0f;
  basalt::Vec3 groundColor{-1.0f, -1.0f, -1.0f};
  int groundPreset = -1;
  int antialiasing = -1;
  float spin = 0.0f;
  float sunAzimuth = -1000.0f, sunElevation = -1000.0f;
  int environmentSun = 1;
  float feedback = -1.0f;
  int testLights = -1;
  int lightShadows = -1;
  int clustered = -1;
  bool wireframe = false;
  bool vsync = true;
  bool showUi = true;
  bool argumentError = false;
  bool pathTemporal = false;
  bool gpuProfile = false;
  float aperture = 0.0f, focusDistance = 0.0f;
  int textureFilter = -1;
  int diEstimator = -1, restirReuse = -1, restirCandidates = -1;
  int wavefrontCapacity = -1;
  int wavefrontAllocation = -1;
  int wavefrontQueueLimit = 0;
  int wavefrontFusion = -1;
  basalt::StartupOptions startup;
  int bvhDiagnostic = -1;
  int bvhBuilder = -1;
  int bvhWidth = -1;
  int pathExecution = -1;
  int pathExecutionSwitchFrame = -1;
  int hybridComparison = -1;
  std::string deviceSelection;
  std::vector<basalt::ScheduledAction> scheduledActions;
  std::array<float, 3> view{};
  bool hasView = false;
  int count = 0;
  // An empty command line makes CommandLineToArgvW return the executable's own path.
  LPWSTR *arguments = commandLine && *commandLine ? CommandLineToArgvW(commandLine, &count) : nullptr;
  if (arguments) {
    // --help prints the usage to the console that started the process (a WIN32 program has
    // none of its own) and exits.
    for (int i = 0; i < count; ++i) {
      const std::wstring argument = arguments[i];
      if (argument != L"--help" && argument != L"-h") continue;
      // A redirected stdout (a pipe or file) is inherited even by a WIN32 program; otherwise
      // write to the console of the shell that started it.
      const HANDLE inherited = GetStdHandle(STD_OUTPUT_HANDLE);
      DWORD written = 0;
      if (inherited && inherited != INVALID_HANDLE_VALUE && GetFileType(inherited) != FILE_TYPE_UNKNOWN) {
        WriteFile(inherited, kUsage, static_cast<DWORD>(std::strlen(kUsage)), &written, nullptr);
      } else if (AttachConsole(ATTACH_PARENT_PROCESS)) {
        FILE *console = nullptr;
        if (freopen_s(&console, "CONOUT$", "w", stdout) == 0) {
          std::fputs(kUsage, stdout);
          std::fflush(stdout);
        }
      }
      basalt::logInfo("{}", kUsage);
      LocalFree(arguments);
      return 0;
    }
    // --log first, so every later argument error reaches the requested file.
    for (int i = 0; i + 1 < count; ++i)
      if (std::filesystem::path(arguments[i]).string() == "--log")
        basalt::openLogFile(std::filesystem::path(arguments[i + 1]).string());
    // Integer options are checked: a malformed or out-of-range value is an error, not zero.
    auto integer = [&](int &i, const std::string &name, long long minimum, long long maximum) -> long long {
      const std::wstring text = arguments[++i];
      wchar_t *end = nullptr;
      const long long value = std::wcstoll(text.c_str(), &end, 10);
      if (text.empty() || end != text.c_str() + text.size() || value < minimum || value > maximum) {
        basalt::logError("{} needs an integer from {} to {}, not '{}'", name, minimum, maximum,
                         std::filesystem::path(text).string());
        argumentError = true;
        return minimum;
      }
      return value;
    };
    for (int i = 0; i < count; ++i) {
      const std::string argument = std::filesystem::path(arguments[i]).string();
      auto endsWith = [&](const char *suffix) {
        const std::string tail(suffix);
        return argument.size() > tail.size() &&
               argument.compare(argument.size() - tail.size(), tail.size(), tail) == 0;
      };
      if (argument == "--log" && i + 1 < count) {
        ++i;  // opened above: test harnesses give each process its own log (no stdout here)
      } else if (argument == "--screenshot" && i + 1 < count) {
        screenshot = std::filesystem::path(arguments[++i]).string();
        if (captureFrame < 0) captureFrame = 8;
      } else if (argument == "--frame" && i + 1 < count) {
        captureFrame = _wtoi(arguments[++i]);
      } else if (argument == "--pfm" && i + 1 < count) {
        linear = std::filesystem::path(arguments[++i]).string();
      } else if (argument == "--denoised-pfm" && i + 1 < count) {
        denoisedLinear = std::filesystem::path(arguments[++i]).string();
      } else if (argument == "--albedo-pfm" && i + 1 < count) {
        albedoLinear = std::filesystem::path(arguments[++i]).string();
      } else if (argument == "--normal-pfm" && i + 1 < count) {
        normalLinear = std::filesystem::path(arguments[++i]).string();
      } else if (argument == "--spp" && i + 1 < count) {
        captureSamples = static_cast<int>(integer(i, "--spp", 1, 1 << 30));
      } else if (argument == "--bounces" && i + 1 < count) {
        // A bounce is a scattering event; every tracer needs at least one (see the docs).
        bounces = static_cast<int>(integer(i, "--bounces", 1, 64));
      } else if (argument == "--intersector" && i + 1 < count) {
        const std::string name = std::filesystem::path(arguments[++i]).string();
        intersector = name == "embree" ? 1 : name == "avx2" || name == "bvh8" ? 2 : 0;
        if (name != "embree" && name != "own" && name != "avx2" && name != "bvh8") {
          basalt::logError("unknown intersector {}: own, embree or avx2/bvh8", name);
          argumentError = true;
        }
      } else if (argument == "--denoise") {
        denoise = true;
      } else if (argument == "--wavefront-capacity" && i + 1 < count) {
        wavefrontCapacity = static_cast<int>(integer(i, "--wavefront-capacity", 0, 1 << 30));
      } else if (argument == "--wavefront-allocation" && i + 1 < count) {
        const std::string name = std::filesystem::path(arguments[++i]).string();
        wavefrontAllocation = name == "auto" ? 0 : name == "subgroup" ? 1 : name == "atomic" ? 2 : -2;
        if (wavefrontAllocation == -2) {
          basalt::logError("unknown wavefront allocation {}: auto, subgroup or atomic", name);
          argumentError = true;
        }
      } else if (argument == "--wavefront-fusion" && i + 1 < count) {
        const std::string name = std::filesystem::path(arguments[++i]).string();
        wavefrontFusion = name == "auto" ? 0 : name == "on" ? 1 : name == "off" ? 2 : -2;
        if (wavefrontFusion == -2) {
          basalt::logError("unknown wavefront fusion {}: auto, on or off", name);
          argumentError = true;
        }
      } else if (argument == "--wavefront-queue-limit" && i + 1 < count) {
        wavefrontQueueLimit = static_cast<int>(integer(i, "--wavefront-queue-limit", 1, 1 << 30));
      } else if ((argument == "--aperture" || argument == "--focus-distance") && i + 1 < count) {
        const std::wstring text = arguments[++i];
        wchar_t *end = nullptr;
        const double value = std::wcstod(text.c_str(), &end);
        if (text.empty() || end != text.c_str() + text.size() || !(value >= 0.0) || value > 1e6) {
          basalt::logError("{} needs a non-negative number, not '{}'", argument, std::filesystem::path(text).string());
          argumentError = true;
        }
        (argument == "--aperture" ? aperture : focusDistance) = static_cast<float>(value);
      } else if (argument == "--texture-filter" && i + 1 < count) {
        const std::string name = std::filesystem::path(arguments[++i]).string();
        textureFilter = name == "level0" ? 0 : name == "raycone" ? 1 : -2;
        if (textureFilter == -2) {
          basalt::logError("unknown texture filter {}: level0 or raycone", name);
          argumentError = true;
        }
      } else if (argument == "--di-estimator" && i + 1 < count) {
        const std::string name = std::filesystem::path(arguments[++i]).string();
        diEstimator = name == "nee" ? 0 : name == "restir" ? 1 : -2;
        if (diEstimator == -2) {
          basalt::logError("unknown direct-light estimator {}: nee or restir", name);
          argumentError = true;
        }
      } else if (argument == "--restir-reuse" && i + 1 < count) {
        const std::string name = std::filesystem::path(arguments[++i]).string();
        restirReuse = name == "none" ? 0 : name == "temporal" ? 1 : name == "spatial" ? 2 : name == "both" ? 3 : -2;
        if (restirReuse == -2) {
          basalt::logError("unknown ReSTIR reuse {}: none, temporal, spatial or both", name);
          argumentError = true;
        }
      } else if (argument == "--restir-candidates" && i + 1 < count) {
        restirCandidates = static_cast<int>(integer(i, "--restir-candidates", 1, 13));
      } else if (argument == "--gpu-profile") {
        gpuProfile = true;
      } else if (argument == "--no-validation") {
        startup.validation = false;
      } else if (argument == "--size" && i + 1 < count) {
        unsigned w = 0, h = 0;
        const std::string size = std::filesystem::path(arguments[++i]).string();
        if (std::sscanf(size.c_str(), "%ux%u", &w, &h) != 2 || w < 8 || h < 8 || w > 16384 || h > 16384) {
          basalt::logError("--size needs WIDTHxHEIGHT from 8x8 to 16384x16384, not '{}'", size);
          argumentError = true;
        }
        startup.width = w;
        startup.height = h;
      } else if (argument == "--path-temporal") {
        pathTemporal = true;
      } else if (argument == "--hybrid-comparison" && i + 1 < count) {
        const std::string name = std::filesystem::path(arguments[++i]).string();
        hybridComparison = name == "traced" ? 0 : name == "raster" ? 1 :
                           name == "difference" ? 2 : name == "split" ? 3 :
                           name == "spp" ? 4 : -2;
        if (hybridComparison == -2) {
          basalt::logError("unknown hybrid comparison {}: traced, raster, difference, split or spp", name);
          argumentError = true;
        }
      } else if (argument == "--bvh-cost" && i + 1 < count) {
        const std::string mode = std::filesystem::path(arguments[++i]).string();
        bvhDiagnostic = mode == "off" ? 0 : mode == "nodes" ? 1 : mode == "triangles" ? 2 : -2;
        if (bvhDiagnostic == -2) {
          basalt::logError("unknown BVH diagnostic {}: off, nodes or triangles", mode);
          argumentError = true;
        }
      } else if (argument == "--device" && i + 1 < count) {
        deviceSelection = std::filesystem::path(arguments[++i]).string();
      } else if (argument == "--bvh-builder" && i + 1 < count) {
        const std::string name = std::filesystem::path(arguments[++i]).string();
        bvhBuilder = name == "cpu" ? 0 : name == "gpu" ? 1 : -2;
        if (bvhBuilder == -2) {
          basalt::logError("unknown BVH builder {}: cpu or gpu", name);
          argumentError = true;
        }
      } else if (argument == "--bvh-width" && i + 1 < count) {
        const std::string name = std::filesystem::path(arguments[++i]).string();
        bvhWidth = name == "binary" ? 0 : name == "4" || name == "bvh4" ? 1 :
                   name == "8" || name == "bvh8" ? 2 : -2;
        if (bvhWidth == -2) {
          basalt::logError("unknown BVH width {}: binary, 4/bvh4 or 8/bvh8", name);
          argumentError = true;
        }
      } else if (argument == "--path-execution" && i + 1 < count) {
        const std::string name = std::filesystem::path(arguments[++i]).string();
        pathExecution = name == "megakernel" ? 0 : name == "wavefront" ? 1 : -2;
        if (pathExecution == -2) {
          basalt::logError("unknown path execution {}: megakernel or wavefront", name);
          argumentError = true;
        }
      } else if (argument == "--at-frame" && i + 2 < count) {
        basalt::ScheduledAction action;
        action.frame = static_cast<int>(integer(i, "--at-frame", 0, 1 << 30));
        const std::string assignment = std::filesystem::path(arguments[++i]).string();
        const std::size_t equals = assignment.find('=');
        if (equals == std::string::npos || equals == 0) {
          basalt::logError("--at-frame needs KEY=VALUE, not '{}'", assignment);
          argumentError = true;
        }
        action.key = assignment.substr(0, equals);
        action.value = equals == std::string::npos ? std::string() : assignment.substr(equals + 1);
        scheduledActions.push_back(action);
      } else if (argument == "--switch-path-execution-frame" && i + 1 < count) {
        pathExecutionSwitchFrame = _wtoi(arguments[++i]);
      } else if (argument == "--strategy" && i + 1 < count) {
        strategy = static_cast<int>(integer(i, "--strategy", 0, 2)); // 0 MIS, 1 BSDF only, 2 lights only
      } else if (argument == "--spf" && i + 1 < count) {
        samplesPerFrame = static_cast<int>(integer(i, "--spf", 1, 1024));
      } else if (argument == "--threads" && i + 1 < count) {
        threads = static_cast<int>(integer(i, "--threads", 1, 4096));
      } else if (argument == "--seed" && i + 1 < count) {
        seed = integer(i, "--seed", 0, 0xffffffffll);
      } else if (argument == "--renderer" && i + 1 < count) {
        const std::string name = std::filesystem::path(arguments[++i]).string();
        rendererKind = basalt::rendererFromName(name);
        if (rendererKind == -2) {
          basalt::logError("unknown renderer {}: raster, cpu, gpu/gpu-rt, gpu-bvh, hybrid or gpu-pipeline", name);
          argumentError = true;
        }
      } else if (argument == "--debug" && i + 1 < count) {
        debugView = _wtoi(arguments[++i]);
      } else if (argument == "--shadows" && i + 1 < count) {
        shadowMode = _wtoi(arguments[++i]); // 0 none, 1 cascaded, 2 traced
      } else if (argument == "--ao" && i + 1 < count) {
        occlusionMode = _wtoi(arguments[++i]); // 0 material map, 1 traced contact, 2 traced sky visibility
        if (occlusionMode < 0 || occlusionMode > 2) {
          basalt::logError("--ao takes 0 (material map), 1 (traced contact) or 2 (traced sky visibility)");
          argumentError = true;
        }
      } else if (argument == "--ao-radius" && i + 1 < count) {
        occlusionRadius = static_cast<float>(_wtof(arguments[++i]));
        if (!(occlusionRadius >= 0.0f)) {
          basalt::logError("--ao-radius needs a non-negative distance (0 = automatic)");
          argumentError = true;
        }
      } else if (argument == "--reflections" && i + 1 < count) {
        reflectionMode = _wtoi(arguments[++i]); // 0 environment, 1 screen space, 2 traced
      } else if (argument == "--ground" && i + 2 < count) {
        groundRoughness = static_cast<float>(_wtof(arguments[++i]));
        groundMetallic = static_cast<float>(_wtof(arguments[++i]));
      } else if (argument == "--ground-color" && i + 3 < count) {
        groundColor.x = static_cast<float>(_wtof(arguments[++i]));
        groundColor.y = static_cast<float>(_wtof(arguments[++i]));
        groundColor.z = static_cast<float>(_wtof(arguments[++i]));
      } else if (argument == "--ground-preset" && i + 1 < count) {
        groundPreset = _wtoi(arguments[++i]);
      } else if (argument == "--clustered" && i + 1 < count) {
        clustered = _wtoi(arguments[++i]) != 0;
      } else if (argument == "--lights" && i + 1 < count) {
        testLights = _wtoi(arguments[++i]);
      } else if (argument == "--light-shadows" && i + 1 < count) {
        lightShadows = _wtoi(arguments[++i]) != 0;
      } else if (argument == "--environment-sun" && i + 1 < count) {
        const std::string name = std::filesystem::path(arguments[++i]).string();
        environmentSun = name == "extract" ? 1 : name == "keep" ? 0 : -2;
        if (environmentSun == -2) {
          basalt::logError("unknown environment sun mode {}: extract or keep", name);
          argumentError = true;
        }
      } else if (argument == "--sun" && i + 2 < count) {
        sunAzimuth = static_cast<float>(_wtof(arguments[++i]));
        sunElevation = static_cast<float>(_wtof(arguments[++i]));
      } else if (argument == "--spin" && i + 1 < count) {
        spin = basalt::radians(static_cast<float>(_wtof(arguments[++i])));
      } else if (argument == "--taa-feedback" && i + 1 < count) {
        feedback = static_cast<float>(_wtof(arguments[++i]));
      } else if (argument == "--aa" && i + 1 < count) {
        antialiasing = _wtoi(arguments[++i]); // 0 none, 1 FXAA, 2 temporal
      } else if (argument == "--no-ui") {
        showUi = false;
      } else if (argument == "--wireframe") {
        wireframe = true;
      } else if (argument == "--no-vsync") {
        vsync = false;
      } else if (argument == "--view" && i + 3 < count) {
        view = {static_cast<float>(_wtof(arguments[i + 1])),
                static_cast<float>(_wtof(arguments[i + 2])),
                static_cast<float>(_wtof(arguments[i + 3]))};
        hasView = true;
        i += 3;
      } else if (endsWith(".hdr")) {
        environment = argument;
      } else if (endsWith(".gltf") || endsWith(".glb")) {
        scene = argument;
      } else {
        // Includes an option given without its value, which the checks above skip.
        basalt::logError("unknown or incomplete argument '{}'", argument);
        argumentError = true;
      }
    }
    LocalFree(arguments);
  }
  if (argumentError) return 2;

  try {
    // Contradictory options fail before any device check: they are wrong on every device.
    if (aperture > 0.0f && rendererKind == 4)
      throw std::runtime_error("--aperture needs traced primary rays; the hybrid renderer's are rasterised");
    if ((aperture > 0.0f || focusDistance > 0.0f) && (rendererKind == 0 || rendererKind == -1))
      throw std::runtime_error("--aperture and --focus-distance require a path tracer");
    if ((restirReuse >= 0 || restirCandidates >= 0) && diEstimator != 1)
      throw std::runtime_error("--restir-reuse and --restir-candidates require --di-estimator restir");
    if (diEstimator == 1 && rendererKind == 4)
      throw std::runtime_error("--di-estimator restir needs traced primary rays; the hybrid renderer's are rasterised");
    if (diEstimator == 1 && (rendererKind == 0 || rendererKind == -1))
      throw std::runtime_error("--di-estimator restir requires a path tracer");
    if (diEstimator == 1 && samplesPerFrame > 1)
      throw std::runtime_error("--di-estimator restir reuses one path per pixel per frame: --spf must be 1");
    if (!deviceSelection.empty() && _putenv_s("BASALT_VULKAN_DEVICE", deviceSelection.c_str()) != 0)
      throw std::runtime_error("could not apply the requested Vulkan device selection");
    basalt::Application application(startup);
    if ((rendererKind == 2 || rendererKind == 4) && !application.context.rayTracingSupported)
      throw std::runtime_error("the selected renderer requires Vulkan ray queries on the selected device");
    if (rendererKind == 5 && !application.context.rayPipelineSupported)
      throw std::runtime_error("the selected renderer requires VK_KHR_ray_tracing_pipeline on the selected device");
    if ((shadowMode == 2 || occlusionMode >= 1 || reflectionMode == 2) && !application.context.rayTracingSupported)
      throw std::runtime_error("traced shadows, occlusion or reflections require Vulkan ray queries on the selected device");
    if (rendererKind == 1 && intersector == 1 && !pt::EmbreeScene::available())
      throw std::runtime_error("the requested Embree intersector is unavailable in this build");
    if (rendererKind == 1 && intersector == 2 && !pt::cpuAvx2Available())
      throw std::runtime_error("the requested AVX2 BVH8 intersector is unavailable on this CPU");
    if (denoise && rendererKind != 1)
      throw std::runtime_error("OIDN denoising is currently supported only by the CPU renderer");
    if (bvhDiagnostic > 0 && rendererKind != 3)
      throw std::runtime_error("--bvh-cost requires --renderer gpu-bvh");
    if (bvhBuilder >= 0 && rendererKind != 3)
      throw std::runtime_error("--bvh-builder requires --renderer gpu-bvh");
    if (bvhWidth >= 0 && rendererKind != 3)
      throw std::runtime_error("--bvh-width requires --renderer gpu-bvh");
    if (bvhWidth > 0 && bvhBuilder == 1)
      throw std::runtime_error("quantized wide BVHs currently require --bvh-builder cpu");
    if (hybridComparison >= 0 && rendererKind != 4)
      throw std::runtime_error("--hybrid-comparison requires --renderer hybrid");
    if (pathExecution == 1 && rendererKind != 2 && rendererKind != 3 && rendererKind != 5)
      throw std::runtime_error("--path-execution wavefront requires --renderer gpu, gpu-bvh or gpu-pipeline");
    if (pathExecutionSwitchFrame >= 0 && rendererKind != 2 && rendererKind != 3 && rendererKind != 5)
      throw std::runtime_error("--switch-path-execution-frame requires a GPU path tracer");
    if (!denoisedLinear.empty() && rendererKind < 2)
      throw std::runtime_error("--denoised-pfm requires a GPU path tracer");
    if ((!albedoLinear.empty() || !normalLinear.empty()) && rendererKind < 2)
      throw std::runtime_error("--albedo-pfm and --normal-pfm require a GPU path tracer");
    if (intersector >= 0 && rendererKind != 1)
      throw std::runtime_error("--intersector requires --renderer cpu");
    const bool capture = !screenshot.empty() || !linear.empty() || !denoisedLinear.empty() ||
                         !albedoLinear.empty() || !normalLinear.empty();
    application.capturePath = screenshot;
    application.linearPath = linear;
    application.denoisedLinearPath = denoisedLinear;
    application.albedoPath = albedoLinear;
    application.normalPath = normalLinear;
    application.failOnLoadError = capture;
    application.captureFrame = capture && captureFrame < 0 ? 8 : captureFrame;
    application.captureSamples = capture ? captureSamples : 0;
    application.pathExecutionSwitchFrame = pathExecutionSwitchFrame;
    application.scheduledActions = scheduledActions;
    application.exitAfterCapture = capture;
    if (rendererKind >= 0) application.renderer.settings.renderer = rendererKind;
    if (bounces >= 0) application.renderer.settings.pathBounces = bounces;
    if (threads > 0) application.renderer.settings.cpuThreads = threads;
    if (samplesPerFrame > 0) application.renderer.settings.pathSamplesPerFrame = samplesPerFrame;
    if (strategy >= 0) application.renderer.settings.pathStrategy = strategy;
    if (intersector >= 0) application.renderer.settings.cpuIntersector = intersector;
    application.renderer.settings.pathDenoise = denoise;
    application.renderer.settings.pathTemporal = pathTemporal;
    application.renderer.settings.pathProfile = gpuProfile;
    application.renderer.settings.pathAperture = aperture;
    application.renderer.settings.pathFocusDistance = focusDistance;
    if (textureFilter >= 0) application.renderer.settings.pathTextureFilter = textureFilter;
    if (diEstimator >= 0) application.renderer.settings.pathDiEstimator = diEstimator;
    if (restirReuse >= 0) application.renderer.settings.pathRestirReuse = restirReuse;
    if (restirCandidates >= 0) application.renderer.settings.pathRestirCandidates = restirCandidates;
    if (diEstimator == 1) application.renderer.settings.pathSamplesPerFrame = 1;
    if (wavefrontCapacity >= 0) application.renderer.settings.pathWaveCapacity = wavefrontCapacity;
    if (wavefrontAllocation >= 0) application.renderer.settings.pathWaveAllocation = wavefrontAllocation;
    application.renderer.settings.pathWaveQueueLimit = wavefrontQueueLimit;
    if (wavefrontFusion >= 0) application.renderer.settings.pathWaveFusion = wavefrontFusion;
    if ((wavefrontCapacity >= 0 || wavefrontAllocation >= 0 || wavefrontQueueLimit > 0 || wavefrontFusion >= 0) &&
        pathExecution != 1)
      throw std::runtime_error("the --wavefront-* options require --path-execution wavefront");
    if (bvhDiagnostic >= 0) application.renderer.settings.pathBvhDiagnostic = bvhDiagnostic;
    if (bvhBuilder >= 0) application.renderer.settings.pathBvhBuilder = bvhBuilder;
    if (bvhWidth >= 0) application.renderer.settings.pathBvhWidth = bvhWidth;
    if (pathExecution >= 0) application.renderer.settings.pathExecution = pathExecution;
    if (hybridComparison >= 0) application.renderer.settings.pathComparison = hybridComparison;
    // A capture that waits for N samples lets the tracer stop there.
    if (capture && captureSamples > 0) application.renderer.settings.pathTargetSamples = captureSamples;
    if (seed >= 0) application.renderer.settings.pathSeed = static_cast<std::uint32_t>(seed);
    application.showUi = showUi;
    application.renderer.settings.debugView = debugView;
    application.renderer.settings.wireframe = wireframe;
    application.renderer.settings.vsync = vsync;
    if (shadowMode >= 0) {
      application.renderer.settings.shadowsEnabled = shadowMode > 0;
      application.renderer.settings.shadowMode = shadowMode > 1 ? 1 : 0;
    }
    if (occlusionMode >= 0) application.renderer.settings.occlusionMode = occlusionMode;
    if (occlusionRadius >= 0.0f) application.renderer.settings.occlusionRadius = occlusionRadius;
    if (reflectionMode >= 0) application.renderer.settings.reflectionMode = reflectionMode;
    if (antialiasing >= 0) application.renderer.settings.antialiasing = antialiasing;
    application.spinPerFrame = spin;
    if (sunAzimuth > -999.0f) {
      application.renderer.settings.sunAzimuth = basalt::radians(sunAzimuth);
      application.renderer.settings.sunElevation = basalt::radians(sunElevation);
      application.fixedSun = true;
    }
    if (testLights >= 0) application.renderer.settings.testLights = testLights;
    if (lightShadows >= 0) application.renderer.settings.lightShadows = lightShadows != 0;
    if (clustered >= 0) application.renderer.settings.clusteredLights = clustered != 0;
    if (feedback > 0.0f) application.renderer.settings.temporalFeedback = feedback;
    if (groundPreset >= 0 && groundPreset < basalt::kGroundPresetCount) {
      const basalt::GroundPreset &chosen = basalt::kGroundPresets[groundPreset];
      application.renderer.settings.groundColor = chosen.colour;
      application.renderer.settings.groundRoughness = chosen.roughness;
      application.renderer.settings.groundMetallic = chosen.metallic;
    }
    if (groundRoughness >= 0.0f) {
      application.renderer.settings.groundRoughness = groundRoughness;
      application.renderer.settings.groundMetallic = groundMetallic;
    }
    if (groundColor.x >= 0.0f) application.renderer.settings.groundColor = groundColor;
    if (!vsync) application.swapchain.recreate(application.window.width(),
                                               application.window.height(), false);
    if (hasView) {
      application.camera.yaw = basalt::radians(view[0]);
      application.camera.pitch = basalt::radians(view[1]);
      application.viewDistanceScale = view[2];
    }
    application.extractEnvironmentSun = environmentSun != 0;
    application.run(scene, environment);
    if (application.context.sawValidationError)
      throw std::runtime_error("Vulkan validation reported an error during rendering");
    if (application.captureFailed)
      throw std::runtime_error("a requested capture output could not be written");
  } catch (const std::exception &failure) {
    basalt::logError("{}", failure.what());
    // Headless and capture runs are scripted: a modal dialog would hang the caller.
    const bool scripted = !screenshot.empty() || !linear.empty() || !denoisedLinear.empty() ||
                          !albedoLinear.empty() || !normalLinear.empty() || !showUi;
    if (!scripted)
      MessageBoxA(nullptr, failure.what(), "Basalt could not start", MB_ICONERROR | MB_OK);
    return 1;
  }
  return 0;
}
