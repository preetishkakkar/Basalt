// Window, interface and frame loop.
#include "core/Log.h"
#include "gpu/Context.h"
#include "gpu/Swapchain.h"
#include "gpu/Uploader.h"
#include "platform/Window.h"
#include "render/Renderer.h"
#include "render/UiPass.h"
#include "scene/Camera.h"

#include <imgui.h>
#include <imgui_impl_win32.h>

#include <commdlg.h>
#include <shellapi.h>

#include <stb_image_write.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <format>
#include <memory>
#include <string>

namespace basalt {
namespace {

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

struct Application {
  Window window{"Basalt: Metal shaders on Vulkan", 1600, 900};
  Context context{window, true};
  Swapchain swapchain{context, window.width(), window.height(), true};
  Uploader uploader{context};
  Renderer renderer{context, swapchain, uploader};
  std::unique_ptr<UiPass> ui;
  Camera camera;

  bool showDemo = false;
  bool showLog = false;
  bool showUi = true;
  bool f1Held = false, f12Held = false;
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
      logError("{}", failure.what());
      status = failure.what();
    }
  }

  // Capture mode: render N frames, write the last one, exit.
  std::string capturePath;
  int captureFrame = -1;
  // Turns the camera every frame, so a capture exercises reprojection.
  float spinPerFrame = 0.0f;
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
};

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
  else
    logError("could not write {}", capturePath);
}

static void alignSunToEnvironment(Renderer &renderer) {
  const Vec3 direction = renderer.environment().brightestDirection();
  renderer.settings.sunAzimuth = std::atan2(direction.x, direction.z);
  renderer.settings.sunElevation = std::asin(std::clamp(direction.y, -1.0f, 1.0f));
}

void Application::drawInterface() {
  const ImGuiViewport *viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + 12, viewport->WorkPos.y + 12),
                          ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(360, 720), ImGuiCond_FirstUseEver);
  ImGui::Begin("Basalt");

  RenderSettings &settings = renderer.settings;
  const FrameStatistics &stats = renderer.statistics();

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
    bool skyChanged = false;
    if (environment.procedural()) {
      skyChanged |= ImGui::SliderFloat("Haze", &settings.skyTurbidity, 1.0f, 10.0f);
      skyChanged |= ImGui::SliderFloat("Sky brightness", &settings.skyIntensity, 0.0f, 4.0f);
    }
    if (ImGui::Button("Open an HDR environment...", ImVec2(-1, 0))) {
      const std::string path = openFileDialog(window.handle(), L"Radiance HDR\0*.hdr\0All files\0*.*\0\0",
                                              L"Open an environment map");
      if (!path.empty()) {
        environment.load(path);
        renderer.environmentChanged();
        alignSunToEnvironment(renderer);
      }
    }
    if (!environment.procedural() && ImGui::Button("Point the sun at the environment's light", ImVec2(-1, 0)))
      alignSunToEnvironment(renderer);
    if (ImGui::Button("Back to the procedural sky", ImVec2(-1, 0))) {
      if (!environment.procedural()) {
        environment.load("");
        renderer.environmentChanged();
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
    ImGui::Combo("Occlusion method", &settings.occlusionMode, "Material map\0Ray traced\0");
    ImGui::EndDisabled();
    tracedNote();
    if (traced && settings.occlusionMode == 1) {
      ImGui::SliderInt("Rays per pixel##ao", &settings.occlusionSamples, 1, 16);
      ImGui::SliderFloat("Radius", &settings.occlusionRadius, 0.0f, 10.0f, "%.2f (0 = automatic)",
                         ImGuiSliderFlags_Logarithmic);
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
                 "Reflection confidence\0");
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
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  io.IniFilename = nullptr;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  applyDarkStyle();
  ImGui_ImplWin32_Init(window.handle());
  ui = std::make_unique<UiPass>(context, uploader, swapchain.format());

  if (!initialEnvironment.empty()) {
    renderer.environment().load(initialEnvironment);
    alignSunToEnvironment(renderer);
  }
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
    const bool capturing = captureFrame >= 0 && framesRendered == captureFrame;
    if (capturing) captureSwapchain(command, imageIndex);

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
      writeCapture();
      status = "wrote " + capturePath;
      captureFrame = -1;
      if (exitAfterCapture) break;
    }
  }

  context.waitIdle();
  ui.reset();
  ImGui_ImplWin32_Shutdown();
  ImGui::DestroyContext();
}

} // namespace basalt

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR commandLine, int) {
  // basalt [model.gltf] [environment.hdr] [--screenshot out.png] [--frame N]
  std::string scene, environment, screenshot;
  int captureFrame = -1;
  int debugView = 0;
  int shadowMode = -1, occlusionMode = -1, reflectionMode = -1;
  float groundRoughness = -1.0f, groundMetallic = 0.0f;
  basalt::Vec3 groundColor{-1.0f, -1.0f, -1.0f};
  int groundPreset = -1;
  int antialiasing = -1;
  float spin = 0.0f;
  float feedback = -1.0f;
  int testLights = -1;
  int lightShadows = -1;
  int clustered = -1;
  bool wireframe = false;
  bool vsync = true;
  bool showUi = true;
  std::array<float, 3> view{};
  bool hasView = false;
  int count = 0;
  if (LPWSTR *arguments = CommandLineToArgvW(commandLine, &count)) {
    for (int i = 0; i < count; ++i) {
      const std::string argument = std::filesystem::path(arguments[i]).string();
      auto endsWith = [&](const char *suffix) {
        const std::string tail(suffix);
        return argument.size() > tail.size() &&
               argument.compare(argument.size() - tail.size(), tail.size(), tail) == 0;
      };
      if (argument == "--screenshot" && i + 1 < count) {
        screenshot = std::filesystem::path(arguments[++i]).string();
        if (captureFrame < 0) captureFrame = 8;
      } else if (argument == "--frame" && i + 1 < count) {
        captureFrame = _wtoi(arguments[++i]);
      } else if (argument == "--debug" && i + 1 < count) {
        debugView = _wtoi(arguments[++i]);
      } else if (argument == "--shadows" && i + 1 < count) {
        shadowMode = _wtoi(arguments[++i]); // 0 none, 1 cascaded, 2 traced
      } else if (argument == "--ao" && i + 1 < count) {
        occlusionMode = _wtoi(arguments[++i]); // 0 material map, 1 traced
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
      }
    }
    LocalFree(arguments);
  }

  basalt::openLogFile("basalt.log");
  try {
    basalt::Application application;
    application.capturePath = screenshot;
    application.captureFrame = !screenshot.empty() && captureFrame < 0 ? 8 : captureFrame;
    application.exitAfterCapture = !screenshot.empty();
    application.showUi = showUi;
    application.renderer.settings.debugView = debugView;
    application.renderer.settings.wireframe = wireframe;
    application.renderer.settings.vsync = vsync;
    if (shadowMode >= 0) {
      application.renderer.settings.shadowsEnabled = shadowMode > 0;
      application.renderer.settings.shadowMode = shadowMode > 1 ? 1 : 0;
    }
    if (occlusionMode >= 0) application.renderer.settings.occlusionMode = occlusionMode;
    if (reflectionMode >= 0) application.renderer.settings.reflectionMode = reflectionMode;
    if (antialiasing >= 0) application.renderer.settings.antialiasing = antialiasing;
    application.spinPerFrame = spin;
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
    application.run(scene, environment);
  } catch (const std::exception &failure) {
    basalt::logError("{}", failure.what());
    if (screenshot.empty())
      MessageBoxA(nullptr, failure.what(), "Basalt could not start", MB_ICONERROR | MB_OK);
    return 1;
  }
  return 0;
}
