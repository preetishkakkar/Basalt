#include "pt/Denoiser.h"

#include <cstring>

#if BASALT_WITH_OIDN
#include <OpenImageDenoise/oidn.h>
#endif

namespace pt {

#if BASALT_WITH_OIDN

struct Denoiser::State {
  OIDNDevice device = nullptr;
};

bool Denoiser::available() { return true; }

Denoiser::Denoiser(Device requested) : state(std::make_unique<State>()) {
  state->device = oidnNewDevice(requested == Device::Cpu ? OIDN_DEVICE_TYPE_CPU : OIDN_DEVICE_TYPE_DEFAULT);
  if (!state->device) {
    lastError = "Open Image Denoise could not create a device";
    return;
  }
  oidnCommitDevice(state->device);
  const char *message = nullptr;
  if (oidnGetDeviceError(state->device, &message) != OIDN_ERROR_NONE) {
    lastError = message ? message : "Open Image Denoise could not start";
    oidnReleaseDevice(state->device);
    state->device = nullptr;
    return;
  }
  switch (oidnGetDeviceInt(state->device, "type")) {
  case OIDN_DEVICE_TYPE_CPU: deviceName = "CPU"; break;
  case OIDN_DEVICE_TYPE_CUDA: deviceName = "CUDA"; break;
  case OIDN_DEVICE_TYPE_SYCL: deviceName = "SYCL"; break;
  case OIDN_DEVICE_TYPE_HIP: deviceName = "HIP"; break;
  case OIDN_DEVICE_TYPE_METAL: deviceName = "Metal"; break;
  default: deviceName = "unknown"; break;
  }
}

Denoiser::~Denoiser() {
  if (state && state->device) oidnReleaseDevice(state->device);
}

std::vector<float> Denoiser::denoise(const std::vector<float> &color, const std::vector<float> &albedo,
                                     const std::vector<float> &normal, unsigned width, unsigned height) {
  if (!state->device) return {};
  const std::size_t bytes = static_cast<std::size_t>(width) * height * 3 * sizeof(float);
  if (color.size() * sizeof(float) != bytes) {
    lastError = "the colour image does not match its size";
    return {};
  }
  // Device buffers, so the images reach a GPU device as well as the CPU one.
  auto upload = [&](const std::vector<float> &image) {
    OIDNBuffer buffer = oidnNewBuffer(state->device, bytes);
    oidnWriteBuffer(buffer, 0, bytes, image.data());
    return buffer;
  };
  OIDNBuffer colorBuffer = upload(color);
  OIDNBuffer albedoBuffer = albedo.size() * sizeof(float) == bytes ? upload(albedo) : nullptr;
  OIDNBuffer normalBuffer = albedoBuffer && normal.size() * sizeof(float) == bytes ? upload(normal) : nullptr;
  OIDNBuffer outputBuffer = oidnNewBuffer(state->device, bytes);

  OIDNFilter filter = oidnNewFilter(state->device, "RT");
  oidnSetFilterImage(filter, "color", colorBuffer, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
  if (albedoBuffer) oidnSetFilterImage(filter, "albedo", albedoBuffer, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
  if (normalBuffer) oidnSetFilterImage(filter, "normal", normalBuffer, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
  oidnSetFilterImage(filter, "output", outputBuffer, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
  oidnSetFilterBool(filter, "hdr", true);
  oidnCommitFilter(filter);
  oidnExecuteFilter(filter);

  std::vector<float> output(color.size());
  oidnReadBuffer(outputBuffer, 0, bytes, output.data());
  const char *message = nullptr;
  const bool failed = oidnGetDeviceError(state->device, &message) != OIDN_ERROR_NONE;
  if (failed) lastError = message ? message : "denoising failed";

  oidnReleaseFilter(filter);
  for (OIDNBuffer buffer : {colorBuffer, albedoBuffer, normalBuffer, outputBuffer})
    if (buffer) oidnReleaseBuffer(buffer);
  if (failed) return {};
  return output;
}

#else

struct Denoiser::State {};
bool Denoiser::available() { return false; }
Denoiser::Denoiser(Device) : state(std::make_unique<State>()) { lastError = "this build has no Open Image Denoise"; }
Denoiser::~Denoiser() = default;
std::vector<float> Denoiser::denoise(const std::vector<float> &, const std::vector<float> &, const std::vector<float> &,
                                     unsigned, unsigned) {
  return {};
}

#endif

} // namespace pt
