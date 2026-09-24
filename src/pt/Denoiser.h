// Intel Open Image Denoise over a path traced image, with the first hit's albedo and normal
// as its auxiliary images. Interactive use may let OIDN choose; headless reference use
// can require its CPU device and therefore never depends on a graphics adapter.
#pragma once
#include <memory>
#include <string>
#include <vector>

namespace pt {

class Denoiser {
public:
  enum class Device { Automatic, Cpu };
  // Whether Open Image Denoise was fetched and compiled in.
  static bool available();
  explicit Denoiser(Device requested = Device::Automatic);
  ~Denoiser();
  Denoiser(const Denoiser &) = delete;
  Denoiser &operator=(const Denoiser &) = delete;

  // RGB float images, rows from the top; albedo and normal may be empty. Returns the
  // denoised colour, or an empty vector (and the reason in error()) on failure.
  std::vector<float> denoise(const std::vector<float> &color, const std::vector<float> &albedo,
                             const std::vector<float> &normal, unsigned width, unsigned height);
  const std::string &device() const { return deviceName; }
  const std::string &error() const { return lastError; }

private:
  struct State;
  std::unique_ptr<State> state;
  std::string deviceName;
  std::string lastError;
};

} // namespace pt
