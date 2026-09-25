#include "pt/EnvironmentSun.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace pt {
namespace {

constexpr double kPiD = 3.14159265358979323846;
// A sun stands this far above the sky's median luminance. Stars and small highlights
// do not; an unclipped sun is thousands of times brighter, a moon hundreds.
constexpr double kContrast = 100.0;
// The region keeps texels above this fraction of the peak (and above the contrast level):
// the disc and its bright core, while the wide aureole stays in the image as sky.
constexpr double kPeakFraction = 0.01;
// Nothing further than this from the peak is taken, so a bright overcast patch or a lit
// wall in the image never becomes one huge "sun".
constexpr double kMaximumRadius = 4.0 * kPiD / 180.0;

struct Direction { double x, y, z; };

Direction texelDirection(uint x, uint y, uint width, uint height) {
  const double theta = (static_cast<double>(y) + 0.5) / static_cast<double>(height) * kPiD;
  const double phi = ((static_cast<double>(x) + 0.5) / static_cast<double>(width) - 0.5) * 2.0 * kPiD;
  return {std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi)};
}

double texelSolidAngle(uint y, uint width, uint height) {
  const double theta = (static_cast<double>(y) + 0.5) / static_cast<double>(height) * kPiD;
  return (2.0 * kPiD / static_cast<double>(width)) * (kPiD / static_cast<double>(height)) * std::sin(theta);
}

double luminanceOf(const float *texel) {
  const double l = 0.2126 * texel[0] + 0.7152 * texel[1] + 0.0722 * texel[2];
  return std::isfinite(l) ? l : 0.0;
}

} // namespace

EnvironmentSun extractEnvironmentSun(std::vector<float> &rgba, uint width, uint height) {
  EnvironmentSun sun;
  if (width == 0 || height == 0 || rgba.size() < static_cast<std::size_t>(width) * height * 4u) return sun;
  const std::size_t count = static_cast<std::size_t>(width) * height;
  std::vector<double> luminance(count);
  double power = 0.0;
  for (uint y = 0; y < height; ++y) {
    const double solidAngle = texelSolidAngle(y, width, height);
    for (uint x = 0; x < width; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * width + x;
      luminance[i] = luminanceOf(&rgba[i * 4]);
      power += luminance[i] * solidAngle;
    }
  }
  auto wrapX = [&](long long x) { return static_cast<uint>(((x % width) + width) % width); };
  auto clampY = [&](long long y) { return static_cast<uint>(std::clamp<long long>(y, 0, height - 1)); };
  auto at = [&](uint x, uint y) { return luminance[static_cast<std::size_t>(y) * width + x]; };

  // The seed: the brightest 3x3 neighbourhood at full resolution, so a single hot pixel (a
  // star, a glint) does not outrank a sun, and a moon a few texels wide is never stepped
  // over; then the brightest texel of that neighbourhood.
  std::vector<double> rows(count);
  for (uint y = 0; y < height; ++y)
    for (uint x = 0; x < width; ++x)
      rows[static_cast<std::size_t>(y) * width + x] =
          at(wrapX(static_cast<long long>(x) - 1), y) + at(x, y) + at(wrapX(static_cast<long long>(x) + 1), y);
  double best = -1.0;
  uint seedX = 0, seedY = 0;
  for (uint y = 0; y < height; ++y)
    for (uint x = 0; x < width; ++x) {
      const double sum = rows[static_cast<std::size_t>(clampY(static_cast<long long>(y) - 1)) * width + x] +
                         rows[static_cast<std::size_t>(y) * width + x] +
                         rows[static_cast<std::size_t>(clampY(static_cast<long long>(y) + 1)) * width + x];
      if (sum > best) { best = sum; seedX = x; seedY = y; }
    }
  {
    const uint cx = seedX, cy = seedY;
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        const uint x = wrapX(static_cast<long long>(cx) + dx), y = clampY(static_cast<long long>(cy) + dy);
        if (at(x, y) > at(seedX, seedY)) { seedX = x; seedY = y; }
      }
  }
  const Direction seed = texelDirection(seedX, seedY, width, height);
  sun.direction = float3(static_cast<float>(seed.x), static_cast<float>(seed.y), static_cast<float>(seed.z));

  std::vector<double> sorted(luminance);
  std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(count / 2), sorted.end());
  const double median = sorted[count / 2];
  const double peak = at(seedX, seedY);
  if (!(peak > 0.0) || peak < kContrast * median) return sun;
  const double threshold = std::max(kContrast * median, kPeakFraction * peak);
  const double cosMaximum = std::cos(kMaximumRadius);

  // The region: texels above the threshold connected to the seed (longitude wraps).
  std::vector<unsigned char> inRegion(count, 0);
  std::vector<std::size_t> region{static_cast<std::size_t>(seedY) * width + seedX};
  inRegion[region[0]] = 1;
  for (std::size_t next = 0; next < region.size(); ++next) {
    const uint x = static_cast<uint>(region[next] % width), y = static_cast<uint>(region[next] / width);
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        const long long ny = static_cast<long long>(y) + dy;
        if (ny < 0 || ny >= static_cast<long long>(height)) continue;
        const uint nx = wrapX(static_cast<long long>(x) + dx);
        const std::size_t i = static_cast<std::size_t>(ny) * width + nx;
        if (inRegion[i] || luminance[i] < threshold) continue;
        const Direction d = texelDirection(nx, static_cast<uint>(ny), width, height);
        if (d.x * seed.x + d.y * seed.y + d.z * seed.z < cosMaximum) continue;
        inRegion[i] = 1;
        region.push_back(i);
      }
  }

  // What replaces it: the mean colour just outside, weighted by solid angle.
  std::array<double, 3> fill{};
  double fillWeight = 0.0;
  for (const std::size_t r : region) {
    const uint x = static_cast<uint>(r % width), y = static_cast<uint>(r / width);
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        const long long ny = static_cast<long long>(y) + dy;
        if (ny < 0 || ny >= static_cast<long long>(height)) continue;
        const std::size_t i = static_cast<std::size_t>(ny) * width + wrapX(static_cast<long long>(x) + dx);
        if (inRegion[i] || luminance[i] >= threshold) continue;
        if (!std::isfinite(rgba[i * 4]) || !std::isfinite(rgba[i * 4 + 1]) || !std::isfinite(rgba[i * 4 + 2]))
          continue;
        const double w = texelSolidAngle(static_cast<uint>(ny), width, height);
        for (int c = 0; c < 3; ++c) fill[c] += w * rgba[i * 4 + c];
        fillWeight += w;
      }
  }
  for (double &c : fill) c = fillWeight > 0.0 ? c / fillWeight : 0.0;

  // Remove it. A texel never gains a channel, so the removed radiance is non-negative and
  // exactly what the disc carries.
  std::array<double, 3> removed{};
  Direction weighted{0.0, 0.0, 0.0};
  double totalWeight = 0.0;
  std::vector<double> energy(region.size());
  for (std::size_t k = 0; k < region.size(); ++k) {
    const std::size_t r = region[k];
    const uint x = static_cast<uint>(r % width), y = static_cast<uint>(r / width);
    const double solidAngle = texelSolidAngle(y, width, height);
    float *texel = &rgba[r * 4];
    double taken[3];
    for (int c = 0; c < 3; ++c) {
      const float replacement = std::min(static_cast<float>(fill[c]), texel[c]);
      taken[c] = static_cast<double>(texel[c]) - static_cast<double>(replacement);
      texel[c] = replacement;
      removed[c] += taken[c] * solidAngle;
    }
    const double w = (0.2126 * taken[0] + 0.7152 * taken[1] + 0.0722 * taken[2]) * solidAngle;
    const Direction d = texelDirection(x, y, width, height);
    weighted = {weighted.x + w * d.x, weighted.y + w * d.y, weighted.z + w * d.z};
    energy[k] = w;
    totalWeight += w;
  }
  const double removedLuminance = 0.2126 * removed[0] + 0.7152 * removed[1] + 0.0722 * removed[2];
  if (!(totalWeight > 0.0) || !(removedLuminance > 0.0)) return sun;
  const double norm = std::sqrt(weighted.x * weighted.x + weighted.y * weighted.y + weighted.z * weighted.z);
  const Direction centre = {weighted.x / norm, weighted.y / norm, weighted.z / norm};

  // A uniform disc of radius R has mean squared angle R^2 / 2 about its centre.
  double spread = 0.0;
  for (std::size_t k = 0; k < region.size(); ++k) {
    const uint x = static_cast<uint>(region[k] % width), y = static_cast<uint>(region[k] / width);
    const Direction d = texelDirection(x, y, width, height);
    const double angle = std::acos(std::clamp(d.x * centre.x + d.y * centre.y + d.z * centre.z, -1.0, 1.0));
    spread += energy[k] * angle * angle;
  }
  spread /= totalWeight;
  const double halfTexel = 0.5 * kPiD / static_cast<double>(height);
  sun.found = true;
  sun.direction = float3(static_cast<float>(centre.x), static_cast<float>(centre.y), static_cast<float>(centre.z));
  sun.irradiance = float3(static_cast<float>(removed[0]), static_cast<float>(removed[1]), static_cast<float>(removed[2]));
  sun.angularRadius = static_cast<float>(std::clamp(std::sqrt(2.0 * spread), halfTexel, kMaximumRadius));
  sun.share = power > 0.0 ? static_cast<float>(removedLuminance / power) : 0.0f;
  sun.texels = static_cast<uint>(region.size());
  return sun;
}

void environmentSunUniforms(float3 direction, float3 irradiance, float angularRadius, Vector<float, 4> &sunDirection,
                            Vector<float, 4> &sunRadiance) {
  sunDirection = float4(0.0f);
  sunRadiance = float4(0.0f);
  if (!(std::max(irradiance.x, std::max(irradiance.y, irradiance.z)) > 0.0f)) return;
  const double radius = std::max(static_cast<double>(angularRadius), 1e-4);
  const double solidAngle = 2.0 * kPiD * (1.0 - std::cos(radius));
  sunDirection = float4(normalize(direction), static_cast<float>(std::cos(radius)));
  sunRadiance = float4(static_cast<float>(irradiance.x / solidAngle), static_cast<float>(irradiance.y / solidAngle),
                       static_cast<float>(irradiance.z / solidAngle), static_cast<float>(solidAngle));
}

} // namespace pt
