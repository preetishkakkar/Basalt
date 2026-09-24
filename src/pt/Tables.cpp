#include "pt/Tables.h"

#include <algorithm>
#include <cmath>
#include <thread>

namespace pt {

void buildEnvironmentDistribution(const float *rgba, uint width, uint height, std::vector<float> &distribution,
                                  float4 &info) {
  const uint columns = std::max(1u, std::min(width, 1024u));
  const uint rows = std::max(1u, std::min(height, 512u));
  const std::size_t functionSize = static_cast<std::size_t>(rows) * columns;
  const std::size_t conditionalOffset = functionSize;
  const std::size_t marginalOffset = conditionalOffset + static_cast<std::size_t>(rows) * (columns + 1);
  const std::size_t marginalCdfOffset = marginalOffset + rows;
  distribution.assign(marginalCdfOffset + rows + 1, 0.0f);

  for (uint row = 0; row < rows; ++row) {
    const uint y0 = row * height / rows, y1 = std::max(y0 + 1, (row + 1) * height / rows);
    const float sinTheta = std::sin((static_cast<float>(row) + 0.5f) / static_cast<float>(rows) * kPi);
    for (uint column = 0; column < columns; ++column) {
      const uint x0 = column * width / columns, x1 = std::max(x0 + 1, (column + 1) * width / columns);
      double sum = 0.0;
      for (uint y = y0; y < y1; ++y)
        for (uint x = x0; x < x1; ++x) {
          const float *t = rgba + (static_cast<std::size_t>(y) * width + x) * 4;
          const float luminance = 0.2126f * t[0] + 0.7152f * t[1] + 0.0722f * t[2];
          // A non-finite or negative texel carries no probability, rather than poisoning all of it.
          if (std::isfinite(luminance) && luminance > 0.0f) sum += luminance;
        }
      distribution[static_cast<std::size_t>(row) * columns + column] =
          static_cast<float>(sum / static_cast<double>((y1 - y0) * (x1 - x0))) * sinTheta;
    }
  }

  // Each row's CDF, and the marginal over rows, both normalised to end at one.
  double total = 0.0;
  std::vector<double> rowIntegrals(rows, 0.0);
  for (uint row = 0; row < rows; ++row) {
    const float *function = distribution.data() + static_cast<std::size_t>(row) * columns;
    float *cdf = distribution.data() + conditionalOffset + static_cast<std::size_t>(row) * (columns + 1);
    double running = 0.0;
    std::vector<double> sums(columns + 1, 0.0);
    for (uint column = 0; column < columns; ++column) {
      running += function[column] / static_cast<double>(columns);
      sums[column + 1] = running;
    }
    rowIntegrals[row] = running;
    for (uint column = 0; column <= columns; ++column)
      cdf[column] = running > 0.0 ? static_cast<float>(sums[column] / running)
                                  : static_cast<float>(column) / static_cast<float>(columns);
    distribution[marginalOffset + row] = static_cast<float>(running);
    total += running / static_cast<double>(rows);
  }
  double running = 0.0;
  distribution[marginalCdfOffset] = 0.0f;
  for (uint row = 0; row < rows; ++row) {
    running += rowIntegrals[row] / static_cast<double>(rows);
    distribution[marginalCdfOffset + row + 1] =
        total > 0.0 ? static_cast<float>(running / total) : static_cast<float>(row + 1) / static_cast<float>(rows);
  }
  distribution[marginalCdfOffset + rows] = 1.0f;
  info = float4(static_cast<float>(columns), static_cast<float>(rows), static_cast<float>(total), total > 0.0 ? 1.0f : 0.0f);
}

std::vector<float> buildSpecularAlbedoTable() {
  const uint size = kAlbedoTableSize;
  const uint samples = 4096;
  std::vector<float> table(static_cast<std::size_t>(size) * size, 1.0f);
  for (uint y = 0; y < size; ++y) {
    const float roughness = static_cast<float>(y) / static_cast<float>(size - 1);
    const float alpha = std::max(roughness * roughness, 1e-6f);
    for (uint x = 0; x < size; ++x) {
      const float cosine = std::max(static_cast<float>(x) / static_cast<float>(size - 1), 1e-4f);
      const float3 view(std::sqrt(std::max(0.0f, 1.0f - cosine * cosine)), 0.0f, cosine);
      double sum = 0.0;
      for (uint i = 0; i < samples; ++i) {
        // Hammersley points, offset half a cell so none sits on the xi = 0 boundary.
        const float2 xi((static_cast<float>(i) + 0.5f) / static_cast<float>(samples),
                        fract(radicalInverse(i) + 0.5f / static_cast<float>(samples)));
        const float3 half = ptSampleVisibleNormal(view, alpha, xi);
        const float3 light = reflect(-view, half);
        if (light.z <= 0.0f) continue;
        // With visible-normal sampling, f cos / pdf for Fresnel at one is G2 / G1(view).
        const float g2 = visibilitySmith(cosine, light.z, alpha) * 4.0f * light.z * cosine;
        sum += g2 / ptSmithG1(cosine, alpha);
      }
      table[static_cast<std::size_t>(y) * size + x] = static_cast<float>(sum / samples);
    }
  }

  // Closed dielectric interfaces: every visible normal splits the view ray by Schlick F into
  // its reflection and its refraction; with visible-normal sampling, each branch's f cos / pdf
  // in importance mode is G2 / G1(view) (the refraction's Jacobian and eta^2 cancel).
  const std::size_t slice = static_cast<std::size_t>(size) * size;
  table.resize(slice * (1 + 2 * kDielectricIorSteps), 1.0f);
  const uint dielectricSamples = 1024;
  const uint slices = 2 * kDielectricIorSteps;
  std::vector<std::thread> workers;
  const uint threads = std::max(1u, std::min(slices, std::thread::hardware_concurrency()));
  for (uint t = 0; t < threads; ++t)
    workers.emplace_back([&, t] {
      for (uint s = t; s < slices; s += threads) {
        const bool leaving = s >= kDielectricIorSteps;
        const uint step = s % kDielectricIorSteps;
        const float ior = 1.0f + (kDielectricIorMax - 1.0f) * static_cast<float>(step) /
                                     static_cast<float>(kDielectricIorSteps - 1);
        const float etap = leaving ? 1.0f / ior : ior;
        const float f0 = ((ior - 1.0f) / (ior + 1.0f)) * ((ior - 1.0f) / (ior + 1.0f));
        float *out = table.data() + slice * (1 + s);
        for (uint y = 0; y < size; ++y) {
          const float roughness = static_cast<float>(y) / static_cast<float>(size - 1);
          const float alpha = std::max(roughness * roughness, 1e-6f);
          for (uint x = 0; x < size; ++x) {
            const float cosine = std::max(static_cast<float>(x) / static_cast<float>(size - 1), 1e-4f);
            const float3 view(std::sqrt(std::max(0.0f, 1.0f - cosine * cosine)), 0.0f, cosine);
            const float g1 = ptSmithG1(cosine, alpha);
            double sum = 0.0;
            for (uint i = 0; i < dielectricSamples; ++i) {
              const float2 xi((static_cast<float>(i) + 0.5f) / static_cast<float>(dielectricSamples),
                              fract(radicalInverse(i) + 0.5f / static_cast<float>(dielectricSamples)));
              const float3 half = ptSampleVisibleNormal(view, alpha, xi);
              const float fresnel = ptFresnelInterface(float3(f0), dot(view, half), etap).x;
              const float3 reflected = reflect(-view, half);
              if (reflected.z > 0.0f)
                sum += fresnel * visibilitySmith(cosine, reflected.z, alpha) * 4.0f * reflected.z * cosine / g1;
              const float3 refracted = refract(-view, half, 1.0f / etap);
              if (fresnel < 1.0f && refracted.z < 0.0f)
                sum += (1.0f - fresnel) * visibilitySmith(cosine, -refracted.z, alpha) * 4.0f * -refracted.z * cosine / g1;
            }
            out[y * size + x] = static_cast<float>(sum / dielectricSamples);
          }
        }
      }
    });
  for (std::thread &worker : workers) worker.join();
  return table;
}

} // namespace pt
