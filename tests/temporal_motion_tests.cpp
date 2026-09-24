#include "temporal_motion_fixture.h"
#include "temporal_motion_v7_fixture.h"
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
using namespace pt;
void require(bool condition, const std::string &why) { if (!condition) throw std::runtime_error(why); }

// A motion corpus: its sequences, their scene and camera per frame, and the directory of its
// frozen references. Both corpora share frames, size, seeds and reference samples.
struct Corpus {
  const char *directory;
  std::vector<const char *> sequences;
  void (*scene)(pt_fixture::Builder &, uint sequence);
  void (*camera)(pt_fixture::Builder &, uint sequence, uint frame);
};
const Corpus kV31{"temporal", {temporal_motion::sequences.begin(), temporal_motion::sequences.end()},
                  [](pt_fixture::Builder &b, uint) { temporal_motion::scene(b); }, temporal_motion::camera};
const Corpus kV7{"temporal_v7", {temporal_motion_v7::sequences.begin(), temporal_motion_v7::sequences.end()},
                 temporal_motion_v7::scene, temporal_motion_v7::camera};

std::string referencePath(const Corpus &corpus, uint sequence) {
  return std::string(BASALT_TEST_DATA_DIR) + "/" + corpus.directory + "/" + corpus.sequences[sequence] + ".rgb32f";
}
std::vector<float> referenceFrame(pt_fixture::Builder &b, uint samples) {
  b.frame.uniforms.image.z = 0;
  b.frame.uniforms.counts.z = 1701;
  std::vector<float> image(temporal_motion::width * temporal_motion::height * 3);
  pt_fixture::sharedPool().parallelFor(temporal_motion::width * temporal_motion::height, [&](uint i) {
    double sum[3]{};
    for (uint s = 0; s < samples; ++s) {
      const auto sample = CpuTracer::tracePixel(b.scene, b.frame, i % temporal_motion::width,
                                               i / temporal_motion::width, s);
      sum[0] += sample.radiance.x; sum[1] += sample.radiance.y; sum[2] += sample.radiance.z;
    }
    for (uint c = 0; c < 3; ++c) image[i * 3 + c] = float(sum[c] / samples);
  });
  return image;
}
void references(const Corpus &corpus) {
  for (uint sequence = 0; sequence < corpus.sequences.size(); ++sequence) {
    pt_fixture::Builder b; corpus.scene(b, sequence);
    std::filesystem::create_directories(std::filesystem::path(referencePath(corpus, sequence)).parent_path());
    std::ofstream file(referencePath(corpus, sequence), std::ios::binary);
    require(bool(file), "cannot create reference");
    for (uint f = 0; f < temporal_motion::frames; ++f) {
      corpus.camera(b, sequence, f);
      const auto image = referenceFrame(b, temporal_motion::referenceSamples);
      file.write(reinterpret_cast<const char *>(image.data()), image.size() * sizeof(float));
      std::printf("reference %s frame %u/%u at %u SPP\n", corpus.sequences[sequence], f + 1,
          temporal_motion::frames, temporal_motion::referenceSamples); std::fflush(stdout);
    }
  }
}
int convergence(const Corpus &corpus) {
  int failures = 0;
  for (uint seq = 0; seq < corpus.sequences.size(); ++seq) {
    pt_fixture::Builder b; corpus.scene(b, seq);
    std::ifstream file(referencePath(corpus, seq), std::ios::binary);
    double error = 0, energy = 0;
    for (uint f = 0; f < temporal_motion::frames; ++f) {
      corpus.camera(b, seq, f);
      const auto twice = referenceFrame(b, temporal_motion::referenceSamples * 2);
      std::vector<float> ref(twice.size());
      file.read(reinterpret_cast<char *>(ref.data()), ref.size() * sizeof(float));
      require(bool(file), "missing reference for convergence check");
      for (std::size_t i = 0; i < ref.size(); ++i) { error += std::abs(ref[i] - twice[i]); energy += twice[i]; }
    }
    const double relative = error / std::max(energy, 1e-3);
    std::printf("%s reference convergence %s 4096->8192 normalized MAE %.6f (maximum 0.01)\n",
        relative <= 0.01 ? "PASS" : "FAIL", corpus.sequences[seq], relative);
    if (relative > 0.01) ++failures;
  }
  return failures == 0 ? 0 : 1;
}
int quality(const Corpus &corpus, const char *reportPath, bool embree = false) {
  if (embree && !EmbreeScene::available()) { std::printf("SKIP: Embree unavailable\n"); return 77; }
  constexpr uint w = temporal_motion::width, h = temporal_motion::height, pixels = w * h;
  std::ofstream report(reportPath);
  require(bool(report), "cannot create metric report");
  report << "sequence,seed,raw_mae,filtered_mae,raw_flicker,filtered_flicker\n";
  int failures = 0;
  for (uint sequence = 0; sequence < corpus.sequences.size(); ++sequence) {
    pt_fixture::Builder b; corpus.scene(b, sequence);
    if (embree) { b.scene.embree = std::make_shared<EmbreeScene>(b.scene); b.frame.intersector = 1; }
    std::ifstream file(referencePath(corpus, sequence), std::ios::binary);
    std::vector<float> reference(pixels * 3 * temporal_motion::frames);
    file.read(reinterpret_cast<char *>(reference.data()), reference.size() * sizeof(float));
    require(file.gcount() == static_cast<std::streamsize>(reference.size() * sizeof(float)), "missing/truncated frozen reference");
    double totalRaw = 0, totalFiltered = 0, totalRawFlicker = 0, totalFilteredFlicker = 0;
    for (uint seed : temporal_motion::seeds) {
      std::vector<PtReconstructionSample> old(pixels), samples(pixels);
      std::vector<PtTemporalHistory> history(pixels);
      std::vector<float3> previousRawError(pixels), previousFilteredError(pixels);
      PathUniforms previousCamera{};
      uint stationarySamples = 0;
      std::vector<float3> rawSum(pixels);
      double rawError = 0, filteredError = 0, rawFlicker = 0, filteredFlicker = 0, energy = 0;
      for (uint f = 0; f < temporal_motion::frames; ++f) {
        corpus.camera(b, sequence, f);
        b.frame.uniforms.counts.z = seed;
        const bool stationary = f != 0 && std::memcmp(&previousCamera.cameraPosition, &b.frame.uniforms.cameraPosition,
            sizeof(float4) * 4) == 0;
        if (!stationary) { stationarySamples = 0; std::fill(rawSum.begin(), rawSum.end(), float3(0)); }
        std::vector<float3> raw(pixels);
        pt_fixture::sharedPool().parallelFor(pixels, [&](uint i) {
          const auto path = CpuTracer::tracePixel(b.scene, b.frame, i % w, i / w, stationarySamples);
          samples[i] = path.guide; rawSum[i] += path.radiance;
          raw[i] = rawSum[i] / float(stationarySamples + 1);
        });
        ++stationarySamples;
        PtTemporalUniforms u{};
        u.previousCamera = f == 0 ? b.frame.uniforms : previousCamera;
        u.image = uint4(w, h, f == 0 ? 1 : 0, 1);
        history = temporal_fixture::temporal(u, samples, old, history);
        const auto filtered = temporal_fixture::filter(u, samples, history);
        for (uint i = 0; i < pixels; ++i) {
          const float *r = &reference[(f * pixels + i) * 3];
          const float3 ref(r[0], r[1], r[2]);
          const float3 output = xyz(filtered[i].diffuse) + xyz(filtered[i].specular);
          require(ptFiniteColor(output) && ptFiniteColor(raw[i]), "nonfinite motion output");
          const float3 re = raw[i] - ref, fe = output - ref;
          auto sum = [](float3 v) { return double(v.x) + v.y + v.z; };
          rawError += sum(abs(re)); filteredError += sum(abs(fe)); energy += ptTemporalLuminance(ref) * 3;
          if (f != 0) {
            rawFlicker += sum(abs(re - previousRawError[i]));
            filteredFlicker += sum(abs(fe - previousFilteredError[i]));
          }
          previousRawError[i] = re; previousFilteredError[i] = fe;
        }
        old = samples; previousCamera = b.frame.uniforms;
      }
      const double scale = 1.0 / std::max(energy, double(pixels * temporal_motion::frames * 3) * 0.001);
      report << corpus.sequences[sequence] << ',' << seed << ',' << rawError * scale << ','
             << filteredError * scale << ',' << rawFlicker * scale << ',' << filteredFlicker * scale << '\n';
      totalRaw += rawError; totalFiltered += filteredError;
      totalRawFlicker += rawFlicker; totalFilteredFlicker += filteredFlicker;
    }
    const bool pass = totalFiltered <= 0.9 * totalRaw && totalFilteredFlicker <= 0.9 * totalRawFlicker;
    std::printf("%s %s spatial %.5f, flicker %.5f (maximum 0.90 each)\n", pass ? "PASS" : "FAIL",
        corpus.sequences[sequence], totalFiltered / totalRaw, totalFilteredFlicker / totalRawFlicker);
    if (!pass) ++failures;
  }
  return failures == 0 ? 0 : 1;
}
}
int main(int argc, char **argv) {
  try {
    // --v7 selects the V7 corpus (glass, clearcoat, depth of field) for any mode.
    std::vector<std::string> args(argv + 1, argv + argc);
    const bool v7 = !args.empty() && args[0] == "--v7";
    if (v7) args.erase(args.begin());
    const Corpus &corpus = v7 ? kV7 : kV31;
    if (args.size() == 1 && args[0] == "--generate-references") { references(corpus); return 0; }
    if (args.size() == 1 && args[0] == "--check-reference-convergence") return convergence(corpus);
    if (!args.empty() && args[0] == "--embree")
      return quality(corpus, args.size() > 1 ? args[1].c_str() : v7 ? "temporal-v7-embree-results.csv"
                                                                   : "temporal-embree-results.csv", true);
    return quality(corpus, !args.empty() ? args[0].c_str() : v7 ? "temporal-v7-motion-results.csv"
                                                               : "temporal-motion-results.csv");
  } catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
