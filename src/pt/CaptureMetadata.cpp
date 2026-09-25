#include "pt/CaptureMetadata.h"

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace pt {
namespace {

std::string quote(const std::string &text) {
  std::string out = "\"";
  for (const char c : text) {
    switch (c) {
    case '"': out += "\\\""; break;
    case '\\': out += "\\\\"; break;
    case '\n': out += "\\n"; break;
    case '\r': out += "\\r"; break;
    case '\t': out += "\\t"; break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        char escaped[8];
        std::snprintf(escaped, sizeof(escaped), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
        out += escaped;
      } else {
        out += c;
      }
    }
  }
  return out + "\"";
}

std::string hex(std::uint64_t value) {
  std::ostringstream text;
  text << std::hex << std::setw(16) << std::setfill('0') << value;
  return quote(text.str());
}

std::string number(double value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream text;
  text << std::setprecision(9) << value;
  return text.str();
}

const char *boolean(bool value) { return value ? "true" : "false"; }

} // namespace

std::uint64_t hashFile(const std::string &path) {
  if (path.empty()) return 0;
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot hash " + path);
  std::uint64_t hash = 14695981039346656037ull;
  std::array<char, 65536> bytes{};
  while (input) {
    input.read(bytes.data(), bytes.size());
    for (std::streamsize i = 0; i < input.gcount(); ++i) {
      hash ^= static_cast<unsigned char>(bytes[static_cast<std::size_t>(i)]);
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

bool writeCaptureMetadata(const std::string &path, const CaptureMetadata &m) {
  std::ofstream out(path, std::ios::binary);
  out << "{\n  \"schema\": \"basalt-capture/1\""
      << ",\n  \"source\": {\"revision\": " << quote(m.revision)
      << ", \"dirty_at_configure\": " << boolean(m.dirtyAtConfigure)
      << ", \"shader_compiler\": " << (m.shaderCompiler.empty() ? std::string("null") : quote(m.shaderCompiler)) << "}"
      << ",\n  \"scene\": {\"path\": " << quote(m.scene) << ", \"file_fnv1a64\": " << hex(m.sceneFileHash)
      << ", \"content_fnv1a64\": " << (m.sceneContentHash ? hex(m.sceneContentHash) : std::string("null"))
      << ", \"environment\": " << quote(m.environment) << "}"
      << ",\n  \"device\": {\"name\": " << quote(m.device) << ", \"driver_version\": " << m.driverVersion
      << ", \"api_version\": " << m.apiVersion << ", \"acceleration_structures\": " << boolean(m.accelerationStructures)
      << ", \"ray_query\": " << boolean(m.rayQuery) << ", \"ray_pipeline\": " << boolean(m.rayPipeline) << "}"
      << ",\n  \"backend\": {\"renderer\": " << quote(m.renderer) << ", \"intersector\": " << quote(m.intersector)
      << ", \"builder\": " << quote(m.builder) << ", \"bvh_layout\": " << quote(m.bvhLayout)
      << ", \"bvh_update\": " << quote(m.bvhUpdate) << ", \"animation\": " << quote(m.animation)
      << ", \"wide_stack\": " << quote(m.wideStack)
      << ", \"execution\": " << quote(m.execution) << ", \"cpu_threads\": " << m.cpuThreads << "}"
      << ",\n  \"integrator\": {\"spp\": " << m.spp << ", \"target_spp\": " << m.targetSpp
      << ", \"samples_per_dispatch\": " << m.samplesPerDispatch << ", \"bounces\": " << m.bounces
      << ", \"roulette_start\": " << number(m.rouletteStart) << ", \"strategy\": " << quote(m.strategy)
      << ", \"seed\": " << m.seed << ", \"firefly_clamp\": " << number(m.fireflyClamp)
      << ", \"texture_filter\": " << quote(m.textureFilter) << ", \"aperture\": " << number(m.aperture)
      << ", \"focus_distance\": " << number(m.focusDistance)
      << ", \"di_estimator\": " << quote(m.diEstimator) << ", \"restir_reuse\": " << quote(m.restirReuse)
      << ", \"restir_candidates\": " << m.restirCandidates << ", \"restir_bias\": " << quote(m.restirBias) << "}"
      << ",\n  \"image\": {\"width\": " << m.width << ", \"height\": " << m.height
      << ", \"raw_color_space\": \"linear scene-referred RGB\", \"pfm_rows\": \"bottom-to-top, little-endian float32\""
      << ", \"exr_rows\": \"top-to-bottom, little-endian float32, channels B G R\"}"
      << ",\n  \"display\": {\"reconstruction\": " << quote(m.reconstruction) << ", \"denoise\": " << quote(m.denoise)
      << ", \"changes_raw_output\": false}";
  out << ",\n  \"outputs\": [";
  for (std::size_t i = 0; i < m.outputs.size(); ++i)
    out << (i ? ", " : "") << "{\"path\": " << quote(m.outputs[i].first) << ", \"kind\": " << quote(m.outputs[i].second) << "}";
  out << "],\n  \"measurements\": {";
  for (std::size_t i = 0; i < m.measurements.size(); ++i)
    out << (i ? ", " : "") << quote(m.measurements[i].first) << ": " << number(m.measurements[i].second);
  out << "},\n  \"series\": {";
  for (std::size_t i = 0; i < m.series.size(); ++i) {
    out << (i ? ", " : "") << quote(m.series[i].first) << ": [";
    for (std::size_t j = 0; j < m.series[i].second.size(); ++j) out << (j ? ", " : "") << number(m.series[i].second[j]);
    out << "]";
  }
  out << "},\n  \"notes\": {";
  for (std::size_t i = 0; i < m.notes.size(); ++i)
    out << (i ? ", " : "") << quote(m.notes[i].first) << ": " << quote(m.notes[i].second);
  out << "}\n}\n";
  return static_cast<bool>(out);
}

} // namespace pt
