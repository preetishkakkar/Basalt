#include "pt/ImageFile.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace pt {
namespace {

// OpenEXR (openexr.com, "OpenEXR File Layout"): little-endian throughout.
constexpr std::uint32_t kExrMagic = 20000630u;
constexpr std::uint32_t kExrVersion = 2u;  // single-part scanline, short names
constexpr std::int32_t kPixelFloat = 2;
constexpr std::uint8_t kNoCompression = 0, kIncreasingY = 0;

class Writer {
public:
  std::vector<char> bytes;
  template <class T> void put(T value) {
    const char *p = reinterpret_cast<const char *>(&value);
    bytes.insert(bytes.end(), p, p + sizeof(T));
  }
  void text(const std::string &s) { bytes.insert(bytes.end(), s.begin(), s.end()); bytes.push_back('\0'); }
  void attribute(const std::string &name, const std::string &type, const std::vector<char> &value) {
    text(name);
    text(type);
    put(static_cast<std::int32_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
  }
};

template <class... T> std::vector<char> pack(T... values) {
  Writer w;
  (w.put(values), ...);
  return w.bytes;
}

bool writeExr(const std::string &path, const std::vector<float> &rgb, std::uint32_t width, std::uint32_t height,
              const std::string &content) {
  Writer w;
  w.put(kExrMagic);
  w.put(kExrVersion);
  // chlist: name, pixel type, pLinear + 3 reserved bytes, x and y sampling; then a null.
  {
    Writer channels;
    for (const char *name : {"B", "G", "R"}) {
      channels.text(name);
      channels.put(kPixelFloat);
      channels.put(static_cast<std::uint32_t>(0));
      channels.put(static_cast<std::int32_t>(1));
      channels.put(static_cast<std::int32_t>(1));
    }
    channels.bytes.push_back('\0');
    w.attribute("channels", "chlist", channels.bytes);
  }
  w.attribute("compression", "compression", {static_cast<char>(kNoCompression)});
  const std::vector<char> window = pack(0, 0, static_cast<std::int32_t>(width) - 1, static_cast<std::int32_t>(height) - 1);
  w.attribute("dataWindow", "box2i", window);
  w.attribute("displayWindow", "box2i", window);
  w.attribute("lineOrder", "lineOrder", {static_cast<char>(kIncreasingY)});
  w.attribute("pixelAspectRatio", "float", pack(1.0f));
  w.attribute("screenWindowCenter", "v2f", pack(0.0f, 0.0f));
  w.attribute("screenWindowWidth", "float", pack(1.0f));
  // Rec. 709 / sRGB primaries, D65 white: red, green, blue, white (x, y).
  w.attribute("chromaticities", "chromaticities",
              pack(0.64f, 0.33f, 0.30f, 0.60f, 0.15f, 0.06f, 0.3127f, 0.3290f));
  w.attribute("basaltContent", "string", std::vector<char>(content.begin(), content.end()));
  w.bytes.push_back('\0');  // end of header

  // One uncompressed scanline per chunk: the offset table, then y, byte count, B, G, R.
  const std::uint64_t lineBytes = static_cast<std::uint64_t>(width) * 3u * sizeof(float);
  const std::uint64_t tableStart = w.bytes.size();
  const std::uint64_t firstChunk = tableStart + static_cast<std::uint64_t>(height) * 8u;
  for (std::uint32_t y = 0; y < height; ++y) w.put(firstChunk + y * (8u + lineBytes));
  std::vector<float> plane(width);
  for (std::uint32_t y = 0; y < height; ++y) {
    w.put(static_cast<std::int32_t>(y));
    w.put(static_cast<std::int32_t>(lineBytes));
    const float *row = rgb.data() + static_cast<std::size_t>(height - 1 - y) * width * 3u;  // top down
    for (int channel = 2; channel >= 0; --channel) {  // B, G, R
      for (std::uint32_t x = 0; x < width; ++x) plane[x] = row[x * 3u + static_cast<std::uint32_t>(channel)];
      const char *p = reinterpret_cast<const char *>(plane.data());
      w.bytes.insert(w.bytes.end(), p, p + width * sizeof(float));
    }
  }
  std::ofstream file(path, std::ios::binary);
  file.write(w.bytes.data(), static_cast<std::streamsize>(w.bytes.size()));
  return static_cast<bool>(file);
}

class Reader {
public:
  explicit Reader(std::vector<char> data) : bytes(std::move(data)) {}
  // Every check is written as `n > size - at`, so no sum can wrap on a malformed offset.
  template <class T> T get() {
    if (at > bytes.size() || sizeof(T) > bytes.size() - at) throw std::runtime_error("OpenEXR file ends early");
    T value;
    std::memcpy(&value, bytes.data() + at, sizeof(T));
    at += sizeof(T);
    return value;
  }
  std::string text() {
    std::string s;
    while (true) {
      if (at >= bytes.size()) throw std::runtime_error("OpenEXR header ends early");
      const char c = bytes[at++];
      if (c == '\0') return s;
      s.push_back(c);
    }
  }
  std::vector<char> take(std::size_t n) {
    if (at > bytes.size() || n > bytes.size() - at) throw std::runtime_error("OpenEXR file ends early");
    std::vector<char> out(bytes.begin() + static_cast<std::ptrdiff_t>(at),
                          bytes.begin() + static_cast<std::ptrdiff_t>(at + n));
    at += n;
    return out;
  }
  std::vector<char> bytes;
  std::size_t at = 0;
};

} // namespace

bool isExrPath(const std::string &path) {
  if (path.size() < 4) return false;
  std::string extension = path.substr(path.size() - 4);
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return extension == ".exr";
}

bool writeLinearImage(const std::string &path, const std::vector<float> &rgb, std::uint32_t width,
                      std::uint32_t height, const std::string &content) {
  if (rgb.size() != static_cast<std::size_t>(width) * height * 3u || width == 0 || height == 0) return false;
  if (isExrPath(path)) return writeExr(path, rgb, width, height, content);
  std::ofstream file(path, std::ios::binary);
  file << "PF\n" << width << ' ' << height << "\n-1.0\n";
  file.write(reinterpret_cast<const char *>(rgb.data()), static_cast<std::streamsize>(rgb.size() * sizeof(float)));
  return static_cast<bool>(file);
}

LinearImage readExr(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open " + path);
  Reader r(std::vector<char>(std::istreambuf_iterator<char>(file), {}));
  if (r.get<std::uint32_t>() != kExrMagic) throw std::runtime_error(path + " is not an OpenEXR file");
  if (r.get<std::uint32_t>() != kExrVersion) throw std::runtime_error(path + ": only single-part scanline files");
  LinearImage image;
  std::int32_t box[4] = {0, 0, -1, -1};
  bool uncompressed = false, increasing = false;
  for (std::string name = r.text(); !name.empty(); name = r.text()) {
    const std::string type = r.text();
    const std::int32_t size = r.get<std::int32_t>();
    if (size < 0) throw std::runtime_error(path + ": negative attribute size");
    Reader value(r.take(static_cast<std::size_t>(size)));
    if (name == "channels") {
      for (std::string channel = value.text(); !channel.empty(); channel = value.text()) {
        if (value.get<std::int32_t>() != kPixelFloat) throw std::runtime_error(path + ": only FLOAT channels");
        value.get<std::uint32_t>();
        if (value.get<std::int32_t>() != 1 || value.get<std::int32_t>() != 1)
          throw std::runtime_error(path + ": only unsampled channels");
        image.channels.push_back(channel);
      }
    } else if (name == "compression") {
      uncompressed = value.get<std::uint8_t>() == kNoCompression;
    } else if (name == "dataWindow") {
      for (std::int32_t &b : box) b = value.get<std::int32_t>();
    } else if (name == "lineOrder") {
      increasing = value.get<std::uint8_t>() == kIncreasingY;
    } else if (type == "string") {
      image.strings.emplace_back(name, std::string(value.bytes.begin(), value.bytes.end()));
    }
  }
  if (!uncompressed || !increasing) throw std::runtime_error(path + ": only uncompressed, increasing-y files");
  if (image.channels != std::vector<std::string>{"B", "G", "R"}) throw std::runtime_error(path + ": expects B, G, R");
  if (box[0] != 0 || box[1] != 0 || box[2] < 0 || box[3] < 0) throw std::runtime_error(path + ": unexpected data window");
  image.width = static_cast<std::uint32_t>(box[2]) + 1u;
  image.height = static_cast<std::uint32_t>(box[3]) + 1u;
  const std::size_t lineBytes = static_cast<std::size_t>(image.width) * 3u * sizeof(float);
  // The file must hold the offset table and every scanline before anything is allocated.
  const std::uint64_t remaining = r.bytes.size() - r.at;
  if (static_cast<std::uint64_t>(image.height) * 8u > remaining ||
      static_cast<std::uint64_t>(image.height) * (8u + lineBytes) > remaining - static_cast<std::uint64_t>(image.height) * 8u)
    throw std::runtime_error(path + ": data window larger than the file");
  std::vector<std::uint64_t> offsets(image.height);
  for (std::uint64_t &offset : offsets) offset = r.get<std::uint64_t>();
  image.rgb.assign(static_cast<std::size_t>(image.width) * image.height * 3u, 0.0f);
  std::vector<float> plane(image.width);
  for (std::uint32_t chunk = 0; chunk < image.height; ++chunk) {
    if (offsets[chunk] >= r.bytes.size()) throw std::runtime_error(path + ": scanline offset outside the file");
    r.at = static_cast<std::size_t>(offsets[chunk]);
    const std::int32_t y = r.get<std::int32_t>();
    if (y < 0 || static_cast<std::uint32_t>(y) >= image.height || r.get<std::int32_t>() != static_cast<std::int32_t>(lineBytes))
      throw std::runtime_error(path + ": malformed scanline");
    float *row = image.rgb.data() + static_cast<std::size_t>(image.height - 1u - static_cast<std::uint32_t>(y)) * image.width * 3u;
    for (int channel = 2; channel >= 0; --channel) {
      const std::vector<char> data = r.take(image.width * sizeof(float));
      std::memcpy(plane.data(), data.data(), data.size());
      for (std::uint32_t x = 0; x < image.width; ++x) row[x * 3u + static_cast<std::uint32_t>(channel)] = plane[x];
    }
  }
  return image;
}

} // namespace pt
