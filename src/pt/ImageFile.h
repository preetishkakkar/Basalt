// Linear float image files for captures (V8): PFM and OpenEXR, chosen by the path's
// extension. Both hold exactly the 32-bit floats given (no conversion, no compression), so a
// capture round-trips bit for bit.
//
// PFM ("PF", little-endian RGB float rows from the bottom up) is the historical format every
// comparison tool reads. OpenEXR files are single-part scanline images with NO_COMPRESSION
// and FLOAT channels B, G, R (stored in the standard's alphabetical order), rows from the top
// down (INCREASING_Y), linear scene-referred Rec. 709 / sRGB primaries with a D65 white point
// (the chromaticities attribute), and a string attribute basaltContent naming the plane
// (e.g. "raw-linear-rgb").
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace pt {

// rgb: width * height * 3 floats, rows from the bottom up (PFM order). A path ending in .exr
// (any case) writes OpenEXR, any other PFM. `content` labels an OpenEXR file; PFM has no
// place for it. False when the file cannot be written.
bool writeLinearImage(const std::string &path, const std::vector<float> &rgb, std::uint32_t width,
                      std::uint32_t height, const std::string &content = "raw-linear-rgb");

bool isExrPath(const std::string &path);

// What readExr returns: rgb rows from the bottom up (as writeLinearImage takes them), the
// channel names in file order and the string attributes.
struct LinearImage {
  std::uint32_t width = 0, height = 0;
  std::vector<float> rgb;
  std::vector<std::string> channels;
  std::vector<std::pair<std::string, std::string>> strings;
};

// Reads an OpenEXR file of the kind writeLinearImage writes (single-part scanline,
// uncompressed, FLOAT R, G and B channels); throws std::runtime_error with the reason
// otherwise.
LinearImage readExr(const std::string &path);

} // namespace pt
