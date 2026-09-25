#include "pt/Tracing.h"

#include <algorithm>
#include <cmath>

namespace pt {
namespace {

// sRGB to linear for each byte value, built once.
const std::array<float, 256> &srgbTable() {
  static const std::array<float, 256> table = [] {
    std::array<float, 256> values{};
    for (int i = 0; i < 256; ++i) {
      const float c = static_cast<float>(i) / 255.0f;
      values[static_cast<std::size_t>(i)] = c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
    }
    return values;
  }();
  return table;
}

float4 decode(std::uint32_t texel, bool srgb) {
  const float r = static_cast<float>(texel & 0xFFu), g = static_cast<float>((texel >> 8) & 0xFFu);
  const float b = static_cast<float>((texel >> 16) & 0xFFu), a = static_cast<float>(texel >> 24);
  if (!srgb) return float4(r, g, b, a) * (1.0f / 255.0f);
  const std::array<float, 256> &table = srgbTable();
  return float4(table[texel & 0xFFu], table[(texel >> 8) & 0xFFu], table[(texel >> 16) & 0xFFu], a / 255.0f);
}

// mode: 0 repeat, 1 clamp to edge, 2 mirrored repeat (glTF's wrap, as packedSampler encodes it).
int wrap(int i, int size, uint mode) {
  if (mode == 1u) return std::clamp(i, 0, size - 1);
  if (mode == 2u) {
    const int period = size * 2;
    int p = i % period;
    if (p < 0) p += period;
    return p < size ? p : period - 1 - p;
  }
  const int m = i % size;
  return m < 0 ? m + size : m;
}

} // namespace

HostTextures::HostTextures() {
  // Slot zero white, slot one the flat normal, as the GPU table starts.
  HostTexture flat;
  flat.texels = {0xFFFF8080u};
  textures = {HostTexture{}, flat};
  slots.fill(0);
  slots[1] = 1;
}

namespace {

// One level: nearest (samplerCode bit 4) or bilinear with texel centres at half-integers, as
// the GPU's filters place them.
float4 sampleLevel(const std::uint32_t *texels, int w, int h, bool srgb, float2 uv, uint samplerCode) {
  const uint wrapU = samplerCode & 3u, wrapV = (samplerCode >> 2u) & 3u;
  if ((samplerCode & 16u) != 0u) {
    const int x = wrap(static_cast<int>(std::floor(uv.x * static_cast<float>(w))), w, wrapU);
    const int y = wrap(static_cast<int>(std::floor(uv.y * static_cast<float>(h))), h, wrapV);
    return decode(texels[static_cast<std::size_t>(y) * w + x], srgb);
  }
  const float x = uv.x * static_cast<float>(w) - 0.5f, y = uv.y * static_cast<float>(h) - 0.5f;
  const float fx = std::floor(x), fy = std::floor(y);
  const float tx = x - fx, ty = y - fy;
  const int x0 = wrap(static_cast<int>(fx), w, wrapU), x1 = wrap(static_cast<int>(fx) + 1, w, wrapU);
  const int y0 = wrap(static_cast<int>(fy), h, wrapV), y1 = wrap(static_cast<int>(fy) + 1, h, wrapV);
  const std::uint32_t *row0 = texels + static_cast<std::size_t>(y0) * w;
  const std::uint32_t *row1 = texels + static_cast<std::size_t>(y1) * w;
  const float4 top = mix(decode(row0[x0], srgb), decode(row0[x1], srgb), tx);
  const float4 bottom = mix(decode(row1[x0], srgb), decode(row1[x1], srgb), tx);
  return mix(top, bottom, ty);
}

std::uint8_t encodeChannel(float linear, bool srgb) {
  float c = std::clamp(linear, 0.0f, 1.0f);
  if (srgb) c = c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
  return static_cast<std::uint8_t>(std::lround(c * 255.0f));
}

} // namespace

void buildMipChain(HostTexture &texture) {
  texture.mips.clear();
  const std::uint32_t *source = texture.texels.data();
  uint w = texture.width, h = texture.height;
  while (w > 1 || h > 1) {
    HostTexture::Level level;
    level.width = std::max(1u, w / 2);
    level.height = std::max(1u, h / 2);
    level.texels.resize(static_cast<std::size_t>(level.width) * level.height);
    for (uint y = 0; y < level.height; ++y)
      for (uint x = 0; x < level.width; ++x) {
        float4 sum(0.0f);
        for (uint dy = 0; dy < 2; ++dy)
          for (uint dx = 0; dx < 2; ++dx) {
            const uint sx = std::min(2 * x + dx, w - 1), sy = std::min(2 * y + dy, h - 1);
            sum = sum + decode(source[static_cast<std::size_t>(sy) * w + sx], texture.srgb);
          }
        sum = sum * 0.25f;
        level.texels[static_cast<std::size_t>(y) * level.width + x] =
            static_cast<std::uint32_t>(encodeChannel(sum.x, texture.srgb)) |
            (static_cast<std::uint32_t>(encodeChannel(sum.y, texture.srgb)) << 8) |
            (static_cast<std::uint32_t>(encodeChannel(sum.z, texture.srgb)) << 16) |
            (static_cast<std::uint32_t>(encodeChannel(sum.w, false)) << 24);
      }
    texture.mips.push_back(std::move(level));
    source = texture.mips.back().texels.data();
    w = texture.mips.back().width;
    h = texture.mips.back().height;
  }
}

float HostTextures::footprint(uint slot, float lodBase) const {
  const HostTexture &texture = textures[slots[slot < kHitTextureSlots ? slot : 0]];
  return ptTextureFootprint(lodBase, static_cast<float>(texture.width), static_cast<float>(texture.height));
}

float4 HostTextures::sample(uint slot, float2 uv, uint samplerCode, float lodBase) const {
  const HostTexture &texture = textures[slots[slot < kHitTextureSlots ? slot : 0]];
  const int w = static_cast<int>(texture.width), h = static_cast<int>(texture.height);
  if (lodBase <= kPtLevelZero || texture.mips.empty())
    return sampleLevel(texture.texels.data(), w, h, texture.srgb, uv, samplerCode);
  // Trilinear between the two levels around the ray-cone level of detail.
  const float lod = ptTextureLevel(lodBase, static_cast<float>(w), static_cast<float>(h),
                                   static_cast<float>(texture.mips.size() + 1));
  const uint lower = static_cast<uint>(lod);
  const float fraction = lod - static_cast<float>(lower);
  auto at = [&](uint level) {
    if (level == 0) return sampleLevel(texture.texels.data(), w, h, texture.srgb, uv, samplerCode);
    const HostTexture::Level &mip = texture.mips[level - 1];
    return sampleLevel(mip.texels.data(), static_cast<int>(mip.width), static_cast<int>(mip.height), texture.srgb,
                       uv, samplerCode);
  };
  const float4 first = at(lower);
  if (fraction <= 0.0f || lower >= texture.mips.size()) return first;
  return mix(first, at(lower + 1), fraction);
}

float3 HostEnvironment::sample(float2 uv) const {
  const int w = static_cast<int>(width), h = static_cast<int>(height);
  const float x = uv.x * static_cast<float>(w) - 0.5f, y = uv.y * static_cast<float>(h) - 0.5f;
  const float fx = std::floor(x), fy = std::floor(y);
  const float tx = x - fx, ty = y - fy;
  const int x0 = wrap(static_cast<int>(fx), w, 0u), x1 = wrap(static_cast<int>(fx) + 1, w, 0u);
  const int y0 = std::clamp(static_cast<int>(fy), 0, h - 1), y1 = std::clamp(static_cast<int>(fy) + 1, 0, h - 1);
  auto at = [&](int px, int py) {
    const float *t = texels.data() + (static_cast<std::size_t>(py) * w + px) * 4;
    return float3(t[0], t[1], t[2]);
  };
  const float3 top = mix(at(x0, y0), at(x1, y0), tx);
  const float3 bottom = mix(at(x0, y1), at(x1, y1), tx);
  return mix(top, bottom, ty);
}

TraceView::TraceView(const HostTextures &hostTextures, const HostEnvironment *hostEnvironment)
    : textures(hostTextures), environment(hostEnvironment) {
  scene.host = reinterpret_cast<std::uint64_t>(this);
}

PtCpuScene TraceView::withInstances(const std::vector<TraceInstance> &instances) const {
  PtCpuScene copy = scene;
  copy.traceInstances = buffer(instances);
  return copy;
}

// What the generated code calls back for (pt_cpu.slang), with the C linkage it declares; only
// the generated C++ calls them, so returning vectors is fine.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4190)
#endif
namespace {
const TraceView &viewOf(std::uint64_t host) { return *reinterpret_cast<const TraceView *>(host); }
} // namespace

extern "C" Vector<float, 4> ptHostSampleTexture(std::uint64_t host, uint slot, Vector<float, 2> uv, uint samplerCode, float lodBase) {
  return viewOf(host).textures.sample(slot, uv, samplerCode, lodBase);
}

extern "C" float ptHostTextureFootprint(std::uint64_t host, uint slot, float lodBase) {
  return viewOf(host).textures.footprint(slot, lodBase);
}

extern "C" Vector<float, 3> ptHostSampleEnvironment(std::uint64_t host, Vector<float, 2> uv) {
  const TraceView &view = viewOf(host);
  return view.environment ? view.environment->sample(uv) : float3(0.0f);
}

extern "C" PtHit ptHostTrace(std::uint64_t host, Vector<float, 3> origin, Vector<float, 3> direction, float tMax, uint mask, uint seed,
                  Vector<float, 2> cone, uint anyHit) {
  const TraceView &view = viewOf(host);
  if (!view.trace) return ptTraceBvh(view, origin, direction, tMax, mask, seed, cone, anyHit);
  return view.trace(view, origin, direction, tMax, mask, seed, cone, anyHit);
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

} // namespace pt
