#include "pt/CpuTracer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace pt {

std::uint16_t floatToHalf(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const std::int32_t exponent = static_cast<std::int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
  std::uint32_t mantissa = bits & 0x7FFFFFu;
  if (((bits >> 23) & 0xFFu) == 0xFFu) return static_cast<std::uint16_t>(sign | 0x7C00u | (mantissa ? 0x200u : 0u));
  if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);  // overflow to infinity
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<std::uint16_t>(sign);
    mantissa |= 0x800000u;
    const std::uint32_t shift = static_cast<std::uint32_t>(14 - exponent);
    std::uint32_t half = mantissa >> shift;
    if ((mantissa >> (shift - 1)) & 1u) ++half;  // round half up
    return static_cast<std::uint16_t>(sign | half);
  }
  std::uint32_t half = sign | (static_cast<std::uint32_t>(exponent) << 10) | (mantissa >> 13);
  if (mantissa & 0x1000u) ++half;  // round half up; a carry into the exponent is correct
  return static_cast<std::uint16_t>(half);
}

// Worker pool. Every worker joins every parallelFor, so no worker can pick up a new task
// through the previous one's function.

WorkerPool::WorkerPool(unsigned threads) {
  for (unsigned i = 1; i < std::max(1u, threads); ++i) workers.emplace_back([this] { work(); });
}

WorkerPool::~WorkerPool() {
  {
    std::lock_guard<std::mutex> lock(mutex);
    quit = true;
  }
  startCondition.notify_all();
  for (std::thread &worker : workers) worker.join();
}

void WorkerPool::work() {
  std::uint64_t seen = 0;
  for (;;) {
    const std::function<void(uint)> *fn = nullptr;
    uint count = 0;
    {
      std::unique_lock<std::mutex> lock(mutex);
      startCondition.wait(lock, [&] { return quit || epoch != seen; });
      if (quit) return;
      seen = epoch;
      fn = task;
      count = taskCount;
    }
    for (uint i; (i = next.fetch_add(1)) < count;) (*fn)(i);
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (++finishedWorkers == workers.size()) finishedCondition.notify_all();
    }
  }
}

void WorkerPool::parallelFor(uint count, const std::function<void(uint)> &fn) {
  {
    std::lock_guard<std::mutex> lock(mutex);
    task = &fn;
    taskCount = count;
    next = 0;
    finishedWorkers = 0;
    ++epoch;
  }
  startCondition.notify_all();
  for (uint i; (i = next.fetch_add(1)) < count;) fn(i);
  std::unique_lock<std::mutex> lock(mutex);
  finishedCondition.wait(lock, [&] { return finishedWorkers == workers.size(); });
}

FrameView::FrameView(const CpuScene &cpu, const CpuFrame &frame) : TraceView(cpu.textures, &cpu.environment) {
  scene.traceInstances = buffer(cpu.instances);
  scene.materials = buffer(frame.materials);
  scene.indices = buffer(cpu.indices);
  scene.vertices = buffer(cpu.vertices);
  scene.lights = buffer(frame.lights);
  scene.environmentDistribution = buffer(cpu.distribution);
  scene.specularAlbedo = buffer(cpu.specularAlbedo);
  scene.emissiveTriangles = buffer(cpu.emissiveTriangles);
  scene.bvhNodes = buffer(cpu.bvh.nodes);
  scene.bvhTriangles = buffer(cpu.bvh.triangles);
  scene.uniforms = frame.uniforms;
  // An intersector the scene was built without falls back to the binary BVH.
  scene.intersector = 0u;
  if (frame.intersector == 1 && cpu.embree) {
    scene.intersector = 1u;
    tracer = cpu.embree.get();
    trace = [](const TraceView &view, float3 origin, float3 direction, float tMax, uint mask, uint seed, float2 cone,
               uint anyHit) {
      return static_cast<const EmbreeScene *>(view.tracer)->trace(view, origin, direction, tMax, mask, seed, cone, anyHit);
    };
  } else if (frame.intersector == 2 && cpu.wideAvx2) {
    scene.intersector = 2u;
    tracer = cpu.wideAvx2.get();
    trace = [](const TraceView &view, float3 origin, float3 direction, float tMax, uint mask, uint seed, float2 cone,
               uint anyHit) {
      return traceWideBvhAvx2(*static_cast<const WideBvh *>(view.tracer), view, origin, direction, tMax, mask, seed, cone,
                              anyHit);
    };
  }
}

namespace {

bool finite(float3 v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); }

} // namespace

PathSample CpuTracer::tracePixel(const TraceView &view, uint pixelX, uint pixelY, uint sampleIndex, float3 restirDirect,
                                 float3 restirDiffuse) {
  const PtPath path = gen::ptCpuTracePath(view.get(), pixelX, pixelY, sampleIndex, restirDirect, restirDiffuse);
  return {path.radiance, path.albedo, path.normal, path.guide};
}

PathSample CpuTracer::tracePixel(const CpuScene &scene, const CpuFrame &frame, uint pixelX, uint pixelY,
                                 uint sampleIndex, float3 restirDirect, float3 restirDiffuse) {
  const FrameView view(scene, frame);
  return tracePixel(view, pixelX, pixelY, sampleIndex, restirDirect, restirDiffuse);
}

// ReSTIR DI: the GPU's reservoir passes (pt_restir.slang), per pixel on the workers.

void CpuTracer::restirFrame(const CpuScene &scene, const CpuFrame &frame, uint sampleIndex, RestirState &state,
                            WorkerPool &pool) {
  const PathUniforms &uniforms = frame.uniforms;
  const uint width = frame.width, height = frame.height, pixels = width * height;
  const uint candidates = std::min(uniforms.estimator.y, kPtRestirMaxCandidates);
  if (state.previous.image.x != width || state.previous.image.y != height ||
      state.previousSurfaces.size() != pixels || state.previousReservoirs.size() != pixels)
    state.previous.image.z = 0u;
  // estimator.z bit 0: temporal reuse.
  if ((uniforms.estimator.z & 1u) == 0u) state.previous.image.z = 0u;
  state.surfaces.assign(pixels, PtRestirSurface{});
  state.reservoirs.assign(pixels, ptEmptyReservoir());
  std::vector<PtReservoir> finals(pixels, ptEmptyReservoir());
  state.direct.assign(pixels, float3(0.0f));
  state.directDiffuse.assign(pixels, float3(0.0f));

  const FrameView view(scene, frame);
  // Primary hits, initial resampling and temporal reuse (reads only the previous frame).
  pool.parallelFor(height, [&](uint y) {
    for (uint x = 0; x < width; ++x) {
      const uint pixel = y * width + x;
      gen::ptCpuRestirPrimary(view.get(), x, y, sampleIndex, candidates, &state.previous, buffer(state.previousSurfaces),
                         buffer(state.previousReservoirs), &state.surfaces[pixel], &state.reservoirs[pixel]);
    }
  });
  // Spatial reuse, then the final reservoir's light with its shadow ray.
  pool.parallelFor(height, [&](uint y) {
    for (uint x = 0; x < width; ++x) {
      const uint pixel = y * width + x;
      gen::ptCpuRestirSpatialShade(view.get(), x, y, sampleIndex, buffer(state.surfaces), buffer(state.reservoirs), &finals[pixel],
                              &state.direct[pixel], &state.directDiffuse[pixel]);
    }
  });

  std::swap(state.previousSurfaces, state.surfaces);
  state.previousReservoirs = std::move(finals);
  state.previous.position = uniforms.cameraPosition;
  state.previous.forward = uniforms.cameraForward;
  state.previous.right = uniforms.cameraRight;
  state.previous.up = uniforms.cameraUp;
  state.previous.image = uint4(width, height, 1u, uniforms.estimator.z);
}

namespace {

constexpr uint kTile = 32;
constexpr uint kSumFloats = 9;  // radiance, albedo, normal

// Adds one sample per pixel of a tile to `sums`; a non-finite path adds nothing. restir: the
// frame's ReSTIR DI state, when the estimator is ReSTIR.
void traceTile(const CpuScene &scene, const CpuFrame &frame, uint tile, uint sampleIndex, float *sums,
               PtReconstructionSample *guides = nullptr, const RestirState *restir = nullptr) {
  const FrameView view(scene, frame);
  const uint tilesX = (frame.width + kTile - 1) / kTile;
  const uint x0 = (tile % tilesX) * kTile, y0 = (tile / tilesX) * kTile;
  for (uint y = y0; y < std::min(y0 + kTile, frame.height); ++y)
    for (uint x = x0; x < std::min(x0 + kTile, frame.width); ++x) {
      const std::size_t pixel = static_cast<std::size_t>(y) * frame.width + x;
      const PathSample sample =
          restir ? CpuTracer::tracePixel(view, x, y, sampleIndex, restir->direct[pixel], restir->directDiffuse[pixel])
                 : CpuTracer::tracePixel(view, x, y, sampleIndex);
      if (guides) guides[static_cast<std::size_t>(y) * frame.width + x] = sample.guide;
      float *p = sums + (static_cast<std::size_t>(y) * frame.width + x) * kSumFloats;
      if (finite(sample.radiance)) {
        p[0] += sample.radiance.x;
        p[1] += sample.radiance.y;
        p[2] += sample.radiance.z;
      }
      p[3] += sample.albedo.x;
      p[4] += sample.albedo.y;
      p[5] += sample.albedo.z;
      p[6] += sample.normal.x;
      p[7] += sample.normal.y;
      p[8] += sample.normal.z;
    }
}

uint tileCount(const CpuFrame &frame) {
  return ((frame.width + kTile - 1) / kTile) * ((frame.height + kTile - 1) / kTile);
}

// One of the three quantities of the sums, averaged, as RGB.
std::vector<float> channel(const std::vector<float> &sums, uint offset, uint samples) {
  const std::size_t pixels = sums.size() / kSumFloats;
  std::vector<float> rgb(pixels * 3);
  const float scale = 1.0f / static_cast<float>(std::max(samples, 1u));
  for (std::size_t i = 0; i < pixels; ++i)
    for (uint c = 0; c < 3; ++c) rgb[i * 3 + c] = sums[i * kSumFloats + offset + c] * scale;
  return rgb;
}

bool powerOfTwo(uint n) { return n != 0 && (n & (n - 1)) == 0; }

} // namespace

CpuTracer::Result CpuTracer::render(const CpuScene &scene, const CpuFrame &frame, uint samples, WorkerPool &pool) {
  std::vector<float> sums(static_cast<std::size_t>(frame.width) * frame.height * kSumFloats, 0.0f);
  const bool restir = frame.uniforms.estimator.x == 1u;
  RestirState state;
  for (uint s = 0; s < samples; ++s) {
    if (restir) restirFrame(scene, frame, s, state, pool);
    pool.parallelFor(tileCount(frame),
                     [&](uint tile) { traceTile(scene, frame, tile, s, sums.data(), nullptr, restir ? &state : nullptr); });
  }
  return {channel(sums, 0, samples), channel(sums, 3, samples), channel(sums, 6, samples)};
}

CpuTracer::CpuTracer(unsigned threads)
    : pool(threads > 0 ? threads : std::max(1u, std::thread::hardware_concurrency() - 1)) {
  coordinator = std::thread([this] { coordinate(); });
}

CpuTracer::~CpuTracer() {
  {
    std::lock_guard<std::mutex> lock(mutex);
    quit = true;
  }
  ++generation;
  wake.notify_all();
  coordinator.join();
}

void CpuTracer::setScene(std::shared_ptr<const CpuScene> next) {
  {
    std::lock_guard<std::mutex> lock(mutex);
    scene = std::move(next);
    frame.reset();  // a new scene must not run with the previous scene's materials
    ++generation;
  }
  wake.notify_all();
}

void CpuTracer::start(CpuFrame next) {
  {
    std::lock_guard<std::mutex> lock(mutex);
    frame = std::make_shared<const CpuFrame>(std::move(next));
    ++generation;
  }
  wake.notify_all();
}

bool CpuTracer::latest(Image &image, std::uint64_t version, bool withFloats) const {
  std::lock_guard<std::mutex> lock(mutex);
  if (published.version == 0 || published.version == version || published.generation != generation.load())
    return false;
  image.half = published.half;
  if (withFloats) {
    image.mean = published.mean;
    image.denoised = published.denoised;
  }
  image.width = published.width;
  image.height = published.height;
  image.samples = published.samples;
  image.denoisedSamples = published.denoisedSamples;
  image.previewScale = published.previewScale;
  image.version = published.version;
  image.generation = published.generation;
  image.uniforms = published.uniforms;
  image.guides = published.guides;
  return true;
}

std::string CpuTracer::denoiserStatus() const {
  std::lock_guard<std::mutex> lock(mutex);
  return denoiserState;
}

void CpuTracer::publish(const std::vector<float> &sums, uint samples, const CpuFrame &f, std::uint64_t forGeneration,
                        const std::vector<PtReconstructionSample> &guides, const PathUniforms *guideCamera) {
  Image image;
  image.width = f.width;
  image.height = f.height;
  image.samples = samples;
  image.generation = forGeneration;
  image.uniforms = guideCamera ? *guideCamera : f.uniforms;
  image.previewScale = samples == 0 ? 4u : 1u;
  image.guides = guides;
  const std::size_t pixels = static_cast<std::size_t>(f.width) * f.height;
  const std::vector<float> color = channel(sums, 0, samples);
  image.mean.resize(pixels * 4);
  for (std::size_t i = 0; i < pixels; ++i) {
    image.mean[i * 4 + 0] = color[i * 3 + 0];
    image.mean[i * 4 + 1] = color[i * 3 + 1];
    image.mean[i * 4 + 2] = color[i * 3 + 2];
    image.mean[i * 4 + 3] = 1.0f;
  }

  // Denoised at every power of two and at the target, so the cost falls as the image
  // converges; in between, the last denoised image stays on screen.
  if (f.denoise && samples > 0 && (powerOfTwo(samples) || samples == f.targetSamples)) {
    if (!denoiser && Denoiser::available()) {
      denoiser = std::make_unique<Denoiser>();
      std::lock_guard<std::mutex> lock(mutex);
      denoiserState = denoiser->error().empty() ? denoiser->device() : denoiser->error();
    }
    if (denoiser) {
      std::vector<float> result =
          denoiser->denoise(color, channel(sums, 3, samples), channel(sums, 6, samples), f.width, f.height);
      if (!result.empty()) {
        lastDenoised = std::move(result);
        lastDenoisedSamples = samples;
      }
    }
  }
  const bool showDenoised = f.denoise && lastDenoised.size() == pixels * 3;
  if (showDenoised) {
    image.denoised.resize(pixels * 4);
    for (std::size_t i = 0; i < pixels; ++i) {
      image.denoised[i * 4 + 0] = lastDenoised[i * 3 + 0];
      image.denoised[i * 4 + 1] = lastDenoised[i * 3 + 1];
      image.denoised[i * 4 + 2] = lastDenoised[i * 3 + 2];
      image.denoised[i * 4 + 3] = 1.0f;
    }
    image.denoisedSamples = lastDenoisedSamples;
  }
  const std::vector<float> &shown = showDenoised ? image.denoised : image.mean;
  image.half.resize(pixels * 4);
  for (std::size_t i = 0; i < pixels * 4; ++i) image.half[i] = floatToHalf(shown[i]);

  std::lock_guard<std::mutex> lock(mutex);
  if (forGeneration != generation.load()) return;  // superseded while it was being made
  image.version = published.version + 1;
  published = std::move(image);
}

void CpuTracer::coordinate() {
  std::uint64_t served = ~0ull;
  for (;;) {
    std::shared_ptr<const CpuScene> currentScene;
    std::shared_ptr<const CpuFrame> currentFrame;
    std::uint64_t current = 0;
    {
      std::unique_lock<std::mutex> lock(mutex);
      wake.wait(lock, [&] { return quit || (scene && frame && generation.load() != served); });
      if (quit) return;
      currentScene = scene;
      currentFrame = frame;
      current = generation.load();
      served = current;
    }
    const CpuScene &s = *currentScene;
    const CpuFrame &f = *currentFrame;
    if (f.width == 0 || f.height == 0) continue;
    const std::size_t pixels = static_cast<std::size_t>(f.width) * f.height;
    auto stale = [&] { return generation.load() != current; };
    lastDenoised.clear();
    lastDenoisedSamples = 0;

    // A preview first: one path per 4 x 4 block, spread over the block, so a moving view
    // answers at once.
    {
      std::vector<float> preview(pixels * kSumFloats, 0.0f);
      const uint blocksX = (f.width + 3) / 4, blocksY = (f.height + 3) / 4;
      CpuFrame previewFrame = f;
      previewFrame.width = blocksX; previewFrame.height = blocksY;
      previewFrame.uniforms.estimator.x = 0u;  // the reduced preview samples its lights directly
      previewFrame.uniforms.image.x = static_cast<float>(blocksX);
      previewFrame.uniforms.image.y = static_cast<float>(blocksY);
      std::vector<PtReconstructionSample> previewGuides;
      if (f.uniforms.image.z > 0.5f) previewGuides.resize(static_cast<std::size_t>(blocksX) * blocksY);
      pool.parallelFor(blocksY, [&](uint by) {
        if (stale()) return;
        const FrameView view(s, previewFrame);
        for (uint bx = 0; bx < blocksX; ++bx) {
          const auto path = tracePixel(view, bx, by, 0);
          if (!previewGuides.empty()) previewGuides[static_cast<std::size_t>(by) * blocksX + bx] = path.guide;
          float3 c = path.radiance;
          if (!finite(c)) c = float3(0.0f);
          for (uint py = (by * f.height + blocksY - 1) / blocksY;
               py < ((by + 1) * f.height + blocksY - 1) / blocksY; ++py)
            for (uint px = (bx * f.width + blocksX - 1) / blocksX;
                 px < ((bx + 1) * f.width + blocksX - 1) / blocksX; ++px) {
              float *p = preview.data() + (static_cast<std::size_t>(py) * f.width + px) * kSumFloats;
              p[0] = c.x;
              p[1] = c.y;
              p[2] = c.z;
            }
        }
      });
      if (stale()) continue;
      publish(preview, 0, f, current, previewGuides, &previewFrame.uniforms);
    }

    std::vector<float> sums(pixels * kSumFloats, 0.0f);
    std::vector<PtReconstructionSample> guides;
    if (f.uniforms.image.z > 0.5f) guides.resize(pixels);
    const auto started = std::chrono::steady_clock::now();
    uint samples = 0;
    // ReSTIR history lives within one accumulation: a restart (a new view) starts afresh.
    const bool restir = f.uniforms.estimator.x == 1u;
    RestirState restirState;
    while (!stale() && (f.targetSamples == 0 || samples < f.targetSamples)) {
      if (restir) restirFrame(s, f, samples, restirState, pool);
      if (stale()) break;
      pool.parallelFor(tileCount(f), [&](uint tile) {
        if (!stale())
          traceTile(s, f, tile, samples, sums.data(), guides.empty() ? nullptr : guides.data(),
                    restir ? &restirState : nullptr);
      });
      if (stale()) break;
      ++samples;
      publish(sums, samples, f, current, guides);
      const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      rate.store(seconds > 0.0 ? static_cast<double>(samples) * static_cast<double>(pixels) / seconds : 0.0);
    }
  }
}

} // namespace pt
