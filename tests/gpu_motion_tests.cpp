#include "core/Log.h"
#include "platform/Window.h"
#include "gpu/Uploader.h"
#include "render/PathReconstruction.h"
#include "render/RayTracing.h"
#include "render/GpuBvhBuilder.h"
#include "temporal_motion_fixture.h"
#include "temporal_motion_v7_fixture.h"
#include <fstream>
#include <memory>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

using namespace basalt;
namespace {
void require(bool condition, const char *why) { if (!condition) throw std::runtime_error(why); }
void barrier(VkCommandBuffer cmd) {
  VkMemoryBarrier2 b{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  b.srcStageMask = b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  b.srcAccessMask = b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
  VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO}; dep.memoryBarrierCount = 1; dep.pMemoryBarriers = &b;
  vkCmdPipelineBarrier2(cmd, &dep);
}
struct Sampler {
  VkDevice device; VkSampler handle = VK_NULL_HANDLE;
  ~Sampler() { if (handle) vkDestroySampler(device, handle, nullptr); }
};

class GpuFixture {
public:
  Context &context; Uploader &uploader;
  Scene scene;
  TraceScene trace;
  std::unique_ptr<SceneAccelerationStructure> acceleration;
  Buffer instances, materials, distribution, albedoTable, lights, emissives, uniforms, bvhNodes, bvhTriangles;
  Buffer restirDormant;  // the ReSTIR DI result bindings, never read with NEE
  std::vector<Image> textures;
  Image environment, accumulation, albedo, normal, output;
  Sampler environmentSampler;
  GltfSamplers tableSamplers;
  Program program;
  Pipeline pipeline;
  DescriptorPool pool;
  VkDescriptorSet set{}, sceneSet{}, structureSet{};
  GpuFixture(Context &c, Uploader &u, const pt_fixture::Builder &b, Buffer &guides, bool software,
             bool gpuBuilder)
      : context(c), uploader(u), environmentSampler{c.device}, tableSamplers(c),
        program(c, software ? (b.scene.emissiveTriangles.empty() ? "path_trace" : "path_trace_emissive")
                            : "path_trace_rt"),
        pipeline(c, program, software ? "motion.path.software" : "motion.path.rt"), pool(c, 8) {
    const auto &s = b.scene;
    scene.vertexCount = static_cast<std::uint32_t>(s.vertices.size() / pt::kVertexFloats);
    scene.indexCount = static_cast<std::uint32_t>(s.indices.size());
    scene.vertexBuffer = u.createBuffer(s.vertices.data(), s.vertices.size() * sizeof(float),
        geometryBufferUsage(c, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT), "motion.vertices");
    scene.indexBuffer = u.createBuffer(s.indices.data(), s.indices.size() * sizeof(pt::uint),
        geometryBufferUsage(c, VK_BUFFER_USAGE_INDEX_BUFFER_BIT), "motion.indices");
    for (std::uint32_t i = 0; i < s.instances.size(); ++i) {
      const auto &source = s.instances[i];
      Primitive p;
      p.firstIndex = source.firstIndex; p.indexCount = s.triangleCounts[i] * 3;
      p.vertexOffset = static_cast<std::int32_t>(source.vertexOffset); p.material = source.material;
      const auto a = source.objectToWorld0, d = source.objectToWorld1, e = source.objectToWorld2;
      p.transform.columns[0] = Vec4(a.x, d.x, e.x, 0);
      p.transform.columns[1] = Vec4(a.y, d.y, e.y, 0);
      p.transform.columns[2] = Vec4(a.z, d.z, e.z, 0);
      p.transform.columns[3] = Vec4(a.w, d.w, e.w, 1);
      scene.primitives.push_back(p); trace.primitives.push_back(i);
    }
    trace.instances = s.instances;
    if (!software) acceleration = std::make_unique<SceneAccelerationStructure>(c, u, scene, trace);
    if (gpuBuilder) {
      GpuBvhBuildResult built = buildGpuBvh(c, u, scene, trace);
      trace.instances = std::move(built.instances);
      bvhNodes = std::move(built.nodes);
      bvhTriangles = std::move(built.triangles);
    }
    auto upload = [&](const void *data, VkDeviceSize bytes, const char *name) {
      return u.createBuffer(data, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, name);
    };
    instances = upload(trace.instances.data(), trace.instances.size() * sizeof(pt::TraceInstance), "motion.instances");
    materials = upload(b.frame.materials.data(), b.frame.materials.size() * sizeof(pt::Material), "motion.materials");
    distribution = upload(s.distribution.data(), s.distribution.size() * sizeof(float), "motion.distribution");
    albedoTable = upload(s.specularAlbedo.data(), s.specularAlbedo.size() * sizeof(float), "motion.table");
    pt::Light zero{}; lights = upload(&zero, sizeof(zero), "motion.lights");
    pt::PtEmissiveTriangle noEmitter{};
    emissives = s.emissiveTriangles.empty()
        ? upload(&noEmitter, sizeof(noEmitter), "motion.emissives")
        : upload(s.emissiveTriangles.data(), s.emissiveTriangles.size() * sizeof(pt::PtEmissiveTriangle),
                 "motion.emissives");
    if (software && !gpuBuilder) {
      bvhNodes = upload(s.bvh.nodes.data(), s.bvh.nodes.size() * sizeof(pt::float4), "motion.bvh.nodes");
      bvhTriangles = upload(s.bvh.triangles.data(), s.bvh.triangles.size() * sizeof(pt::float4),
                            "motion.bvh.triangles");
    }
    uniforms = Buffer(c, sizeof(pt::PathUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT);
    for (const auto &t : s.textures.textures)
      textures.push_back(u.createTexture(t.texels.data(), t.texels.size() * 4, t.width, t.height,
          t.srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM, 1, "motion.texture"));
    std::vector<VkImageView> views;
    for (auto slot : s.textures.slots) views.push_back(textures.at(slot).view);
    environment = u.createTexture(s.environment.texels.data(), s.environment.texels.size() * sizeof(float),
        s.environment.width, s.environment.height, VK_FORMAT_R32G32B32A32_SFLOAT, 1, "motion.environment");
    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.magFilter = sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    check(vkCreateSampler(c.device, &sampler, nullptr, &environmentSampler.handle), "motion environment sampler");
    ImageDescription image;
    image.width = temporal_motion::width; image.height = temporal_motion::height;
    image.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    image.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    accumulation = Image(c, image); albedo = Image(c, image); normal = Image(c, image); output = Image(c, image);
    u.runImmediate([&](VkCommandBuffer cmd) {
      for (Image *img : {&accumulation, &albedo, &normal, &output})
        transitionImage(cmd, *img, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    });
    restirDormant = Buffer(c, 48, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, 0, "motion.restir");
    sceneSet = program.allocate(pool, "scene");
    DescriptorWriter sceneWriter(c, program, sceneSet, "scene");
    sceneWriter.buffer("geometry.traceInstances", instances).buffer("geometry.materials", materials)
        .buffer("geometry.indices", scene.indexBuffer).buffer("geometry.vertices", scene.vertexBuffer)
        .textureArray("maps.table", views, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        .texture("maps.environmentMap", environment).sampler("maps.environmentSampler", environmentSampler.handle)
        .buffer("lights", lights).buffer("environmentDistribution", distribution).buffer("specularAlbedo", albedoTable)
        .buffer("emissiveTriangles", emissives);
    for (int i = 0; i < 6; ++i)
      sceneWriter.sampler(std::string("maps.") + GltfSamplers::names[i], tableSamplers.handles[i]);
    sceneWriter.apply();
    structureSet = program.allocate(pool, "structure");
    DescriptorWriter structure(c, program, structureSet, "structure");
    if (software)
      structure.buffer("bvhNodes", bvhNodes).buffer("bvhTriangles", bvhTriangles);
    else
      structure.accelerationStructure("accelerationStructure", acceleration->topLevel);
    structure.apply();
    set = program.allocate(pool);
    DescriptorWriter(c, program, set)
        .buffer("uniforms", uniforms).buffer("reconstructionSamples", guides)
        .buffer("restirResults", restirDormant).buffer("restirGuides", restirDormant)
        .storageTexture("accumulation", accumulation.view)
        .storageTexture("albedoAccumulation", albedo.view).storageTexture("normalAccumulation", normal.view)
        .storageTexture("output", output.view).apply();
  }
  void dispatch(VkCommandBuffer cmd, const pt::PathUniforms &u) {
    uniforms.write(&u, sizeof(u)); barrier(cmd);
    transitionImage(cmd, output, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
    program.bind(cmd, set);
    program.bind(cmd, sceneSet, "scene");
    program.bind(cmd, structureSet, "structure");
    vkCmdDispatch(cmd, (temporal_motion::width + 7) / 8, (temporal_motion::height + 7) / 8, 1);
    barrier(cmd);
  }
  std::vector<pt::float4> read(Image &image) {
    const auto bytes = uploader.readImage(image, 16);
    std::vector<pt::float4> result(bytes.size() / 16);
    std::memcpy(result.data(), bytes.data(), bytes.size()); return result;
  }
};

int runEmissive(bool software, bool gpuBuilder) {
  openLogFile(gpuBuilder ? "gpu-bvh-build-emissive-validation.log" :
              software ? "gpu-bvh-emissive-validation.log" : "gpu-emissive-validation.log");
  Window window("GPU emissive path oracle", 64, 64, false);
  Context context(window, true);
  if (!software && !context.rayTracingSupported) { std::printf("SKIP: ray queries unavailable\n"); return 77; }
  require(context.validationEnabled, "GPU emissive oracle requires Vulkan validation");
  {
    Uploader uploader(context);
    pt_fixture::Builder b;
    b.floor(8.0f, 0.0f, b.material(pt::float3(0.8f), 0.0f, 1.0f));
    b.ceiling(1.5f, 3.0f, b.emissiveMaterial(pt::float3(8.0f, 6.0f, 4.0f)));
    b.environment([](pt::float3) { return pt::float3(0.0f); }, 8, 4);
    b.camera(pt::float3(0.0f, 1.2f, 5.0f), pt::float3(0.0f), 0.65f,
             temporal_motion::width, temporal_motion::height);
    b.finish(2, 0, 0x4281u);
    constexpr pt::uint samples = 1024;
    const auto cpu = b.render(samples);
    PathReconstruction guides(context, temporal_motion::width, temporal_motion::height);
    GpuFixture gpu(context, uploader, b, guides.samples(), software, gpuBuilder);
    auto uniforms = b.frame.uniforms;
    uniforms.image.z = 0.0f;
    uniforms.image.w = static_cast<float>(samples);
    uploader.runImmediate([&](VkCommandBuffer cmd) { gpu.dispatch(cmd, uniforms); });
    const auto sum = gpu.read(gpu.accumulation);
    double error = 0.0, energy = 0.0;
    pt::uint litPixels = 0;
    for (std::size_t i = 0; i < sum.size(); ++i) {
      require(sum[i].w == float(samples), "GPU emissive batch changed exact SPP");
      const pt::float3 actual = pt::xyz(sum[i]) / sum[i].w;
      const pt::float3 expected(cpu[i * 3], cpu[i * 3 + 1], cpu[i * 3 + 2]);
      require(pt::ptFiniteColor(actual), "nonfinite GPU emissive output");
      error += pt::ptLuminance(pt::abs(actual - expected));
      energy += pt::ptLuminance(expected);
      if (pt::ptLuminance(expected) > 0.01f) ++litPixels;
    }
    const double relative = error / std::max(energy, 1e-9);
    require(litPixels > sum.size() / 8, "emissive fixture did not illuminate enough receiver pixels");
    if (relative > 0.02)
      throw std::runtime_error("GPU emissive result differs from CPU by " + std::to_string(relative));
    std::printf("PASS GPU %s emissive NEE/MIS: %u SPP, relative luminance error %.6f\n",
                gpuBuilder ? "GPU-built software-BVH" : software ? "software-BVH" : "ray-query",
                samples, relative);

    // Temporal guides with emissive triangles: collecting them leaves the raw sample bitwise
    // unchanged, and the guide's diffuse + specular split is that sample's radiance.
    uniforms.image.w = 1.0f;
    uniforms.image.z = 0.0f;
    uploader.runImmediate([&](VkCommandBuffer cmd) { gpu.dispatch(cmd, uniforms); });
    const auto withoutGuides = gpu.read(gpu.accumulation);
    uniforms.image.z = 1.0f;
    uploader.runImmediate([&](VkCommandBuffer cmd) { gpu.dispatch(cmd, uniforms); });
    const auto withGuides = gpu.read(gpu.accumulation);
    require(std::memcmp(withoutGuides.data(), withGuides.data(), withGuides.size() * sizeof(pt::float4)) == 0,
            "GPU emissive raw sample changed when guide collection toggled");
    const auto bytes = uploader.readBuffer(guides.samples(), withGuides.size() * sizeof(pt::PtReconstructionSample));
    std::vector<pt::PtReconstructionSample> collected(withGuides.size());
    std::memcpy(collected.data(), bytes.data(), collected.size() * sizeof(pt::PtReconstructionSample));
    pt::uint covered = 0;
    for (std::size_t i = 0; i < collected.size(); ++i) {
      if (collected[i].identity.z == 0u) continue;
      ++covered;
      const pt::float3 split = pt::xyz(collected[i].diffuse) + pt::xyz(collected[i].specular);
      const pt::float3 raw = pt::xyz(withGuides[i]);
      require(pt::ptFiniteColor(split), "nonfinite GPU emissive guide radiance");
      if (pt::ptLuminance(pt::abs(split - raw)) > 1e-4f * std::max(1.0f, pt::ptLuminance(raw)))
        throw std::runtime_error("GPU emissive guide split differs from its raw sample at pixel " + std::to_string(i));
    }
    require(covered > collected.size() / 2, "GPU emissive guides covered too few pixels");
    std::printf("PASS GPU %s emissive temporal guides: raw invariant, split matches the sample on %u pixels\n",
                gpuBuilder ? "GPU-built software-BVH" : software ? "software-BVH" : "ray-query", covered);
    context.waitIdle();
  }
  require(!context.sawValidationError, "GPU emissive validation errors");
  return 0;
}

// v7: the V7 motion corpus (glass, clearcoat, depth of field; tests/data/temporal_v7), whose
// scene changes per sequence, so the GPU fixture is rebuilt for each.
int run(bool software, bool gpuBuilder, bool v7) {
  const std::string prefix = v7 ? "gpu-v7-" : "gpu-";
  openLogFile(prefix + (gpuBuilder ? "bvh-build-motion-validation.log" :
              software ? "bvh-motion-validation.log" : "motion-validation.log"));
  Window window("GPU traced motion oracle", 64, 64, false);
  Context context(window, true);
  if (!software && !context.rayTracingSupported) { std::printf("SKIP: ray queries unavailable\n"); return 77; }
  require(context.validationEnabled, "GPU motion oracle requires Vulkan validation");
  int failures = 0;
  {
    Uploader uploader(context);
    pt_fixture::Builder b;
    if (v7) temporal_motion_v7::scene(b, 0); else temporal_motion::scene(b);
    const std::vector<const char *> sequences = v7
        ? std::vector<const char *>(temporal_motion_v7::sequences.begin(), temporal_motion_v7::sequences.end())
        : std::vector<const char *>(temporal_motion::sequences.begin(), temporal_motion::sequences.end());
    constexpr pt::uint w = temporal_motion::width, h = temporal_motion::height, pixels = w * h;
    PathReconstruction filter(context, w, h);
    auto gpu = std::make_unique<GpuFixture>(context, uploader, b, filter.samples(), software, gpuBuilder);
    std::ofstream report(prefix + (gpuBuilder ? "bvh-build-motion-results.csv" :
                         software ? "bvh-motion-results.csv" : "motion-results.csv"));
    report << "sequence,seed,raw_mae,filtered_mae,raw_flicker,filtered_flicker\n";
    std::uint64_t key = 0;
    for (pt::uint seq = 0; seq < sequences.size(); ++seq) {
      if (v7 && seq > 0) {
        context.waitIdle();
        gpu.reset();
        b = pt_fixture::Builder{};
        temporal_motion_v7::scene(b, seq);
        gpu = std::make_unique<GpuFixture>(context, uploader, b, filter.samples(), software, gpuBuilder);
      }
      std::ifstream file(std::string(BASALT_TEST_DATA_DIR) + (v7 ? "/temporal_v7/" : "/temporal/") + sequences[seq] +
          ".rgb32f", std::ios::binary);
      std::vector<pt::float3> refs(pixels * temporal_motion::frames);
      file.read(reinterpret_cast<char *>(refs.data()), refs.size() * sizeof(pt::float3));
      require(bool(file), "missing frozen motion reference");
      double sr = 0, sf = 0, tr = 0, tf = 0;
      for (pt::uint seed : temporal_motion::seeds) {
        ++key;
        std::vector<pt::float3> previousRaw(pixels), previousFiltered(pixels);
        pt::PathUniforms previousCamera{};
        pt::uint stationarySamples = 0;
        double er = 0, ef = 0, fr = 0, ff = 0, energy = 0;
        for (pt::uint frame = 0; frame < temporal_motion::frames; ++frame) {
          if (v7) temporal_motion_v7::camera(b, seq, frame); else temporal_motion::camera(b, seq, frame);
          b.frame.uniforms.counts.z = seed;
          const bool same = frame != 0 && std::memcmp(&previousCamera.cameraPosition, &b.frame.uniforms.cameraPosition,
              4 * sizeof(pt::float4)) == 0;
          if (!same) stationarySamples = 0;
          auto u = b.frame.uniforms; u.counts.w = stationarySamples; u.image.w = 1;
          uploader.runImmediate([&](VkCommandBuffer cmd) {
            gpu->dispatch(cmd, u);
            filter.record(cmd, frame % kFramesInFlight, gpu->output, u, key, stationarySamples + 1, true, 1);
          });
          ++stationarySamples;
          const auto raw = gpu->read(gpu->accumulation), output = gpu->read(gpu->output);
          for (pt::uint i = 0; i < pixels; ++i) {
            require(raw[i].w == float(stationarySamples), "GPU changed exact raw SPP");
            const pt::float3 r = pt::xyz(raw[i]) / raw[i].w, f = pt::xyz(output[i]);
            require(pt::ptFiniteColor(r) && pt::ptFiniteColor(f), "nonfinite GPU motion output");
            const auto re = r - refs[frame * pixels + i], fe = f - refs[frame * pixels + i];
            auto sum = [](pt::float3 v) { return double(v.x) + v.y + v.z; };
            er += sum(pt::abs(re)); ef += sum(pt::abs(fe)); energy += 3 * pt::ptTemporalLuminance(refs[frame * pixels + i]);
            if (frame) { fr += sum(pt::abs(re - previousRaw[i])); ff += sum(pt::abs(fe - previousFiltered[i])); }
            previousRaw[i] = re; previousFiltered[i] = fe;
          }
          previousCamera = b.frame.uniforms;
        }
        const double scale = 1.0 / std::max(energy, 0.001 * pixels * temporal_motion::frames * 3);
        report << sequences[seq] << ',' << seed << ',' << er * scale << ',' << ef * scale << ','
               << fr * scale << ',' << ff * scale << '\n';
        sr += er; sf += ef; tr += fr; tf += ff;
      }
      const bool pass = sf <= sr * 0.9 && tf <= tr * 0.9;
      std::printf("%s GPU %s %smotion %s spatial %.5f flicker %.5f (limit 0.90)\n", pass ? "PASS" : "FAIL",
          gpuBuilder ? "GPU-built software-BVH" : software ? "software-BVH" : "ray-query", v7 ? "V7 " : "",
          sequences[seq], sf / sr, tf / tr);
      if (!pass) ++failures;
    }
    // Identical seeded raw results with guides disabled/enabled, outside the display filter.
    auto u = b.frame.uniforms; u.counts.w = 0; u.image.z = 0; u.image.w = 7;
    uploader.runImmediate([&](VkCommandBuffer cmd) { gpu->dispatch(cmd, u); });
    const auto off = gpu->read(gpu->accumulation);
    u.image.z = 1;
    uploader.runImmediate([&](VkCommandBuffer cmd) { gpu->dispatch(cmd, u); });
    const auto on = gpu->read(gpu->accumulation);
    require(std::memcmp(off.data(), on.data(), off.size() * sizeof(pt::float4)) == 0,
        "GPU raw accumulation changed when guide collection toggled");
    for (const auto &p : on) require(p.w == 7, "multi-SPP batch changed exact target");
    std::printf("PASS GPU raw bitwise invariance and exact seven-SPP batch\n");
    context.waitIdle();
  }
  require(!context.sawValidationError, "GPU motion validation errors (gpu-motion-validation.log)");
  return failures == 0 ? 0 : 1;
}
}
int main(int argc, char **argv) {
  try {
    // A leading --v7 selects the V7 motion corpus.
    const bool v7 = argc > 1 && std::string(argv[1]) == "--v7";
    if (v7) { --argc; ++argv; }
    const bool emissive = argc > 1 && std::string(argv[1]).find("emissive") != std::string::npos;
    const bool gpuBuilder = argc > 1 && std::string(argv[1]) == "--gpu-builder";
    if (emissive) {
      const bool emissiveGpuBuilder = std::string(argv[1]) == "--emissive-gpu-builder";
      const bool emissiveSoftware = emissiveGpuBuilder || std::string(argv[1]) == "--emissive-software";
      return runEmissive(emissiveSoftware, emissiveGpuBuilder);
    }
    return run(gpuBuilder || (argc > 1 && std::string(argv[1]) == "--software"), gpuBuilder, v7);
  }
  catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
