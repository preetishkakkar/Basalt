#include "render/GpuBvhBuilder.h"

#include "core/Log.h"
#include "gpu/Descriptors.h"
#include "gpu/Pipeline.h"
#include "gpu/Shader.h"
#include "gpu/Uploader.h"
#include "pt/Shared.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>

namespace basalt {
namespace {

static_assert(sizeof(pt::BvhSegment) == 32);
static_assert(sizeof(pt::BvhLbvhControl) == 80);
static_assert(sizeof(pt::BvhSortControl) == 32);

constexpr std::uint32_t kFitIterations = pt::kBvhStack; // tree levels the fit publishes
constexpr std::uint32_t kSortTile = 1024;          // keys per sort workgroup (bvh_sort.slang)
constexpr std::uint32_t kSortMaximumBlocks = 1024; // one workgroup scans the block sums
constexpr VkDeviceSize kSlot = 256;                // the largest offset alignment Vulkan allows
constexpr std::uint32_t kBatchedLevel = 256;       // the batched build's largest level (bvh_batched.slang)
// The binned SAH's large tasks (bvh_sah.slang): records per chunk, the size from which a task is
// chunked, the levels that may hold them, and a task's slot in words.
constexpr std::uint32_t kSahChunk = 256, kSahLarge = 4 * kSahChunk, kSahLargeLevels = 16, kSahSlotWords = 356;

const char *const kEntries[] = {"bvh_lbvh_extents",  "bvh_lbvh_instances", "bvh_lbvh_morton",
                                "bvh_sort_histogram", "bvh_sort_reduce",    "bvh_sort_blocks",
                                "bvh_sort_apply",     "bvh_sort_scatter",   "bvh_lbvh_topology",
                                "bvh_lbvh_seed",      "bvh_lbvh_fit",       "bvh_lbvh_number",
                                "bvh_lbvh_emit",      "bvh_lbvh_finish",    "bvh_ploc_init",
                                "bvh_ploc_single",    "bvh_ploc_neighbours", "bvh_ploc_scan_sums",
                                "bvh_ploc_compact",   "bvh_ploc_finish",    "bvh_apetrei_single",
                                "bvh_apetrei_climb",  "bvh_apetrei_levels", "bvh_apetrei_scatter",
                                "bvh_batched_build",  "bvh_ploc_fused",     "bvh_hploc",
                                "bvh_sah_level",      "bvh_sah_sizes",      "bvh_sah_bases",
                                "bvh_sah_emit",       "bvh_sah_large_bounds", "bvh_sah_large_bins",
                                "bvh_sah_large_scatter", "bvh_clip_area_blocks", "bvh_clip_sums",
                                "bvh_clip_count",     "bvh_clip_scan",      "bvh_clip_instances",
                                "bvh_clip_emit"};

// Timestamps after each stage; stage i lasts from mark i to mark i + 1.
enum Mark : std::uint32_t { kMarkStart, kMarkExtents, kMarkMorton, kMarkSort, kMarkTopology, kMarkFit, kMarkEmit,
                            kMarkCount };

std::uint32_t nodeCount(std::uint32_t leaves) { return leaves > 1 ? leaves - 1 : 1; }

// The binned SAH's levels with large-task slots: until the largest segment's halves would be
// small, and four more for uneven splits. A large task deeper takes the per-task path.
std::uint32_t sahLargeLevels(const std::vector<pt::BvhSegment> &segments) {
  std::uint32_t largest = 0;
  for (const pt::BvhSegment &segment : segments) largest = std::max(largest, segment.range.y);
  if (largest <= kSahLarge) return 0;
  return std::min<std::uint32_t>(kSahLargeLevels, std::bit_width((largest - 1u) / kSahLarge) + 4u);
}
std::uint32_t groups(std::uint32_t threads, std::uint32_t size) { return (threads + size - 1) / size; }
VkDeviceSize aligned(VkDeviceSize bytes) { return (bytes + kSlot - 1) / kSlot * kSlot; }

// Orders every earlier compute write before later compute reads and writes, indirect
// dispatch arguments and transfer reads.
void barrier(VkCommandBuffer command) {
  VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  memory.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  memory.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
  memory.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  memory.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                         VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.memoryBarrierCount = 1;
  dependency.pMemoryBarriers = &memory;
  vkCmdPipelineBarrier2(command, &dependency);
}

// Orders transfer writes (restored setup sections, filled counters) before compute reads.
void transferBarrier(VkCommandBuffer command) {
  VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  memory.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  memory.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  memory.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
  memory.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                         VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.memoryBarrierCount = 1;
  dependency.pMemoryBarriers = &memory;
  vkCmdPipelineBarrier2(command, &dependency);
}

// Orders earlier compute and transfer work before transfer writes and reads (an update's copies
// into the setup and the status copy out of it).
void transferAfterCompute(VkCommandBuffer command) {
  VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  memory.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  memory.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
  memory.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  memory.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.memoryBarrierCount = 1;
  dependency.pMemoryBarriers = &memory;
  vkCmdPipelineBarrier2(command, &dependency);
}

const char *errorName(std::uint32_t error) {
  switch (error) {
  case pt::kBvhBuildErrorCapacity: return "buffer capacity or layout contract";
  case pt::kBvhBuildErrorGeometry: return "geometry index";
  case pt::kBvhBuildErrorNonFinite: return "non-finite vertex";
  case pt::kBvhBuildErrorDepth: return "builder or traversal depth";
  default: return "unknown";
  }
}

} // namespace

struct GpuLbvhBuilder::Kernel {
  std::string entry;
  std::unique_ptr<Program> program;
  Pipeline pipeline;
};

GpuLbvhBuilder::GpuLbvhBuilder(const Context &ctx) : context(ctx) {
  for (const char *entry : kEntries) {
    // The binned SAH's level kernel needs subgroup operations and the split clipping 64-bit
    // floats; without them only those fail.
    if (!deviceSupports(context, Shader(context, entry, VK_SHADER_STAGE_COMPUTE_BIT))) continue;
    auto k = std::make_unique<Kernel>();
    k->entry = entry;
    k->program = std::make_unique<Program>(context, entry);
    k->pipeline = Pipeline(context, *k->program, std::string("GPU parallel LBVH ") + entry);
    kernels.push_back(std::move(k));
  }
  VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  info.queryCount = kMarkCount;
  check(vkCreateQueryPool(context.device, &info, nullptr, &timestamps), "vkCreateQueryPool (parallel LBVH)");
}

GpuLbvhBuilder::~GpuLbvhBuilder() {
  if (timestamps) vkDestroyQueryPool(context.device, timestamps, nullptr);
}

const GpuLbvhBuilder::Kernel &GpuLbvhBuilder::kernel(const char *entry) const {
  for (const auto &k : kernels)
    if (k->entry == entry) return *k;
  throw std::runtime_error(std::string("parallel LBVH kernel ") + entry + " is missing");
}

std::vector<GpuLbvhBuilder::SortSets> GpuLbvhBuilder::sortSets(
    DescriptorPool &pool, const Buffer *keysLo, const Buffer *keysHi, const Buffer *values, const Buffer &histogram,
    const Buffer &blockSums, const Buffer &controls, VkDeviceSize controlBase, std::uint32_t passes) const {
  auto set = [&](const char *entry, const std::function<void(DescriptorWriter &)> &write) {
    const Kernel &k = kernel(entry);
    const VkDescriptorSet s = k.program->allocate(pool);
    DescriptorWriter writer(context, *k.program, s);
    write(writer);
    writer.apply();
    return s;
  };
  std::vector<SortSets> sets;
  for (std::uint32_t pass = 0; pass < passes; ++pass) {
    const std::uint32_t in = pass & 1u, out = in ^ 1u;
    const VkDeviceSize offset = controlBase + pass * kSlot, bytes = sizeof(pt::BvhSortControl);
    SortSets s{};
    s.histogram = set("bvh_sort_histogram", [&](DescriptorWriter &w) {
      w.buffer("keysLo", keysLo[in]).buffer("keysHi", keysHi[in]).buffer("histogram", histogram)
          .buffer("control", controls, offset, bytes);
    });
    s.reduce = set("bvh_sort_reduce", [&](DescriptorWriter &w) {
      w.buffer("histogram", histogram).buffer("blockSums", blockSums).buffer("control", controls, offset, bytes);
    });
    s.blocks = set("bvh_sort_blocks", [&](DescriptorWriter &w) {
      w.buffer("blockSums", blockSums).buffer("control", controls, offset, bytes);
    });
    s.apply = set("bvh_sort_apply", [&](DescriptorWriter &w) {
      w.buffer("histogram", histogram).buffer("blockSums", blockSums).buffer("control", controls, offset, bytes);
    });
    s.scatter = set("bvh_sort_scatter", [&](DescriptorWriter &w) {
      w.buffer("keysLo", keysLo[in]).buffer("keysHi", keysHi[in]).buffer("values", values[in])
          .buffer("outLo", keysLo[out]).buffer("outHi", keysHi[out]).buffer("outValues", values[out])
          .buffer("histogram", histogram).buffer("control", controls, offset, bytes);
    });
    sets.push_back(s);
  }
  return sets;
}

std::uint32_t GpuLbvhBuilder::recordSort(VkCommandBuffer command, const std::vector<SortSets> &sets,
                                         std::uint32_t tiles, std::uint32_t blocks) const {
  std::uint32_t dispatches = 0;
  auto run = [&](const char *entry, VkDescriptorSet s, std::uint32_t x) {
    const Kernel &k = kernel(entry);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline.handle);
    k.program->bind(command, s);
    if (x > 0) vkCmdDispatch(command, x, 1, 1);
    barrier(command);
    ++dispatches;
  };
  for (const SortSets &s : sets) {
    run("bvh_sort_histogram", s.histogram, tiles);
    run("bvh_sort_reduce", s.reduce, blocks);
    run("bvh_sort_blocks", s.blocks, 1);
    run("bvh_sort_apply", s.apply, blocks);
    run("bvh_sort_scatter", s.scatter, tiles);
  }
  return dispatches;
}

void GpuLbvhBuilder::sortTriples(Uploader &uploader, std::vector<std::uint32_t> &lo, std::vector<std::uint32_t> &hi,
                                 std::vector<std::uint32_t> &values, std::uint32_t highPasses) {
  const auto count = static_cast<std::uint32_t>(lo.size());
  if (hi.size() != count || values.size() != count) throw std::runtime_error("sortTriples needs equal arrays");
  const std::uint32_t tiles = groups(count, kSortTile), blocks = groups(256u * tiles, kSortTile);
  if (blocks > kSortMaximumBlocks) throw std::runtime_error("sortTriples: too many keys");
  const std::uint32_t passes = 4u + highPasses;
  std::vector<std::uint8_t> controlBytes(passes * kSlot, 0);
  for (std::uint32_t pass = 0; pass < passes; ++pass) {
    pt::BvhSortControl sort{};
    sort.sizes = {count, tiles, blocks, 0u};
    const bool high = pass >= 4u;
    sort.pass = {pass, 0u, (high ? pass - 4u : pass) * 8u, high ? 1u : 0u};
    std::memcpy(controlBytes.data() + pass * kSlot, &sort, sizeof(sort));
  }
  Buffer controls = uploader.createBuffer(controlBytes.data(), controlBytes.size(), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                          "gpu-lbvh.sort-test.controls");
  const VkDeviceSize bytes = std::max<VkDeviceSize>(16, VkDeviceSize(count) * 4u);
  const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  auto upload = [&](const std::vector<std::uint32_t> &data, const char *name) {
    std::vector<std::uint32_t> padded(static_cast<std::size_t>(bytes / 4u), 0u);
    std::copy(data.begin(), data.end(), padded.begin());
    return uploader.createBuffer(padded.data(), bytes, usage, name);
  };
  auto blank = [&](const char *name) { return Buffer(context, bytes, usage, VMA_MEMORY_USAGE_AUTO, 0, name); };
  Buffer keysLo[2] = {upload(lo, "gpu-lbvh.sort-test.lo-a"), blank("gpu-lbvh.sort-test.lo-b")};
  Buffer keysHi[2] = {upload(hi, "gpu-lbvh.sort-test.hi-a"), blank("gpu-lbvh.sort-test.hi-b")};
  Buffer valueBuffers[2] = {upload(values, "gpu-lbvh.sort-test.values-a"), blank("gpu-lbvh.sort-test.values-b")};
  Buffer histogram(context, std::max<VkDeviceSize>(16, 256ull * tiles * 4ull), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   VMA_MEMORY_USAGE_AUTO, 0, "gpu-lbvh.sort-test.histogram");
  Buffer blockSums(context, std::max<VkDeviceSize>(16, blocks * 4ull), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   VMA_MEMORY_USAGE_AUTO, 0, "gpu-lbvh.sort-test.block-sums");
  DescriptorPool pool(context, 256);
  const std::vector<SortSets> sets =
      sortSets(pool, keysLo, keysHi, valueBuffers, histogram, blockSums, controls, 0, passes);
  uploader.runImmediate([&](VkCommandBuffer command) { recordSort(command, sets, tiles, blocks); });
  if (count == 0) return;
  const std::uint32_t result = passes & 1u;
  auto download = [&](const Buffer &buffer, std::vector<std::uint32_t> &out) {
    const std::vector<std::uint8_t> data = uploader.readBuffer(buffer, VkDeviceSize(count) * 4u);
    std::memcpy(out.data(), data.data(), data.size());
  };
  download(keysLo[result], lo);
  download(keysHi[result], hi);
  download(valueBuffers[result], values);
}

// A build's setup and scratch. build() keeps it when asked, so refit() and rebuildTopLevel() can
// update the tree without rebuilding what did not change.
struct GpuLbvhBuilder::Kept {
  Buffer setup;
  Buffer pristine;  // the host-initialised setup bytes, whose sections an update restores
  Buffer records, keysLo[2], keysHi[2], values[2], histogram, blockSums;
  Buffer nodeChildren, nodeParent, nodeCounter, nodeBox, nodeSize, nodeNumber, entries;
  Buffer nodeSpan;  // the single-pass LBVH's node ranges
  Buffer referencePlaceholder;           // bound when a build has no references
  const Buffer *references = nullptr;     // the build's references (the caller's), during the build
  // The TLAS's own sort: its records' keys copied out, sorted, and copied back into its range.
  Buffer topLo[2], topHi[2], topValues[2], topHistogram, topBlockSums;
  VkDeviceSize statusOffset = 0, errorOffset = 0, controlOffset = 0, segmentOffset = 0, boxOffset = 0,
               heightOffset = 0, headerOffset = 0, levelOffset = 0, instanceOffset = 0;
  VkDeviceSize segmentBytes = 0, boxBytes = 0, heightBytes = 0, headerBytes = 0, levelBytes = 0, instanceBytes = 0;
  std::uint32_t recordCount = 0, segmentCount = 0, internalCount = 0, triangleTotal = 0, instanceCount = 0;
  std::uint32_t topNodes = 0, passes = 0, tiles = 0, blocks = 0, topTiles = 0, topBlocks = 0;
  std::uint32_t topSlot = 0;  // control slot of the TLAS part; its four sort passes follow
  std::uint32_t batchSlot = 0;  // control slot of a batched build (slot 0's, with the batched level size)
  std::vector<pt::TraceInstance> instances;       // the published rows (blasRoot, triangleOffset)
  std::vector<std::uint32_t> triangleCounts;      // per instance: an update keeps them
  VkDeviceSize scratchBytes = 0;
  // In-frame updates: their descriptor sets (for these nodes, triangles and vertices) and each
  // frame slot's status, copied to host-visible memory.
  std::unique_ptr<DescriptorPool> framePool;
  VkBuffer frameKey[3]{};
  CommonSets frameRefit, frameTop;
  std::vector<SortSets> frameSorting;
  Buffer frameStatus;
  bool framePending[kFrameSlots]{};
  VkDeviceSize controlAt(std::uint32_t slot) const { return controlOffset + slot * kSlot; }
  std::uint32_t topFirstNode() const { return internalCount - topNodes; }
};

GpuLbvhBuilder::CommonSets GpuLbvhBuilder::commonSets(DescriptorPool &pool, const Kept &k, const Scene &scene,
                                                      const Buffer &nodes, const Buffer &triangles,
                                                      std::uint32_t partSlot) const {
  auto set = [&](const Kernel &kernel, const std::function<void(DescriptorWriter &)> &write) {
    const VkDescriptorSet s = kernel.program->allocate(pool);
    DescriptorWriter writer(context, *kernel.program, s);
    write(writer);
    writer.apply();
    return s;
  };
  const std::uint32_t sorted = k.passes & 1u;
  const VkDeviceSize controlBytes = sizeof(pt::BvhLbvhControl);
  const VkDeviceSize part = k.controlAt(partSlot);
  CommonSets sets;
  sets.extents = set(kernel("bvh_lbvh_extents"), [&](DescriptorWriter &w) {
    w.buffer("segments", k.setup, k.segmentOffset, k.segmentBytes)
        .buffer("instances", k.setup, k.instanceOffset, k.instanceBytes)
        .buffer("indices", scene.indexBuffer).buffer("vertices", scene.vertexBuffer)
        .buffer("records", k.records).buffer("references", *k.references)
        .buffer("segmentBox", k.setup, k.boxOffset, k.boxBytes)
        .buffer("error", k.setup, k.errorOffset, 4).buffer("control", k.setup, part, controlBytes);
  });
  sets.instances = set(kernel("bvh_lbvh_instances"), [&](DescriptorWriter &w) {
    w.buffer("segments", k.setup, k.segmentOffset, k.segmentBytes)
        .buffer("instances", k.setup, k.instanceOffset, k.instanceBytes).buffer("records", k.records)
        .buffer("segmentBox", k.setup, k.boxOffset, k.boxBytes)
        .buffer("control", k.setup, k.controlAt(0), controlBytes);
  });
  sets.morton = set(kernel("bvh_lbvh_morton"), [&](DescriptorWriter &w) {
    w.buffer("segments", k.setup, k.segmentOffset, k.segmentBytes).buffer("records", k.records).buffer("segmentBox", k.setup, k.boxOffset, k.boxBytes)
        .buffer("keysLo", k.keysLo[0]).buffer("keysHi", k.keysHi[0]).buffer("values", k.values[0])
        .buffer("control", k.setup, part, controlBytes);
  });
  sets.topology = set(kernel("bvh_lbvh_topology"), [&](DescriptorWriter &w) {
    w.buffer("segments", k.setup, k.segmentOffset, k.segmentBytes).buffer("records", k.records)
        .buffer("codes", k.keysLo[sorted]).buffer("sortedIndex", k.values[sorted])
        .buffer("nodeChildren", k.nodeChildren).buffer("nodeParent", k.nodeParent).buffer("nodeCounter", k.nodeCounter)
        .buffer("entries", k.entries).buffer("headers", k.setup, k.headerOffset, k.headerBytes)
        .buffer("control", k.setup, part, controlBytes);
  });
  sets.seed = set(kernel("bvh_lbvh_seed"), [&](DescriptorWriter &w) {
    w.buffer("nodeChildren", k.nodeChildren).buffer("nodeCounter", k.nodeCounter).buffer("entries", k.entries)
        .buffer("headers", k.setup, k.headerOffset, k.headerBytes).buffer("control", k.setup, part, controlBytes);
  });
  for (std::uint32_t i = 0; i < kFitIterations; ++i) {
    sets.fit.push_back(set(kernel("bvh_lbvh_fit"), [&](DescriptorWriter &w) {
      w.buffer("nodeChildren", k.nodeChildren).buffer("nodeParent", k.nodeParent).buffer("nodeCounter", k.nodeCounter)
          .buffer("records", k.records).buffer("sortedIndex", k.values[sorted]).buffer("nodeBox", k.nodeBox)
          .buffer("nodeSize", k.nodeSize).buffer("entries", k.entries)
          .buffer("levelBase", k.setup, k.levelOffset, k.levelBytes)
          .buffer("headers", k.setup, k.headerOffset, k.headerBytes)
          .buffer("segmentHeight", k.setup, k.heightOffset, k.heightBytes)
          .buffer("control", k.setup, k.controlAt(i), controlBytes);
    }));
    sets.number.push_back(set(kernel("bvh_lbvh_number"), [&](DescriptorWriter &w) {
      w.buffer("nodeChildren", k.nodeChildren).buffer("nodeParent", k.nodeParent).buffer("nodeSize", k.nodeSize)
          .buffer("nodeNumber", k.nodeNumber).buffer("entries", k.entries)
          .buffer("levelBase", k.setup, k.levelOffset, k.levelBytes)
          .buffer("headers", k.setup, k.headerOffset, k.headerBytes)
          .buffer("control", k.setup, k.controlAt(i), controlBytes);
    }));
  }
  sets.emit = set(kernel("bvh_lbvh_emit"), [&](DescriptorWriter &w) {
    w.buffer("segments", k.setup, k.segmentOffset, k.segmentBytes).buffer("nodeChildren", k.nodeChildren)
        .buffer("nodeBox", k.nodeBox).buffer("nodeNumber", k.nodeNumber).buffer("records", k.records)
        .buffer("sortedIndex", k.values[sorted])
        .buffer("instances", k.setup, k.instanceOffset, k.instanceBytes)
        .buffer("indices", scene.indexBuffer).buffer("vertices", scene.vertexBuffer)
        .buffer("nodes", nodes).buffer("triangles", triangles)
        .buffer("control", k.setup, part, controlBytes);
  });
  sets.finish = set(kernel("bvh_lbvh_finish"), [&](DescriptorWriter &w) {
    w.buffer("segmentHeight", k.setup, k.heightOffset, k.heightBytes)
        .buffer("headers", k.setup, k.headerOffset, k.headerBytes).buffer("error", k.setup, k.errorOffset, 4)
        .buffer("status", k.setup, k.statusOffset, sizeof(pt::BvhBuildStatus2))
        .buffer("control", k.setup, k.controlAt(0), controlBytes);
  });
  return sets;
}

GpuBvhBuildResult GpuLbvhBuilder::build(Uploader &uploader, const Scene &scene, const TraceScene &trace,
                                        bool refittable, GpuBvhTopology topology,
                                        const GpuBvhReferences *references) {
  if (topology == GpuBvhTopology::BinnedSah && refittable)
    throw std::invalid_argument("the GPU binned SAH build keeps no LBVH to refit");
  // A refit makes a record per triangle; the references' boxes are the clipping's.
  if (references && refittable) throw std::invalid_argument("a build over split clipping's references cannot be refitted");
  if (references && references->counts.size() != trace.instances.size())
    throw std::runtime_error("GPU BVH build received references for other instances");
  const auto started = std::chrono::steady_clock::now();
  kept.reset();  // an earlier build's scratch
  auto state = std::make_unique<Kept>();
  Kept &k = *state;
  GpuBvhBuildResult result;
  result.instances = trace.instances;
  const auto instanceCount = static_cast<std::uint32_t>(result.instances.size());

  // Segments: each instance's bottom level at the serial builder's node and triangle bases
  // (its records are its triangles in primitive order, or its references in their order), then
  // the TLAS over the instances.
  std::vector<pt::BvhSegment> segments;
  segments.reserve(instanceCount + 1u);
  const std::uint32_t topNodes = nodeCount(instanceCount);
  std::uint32_t nodeOffset = topNodes, triangleOffset = 0, internalFirst = 0;
  for (std::uint32_t i = 0; i < instanceCount; ++i) {
    const std::uint32_t primitiveIndex = trace.primitives[i];
    if (primitiveIndex >= scene.primitives.size())
      throw std::runtime_error("GPU BVH build received an invalid primitive mapping");
    const std::uint32_t triangles =
        references ? references->counts[i] : scene.primitives[primitiveIndex].indexCount / 3u;
    pt::BvhSegment segment{};
    segment.range = {triangleOffset, triangles, nodeOffset, triangleOffset};
    segment.flags = {0u, i, nodeCount(triangles), internalFirst};
    segments.push_back(segment);
    result.instances[i].blasRoot = nodeOffset;
    result.instances[i].triangleOffset = triangleOffset;
    k.triangleCounts.push_back(triangles);
    nodeOffset += nodeCount(triangles);
    internalFirst += nodeCount(triangles);
    triangleOffset += triangles;
  }
  pt::BvhSegment top{};
  top.range = {triangleOffset, instanceCount, 0u, 0u};
  top.flags = {1u, 0u, topNodes, internalFirst};
  segments.push_back(top);
  internalFirst += topNodes;

  const std::uint32_t triangleTotal = triangleOffset;
  const std::uint32_t records = triangleTotal + instanceCount;
  const auto segmentCount = static_cast<std::uint32_t>(segments.size());
  const std::uint32_t internalCount = internalFirst;  // = every node: one per internal node
  const std::uint32_t tiles = groups(records, kSortTile);
  const std::uint32_t blocks = groups(256u * tiles, kSortTile);
  if (blocks > kSortMaximumBlocks)
    throw std::runtime_error("the parallel GPU LBVH sorts at most 4M triangles and instances; use --bvh-builder cpu");
  const std::uint32_t highPasses = segmentCount > 1u ? (std::bit_width(segmentCount - 1u) + 7u) / 8u : 0u;
  const std::uint32_t passes = 4u + highPasses;
  k.recordCount = records;
  k.segmentCount = segmentCount;
  k.internalCount = internalCount;
  k.triangleTotal = triangleTotal;
  k.instanceCount = instanceCount;
  k.topNodes = topNodes;
  k.passes = passes;
  k.tiles = tiles;
  k.blocks = blocks;
  k.topTiles = groups(std::max(1u, instanceCount), kSortTile);
  k.topBlocks = groups(256u * k.topTiles, kSortTile);

  // One host-initialised buffer: status, error word, per-dispatch controls (the fit levels',
  // the sort passes', then the TLAS part's and its four sort passes'), segments, segment boxes
  // (empty: min words all ones, max words zero), root heights, level queue headers (groups, 1,
  // 1, count), level entry offsets (zero) and the instance table. Sections start at 256-byte
  // offsets.
  k.statusOffset = 0;
  k.errorOffset = kSlot;
  k.controlOffset = 2 * kSlot;
  k.topSlot = kFitIterations + passes;
  k.batchSlot = k.topSlot + 5u;
  const std::uint32_t controlSlots = k.batchSlot + 1u;
  k.segmentOffset = aligned(k.controlOffset + controlSlots * kSlot);
  k.segmentBytes = segmentCount * sizeof(pt::BvhSegment);
  k.boxOffset = aligned(k.segmentOffset + k.segmentBytes);
  k.boxBytes = segmentCount * 6u * sizeof(std::uint32_t);
  k.heightOffset = aligned(k.boxOffset + k.boxBytes);
  k.heightBytes = segmentCount * sizeof(std::uint32_t);
  k.headerOffset = aligned(k.heightOffset + k.heightBytes);
  k.headerBytes = (kFitIterations + 1u) * sizeof(pt::uint4);
  k.levelOffset = aligned(k.headerOffset + k.headerBytes);
  k.levelBytes = (kFitIterations + 1u) * sizeof(std::uint32_t);
  k.instanceOffset = aligned(k.levelOffset + k.levelBytes);
  const std::uint32_t instanceRows = std::max(1u, instanceCount);
  k.instanceBytes = instanceRows * sizeof(pt::TraceInstance);
  const VkDeviceSize setupBytes = k.instanceOffset + k.instanceBytes;
  std::vector<std::uint8_t> setup(setupBytes, 0);
  pt::BvhLbvhControl control{};
  control.sizes = {records, segmentCount, internalCount, triangleTotal};
  control.geometry = {scene.indexCount, scene.vertexCount, pt::kBvhStackDeep, pt::kBvhBuildLayoutVersion};
  control.range = {instanceCount, triangleTotal, 0u, 0u};
  control.part = {0u, 0u, internalCount, triangleTotal};
  control.source = {references ? 1u : 0u, topology == GpuBvhTopology::BinnedSah ? sahLargeLevels(segments) : 0u, 0u,
                    0u};
  for (std::uint32_t i = 0; i < kFitIterations; ++i) {
    control.range.z = i;
    std::memcpy(setup.data() + k.controlAt(i), &control, sizeof(control));
  }
  control.range.z = 0;
  control.range.w = kBatchedLevel;
  std::memcpy(setup.data() + k.controlAt(k.batchSlot), &control, sizeof(control));
  control.range.w = 0;
  control.part = {triangleTotal, k.topFirstNode(), topNodes, 0u};
  std::memcpy(setup.data() + k.controlAt(k.topSlot), &control, sizeof(control));
  for (std::uint32_t pass = 0; pass < passes; ++pass) {
    pt::BvhSortControl sort{};
    sort.sizes = {records, tiles, blocks, 0u};
    const bool high = pass >= 4u;
    sort.pass = {pass, 0u, (high ? pass - 4u : pass) * 8u, high ? 1u : 0u};
    std::memcpy(setup.data() + k.controlAt(kFitIterations + pass), &sort, sizeof(sort));
  }
  for (std::uint32_t pass = 0; pass < 4u; ++pass) {
    pt::BvhSortControl sort{};
    sort.sizes = {instanceCount, k.topTiles, k.topBlocks, 0u};
    sort.pass = {pass, 0u, pass * 8u, 0u};
    std::memcpy(setup.data() + k.controlAt(k.topSlot + 1u + pass), &sort, sizeof(sort));
  }
  std::memcpy(setup.data() + k.segmentOffset, segments.data(), segments.size() * sizeof(pt::BvhSegment));
  for (std::uint32_t g = 0; g < segmentCount; ++g)
    std::memset(setup.data() + k.boxOffset + g * 6u * sizeof(std::uint32_t), 0xFF, 3u * sizeof(std::uint32_t));
  for (std::uint32_t i = 0; i <= kFitIterations; ++i) {
    const pt::uint4 header{0u, 1u, 1u, 0u};
    std::memcpy(setup.data() + k.headerOffset + i * sizeof(pt::uint4), &header, sizeof(header));
  }
  if (instanceCount > 0)
    std::memcpy(setup.data() + k.instanceOffset, result.instances.data(), instanceCount * sizeof(pt::TraceInstance));
  k.setup = uploader.createBuffer(setup.data(), setupBytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      "path.gpu-lbvh.setup");
  if (refittable)
    k.pristine = uploader.createBuffer(setup.data(), setupBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                       "path.gpu-lbvh.pristine-setup");
  const pt::BvhReference emptyReference{};
  k.referencePlaceholder = uploader.createBuffer(&emptyReference, sizeof(emptyReference), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                 "path.gpu-lbvh.no-references");
  k.references = references ? &references->references : &k.referencePlaceholder;
  result.nodes = Buffer(context, VkDeviceSize(internalCount) * 4u * sizeof(pt::float4),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VMA_MEMORY_USAGE_AUTO, 0, "path.gpu-bvh.nodes");
  result.triangles = Buffer(context, VkDeviceSize(std::max(1u, triangleTotal)) * 3u * sizeof(pt::float4),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VMA_MEMORY_USAGE_AUTO, 0, "path.gpu-bvh.triangles");

  k.scratchBytes = setupBytes + (refittable ? setupBytes : 0);
  if (topology == GpuBvhTopology::BinnedSah) {
    buildBinnedSah(uploader, scene, k, segments, result);
    result.statistics.milliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    result.milliseconds = result.statistics.milliseconds;
    return result;
  }
  auto scratch = [&](VkDeviceSize bytes, const char *name, VkBufferUsageFlags extra = 0) {
    bytes = std::max<VkDeviceSize>(bytes, 16);
    k.scratchBytes += bytes;
    return Buffer(context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | extra, VMA_MEMORY_USAGE_AUTO, 0, name);
  };
  const VkBufferUsageFlags copies = refittable ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT : 0;
  k.records = scratch(VkDeviceSize(records) * sizeof(pt::BvhBuildRecord), "path.gpu-lbvh.records");
  k.keysLo[0] = scratch(records * 4ull, "path.gpu-lbvh.keys-lo-a", copies);
  k.keysLo[1] = scratch(records * 4ull, "path.gpu-lbvh.keys-lo-b", copies);
  k.keysHi[0] = scratch(records * 4ull, "path.gpu-lbvh.keys-hi-a");
  k.keysHi[1] = scratch(records * 4ull, "path.gpu-lbvh.keys-hi-b");
  k.values[0] = scratch(records * 4ull, "path.gpu-lbvh.values-a", copies);
  k.values[1] = scratch(records * 4ull, "path.gpu-lbvh.values-b", copies);
  k.histogram = scratch(256ull * tiles * 4ull, "path.gpu-lbvh.histogram");
  k.blockSums = scratch(blocks * 4ull, "path.gpu-lbvh.block-sums");
  // PLOC++ is PLOC with each wide iteration one fused dispatch (bvh_ploc_fused).
  const bool plocFused = topology == GpuBvhTopology::PlocPlusPlus;
  // H-PLOC shares PLOC's node records and publication, not its iterations.
  const bool hploc = topology == GpuBvhTopology::Hploc;
  const bool ploc = topology == GpuBvhTopology::Ploc || plocFused || hploc;
  // The batched build is the single-pass LBVH with small bottom levels built one workgroup each.
  const bool batched = topology == GpuBvhTopology::BatchedLbvh;
  const bool singlePass = topology == GpuBvhTopology::SinglePassLbvh || batched;
  const std::uint32_t buildSlot = batched ? k.batchSlot : 0u;
  if (singlePass) k.nodeSpan = scratch(internalCount * 8ull, "path.gpu-lbvh.node-span");
  const VkBufferUsageFlags filled = ploc ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : 0;  // PLOC starts them empty
  k.nodeChildren = scratch(internalCount * sizeof(pt::uint4), "path.gpu-lbvh.node-children", filled);
  k.nodeParent = scratch(internalCount * 4ull, "path.gpu-lbvh.node-parent", filled);
  k.nodeCounter = scratch(internalCount * 4ull, "path.gpu-lbvh.node-counter", VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  k.nodeBox = scratch(internalCount * 2ull * sizeof(pt::float4), "path.gpu-lbvh.node-box");
  k.nodeSize = scratch(internalCount * 4ull, "path.gpu-lbvh.node-size");
  k.nodeNumber = scratch(internalCount * 8ull, "path.gpu-lbvh.node-number");
  k.entries = scratch(internalCount * 4ull, "path.gpu-lbvh.level-entries");
  // PLOC: the clusters and their boxes (two halves of a buffer each, 256-byte aligned), nearest
  // neighbours, kept flags scanned per block of 128 and the block sums, each segment's node
  // cursor, and two iteration sizes (bvh_ploc.slang), a slot apart: an iteration reads one and
  // writes the next's into the other. The iterations over all workgroups run while the clusters
  // are many: about until the finish's threadgroup tile holds them (kPlocShared), at the fifth
  // that a PLOC iteration typically removes (the rate only sets how the work splits; the
  // one-workgroup finish ends every segment as one tree from any count).
  constexpr std::uint32_t kPlocGroup = 128, kPlocShared = 256;
  const std::uint32_t plocWideIterations =
      records > kPlocShared
          ? static_cast<std::uint32_t>(std::ceil(std::log(double(records) / kPlocShared) / std::log(1.25)))
          : 0u;
  const VkDeviceSize clusterHalf = aligned(std::max(records, 1u) * 8ull);
  const VkDeviceSize boxHalf = clusterHalf * 4u;  // two float4 per cluster
  Buffer clusters, clusterBoxes, segmentNext, neighbour, plocKept, plocSums, plocSizes, lookback, hplocCluster, hplocMeet;
  if (ploc) {
    clusters = scratch(2u * clusterHalf, "path.gpu-ploc.clusters");
    clusterBoxes = scratch(2u * boxHalf, "path.gpu-ploc.cluster-boxes");
    segmentNext = scratch(segmentCount * 4ull, "path.gpu-ploc.segment-next");
    neighbour = scratch(records * 4ull, "path.gpu-ploc.neighbours");
    plocKept = scratch(records * 4ull, "path.gpu-ploc.kept");
    plocSums = scratch(groups(records, kPlocGroup) * 4ull, "path.gpu-ploc.block-sums");
    if (plocFused)
      lookback = scratch((1ull + groups(records, kPlocGroup)) * 4ull, "path.gpu-ploc.look-back",
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (hploc) {
      hplocCluster = scratch(records * 4ull, "path.gpu-hploc.clusters");
      hplocMeet = scratch(records * 4ull, "path.gpu-hploc.meet", VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }
    std::vector<std::uint32_t> sizes(2u * kSlot / 4u, 0u);
    const std::uint32_t first[9] = {groups(records, kPlocGroup), 1u, 1u, records, 0u, 0u, 0u, 0u, 0u};
    std::copy(std::begin(first), std::end(first), sizes.begin());
    sizes[kSlot / 4u + 8u] = 1u;  // the second slot's parity
    plocSizes = uploader.createBuffer(sizes.data(), sizes.size() * 4u,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        "path.gpu-ploc.sizes");
    k.scratchBytes += plocSizes.size;
  }
  if (refittable) {
    const VkDeviceSize topKeys = std::max(1u, instanceCount) * 4ull;
    const VkBufferUsageFlags both = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    for (std::uint32_t i = 0; i < 2u; ++i) {
      k.topLo[i] = scratch(topKeys, "path.gpu-lbvh.top-keys-lo", both);
      k.topHi[i] = scratch(topKeys, "path.gpu-lbvh.top-keys-hi", both);
      k.topValues[i] = scratch(topKeys, "path.gpu-lbvh.top-values", both);
    }
    k.topHistogram = scratch(256ull * k.topTiles * 4ull, "path.gpu-lbvh.top-histogram");
    k.topBlockSums = scratch(k.topBlocks * 4ull, "path.gpu-lbvh.top-block-sums");
  }

  DescriptorPool pool(context, 256);
  const CommonSets common = commonSets(pool, k, scene, result.nodes, result.triangles, buildSlot);
  const std::vector<SortSets> sorting = sortSets(pool, k.keysLo, k.keysHi, k.values, k.histogram, k.blockSums,
                                                 k.setup, k.controlAt(kFitIterations), passes);
  // PLOC's sets: one per parity for the iterations (clusters and sizes read from [p], written
  // to [p ^ 1]).
  struct PlocSets {
    VkDescriptorSet init = VK_NULL_HANDLE, single = VK_NULL_HANDLE, finish = VK_NULL_HANDLE, hploc = VK_NULL_HANDLE;
    VkDescriptorSet neighbours[2]{}, scanSums[2]{}, compact[2]{}, fused[2]{};
  } plocSets;
  if (ploc) {
    const std::uint32_t sortedSlot = passes & 1u;
    auto set = [&](const char *entry, const std::function<void(DescriptorWriter &)> &write) {
      const Kernel &kk = kernel(entry);
      const VkDescriptorSet s = kk.program->allocate(pool);
      DescriptorWriter writer(context, *kk.program, s);
      write(writer);
      writer.apply();
      return s;
    };
    const VkDeviceSize controlBytes = sizeof(pt::BvhLbvhControl);
    plocSets.init = set("bvh_ploc_init", [&](DescriptorWriter &w) {
      w.buffer("records", k.records).buffer("sortedIndex", k.values[sortedSlot]).buffer("clusters", clusters, 0, clusterHalf)
          .buffer("boxes", clusterBoxes, 0, boxHalf).buffer("control", k.setup, k.controlAt(0), controlBytes);
    });
    plocSets.single = set("bvh_ploc_single", [&](DescriptorWriter &w) {
      w.buffer("segments", k.setup, k.segmentOffset, k.segmentBytes).buffer("nodeChildren", k.nodeChildren)
          .buffer("segmentNext", segmentNext).buffer("control", k.setup, k.controlAt(0), controlBytes);
    });
    if (hploc)
      plocSets.hploc = set("bvh_hploc", [&](DescriptorWriter &w) {
        w.buffer("control", k.setup, k.controlAt(0), controlBytes)
            .buffer("segments", k.setup, k.segmentOffset, k.segmentBytes).buffer("records", k.records)
            .buffer("sortedIndex", k.values[sortedSlot]).buffer("codes", k.keysLo[sortedSlot])
            .buffer("nodeChildren", k.nodeChildren).buffer("nodeParent", k.nodeParent)
            .buffer("segmentNext", segmentNext).buffer("hplocCluster", hplocCluster).buffer("hplocMeet", hplocMeet)
            .buffer("nodeBox", k.nodeBox);
      });
    const VkDeviceSize sizeBytes = 48;
    plocSets.finish = set("bvh_ploc_finish", [&](DescriptorWriter &w) {
      w.buffer("nodeChildren", k.nodeChildren).buffer("nodeParent", k.nodeParent).buffer("clusters", clusters)
          .buffer("boxes", clusterBoxes).buffer("neighbour", neighbour).buffer("segmentNext", segmentNext)
          .buffer("current", plocSizes, (plocWideIterations & 1u) * kSlot, sizeBytes)
          .buffer("control", k.setup, k.controlAt(0), controlBytes);
    });
    for (std::uint32_t p = 0; p < 2u; ++p) {
      const VkDeviceSize current = p * kSlot, next = (p ^ 1u) * kSlot;
      const VkDeviceSize half = p * clusterHalf, nextHalf = (p ^ 1u) * clusterHalf;
      const VkDeviceSize boxes = p * boxHalf, nextBoxes = (p ^ 1u) * boxHalf;
      plocSets.neighbours[p] = set("bvh_ploc_neighbours", [&](DescriptorWriter &w) {
        w.buffer("clusters", clusters, half, clusterHalf).buffer("boxes", clusterBoxes, boxes, boxHalf)
            .buffer("neighbour", neighbour).buffer("kept", plocKept).buffer("blockSums", plocSums)
            .buffer("current", plocSizes, current, 32);
      });
      plocSets.scanSums[p] = set("bvh_ploc_scan_sums", [&](DescriptorWriter &w) {
        w.buffer("blockSums", plocSums).buffer("current", plocSizes, current, 32).buffer("next", plocSizes, next, 32);
      });
      if (plocFused)
        plocSets.fused[p] = set("bvh_ploc_fused", [&](DescriptorWriter &w) {
          w.buffer("nodeChildren", k.nodeChildren).buffer("nodeParent", k.nodeParent)
              .buffer("clusters", clusters, half, clusterHalf).buffer("boxes", clusterBoxes, boxes, boxHalf)
              .buffer("nextClusters", clusters, nextHalf, clusterHalf)
              .buffer("nextBoxes", clusterBoxes, nextBoxes, boxHalf).buffer("segmentNext", segmentNext)
              .buffer("current", plocSizes, current, 32).buffer("next", plocSizes, next, 32)
              .buffer("lookback", lookback);
        });
      plocSets.compact[p] = set("bvh_ploc_compact", [&](DescriptorWriter &w) {
        w.buffer("nodeChildren", k.nodeChildren).buffer("nodeParent", k.nodeParent)
            .buffer("clusters", clusters, half, clusterHalf).buffer("boxes", clusterBoxes, boxes, boxHalf)
            .buffer("neighbour", neighbour).buffer("kept", plocKept).buffer("blockSums", plocSums)
            .buffer("nextClusters", clusters, nextHalf, clusterHalf)
            .buffer("nextBoxes", clusterBoxes, nextBoxes, boxHalf).buffer("segmentNext", segmentNext)
            .buffer("current", plocSizes, current, 32);
      });
    }
  }

  // The single-pass LBVH's sets: its kernels read the sorted keys and write the node records the
  // numbering and emit read, as the Karras topology and fit do.
  struct SinglePassSets {
    VkDescriptorSet single = VK_NULL_HANDLE, climb = VK_NULL_HANDLE, levels = VK_NULL_HANDLE, scatter = VK_NULL_HANDLE,
                    batched = VK_NULL_HANDLE;
  } singlePassSets;
  if (singlePass) {
    const std::uint32_t sortedSlot = passes & 1u;
    auto set = [&](const char *entry, const std::function<void(DescriptorWriter &)> &write) {
      const Kernel &kk = kernel(entry);
      const VkDescriptorSet s = kk.program->allocate(pool);
      DescriptorWriter writer(context, *kk.program, s);
      write(writer);
      writer.apply();
      return s;
    };
    const VkDeviceSize controlBytes = sizeof(pt::BvhLbvhControl);
    auto nodes = [&](DescriptorWriter &w) {
      w.buffer("control", k.setup, k.controlAt(buildSlot), controlBytes)
          .buffer("segments", k.setup, k.segmentOffset, k.segmentBytes).buffer("records", k.records)
          .buffer("codes", k.keysLo[sortedSlot]).buffer("sortedIndex", k.values[sortedSlot])
          .buffer("nodeChildren", k.nodeChildren).buffer("nodeParent", k.nodeParent).buffer("nodeSpan", k.nodeSpan)
          .buffer("nodeBox", k.nodeBox).buffer("nodeSize", k.nodeSize).buffer("nodeNumber", k.nodeNumber)
          .buffer("nodeCounter", k.nodeCounter).buffer("headers", k.setup, k.headerOffset, k.headerBytes)
          .buffer("segmentHeight", k.setup, k.heightOffset, k.heightBytes).buffer("error", k.setup, k.errorOffset, 4);
    };
    singlePassSets.single = set("bvh_apetrei_single", nodes);
    singlePassSets.climb = set("bvh_apetrei_climb", nodes);
    singlePassSets.levels = set("bvh_apetrei_levels", [&](DescriptorWriter &w) {
      w.buffer("headers", k.setup, k.headerOffset, k.headerBytes).buffer("levelBase", k.setup, k.levelOffset, k.levelBytes);
    });
    if (batched)
      singlePassSets.batched = set("bvh_batched_build", [&](DescriptorWriter &w) {
        w.buffer("control", k.setup, k.controlAt(buildSlot), controlBytes)
            .buffer("segments", k.setup, k.segmentOffset, k.segmentBytes)
            .buffer("instances", k.setup, k.instanceOffset, k.instanceBytes)
            .buffer("indices", scene.indexBuffer).buffer("vertices", scene.vertexBuffer).buffer("records", k.records)
            .buffer("references", *k.references)
            .buffer("segmentBox", k.setup, k.boxOffset, k.boxBytes).buffer("keysLo", k.keysLo[0])
            .buffer("keysHi", k.keysHi[0]).buffer("values", k.values[0]).buffer("nodeChildren", k.nodeChildren)
            .buffer("nodeParent", k.nodeParent).buffer("nodeBox", k.nodeBox).buffer("nodeSize", k.nodeSize)
            .buffer("nodeNumber", k.nodeNumber).buffer("headers", k.setup, k.headerOffset, k.headerBytes)
            .buffer("segmentHeight", k.setup, k.heightOffset, k.heightBytes).buffer("error", k.setup, k.errorOffset, 4);
      });
    singlePassSets.scatter = set("bvh_apetrei_scatter", [&](DescriptorWriter &w) {
      w.buffer("control", k.setup, k.controlAt(0), controlBytes).buffer("nodeNumber", k.nodeNumber)
          .buffer("levelBase", k.setup, k.levelOffset, k.levelBytes).buffer("entries", k.entries);
    });
  }

  std::uint32_t dispatches = 0;
  uploader.runImmediate([&](VkCommandBuffer command) {
    auto run = [&](const char *entry, VkDescriptorSet s, std::uint32_t x) {
      const Kernel &kk = kernel(entry);
      vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, kk.pipeline.handle);
      kk.program->bind(command, s);
      if (x > 0) vkCmdDispatch(command, x, 1, 1);
      barrier(command);
      ++dispatches;
    };
    // A timestamp at each stage boundary, and the stage as a named range for GPU profilers.
    const char *const stages[kMarkCount] = {batched ? "batched levels and LBVH extents" : "LBVH extents",
                                            "LBVH morton", "LBVH sort",
                                            hploc ? "H-PLOC topology" : plocFused ? "PLOC++ topology" :
                                            ploc ? "PLOC topology" :
                                            singlePass ? "single-pass LBVH climb" : "LBVH topology",
                                            singlePass ? "single-pass LBVH levels" : "LBVH fit",
                                            "LBVH number and emit", nullptr};
    auto mark = [&](Mark m) {
      vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, timestamps, m);
      if (m != kMarkStart) context.endLabel(command);
      if (stages[m]) context.beginLabel(command, stages[m]);
    };
    vkCmdResetQueryPool(command, timestamps, 0, kMarkCount);
    mark(kMarkStart);
    // A batched build's small bottom levels first: their records, boxes, keys and nodes.
    if (batched) run("bvh_batched_build", singlePassSets.batched, std::max(1u, segmentCount - 1u));
    run("bvh_lbvh_extents", common.extents, groups(triangleTotal, 64));
    run("bvh_lbvh_instances", common.instances, groups(instanceCount, 64));
    mark(kMarkExtents);
    run("bvh_lbvh_morton", common.morton, groups(records, 64));
    mark(kMarkMorton);
    dispatches += recordSort(command, sorting, tiles, blocks);
    mark(kMarkSort);
    if (ploc) {
      // PLOC: every node's parent starts empty (a segment's root keeps it) and every node
      // unmade (children and segment empty: never fitted or emitted, should a segment not
      // finish); segments of at most one record get their node; then the merge iterations
      // over all workgroups, each sized by the cluster count the previous one left (three
      // dispatches each), and the one-workgroup finish; the fit starts from the nodes without
      // internal children.
      vkCmdFillBuffer(command, k.nodeParent.handle, 0, VK_WHOLE_SIZE, 0xFFFFFFFFu);
      vkCmdFillBuffer(command, k.nodeChildren.handle, 0, VK_WHOLE_SIZE, 0xFFFFFFFFu);
      if (hploc) vkCmdFillBuffer(command, hplocMeet.handle, 0, VK_WHOLE_SIZE, 0xFFFFFFFFu);
      transferBarrier(command);
      if (hploc) {
        // The segments' node cursors and one-record nodes, then the whole climb in one dispatch.
        run("bvh_ploc_single", plocSets.single, groups(segmentCount, kPlocGroup));
        run("bvh_hploc", plocSets.hploc, groups(records, 32));
      }
      if (!hploc) run("bvh_ploc_init", plocSets.init, groups(records, kPlocGroup));
      if (!hploc) run("bvh_ploc_single", plocSets.single, groups(segmentCount, kPlocGroup));
      auto indirect = [&](const char *entry, VkDescriptorSet s, VkDeviceSize offset) {
        const Kernel &kk = kernel(entry);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, kk.pipeline.handle);
        kk.program->bind(command, s);
        vkCmdDispatchIndirect(command, plocSizes.handle, offset);
        barrier(command);
        ++dispatches;
      };
      for (std::uint32_t iteration = 0; !hploc && iteration < plocWideIterations; ++iteration) {
        const std::uint32_t p = iteration & 1u;
        if (plocFused) {
          // The tickets and the blocks' look-back states start at zero every iteration.
          vkCmdFillBuffer(command, lookback.handle, 0, VK_WHOLE_SIZE, 0u);
          transferBarrier(command);
          indirect("bvh_ploc_fused", plocSets.fused[p], p * kSlot);
          continue;
        }
        indirect("bvh_ploc_neighbours", plocSets.neighbours[p], p * kSlot);
        run("bvh_ploc_scan_sums", plocSets.scanSums[p], 1);
        indirect("bvh_ploc_compact", plocSets.compact[p], p * kSlot);
      }
      if (!hploc) run("bvh_ploc_finish", plocSets.finish, 1);
      run("bvh_lbvh_seed", common.seed, groups(internalCount, 64));
    } else if (singlePass) {
      // Arrivals start at zero; segments of at most one record get their node; the climb builds
      // and fits the rest.
      vkCmdFillBuffer(command, k.nodeCounter.handle, 0, VK_WHOLE_SIZE, 0u);
      transferBarrier(command);
      run("bvh_apetrei_single", singlePassSets.single, groups(segmentCount, 64));
      run("bvh_apetrei_climb", singlePassSets.climb, groups(records, 64));
    } else {
      run("bvh_lbvh_topology", common.topology, groups(internalCount, 64));
    }
    mark(kMarkTopology);
    if (singlePass) {
      // The climb queued every node at its height; place the queues for the numbering.
      run("bvh_apetrei_levels", singlePassSets.levels, 1);
      run("bvh_apetrei_scatter", singlePassSets.scatter, groups(internalCount, 64));
    } else {
      dispatches += recordFit(command, k, common);
    }
    mark(kMarkFit);
    dispatches += recordNumber(command, k, common);
    run("bvh_lbvh_emit", common.emit, groups(internalCount + triangleTotal, 64));
    run("bvh_lbvh_finish", common.finish, 1);
    mark(kMarkEmit);
  });

  if (ploc && !hploc) {
    // Finished: one cluster left per segment with records (in the sizes the finish wrote).
    std::uint32_t plocResult[8]{};
    const std::vector<std::uint8_t> sizes = uploader.readBuffer(plocSizes, plocSizes.size);
    std::memcpy(plocResult, sizes.data() + (plocWideIterations & 1u) * kSlot, sizeof(plocResult));
    std::uint32_t roots = 0;
    for (const pt::BvhSegment &segment : segments) roots += segment.range.y != 0u ? 1u : 0u;
    if (plocResult[3] != roots)
      throw std::runtime_error("GPU PLOC build did not reduce every segment to one tree (" +
                               std::to_string(plocResult[3]) + " clusters for " + std::to_string(roots) + " trees)");
    result.status.plocIterations = plocResult[7];
  }
  const pt::BvhBuildStatus2 status = readStatus(uploader, k, hploc        ? "GPU H-PLOC build" :
                                                             plocFused    ? "GPU PLOC++ build" :
                                                             ploc         ? "GPU PLOC build" :
                                                             batched    ? "GPU batched LBVH build" :
                                                             singlePass ? "GPU single-pass LBVH build" :
                                                                          "GPU parallel LBVH build",
                                                internalCount);
  double gpuMilliseconds = 0.0;
  result.stageMilliseconds = readStages(gpuMilliseconds);

  result.statistics.instances = instanceCount;
  result.statistics.triangles = triangleTotal;
  result.statistics.topNodes = topNodes;
  result.statistics.bottomNodes = internalCount - topNodes;
  result.statistics.topDepth = status.result.y;
  result.statistics.bottomDepth = status.result.z;
  result.statistics.milliseconds =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  result.milliseconds = result.statistics.milliseconds;
  result.scratchBytes = k.scratchBytes;
  result.outputBytes = result.nodes.size + result.triangles.size;
  result.radixPasses = passes;
  result.maximumBuilderStack = 0;
  result.status.error = status.result.x;
  result.status.topDepth = status.result.y;
  result.status.bottomDepth = status.result.z;
  result.status.nodes = status.counts.x;
  result.status.triangles = status.counts.y;
  result.status.instances = status.counts.z;
  result.status.sortPasses = passes;
  result.status.dispatches = dispatches;
  result.status.fitIterations = status.extra.z;

  const pt::LayoutCost cost = gpuBvhSahCost(context, uploader, result.nodes, internalCount, result.instances, false);
  result.status.topCost = cost.top;
  result.status.bottomCost = cost.bottom;
  result.statistics.sahCost = cost.bottom;
  if (refittable) {
    k.instances = result.instances;
    kept = std::move(state);
  }
  return result;
}

// The binned SAH build (bvh_sah.slang) after the setup: the LBVH's records, one dispatch per tree
// level (indirect, sized by the level before), the subtree sizes, the segments' bases and the
// emit. Each segment's nodes are stored in its LBVH node range, which holds as many as its tree
// can have, and published contiguously in preorder, so the host reads the bottom levels' roots.
// In the first levels, large tasks' chunks are bounded (level 0), binned before and scattered
// after the level's dispatch; the host makes level 0's slots.
void GpuLbvhBuilder::buildBinnedSah(Uploader &uploader, const Scene &scene, Kept &k,
                                    const std::vector<pt::BvhSegment> &segments, GpuBvhBuildResult &result) {
  constexpr std::uint32_t kSahLevels = 64;  // bvh_sah.slang; level l reads control slot l (range.z = l)
  static_assert(kSahLevels <= kFitIterations);
  constexpr std::uint32_t kEmpty = 0xFFFFFFFFu;
  if (!sahAvailable())
    throw std::runtime_error("the GPU binned SAH build needs subgroups of 32 to 128 lanes with arithmetic, ballot and "
                             "broadcast operations");
  const std::uint32_t records = k.recordCount, segmentCount = k.segmentCount, internalCount = k.internalCount;
  const std::uint32_t triangleTotal = k.triangleTotal, instanceCount = k.instanceCount;
  // A level's tasks: a segment's root each at level 0, then at most one per record.
  const std::uint32_t taskCapacity = std::max(records, segmentCount);
  auto scratch = [&](VkDeviceSize bytes, const char *name, VkBufferUsageFlags extra = 0) {
    bytes = std::max<VkDeviceSize>(bytes, 16);
    k.scratchBytes += bytes;
    return Buffer(context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | extra, VMA_MEMORY_USAGE_AUTO, 0, name);
  };
  auto upload = [&](std::vector<std::uint32_t> words, VkDeviceSize bytes, VkBufferUsageFlags extra, const char *name) {
    words.resize(static_cast<std::size_t>(std::max<VkDeviceSize>(bytes, 16) / 4u), 0u);
    k.scratchBytes += words.size() * 4u;
    return uploader.createBuffer(words.data(), words.size() * 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | extra, name);
  };
  std::vector<std::uint32_t> roots, identity(records), args(4u, 1u);
  for (std::uint32_t g = 0; g < segmentCount; ++g)
    roots.insert(roots.end(), {g, segments[g].range.x, segments[g].range.y, kEmpty});
  for (std::uint32_t i = 0; i < records; ++i) identity[i] = i;
  args[0] = args[3] = segmentCount;  // level 0: a workgroup per segment (its tasks, whatever the subgroup size)
  const VkBufferUsageFlags zeroed = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  k.records = scratch(VkDeviceSize(records) * sizeof(pt::BvhBuildRecord), "path.gpu-lbvh.records");
  const Buffer tasks[2] = {upload(roots, taskCapacity * 16ull, 0, "path.gpu-sah.tasks-a"),
                           scratch(taskCapacity * 16ull, "path.gpu-sah.tasks-b")};
  const Buffer order[2] = {upload(identity, records * 4ull, 0, "path.gpu-sah.order-a"),
                           scratch(records * 4ull, "path.gpu-sah.order-b")};
  const Buffer cursor[2] = {scratch(segmentCount * 4ull, "path.gpu-sah.segment-cursor-a"),
                            scratch(segmentCount * 4ull, "path.gpu-sah.segment-cursor-b")};
  const Buffer finalOrder = scratch(records * 4ull, "path.gpu-sah.final-order");
  const Buffer nodeSlots = scratch(internalCount * 4ull * sizeof(pt::float4), "path.gpu-sah.node-slots");
  const Buffer nodeParent = scratch(internalCount * 4ull, "path.gpu-sah.node-parent");
  const Buffer subtree = scratch(internalCount * 4ull, "path.gpu-sah.subtree");
  const Buffer arrivals = scratch(internalCount * 4ull, "path.gpu-sah.arrivals", zeroed);
  const Buffer segmentNodes = scratch(segmentCount * 4ull, "path.gpu-sah.segment-nodes", zeroed);
  const Buffer segmentDepth = scratch(segmentCount * 4ull, "path.gpu-sah.segment-depth", zeroed);
  const Buffer segmentStart = scratch(segmentCount * 4ull, "path.gpu-sah.segment-start", zeroed);
  const Buffer segmentBase = scratch((segmentCount + 1ull) * 4ull, "path.gpu-sah.segment-base",
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  const Buffer levelArgs = upload(args, (kSahLevels + 1ull) * 16ull, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                                  "path.gpu-sah.level-args");
  const Buffer lookback = scratch((kSahLevels + taskCapacity) * 4ull, "path.gpu-sah.look-back", zeroed);

  // Large tasks: level 0's slots (the large roots), its chunk table and the per-level dispatches.
  // Each level's slots: its large tasks are at most one per kSahLarge records, and at most twice
  // the level before's (a large task's parent is large).
  const std::uint32_t largeLevels = sahLargeLevels(segments);
  std::uint32_t largeRoots = 0;
  for (const pt::BvhSegment &segment : segments) largeRoots += segment.range.y > kSahLarge ? 1u : 0u;
  std::vector<std::uint32_t> largeBase(kSahLargeLevels + 2u, 0u);
  std::uint32_t levelSlots = largeRoots, mostSlots = 0;
  for (std::uint32_t l = 0; l < kSahLargeLevels + 1u; ++l) {
    const std::uint32_t slotsHere = l < largeLevels ? std::min(levelSlots, records / kSahLarge + 1u) : 0u;
    largeBase[l + 1u] = largeBase[l] + slotsHere;
    mostSlots = std::max(mostSlots, slotsHere);
    levelSlots = std::min<std::uint64_t>(std::uint64_t(levelSlots) * 2u, records);
  }
  const std::uint32_t chunkCapacity = records / kSahChunk + mostSlots + 1u;
  std::vector<std::uint32_t> slots(std::size_t(std::max(1u, largeBase.back())) * kSahSlotWords, 0u);
  std::vector<std::uint32_t> rootSlot(segmentCount, kEmpty), rootChunks, largeArgs(kSahLargeLevels * 4u, 1u);
  for (std::uint32_t l = 0; l < kSahLargeLevels; ++l) largeArgs[l * 4u] = largeArgs[l * 4u + 3u] = 0u;
  for (std::uint32_t g = 0; g < segmentCount && largeLevels > 0; ++g) {
    const std::uint32_t count = segments[g].range.y;
    if (count <= kSahLarge) continue;
    const std::uint32_t slot = largeArgs[3]++, chunks = (count + kSahChunk - 1u) / kSahChunk;
    std::uint32_t *words = slots.data() + std::size_t(slot) * kSahSlotWords;
    for (std::uint32_t w = 0; w < 12u; ++w) words[w] = w % 6u < 3u ? kEmpty : 0u;
    for (std::uint32_t w = 0; w < 3u * 16u * 7u; ++w) words[12u + w] = w % 7u < 3u ? kEmpty : 0u;
    words[kSahSlotWords - 4u] = g;                              // its task: level 0's are the segments
    words[kSahSlotWords - 3u] = largeArgs[0];                   // its first chunk
    words[kSahSlotWords - 2u] = chunks;
    rootSlot[g] = slot;
    rootChunks.insert(rootChunks.end(), chunks, slot);
    largeArgs[0] += chunks;
  }
  const Buffer large = upload(slots, slots.size() * 4ull, 0, "path.gpu-sah.large-slots");
  const Buffer largeBases = upload(largeBase, largeBase.size() * 4ull, 0, "path.gpu-sah.large-bases");
  const Buffer largeOf[2] = {upload(rootSlot, taskCapacity * 4ull, 0, "path.gpu-sah.large-of-a"),
                             scratch(taskCapacity * 4ull, "path.gpu-sah.large-of-b")};
  const Buffer chunkSlot[2] = {upload(rootChunks, chunkCapacity * 4ull, 0, "path.gpu-sah.chunk-slot-a"),
                               scratch(chunkCapacity * 4ull, "path.gpu-sah.chunk-slot-b")};
  const Buffer largeDispatch = upload(largeArgs, largeArgs.size() * 4ull, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                                      "path.gpu-sah.large-args");
  const Buffer chunkState = scratch((kSahLargeLevels + chunkCapacity) * 4ull, "path.gpu-sah.chunk-state", zeroed);

  DescriptorPool pool(context, 128);
  auto set = [&](const char *entry, const std::function<void(DescriptorWriter &)> &write) {
    const Kernel &kk = kernel(entry);
    const VkDescriptorSet s = kk.program->allocate(pool);
    DescriptorWriter writer(context, *kk.program, s);
    write(writer);
    writer.apply();
    return s;
  };
  const VkDeviceSize controlBytes = sizeof(pt::BvhLbvhControl);
  const VkDescriptorSet extents = set("bvh_lbvh_extents", [&](DescriptorWriter &w) {
    w.buffer("segments", k.setup, k.segmentOffset, k.segmentBytes)
        .buffer("instances", k.setup, k.instanceOffset, k.instanceBytes)
        .buffer("indices", scene.indexBuffer).buffer("vertices", scene.vertexBuffer)
        .buffer("records", k.records).buffer("references", *k.references)
        .buffer("segmentBox", k.setup, k.boxOffset, k.boxBytes)
        .buffer("error", k.setup, k.errorOffset, 4).buffer("control", k.setup, k.controlAt(0), controlBytes);
  });
  const VkDescriptorSet instances = set("bvh_lbvh_instances", [&](DescriptorWriter &w) {
    w.buffer("segments", k.setup, k.segmentOffset, k.segmentBytes)
        .buffer("instances", k.setup, k.instanceOffset, k.instanceBytes).buffer("records", k.records)
        .buffer("segmentBox", k.setup, k.boxOffset, k.boxBytes)
        .buffer("control", k.setup, k.controlAt(0), controlBytes);
  });
  // Level l reads the tasks, order and segment cursors at [l & 1] and writes the next level's.
  std::vector<VkDescriptorSet> levels;
  for (std::uint32_t level = 0; level < kSahLevels; ++level) {
    const std::uint32_t p = level & 1u;
    levels.push_back(set("bvh_sah_level", [&](DescriptorWriter &w) {
      w.buffer("control", k.setup, k.controlAt(level), controlBytes)
          .buffer("segments", k.setup, k.segmentOffset, k.segmentBytes).buffer("records", k.records)
          .buffer("tasks", tasks[p]).buffer("nextTasks", tasks[p ^ 1u]).buffer("order", order[p])
          .buffer("nextOrder", order[p ^ 1u]).buffer("finalOrder", finalOrder).buffer("nodeSlots", nodeSlots)
          .buffer("nodeParent", nodeParent).buffer("segmentNodes", segmentNodes)
          .buffer("segmentDepth", segmentDepth).buffer("levelArgs", levelArgs)
          .buffer("error", k.setup, k.errorOffset, 4).buffer("lookback", lookback)
          .buffer("segmentStart", segmentStart).buffer("segmentCursor", cursor[p])
          .buffer("nextSegmentCursor", cursor[p ^ 1u]).buffer("large", large).buffer("largeOf", largeOf[p])
          .buffer("nextLargeOf", largeOf[p ^ 1u]).buffer("largeArgs", largeDispatch).buffer("largeBase", largeBases)
          .buffer("nextChunkSlot", chunkSlot[p ^ 1u]);
    }));
  }
  // The chunk kernels of the levels with large tasks.
  auto chunks = [&](DescriptorWriter &w, std::uint32_t level) {
    const std::uint32_t p = level & 1u;
    w.buffer("control", k.setup, k.controlAt(level), controlBytes).buffer("records", k.records)
        .buffer("tasks", tasks[p]).buffer("order", order[p]).buffer("large", large).buffer("chunkSlot", chunkSlot[p]);
  };
  const VkDescriptorSet largeBounds =
      largeLevels > 0 ? set("bvh_sah_large_bounds", [&](DescriptorWriter &w) { chunks(w, 0); }) : VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> largeBins, largeScatter;
  for (std::uint32_t level = 0; level < largeLevels; ++level) {
    const std::uint32_t p = level & 1u;
    largeBins.push_back(set("bvh_sah_large_bins", [&](DescriptorWriter &w) { chunks(w, level); }));
    largeScatter.push_back(set("bvh_sah_large_scatter", [&](DescriptorWriter &w) {
      chunks(w, level);
      w.buffer("nextOrder", order[p ^ 1u]).buffer("chunkState", chunkState).buffer("nextLargeOf", largeOf[p ^ 1u]);
    }));
  }
  const VkDescriptorSet sizes = set("bvh_sah_sizes", [&](DescriptorWriter &w) {
    w.buffer("control", k.setup, k.controlAt(0), controlBytes)
        .buffer("segments", k.setup, k.segmentOffset, k.segmentBytes).buffer("nodeSlots", nodeSlots)
        .buffer("nodeParent", nodeParent).buffer("subtree", subtree).buffer("arrivals", arrivals)
        .buffer("segmentNodes", segmentNodes);
  });
  const VkDescriptorSet bases = set("bvh_sah_bases", [&](DescriptorWriter &w) {
    w.buffer("control", k.setup, k.controlAt(0), controlBytes).buffer("segmentNodes", segmentNodes)
        .buffer("segmentDepth", segmentDepth).buffer("segmentBase", segmentBase)
        .buffer("error", k.setup, k.errorOffset, 4)
        .buffer("status", k.setup, k.statusOffset, sizeof(pt::BvhBuildStatus2));
  });
  const VkDescriptorSet emit = set("bvh_sah_emit", [&](DescriptorWriter &w) {
    w.buffer("control", k.setup, k.controlAt(0), controlBytes)
        .buffer("segments", k.setup, k.segmentOffset, k.segmentBytes).buffer("records", k.records)
        .buffer("instances", k.setup, k.instanceOffset, k.instanceBytes)
        .buffer("indices", scene.indexBuffer).buffer("vertices", scene.vertexBuffer)
        .buffer("finalOrder", finalOrder).buffer("nodeSlots", nodeSlots).buffer("nodeParent", nodeParent)
        .buffer("subtree", subtree).buffer("segmentNodes", segmentNodes).buffer("segmentBase", segmentBase)
        .buffer("nodes", result.nodes).buffer("triangles", result.triangles);
  });

  std::uint32_t dispatches = 0;
  uploader.runImmediate([&](VkCommandBuffer command) {
    auto run = [&](const char *entry, VkDescriptorSet s, std::uint32_t x) {
      const Kernel &kk = kernel(entry);
      vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, kk.pipeline.handle);
      kk.program->bind(command, s);
      if (x > 0) vkCmdDispatch(command, x, 1, 1);
      barrier(command);
      ++dispatches;
    };
    // The stages the LBVH reports; the SAH has no Morton codes or sort (they last 0 ms), its
    // levels are the topology, the subtree sizes and bases the fit.
    const char *const stages[kMarkCount] = {"LBVH extents", nullptr, nullptr, "SAH levels", "SAH subtree sizes",
                                            "SAH emit", nullptr};
    bool labelled = false;
    auto mark = [&](Mark m) {
      vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, timestamps, m);
      if (labelled) context.endLabel(command);
      labelled = stages[m] != nullptr;
      if (labelled) context.beginLabel(command, stages[m]);
    };
    vkCmdResetQueryPool(command, timestamps, 0, kMarkCount);
    mark(kMarkStart);
    for (const Buffer *zero : {&arrivals, &segmentNodes, &segmentDepth, &segmentStart, &lookback, &chunkState})
      vkCmdFillBuffer(command, zero->handle, 0, VK_WHOLE_SIZE, 0u);
    transferBarrier(command);
    run("bvh_lbvh_extents", extents, groups(triangleTotal, 64));
    run("bvh_lbvh_instances", instances, groups(instanceCount, 64));
    mark(kMarkExtents);
    mark(kMarkMorton);
    mark(kMarkSort);
    auto indirect = [&](const char *entry, VkDescriptorSet s, const Buffer &args, VkDeviceSize offset) {
      const Kernel &kk = kernel(entry);
      vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, kk.pipeline.handle);
      kk.program->bind(command, s);
      vkCmdDispatchIndirect(command, args.handle, offset);
      barrier(command);
      ++dispatches;
    };
    for (std::uint32_t l = 0; l < kSahLevels; ++l) {
      if (l == 0 && largeLevels > 0) indirect("bvh_sah_large_bounds", largeBounds, largeDispatch, 0);
      if (l < largeLevels) indirect("bvh_sah_large_bins", largeBins[l], largeDispatch, l * 16ull);
      indirect("bvh_sah_level", levels[l], levelArgs, l * 16ull);
      if (l < largeLevels) indirect("bvh_sah_large_scatter", largeScatter[l], largeDispatch, l * 16ull);
    }
    mark(kMarkTopology);
    run("bvh_sah_sizes", sizes, groups(internalCount, 64));
    run("bvh_sah_bases", bases, 1);
    mark(kMarkFit);
    run("bvh_sah_emit", emit, groups(std::max(internalCount, triangleTotal), 64));
    mark(kMarkEmit);
  });

  std::vector<std::uint32_t> base(segmentCount + 1u);
  const std::vector<std::uint8_t> baseBytes = uploader.readBuffer(segmentBase, base.size() * 4u);
  std::memcpy(base.data(), baseBytes.data(), baseBytes.size());
  const std::uint32_t nodes = base[segmentCount];
  const pt::BvhBuildStatus2 status = readStatus(uploader, k, "GPU binned SAH build", nodes);
  // The published nodes alone, as every other build's buffer holds them; the LBVH-sized buffer
  // the emit wrote was scratch.
  Buffer published(context, VkDeviceSize(nodes) * 4u * sizeof(pt::float4),
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   VMA_MEMORY_USAGE_AUTO, 0, "path.gpu-bvh.nodes");
  uploader.runImmediate([&](VkCommandBuffer command) {
    const VkBufferCopy region{0, 0, published.size};
    vkCmdCopyBuffer(command, result.nodes.handle, published.handle, 1, &region);
  });
  k.scratchBytes += result.nodes.size;
  result.nodes = std::move(published);
  for (std::uint32_t i = 0; i < instanceCount; ++i) result.instances[i].blasRoot = base[i];
  const std::uint32_t topNodes = instanceCount > 0 ? base[0] : nodes;
  double gpuMilliseconds = 0.0;
  result.stageMilliseconds = readStages(gpuMilliseconds);

  result.statistics.instances = instanceCount;
  result.statistics.triangles = triangleTotal;
  result.statistics.topNodes = topNodes;
  result.statistics.bottomNodes = nodes - topNodes;
  result.statistics.topDepth = status.result.y;
  result.statistics.bottomDepth = status.result.z;
  result.scratchBytes = k.scratchBytes;
  result.outputBytes = result.nodes.size + result.triangles.size;
  result.status.error = status.result.x;
  result.status.topDepth = status.result.y;
  result.status.bottomDepth = status.result.z;
  result.status.nodes = nodes;
  result.status.triangles = status.counts.y;
  result.status.instances = status.counts.z;
  result.status.dispatches = dispatches;
  const pt::LayoutCost cost = gpuBvhSahCost(context, uploader, result.nodes, nodes, result.instances, false);
  result.status.topCost = cost.top;
  result.status.bottomCost = cost.bottom;
  result.statistics.sahCost = cost.bottom;
}

// A task per subgroup, up to four to a workgroup of 128 (bvh_sah_level).
bool GpuLbvhBuilder::sahAvailable() const {
  VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  properties.pNext = &subgroup;
  vkGetPhysicalDeviceProperties2(context.physical, &properties);
  return subgroup.subgroupSize >= 32 && subgroup.subgroupSize <= 128 &&
         std::any_of(kernels.begin(), kernels.end(), [](const auto &k) { return k->entry == "bvh_sah_level"; });
}

bool GpuLbvhBuilder::clipAvailable() const {
  return std::any_of(kernels.begin(), kernels.end(), [](const auto &k) { return k->entry == "bvh_clip_emit"; });
}

// The clipping's kernels (bvh_clip.slang): the threshold sums, the counts and their scan, the
// per-instance ranges read back (the counts the build's layout needs), then the references
// written into a buffer of their total.
GpuBvhReferences GpuLbvhBuilder::clipReferences(Uploader &uploader, const Scene &scene, const TraceScene &trace,
                                                float factor) {
  if (!clipAvailable()) throw std::runtime_error("GPU split clipping needs 64-bit floats in shaders");
  const auto started = std::chrono::steady_clock::now();
  constexpr std::uint32_t kAreaBlock = 256, kScanBlock = 256;  // pt::clipReferences's, bvh_clip.slang's
  struct Pair {
    std::uint32_t x, y;
  };
  const auto instanceCount = static_cast<std::uint32_t>(trace.instances.size());
  std::vector<pt::uint4> levels;
  std::vector<Pair> areaBlocks;
  std::uint32_t triangles = 0;
  for (std::uint32_t i = 0; i < instanceCount; ++i) {
    const std::uint32_t count = scene.primitives[trace.primitives[i]].indexCount / 3u;
    const std::uint32_t blocks = groups(count, kAreaBlock);
    levels.push_back({triangles, count, static_cast<std::uint32_t>(areaBlocks.size()), blocks});
    for (std::uint32_t b = 0; b < blocks; ++b) areaBlocks.push_back({i, b * kAreaBlock});
    triangles += count;
  }
  const std::uint32_t scanBlocks = std::max(1u, groups(triangles, kScanBlock));
  pt::BvhClipControl control{};
  control.sizes = {instanceCount, triangles, static_cast<std::uint32_t>(areaBlocks.size()), scanBlocks};
  control.factor = {factor, 0.0f, 0.0f, 0.0f};
  const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  auto upload = [&](const void *data, std::size_t bytes, const char *name) {
    std::vector<std::uint8_t> padded(std::max<std::size_t>(bytes, 16), 0);
    if (bytes) std::memcpy(padded.data(), data, bytes);
    return uploader.createBuffer(padded.data(), padded.size(), storage, name);
  };
  auto scratch = [&](std::size_t bytes, const char *name, VkBufferUsageFlags extra = 0) {
    return Buffer(context, std::max<std::size_t>(bytes, 16), storage | extra, VMA_MEMORY_USAGE_AUTO, 0, name);
  };
  const Buffer controlBuffer = uploader.createBuffer(&control, sizeof(control), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                     "path.gpu-clip.control");
  const Buffer instances = upload(trace.instances.data(), trace.instances.size() * sizeof(pt::TraceInstance),
                                  "path.gpu-clip.instances");
  const Buffer levelBuffer = upload(levels.data(), levels.size() * sizeof(pt::uint4), "path.gpu-clip.levels");
  const Buffer areaBuffer = upload(areaBlocks.data(), areaBlocks.size() * sizeof(Pair), "path.gpu-clip.area-blocks");
  const Buffer blockSums = scratch(areaBlocks.size() * sizeof(double), "path.gpu-clip.block-sums");
  const Buffer factorSums = scratch(std::size_t(instanceCount) * sizeof(double), "path.gpu-clip.factor-sums");
  const Buffer inBlock = scratch(std::size_t(triangles) * 4u, "path.gpu-clip.in-block");
  const Buffer scanTotals = scratch(std::size_t(scanBlocks) * 4u, "path.gpu-clip.scan-totals");
  const Buffer scanPrefix = scratch((scanBlocks + 1ull) * 4u, "path.gpu-clip.scan-prefix");
  const Buffer ranges = scratch(std::size_t(instanceCount) * 8u, "path.gpu-clip.ranges", VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

  DescriptorPool pool(context, 16);
  auto set = [&](const char *entry, const std::function<void(DescriptorWriter &)> &write) {
    const Kernel &kk = kernel(entry);
    const VkDescriptorSet s = kk.program->allocate(pool);
    DescriptorWriter writer(context, *kk.program, s);
    write(writer);
    writer.apply();
    return s;
  };
  GpuBvhReferences result;
  auto common = [&](DescriptorWriter &w) {
    w.buffer("control", controlBuffer).buffer("instances", instances).buffer("indices", scene.indexBuffer)
        .buffer("vertexBits", scene.vertexBuffer).buffer("levels", levelBuffer).buffer("areaBlocks", areaBuffer)
        .buffer("blockSums", blockSums).buffer("factorSums", factorSums).buffer("inBlock", inBlock)
        .buffer("scanTotals", scanTotals).buffer("scanPrefix", scanPrefix).buffer("ranges", ranges);
  };
  const char *const counting[] = {"bvh_clip_area_blocks", "bvh_clip_sums", "bvh_clip_count", "bvh_clip_scan",
                                  "bvh_clip_instances"};
  const std::uint32_t grid[] = {groups(control.sizes.z, 64), groups(instanceCount, 64), scanBlocks, 1,
                                groups(instanceCount, 64)};
  VkDescriptorSet countSets[5];
  for (int i = 0; i < 5; ++i) countSets[i] = set(counting[i], common);
  auto dispatch = [&](VkCommandBuffer command, const char *entry, VkDescriptorSet s, std::uint32_t x) {
    const Kernel &kk = kernel(entry);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, kk.pipeline.handle);
    kk.program->bind(command, s);
    if (x > 0) vkCmdDispatch(command, x, 1, 1);
    barrier(command);
  };
  double gpu = 0.0;
  auto timed = [&](const std::function<void(VkCommandBuffer)> &work) {
    uploader.runImmediate([&](VkCommandBuffer command) {
      vkCmdResetQueryPool(command, timestamps, 0, 2);
      vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, timestamps, 0);
      work(command);
      vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, timestamps, 1);
    });
    std::uint64_t stamps[2]{};
    if (vkGetQueryPoolResults(context.device, timestamps, 0, 2, sizeof(stamps), stamps, sizeof(std::uint64_t),
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS)
      gpu += double(stamps[1] - stamps[0]) * context.properties.limits.timestampPeriod * 1e-6;
  };
  timed([&](VkCommandBuffer command) {
    for (int i = 0; i < 5; ++i) dispatch(command, counting[i], countSets[i], grid[i]);
  });
  std::vector<Pair> range(instanceCount);
  if (instanceCount) {
    const std::vector<std::uint8_t> bytes = uploader.readBuffer(ranges, range.size() * sizeof(Pair));
    std::memcpy(range.data(), bytes.data(), bytes.size());
  }
  std::size_t total = 0;
  for (const Pair &r : range) {
    result.counts.push_back(r.y);
    total += r.y;
  }
  result.references = Buffer(context, std::max<std::size_t>(1, total) * sizeof(pt::BvhReference),
                             storage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO, 0, "path.gpu-bvh.references");
  const VkDescriptorSet emit = set("bvh_clip_emit", [&](DescriptorWriter &w) {
    common(w);
    w.buffer("references", result.references);
  });
  timed([&](VkCommandBuffer command) { dispatch(command, "bvh_clip_emit", emit, groups(triangles, 64)); });
  result.gpuMilliseconds = gpu;
  result.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  return result;
}

// Bottom-up fit, one dispatch per tree level from the seeds queued at level 0.
std::uint32_t GpuLbvhBuilder::recordFit(VkCommandBuffer command, const Kept &k, const CommonSets &sets) const {
  const Kernel &fit = kernel("bvh_lbvh_fit");
  vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, fit.pipeline.handle);
  for (std::uint32_t i = 0; i < kFitIterations; ++i) {
    fit.program->bind(command, sets.fit[i]);
    vkCmdDispatchIndirect(command, k.setup.handle, k.headerOffset + i * sizeof(pt::uint4));
    barrier(command);
  }
  return kFitIterations;
}

// Top-down numbering, from the highest level (the tallest root) to the leaves' parents.
std::uint32_t GpuLbvhBuilder::recordNumber(VkCommandBuffer command, const Kept &k, const CommonSets &sets) const {
  const Kernel &number = kernel("bvh_lbvh_number");
  vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, number.pipeline.handle);
  for (std::uint32_t level = kFitIterations; level-- > 0;) {
    number.program->bind(command, sets.number[level]);
    vkCmdDispatchIndirect(command, k.setup.handle, k.headerOffset + level * sizeof(pt::uint4));
    barrier(command);
  }
  return kFitIterations;
}

pt::BvhBuildStatus2 GpuLbvhBuilder::readStatus(Uploader &uploader, const Kept &k, const char *what,
                                               std::uint32_t nodes) const {
  const std::vector<std::uint8_t> statusBytes = uploader.readBuffer(k.setup, sizeof(pt::BvhBuildStatus2));
  pt::BvhBuildStatus2 status{};
  std::memcpy(&status, statusBytes.data(), sizeof(status));
  if (status.result.x != pt::kBvhBuildErrorNone) {
    // A tree too deep for the traversal stack says how deep.
    const std::string depth = status.result.x == pt::kBvhBuildErrorDepth
        ? "; depth " + std::to_string(status.result.y) + " + " + std::to_string(status.result.z) + ", the stack holds " +
              std::to_string(pt::kBvhStackDeep)
        : "";
    throw std::runtime_error(std::string(what) + " failed: " + errorName(status.result.x) + " (code " +
                             std::to_string(status.result.x) + depth + ")");
  }
  if (status.counts.x != nodes || status.counts.y != k.triangleTotal || status.counts.z != k.instanceCount)
    throw std::runtime_error(std::string(what) + " published incomplete output counts");
  return status;
}

// Every stage's time from the marks (written in order by every build and update).
std::vector<double> GpuLbvhBuilder::readStages(double &total) const {
  std::vector<double> stages(kGpuBvhStageCount, 0.0);
  std::uint64_t stamps[kMarkCount]{};
  if (vkGetQueryPoolResults(context.device, timestamps, 0, kMarkCount, sizeof(stamps), stamps, sizeof(std::uint64_t),
                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS)
    return stages;
  const double period = context.properties.limits.timestampPeriod * 1e-6;  // ticks to ms
  for (std::uint32_t stage = 0; stage + 1 < kMarkCount; ++stage)
    stages[stage] = static_cast<double>(stamps[stage + 1] - stamps[stage]) * period;
  total = static_cast<double>(stamps[kMarkCount - 1] - stamps[0]) * period;
  return stages;
}

// The trace's instance rows (new transforms) with the kept build's roots and triangle offsets,
// after checking that the update keeps the build's instances and triangle counts.
std::vector<pt::TraceInstance> GpuLbvhBuilder::keptRows(const Scene &scene, const TraceScene &trace) const {
  if (!kept) throw std::runtime_error("a GPU LBVH update needs a build made refittable");
  const Kept &k = *kept;
  if (trace.instances.size() != k.instanceCount || trace.primitives.size() != k.instanceCount)
    throw std::runtime_error("a GPU LBVH update needs the built instances");
  for (std::uint32_t i = 0; i < k.instanceCount; ++i)
    if (trace.primitives[i] >= scene.primitives.size() ||
        scene.primitives[trace.primitives[i]].indexCount / 3u != k.triangleCounts[i])
      throw std::runtime_error("a GPU LBVH update needs the built triangle counts");
  std::vector<pt::TraceInstance> rows = trace.instances;
  for (std::uint32_t i = 0; i < k.instanceCount; ++i) {
    rows[i].blasRoot = k.instances[i].blasRoot;
    rows[i].triangleOffset = k.instances[i].triangleOffset;
  }
  return rows;
}

std::vector<pt::TraceInstance> GpuLbvhBuilder::updateRows(const Scene &scene, const TraceScene &trace) const {
  return keptRows(scene, trace);
}

// A refit's commands: restore what the build consumed (the error word, empty segment boxes, the
// level headers), take the new instance rows from `rows`, then extents, instance records, the
// fit seeded from the kept topology, emit and the status. `mark` stamps stage boundaries.
std::uint32_t GpuLbvhBuilder::recordRefitCommands(VkCommandBuffer command, const Kept &k, const CommonSets &common,
                                                  const Buffer &rows, const std::function<void(std::uint32_t)> &mark) const {
  std::uint32_t dispatches = 0;
  auto run = [&](const char *entry, VkDescriptorSet s, std::uint32_t x) {
    const Kernel &kk = kernel(entry);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, kk.pipeline.handle);
    kk.program->bind(command, s);
    if (x > 0) vkCmdDispatch(command, x, 1, 1);
    barrier(command);
    ++dispatches;
  };
  mark(kMarkStart);
  transferAfterCompute(command);
  const VkBufferCopy restore[] = {{k.errorOffset, k.errorOffset, 4},
                                  {k.boxOffset, k.boxOffset, k.boxBytes},
                                  {k.headerOffset, k.headerOffset, k.headerBytes}};
  vkCmdCopyBuffer(command, k.pristine.handle, k.setup.handle, static_cast<std::uint32_t>(std::size(restore)), restore);
  const VkBufferCopy instanceRegion{0, k.instanceOffset, k.instanceBytes};
  vkCmdCopyBuffer(command, rows.handle, k.setup.handle, 1, &instanceRegion);
  transferBarrier(command);
  run("bvh_lbvh_extents", common.extents, groups(k.triangleTotal, 64));
  run("bvh_lbvh_instances", common.instances, groups(k.instanceCount, 64));
  mark(kMarkExtents);
  mark(kMarkMorton);
  mark(kMarkSort);
  mark(kMarkTopology);
  run("bvh_lbvh_seed", common.seed, groups(k.internalCount, 64));
  dispatches += recordFit(command, k, common);
  mark(kMarkFit);
  run("bvh_lbvh_emit", common.emit, groups(k.internalCount + k.triangleTotal, 64));
  run("bvh_lbvh_finish", common.finish, 1);
  mark(kMarkEmit);
  return dispatches;
}

// A TLAS rebuild's commands: restore the error word, the TLAS segment's box and the level
// headers, take the new rows, then the TLAS records, their Morton codes, their sort alone
// (copied out and back into the TLAS's range of the sorted keys), topology, fit, numbering and
// emit for the TLAS's nodes, and the status.
std::uint32_t GpuLbvhBuilder::recordTopLevelCommands(VkCommandBuffer command, const Kept &k, const CommonSets &common,
                                                     const std::vector<SortSets> &sorting, const Buffer &rows,
                                                     const std::function<void(std::uint32_t)> &mark) const {
  std::uint32_t dispatches = 0;
  auto run = [&](const char *entry, VkDescriptorSet s, std::uint32_t x) {
    const Kernel &kk = kernel(entry);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, kk.pipeline.handle);
    kk.program->bind(command, s);
    if (x > 0) vkCmdDispatch(command, x, 1, 1);
    barrier(command);
    ++dispatches;
  };
  const std::uint32_t sorted = k.passes & 1u;
  const std::uint32_t tlas = k.segmentCount - 1u;
  const VkDeviceSize topFirst = VkDeviceSize(k.triangleTotal) * 4u, topBytes = std::max(1u, k.instanceCount) * 4ull;
  mark(kMarkStart);
  transferAfterCompute(command);
  const VkDeviceSize topBox = k.boxOffset + tlas * 6u * sizeof(std::uint32_t);
  const VkBufferCopy restore[] = {{k.errorOffset, k.errorOffset, 4},
                                  {topBox, topBox, 6u * sizeof(std::uint32_t)},
                                  {k.headerOffset, k.headerOffset, k.headerBytes}};
  vkCmdCopyBuffer(command, k.pristine.handle, k.setup.handle, static_cast<std::uint32_t>(std::size(restore)), restore);
  const VkBufferCopy instanceRegion{0, k.instanceOffset, k.instanceBytes};
  vkCmdCopyBuffer(command, rows.handle, k.setup.handle, 1, &instanceRegion);
  transferBarrier(command);
  run("bvh_lbvh_instances", common.instances, groups(k.instanceCount, 64));
  mark(kMarkExtents);
  run("bvh_lbvh_morton", common.morton, groups(k.instanceCount, 64));
  mark(kMarkMorton);
  if (k.instanceCount > 0) {
    const VkBufferCopy out{topFirst, 0, topBytes};
    vkCmdCopyBuffer(command, k.keysLo[0].handle, k.topLo[0].handle, 1, &out);
    vkCmdCopyBuffer(command, k.values[0].handle, k.topValues[0].handle, 1, &out);
    vkCmdFillBuffer(command, k.topHi[0].handle, 0, VK_WHOLE_SIZE, 0u);
    transferBarrier(command);
    dispatches += recordSort(command, sorting, k.topTiles, k.topBlocks);
    const VkBufferCopy back{0, topFirst, topBytes};
    vkCmdCopyBuffer(command, k.topLo[0].handle, k.keysLo[sorted].handle, 1, &back);
    vkCmdCopyBuffer(command, k.topValues[0].handle, k.values[sorted].handle, 1, &back);
    transferBarrier(command);
  }
  mark(kMarkSort);
  run("bvh_lbvh_topology", common.topology, groups(k.topNodes, 64));
  mark(kMarkTopology);
  dispatches += recordFit(command, k, common);
  mark(kMarkFit);
  dispatches += recordNumber(command, k, common);
  run("bvh_lbvh_emit", common.emit, groups(k.topNodes, 64));
  run("bvh_lbvh_finish", common.finish, 1);
  mark(kMarkEmit);
  return dispatches;
}

GpuBvhUpdateResult GpuLbvhBuilder::refit(Uploader &uploader, const Scene &scene, const TraceScene &trace,
                                         const Buffer &nodes, const Buffer &triangles) {
  const auto started = std::chrono::steady_clock::now();
  std::vector<pt::TraceInstance> rows = keptRows(scene, trace);
  Kept &k = *kept;
  GpuBvhUpdateResult result;
  result.instances = rows;
  if (rows.empty()) rows.push_back({});
  Buffer rowBuffer = uploader.createBuffer(rows.data(), rows.size() * sizeof(pt::TraceInstance),
                                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT, "path.gpu-lbvh.update-instances");
  DescriptorPool pool(context, 256);
  const CommonSets common = commonSets(pool, k, scene, nodes, triangles, 0u);
  uploader.runImmediate([&](VkCommandBuffer command) {
    const char *const labels[kMarkCount] = {"LBVH refit extents", nullptr, nullptr, "LBVH refit fit", "LBVH refit emit",
                                            nullptr, nullptr};
    bool labelled = false;  // a stage range is open
    vkCmdResetQueryPool(command, timestamps, 0, kMarkCount);
    result.dispatches = recordRefitCommands(command, k, common, rowBuffer, [&](std::uint32_t m) {
      vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, timestamps, m);
      if (labelled) context.endLabel(command);
      labelled = labels[m] != nullptr;
      if (labelled) context.beginLabel(command, labels[m]);
    });
  });
  const pt::BvhBuildStatus2 status = readStatus(uploader, k, "GPU parallel LBVH refit", k.internalCount);
  result.topDepth = status.result.y;
  result.bottomDepth = status.result.z;
  result.stageMilliseconds = readStages(result.gpuMilliseconds);
  k.instances = result.instances;
  result.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  return result;
}

GpuBvhUpdateResult GpuLbvhBuilder::rebuildTopLevel(Uploader &uploader, const Scene &scene, const TraceScene &trace,
                                                   const Buffer &nodes, const Buffer &triangles) {
  const auto started = std::chrono::steady_clock::now();
  std::vector<pt::TraceInstance> rows = keptRows(scene, trace);
  Kept &k = *kept;
  GpuBvhUpdateResult result;
  result.instances = rows;
  if (rows.empty()) rows.push_back({});
  Buffer rowBuffer = uploader.createBuffer(rows.data(), rows.size() * sizeof(pt::TraceInstance),
                                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT, "path.gpu-lbvh.update-instances");
  DescriptorPool pool(context, 256);
  const CommonSets common = commonSets(pool, k, scene, nodes, triangles, k.topSlot);
  const std::vector<SortSets> sorting = sortSets(pool, k.topLo, k.topHi, k.topValues, k.topHistogram,
                                                 k.topBlockSums, k.setup, k.controlAt(k.topSlot + 1u), 4u);
  uploader.runImmediate([&](VkCommandBuffer command) {
    const char *const labels[kMarkCount] = {"TLAS instances", "TLAS morton", "TLAS sort", "TLAS topology",
                                            "TLAS fit", "TLAS number and emit", nullptr};
    vkCmdResetQueryPool(command, timestamps, 0, kMarkCount);
    result.dispatches = recordTopLevelCommands(command, k, common, sorting, rowBuffer, [&](std::uint32_t m) {
      vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, timestamps, m);
      if (m != kMarkStart) context.endLabel(command);
      if (labels[m]) context.beginLabel(command, labels[m]);
    });
  });
  const pt::BvhBuildStatus2 status = readStatus(uploader, k, "GPU parallel LBVH TLAS rebuild", k.internalCount);
  result.topDepth = status.result.y;
  result.bottomDepth = status.result.z;
  result.stageMilliseconds = readStages(result.gpuMilliseconds);
  k.instances = result.instances;
  result.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  return result;
}

// The sets an in-frame update binds, made once for the kept build and these buffers.
const GpuLbvhBuilder::CommonSets &GpuLbvhBuilder::frameSets(const Scene &scene, const Buffer &nodes,
                                                            const Buffer &triangles, bool top) {
  Kept &k = *kept;
  const VkBuffer key[3] = {nodes.handle, triangles.handle, scene.vertexBuffer.handle};
  if (!k.framePool || std::memcmp(key, k.frameKey, sizeof(key)) != 0) {
    k.framePool = std::make_unique<DescriptorPool>(context, 512);
    k.frameRefit = commonSets(*k.framePool, k, scene, nodes, triangles, 0u);
    k.frameTop = commonSets(*k.framePool, k, scene, nodes, triangles, k.topSlot);
    k.frameSorting = sortSets(*k.framePool, k.topLo, k.topHi, k.topValues, k.topHistogram, k.topBlockSums, k.setup,
                              k.controlAt(k.topSlot + 1u), 4u);
    std::memcpy(k.frameKey, key, sizeof(key));
  }
  return top ? k.frameTop : k.frameRefit;
}

// The status of the update recorded for `slot`, copied to host-visible memory with the update.
void GpuLbvhBuilder::copyFrameStatus(VkCommandBuffer command, std::uint32_t slot) {
  Kept &k = *kept;
  if (!k.frameStatus.handle)
    k.frameStatus = Buffer(context, kFrameSlots * sizeof(pt::BvhBuildStatus2), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           VMA_MEMORY_USAGE_AUTO,
                           VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                           "path.gpu-lbvh.frame-status");
  transferAfterCompute(command);
  const VkBufferCopy region{k.statusOffset, slot * sizeof(pt::BvhBuildStatus2), sizeof(pt::BvhBuildStatus2)};
  vkCmdCopyBuffer(command, k.setup.handle, k.frameStatus.handle, 1, &region);
  k.framePending[slot] = true;
}

std::uint32_t GpuLbvhBuilder::recordRefit(VkCommandBuffer command, const Scene &scene, const Buffer &rows,
                                          const Buffer &nodes, const Buffer &triangles, std::uint32_t slot) {
  if (!kept) throw std::runtime_error("a GPU LBVH update needs a build made refittable");
  if (slot >= kFrameSlots) throw std::runtime_error("GPU LBVH frame slot out of range");
  const CommonSets &sets = frameSets(scene, nodes, triangles, false);
  context.beginLabel(command, "LBVH refit");
  const std::uint32_t dispatches = recordRefitCommands(command, *kept, sets, rows, [](std::uint32_t) {});
  context.endLabel(command);
  copyFrameStatus(command, slot);
  return dispatches;
}

std::uint32_t GpuLbvhBuilder::recordRebuildTopLevel(VkCommandBuffer command, const Scene &scene, const Buffer &rows,
                                                    const Buffer &nodes, const Buffer &triangles, std::uint32_t slot) {
  if (!kept) throw std::runtime_error("a GPU LBVH update needs a build made refittable");
  if (slot >= kFrameSlots) throw std::runtime_error("GPU LBVH frame slot out of range");
  const CommonSets &sets = frameSets(scene, nodes, triangles, true);
  context.beginLabel(command, "TLAS rebuild");
  const std::uint32_t dispatches = recordTopLevelCommands(command, *kept, sets, kept->frameSorting, rows, [](std::uint32_t) {});
  context.endLabel(command);
  copyFrameStatus(command, slot);
  return dispatches;
}

std::string GpuLbvhBuilder::frameError(std::uint32_t slot) {
  if (!kept || slot >= kFrameSlots || !kept->framePending[slot]) return {};
  Kept &k = *kept;
  k.framePending[slot] = false;
  pt::BvhBuildStatus2 status{};
  std::memcpy(&status, static_cast<const std::uint8_t *>(k.frameStatus.mapped) + slot * sizeof(status), sizeof(status));
  if (status.result.x != pt::kBvhBuildErrorNone)
    return std::string("GPU parallel LBVH update failed: ") + errorName(status.result.x) + " (code " +
           std::to_string(status.result.x) + ")";
  if (status.counts.x != k.internalCount || status.counts.y != k.triangleTotal || status.counts.z != k.instanceCount)
    return "GPU parallel LBVH update published incomplete output counts";
  return {};
}

} // namespace basalt
