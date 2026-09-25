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
static_assert(sizeof(pt::BvhLbvhControl) == 64);
static_assert(sizeof(pt::BvhSortControl) == 32);

constexpr std::uint32_t kFitIterations = pt::kBvhStack; // tree levels the fit publishes
constexpr std::uint32_t kSortTile = 1024;          // keys per sort workgroup (bvh_sort.slang)
constexpr std::uint32_t kSortMaximumBlocks = 1024; // one workgroup scans the block sums
constexpr VkDeviceSize kSlot = 256;                // the largest offset alignment Vulkan allows

const char *const kEntries[] = {"bvh_lbvh_extents",  "bvh_lbvh_instances", "bvh_lbvh_morton",
                                "bvh_sort_histogram", "bvh_sort_reduce",    "bvh_sort_blocks",
                                "bvh_sort_apply",     "bvh_sort_scatter",   "bvh_lbvh_topology",
                                "bvh_lbvh_seed",      "bvh_lbvh_fit",       "bvh_lbvh_number",
                                "bvh_lbvh_emit",      "bvh_lbvh_finish",    "bvh_ploc_init",
                                "bvh_ploc_single",    "bvh_ploc_neighbours", "bvh_ploc_scan_sums",
                                "bvh_ploc_compact",   "bvh_ploc_finish"};

// Timestamps after each stage; stage i lasts from mark i to mark i + 1.
enum Mark : std::uint32_t { kMarkStart, kMarkExtents, kMarkMorton, kMarkSort, kMarkTopology, kMarkFit, kMarkEmit,
                            kMarkCount };

std::uint32_t nodeCount(std::uint32_t leaves) { return leaves > 1 ? leaves - 1 : 1; }
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
  // The TLAS's own sort: its records' keys copied out, sorted, and copied back into its range.
  Buffer topLo[2], topHi[2], topValues[2], topHistogram, topBlockSums;
  VkDeviceSize statusOffset = 0, errorOffset = 0, controlOffset = 0, segmentOffset = 0, boxOffset = 0,
               heightOffset = 0, headerOffset = 0, levelOffset = 0, instanceOffset = 0;
  VkDeviceSize segmentBytes = 0, boxBytes = 0, heightBytes = 0, headerBytes = 0, levelBytes = 0, instanceBytes = 0;
  std::uint32_t recordCount = 0, segmentCount = 0, internalCount = 0, triangleTotal = 0, instanceCount = 0;
  std::uint32_t topNodes = 0, passes = 0, tiles = 0, blocks = 0, topTiles = 0, topBlocks = 0;
  std::uint32_t topSlot = 0;  // control slot of the TLAS part; its four sort passes follow
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
        .buffer("records", k.records).buffer("segmentBox", k.setup, k.boxOffset, k.boxBytes)
        .buffer("error", k.setup, k.errorOffset, 4).buffer("control", k.setup, k.controlAt(0), controlBytes);
  });
  sets.instances = set(kernel("bvh_lbvh_instances"), [&](DescriptorWriter &w) {
    w.buffer("segments", k.setup, k.segmentOffset, k.segmentBytes)
        .buffer("instances", k.setup, k.instanceOffset, k.instanceBytes).buffer("records", k.records)
        .buffer("segmentBox", k.setup, k.boxOffset, k.boxBytes)
        .buffer("control", k.setup, k.controlAt(0), controlBytes);
  });
  sets.morton = set(kernel("bvh_lbvh_morton"), [&](DescriptorWriter &w) {
    w.buffer("records", k.records).buffer("segmentBox", k.setup, k.boxOffset, k.boxBytes)
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
                                        bool refittable, GpuBvhTopology topology) {
  const auto started = std::chrono::steady_clock::now();
  kept.reset();  // an earlier build's scratch
  auto state = std::make_unique<Kept>();
  Kept &k = *state;
  GpuBvhBuildResult result;
  result.instances = trace.instances;
  const auto instanceCount = static_cast<std::uint32_t>(result.instances.size());

  // Segments: each instance's bottom level at the serial builder's node and triangle bases
  // (its records are its triangles, in primitive order), then the TLAS over the instances.
  std::vector<pt::BvhSegment> segments;
  segments.reserve(instanceCount + 1u);
  const std::uint32_t topNodes = nodeCount(instanceCount);
  std::uint32_t nodeOffset = topNodes, triangleOffset = 0, internalFirst = 0;
  for (std::uint32_t i = 0; i < instanceCount; ++i) {
    const std::uint32_t primitiveIndex = trace.primitives[i];
    if (primitiveIndex >= scene.primitives.size())
      throw std::runtime_error("GPU BVH build received an invalid primitive mapping");
    const std::uint32_t triangles = scene.primitives[primitiveIndex].indexCount / 3u;
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
  const std::uint32_t controlSlots = k.topSlot + 5u;
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
  control.geometry = {scene.indexCount, scene.vertexCount, pt::kBvhStack, pt::kBvhBuildLayoutVersion};
  control.range = {instanceCount, triangleTotal, 0u, 0u};
  control.part = {0u, 0u, internalCount, triangleTotal};
  for (std::uint32_t i = 0; i < kFitIterations; ++i) {
    control.range.z = i;
    std::memcpy(setup.data() + k.controlAt(i), &control, sizeof(control));
  }
  control.range.z = 0;
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

  k.scratchBytes = setupBytes + (refittable ? setupBytes : 0);
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
  const bool ploc = topology == GpuBvhTopology::Ploc;
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
  Buffer clusters, clusterBoxes, segmentNext, neighbour, plocKept, plocSums, plocSizes;
  if (ploc) {
    clusters = scratch(2u * clusterHalf, "path.gpu-ploc.clusters");
    clusterBoxes = scratch(2u * boxHalf, "path.gpu-ploc.cluster-boxes");
    segmentNext = scratch(segmentCount * 4ull, "path.gpu-ploc.segment-next");
    neighbour = scratch(records * 4ull, "path.gpu-ploc.neighbours");
    plocKept = scratch(records * 4ull, "path.gpu-ploc.kept");
    plocSums = scratch(groups(records, kPlocGroup) * 4ull, "path.gpu-ploc.block-sums");
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
  result.nodes = Buffer(context, VkDeviceSize(internalCount) * 4u * sizeof(pt::float4),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VMA_MEMORY_USAGE_AUTO, 0, "path.gpu-bvh.nodes");
  result.triangles = Buffer(context, VkDeviceSize(std::max(1u, triangleTotal)) * 3u * sizeof(pt::float4),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VMA_MEMORY_USAGE_AUTO, 0, "path.gpu-bvh.triangles");

  DescriptorPool pool(context, 256);
  const CommonSets common = commonSets(pool, k, scene, result.nodes, result.triangles, 0u);
  const std::vector<SortSets> sorting = sortSets(pool, k.keysLo, k.keysHi, k.values, k.histogram, k.blockSums,
                                                 k.setup, k.controlAt(kFitIterations), passes);
  // PLOC's sets: one per parity for the iterations (clusters and sizes read from [p], written
  // to [p ^ 1]).
  struct PlocSets {
    VkDescriptorSet init = VK_NULL_HANDLE, single = VK_NULL_HANDLE, finish = VK_NULL_HANDLE;
    VkDescriptorSet neighbours[2]{}, scanSums[2]{}, compact[2]{};
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
    const char *const stages[kMarkCount] = {"LBVH extents", "LBVH morton", "LBVH sort",
                                            ploc ? "PLOC topology" : "LBVH topology", "LBVH fit",
                                            "LBVH number and emit", nullptr};
    auto mark = [&](Mark m) {
      vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, timestamps, m);
      if (m != kMarkStart) context.endLabel(command);
      if (stages[m]) context.beginLabel(command, stages[m]);
    };
    vkCmdResetQueryPool(command, timestamps, 0, kMarkCount);
    mark(kMarkStart);
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
      transferBarrier(command);
      run("bvh_ploc_init", plocSets.init, groups(records, kPlocGroup));
      run("bvh_ploc_single", plocSets.single, groups(segmentCount, kPlocGroup));
      auto indirect = [&](const char *entry, VkDescriptorSet s, VkDeviceSize offset) {
        const Kernel &kk = kernel(entry);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, kk.pipeline.handle);
        kk.program->bind(command, s);
        vkCmdDispatchIndirect(command, plocSizes.handle, offset);
        barrier(command);
        ++dispatches;
      };
      for (std::uint32_t iteration = 0; iteration < plocWideIterations; ++iteration) {
        const std::uint32_t p = iteration & 1u;
        indirect("bvh_ploc_neighbours", plocSets.neighbours[p], p * kSlot);
        run("bvh_ploc_scan_sums", plocSets.scanSums[p], 1);
        indirect("bvh_ploc_compact", plocSets.compact[p], p * kSlot);
      }
      run("bvh_ploc_finish", plocSets.finish, 1);
      run("bvh_lbvh_seed", common.seed, groups(internalCount, 64));
    } else {
      run("bvh_lbvh_topology", common.topology, groups(internalCount, 64));
    }
    mark(kMarkTopology);
    dispatches += recordFit(command, k, common);
    mark(kMarkFit);
    dispatches += recordNumber(command, k, common);
    run("bvh_lbvh_emit", common.emit, groups(internalCount + triangleTotal, 64));
    run("bvh_lbvh_finish", common.finish, 1);
    mark(kMarkEmit);
  });

  if (ploc) {
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
  const pt::BvhBuildStatus2 status = readStatus(uploader, k, ploc ? "GPU PLOC build" : "GPU parallel LBVH build");
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

pt::BvhBuildStatus2 GpuLbvhBuilder::readStatus(Uploader &uploader, const Kept &k, const char *what) const {
  const std::vector<std::uint8_t> statusBytes = uploader.readBuffer(k.setup, sizeof(pt::BvhBuildStatus2));
  pt::BvhBuildStatus2 status{};
  std::memcpy(&status, statusBytes.data(), sizeof(status));
  if (status.result.x != pt::kBvhBuildErrorNone)
    throw std::runtime_error(std::string(what) + " failed: " + errorName(status.result.x) + " (code " +
                             std::to_string(status.result.x) + ")");
  if (status.counts.x != k.internalCount || status.counts.y != k.triangleTotal || status.counts.z != k.instanceCount)
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
  const pt::BvhBuildStatus2 status = readStatus(uploader, k, "GPU parallel LBVH refit");
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
  const pt::BvhBuildStatus2 status = readStatus(uploader, k, "GPU parallel LBVH TLAS rebuild");
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
