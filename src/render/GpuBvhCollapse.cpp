#include "render/GpuBvhBuilder.h"

#include "core/Log.h"
#include "gpu/Descriptors.h"
#include "gpu/Pipeline.h"
#include "gpu/Shader.h"
#include "gpu/Uploader.h"
#include "pt/Shared.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace basalt {
namespace {

static_assert(sizeof(pt::BvhCollapseControl) == 32);
static_assert(sizeof(pt::BvhCollapseStatus) == 16);

constexpr std::uint32_t kLevels = pt::kBvhStack;  // bvh_collapse.slang's kCollapseLevels
constexpr VkDeviceSize kSlot = 256;    // the largest offset alignment Vulkan allows

const char *const kEntries[] = {"bvh_collapse_gather", "bvh_collapse_size", "bvh_collapse_roots",
                                "bvh_collapse_emit", "bvh_collapse_emit_args", "bvh_collapse_rows"};

enum Mark : std::uint32_t { kMarkStart, kMarkGathered, kMarkSized, kMarkShaped, kMarkEmitStart, kMarkEmitted, kMarkCount };

VkDeviceSize aligned(VkDeviceSize bytes) { return (bytes + kSlot - 1) / kSlot * kSlot; }

// Orders earlier compute and transfer writes before later compute reads and writes, indirect
// dispatch arguments and transfer reads.
void barrier(VkCommandBuffer command) {
  VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  memory.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  memory.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
  memory.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  memory.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                         VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.memoryBarrierCount = 1;
  dependency.pMemoryBarriers = &memory;
  vkCmdPipelineBarrier2(command, &dependency);
}

// The host-initialised buffer: status, level queue headers (level 0: the TLAS root) and the
// bottom levels' root numbers (read back together), error word, per-level controls, level
// bases, the instances' binary roots, their root tasks (none yet) and the emit's indirect
// dispatches (in-frame collapses). Sections at 256-byte offsets.
struct Layout {
  VkDeviceSize status = 0, header = 0, headerBytes = 0, rootNumber = 0, rootBytes = 0, error = 0, control = 0,
               level = 0, levelBytes = 0, instanceRoot = 0, rootTask = 0, emitHeader = 0, bytes = 0;
  VkDeviceSize controlAt(std::uint32_t depth) const { return control + depth * kSlot; }
};

Layout layoutFor(std::uint32_t instanceCount) {
  Layout l;
  l.header = kSlot;
  l.headerBytes = (kLevels + 1u) * sizeof(pt::uint4);
  l.rootNumber = aligned(l.header + l.headerBytes);
  l.rootBytes = VkDeviceSize(std::max(1u, instanceCount)) * sizeof(std::uint32_t);
  l.error = aligned(l.rootNumber + l.rootBytes);
  l.control = l.error + kSlot;
  l.level = aligned(l.control + kLevels * kSlot);
  l.levelBytes = (kLevels + 1u) * sizeof(std::uint32_t);
  l.instanceRoot = aligned(l.level + l.levelBytes);
  l.rootTask = aligned(l.instanceRoot + l.rootBytes);
  l.emitHeader = aligned(l.rootTask + l.rootBytes);
  l.bytes = l.emitHeader + l.headerBytes;
  return l;
}

std::vector<std::uint8_t> setupBytes(const Layout &l, std::uint32_t capacity, std::uint32_t binaryNodeCount,
                                     std::uint32_t width, std::uint32_t levels,
                                     const std::vector<pt::TraceInstance> &instances) {
  std::vector<std::uint8_t> setup(l.bytes, 0);
  pt::BvhCollapseControl control{};
  control.sizes = {capacity, binaryNodeCount, width, static_cast<std::uint32_t>(instances.size())};
  for (std::uint32_t level = 0; level < kLevels; ++level) {
    control.level = {level, levels, 0u, 0u};
    std::memcpy(setup.data() + l.controlAt(level), &control, sizeof(control));
  }
  for (std::uint32_t level = 0; level <= kLevels; ++level) {
    const pt::uint4 header = level == 0 ? pt::uint4{1u, 1u, 1u, 1u} : pt::uint4{0u, 1u, 1u, 0u};
    std::memcpy(setup.data() + l.header + level * sizeof(pt::uint4), &header, sizeof(header));
  }
  for (std::size_t i = 0; i < instances.size(); ++i)
    std::memcpy(setup.data() + l.instanceRoot + i * sizeof(std::uint32_t), &instances[i].blasRoot,
                sizeof(std::uint32_t));
  std::memset(setup.data() + l.rootTask, 0xFF, l.rootBytes);
  return setup;
}

// Every wide node starts at a distinct binary node when each instance has its own bottom level,
// as every builder publishes; the binary node count then bounds the tasks.
void checkRoots(std::uint32_t binaryNodeCount, const std::vector<pt::TraceInstance> &instances) {
  if (binaryNodeCount == 0u || binaryNodeCount > 0x7FFFFFFFu)
    throw std::runtime_error("the GPU wide collapse needs a binary tree of 1 to 2^31 - 1 nodes");
  std::unordered_set<std::uint32_t> roots;
  for (const pt::TraceInstance &instance : instances)
    if (instance.blasRoot >= binaryNodeCount || !roots.insert(instance.blasRoot).second)
      throw std::runtime_error("the GPU wide collapse needs one bottom level per instance inside the tree");
}

// Orders earlier compute and transfer work before transfer writes and reads.
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
  case pt::kBvhBuildErrorCapacity: return "malformed binary tree or task capacity";
  case pt::kBvhBuildErrorDepth: return "wide tree deeper than the traversal stack";
  default: return "unknown";
  }
}

} // namespace

struct GpuBvhCollapser::Kernel {
  std::string entry;
  std::unique_ptr<Program> program;
  Pipeline pipeline;
};

// A collapse's tasks, kept for reemit().
struct GpuBvhCollapser::Kept {
  Buffer setup, taskNode, taskSlots, taskChild, taskSize, taskNumber;
  VkDeviceSize controlOffset = 0, headerOffset = 0, headerBytes = 0, levelOffset = 0, levelBytes = 0;
  std::vector<std::uint32_t> levelTasks;  // tasks per level
  std::uint32_t nodeCount = 0, maximumStack = 0;
  std::vector<pt::TraceInstance> instances;
};

GpuBvhCollapser::GpuBvhCollapser(const Context &ctx) : context(ctx) {
  for (const char *entry : kEntries) {
    auto k = std::make_unique<Kernel>();
    k->entry = entry;
    k->program = std::make_unique<Program>(context, entry);
    k->pipeline = Pipeline(context, *k->program, std::string("GPU wide collapse ") + entry);
    kernels.push_back(std::move(k));
  }
  VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  info.queryCount = kMarkCount;
  check(vkCreateQueryPool(context.device, &info, nullptr, &timestamps), "vkCreateQueryPool (wide collapse)");
}


const GpuBvhCollapser::Kernel &GpuBvhCollapser::kernel(const char *entry) const {
  for (const auto &k : kernels)
    if (k->entry == entry) return *k;
  throw std::runtime_error(std::string("wide collapse kernel ") + entry + " is missing");
}

GpuWideCollapseResult GpuBvhCollapser::collapse(Uploader &uploader, const Buffer &binaryNodes,
                                                std::uint32_t binaryNodeCount,
                                                const std::vector<pt::TraceInstance> &instances, std::uint32_t width,
                                                std::uint32_t binaryDepth, bool keep) {
  const auto started = std::chrono::steady_clock::now();
  kept.reset();
  if (width != 4u && width != 8u) throw std::runtime_error("wide BVH width must be four or eight");
  // Each wide level descends at least one binary level, and a TLAS leaf's bottom level starts
  // one level below it: the wide tree has at most as many levels as the TLAS and the deepest
  // bottom level together.
  const std::uint32_t levels = binaryDepth == 0u ? kLevels : std::min(binaryDepth, kLevels);
  checkRoots(binaryNodeCount, instances);
  const auto instanceCount = static_cast<std::uint32_t>(instances.size());
  const std::uint32_t capacity = binaryNodeCount;
  const Layout layout = layoutFor(instanceCount);
  const VkDeviceSize statusOffset = layout.status, headerOffset = layout.header, headerBytes = layout.headerBytes;
  const VkDeviceSize rootNumberOffset = layout.rootNumber, rootBytes = layout.rootBytes, errorOffset = layout.error;
  const VkDeviceSize controlOffset = layout.control, levelOffset = layout.level, levelBytes = layout.levelBytes;
  const VkDeviceSize instanceRootOffset = layout.instanceRoot, rootTaskOffset = layout.rootTask;
  const VkDeviceSize setupBytes = layout.bytes;
  const std::vector<std::uint8_t> setup = ::basalt::setupBytes(layout, capacity, binaryNodeCount, width, levels, instances);
  pt::BvhCollapseControl control{};
  Buffer setupBuffer = uploader.createBuffer(setup.data(), setupBytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      "path.wide-collapse.setup");

  GpuWideCollapseResult result;
  result.scratchBytes = setupBytes;
  auto scratch = [&](VkDeviceSize bytes, const char *name, VkBufferUsageFlags extra = 0) {
    result.scratchBytes += bytes;
    return Buffer(context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | extra, VMA_MEMORY_USAGE_AUTO, 0, name);
  };
  const VkDeviceSize tasks = capacity;
  Buffer taskNode = scratch(tasks * 4u, "path.wide-collapse.task-node", VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  Buffer taskSlots = scratch(tasks * 32u, "path.wide-collapse.task-slots");
  Buffer taskChild = scratch(tasks * 32u, "path.wide-collapse.task-child");
  Buffer taskSize = scratch(tasks * 4u, "path.wide-collapse.task-size");
  Buffer taskStack = scratch(tasks * 4u, "path.wide-collapse.task-stack");
  Buffer taskNumber = scratch(tasks * 4u, "path.wide-collapse.task-number");

  DescriptorPool pool(context, 256);
  auto set = [&](const Kernel &k, const std::function<void(DescriptorWriter &)> &write) {
    const VkDescriptorSet s = k.program->allocate(pool);
    DescriptorWriter writer(context, *k.program, s);
    write(writer);
    writer.apply();
    return s;
  };
  auto levelControl = [&](std::uint32_t level) { return controlOffset + level * kSlot; };
  const Kernel &gather = kernel("bvh_collapse_gather"), &size = kernel("bvh_collapse_size"),
               &rootsKernel = kernel("bvh_collapse_roots"), &emit = kernel("bvh_collapse_emit");
  std::vector<VkDescriptorSet> gatherSets, sizeSets;
  for (std::uint32_t level = 0; level < kLevels; ++level) {
    gatherSets.push_back(set(gather, [&](DescriptorWriter &w) {
      w.buffer("binary", binaryNodes).buffer("instanceRoots", setupBuffer, instanceRootOffset, rootBytes)
          .buffer("taskNode", taskNode).buffer("taskSlots", taskSlots).buffer("taskChild", taskChild)
          .buffer("rootTask", setupBuffer, rootTaskOffset, rootBytes)
          .buffer("levelBase", setupBuffer, levelOffset, levelBytes)
          .buffer("headers", setupBuffer, headerOffset, headerBytes).buffer("error", setupBuffer, errorOffset, 4)
          .buffer("control", setupBuffer, levelControl(level), sizeof(control));
    }));
    sizeSets.push_back(set(size, [&](DescriptorWriter &w) {
      w.buffer("binary", binaryNodes).buffer("taskNode", taskNode).buffer("taskSlots", taskSlots)
          .buffer("taskChild", taskChild).buffer("taskSize", taskSize).buffer("taskStack", taskStack)
          .buffer("levelBase", setupBuffer, levelOffset, levelBytes)
          .buffer("headers", setupBuffer, headerOffset, headerBytes)
          .buffer("control", setupBuffer, levelControl(level), sizeof(control));
    }));
  }
  const VkDescriptorSet rootsSet = set(rootsKernel, [&](DescriptorWriter &w) {
    w.buffer("taskSize", taskSize).buffer("taskStack", taskStack)
        .buffer("rootTask", setupBuffer, rootTaskOffset, rootBytes).buffer("taskNumber", taskNumber)
        .buffer("rootNumber", setupBuffer, rootNumberOffset, rootBytes)
        .buffer("headers", setupBuffer, headerOffset, headerBytes).buffer("error", setupBuffer, errorOffset, 4)
        .buffer("status", setupBuffer, statusOffset, sizeof(pt::BvhCollapseStatus))
        .buffer("control", setupBuffer, levelControl(0), sizeof(control));
  });

  auto indirectLevel = [&](VkCommandBuffer command, const Kernel &k, VkDescriptorSet s, std::uint32_t level) {
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline.handle);
    k.program->bind(command, s);
    vkCmdDispatchIndirect(command, setupBuffer.handle, headerOffset + level * sizeof(pt::uint4));
    barrier(command);
    ++result.dispatches;
  };
  // A timestamp at each stage boundary, and the stage as a named range for GPU profilers.
  auto mark = [&](VkCommandBuffer command, Mark m) {
    vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, timestamps, m);
    const char *const stages[kMarkCount] = {"collapse gather", "collapse size", "collapse roots", nullptr,
                                            "collapse emit", nullptr};
    if (m != kMarkStart && m != kMarkEmitStart) context.endLabel(command);
    if (stages[m]) context.beginLabel(command, stages[m]);
  };
  // Shape: the frontiers top-down, the subtree sizes bottom-up, then the roots' numbers.
  uploader.runImmediate([&](VkCommandBuffer command) {
    vkCmdResetQueryPool(command, timestamps, 0, kMarkCount);
    mark(command, kMarkStart);
    vkCmdFillBuffer(command, taskNode.handle, 0, sizeof(std::uint32_t), 0u);  // task 0: the TLAS root
    barrier(command);
    for (std::uint32_t level = 0; level < levels; ++level) indirectLevel(command, gather, gatherSets[level], level);
    mark(command, kMarkGathered);
    for (std::uint32_t level = levels; level-- > 0;) indirectLevel(command, size, sizeSets[level], level);
    mark(command, kMarkSized);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, rootsKernel.pipeline.handle);
    rootsKernel.program->bind(command, rootsSet);
    vkCmdDispatch(command, 1, 1, 1);
    barrier(command);
    ++result.dispatches;
    mark(command, kMarkShaped);
  });

  const std::vector<std::uint8_t> head = uploader.readBuffer(setupBuffer, rootNumberOffset + rootBytes);
  std::vector<pt::uint4> headers(kLevels + 1u);
  std::memcpy(headers.data(), head.data() + headerOffset, headerBytes);
  pt::BvhCollapseStatus status{};
  std::memcpy(&status, head.data() + statusOffset, sizeof(status));
  if (status.result.x != pt::kBvhBuildErrorNone)
    throw std::runtime_error(std::string("GPU wide collapse failed: ") + errorName(status.result.x) + " (code " +
                             std::to_string(status.result.x) + "; traversal stack bound " +
                             std::to_string(status.result.w) + " of " + std::to_string(pt::kWideStackDeep) + ", " +
                             std::to_string(status.result.y) + " levels)");
  result.nodeCount = status.result.z;
  result.levels = status.result.y;
  result.maximumStack = status.result.w;
  if (result.nodeCount == 0u || result.nodeCount > capacity)
    throw std::runtime_error("GPU wide collapse published an impossible node count");
  result.instances = instances;
  for (std::uint32_t i = 0; i < instanceCount; ++i)
    std::memcpy(&result.instances[i].blasRoot, head.data() + rootNumberOffset + i * sizeof(std::uint32_t),
                sizeof(std::uint32_t));

  // Emit: every node at its number, top-down, eight threads per task (one per slot).
  result.nodes = Buffer(context, VkDeviceSize(result.nodeCount) * sizeof(pt::QuantizedWideNode),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO,
                        0, "path.gpu-bvh.wide-nodes");
  std::vector<VkDescriptorSet> emitSets;
  for (std::uint32_t level = 0; level < result.levels; ++level)
    emitSets.push_back(set(emit, [&](DescriptorWriter &w) {
      w.buffer("binary", binaryNodes).buffer("taskNode", taskNode).buffer("taskSlots", taskSlots)
          .buffer("taskChild", taskChild).buffer("taskSize", taskSize).buffer("taskNumber", taskNumber)
          .buffer("levelBase", setupBuffer, levelOffset, levelBytes)
          .buffer("headers", setupBuffer, headerOffset, headerBytes).buffer("nodes", result.nodes)
          .buffer("control", setupBuffer, levelControl(level), sizeof(control));
    }));
  uploader.runImmediate([&](VkCommandBuffer command) {
    mark(command, kMarkEmitStart);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, emit.pipeline.handle);
    for (std::uint32_t level = 0; level < result.levels; ++level) {
      emit.program->bind(command, emitSets[level]);
      vkCmdDispatch(command, (headers[level].w * 8u + 63u) / 64u, 1, 1);
      barrier(command);
      ++result.dispatches;
    }
    mark(command, kMarkEmitted);
  });

  std::uint64_t stamps[kMarkCount]{};
  if (vkGetQueryPoolResults(context.device, timestamps, 0, kMarkCount, sizeof(stamps), stamps, sizeof(std::uint64_t),
                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
    const double period = context.properties.limits.timestampPeriod * 1e-6;  // ticks to ms
    result.gatherMilliseconds = static_cast<double>(stamps[kMarkGathered] - stamps[kMarkStart]) * period;
    result.sizeMilliseconds = static_cast<double>(stamps[kMarkShaped] - stamps[kMarkGathered]) * period;
    result.emitMilliseconds = static_cast<double>(stamps[kMarkEmitted] - stamps[kMarkEmitStart]) * period;
    result.gpuMilliseconds = result.gatherMilliseconds + result.sizeMilliseconds + result.emitMilliseconds;
  }
  result.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  if (keep) {
    kept = std::make_unique<Kept>();
    kept->setup = std::move(setupBuffer);
    kept->taskNode = std::move(taskNode);
    kept->taskSlots = std::move(taskSlots);
    kept->taskChild = std::move(taskChild);
    kept->taskSize = std::move(taskSize);
    kept->taskNumber = std::move(taskNumber);
    kept->controlOffset = controlOffset;
    kept->headerOffset = headerOffset;
    kept->headerBytes = headerBytes;
    kept->levelOffset = levelOffset;
    kept->levelBytes = levelBytes;
    for (std::uint32_t level = 0; level < result.levels; ++level) kept->levelTasks.push_back(headers[level].w);
    kept->nodeCount = result.nodeCount;
    kept->maximumStack = result.maximumStack;
    kept->instances = result.instances;
  }
  return result;
}

GpuWideCollapseResult GpuBvhCollapser::reemit(Uploader &uploader, const Buffer &binaryNodes, const Buffer &wideNodes) {
  const auto started = std::chrono::steady_clock::now();
  if (!kept) throw std::runtime_error("a wide re-emit needs a kept collapse");
  const Kept &k = *kept;
  if (wideNodes.size < VkDeviceSize(k.nodeCount) * sizeof(pt::QuantizedWideNode))
    throw std::runtime_error("a wide re-emit needs the kept collapse's node buffer");
  const Kernel &emit = kernel("bvh_collapse_emit");
  DescriptorPool pool(context, 128);
  std::vector<VkDescriptorSet> emitSets;
  for (std::uint32_t level = 0; level < k.levelTasks.size(); ++level) {
    const VkDescriptorSet s = emit.program->allocate(pool);
    DescriptorWriter(context, *emit.program, s)
        .buffer("binary", binaryNodes).buffer("taskNode", k.taskNode).buffer("taskSlots", k.taskSlots)
        .buffer("taskChild", k.taskChild).buffer("taskSize", k.taskSize).buffer("taskNumber", k.taskNumber)
        .buffer("levelBase", k.setup, k.levelOffset, k.levelBytes)
        .buffer("headers", k.setup, k.headerOffset, k.headerBytes).buffer("nodes", wideNodes)
        .buffer("control", k.setup, k.controlOffset + level * kSlot, sizeof(pt::BvhCollapseControl))
        .apply();
    emitSets.push_back(s);
  }
  GpuWideCollapseResult result;
  uploader.runImmediate([&](VkCommandBuffer command) {
    vkCmdResetQueryPool(command, timestamps, 0, kMarkCount);
    vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, timestamps, kMarkEmitStart);
    context.beginLabel(command, "collapse re-emit");
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, emit.pipeline.handle);
    for (std::uint32_t level = 0; level < k.levelTasks.size(); ++level) {
      emit.program->bind(command, emitSets[level]);
      vkCmdDispatch(command, (k.levelTasks[level] * 8u + 63u) / 64u, 1, 1);
      barrier(command);
      ++result.dispatches;
    }
    context.endLabel(command);
    vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, timestamps, kMarkEmitted);
  });
  std::uint64_t stamps[2]{};
  if (vkGetQueryPoolResults(context.device, timestamps, kMarkEmitStart, 2, sizeof(stamps), stamps, sizeof(std::uint64_t),
                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS)
    result.emitMilliseconds = result.gpuMilliseconds =
        static_cast<double>(stamps[1] - stamps[0]) * context.properties.limits.timestampPeriod * 1e-6;
  result.nodeCount = k.nodeCount;
  result.levels = static_cast<std::uint32_t>(k.levelTasks.size());
  result.maximumStack = k.maximumStack;
  result.instances = k.instances;
  result.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  return result;
}

// In-frame collapses: the scratch, the output (as many nodes as the binary tree, the most a
// collapse can emit) and the descriptor sets, made once; everything a collapse sizes from a
// readback in collapse() it reads on the GPU here.
struct GpuBvhCollapser::Frames {
  Layout layout;
  std::uint32_t capacity = 0, instanceCount = 0;
  Buffer setup, pristine, taskNode, taskSlots, taskChild, taskSize, taskStack, taskNumber, status;
  std::unique_ptr<DescriptorPool> pool;
  std::vector<VkDescriptorSet> gather, size, emit;
  VkDescriptorSet roots = VK_NULL_HANDLE, emitArgs = VK_NULL_HANDLE, rows = VK_NULL_HANDLE;
  VkBuffer rowsKey[2]{};
  bool pending[kFrameSlots]{};
};

void GpuBvhCollapser::prepareFrames(Uploader &uploader, const Buffer &binaryNodes, std::uint32_t binaryNodeCount,
                                    const std::vector<pt::TraceInstance> &instances, std::uint32_t width,
                                    const Buffer &output) {
  if (width != 4u && width != 8u) throw std::runtime_error("wide BVH width must be four or eight");
  checkRoots(binaryNodeCount, instances);
  if (output.size < VkDeviceSize(binaryNodeCount) * sizeof(pt::QuantizedWideNode))
    throw std::runtime_error("an in-frame wide collapse needs an output of as many nodes as the binary tree");
  frames = std::make_unique<Frames>();
  Frames &f = *frames;
  f.capacity = binaryNodeCount;
  f.instanceCount = static_cast<std::uint32_t>(instances.size());
  f.layout = layoutFor(f.instanceCount);
  const Layout &l = f.layout;
  const std::vector<std::uint8_t> setup = setupBytes(l, f.capacity, binaryNodeCount, width, kLevels, instances);
  f.setup = uploader.createBuffer(setup.data(), l.bytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      "path.wide-collapse.frame-setup");
  f.pristine = uploader.createBuffer(setup.data(), l.bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                     "path.wide-collapse.frame-pristine");
  auto scratch = [&](VkDeviceSize bytes, const char *name, VkBufferUsageFlags extra = 0) {
    return Buffer(context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | extra, VMA_MEMORY_USAGE_AUTO, 0, name);
  };
  const VkDeviceSize tasks = f.capacity;
  f.taskNode = scratch(tasks * 4u, "path.wide-collapse.frame-task-node", VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  f.taskSlots = scratch(tasks * 32u, "path.wide-collapse.frame-task-slots");
  f.taskChild = scratch(tasks * 32u, "path.wide-collapse.frame-task-child");
  f.taskSize = scratch(tasks * 4u, "path.wide-collapse.frame-task-size");
  f.taskStack = scratch(tasks * 4u, "path.wide-collapse.frame-task-stack");
  f.taskNumber = scratch(tasks * 4u, "path.wide-collapse.frame-task-number");
  f.status = Buffer(context, kFrameSlots * sizeof(pt::BvhCollapseStatus), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                    "path.wide-collapse.frame-status");

  f.pool = std::make_unique<DescriptorPool>(context, 256);
  auto set = [&](const char *entry, const std::function<void(DescriptorWriter &)> &write) {
    const Kernel &k = kernel(entry);
    const VkDescriptorSet s = k.program->allocate(*f.pool);
    DescriptorWriter writer(context, *k.program, s);
    write(writer);
    writer.apply();
    return s;
  };
  const VkDeviceSize controlBytes = sizeof(pt::BvhCollapseControl);
  for (std::uint32_t level = 0; level < kLevels; ++level) {
    f.gather.push_back(set("bvh_collapse_gather", [&](DescriptorWriter &w) {
      w.buffer("binary", binaryNodes).buffer("instanceRoots", f.setup, l.instanceRoot, l.rootBytes)
          .buffer("taskNode", f.taskNode).buffer("taskSlots", f.taskSlots).buffer("taskChild", f.taskChild)
          .buffer("rootTask", f.setup, l.rootTask, l.rootBytes).buffer("levelBase", f.setup, l.level, l.levelBytes)
          .buffer("headers", f.setup, l.header, l.headerBytes).buffer("error", f.setup, l.error, 4)
          .buffer("control", f.setup, l.controlAt(level), controlBytes);
    }));
    f.size.push_back(set("bvh_collapse_size", [&](DescriptorWriter &w) {
      w.buffer("binary", binaryNodes).buffer("taskNode", f.taskNode).buffer("taskSlots", f.taskSlots)
          .buffer("taskChild", f.taskChild).buffer("taskSize", f.taskSize).buffer("taskStack", f.taskStack)
          .buffer("levelBase", f.setup, l.level, l.levelBytes).buffer("headers", f.setup, l.header, l.headerBytes)
          .buffer("control", f.setup, l.controlAt(level), controlBytes);
    }));
    f.emit.push_back(set("bvh_collapse_emit", [&](DescriptorWriter &w) {
      w.buffer("binary", binaryNodes).buffer("taskNode", f.taskNode).buffer("taskSlots", f.taskSlots)
          .buffer("taskChild", f.taskChild).buffer("taskSize", f.taskSize).buffer("taskNumber", f.taskNumber)
          .buffer("levelBase", f.setup, l.level, l.levelBytes).buffer("headers", f.setup, l.header, l.headerBytes)
          .buffer("nodes", output).buffer("control", f.setup, l.controlAt(level), controlBytes);
    }));
  }
  f.roots = set("bvh_collapse_roots", [&](DescriptorWriter &w) {
    w.buffer("taskSize", f.taskSize).buffer("taskStack", f.taskStack).buffer("rootTask", f.setup, l.rootTask, l.rootBytes)
        .buffer("taskNumber", f.taskNumber).buffer("rootNumber", f.setup, l.rootNumber, l.rootBytes)
        .buffer("headers", f.setup, l.header, l.headerBytes).buffer("error", f.setup, l.error, 4)
        .buffer("status", f.setup, l.status, sizeof(pt::BvhCollapseStatus))
        .buffer("control", f.setup, l.controlAt(0), controlBytes);
  });
  f.emitArgs = set("bvh_collapse_emit_args", [&](DescriptorWriter &w) {
    w.buffer("headers", f.setup, l.header, l.headerBytes).buffer("emitHeaders", f.setup, l.emitHeader, l.headerBytes);
  });
}

// The traced rows: `input` (the binary tree's roots, current transforms) with the wide roots.
void GpuBvhCollapser::recordRows(VkCommandBuffer command, const Buffer &input, const Buffer &output) {
  Frames &f = *frames;
  const VkBuffer key[2] = {input.handle, output.handle};
  if (!f.rows || std::memcmp(key, f.rowsKey, sizeof(key)) != 0) {
    const Kernel &k = kernel("bvh_collapse_rows");
    f.rows = k.program->allocate(*f.pool);
    DescriptorWriter(context, *k.program, f.rows)
        .buffer("input", input).buffer("output", output)
        .buffer("rootNumber", f.setup, f.layout.rootNumber, f.layout.rootBytes)
        .buffer("control", f.setup, f.layout.controlAt(0), sizeof(pt::BvhCollapseControl))
        .apply();
    std::memcpy(f.rowsKey, key, sizeof(key));
  }
  const Kernel &k = kernel("bvh_collapse_rows");
  vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline.handle);
  k.program->bind(command, f.rows);
  vkCmdDispatch(command, (std::max(1u, f.instanceCount) + 63u) / 64u, 1, 1);
  barrier(command);
}

void GpuBvhCollapser::recordEmit(VkCommandBuffer command) {
  Frames &f = *frames;
  const Kernel &emit = kernel("bvh_collapse_emit");
  vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, emit.pipeline.handle);
  for (std::uint32_t level = 0; level < kLevels; ++level) {
    emit.program->bind(command, f.emit[level]);
    vkCmdDispatchIndirect(command, f.setup.handle, f.layout.emitHeader + level * sizeof(pt::uint4));
    barrier(command);
  }
}

void GpuBvhCollapser::copyFrameStatus(VkCommandBuffer command, std::uint32_t slot) {
  Frames &f = *frames;
  transferAfterCompute(command);
  const VkBufferCopy region{f.layout.status, slot * sizeof(pt::BvhCollapseStatus), sizeof(pt::BvhCollapseStatus)};
  vkCmdCopyBuffer(command, f.setup.handle, f.status.handle, 1, &region);
  f.pending[slot] = true;
}

void GpuBvhCollapser::recordCollapse(VkCommandBuffer command, const Buffer &inputRows, const Buffer &outputRows,
                                     std::uint32_t slot) {
  if (!frames) throw std::runtime_error("the GPU wide collapse has no in-frame state");
  if (slot >= kFrameSlots) throw std::runtime_error("wide collapse frame slot out of range");
  Frames &f = *frames;
  const Layout &l = f.layout;
  context.beginLabel(command, "collapse (in frame)");
  // Start afresh: the level headers and bases, the root tasks, the error word; task 0 is the
  // TLAS root.
  transferAfterCompute(command);
  const VkBufferCopy restore[] = {{l.header, l.header, l.headerBytes},
                                  {l.level, l.level, l.levelBytes},
                                  {l.rootTask, l.rootTask, l.rootBytes},
                                  {l.error, l.error, 4}};
  vkCmdCopyBuffer(command, f.pristine.handle, f.setup.handle, static_cast<std::uint32_t>(std::size(restore)), restore);
  vkCmdFillBuffer(command, f.taskNode.handle, 0, sizeof(std::uint32_t), 0u);
  barrier(command);
  auto indirect = [&](const char *entry, VkDescriptorSet s, std::uint32_t level) {
    const Kernel &k = kernel(entry);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline.handle);
    k.program->bind(command, s);
    vkCmdDispatchIndirect(command, f.setup.handle, l.header + level * sizeof(pt::uint4));
    barrier(command);
  };
  auto direct = [&](const char *entry, VkDescriptorSet s, std::uint32_t groups) {
    const Kernel &k = kernel(entry);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline.handle);
    k.program->bind(command, s);
    vkCmdDispatch(command, groups, 1, 1);
    barrier(command);
  };
  for (std::uint32_t level = 0; level < kLevels; ++level) indirect("bvh_collapse_gather", f.gather[level], level);
  for (std::uint32_t level = kLevels; level-- > 0;) indirect("bvh_collapse_size", f.size[level], level);
  direct("bvh_collapse_roots", f.roots, 1u);
  direct("bvh_collapse_emit_args", f.emitArgs, (kLevels + 1u + 63u) / 64u);
  recordEmit(command);
  recordRows(command, inputRows, outputRows);
  context.endLabel(command);
  copyFrameStatus(command, slot);
}

void GpuBvhCollapser::recordReemit(VkCommandBuffer command, const Buffer &inputRows, const Buffer &outputRows,
                                   std::uint32_t slot) {
  if (!frames) throw std::runtime_error("the GPU wide collapse has no in-frame state");
  if (slot >= kFrameSlots) throw std::runtime_error("wide collapse frame slot out of range");
  context.beginLabel(command, "collapse re-emit (in frame)");
  recordEmit(command);
  recordRows(command, inputRows, outputRows);
  context.endLabel(command);
  copyFrameStatus(command, slot);
}

std::string GpuBvhCollapser::frameError(std::uint32_t slot, std::uint32_t *maximumStack, std::uint32_t *nodeCount) {
  if (!frames || slot >= kFrameSlots || !frames->pending[slot]) return {};
  Frames &f = *frames;
  f.pending[slot] = false;
  pt::BvhCollapseStatus status{};
  std::memcpy(&status, static_cast<const std::uint8_t *>(f.status.mapped) + slot * sizeof(status), sizeof(status));
  if (maximumStack) *maximumStack = status.result.w;
  if (nodeCount) *nodeCount = status.result.z;
  if (status.result.x != pt::kBvhBuildErrorNone)
    return std::string("GPU wide collapse failed: ") + errorName(status.result.x) + " (code " +
           std::to_string(status.result.x) + "; traversal stack bound " + std::to_string(status.result.w) + " of " +
           std::to_string(pt::kWideStackDeep) + ", " + std::to_string(status.result.y) + " levels)";
  if (status.result.z == 0u || status.result.z > f.capacity)
    return "GPU wide collapse published an impossible node count";
  return {};
}

// Here, where Kept and Frames are complete.
GpuBvhCollapser::~GpuBvhCollapser() {
  if (timestamps) vkDestroyQueryPool(context.device, timestamps, nullptr);
}

} // namespace basalt
