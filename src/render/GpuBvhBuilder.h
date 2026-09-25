#pragma once
#include "gpu/Resources.h"
#include "pt/Bvh.h"
#include "pt/BvhVariants.h"
#include "render/TraceScene.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace pt {
struct BvhBuildStatus2;  // shaders/pt/bvh_build.h
}

namespace basalt {

class Context;
class DescriptorPool;
class Uploader;

// The GPU builders' shared status record (shaders/pt/bvh_build.h), widened for the host.
struct GpuBvhStatus {
  std::uint32_t error = 0, topDepth = 0, bottomDepth = 0, maximumBuilderStack = 0;
  std::uint32_t nodes = 0, triangles = 0, instances = 0, sortPasses = 0;
  std::uint32_t dispatches = 0, fitIterations = 0;
  std::uint32_t plocIterations = 0;  // PLOC topology: iterations that merged clusters
  double topCost = 0.0, bottomCost = 0.0;
};

// Stage names every GPU builder reports times for, in this order (absent stages are 0).
inline constexpr const char *kGpuBvhStageNames[] = {"extents", "morton", "sort", "topology", "fit", "emit", "collapse"};
inline constexpr std::size_t kGpuBvhStageCount = sizeof(kGpuBvhStageNames) / sizeof(kGpuBvhStageNames[0]);

struct GpuBvhBuildResult {
  Buffer nodes;
  Buffer triangles;
  std::vector<pt::TraceInstance> instances;
  pt::BvhStatistics statistics{};  // sahCost: the bottom levels' cost, as the CPU builder reports
  GpuBvhStatus status{};
  std::vector<double> stageMilliseconds;  // kGpuBvhStageNames order; empty when not timed
  VkDeviceSize scratchBytes = 0;
  VkDeviceSize outputBytes = 0;
  double milliseconds = 0.0;
  std::uint32_t radixPasses = 0;
  std::uint32_t maximumBuilderStack = 0;
};

// A refit or TLAS rebuild of a kept GPU LBVH.
struct GpuBvhUpdateResult {
  std::vector<pt::TraceInstance> instances;  // the build's rows with the new transforms
  std::uint32_t dispatches = 0;
  std::uint32_t topDepth = 0, bottomDepth = 0;  // of the updated tree, as a build reports them
  std::vector<double> stageMilliseconds;     // kGpuBvhStageNames order; stages not run are 0
  double gpuMilliseconds = 0.0;
  double milliseconds = 0.0;                 // wall clock
};

// Builds the traversal structure entirely from the resident GPU geometry. The host only
// supplies primitive counts/offsets and reads the small failure/status record.
GpuBvhBuildResult buildGpuBvh(const Context &context, Uploader &uploader,
                              const Scene &scene, const TraceScene &trace);

// The parallel builder's topology: Karras's LBVH (the serial builder's tree), or PLOC's
// agglomerative clustering in Morton order (shaders/bvh_ploc.metal), numbered and published alike.
enum class GpuBvhTopology { Lbvh, Ploc };

// The parallel LBVH (shaders/bvh_lbvh.metal, bvh_sort.metal): the serial builder's tree for
// every bottom level and the TLAS in one set of dispatches, with its numbering; or, with the PLOC
// topology, the same sort, fit, numbering and layout around PLOC's tree. The programs and
// pipelines are made once; build() allocates one scene's scratch and releases it unless asked to
// keep it for refit() and rebuildTopLevel().
class GpuLbvhBuilder {
public:
  explicit GpuLbvhBuilder(const Context &context);
  ~GpuLbvhBuilder();
  GpuLbvhBuilder(const GpuLbvhBuilder &) = delete;
  GpuLbvhBuilder &operator=(const GpuLbvhBuilder &) = delete;

  // refittable: keep this build's scratch so refit() can update the tree; a later build
  // releases it.
  GpuBvhBuildResult build(Uploader &uploader, const Scene &scene, const TraceScene &trace, bool refittable = false,
                          GpuBvhTopology topology = GpuBvhTopology::Lbvh);

  // The kept build's boxes and triangles from the scene's current vertices and the trace's
  // instance transforms, into that build's `nodes` and `triangles`: topology and numbering
  // unchanged (no sort), so only the geometry may move, not the triangle counts.
  GpuBvhUpdateResult refit(Uploader &uploader, const Scene &scene, const TraceScene &trace, const Buffer &nodes,
                           const Buffer &triangles);

  // The kept build's TLAS rebuilt for the trace's instance transforms (its bottom levels kept):
  // what build() would publish for the moved instances, when their bottom levels are unchanged.
  // The rebuilt TLAS is an LBVH whichever topology the build used (a PLOC build's bottom levels
  // under an LBVH top level).
  GpuBvhUpdateResult rebuildTopLevel(Uploader &uploader, const Scene &scene, const TraceScene &trace,
                                     const Buffer &nodes, const Buffer &triangles);

  // In-frame updates: the same work recorded into `command` (nothing waits), reading the new
  // instance rows from `rows` (updateRows()' layout, written before the commands run). The
  // status goes to frame slot `slot` (< kFrameSlots); frameError(slot) reads it once the
  // frame's commands have completed ("" when the update succeeded or none was recorded).
  static constexpr std::uint32_t kFrameSlots = 4;
  std::vector<pt::TraceInstance> updateRows(const Scene &scene, const TraceScene &trace) const;
  std::uint32_t recordRefit(VkCommandBuffer command, const Scene &scene, const Buffer &rows, const Buffer &nodes,
                            const Buffer &triangles, std::uint32_t slot);
  std::uint32_t recordRebuildTopLevel(VkCommandBuffer command, const Scene &scene, const Buffer &rows,
                                      const Buffer &nodes, const Buffer &triangles, std::uint32_t slot);
  std::string frameError(std::uint32_t slot);

  // The build's radix sort alone, for tests: sorts the triples stably by (hi, lo) in place.
  // highPasses: 8-bit passes over hi (the build uses enough for its segment count).
  void sortTriples(Uploader &uploader, std::vector<std::uint32_t> &lo, std::vector<std::uint32_t> &hi,
                   std::vector<std::uint32_t> &values, std::uint32_t highPasses);

private:
  struct Kernel;
  struct Kept;
  struct SortSets {
    VkDescriptorSet histogram, reduce, blocks, apply, scatter;
  };
  // The kernels' sets; morton, topology, seed and emit read the part at control slot `partSlot`.
  struct CommonSets {
    VkDescriptorSet extents = VK_NULL_HANDLE, instances = VK_NULL_HANDLE, morton = VK_NULL_HANDLE,
                    topology = VK_NULL_HANDLE, seed = VK_NULL_HANDLE, emit = VK_NULL_HANDLE,
                    finish = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> fit, number;
  };
  const Kernel &kernel(const char *entry) const;
  CommonSets commonSets(DescriptorPool &pool, const Kept &kept, const Scene &scene, const Buffer &nodes,
                        const Buffer &triangles, std::uint32_t partSlot) const;
  std::uint32_t recordFit(VkCommandBuffer command, const Kept &kept, const CommonSets &sets) const;
  std::uint32_t recordNumber(VkCommandBuffer command, const Kept &kept, const CommonSets &sets) const;
  std::vector<pt::TraceInstance> keptRows(const Scene &scene, const TraceScene &trace) const;
  std::uint32_t recordRefitCommands(VkCommandBuffer command, const Kept &kept, const CommonSets &common,
                                    const Buffer &rows, const std::function<void(std::uint32_t)> &mark) const;
  std::uint32_t recordTopLevelCommands(VkCommandBuffer command, const Kept &kept, const CommonSets &common,
                                       const std::vector<SortSets> &sorting, const Buffer &rows,
                                       const std::function<void(std::uint32_t)> &mark) const;
  const CommonSets &frameSets(const Scene &scene, const Buffer &nodes, const Buffer &triangles, bool top);
  void copyFrameStatus(VkCommandBuffer command, std::uint32_t slot);
  std::vector<double> readStages(double &total) const;
  pt::BvhBuildStatus2 readStatus(Uploader &uploader, const Kept &kept, const char *what) const;
  // Pass p reads keys[p & 1] and writes keys[(p & 1) ^ 1]; its BvhSortControl is at
  // controls + controlBase + p * 256.
  std::vector<SortSets> sortSets(DescriptorPool &pool, const Buffer *keysLo, const Buffer *keysHi,
                                 const Buffer *values, const Buffer &histogram, const Buffer &blockSums,
                                 const Buffer &controls, VkDeviceSize controlBase, std::uint32_t passes) const;
  std::uint32_t recordSort(VkCommandBuffer command, const std::vector<SortSets> &sets, std::uint32_t tiles,
                           std::uint32_t blocks) const;
  const Context &context;
  std::vector<std::unique_ptr<Kernel>> kernels;
  VkQueryPool timestamps = VK_NULL_HANDLE;
  std::unique_ptr<Kept> kept;
};

struct GpuWideCollapseResult {
  Buffer nodes;                              // pt::QuantizedWideNode x nodeCount
  std::vector<pt::TraceInstance> instances;  // blasRoot: the bottom level's wide root
  std::uint32_t nodeCount = 0, maximumStack = 0, levels = 0, dispatches = 0;
  VkDeviceSize scratchBytes = 0;
  double gpuMilliseconds = 0.0;  // the kernels: gather + size (with the roots) + emit
  double gatherMilliseconds = 0.0, sizeMilliseconds = 0.0, emitMilliseconds = 0.0;
  double milliseconds = 0.0;     // wall clock, host work and the readback included
};

// Collapses a resident binary tree (the TLAS at node 0 and one bottom level per instance, as
// every builder publishes) into pt::buildWideBvh's quantized BVH4/BVH8 nodes, byte for byte
// and in its node order (shaders/bvh_collapse.metal). The programs are made once.
class GpuBvhCollapser {
public:
  explicit GpuBvhCollapser(const Context &context);
  ~GpuBvhCollapser();
  GpuBvhCollapser(const GpuBvhCollapser &) = delete;
  GpuBvhCollapser &operator=(const GpuBvhCollapser &) = delete;

  // binaryDepth: the TLAS depth plus the deepest bottom level's (node levels), which bounds the
  // wide tree's levels and so the dispatches; 0 when unknown (64 levels).
  GpuWideCollapseResult collapse(Uploader &uploader, const Buffer &binaryNodes, std::uint32_t binaryNodeCount,
                                 const std::vector<pt::TraceInstance> &instances, std::uint32_t width,
                                 std::uint32_t binaryDepth = 0, bool keep = false);

  // After a refit of the binary tree a kept collapse (keep = true) re-emits its nodes into
  // `wideNodes`: the same frontiers, slots and numbering, the refit boxes quantised again. A
  // wide refit, not a new collapse, whose choices depend on the boxes; a later collapse releases
  // the kept one.
  GpuWideCollapseResult reemit(Uploader &uploader, const Buffer &binaryNodes, const Buffer &wideNodes);

  // In-frame collapses, recorded into the frame's command buffer with nothing read back:
  // prepareFrames() makes the scratch and the descriptor sets for this binary tree, its instances
  // (binary roots) and `output`, which holds as many nodes as the binary tree (the most a
  // collapse emits) and must outlive the frames that use it. recordCollapse()
  // collapses anew; recordReemit() re-emits the last recorded collapse over a refit tree. Both then
  // write the traced rows: `inputRows` (binary roots, current transforms) with the wide roots, to
  // `outputRows`. frameError(slot) reads the recorded status once the frame has completed.
  static constexpr std::uint32_t kFrameSlots = 4;
  void prepareFrames(Uploader &uploader, const Buffer &binaryNodes, std::uint32_t binaryNodeCount,
                     const std::vector<pt::TraceInstance> &instances, std::uint32_t width, const Buffer &output);
  void recordCollapse(VkCommandBuffer command, const Buffer &inputRows, const Buffer &outputRows, std::uint32_t slot);
  void recordReemit(VkCommandBuffer command, const Buffer &inputRows, const Buffer &outputRows, std::uint32_t slot);
  std::string frameError(std::uint32_t slot, std::uint32_t *maximumStack = nullptr, std::uint32_t *nodeCount = nullptr);

private:
  struct Kernel;
  struct Kept;
  struct Frames;
  const Kernel &kernel(const char *entry) const;
  void recordRows(VkCommandBuffer command, const Buffer &input, const Buffer &output);
  void recordEmit(VkCommandBuffer command);
  void copyFrameStatus(VkCommandBuffer command, std::uint32_t slot);
  const Context &context;
  std::vector<std::unique_ptr<Kernel>> kernels;
  VkQueryPool timestamps = VK_NULL_HANDLE;
  std::unique_ptr<Kept> kept;
  std::unique_ptr<Frames> frames;
};

// SAH cost of a resident tree (binary float4 nodes, or quantized wide nodes when wide),
// computed per node on the GPU and summed per tree in double on the host. Every tree must
// occupy a contiguous node range: the TLAS from node 0, then each instance's bottom level
// from its blasRoot. Matches pt::binaryLayoutCost / pt::wideLayoutCost.
pt::LayoutCost gpuBvhSahCost(const Context &context, Uploader &uploader, const Buffer &nodes,
                             std::uint32_t nodeCount, const std::vector<pt::TraceInstance> &instances, bool wide);

} // namespace basalt

