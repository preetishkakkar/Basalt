#pragma once
// Metal2Vulkan host library: reflection reading, descriptor layouts, host data
// packing, argument-buffer encoding and exact-grid dispatch plans for
// applications that run msl2spirv output. No LLVM dependency; Vulkan headers
// only. Every function throws std::runtime_error on contract violations.
// See docs/HOST_LIBRARY.md.
#include <vulkan/vulkan.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <functional>

namespace m2v::host {

// A minimal JSON value: objects, arrays, strings, numbers, booleans and null,
// enough for reflection files and host layout descriptions.
struct Json {
  enum class Kind { Null, Boolean, Number, String, Array, Object };
  Kind kind = Kind::Null;
  bool boolean = false;
  double number = 0;
  std::string string;
  std::vector<Json> array;
  std::vector<std::pair<std::string, Json>> object;
  static Json parse(const std::string &text);
  static Json parseFile(const std::string &path);
  bool isObject() const { return kind == Kind::Object; }
  bool isArray() const { return kind == Kind::Array; }
  const Json *find(const std::string &key) const; // Objects: the member or null.
  const Json &at(const std::string &key) const;   // Objects: the member, or throws.
  std::string text(const std::string &key) const;
  std::int64_t integer(const std::string &key, std::int64_t maximum) const; // 0..maximum, or throws.
  bool flag(const std::string &key) const;
  const std::vector<Json> &list(const std::string &key) const;
};

// One shader stage's reflected resources (schema 1, binding contract set_per_stage_class_offset).
struct Field { std::string name, type; unsigned offset = 0, size = 0; std::string role; }; // role: compute_pipeline_state / render_pipeline_state (docs/INDIRECT_COMMANDS.md).
// An optional argument ([[function_constant(name)]] on the parameter): the bool constant that says
// whether it exists and the spec id the host sets it under. The descriptor is in the layout either
// way; where the constant is false the shader never reaches the argument and the host binds a
// placeholder (docs/FUNCTION_CONSTANTS.md, absentOptionalResources).
struct Condition { std::string constant; std::uint32_t specId = 0; };
struct Buffer {
  std::string name;
  std::uint32_t set = 0, binding = 0, metalIndex = 0;
  unsigned stride = 0, minimumBytes = 0;
  bool reference = false, readOnly = true;
  // Whether the entry loads from and stores to this buffer. A chain of compute passes takes its
  // dependencies from exactly these: a buffer one pass writes and a later one reads is the
  // dependency between them, and a buffer a pass only writes needs no host contents, because it is
  // that pass's output (docs/COMPUTE_DISPATCH.md).
  bool reads = true, writes = false;
  std::optional<Condition> condition; // Set for an optional argument.
  bool uniform = false;  // descriptor_type uniform_buffer.
  bool argument = false; // A member of an argument buffer: (argumentBuffer, argumentId) instead of a Metal index.
  std::uint32_t argumentBuffer = 0, argumentId = 0;
  // Descriptor array length: an argument-buffer pointer array (T* w[N]) is one
  // binding with descriptorCount N (element e answers to host key() + e); a
  // member of an argument-buffer array (capacity C, --argument-capacity) has
  // C times its own count descriptors, element e's at e*own.., keyed
  // BUFFER[e].ID (+ j). elementKey(d) is the host key of descriptor d.
  std::uint32_t count = 1, capacity = 1;
  // Ray tracing (docs/RAY_TRACING.md): an intersection function's own [[buffer(n)]] names its
  // function and binds in the function class (1024 + 32 * function index + n); an intersection
  // function table is a read-only buffer of u32 words, one function index per slot, that the host
  // fills with functionTableWords.
  std::string intersectionFunction;
  bool functionTable = false;
  std::string signature; // A visible function table (docs/INDIRECT_CALLS.md): the type its functions share.
  // A command_buffer member (docs/INDIRECT_COMMANDS.md): the host allocates commandWords * count words,
  // the encoding kernel writes records into it, decodeComputeCommands reads them back.
  bool commandBuffer = false;
  std::uint32_t commandWords = 0;
  std::string commandKind; // compute, render, or unused (the entry never encodes into it).
  // A member of an array of argument structs (Inner items[N]): `groups` = N and
  // `idStride` the Metal ids one element spans; element g's descriptors sit at
  // g*own.. and answer to key() + g*idStride (+ j).
  std::uint32_t groups = 1, idStride = 0;
  std::vector<Field> fields;
  // Host lookup key: the Metal index, or an encoding of BUFFER.ID for argument members (see resourceKey).
  std::uint32_t key() const { return argument ? (0x10000u | (argumentBuffer << 8) | argumentId) : metalIndex; }
  std::uint32_t elementKey(std::uint32_t descriptor) const {
    const auto perElement = count / capacity, own = perElement / groups;
    const auto element = descriptor / perElement, group = descriptor % perElement / own, index = descriptor % own;
    return key() + group * idStride + index + (element << 20);
  }
  VkDescriptorType descriptorType() const { return uniform ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; }
};
struct Handle {
  std::string name;
  std::uint32_t set = 0, binding = 0, metalIndex = 0;
  std::optional<Condition> condition; // Set for an optional argument.
  bool argument = false;
  std::uint32_t argumentBuffer = 0, argumentId = 0;
  // A fixed array of handles (array<T, N>): one binding with descriptorCount
  // count; element e answers to host key() + e (Metal index or argument id).
  // A member of an argument-buffer array (capacity C) has C times its own
  // count descriptors, keyed BUFFER[e].ID (+ j): elementKey(d) for descriptor d.
  std::uint32_t count = 1, capacity = 1;
  // Textures: the MSL access qualifier (sample, read, write, read_write); every
  // access but sample is a storage image (VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) that
  // the host binds in the General layout with a format supporting storage
  // reads/writes without a format.
  // Sampled textures reflect mip_levels 0: the host chooses the mip chain
  // (sampled explicitly by level or gradient, implicitly in fragment entries).
  std::string access = "sample";
  std::string component = "f32"; // f32/f16, i32/i16 (SINT formats) or u32/u16 (UINT formats); narrow texels convert from 32-bit image operations.
  std::string dimension = "2d", view = "2d"; // 1d, 1d_array, 2d, 2d_array, 3d, cube, cube_array; storage cubes bind 2d_array views.
  bool storage = false;
  bool multisampled = false;
  VkDescriptorType descriptorType() const {
    return dimension == "buffer" ? (storage ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER) :
      storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  }
  std::vector<std::string> formatFeatures; // storage_read_without_format / storage_write_without_format: VkFormatProperties3 bits the host's format must offer.
  // The software sampling paths this texture's samples take (docs/TEXTURES.md), such as
  // pixel_2d_clamp_edge; a clamp_zero or clamp_border path also appears in the stage's
  // softwareBorderChannels, whose mask the host sets from the format it binds.
  std::vector<std::string> softwareSampling;
  // Samplers: a constexpr sampler's reflected state (samplerCreateInfo builds
  // the VkSamplerCreateInfo); runtime samplers leave it empty and take host settings.
  struct ConstexprSampler {
    std::string magFilter = "nearest", minFilter = "nearest", mipFilter = "none";
    std::array<std::string, 3> address{"clamp_to_edge", "clamp_to_edge", "clamp_to_edge"};
    bool pixelCoordinates = false;
    std::string borderColor = "transparent_black";
    std::string reduction = "weighted_average";
    std::uint32_t maxAnisotropy = 1;
    std::optional<std::pair<float, float>> lodClamp;
    std::string compare = "none"; // compare_func: none, less, less_equal, greater, greater_equal, equal, not_equal, always, never.
  };
  std::optional<ConstexprSampler> constexprSampler;
  // Samplers: used with sample_compare/gather_compare; a host (runtime) comparison sampler must enable compareEnable.
  bool comparison = false;
  // Textures: a depth texture (sampled from a depth format whose features include SAMPLED_IMAGE_DEPTH_COMPARISON).
  bool depth = false;
  // True only when this image is queried for LOD/sizes without texel access.
  // Missing in older reflection: preserve the conservative sampler checks.
  bool lodQueryOnly = false;
  // Textures: an atomic texture binds exactly `format` (r32uint or r32sint) with
  // STORAGE_IMAGE_ATOMIC or, for texel buffers, STORAGE_TEXEL_BUFFER_ATOMIC support.
  bool atomic = false;
  std::string format; // Empty unless the reflection fixes the format.
  std::uint32_t key() const { return argument ? (0x10000u | (argumentBuffer << 8) | argumentId) : metalIndex; }
  std::uint32_t elementKey(std::uint32_t descriptor) const { const auto own = count / capacity; return key() + descriptor % own + ((descriptor / own) << 20); }
};
struct ArgumentMember {
  std::uint32_t id = 0;
  std::string name, kind; // buffer, texture, sampler or constant.
  std::uint32_t set = 0, binding = 0; // Resource members.
  std::string descriptorType;
  unsigned offset = 0, size = 0; // Inline constants: within the argument buffer's inline block.
  std::string type;
};
struct ArgumentBuffer {
  std::string name;
  std::uint32_t metalIndex = 0;
  std::uint32_t capacity = 1; // > 1: an array of argument buffers (a pointer parameter compiled with --argument-capacity).
  bool inlineBlock = false;
  std::uint32_t inlineSet = 0, inlineBinding = 0;
  std::vector<ArgumentMember> members;
};
// An acceleration structure the entry traverses (docs/RAY_TRACING.md): a descriptor of type
// VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR at (set, binding), named by its Metal buffer index.
// `instance` says the shader was written against an instance_acceleration_structure rather than a
// primitive one; Vulkan binds a top-level structure either way, so this is for the host to check.
struct AccelerationStructure {
  std::string name;
  std::uint32_t set = 0, binding = 0, metalIndex = 0;
  bool instance = false;
};
// An [[intersection(kind)]] function of the source (docs/RAY_TRACING.md): what a function table's
// slots may name, by index; the compiler inlines it where a candidate of its kind selects it.
struct IntersectionFunction {
  std::string name, kind; // kind: bounding_box or triangle.
  std::uint32_t index = 0;
};
// A [[visible]] function the stage may dispatch to (docs/INDIRECT_CALLS.md): a function pointer
// value is its index, and a visible_function_table's words are such indices (0xFFFFFFFF empty).
struct VisibleFunction {
  std::string name, signature;
  std::uint32_t index = 0;
};
// A ray-pipeline interface of a ray stage (docs/RAY_PIPELINES.md): payload, incoming_payload,
// callable_data, incoming_callable_data, hit_attribute or shader_record. Two stages agree on an
// interface exactly when their fingerprints do; a shader record's offsets and size are the MSL layout
// the host writes into the SBT record after the group handle.
struct RayInterface {
  std::string kind, name, fingerprint;
  std::uint32_t packedBytes = 0, size = 0, alignment = 0;
  bool readOnly = false;
  struct Field { std::string name, type; std::uint32_t offset = 0, count = 1, size = 0; };
  std::vector<Field> fields;
};
// What a ray stage's reflection says beyond its descriptors: its interfaces, the built-ins it reads and
// the shader calls it makes. `present` is false for every other stage.
struct RayStageInfo {
  bool present = false;
  std::vector<RayInterface> interfaces;
  std::vector<std::string> builtins;
  bool traces = false, executesCallables = false, reportsIntersections = false, ignoresIntersections = false, terminatesRays = false;
  const RayInterface *find(const std::string &kind) const {
    for (const auto &interface : interfaces) if (interface.kind == kind) return &interface;
    return nullptr;
  }
};
struct DispatchPlanInfo {
  std::array<std::uint32_t, 3> defaultLocalSize{1, 1, 1};
  std::array<std::uint32_t, 3> specIds{0, 1, 2};
  std::uint32_t pushConstantBytes = 64;
};
struct Stage {
  std::string stage, entry;
  std::uint32_t set = 0;
  std::array<std::uint32_t, 3> localSize{1, 1, 1}; // Compute only.
  bool exactGrid = false;                          // dispatch: exact_grid_plan.
  std::optional<DispatchPlanInfo> plan;
  std::vector<std::string> requiredFeatures;
  // Device properties (not features) the entry needs: quadDivergentImplicitLod, which a host reads from
  // VkPhysicalDeviceDescriptorIndexingProperties, and the subgroup operation classes (subgroupBasic,
  // subgroupVote, subgroupBallot, subgroupShuffle, ...) and subgroupWidth8 (SIMD-group matrices) read
  // from VkPhysicalDeviceSubgroupProperties (docs/SUBGROUPS.md). See supportsProperty.
  std::vector<std::string> requiredProperties;
  // Runtime samplers whose LOD state a clamped LOD query needs (docs/TEXTURES.md). The host publishes
  // lod_min, lod_max and max_anisotropy as three consecutive floats at `offset` in the fragment
  // push-constant block; clampedLodBounds() derives them from the sampler's own VkSamplerCreateInfo.
  struct ClampedLodBound { std::string name; std::uint32_t metalIndex = 0, set = 0, binding = 0, offset = 0; };
  std::vector<ClampedLodBound> clampedLodBounds;
  // Visible function tables whose size() the entry reads (docs/INDIRECT_CALLS.md). The host publishes
  // each table's slot count -- the bound buffer's byte size over four, since functionTableWords writes
  // one u32 per slot -- as one u32 at `offset` in the stage's push-constant block, before every draw
  // or dispatch, exactly as it pushes the residency word.
  struct FunctionTableLength { std::string name; std::uint32_t metalIndex = 0, set = 0, binding = 0, offset = 0; };
  std::vector<FunctionTableLength> functionTableLengths;
  // Textures sampled with clamp_to_zero or clamp_to_border on the software pixel path
  // (docs/TEXTURES.md). The border colour is expanded for the bound format as a texel is -- a
  // channel the format lacks reads 0, alpha reads 1 -- which is what Metal does; the format is the
  // host's, so the host sets each texture's channel mask (bit c for channel c) at `specId` with
  // specializeTextureChannels. An unset mask is 15, every channel.
  struct TextureChannels { std::string name; std::uint32_t metalIndex = 0, set = 0, binding = 0, specId = 0; };
  std::vector<TextureChannels> softwareBorderChannels;
  // Runtime samplers whose unclamped LOD queries run through the compiler's neutral sampler
  // (docs/TEXTURES.md): the host checks every bound element with checkLodQuerySampler().
  struct LodQuerySampler { std::string name; std::uint32_t metalIndex = 0, set = 0, binding = 0, count = 1; };
  std::vector<LodQuerySampler> lodQuerySamplers;
  // get_num_samples()/get_sample_position() (docs/STAGE_INTERFACES.md): the host writes `capacity`
  // float2 positions at positionsOffset and the attachment's sample count at countOffset, in the
  // fragment push-constant block. Absent when the entry asks for neither.
  struct SampleQueries { bool used = false; std::uint32_t positionsOffset = 0, countOffset = 0, capacity = 0; };
  SampleQueries sampleQueries;
  std::vector<std::string> builtinInputs;  // Built-in inputs the entry reads: instance_id, base_instance, base_vertex, front_facing, point_coord, primitive_id, sample_id.
  std::vector<std::string> builtinOutputs; // Built-in outputs: point_size, clip_distance, render_target_array_index, viewport_array_index (vertex), depth (fragment).
  // How many clip distances the vertex entry writes (MSL's float clip [[clip_distance]] [N]); zero
  // when it writes none. Vulkan guarantees only maxClipDistances of them, so the host checks it.
  std::uint32_t clipDistances = 0;
  std::vector<std::uint32_t> colorOutputs;  // Fragment color attachment indices the entry writes, attachment 0 first.
  // The reflected type of each of those outputs, in the same order ("f32x4", "u32x4", ...). An
  // attachment's format must be of the same numeric class, which checkAttachments enforces.
  std::vector<std::string> colorOutputTypes;
  // An attachment a [[function_constant(name)]] field declares exists only where that constant is
  // true, so a host that specializes it false creates a pipeline without it (docs/FUNCTION_CONSTANTS.md).
  struct ConditionalOutput { std::uint32_t index = 0; std::string constant; std::uint32_t specId = 0; };
  std::vector<ConditionalOutput> conditionalOutputs;
  // The entry writes a [[color(0), index(1)]] second blend source. The pipeline must enable the
  // dualSrcBlend feature and blend attachment zero with SRC1 factors, and Vulkan bounds the number
  // of attachments by maxFragmentDualSrcAttachments while it does (see supportsDualSourceBlending).
  bool dualSourceBlending = false;
  // Mesh entries: the capacities and topology the [[mesh]] entry declared, which bound what one
  // threadgroup may produce. Zero vertices means the stage is not a mesh entry.
  struct Mesh { std::uint32_t maxVertices = 0, maxPrimitives = 0, indicesPerPrimitive = 0; std::string topology; };
  Mesh mesh;
  // Every varying location the stage declares, and the ones that cross the interface once per
  // primitive rather than once per vertex: a mesh entry's per_primitive_varyings and the fragment
  // inputs marked per_primitive. Each module fixes the decoration at compile time, so the two stages
  // have to agree and a host can see whether they do (docs/MESH_SHADERS.md).
  std::vector<std::uint32_t> varyingLocations, perPrimitiveVaryings;
  // Vertex amplification (MSL 5.2.3.4): the entry reads [[amplification_id]], which is the view, and
  // possibly [[amplification_count]], which the host specializes to the number of views it renders.
  struct Amplification { bool views = false, readsCount = false; std::uint32_t viewCountSpecId = 0; };
  Amplification amplification;
  // The object_data payload an object entry writes and the mesh entries it launches read: shared
  // threadgroup storage rather than a descriptor, so the host binds nothing for it and only checks
  // that the two stages agree and that it fits maxTaskPayloadSize (docs/OBJECT_ENTRIES.md).
  struct Payload {
    std::string name;
    std::uint32_t size = 0, alignment = 0;
    bool readOnly = false;
    struct Field { std::string name, type; std::uint32_t offset = 0; };
    std::vector<Field> fields;
  };
  Payload payload;
  // A post-tessellation vertex entry: the patch it runs for, the tessellator state its module fixes,
  // and where the generated control stage reads Metal's factor buffer (docs/TESSELLATION.md).
  struct Patch {
    std::string type, partition, winding;
    std::uint32_t controlPoints = 0, factorBufferBinding = 0;
  };
  Patch patch;
  // [[function_constant(index)]] constants the entry uses. `required` means the entry reads the value
  // outside an is_function_constant_defined guard, so a pipeline without a value for it is refused.
  struct FunctionConstant {
    std::string name, type; // type is bool, i32, u32 or f32, or a vector of them (f32x4, i32x2, ...).
    std::uint32_t index = 0, specId = 0, definedSpecId = 0;
    bool required = false, queried = false;
    // A vector constant is one specialization constant per lane, at laneSpecIds; a scalar has one
    // lane and uses specId.
    std::uint32_t lanes = 1;
    std::vector<std::uint32_t> laneSpecIds;
  };
  std::vector<FunctionConstant> functionConstants;
  std::vector<Buffer> buffers;
  std::vector<Handle> textures, samplers;
  std::vector<ArgumentBuffer> argumentBuffers;
  // The acceleration structures the entry traverses; requiredFeatures then names rayQuery and
  // accelerationStructure, which the host enables with the extensions (VK_KHR_acceleration_structure,
  // VK_KHR_ray_query, VK_KHR_deferred_host_operations) and bufferDeviceAddress for the build.
  std::vector<AccelerationStructure> accelerationStructures;
  std::vector<IntersectionFunction> intersectionFunctions;
  std::vector<VisibleFunction> visibleFunctions;
  unsigned threadgroupBytes = 0;
  // Nullable resources (compute): parameter names the host may leave unbound.
  // The host binds a placeholder for each unbound one and pushes residencyMask()
  // at residencyOffset (a 4-byte push constant after the dispatch plan).
  std::vector<std::string> nullable;
  std::uint32_t residencyOffset = 0;
  // Device addresses (--device-addresses): the entry reads a table of one 64-bit
  // device address per buffer (in `buffers` order) from a read-only storage
  // buffer at (set, deviceAddressBinding); the host creates every buffer with
  // VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT and enables bufferDeviceAddress.
  bool deviceAddresses = false;
  std::uint32_t deviceAddressBinding = 0;
  // Ray-pipeline stages (ray_generation, miss, closest_hit, any_hit, intersection, callable).
  RayStageInfo rayPipeline;
};
// The standard sample locations Vulkan defines for a sample count, as normalized offsets inside the
// pixel, in sample-index order. Only counts a device reporting standardSampleLocations must honour
// are known: 1, 2, 4, 8 and 16. Throws for any other count, so a host cannot publish invented
// positions. A host that programs custom sample locations supplies its own instead.
std::vector<std::array<float, 2>> standardSamplePositions(VkSampleCountFlagBits samples);
// The lod_min, lod_max and max_anisotropy a clamped LOD query reads from a runtime sampler, taken from
// the very VkSamplerCreateInfo the host passes to vkCreateSampler so the published values cannot drift
// from the sampler they describe. An absent Metal lod_clamp is Vulkan's 0 and VK_LOD_CLAMP_NONE, which
// the shader's arithmetic already reduces to the unclamped case, and a sampler without anisotropy
// reports 1. Throws for the states the constexpr clamped LOD profile rejects: a comparison sampler,
// unnormalized coordinates, or an active min/max reduction.
std::array<float, 3> clampedLodBounds(const VkSamplerCreateInfo &info);
// The channel mask of a format: bit c set when the format stores channel c (r, g, b, a); a depth
// format is one channel. Throws for a format the table does not know, rather than guess.
std::uint32_t textureChannelMask(VkFormat format);
// Throws unless a runtime sampler named in Stage::lodQuerySamplers has the state an unclamped LOD
// query admits: normalized coordinates, no comparison, no min/max reduction and no anisotropy.
void checkLodQuerySampler(const VkSamplerCreateInfo &info);
// One function constant value a host supplies, as parsed from a command line: the source index and the
// 4 bytes per lane the shader reads, already in the constant's own type.
struct FunctionConstantValue { std::uint32_t index = 0; std::vector<std::byte> bytes; };
// Parse "INDEX:VALUE" against a stage's declared constants: "3:1.5" for a float, "0:true"/"0:1" for a bool,
// decimal (or 0x) integers for int and uint, and one value per lane separated by commas for a vector
// ("4:1.0,2.0,3.0,4.0"). Throws when the index is not declared or the text does not parse in the
// constant's type and lane count.
FunctionConstantValue parseFunctionConstant(const Stage &stage, const std::string &text);
// The specialization a stage needs, given the values the host supplies. Every constant the stage declares
// contributes its "defined" flag, true only for the supplied ones, so is_function_constant_defined is
// answered for all of them. Throws when a required constant has no value: Vulkan would otherwise
// specialize it to zero silently, where Metal fails to create the pipeline state.
struct Specialization {
  std::vector<VkSpecializationMapEntry> entries;
  std::vector<std::byte> data;
  VkSpecializationInfo info() const {
    return {static_cast<std::uint32_t>(entries.size()), entries.data(), data.size(), data.data()};
  }
};
Specialization functionConstantSpecialization(const Stage &stage, const std::vector<FunctionConstantValue> &values);
// Adds (or replaces) the channel-mask specialization of every texture in stage.softwareBorderChannels,
// asking `format` for the VkFormat bound at (set, binding); a texture with no bound format -- a
// nullable one left unbound -- keeps the default of every channel (docs/TEXTURES.md).
void specializeTextureChannels(Specialization &specialization, const Stage &stage,
                               const std::function<std::optional<VkFormat>(std::uint32_t set, std::uint32_t binding)> &format);
// An encoded compute command (docs/INDIRECT_COMMANDS.md): what the record's 64 words say. A buffer
// binding names the encoding entry's reflected buffer by ordinal (buffers[ordinal]) plus a descriptor
// element and a byte offset; the host binds that buffer, at that offset, at the dispatched pipeline's
// buffer index.
struct ComputeCommand {
  enum class Kind : std::uint32_t { Empty = 0, DispatchThreadgroups = 1, DispatchThreads = 2 };
  struct BufferBinding { std::uint32_t index = 0, buffer = 0, element = 0, offset = 0, stride = 0; }; // buffer: reflected ordinal.
  struct ThreadgroupMemory { std::uint32_t index = 0, length = 0; };
  Kind kind = Kind::Empty;
  std::uint32_t pipeline = 0xFFFFFFFFu; // The pipeline id the host wrote into the pipeline state; 0xFFFFFFFF none.
  std::array<std::uint32_t, 3> grid{0, 0, 0}, threadsPerThreadgroup{0, 0, 0}; // grid: threadgroups (kind 1) or threads (kind 2).
  bool barrier = false;
  std::vector<BufferBinding> buffers;
  std::vector<ThreadgroupMemory> threadgroupMemory;
};
constexpr std::uint32_t kComputeCommandWords = 64;
constexpr std::uint32_t kCommandBufferBindings = 6, kCommandThreadgroupMemories = 4;
// An encoded render command (docs/INDIRECT_COMMANDS.md): a draw or an indexed draw with its vertex and
// fragment buffer bindings and the Metal 4 render state the record sets (stateMask says which).
struct RenderCommand {
  enum class Kind : std::uint32_t { Empty = 0, Draw = 3, DrawIndexed = 4 };
  enum State : std::uint32_t { CullMode = 1, Winding = 2, FillMode = 4, DepthClip = 8, DepthBias = 16, DepthStencil = 32 };
  Kind kind = Kind::Empty;
  std::uint32_t pipeline = 0xFFFFFFFFu;
  std::uint32_t primitive = 0;   // 0 point, 1 line, 2 line strip, 3 triangle, 4 triangle strip.
  std::uint32_t vertexStart = 0, vertexCount = 0, indexCount = 0, baseVertex = 0, instanceCount = 0, baseInstance = 0;
  ComputeCommand::BufferBinding indexBuffer; // buffer 0 when not indexed; stride is the index size (2 or 4).
  std::vector<ComputeCommand::BufferBinding> vertexBuffers, fragmentBuffers;
  std::uint32_t stateMask = 0, cullMode = 0, winding = 0, fillMode = 0, depthClipMode = 0, depthStencil = 0xFFFFFFFFu;
  float depthBias = 0, slopeScale = 0, depthBiasClamp = 0;
  bool barrier = false;
};
constexpr std::uint32_t kRenderCommandBindings = 4;
std::vector<RenderCommand> decodeRenderCommands(const std::vector<std::uint32_t> &words);
// Decodes the records of a command buffer read back from the device. Throws on a record whose kind or
// counts the format does not define, so a corrupt or foreign buffer is never replayed.
std::vector<ComputeCommand> decodeComputeCommands(const std::vector<std::uint32_t> &words);
// A rasterization rate map in this compiler's layout (docs/RATE_MAPS.md): up to four layers, each a
// grid of up to 16 x 16 cells whose column and row qualities in (0, 1] scale screen distance into
// physical distance. rateMapWords packs it as the constant buffer the decoder reads, rateMapPhysical
// maps a screen point as the decoder does (to size render targets), and shadingRateTexels quantizes a
// layer into VK_KHR_fragment_shading_rate attachment texels (the nearest of 1, 2 and 4 pixels per
// fragment on each axis) for a device that offers attachment rates.
struct RateMapLayer {
  std::array<std::uint32_t, 2> screen{0, 0};
  std::vector<float> horizontal, vertical; // One quality per column and per row, 1 to 16 each.
};
std::vector<std::uint32_t> rateMapWords(const std::vector<RateMapLayer> &layers);
// The 64-byte header of a bound tensor (docs/TENSORS.md): word 0 the rank (1 to 4), words 2 to 5 the
// extents, words 6 to 9 the strides in elements (dense with dimension 0 fastest when omitted), the
// rest zero. The elements follow the header in the same buffer; tensorElementCount is how many the
// extents and strides reach, so a host can size the buffer (header + count * element size).
std::vector<std::uint32_t> tensorHeaderWords(const std::vector<std::uint32_t> &extents, const std::vector<std::uint32_t> &strides = {});
std::uint32_t tensorElementCount(const std::vector<std::uint32_t> &extents, const std::vector<std::uint32_t> &strides = {});
constexpr std::uint32_t kTensorHeaderBytes = 64;
std::array<float, 2> rateMapPhysical(const RateMapLayer &layer, float x, float y);
std::array<std::uint32_t, 2> rateMapPhysicalSize(const RateMapLayer &layer);
// Fragment shading rate attachment texels for `layer`, one per `texelSize` x `texelSize` screen block,
// row-major; each texel is the extension's encoding (log2(height) << 2 | log2(width)).
std::vector<std::uint8_t> shadingRateTexels(const RateMapLayer &layer, std::uint32_t texelSize);
// Whether a physical device reports a property an entry's required_properties names. Throws for a name
// this library does not know, so a reflection from a newer compiler cannot pass unchecked.
bool supportsProperty(VkPhysicalDevice physical, const std::string &name);
// Whether a device can run a stage that declares a second blend source: dual-source blending is an
// optional feature, and while it is enabled Vulkan bounds the pipeline's attachments by
// maxFragmentDualSrcAttachments (1 on both local GPUs), which no shader property can relax.
bool supportsDualSourceBlending(VkPhysicalDevice physical, std::uint32_t attachments);
// Whether a device can run this mesh entry with that many threadgroups: VK_EXT_mesh_shader publishes
// per-stage output limits and workgroup counts, and a pipeline beyond them is refused by name.
void checkMeshLimits(VkPhysicalDevice physical, const Stage &stage, const std::array<std::uint32_t, 3> &groups);
// An object (task) entry against VK_EXT_mesh_shader's task limits: the threadgroups the draw
// dispatches, the invocations one of them runs and the payload's size.
void checkTaskLimits(VkPhysicalDevice physical, const Stage &stage, const std::array<std::uint32_t, 3> &groups);
// The payloads of an object entry and the mesh entry it launches must be the same struct; a
// mismatch is refused by name rather than read as bytes.
void checkPayloadMatch(const Stage &object, const Stage &mesh);
// The optional arguments (buffers, textures and samplers, by name) whose constant the host has set to
// false in `values`: the shader never reaches them in that specialization, so the host binds a
// placeholder of the right descriptor type for each. A constant left unset is not an absence -- the
// entry reads it, so functionConstantSpecialization refuses the pipeline for the missing value.
std::vector<std::string> absentOptionalResources(const Stage &stage, const std::vector<FunctionConstantValue> &values);
// Checks that a fragment entry reads a mesh entry's per-primitive outputs as per-primitive inputs.
// A fragment compiled without --mesh-entry declares ordinary varyings, which is right for a vertex
// stage and wrong here, and the mismatch is refused by name rather than drawn (docs/MESH_SHADERS.md).
void checkPerPrimitiveMatch(const Stage &mesh, const Stage &fragment);

// One color attachment as the host describes it: which index it fills and the format of its image
// (docs/PIPELINE_CONTRACT.md). An attachment is described by data rather than by a flag, so every
// host states the same three things and gets the same checks.
struct AttachmentDescription {
  std::uint32_t index = 0;
  VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
};

// Checks a set of attachment descriptions against a fragment entry's reflection: every described
// index is one the entry writes, no index is described twice, each format's numeric class matches
// that output's type, and the device advertises each format as a color attachment. Describing fewer
// attachments than the entry writes is allowed -- an index nobody describes is rgba8 -- so a host
// that wants today's behaviour passes nothing. Throws naming the attachment and the reason.
void checkAttachments(VkPhysicalDevice physical, const Stage &fragment,
                      const std::vector<AttachmentDescription> &attachments);
// The Vulkan sampler a constexpr sampler describes (clamp_to_zero: clamp to a transparent-black border; pixel
// coordinates are normalized in the shader, so the sampler stays normalized). Throws for a runtime sampler.
VkSamplerCreateInfo samplerCreateInfo(const Handle &sampler);
// Runtime LOD state. Bias uses exactly representable S4.6 values; no implicit
// rounding policy is imposed on host inputs. Check the selected device's limit.
struct SamplerLod { float bias = 0, min = 0, max = VK_LOD_CLAMP_NONE; };
void applySamplerLod(VkSamplerCreateInfo &info, const SamplerLod &lod, float maxSamplerLodBias);
// Requires an empty pNext chain. Installs an immutable, copy-safe reduction node
// only when all three filters are linear (Metal ignores reduction otherwise).
void applySamplerReduction(VkSamplerCreateInfo &info, const std::string &mode);
// Shared parser for the eight Metal comparison functions; throws for unknown names.
VkCompareOp samplerCompareOp(const std::string &name);
// The address table's words: one VkDeviceAddress per buffer of the stage, in `buffers` order, from `addressOf`.
std::vector<std::uint32_t> deviceAddressTable(const Stage &stage, const std::function<VkDeviceAddress(const Buffer &)> &addressOf);
// The residency word: bit i set when nullable[i] is bound; `unbound` names must be nullable.
std::uint32_t residencyMask(const Stage &stage, const std::vector<std::string> &unbound);
// Reads one stage's reflection ("vertex", "fragment" or "compute") and checks
// the stage/entry and the descriptor contract. Graphics buffers must be
// read-only; compute stages bind buffers only.
Stage readStage(const std::string &reflectionPath, const std::string &stage, const std::string &entry);
// Parses a host resource key: a Metal index (at most `limit`), BUFFER.ID for an
// argument-buffer member, or BUFFER[e].ID for a member of element e of an argument-buffer array.
std::uint32_t resourceKey(const std::string &text, unsigned limit);

// Acceleration structure builds (docs/RAY_TRACING.md). The library describes what Vulkan builds; the
// host owns the buffers, the device addresses and the command that builds. A primitive structure is
// a bottom-level build over float3 vertices, three per triangle, at a 12-byte stride; an instance
// structure is a top-level build over instance records, each placing that primitive structure under a
// transform with the mask a query's mask must meet and the custom index the shader reads as
// user_instance_id. A shader written against a primitive structure traverses a top-level structure
// holding one identity instance with every mask bit -- the default StructureInstance.
struct StructureInstance {
  std::array<float, 12> transform{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}; // Row-major 3x4: rotation columns, then translation.
  std::uint32_t mask = 0xFF, customIndex = 0;
  // The instance's shader-binding-table record offset (24 bits): with an intersection function
  // table, a candidate in this instance runs the function at slot recordOffset + geometry index.
  std::uint32_t recordOffset = 0;
};
// The words of an intersection function table: `capacity` slots, each the index of the function
// named for it in `slots` (by name, in slot order), and an index no function has (0xFFFFFFFF) for
// every slot left unnamed, so a candidate that lands there runs nothing. Throws for a name the
// stage's intersection_functions does not list or more names than slots.
std::vector<std::uint32_t> functionTableWords(const Stage &stage, const std::vector<std::string> &slots, std::uint32_t capacity = 16);
// The instance records of a top-level build, all over the primitive structure at `primitiveStructure`
// (its vkGetAccelerationStructureDeviceAddressKHR address) and none carrying instance flags: both
// faces hit unless a ray's own flags cull one, which is what an intersector's cull modes rely on.
// Throws for no instances, a mask beyond eight bits or a custom index beyond 24.
std::vector<VkAccelerationStructureInstanceKHR> instanceRecords(const std::vector<StructureInstance> &instances, VkDeviceAddress primitiveStructure);
// The geometry of a bottom-level build over `vertexCount` float3 vertices at `vertices`, opaque
// unless told otherwise (non-opaque triangles are the ones a stepped query sees as candidates).
// Throws unless the count is a positive multiple of three.
VkAccelerationStructureGeometryKHR triangleGeometry(VkDeviceAddress vertices, std::uint32_t vertexCount, bool opaque);
// The geometry of a bottom-level build over `boxCount` axis-aligned boxes at `boxes`, six floats each
// (min xyz, max xyz; VkAabbPositionsKHR at a 24-byte stride). A box is always a candidate of a stepped
// query, whatever its opacity, because only the shader can say where the ray meets what it bounds.
VkAccelerationStructureGeometryKHR boxGeometry(VkDeviceAddress boxes, std::uint32_t boxCount, bool opaque);
// The geometry of a top-level build over the records at `instances`.
VkAccelerationStructureGeometryKHR instanceGeometry(VkDeviceAddress instances);

// Descriptor layouts.
struct LayoutBinding { std::uint32_t set = 0, binding = 0; VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; std::uint32_t count = 1; };
std::vector<LayoutBinding> descriptorBindings(const Stage &stage);
std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings(const std::vector<LayoutBinding> &bindings, std::uint32_t set, VkShaderStageFlags stages);
std::vector<VkDescriptorPoolSize> poolSizes(const std::vector<LayoutBinding> &bindings);

// Host data packing into reflected element layouts (one element).
struct Words { std::string type; std::vector<std::uint32_t> words; };
std::vector<std::uint32_t> packWords(const Buffer &buffer, const std::map<std::string, Words> &values);
std::vector<std::uint32_t> packFloats(const Buffer &buffer, const std::map<std::string, std::vector<float>> &values);

// Argument-buffer encoding: member lookups and the inline block from per-id values.
const ArgumentMember *findMember(const ArgumentBuffer &argument, std::uint32_t id);
const ArgumentMember *findMember(const ArgumentBuffer &argument, const std::string &name);
std::vector<std::uint32_t> encodeInlineConstants(const ArgumentBuffer &argument, const Buffer &inlineBlock, const std::map<std::uint32_t, Words> &valuesById);

// Exact-grid dispatch plans (docs/COMPUTE_DISPATCH.md): full workgroups of the
// default size per axis, then one remainder group; regions are the product.
struct Region { std::array<std::uint32_t, 3> origin{}, groupOrigin{}, groups{}, local{}; };
struct DispatchPlan {
  std::array<std::uint32_t, 3> grid{}, logicalGroups{};
  std::vector<Region> regions;
};
DispatchPlan planExactGrid(std::array<std::uint32_t, 3> grid, std::array<std::uint32_t, 3> defaultLocal);
// The 64-byte push-constant block for one region (thread origin, group origin, threads per grid, threadgroups per grid).
std::array<std::uint32_t, 16> pushConstants(const DispatchPlan &plan, const Region &region);
// Specialization entries binding spec ids 0..2 to a uint32[3] of a region's workgroup size.
std::array<VkSpecializationMapEntry, 3> workgroupSizeSpecialization();

// ---------------------------------------------------------------- ray-tracing pipelines (docs/RAY_PIPELINES.md)
// The reflected stage names of the six ray stages, and their Vulkan stage bits.
bool isRayStage(const std::string &stage);
VkShaderStageFlagBits rayShaderStage(const std::string &stage); // Throws for a stage that is not a ray stage.
// A shader group over indices into the pipeline's stage list (VK_SHADER_UNUSED_KHR for an empty slot).
struct RayShaderGroup {
  VkRayTracingShaderGroupTypeKHR type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  std::uint32_t general = VK_SHADER_UNUSED_KHR, closestHit = VK_SHADER_UNUSED_KHR, anyHit = VK_SHADER_UNUSED_KHR, intersection = VK_SHADER_UNUSED_KHR;
};
// What each group is for, in group order: "raygen", "miss", "callable" or "hit". Throws, before any
// Vulkan call, for a stage that is not a ray stage, a slot holding the wrong stage, a general group
// without a raygen, miss or callable stage, a triangle group with an intersection stage or a
// procedural group without one, a hit group whose stages disagree on the incoming payload, a hit
// attribute a procedural group's intersection stage does not report or a triangle hit group that
// reads more than its two barycentric floats, an incoming payload or callable data no stage in the
// pipeline sends, and a pipeline without a ray-generation group.
std::vector<std::string> checkRayShaderGroups(const std::vector<Stage> &stages, const std::vector<RayShaderGroup> &groups);
// The one descriptor-set layout (set 0) every stage of a ray pipeline shares: each binding's type and
// count, and the stages that use it. Throws when two stages disagree about a binding.
struct RayDescriptorBinding { std::uint32_t binding = 0; VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; std::uint32_t count = 1; VkShaderStageFlags stages = 0; };
std::vector<RayDescriptorBinding> rayDescriptorBindings(const std::vector<Stage> &stages);
// Checks a pipeline's maxPipelineRayRecursionDepth: at least 1, at least 2 when a closest-hit or miss
// stage traces (a trace from there is a second level), and within the device's maxRayRecursionDepth.
// A deeper recursion is the application's own; exceeding the depth it creates the pipeline with is
// undefined in Vulkan, so the host must bound it.
void checkRayRecursionDepth(const std::vector<Stage> &stages, std::uint32_t depth, const VkPhysicalDeviceRayTracingPipelinePropertiesKHR &properties);
// A shader binding table: one raygen record, then the miss, hit and callable regions, each record the
// group's handle followed by its shader-record bytes. The regions' addresses are offsets from the
// table's start (add the buffer's device address, which must be shaderGroupBaseAlignment-aligned).
struct ShaderBindingRecord { std::uint32_t group = 0; std::vector<std::byte> data; };
struct ShaderBindingTable {
  VkStridedDeviceAddressRegionKHR raygen{}, miss{}, hit{}, callable{};
  std::vector<std::byte> bytes;
};
// Lays out and fills the table from the handles vkGetRayTracingShaderGroupHandlesKHR returned (one per
// group, shaderGroupHandleSize bytes each). Every record of a region shares its stride; a record's data
// must cover the shader_record every stage of its group reads. Throws for a record whose group has
// the wrong role for its region, oversized data, a stride beyond maxShaderGroupStride, or arithmetic
// that would overflow.
ShaderBindingTable buildShaderBindingTable(const std::vector<Stage> &stages, const std::vector<RayShaderGroup> &groups,
                                           const VkPhysicalDeviceRayTracingPipelinePropertiesKHR &properties,
                                           const std::vector<std::byte> &handles, const ShaderBindingRecord &raygen,
                                           const std::vector<ShaderBindingRecord> &miss, const std::vector<ShaderBindingRecord> &hit,
                                           const std::vector<ShaderBindingRecord> &callable);
// Pipeline libraries (VK_KHR_pipeline_library): every library and the pipeline linking them must
// declare the same VkRayTracingPipelineInterfaceCreateInfoKHR. This is the smallest one that holds the
// stages' interfaces as laid out (a float3 takes 16 bytes, as the compiler measures hit attributes):
// the largest payload or callable data, and the largest hit attributes, at least the 8 bytes a
// triangle's barycentrics take.
struct RayPipelineInterface { std::uint32_t maxPayloadSize = 0, maxHitAttributeSize = 0; };
RayPipelineInterface rayPipelineInterface(const std::vector<Stage> &stages);
// Deferred destruction for a renderer with frames in flight: an object replaced at frame N is still
// used by every frame submitted before N, so it is destroyed only once those frames have completed.
// retire() records the last frame that may use the object; completed(F) destroys, in retirement
// order, every object whose last use is at or before F; flush() destroys the rest (after the device
// is idle).
class RetirementQueue {
public:
  void retire(std::uint64_t lastUse, std::function<void()> destroy);
  void completed(std::uint64_t frame);
  void flush();
  std::size_t pending() const { return items.size(); }
private:
  std::vector<std::pair<std::uint64_t, std::function<void()>>> items;
};

} // namespace m2v::host
