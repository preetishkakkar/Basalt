#include "m2v_host.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace m2v::host {
namespace {

// ---------------------------------------------------------------- JSON
struct Parser {
  const std::string &text;
  std::size_t at = 0;
  [[noreturn]] void fail(const char *what) const { throw std::runtime_error(std::string("JSON: ") + what + " at offset " + std::to_string(at)); }
  void space() { while (at < text.size() && (text[at] == ' ' || text[at] == '\t' || text[at] == '\n' || text[at] == '\r')) ++at; }
  bool take(char c) { space(); if (at < text.size() && text[at] == c) { ++at; return true; } return false; }
  void expect(char c) { if (!take(c)) fail("unexpected character"); }
  static void appendUtf8(std::string &out, unsigned code) {
    if (code < 0x80) out += static_cast<char>(code);
    else if (code < 0x800) { out += static_cast<char>(0xc0 | (code >> 6)); out += static_cast<char>(0x80 | (code & 0x3f)); }
    else if (code < 0x10000) { out += static_cast<char>(0xe0 | (code >> 12)); out += static_cast<char>(0x80 | ((code >> 6) & 0x3f)); out += static_cast<char>(0x80 | (code & 0x3f)); }
    else { out += static_cast<char>(0xf0 | (code >> 18)); out += static_cast<char>(0x80 | ((code >> 12) & 0x3f)); out += static_cast<char>(0x80 | ((code >> 6) & 0x3f)); out += static_cast<char>(0x80 | (code & 0x3f)); }
  }
  unsigned hex4() {
    if (at + 4 > text.size()) fail("truncated escape");
    unsigned value = 0;
    for (unsigned i = 0; i < 4; ++i) {
      const char c = text[at++];
      value <<= 4;
      if (c >= '0' && c <= '9') value |= static_cast<unsigned>(c - '0');
      else if (c >= 'a' && c <= 'f') value |= static_cast<unsigned>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') value |= static_cast<unsigned>(c - 'A' + 10);
      else fail("invalid escape");
    }
    return value;
  }
  std::string string() {
    expect('"');
    std::string out;
    while (true) {
      if (at >= text.size()) fail("unterminated string");
      const char c = text[at++];
      if (c == '"') return out;
      if (c != '\\') { out += c; continue; }
      if (at >= text.size()) fail("unterminated escape");
      const char e = text[at++];
      switch (e) {
      case '"': out += '"'; break;
      case '\\': out += '\\'; break;
      case '/': out += '/'; break;
      case 'b': out += '\b'; break;
      case 'f': out += '\f'; break;
      case 'n': out += '\n'; break;
      case 'r': out += '\r'; break;
      case 't': out += '\t'; break;
      case 'u': {
        unsigned code = hex4();
        if (code >= 0xd800 && code < 0xdc00 && at + 6 <= text.size() && text[at] == '\\' && text[at + 1] == 'u') {
          at += 2;
          const unsigned low = hex4();
          code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
        }
        appendUtf8(out, code);
        break;
      }
      default: fail("invalid escape");
      }
    }
  }
  Json value() {
    space();
    if (at >= text.size()) fail("unexpected end");
    Json result;
    const char c = text[at];
    if (c == '{') {
      ++at;
      result.kind = Json::Kind::Object;
      if (take('}')) return result;
      while (true) {
        space();
        auto key = string();
        expect(':');
        result.object.emplace_back(std::move(key), value());
        if (take(',')) continue;
        expect('}');
        return result;
      }
    }
    if (c == '[') {
      ++at;
      result.kind = Json::Kind::Array;
      if (take(']')) return result;
      while (true) {
        result.array.push_back(value());
        if (take(',')) continue;
        expect(']');
        return result;
      }
    }
    if (c == '"') { result.kind = Json::Kind::String; result.string = string(); return result; }
    if (text.compare(at, 4, "true") == 0) { at += 4; result.kind = Json::Kind::Boolean; result.boolean = true; return result; }
    if (text.compare(at, 5, "false") == 0) { at += 5; result.kind = Json::Kind::Boolean; return result; }
    if (text.compare(at, 4, "null") == 0) { at += 4; return result; }
    const char *begin = text.c_str() + at;
    char *end = nullptr;
    const double number = std::strtod(begin, &end);
    if (end == begin) fail("unexpected token");
    at += static_cast<std::size_t>(end - begin);
    result.kind = Json::Kind::Number;
    result.number = number;
    return result;
  }
};

const Json &member(const Json &json, const std::string &key) {
  const auto *found = json.find(key);
  if (!found) throw std::runtime_error("missing reflection value " + key);
  return *found;
}

} // namespace

Json Json::parse(const std::string &text) {
  Parser parser{text};
  auto result = parser.value();
  parser.space();
  if (parser.at != text.size()) parser.fail("trailing characters");
  return result;
}

Json Json::parseFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot read " + path);
  std::stringstream buffer;
  buffer << file.rdbuf();
  return parse(buffer.str());
}

const Json *Json::find(const std::string &key) const {
  if (kind != Kind::Object) return nullptr;
  for (const auto &[name, value] : object) if (name == key) return &value;
  return nullptr;
}

const Json &Json::at(const std::string &key) const { return member(*this, key); }

std::string Json::text(const std::string &key) const {
  const auto &value = member(*this, key);
  if (value.kind != Kind::String) throw std::runtime_error("missing reflection string " + key);
  return value.string;
}

std::int64_t Json::integer(const std::string &key, std::int64_t maximum) const {
  const auto &value = member(*this, key);
  if (value.kind != Kind::Number || value.number < 0 || value.number > static_cast<double>(maximum) || value.number != static_cast<double>(static_cast<std::int64_t>(value.number)))
    throw std::runtime_error("invalid reflection value " + key);
  return static_cast<std::int64_t>(value.number);
}

bool Json::flag(const std::string &key) const {
  const auto &value = member(*this, key);
  if (value.kind != Kind::Boolean) throw std::runtime_error("missing reflection flag " + key);
  return value.boolean;
}

const std::vector<Json> &Json::list(const std::string &key) const {
  const auto &value = member(*this, key);
  if (value.kind != Kind::Array) throw std::runtime_error("missing reflection array " + key);
  return value.array;
}

// ---------------------------------------------------------------- reflection
std::uint32_t resourceKey(const std::string &text, unsigned limit) {
  const auto dot = text.find('.');
  try {
    if (dot == std::string::npos) {
      const auto index = std::stoul(text);
      if (index > limit) throw std::runtime_error("resource index " + text + " exceeds " + std::to_string(limit));
      return static_cast<std::uint32_t>(index);
    }
    auto head = text.substr(0, dot);
    unsigned long element = 0;
    if (const auto open = head.find('['); open != std::string::npos) {
      if (head.size() < open + 3 || head.back() != ']') throw std::runtime_error("argument-buffer array key " + text + " must be BUFFER[ELEMENT].ID");
      element = std::stoul(head.substr(open + 1, head.size() - open - 2));
      head = head.substr(0, open);
    }
    const auto buffer = std::stoul(head), id = std::stoul(text.substr(dot + 1));
    if (buffer > 30 || id > 255 || element > 255) throw std::runtime_error("argument-buffer key " + text + " must be BUFFER(0..30)[ELEMENT(0..255)].ID(0..255)");
    return 0x10000u | (static_cast<std::uint32_t>(buffer) << 8) | static_cast<std::uint32_t>(id) | (static_cast<std::uint32_t>(element) << 20);
  } catch (const std::invalid_argument &) {
    throw std::runtime_error("resource key must be an index or BUFFER.ID: " + text);
  }
}

Stage readStage(const std::string &reflectionPath, const std::string &stage, const std::string &entry) {
  const auto root = Json::parseFile(reflectionPath);
  if (!root.isObject()) throw std::runtime_error("expected reflection object: root");
  Stage result;
  result.stage = root.text("stage");
  result.entry = root.text("entry");
  if (result.stage != stage || result.entry != entry) throw std::runtime_error("reflection must describe " + stage + " entry " + entry);
  const auto &contract = root.at("binding_contract");
  if (!contract.isObject() || contract.text("policy") != "set_per_stage_class_offset") throw std::runtime_error("unsupported reflection binding contract");
  const bool compute = stage == "compute";
  // A mesh or object entry binds in set 0, as a vertex entry does; only the fragment stage uses set 1.
  const std::uint32_t expectedSet = stage == "fragment" ? 1u : stage == "object" ? 2u : 0u;
  // A post-tessellation vertex entry binds in set 0, as a vertex entry does.
  if (contract.integer("set", 2) != expectedSet) throw std::runtime_error("reflection descriptor set does not match the stage");
  result.set = expectedSet;
  const auto bufferBase = static_cast<std::uint32_t>(contract.integer("buffer_base", 255));
  const auto samplerBase = static_cast<std::uint32_t>(contract.integer("sampler_base", 255));
  const auto textureBase = static_cast<std::uint32_t>(contract.integer("texture_base", 255));
  std::set<std::uint32_t> bindings;
  auto argumentOf = [](const Json &entryObject, bool &argument, std::uint32_t &argumentBuffer, std::uint32_t &argumentId) {
    const auto *found = entryObject.find("argument");
    if (!found || !found->isObject()) return;
    argument = true;
    argumentBuffer = static_cast<std::uint32_t>(found->integer("buffer", 30));
    argumentId = static_cast<std::uint32_t>(found->integer("id", 255));
  };
  if (const auto *functions = root.find("intersection_functions"); functions && functions->isArray())
    for (const auto &entryObject : functions->array) {
      if (!entryObject.isObject()) throw std::runtime_error("reflection intersection_functions must be objects");
      IntersectionFunction function{entryObject.text("name"), entryObject.text("kind"), static_cast<std::uint32_t>(entryObject.integer("index", 15))};
      if (function.index != result.intersectionFunctions.size() || function.name.empty() || (function.kind != "bounding_box" && function.kind != "triangle"))
        throw std::runtime_error("reflected intersection functions are indexed in order with a known kind");
      result.intersectionFunctions.push_back(std::move(function));
    }
  if (const auto *functions = root.find("visible_functions"); functions && functions->isArray())
    for (const auto &entryObject : functions->array) {
      if (!entryObject.isObject()) throw std::runtime_error("reflection visible_functions must be objects");
      VisibleFunction function{entryObject.text("name"), entryObject.text("signature"), static_cast<std::uint32_t>(entryObject.integer("index", 1 << 20))};
      if (function.index != result.visibleFunctions.size() || function.name.empty() || function.signature.empty())
        throw std::runtime_error("reflected visible functions are indexed in order with a signature");
      result.visibleFunctions.push_back(std::move(function));
    }
  for (const auto &entryObject : root.list("buffers")) {
    if (!entryObject.isObject()) throw std::runtime_error("expected reflection object: buffer");
    Buffer buffer;
    buffer.name = entryObject.text("name");
    buffer.set = static_cast<std::uint32_t>(entryObject.integer("set", 2));
    if (const auto *fields = entryObject.find("fields"); fields && fields->isArray())
      for (const auto &fieldObject : fields->array)
        if (fieldObject.isObject())
          if (const auto *role = fieldObject.find("role"); role && role->kind == Json::Kind::String && role->string != "compute_pipeline_state" &&
              role->string != "render_pipeline_state" && role->string != "depth_stencil_state")
            throw std::runtime_error("reflected field role " + role->string + " is unknown");
    buffer.binding = static_cast<std::uint32_t>(entryObject.integer("binding", 1535) /* the function class reaches 1024 + 32 * 15 + 30 */);
    buffer.metalIndex = static_cast<std::uint32_t>(entryObject.integer("metal_index", 30));
    buffer.stride = static_cast<unsigned>(entryObject.integer("stride", 1 << 20));
    buffer.minimumBytes = static_cast<unsigned>(entryObject.integer("minimum_bytes", 1 << 20));
    buffer.reference = entryObject.flag("reference");
    buffer.readOnly = entryObject.flag("read_only");
    // Older reflections name neither; a buffer that names no direction is assumed to be read, which
    // is the conservative reading (it then needs host contents).
    buffer.reads = entryObject.find("reads") ? entryObject.flag("reads") : true;
    buffer.writes = entryObject.find("writes") ? entryObject.flag("writes") : !buffer.readOnly;
    const auto descriptor = entryObject.text("descriptor_type");
    buffer.uniform = descriptor == "uniform_buffer";
    if ((descriptor != "storage_buffer" && descriptor != "uniform_buffer") || entryObject.text("resource_class") != "buffer")
      throw std::runtime_error("unsupported reflected resource class");
    if (buffer.uniform && (!buffer.reference || !buffer.readOnly)) throw std::runtime_error("reflected uniform buffer " + buffer.name + " must be a read-only reference");
    argumentOf(entryObject, buffer.argument, buffer.argumentBuffer, buffer.argumentId);
    if (const auto *condition = entryObject.find("condition"); condition && condition->isObject())
      buffer.condition = Condition{condition->text("constant"), static_cast<std::uint32_t>(condition->integer("spec_id", 1 << 20))};
    if (entryObject.find("count")) buffer.count = static_cast<std::uint32_t>(entryObject.integer("count", 33));
    if (entryObject.find("elements")) {
      buffer.groups = static_cast<std::uint32_t>(entryObject.integer("elements", 32));
      buffer.idStride = static_cast<std::uint32_t>(entryObject.integer("id_stride", 255));
      if (!buffer.groups || !buffer.idStride || buffer.count % buffer.groups) throw std::runtime_error("reflected buffer " + buffer.name + " has an invalid element group");
    }
    if (!buffer.count || buffer.count > 256 || (buffer.count > 1 && (!buffer.argument || buffer.reference || buffer.uniform)))
      throw std::runtime_error("reflected buffer " + buffer.name + " has an invalid array count");
    if (const auto *function = entryObject.find("intersection_function"); function && function->kind == Json::Kind::String)
      buffer.intersectionFunction = function->string;
    if (entryObject.find("function_table")) buffer.functionTable = entryObject.flag("function_table");
    if (const auto *signature = entryObject.find("signature"); signature && signature->kind == Json::Kind::String) buffer.signature = signature->string;
    if (const auto *commands = entryObject.find("command_buffer"); commands && commands->isObject()) {
      buffer.commandBuffer = true;
      buffer.commandWords = static_cast<std::uint32_t>(commands->integer("record_words", kComputeCommandWords)); // The second argument is the largest admitted value.
      buffer.commandKind = commands->text("kind");
      if ((buffer.commandKind != "compute" && buffer.commandKind != "render" && buffer.commandKind != "unused") || buffer.commandWords != kComputeCommandWords ||
          !buffer.argument || buffer.readOnly || buffer.stride != 4)
        throw std::runtime_error("reflected command buffer " + buffer.name + " is not a read-write argument-buffer member of 64-word records");
    }
    if (!buffer.signature.empty() && !buffer.functionTable) throw std::runtime_error("reflected buffer " + buffer.name + " has a signature but is not a function table");
    std::optional<std::uint32_t> functionIndex;
    for (const auto &function : result.intersectionFunctions)
      if (function.name == buffer.intersectionFunction) functionIndex = function.index;
    if (!buffer.intersectionFunction.empty() && (!functionIndex || buffer.argument || buffer.reference || buffer.uniform || !buffer.readOnly))
      throw std::runtime_error("reflected buffer " + buffer.name + " names an intersection function the stage does not declare, or is not its read-only pointer");
    // A visible function table (one with a signature) may be an argument-buffer member; an
    // intersection table is an entry parameter only (docs/INDIRECT_CALLS.md).
    if (buffer.functionTable && ((buffer.argument && buffer.signature.empty()) || buffer.reference || buffer.uniform || !buffer.readOnly ||
                                 buffer.stride != 4 || !buffer.intersectionFunction.empty()))
      throw std::runtime_error("reflected function table " + buffer.name + " is a read-only buffer of u32 words");
    const bool bindingOk = functionIndex ? buffer.binding == 1024 + 32 * *functionIndex + buffer.metalIndex
                         : buffer.argument ? (buffer.binding >= 256 && buffer.binding < 512) : buffer.binding == bufferBase + buffer.metalIndex;
    if (buffer.set != expectedSet || !bindingOk || !buffer.stride || buffer.minimumBytes < buffer.stride || !bindings.insert(buffer.binding).second)
      throw std::runtime_error("reflected buffer " + buffer.name + " violates the descriptor contract");
    // A fragment entry may store into a buffer (it declares fragmentStoresAndAtomics for it); the
    // vertex and mesh stages bind read-only buffers only (docs/RESOURCE_BINDINGS.md).
    if (!buffer.readOnly && !compute && stage != "fragment")
      throw std::runtime_error(stage + " buffer " + buffer.name + " is not read-only");
    if (const auto *fields = entryObject.find("fields"); fields && fields->isArray())
      for (const auto &field : fields->array) {
        buffer.fields.push_back({field.text("name"), field.text("type"), static_cast<unsigned>(field.integer("offset", 1 << 20)), static_cast<unsigned>(field.integer("size", 1 << 16))}); // An array field may span kilobytes (a rate map layer, docs/RATE_MAPS.md).
        if (const auto *role = field.find("role"); role && role->kind == Json::Kind::String) buffer.fields.back().role = role->string; // A pipeline state (docs/INDIRECT_COMMANDS.md).
      }
    result.buffers.push_back(std::move(buffer));
  }
  auto handles = [&](const char *key, const char *resourceClass, const char *descriptorType, std::uint32_t base, unsigned limit, std::vector<Handle> &out) {
    const auto *list = root.find(key);
    if (!list || !list->isArray()) return;
    for (const auto &entryObject : list->array) {
      if (!entryObject.isObject()) throw std::runtime_error(std::string("expected reflection object: ") + key);
      Handle handle;
      handle.name = entryObject.text("name");
      handle.set = static_cast<std::uint32_t>(entryObject.integer("set", 2));
      handle.binding = static_cast<std::uint32_t>(entryObject.integer("binding", 1023));
      handle.metalIndex = static_cast<std::uint32_t>(entryObject.integer("metal_index", limit));
      argumentOf(entryObject, handle.argument, handle.argumentBuffer, handle.argumentId);
      if (const auto *condition = entryObject.find("condition"); condition && condition->isObject())
        handle.condition = Condition{condition->text("constant"), static_cast<std::uint32_t>(condition->integer("spec_id", 1 << 20))};
      if (entryObject.find("count")) handle.count = static_cast<std::uint32_t>(entryObject.integer("count", limit + 1));
      if (!handle.count || handle.count > 256 || (!handle.argument && handle.metalIndex + handle.count > limit + 1))
        throw std::runtime_error("reflected " + std::string(resourceClass) + " " + handle.name + " has an invalid array count");
      const std::uint32_t argumentBase = std::string(resourceClass) == "texture" ? 768u : 512u;
      const bool bindingOk = handle.argument ? (handle.binding >= argumentBase && handle.binding < argumentBase + 256) : handle.binding == base + handle.metalIndex;
      if (std::string(resourceClass) == "texture") {
        // Sampled textures (access sample) are sampled images; the other accesses are storage images.
        handle.access = entryObject.find("access") ? entryObject.text("access") : "sample";
        handle.component = entryObject.text("component");
        if (handle.component != "f32" && handle.component != "f16" && handle.component != "i16" && handle.component != "u16" && handle.component != "i32" && handle.component != "u32")
          throw std::runtime_error("reflected texture " + handle.name + " has an unknown component type");
        handle.dimension = entryObject.text("dimension");
        if (const auto *depth = entryObject.find("depth")) handle.depth = depth->boolean;
        if (const auto *ms = entryObject.find("multisampled")) handle.multisampled = ms->boolean;
        if (const auto *atomic = entryObject.find("atomic")) handle.atomic = atomic->boolean;
        if (const auto *format = entryObject.find("format"); format && format->kind == Json::Kind::String) handle.format = format->string;
        if (handle.atomic && handle.format != "r32uint" && handle.format != "r32sint") throw std::runtime_error("reflected atomic texture " + handle.name + " needs an r32uint or r32sint format");
        handle.view = entryObject.find("view") ? entryObject.text("view") : handle.dimension;
        if (const auto *queryOnly = entryObject.find("lod_query_only")) {
          if (queryOnly->kind != Json::Kind::Boolean) throw std::runtime_error("reflected lod_query_only must be boolean");
          handle.lodQueryOnly = queryOnly->boolean;
          if (handle.lodQueryOnly && handle.access != "sample") throw std::runtime_error("reflected lod_query_only requires sample access");
        }
        static const char *const kDimensions[] = {"1d", "1d_array", "2d", "2d_array", "3d", "cube", "cube_array", "buffer"};
        if (std::find(std::begin(kDimensions), std::end(kDimensions), handle.dimension) == std::end(kDimensions) ||
            std::find(std::begin(kDimensions), std::end(kDimensions), handle.view) == std::end(kDimensions))
          throw std::runtime_error("reflected texture " + handle.name + " has an unknown dimension");
        handle.storage = handle.access != "sample";
        if (handle.storage && handle.access != "read" && handle.access != "write" && handle.access != "read_write")
          throw std::runtime_error("reflected texture " + handle.name + " has an unknown access");
        // Depth fetches use sampled-image descriptors even with access::read.
        // Vulkan has no storage depth images; preserve the compiler's mapping.
        if (handle.dimension == "buffer" && handle.access == "read") handle.storage = false;
        if (handle.multisampled) {
          if ((handle.dimension != "2d" && handle.dimension != "2d_array") || handle.view != handle.dimension || handle.access != "read" || handle.atomic || handle.count != 1 || entryObject.integer("mip_levels", 16) != 1)
            throw std::runtime_error("reflected multisampled textures require single readable 2D images");
          handle.storage = false;
        }
        if (handle.dimension == "buffer" && (handle.view != "buffer" || handle.access == "sample" || handle.depth || handle.count != 1 || (stage != "compute" && handle.access != "read")))
          throw std::runtime_error("reflected texel buffers require single readable textures or writable kernel textures");
        if (handle.dimension == "buffer" && handle.atomic &&
            (handle.access != "read_write" || (handle.component != "u32" && handle.component != "i32") ||
             handle.format != (handle.component == "u32" ? "r32uint" : "r32sint")))
          throw std::runtime_error("reflected atomic texel buffers require matching R32 integer read_write textures");
        if (handle.dimension != "buffer" && handle.atomic &&
            (stage != "compute" || handle.count != 1 || handle.depth || handle.multisampled ||
             handle.access != "read_write" || (handle.component != "u32" && handle.component != "i32") ||
             handle.format != (handle.component == "u32" ? "r32uint" : "r32sint")))
          throw std::runtime_error("reflected atomic images require single matching R32 integer read_write kernel textures");
        if (handle.depth) {
          if (handle.access != "sample" && handle.access != "read")
            throw std::runtime_error("reflected depth textures are read-only");
          handle.storage = false;
        }
        if (const auto *bits = entryObject.find("format_features"); bits && bits->isArray())
          for (const auto &bit : bits->array) handle.formatFeatures.push_back(bit.string);
        if (const auto *paths = entryObject.find("software_sampling"); paths && paths->isArray())
          for (const auto &path : paths->array) handle.softwareSampling.push_back(path.string);
        if (handle.dimension == "buffer" && handle.atomic &&
            handle.formatFeatures != std::vector<std::string>{"storage_texel_buffer_atomic"})
          throw std::runtime_error("reflected atomic texel buffers require storage_texel_buffer_atomic");
        if (handle.dimension != "buffer" && handle.atomic &&
            handle.formatFeatures != std::vector<std::string>{"storage_image_atomic"})
          throw std::runtime_error("reflected atomic images require storage_image_atomic");
      }
      if (const auto *state = entryObject.find("constexpr"); state && state->isObject()) {
        Handle::ConstexprSampler description;
        description.magFilter = state->text("mag_filter");
        description.minFilter = state->text("min_filter");
        description.mipFilter = state->text("mip_filter");
        const auto &address = state->list("address");
        if (address.size() != 3) throw std::runtime_error("reflected constexpr sampler " + handle.name + " needs three address modes");
        for (std::size_t i = 0; i < 3; ++i) description.address[i] = address[i].string;
        const auto coordinateMode = state->text("coord");
        if (coordinateMode != "pixel" && coordinateMode != "normalized") throw std::runtime_error("unknown reflected sampler coordinate mode " + coordinateMode);
        description.pixelCoordinates = coordinateMode == "pixel";
        description.borderColor = state->text("border_color");
        description.maxAnisotropy = static_cast<std::uint32_t>(state->integer("max_anisotropy", 16));
        if (state->find("reduction")) description.reduction = state->text("reduction");
        if (state->find("compare")) description.compare = state->text("compare");
        if (const auto *clamp = state->find("lod_clamp"); clamp && clamp->isArray()) {
          if (clamp->array.size() != 2) throw std::runtime_error("reflected constexpr sampler " + handle.name + " has an invalid lod_clamp");
          description.lodClamp = std::pair{static_cast<float>(clamp->array[0].number), static_cast<float>(clamp->array[1].number)};
        }
        handle.constexprSampler = std::move(description);
        (void)samplerCreateInfo(handle); // Validates the enumerators.
      }
      if (const auto *comparison = entryObject.find("comparison")) handle.comparison = comparison->boolean;
      if (handle.constexprSampler && handle.comparison != (handle.constexprSampler->compare != "none"))
        throw std::runtime_error("reflected sampler " + handle.name + " disagrees about its comparison state");
      const std::string expectedType = handle.dimension == "buffer" ? (handle.storage ? "storage_texel_buffer" : "uniform_texel_buffer") : handle.storage ? "storage_image" : descriptorType;
      if (entryObject.text("resource_class") != resourceClass || entryObject.text("descriptor_type") != expectedType ||
          handle.set != expectedSet || !bindingOk || !bindings.insert(handle.binding).second)
        throw std::runtime_error("reflected " + std::string(resourceClass) + " " + handle.name + " violates the descriptor contract");
      if (std::string(resourceClass) == "texture" &&
          (entryObject.integer("mip_levels", 16) > 1 ||
           ((handle.dimension == "1d" || handle.dimension == "1d_array" || handle.dimension == "buffer") && entryObject.integer("mip_levels", 16) != 1) ||
           entryObject.text("coordinates") != (handle.storage || handle.multisampled || handle.dimension == "buffer" || compute ? "texel" : "normalized")))
        throw std::runtime_error("unsupported texture contract for " + handle.name);
      out.push_back(std::move(handle));
    }
  };
  handles("textures", "texture", "sampled_image", textureBase, 127, result.textures);
  handles("samplers", "sampler", "sampler", samplerBase, 15, result.samplers);
  if (const auto *structures = root.find("acceleration_structures"); structures && structures->isArray())
    for (const auto &entryObject : structures->array) {
      if (!entryObject.isObject()) throw std::runtime_error("reflection acceleration_structures must be objects");
      AccelerationStructure structure;
      structure.name = entryObject.text("name");
      structure.metalIndex = static_cast<std::uint32_t>(entryObject.integer("metal_index", 30));
      structure.set = static_cast<std::uint32_t>(entryObject.integer("set", 2));
      structure.binding = static_cast<std::uint32_t>(entryObject.integer("binding", 1023));
      const auto kind = entryObject.text("kind");
      if (kind != "primitive" && kind != "instance") throw std::runtime_error("reflected acceleration structure " + structure.name + " has an unknown kind");
      structure.instance = kind == "instance";
      if (entryObject.text("resource_class") != "acceleration_structure" || entryObject.text("descriptor_type") != "acceleration_structure" ||
          structure.set != expectedSet || !bindings.insert(structure.binding).second)
        throw std::runtime_error("reflected acceleration structure " + structure.name + " violates the descriptor contract");
      result.accelerationStructures.push_back(std::move(structure));
    }
  if (const auto *features = root.find("required_features"); features && features->isArray())
    for (const auto &feature : features->array) {
      if (feature.kind != Json::Kind::String) throw std::runtime_error("reflection required_features must be strings");
      result.requiredFeatures.push_back(feature.string);
  if (const auto *properties = root.find("required_properties"); properties && properties->isArray())
    for (const auto &property : properties->array) {
      if (property.kind != Json::Kind::String) throw std::runtime_error("reflection required_properties must be strings");
      result.requiredProperties.push_back(property.string);
    }
    }
  if (!result.accelerationStructures.empty())
    for (const char *needed : {"rayQuery", "accelerationStructure"})
      if (std::find(result.requiredFeatures.begin(), result.requiredFeatures.end(), needed) == result.requiredFeatures.end())
        throw std::runtime_error(std::string("a stage with acceleration structures must require ") + needed);
  if (const auto *queries = root.find("sample_queries"); queries && queries->isObject()) {
    result.sampleQueries.used = true;
    result.sampleQueries.positionsOffset = static_cast<std::uint32_t>(queries->integer("positions_offset", 4096));
    result.sampleQueries.countOffset = static_cast<std::uint32_t>(queries->integer("count_offset", 4096));
    result.sampleQueries.capacity = static_cast<std::uint32_t>(queries->integer("capacity", 64));
    if (!result.sampleQueries.capacity || result.sampleQueries.positionsOffset % 8)
      throw std::runtime_error("reflected sample queries need a capacity and an eight-byte aligned position array");
  }
  if (const auto *bounds = root.find("clamped_lod_bounds"); bounds && bounds->isArray())
    for (const auto &described : bounds->array) {
      if (!described.isObject()) throw std::runtime_error("reflection clamped_lod_bounds must be objects");
      Stage::ClampedLodBound bound;
      bound.name = described.text("name");
      bound.metalIndex = static_cast<std::uint32_t>(described.integer("metal_index", 15));
      bound.set = static_cast<std::uint32_t>(described.integer("set", 3));
      bound.binding = static_cast<std::uint32_t>(described.integer("binding", 1023));
      bound.offset = static_cast<std::uint32_t>(described.integer("offset", 1024));
      if (bound.offset % 4) throw std::runtime_error("a clamped LOD bounds offset must be a multiple of four");
      for (const auto &other : result.clampedLodBounds)
        if (other.offset == bound.offset || other.binding == bound.binding)
          throw std::runtime_error("reflection clamped_lod_bounds repeat a sampler or an offset");
      result.clampedLodBounds.push_back(bound);
    }
  if (const auto *lengths = root.find("function_table_lengths"); lengths && lengths->isArray())
    for (const auto &described : lengths->array) {
      if (!described.isObject()) throw std::runtime_error("reflection function_table_lengths must be objects");
      Stage::FunctionTableLength length;
      length.name = described.text("name");
      length.metalIndex = static_cast<std::uint32_t>(described.integer("metal_index", 1023));
      length.set = static_cast<std::uint32_t>(described.integer("set", 3));
      length.binding = static_cast<std::uint32_t>(described.integer("binding", 1023));
      length.offset = static_cast<std::uint32_t>(described.integer("offset", 1024));
      if (length.offset % 4) throw std::runtime_error("a function table length offset must be a multiple of four");
      for (const auto &other : result.functionTableLengths)
        if (other.offset == length.offset || other.binding == length.binding)
          throw std::runtime_error("reflection function_table_lengths repeat a table or an offset");
      result.functionTableLengths.push_back(length);
    }
  if (const auto *channels = root.find("software_border_channels"); channels && channels->isArray())
    for (const auto &described : channels->array) {
      if (!described.isObject()) throw std::runtime_error("reflection software_border_channels must be objects");
      Stage::TextureChannels texture;
      texture.name = described.text("name");
      texture.metalIndex = static_cast<std::uint32_t>(described.integer("metal_index", 1023));
      texture.set = static_cast<std::uint32_t>(described.integer("set", 3));
      texture.binding = static_cast<std::uint32_t>(described.integer("binding", 1023));
      texture.specId = static_cast<std::uint32_t>(described.integer("spec_id", 1 << 20));
      result.softwareBorderChannels.push_back(texture);
    }
  if (const auto *queried = root.find("lod_query_samplers"); queried && queried->isArray())
    for (const auto &described : queried->array) {
      if (!described.isObject()) throw std::runtime_error("reflection lod_query_samplers must be objects");
      Stage::LodQuerySampler sampler;
      sampler.name = described.text("name");
      sampler.metalIndex = static_cast<std::uint32_t>(described.integer("metal_index", 15));
      sampler.set = static_cast<std::uint32_t>(described.integer("set", 3));
      sampler.binding = static_cast<std::uint32_t>(described.integer("binding", 1023));
      sampler.count = static_cast<std::uint32_t>(described.integer("count", 256));
      if (!sampler.count) throw std::runtime_error("a LOD query sampler has at least one element");
      for (const auto &other : result.lodQuerySamplers)
        if (other.binding == sampler.binding) throw std::runtime_error("reflection lod_query_samplers repeat a sampler");
      result.lodQuerySamplers.push_back(sampler);
    }
  if (const auto *inputs = root.find("builtin_inputs"); inputs && inputs->isArray())
    for (const auto &input : inputs->array) {
      if (input.kind != Json::Kind::String) throw std::runtime_error("reflection builtin_inputs must be strings");
      result.builtinInputs.push_back(input.string);
    }
  if (const auto *outputs = root.find("builtin_outputs"); outputs && outputs->isArray())
    for (const auto &output : outputs->array) {
      if (output.kind != Json::Kind::String || (output.string != "point_size" && output.string != "clip_distance" &&
                                                output.string != "depth" && output.string != "render_target_array_index" &&
                                                output.string != "viewport_array_index" && output.string != "stencil" &&
                                                output.string != "sample_mask"))
        throw std::runtime_error("reflection builtin_outputs must name point_size, clip_distance, depth, render_target_array_index, viewport_array_index, stencil or sample_mask");
      result.builtinOutputs.push_back(output.string);
    }
  // One clip distance unless the reflection counts more; older reflections name the built-in alone.
  if (std::find(result.builtinOutputs.begin(), result.builtinOutputs.end(), "clip_distance") != result.builtinOutputs.end())
    result.clipDistances = root.find("clip_distances") ? static_cast<std::uint32_t>(root.integer("clip_distances", 64)) : 1;
  if (const auto *colors = root.find("color_outputs"); colors && colors->isArray())
    for (const auto &color : colors->array) {
      if (!color.isObject()) throw std::runtime_error("reflection color_outputs must be objects");
      const auto index = color.integer("index", 7);
      // The second blend source shares attachment zero's location instead of owning an attachment,
      // so it is a pipeline blend property rather than another target to allocate.
      if (color.find("source_index") && color.integer("source_index", 1) == 1) {
        if (index != 0) throw std::runtime_error("reflection places a second blend source outside attachment zero");
        if (result.dualSourceBlending) throw std::runtime_error("reflection repeats the second blend source");
        result.dualSourceBlending = true;
        continue;
      }
      if (std::find(result.colorOutputs.begin(), result.colorOutputs.end(), index) != result.colorOutputs.end())
        throw std::runtime_error("reflection color_outputs repeat an attachment index");
      result.colorOutputs.push_back(static_cast<std::uint32_t>(index));
      result.colorOutputTypes.push_back(color.text("type"));
      if (const auto *condition = color.find("condition"); condition && condition->isObject())
        result.conditionalOutputs.push_back({static_cast<std::uint32_t>(index), condition->text("constant"),
                                             static_cast<std::uint32_t>(condition->integer("spec_id", 1 << 20))});
    }
  if (const auto *meshBlock = root.find("mesh"); meshBlock && meshBlock->isObject()) {
    result.mesh.maxVertices = static_cast<std::uint32_t>(meshBlock->integer("max_vertices", 4096));
    result.mesh.maxPrimitives = static_cast<std::uint32_t>(meshBlock->integer("max_primitives", 4096));
    result.mesh.indicesPerPrimitive = static_cast<std::uint32_t>(meshBlock->integer("indices_per_primitive", 3));
    result.mesh.topology = meshBlock->text("topology");
    if (const auto *perPrimitive = meshBlock->find("per_primitive_varyings"); perPrimitive && perPrimitive->isArray())
      for (const auto &varying : perPrimitive->array)
        result.perPrimitiveVaryings.push_back(static_cast<std::uint32_t>(varying.integer("location", 255)));
  }
  // The stage's varyings: their locations, and which of them are per primitive.
  if (const auto *varyings = root.find("varyings"); varyings && varyings->isArray())
    for (const auto &varying : varyings->array) {
      const auto location = static_cast<std::uint32_t>(varying.integer("location", 255));
      result.varyingLocations.push_back(location);
      // The key is present only on a per-primitive input, so its absence is the ordinary case.
      if (const auto *flag = varying.find("per_primitive"); flag && flag->boolean)
        result.perPrimitiveVaryings.push_back(location);
    }
  if (const auto *block = root.find("patch"); block && block->isObject()) {
    result.patch.type = block->text("type");
    result.patch.partition = block->text("partition");
    result.patch.winding = block->text("winding");
    result.patch.controlPoints = static_cast<std::uint32_t>(block->integer("control_points", 32));
    result.patch.factorBufferBinding = static_cast<std::uint32_t>(block->integer("factor_buffer_binding", 255));
    if ((result.patch.type != "triangle" && result.patch.type != "quad") || !result.patch.controlPoints)
      throw std::runtime_error("reflected patch is invalid");
  }
  if (const auto *block = root.find("amplification"); block && block->isObject()) {
    if (block->text("views") != "multiview") throw std::runtime_error("unsupported reflection amplification mode");
    result.amplification.views = true;
    result.amplification.readsCount = block->flag("reads_count");
    result.amplification.viewCountSpecId = static_cast<std::uint32_t>(block->integer("view_count_spec_id", 65535));
  }
  if (const auto *payloadBlock = root.find("payload"); payloadBlock && payloadBlock->isObject()) {
    result.payload.name = payloadBlock->text("name");
    result.payload.size = static_cast<std::uint32_t>(payloadBlock->integer("size", 1 << 20));
    result.payload.alignment = static_cast<std::uint32_t>(payloadBlock->integer("alignment", 4));
    result.payload.readOnly = payloadBlock->flag("read_only");
    if (!result.payload.size || result.payload.size > 65536) throw std::runtime_error("reflected payload has an invalid size");
    for (const auto &field : payloadBlock->list("fields")) {
      if (!field.isObject()) throw std::runtime_error("reflection payload fields must be objects");
      result.payload.fields.push_back({field.text("name"), field.text("type"),
                                       static_cast<std::uint32_t>(field.integer("offset", 1 << 20))});
    }
    if (result.payload.fields.empty()) throw std::runtime_error("reflected payload has no fields");
  }
  if (const auto *constants = root.find("function_constants"); constants && constants->isArray())
    for (const auto &constant : constants->array) {
      if (!constant.isObject()) throw std::runtime_error("reflection function_constants must be objects");
      Stage::FunctionConstant declared;
      declared.name = constant.text("name");
      declared.type = constant.text("type");
      declared.index = static_cast<std::uint32_t>(constant.integer("index", 65535));
      declared.specId = static_cast<std::uint32_t>(constant.integer("spec_id", 0xffffffffu));
      declared.definedSpecId = static_cast<std::uint32_t>(constant.integer("defined_spec_id", 0xffffffffu));
      declared.required = constant.flag("required");
      declared.queried = constant.flag("queried");
      if (constant.find("lanes")) declared.lanes = static_cast<std::uint32_t>(constant.integer("lanes", 4));
      const auto element = declared.type.substr(0, declared.type.find('x'));
      const auto width = declared.type.find('x') == std::string::npos ? std::string("1") : declared.type.substr(declared.type.find('x') + 1);
      if ((element != "bool" && element != "i32" && element != "u32" && element != "f32") || width != std::to_string(declared.lanes) ||
          !declared.lanes || declared.lanes > 4)
        throw std::runtime_error("reflected function constant " + declared.name + " has an unsupported type " + declared.type);
      if (const auto *laneIds = constant.find("lane_spec_ids"); laneIds && laneIds->isArray())
        for (const auto &id : laneIds->array) declared.laneSpecIds.push_back(static_cast<std::uint32_t>(id.number));
      if (declared.lanes > 1 && declared.laneSpecIds.size() != declared.lanes)
        throw std::runtime_error("reflected vector function constant " + declared.name + " lists one spec id per lane");
      for (const auto &other : result.functionConstants)
        if (other.index == declared.index) throw std::runtime_error("reflection function_constants repeat an index");
      result.functionConstants.push_back(declared);
    }
  if (const auto *arguments = root.find("argument_buffers"); arguments && arguments->isArray())
    for (const auto &entryObject : arguments->array) {
      ArgumentBuffer argument;
      argument.name = entryObject.text("name");
      argument.metalIndex = static_cast<std::uint32_t>(entryObject.integer("metal_index", 30));
      if (entryObject.find("capacity")) argument.capacity = static_cast<std::uint32_t>(entryObject.integer("capacity", 256));
      if (!argument.capacity) throw std::runtime_error("reflected argument buffer " + argument.name + " has an invalid capacity");
      if (const auto *block = entryObject.find("inline_block"); block && block->isObject()) {
        argument.inlineBlock = true;
        argument.inlineSet = static_cast<std::uint32_t>(block->integer("set", 2));
        argument.inlineBinding = static_cast<std::uint32_t>(block->integer("binding", 1023));
      }
      for (const auto &memberObject : entryObject.list("members")) {
        ArgumentMember item;
        item.id = static_cast<std::uint32_t>(memberObject.integer("id", 65535));
        item.name = memberObject.text("name");
        item.kind = memberObject.text("kind");
        if (item.kind == "constant") {
          if (memberObject.find("offset")) {
            item.offset = static_cast<unsigned>(memberObject.integer("offset", 1 << 20));
            item.size = static_cast<unsigned>(memberObject.integer("size", 1 << 20));
            item.type = memberObject.text("type");
          }
        } else if (item.kind == "buffer" || item.kind == "texture" || item.kind == "sampler") {
          item.set = static_cast<std::uint32_t>(memberObject.integer("set", 2));
          item.binding = static_cast<std::uint32_t>(memberObject.integer("binding", 1023));
          item.descriptorType = memberObject.text("descriptor_type");
        } else throw std::runtime_error("unknown argument-buffer member kind " + item.kind);
        argument.members.push_back(std::move(item));
      }
      result.argumentBuffers.push_back(std::move(argument));
    }
  // Members of argument-buffer arrays: their descriptor count is the capacity times their own count.
  auto capacityOf = [&](bool argument, std::uint32_t argumentBuffer, std::uint32_t count, const std::string &name) {
    if (!argument) return 1u;
    for (const auto &entry : result.argumentBuffers)
      if (entry.metalIndex == argumentBuffer) {
        if (count % entry.capacity) throw std::runtime_error("reflected member " + name + " does not span its argument-buffer array");
        if (count > 256) throw std::runtime_error("reflected member " + name + " exceeds 256 descriptors");
        return entry.capacity;
      }
    throw std::runtime_error("reflected member " + name + " names an argument buffer the stage does not declare");
  };
  for (auto &buffer : result.buffers) buffer.capacity = capacityOf(buffer.argument, buffer.argumentBuffer, buffer.count, buffer.name);
  for (auto &texture : result.textures) texture.capacity = capacityOf(texture.argument, texture.argumentBuffer, texture.count, texture.name);
  for (auto &sampler : result.samplers) sampler.capacity = capacityOf(sampler.argument, sampler.argumentBuffer, sampler.count, sampler.name);
  if (compute) {
    const auto &localSize = root.list("local_size");
    const auto dispatch = root.text("dispatch");
    if (localSize.size() != 3 || (dispatch != "whole_workgroups" && dispatch != "exact_grid_plan"))
      throw std::runtime_error("compute reflection lacks a whole-workgroup or exact-grid local size");
    for (std::size_t i = 0; i < 3; ++i) {
      if (localSize[i].kind != Json::Kind::Number || localSize[i].number < 1 || localSize[i].number > 1024) throw std::runtime_error("invalid reflected local size");
      result.localSize[i] = static_cast<std::uint32_t>(localSize[i].number);
    }
    result.exactGrid = dispatch == "exact_grid_plan";
    if (result.exactGrid) {
      const auto &plan = root.at("dispatch_plan");
      DispatchPlanInfo info;
      info.defaultLocalSize = result.localSize;
      const auto &specIds = plan.at("workgroup_size").list("spec_ids");
      if (specIds.size() != 3) throw std::runtime_error("dispatch plan needs three workgroup-size spec ids");
      for (std::size_t i = 0; i < 3; ++i) info.specIds[i] = static_cast<std::uint32_t>(specIds[i].number);
      info.pushConstantBytes = static_cast<std::uint32_t>(plan.at("push_constants").integer("size", 128));
      if (info.pushConstantBytes != 64) throw std::runtime_error("dispatch plan push constants must be 64 bytes");
      result.plan = info;
    }
    if (const auto *memory = root.find("threadgroup_memory"); memory && memory->isObject())
      result.threadgroupBytes = static_cast<unsigned>(memory->integer("bytes", 1 << 20));
  }
  if (const auto *addresses = root.find("device_addresses"); addresses && addresses->isObject()) {
    result.deviceAddresses = true;
    result.deviceAddressBinding = static_cast<std::uint32_t>(addresses->integer("binding", 1023));
    const auto &order = addresses->list("buffers");
    if (addresses->integer("set", 2) != expectedSet || addresses->text("descriptor_type") != "storage_buffer" || order.size() != result.buffers.size() ||
        !bindings.insert(result.deviceAddressBinding).second)
      throw std::runtime_error("reflected device-address table violates the descriptor contract");
    for (std::size_t i = 0; i < order.size(); ++i)
      if (order[i].string != result.buffers[i].name) throw std::runtime_error("reflected device-address table does not follow the buffer order");
    if (std::find(result.requiredFeatures.begin(), result.requiredFeatures.end(), "bufferDeviceAddress") == result.requiredFeatures.end())
      throw std::runtime_error("reflected device-address table without the bufferDeviceAddress feature");
  }
  if (const auto *names = root.find("nullable"); names && names->isArray()) {
    // Nullable resources: buffer pointer parameters (compute) or single textures (fragment); the word follows the plan block.
    for (const auto &name : names->array) result.nullable.push_back(name.string);
    const auto &residency = root.at("residency");
    result.residencyOffset = static_cast<std::uint32_t>(residency.integer("push_constant_offset", 64));
    // A post-tessellation vertex entry runs in the vertex stage's place and owns that stage's word.
    const bool vertexLike = stage == "vertex" || stage == "post_tessellation_vertex";
    if (residency.integer("size", 8) != 4 || result.residencyOffset != (result.exactGrid ? 64u : vertexLike ? 4u : 0u) || result.nullable.size() > 32 ||
        residency.text("stage") != (vertexLike ? "vertex" : stage))
      throw std::runtime_error("reflected residency word violates the push-constant contract");
    for (const auto &name : result.nullable)
      if (std::none_of(result.buffers.begin(), result.buffers.end(), [&](const Buffer &b) { return b.name == name && !b.argument && !b.reference; }) &&
          std::none_of(result.textures.begin(), result.textures.end(), [&](const Handle &t) { return t.name == name && !t.argument && t.count == 1; }))
        throw std::runtime_error("reflected nullable resource " + name + " is not a buffer pointer or texture parameter");
  }
  return result;
}

// ---------------------------------------------------------------- acceleration structures
std::vector<ComputeCommand> decodeComputeCommands(const std::vector<std::uint32_t> &words) {
  if (words.size() % kComputeCommandWords) throw std::runtime_error("a command buffer holds whole 64-word records");
  std::vector<ComputeCommand> commands;
  for (std::size_t base = 0; base < words.size(); base += kComputeCommandWords) {
    const auto *w = words.data() + base;
    ComputeCommand command;
    if (w[0] > 2) throw std::runtime_error("command record " + std::to_string(base / kComputeCommandWords) + " has an unknown kind");
    command.kind = static_cast<ComputeCommand::Kind>(w[0]);
    command.pipeline = w[1];
    command.grid = {w[2], w[3], w[4]};
    command.threadsPerThreadgroup = {w[5], w[6], w[7]};
    command.barrier = w[8] != 0;
    const auto bindings = w[9];
    if (bindings > kCommandBufferBindings) throw std::runtime_error("command record binds more buffers than the format holds");
    for (std::uint32_t i = 0; i < bindings; ++i) {
      const auto *b = w + 10 + 5 * i;
      command.buffers.push_back({b[0], b[1], b[2], b[3], b[4]});
    }
    for (std::uint32_t i = 0; i < kCommandThreadgroupMemories; ++i) {
      const auto *m = w + 40 + 2 * i;
      if (m[1]) command.threadgroupMemory.push_back({m[0], m[1]});
    }
    commands.push_back(std::move(command));
  }
  return commands;
}

std::vector<RenderCommand> decodeRenderCommands(const std::vector<std::uint32_t> &words) {
  if (words.size() % kComputeCommandWords) throw std::runtime_error("a command buffer holds whole 64-word records");
  std::vector<RenderCommand> commands;
  for (std::size_t base = 0; base < words.size(); base += kComputeCommandWords) {
    const auto *w = words.data() + base;
    RenderCommand command;
    if (w[0] != 0 && w[0] != 3 && w[0] != 4) throw std::runtime_error("command record " + std::to_string(base / kComputeCommandWords) + " is not a render command");
    command.kind = static_cast<RenderCommand::Kind>(w[0]);
    command.pipeline = w[1];
    command.primitive = w[2];
    if (command.kind == RenderCommand::Kind::DrawIndexed) { command.indexCount = w[3]; command.baseVertex = w[4]; }
    else { command.vertexStart = w[3]; command.vertexCount = w[4]; }
    command.instanceCount = w[5];
    command.baseInstance = w[6];
    command.indexBuffer = {0, w[7], w[8], w[9], w[10]};
    auto bindings = [&](std::uint32_t countWord, std::uint32_t first, std::vector<ComputeCommand::BufferBinding> &out) {
      if (w[countWord] > kRenderCommandBindings) throw std::runtime_error("command record binds more buffers than the format holds");
      for (std::uint32_t i = 0; i < w[countWord]; ++i) {
        const auto *b = w + first + 5 * i;
        out.push_back({b[0], b[1], b[2], b[3], b[4]});
      }
    };
    bindings(11, 12, command.vertexBuffers);
    bindings(32, 33, command.fragmentBuffers);
    command.stateMask = w[53];
    command.cullMode = w[54];
    command.winding = w[55];
    command.fillMode = w[56];
    command.depthClipMode = w[57];
    std::memcpy(&command.depthBias, w + 58, 4);
    std::memcpy(&command.slopeScale, w + 59, 4);
    std::memcpy(&command.depthBiasClamp, w + 60, 4);
    command.depthStencil = w[61];
    command.barrier = w[62] != 0;
    commands.push_back(std::move(command));
  }
  return commands;
}

static void checkRateMapLayer(const RateMapLayer &layer) {
  if (!layer.screen[0] || !layer.screen[1]) throw std::runtime_error("a rate map layer needs a screen extent");
  if (layer.horizontal.empty() || layer.horizontal.size() > 16 || layer.vertical.empty() || layer.vertical.size() > 16)
    throw std::runtime_error("a rate map layer has 1 to 16 columns and rows");
  for (const auto quality : layer.horizontal) if (!(quality > 0.0f && quality <= 1.0f)) throw std::runtime_error("rate map qualities lie in (0, 1]");
  for (const auto quality : layer.vertical) if (!(quality > 0.0f && quality <= 1.0f)) throw std::runtime_error("rate map qualities lie in (0, 1]");
}

static float rateMapAxis(float coordinate, float extent, const std::vector<float> &qualities) {
  const float width = extent / static_cast<float>(qualities.size());
  float physical = 0.0f;
  for (std::size_t cell = 0; cell < qualities.size(); ++cell) {
    const float start = static_cast<float>(cell) * width;
    if (coordinate >= start + width) physical += width * qualities[cell];
    else if (coordinate > start) physical += (coordinate - start) * qualities[cell];
  }
  return physical;
}

std::array<float, 2> rateMapPhysical(const RateMapLayer &layer, float x, float y) {
  checkRateMapLayer(layer);
  return {rateMapAxis(x, static_cast<float>(layer.screen[0]), layer.horizontal), rateMapAxis(y, static_cast<float>(layer.screen[1]), layer.vertical)};
}

std::array<std::uint32_t, 2> rateMapPhysicalSize(const RateMapLayer &layer) {
  const auto physical = rateMapPhysical(layer, static_cast<float>(layer.screen[0]), static_cast<float>(layer.screen[1]));
  return {static_cast<std::uint32_t>(std::ceil(physical[0])), static_cast<std::uint32_t>(std::ceil(physical[1]))};
}

static std::vector<std::uint32_t> tensorStrides(const std::vector<std::uint32_t> &extents, const std::vector<std::uint32_t> &strides) {
  if (extents.empty() || extents.size() > 4) throw std::runtime_error("a tensor has rank 1 to 4");
  for (const auto extent : extents)
    if (extent == 0 || extent > 0x7fffffffu) throw std::runtime_error("a tensor extent is 1 to 2^31 - 1");
  if (!strides.empty() && strides.size() != extents.size()) throw std::runtime_error("tensor strides come one per extent");
  std::vector<std::uint32_t> dense{1};
  for (std::size_t r = 1; r < extents.size(); ++r) dense.push_back(dense[r - 1] * extents[r - 1]);
  const auto &chosen = strides.empty() ? dense : strides;
  for (const auto stride : chosen)
    if (stride == 0 || stride > 0x7fffffffu) throw std::runtime_error("a tensor stride is 1 to 2^31 - 1 elements");
  return chosen;
}

std::uint32_t tensorElementCount(const std::vector<std::uint32_t> &extents, const std::vector<std::uint32_t> &strides) {
  const auto chosen = tensorStrides(extents, strides);
  std::uint64_t last = 0;
  for (std::size_t r = 0; r < extents.size(); ++r) last += std::uint64_t(extents[r] - 1) * chosen[r];
  if (last + 1 > 0x7fffffffu) throw std::runtime_error("a tensor's elements must fit 2^31 - 1 elements");
  return static_cast<std::uint32_t>(last + 1);
}

std::vector<std::uint32_t> tensorHeaderWords(const std::vector<std::uint32_t> &extents, const std::vector<std::uint32_t> &strides) {
  const auto chosen = tensorStrides(extents, strides);
  std::vector<std::uint32_t> words(16, 0);
  words[0] = static_cast<std::uint32_t>(extents.size());
  for (std::size_t r = 0; r < extents.size(); ++r) {
    words[2 + r] = extents[r];
    words[6 + r] = chosen[r];
  }
  return words;
}

std::vector<std::uint32_t> rateMapWords(const std::vector<RateMapLayer> &layers) {
  if (layers.empty() || layers.size() > 4) throw std::runtime_error("a rate map has 1 to 4 layers");
  std::vector<std::uint32_t> words(4 + 8 + 8 + 4 + 4 + 64 + 64, 0);
  words[0] = static_cast<std::uint32_t>(layers.size());
  auto put = [&](std::size_t index, float value) { std::memcpy(&words[index], &value, 4); };
  for (std::size_t l = 0; l < layers.size(); ++l) {
    checkRateMapLayer(layers[l]);
    const auto physical = rateMapPhysicalSize(layers[l]);
    words[4 + 2 * l] = layers[l].screen[0];
    words[5 + 2 * l] = layers[l].screen[1];
    words[12 + 2 * l] = physical[0];
    words[13 + 2 * l] = physical[1];
    words[20 + l] = static_cast<std::uint32_t>(layers[l].horizontal.size());
    words[24 + l] = static_cast<std::uint32_t>(layers[l].vertical.size());
    for (std::size_t c = 0; c < layers[l].horizontal.size(); ++c) put(28 + 16 * l + c, layers[l].horizontal[c]);
    for (std::size_t r = 0; r < layers[l].vertical.size(); ++r) put(92 + 16 * l + r, layers[l].vertical[r]);
  }
  return words;
}

std::vector<std::uint8_t> shadingRateTexels(const RateMapLayer &layer, std::uint32_t texelSize) {
  checkRateMapLayer(layer);
  if (!texelSize) throw std::runtime_error("a shading rate texel covers at least one pixel");
  // The fragment size an axis quality asks for: 1 pixel above 0.75, 2 above 0.375, 4 below (the nearest of 1, 2, 4).
  auto fragmentLog2 = [](float quality) { return quality > 0.75f ? 0u : quality > 0.375f ? 1u : 2u; };
  const auto width = (layer.screen[0] + texelSize - 1) / texelSize, height = (layer.screen[1] + texelSize - 1) / texelSize;
  std::vector<std::uint8_t> texels(static_cast<std::size_t>(width) * height);
  for (std::uint32_t ty = 0; ty < height; ++ty)
    for (std::uint32_t tx = 0; tx < width; ++tx) {
      // The cell the texel's centre falls in.
      const auto column = std::min<std::size_t>(layer.horizontal.size() - 1, static_cast<std::size_t>((tx * texelSize + texelSize / 2) * layer.horizontal.size() / layer.screen[0]));
      const auto row = std::min<std::size_t>(layer.vertical.size() - 1, static_cast<std::size_t>((ty * texelSize + texelSize / 2) * layer.vertical.size() / layer.screen[1]));
      texels[static_cast<std::size_t>(ty) * width + tx] = static_cast<std::uint8_t>((fragmentLog2(layer.vertical[row]) << 2) | fragmentLog2(layer.horizontal[column]));
    }
  return texels;
}

std::vector<std::uint32_t> functionTableWords(const Stage &stage, const std::vector<std::string> &slots, std::uint32_t capacity) {
  if (!capacity) throw std::runtime_error("a function table has at least one slot");
  if (slots.size() > capacity) throw std::runtime_error("more functions named than the table has slots");
  std::vector<std::uint32_t> words(capacity, 0xFFFFFFFFu);
  for (std::size_t slot = 0; slot < slots.size(); ++slot) {
    if (slots[slot].empty()) continue;
    // A name is an intersection function's or a visible function's; the two lists never share one.
    const auto found = std::find_if(stage.intersectionFunctions.begin(), stage.intersectionFunctions.end(),
                                    [&](const IntersectionFunction &f) { return f.name == slots[slot]; });
    if (found != stage.intersectionFunctions.end()) { words[slot] = found->index; continue; }
    const auto visible = std::find_if(stage.visibleFunctions.begin(), stage.visibleFunctions.end(),
                                      [&](const VisibleFunction &f) { return f.name == slots[slot]; });
    if (visible == stage.visibleFunctions.end()) throw std::runtime_error("the stage declares no intersection or visible function named " + slots[slot]);
    words[slot] = visible->index;
  }
  return words;
}

std::vector<VkAccelerationStructureInstanceKHR> instanceRecords(const std::vector<StructureInstance> &instances, VkDeviceAddress primitiveStructure) {
  if (instances.empty()) throw std::runtime_error("an instance acceleration structure needs at least one instance");
  if (!primitiveStructure) throw std::runtime_error("an instance record needs the primitive structure's device address");
  std::vector<VkAccelerationStructureInstanceKHR> result;
  for (const auto &instance : instances) {
    if (instance.mask > 0xFF) throw std::runtime_error("an instance mask has eight bits");
    if (instance.customIndex > 0xFFFFFF) throw std::runtime_error("an instance custom index has 24 bits");
    if (instance.recordOffset > 0xFFFFFF) throw std::runtime_error("an instance record offset has 24 bits");
    VkAccelerationStructureInstanceKHR record{};
    for (unsigned row = 0; row < 3; ++row)
      for (unsigned column = 0; column < 4; ++column) record.transform.matrix[row][column] = instance.transform[row * 4 + column];
    record.instanceCustomIndex = instance.customIndex;
    record.mask = instance.mask;
    record.instanceShaderBindingTableRecordOffset = instance.recordOffset;
    record.flags = 0;
    record.accelerationStructureReference = primitiveStructure;
    result.push_back(record);
  }
  return result;
}

VkAccelerationStructureGeometryKHR triangleGeometry(VkDeviceAddress vertices, std::uint32_t vertexCount, bool opaque) {
  if (!vertexCount || vertexCount % 3) throw std::runtime_error("triangle geometry takes three vertices per triangle");
  if (!vertices) throw std::runtime_error("triangle geometry needs the vertex buffer's device address");
  VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  geometry.flags = opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;
  auto &triangles = geometry.geometry.triangles;
  triangles = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
  triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  triangles.vertexData.deviceAddress = vertices;
  triangles.vertexStride = 12;
  triangles.maxVertex = vertexCount - 1;
  triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
  return geometry;
}

VkAccelerationStructureGeometryKHR boxGeometry(VkDeviceAddress boxes, std::uint32_t boxCount, bool opaque) {
  if (!boxCount) throw std::runtime_error("box geometry takes at least one box");
  if (!boxes) throw std::runtime_error("box geometry needs the box buffer's device address");
  VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
  geometry.flags = opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;
  geometry.geometry.aabbs = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR};
  geometry.geometry.aabbs.data.deviceAddress = boxes;
  geometry.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);
  return geometry;
}

VkAccelerationStructureGeometryKHR instanceGeometry(VkDeviceAddress instances) {
  if (!instances) throw std::runtime_error("instance geometry needs the record buffer's device address");
  VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  geometry.geometry.instances = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
  geometry.geometry.instances.data.deviceAddress = instances;
  return geometry;
}

// ---------------------------------------------------------------- descriptor layouts
std::vector<LayoutBinding> descriptorBindings(const Stage &stage) {
  std::vector<LayoutBinding> result;
  for (const auto &buffer : stage.buffers) result.push_back({buffer.set, buffer.binding, buffer.descriptorType(), buffer.count});
  for (const auto &structure : stage.accelerationStructures)
    result.push_back({structure.set, structure.binding, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1});
  if (stage.deviceAddresses) result.push_back({stage.set, stage.deviceAddressBinding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1});
  for (const auto &texture : stage.textures)
    result.push_back({texture.set, texture.binding, texture.descriptorType(), texture.count});
  for (const auto &sampler : stage.samplers) result.push_back({sampler.set, sampler.binding, VK_DESCRIPTOR_TYPE_SAMPLER, sampler.count});
  std::sort(result.begin(), result.end(), [](const LayoutBinding &a, const LayoutBinding &b) { return std::tie(a.set, a.binding) < std::tie(b.set, b.binding); });
  return result;
}

std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings(const std::vector<LayoutBinding> &bindings, std::uint32_t set, VkShaderStageFlags stages) {
  std::vector<VkDescriptorSetLayoutBinding> result;
  for (const auto &binding : bindings)
    if (binding.set == set) result.push_back({binding.binding, binding.type, binding.count, stages, nullptr});
  return result;
}

std::vector<VkDescriptorPoolSize> poolSizes(const std::vector<LayoutBinding> &bindings) {
  std::vector<VkDescriptorPoolSize> result;
  for (const auto &binding : bindings) {
    auto found = std::find_if(result.begin(), result.end(), [&](const VkDescriptorPoolSize &size) { return size.type == binding.type; });
    if (found == result.end()) result.push_back({binding.type, binding.count});
    else found->descriptorCount += binding.count;
  }
  return result;
}

// ---------------------------------------------------------------- packing
std::vector<std::uint32_t> packWords(const Buffer &buffer, const std::map<std::string, Words> &values) {
  if (buffer.stride % 4) throw std::runtime_error("reflected stride is not word aligned");
  std::vector<std::uint32_t> words(buffer.stride / 4, 0);
  std::size_t used = 0;
  for (const auto &field : buffer.fields) {
    const auto found = values.find(field.name);
    if (found == values.end()) throw std::runtime_error("no host value for field " + field.name);
    if (field.type != found->second.type || found->second.words.size() * 4 != field.size || field.offset + field.size > buffer.stride)
      throw std::runtime_error("host value does not match reflected field " + field.name);
    std::memcpy(words.data() + field.offset / 4, found->second.words.data(), field.size);
    ++used;
  }
  if (used != values.size()) throw std::runtime_error("host supplied values for unknown fields");
  return words;
}

std::vector<std::uint32_t> packFloats(const Buffer &buffer, const std::map<std::string, std::vector<float>> &values) {
  std::map<std::string, Words> words;
  for (const auto &[name, floats] : values) {
    Words entry;
    const auto field = std::find_if(buffer.fields.begin(), buffer.fields.end(), [&](const Field &f) { return f.name == name; });
    entry.type = field == buffer.fields.end() ? "f32" : field->type;
    if (entry.type.rfind("f32", 0) != 0) throw std::runtime_error("host value does not match reflected field " + name);
    entry.words.resize(floats.size());
    std::memcpy(entry.words.data(), floats.data(), floats.size() * sizeof(float));
    words.emplace(name, std::move(entry));
  }
  return packWords(buffer, words);
}

// ---------------------------------------------------------------- argument buffers
const ArgumentMember *findMember(const ArgumentBuffer &argument, std::uint32_t id) {
  for (const auto &item : argument.members) if (item.id == id) return &item;
  return nullptr;
}

const ArgumentMember *findMember(const ArgumentBuffer &argument, const std::string &name) {
  for (const auto &item : argument.members) if (item.name == name) return &item;
  return nullptr;
}

std::vector<std::uint32_t> encodeInlineConstants(const ArgumentBuffer &argument, const Buffer &inlineBlock, const std::map<std::uint32_t, Words> &valuesById) {
  if (!argument.inlineBlock) throw std::runtime_error("argument buffer " + argument.name + " has no inline constants");
  if (inlineBlock.stride % 4) throw std::runtime_error("reflected stride is not word aligned");
  std::vector<std::uint32_t> words(inlineBlock.stride / 4, 0);
  std::size_t used = 0;
  for (const auto &item : argument.members) {
    if (item.kind != "constant") continue;
    const auto found = valuesById.find(item.id);
    if (found == valuesById.end()) throw std::runtime_error("no host value for argument member " + item.name);
    if (item.type != found->second.type || found->second.words.size() * 4 != item.size || item.offset + item.size > inlineBlock.stride)
      throw std::runtime_error("host value does not match argument member " + item.name);
    std::memcpy(words.data() + item.offset / 4, found->second.words.data(), item.size);
    ++used;
  }
  if (used != valuesById.size()) throw std::runtime_error("host supplied values for unknown argument members");
  return words;
}

VkCompareOp samplerCompareOp(const std::string &name) {
  static const std::pair<const char *, VkCompareOp> kCompareOps[] = {
    {"never", VK_COMPARE_OP_NEVER}, {"less", VK_COMPARE_OP_LESS}, {"less_equal", VK_COMPARE_OP_LESS_OR_EQUAL}, {"greater", VK_COMPARE_OP_GREATER},
    {"greater_equal", VK_COMPARE_OP_GREATER_OR_EQUAL}, {"equal", VK_COMPARE_OP_EQUAL}, {"not_equal", VK_COMPARE_OP_NOT_EQUAL}, {"always", VK_COMPARE_OP_ALWAYS}};
  const auto op = std::find_if(std::begin(kCompareOps), std::end(kCompareOps), [&](const auto &entry) { return name == entry.first; });
  if (op == std::end(kCompareOps)) throw std::runtime_error("unknown sampler compare function " + name);
  return op->second;
}

void applySamplerLod(VkSamplerCreateInfo &info, const SamplerLod &lod, float maxSamplerLodBias) {
  if (!std::isfinite(lod.bias) || !std::isfinite(lod.min) || !std::isfinite(lod.max))
    throw std::runtime_error("sampler LOD values must be finite");
  if (lod.bias < -16.0f || lod.bias > 1023.0f / 64.0f || std::trunc(lod.bias * 64.0f) != lod.bias * 64.0f)
    throw std::runtime_error("sampler LOD bias requires an exact S4.6 value in [-16, 15.984375]");
  if (lod.min < 0 || lod.max < lod.min)
    throw std::runtime_error("sampler LOD clamps require 0 <= min <= max");
  if (std::abs(lod.bias) > maxSamplerLodBias)
    throw std::runtime_error("sampler LOD bias exceeds maxSamplerLodBias");
  info.mipLodBias = lod.bias;
  info.minLod = lod.min;
  info.maxLod = lod.max;
}

void applySamplerReduction(VkSamplerCreateInfo &info, const std::string &mode) {
  if (mode != "weighted_average" && mode != "minimum" && mode != "maximum")
    throw std::runtime_error("unknown sampler reduction mode " + mode);
  if (info.pNext) throw std::runtime_error("sampler reduction requires an empty pNext chain");
  if (mode == "weighted_average" || info.magFilter != VK_FILTER_LINEAR || info.minFilter != VK_FILTER_LINEAR ||
      info.mipmapMode != VK_SAMPLER_MIPMAP_MODE_LINEAR) return;
  if (info.compareEnable || info.anisotropyEnable || info.unnormalizedCoordinates)
    throw std::runtime_error("min/max reduction requires normalized non-comparison sampling without anisotropy");
  static const VkSamplerReductionModeCreateInfo minimum{VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO, nullptr, VK_SAMPLER_REDUCTION_MODE_MIN};
  static const VkSamplerReductionModeCreateInfo maximum{VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO, nullptr, VK_SAMPLER_REDUCTION_MODE_MAX};
  info.pNext = mode == "minimum" ? &minimum : &maximum;
}

VkSamplerCreateInfo samplerCreateInfo(const Handle &sampler) {
  if (!sampler.constexprSampler) throw std::runtime_error("sampler " + sampler.name + " is a runtime sampler: the host chooses its state");
  const auto &state = *sampler.constexprSampler;
  auto clampAddress = [](const std::string &mode) { return mode == "clamp_to_zero" || mode == "clamp_to_edge" || mode == "clamp_to_border"; };
  if (state.pixelCoordinates && (state.magFilter != state.minFilter || state.mipFilter != "none" || state.compare != "none" ||
      !std::all_of(state.address.begin(), state.address.end(), clampAddress)))
    throw std::runtime_error("coord::pixel requires matching min/mag filters, mip_filter::none, compare_func::none and clamp address modes (MSL sampler contract)");
  auto filter = [&](const std::string &name) {
    if (name == "nearest") return VK_FILTER_NEAREST;
    if (name == "linear") return VK_FILTER_LINEAR;
    throw std::runtime_error("unknown sampler filter " + name);
  };
  auto address = [&](const std::string &name) {
    if (name == "clamp_to_edge") return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (name == "repeat") return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    if (name == "mirrored_repeat") return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    if (name == "clamp_to_zero" || name == "clamp_to_border") return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    throw std::runtime_error("unknown sampler address mode " + name);
  };
  VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  if (state.reduction != "weighted_average" && state.reduction != "minimum" && state.reduction != "maximum")
    throw std::runtime_error("unknown sampler reduction mode " + state.reduction);
  // Apple ignores reduction unless all three filters are linear. Static immutable
  // extension structures keep pNext valid when callers copy the returned info.
  if (state.reduction != "weighted_average" && state.magFilter == "linear" && state.minFilter == "linear" && state.mipFilter == "linear") {
    if (state.compare != "none" || state.maxAnisotropy > 1 || state.pixelCoordinates)
      throw std::runtime_error("min/max reduction requires normalized non-comparison sampling without anisotropy");
    static const VkSamplerReductionModeCreateInfo minimum{VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO, nullptr, VK_SAMPLER_REDUCTION_MODE_MIN};
    static const VkSamplerReductionModeCreateInfo maximum{VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO, nullptr, VK_SAMPLER_REDUCTION_MODE_MAX};
    info.pNext = state.reduction == "minimum" ? &minimum : &maximum;
  }
  info.magFilter = filter(state.magFilter);
  info.minFilter = filter(state.minFilter);
  if (state.mipFilter == "none") { info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST; info.maxLod = 0.25f; } // Level zero only.
  else if (state.mipFilter == "nearest") { info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST; info.maxLod = VK_LOD_CLAMP_NONE; }
  else if (state.mipFilter == "linear") { info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR; info.maxLod = VK_LOD_CLAMP_NONE; }
  else throw std::runtime_error("unknown sampler mip filter " + state.mipFilter);
  info.addressModeU = address(state.address[0]);
  info.addressModeV = address(state.address[1]);
  info.addressModeW = address(state.address[2]);
  // clamp_to_zero borders are transparent black whatever border_color says; clamp_to_border takes it.
  const bool zero = std::any_of(state.address.begin(), state.address.end(), [](const std::string &mode) { return mode == "clamp_to_zero"; });
  if (zero || state.borderColor == "transparent_black") info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
  else if (state.borderColor == "opaque_black") info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
  else if (state.borderColor == "opaque_white") info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
  else throw std::runtime_error("unknown sampler border color " + state.borderColor);
  if (state.maxAnisotropy < 1 || state.maxAnisotropy > 16) throw std::runtime_error("sampler anisotropy must be 1..16");
  if (state.compare != "none") {
    info.compareEnable = VK_TRUE;
    info.compareOp = samplerCompareOp(state.compare);
  }
  info.anisotropyEnable = state.maxAnisotropy > 1 ? VK_TRUE : VK_FALSE;
  info.maxAnisotropy = static_cast<float>(state.maxAnisotropy);
  if (state.lodClamp) {
    if (state.lodClamp->first < 0.0f || state.lodClamp->second < state.lodClamp->first) throw std::runtime_error("sampler lod_clamp must be ordered and non-negative");
    info.minLod = state.lodClamp->first;
    info.maxLod = std::min(info.maxLod, state.lodClamp->second);
  }
  return info;
}

std::vector<std::uint32_t> deviceAddressTable(const Stage &stage, const std::function<VkDeviceAddress(const Buffer &)> &addressOf) {
  if (!stage.deviceAddresses) throw std::runtime_error("the stage reads no device-address table");
  std::vector<std::uint32_t> words;
  for (const auto &buffer : stage.buffers) {
    const std::uint64_t address = addressOf(buffer);
    words.push_back(static_cast<std::uint32_t>(address & 0xffffffffu));
    words.push_back(static_cast<std::uint32_t>(address >> 32));
  }
  return words;
}

std::uint32_t residencyMask(const Stage &stage, const std::vector<std::string> &unbound) {
  std::uint32_t mask = 0;
  for (std::size_t i = 0; i < stage.nullable.size(); ++i)
    if (std::find(unbound.begin(), unbound.end(), stage.nullable[i]) == unbound.end()) mask |= 1u << i;
  for (const auto &name : unbound)
    if (std::find(stage.nullable.begin(), stage.nullable.end(), name) == stage.nullable.end())
      throw std::runtime_error("resource " + name + " is not nullable in this entry");
  return mask;
}

// ---------------------------------------------------------------- dispatch plans
DispatchPlan planExactGrid(std::array<std::uint32_t, 3> grid, std::array<std::uint32_t, 3> defaultLocal) {
  DispatchPlan plan;
  plan.grid = grid;
  struct Part { std::uint32_t origin, groupOrigin, groups, local; };
  std::array<std::vector<Part>, 3> parts;
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (!grid[axis] || !defaultLocal[axis]) throw std::runtime_error("exact grid and workgroup sizes must be positive on every axis");
    const auto full = grid[axis] / defaultLocal[axis], remainder = grid[axis] % defaultLocal[axis];
    plan.logicalGroups[axis] = full + (remainder ? 1u : 0u);
    if (full) parts[axis].push_back({0, 0, full, defaultLocal[axis]});
    if (remainder) parts[axis].push_back({full * defaultLocal[axis], full, 1, remainder});
  }
  for (const auto &x : parts[0])
    for (const auto &y : parts[1])
      for (const auto &z : parts[2])
        plan.regions.push_back({{x.origin, y.origin, z.origin}, {x.groupOrigin, y.groupOrigin, z.groupOrigin}, {x.groups, y.groups, z.groups}, {x.local, y.local, z.local}});
  return plan;
}

std::array<std::uint32_t, 16> pushConstants(const DispatchPlan &plan, const Region &region) {
  return {region.origin[0], region.origin[1], region.origin[2], 0,
          region.groupOrigin[0], region.groupOrigin[1], region.groupOrigin[2], 0,
          plan.grid[0], plan.grid[1], plan.grid[2], 0,
          plan.logicalGroups[0], plan.logicalGroups[1], plan.logicalGroups[2], 0};
}

bool supportsProperty(VkPhysicalDevice physical, const std::string &name) {
  VkPhysicalDeviceDescriptorIndexingProperties indexing{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES};
  VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceFloatControlsProperties floatControls{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
  indexing.pNext = &subgroup;
  subgroup.pNext = &floatControls;
  VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  properties.pNext = &indexing;
  vkGetPhysicalDeviceProperties2(physical, &properties);
  if (name == "quadDivergentImplicitLod") return indexing.quadDivergentImplicitLod == VK_TRUE;
  // A module holding a half declares RoundingModeRTE for its 16-bit conversions, which the device
  // has to support (docs/NUMERICS.md).
  if (name == "shaderRoundingModeRTEFloat16") return floatControls.shaderRoundingModeRTEFloat16 == VK_TRUE;
  // Quad operations, and whether the fragment stage may use them. Vulkan restricts quad operations
  // to the fragment and compute stages unless quadOperationsInAllStages is set, so a fragment entry
  // needs the class and the stage, and nothing more (docs/SUBGROUPS.md).
  if (name == "subgroupQuad") return (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT) != 0;
  if (name == "subgroupQuadInFragment")
    return (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT) != 0 &&
           (subgroup.supportedStages & VK_SHADER_STAGE_FRAGMENT_BIT) != 0;
  // SIMD-group matrices keep one matrix row per lane of the first eight (docs/SUBGROUPS.md).
  if (name == "subgroupWidth8") return subgroup.subgroupSize >= 8 && (subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT);
  // The SIMD-group operation classes (docs/SUBGROUPS.md): a kernel that uses one needs the device to
  // report the class for the compute stage, which Vulkan requires of every device for the basic class.
  const std::map<std::string, VkSubgroupFeatureFlags> classes{
    {"subgroupBasic", VK_SUBGROUP_FEATURE_BASIC_BIT}, {"subgroupVote", VK_SUBGROUP_FEATURE_VOTE_BIT},
    {"subgroupBallot", VK_SUBGROUP_FEATURE_BALLOT_BIT}, {"subgroupShuffle", VK_SUBGROUP_FEATURE_SHUFFLE_BIT},
    {"subgroupArithmetic", VK_SUBGROUP_FEATURE_ARITHMETIC_BIT}, {"subgroupClustered", VK_SUBGROUP_FEATURE_CLUSTERED_BIT},
    {"subgroupQuad", VK_SUBGROUP_FEATURE_QUAD_BIT}};
  if (const auto found = classes.find(name); found != classes.end())
    return (subgroup.supportedOperations & found->second) == found->second && (subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT);
  throw std::runtime_error("unknown required device property " + name);
}

void checkMeshLimits(VkPhysicalDevice physical, const Stage &stage, const std::array<std::uint32_t, 3> &groups) {
  if (!stage.mesh.maxVertices) throw std::runtime_error("the reflection does not describe a mesh entry");
#ifdef VK_EXT_mesh_shader
  VkPhysicalDeviceMeshShaderPropertiesEXT mesh{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
  VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  properties.pNext = &mesh;
  vkGetPhysicalDeviceProperties2(physical, &properties);
  if (stage.mesh.maxVertices > mesh.maxMeshOutputVertices || stage.mesh.maxPrimitives > mesh.maxMeshOutputPrimitives)
    throw std::runtime_error("the mesh entry declares more vertices or primitives than this device supports");
  const std::uint32_t limits[] = {mesh.maxMeshWorkGroupCount[0], mesh.maxMeshWorkGroupCount[1], mesh.maxMeshWorkGroupCount[2]};
  for (std::size_t axis = 0; axis < 3; ++axis)
    if (!groups[axis] || groups[axis] > limits[axis])
      throw std::runtime_error("the mesh draw dispatches more threadgroups than this device supports");
  if (std::uint64_t(groups[0]) * groups[1] * groups[2] > mesh.maxMeshWorkGroupTotalCount)
    throw std::runtime_error("the mesh draw dispatches more threadgroups than this device supports");
#else
  // Headers without VK_EXT_mesh_shader cannot query the limits; the device cannot offer the
  // extension to a build that does not know it either, so the pipeline is refused before this.
  (void)physical;
  for (const auto count : groups)
    if (!count) throw std::runtime_error("a mesh draw dispatches at least one threadgroup per axis");
#endif
}

void checkTaskLimits(VkPhysicalDevice physical, const Stage &stage, const std::array<std::uint32_t, 3> &groups) {
  if (stage.stage != "object") throw std::runtime_error("the reflection does not describe an object entry");
  if (stage.payload.fields.empty()) throw std::runtime_error("an object entry writes an object_data payload");
#ifdef VK_EXT_mesh_shader
  VkPhysicalDeviceMeshShaderPropertiesEXT mesh{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
  VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  properties.pNext = &mesh;
  vkGetPhysicalDeviceProperties2(physical, &properties);
  if (stage.payload.size > mesh.maxTaskPayloadSize)
    throw std::runtime_error("the object entry's payload is larger than this device supports");
  const std::uint32_t limits[] = {mesh.maxTaskWorkGroupCount[0], mesh.maxTaskWorkGroupCount[1], mesh.maxTaskWorkGroupCount[2]};
  for (std::size_t axis = 0; axis < 3; ++axis)
    if (!groups[axis] || groups[axis] > limits[axis])
      throw std::runtime_error("the mesh draw dispatches more object threadgroups than this device supports");
  if (std::uint64_t(groups[0]) * groups[1] * groups[2] > mesh.maxTaskWorkGroupTotalCount)
    throw std::runtime_error("the mesh draw dispatches more object threadgroups than this device supports");
  const std::uint64_t invocations = std::uint64_t(stage.localSize[0]) * stage.localSize[1] * stage.localSize[2];
  if (invocations > mesh.maxTaskWorkGroupInvocations)
    throw std::runtime_error("the object entry's threadgroup is larger than this device supports");
#else
  (void)physical;
  for (const auto count : groups)
    if (!count) throw std::runtime_error("a mesh draw dispatches at least one threadgroup per axis");
#endif
}

void checkPayloadMatch(const Stage &object, const Stage &mesh) {
  if (object.payload.fields.empty() || mesh.payload.fields.empty())
    throw std::runtime_error("an object entry and the mesh entry it launches both declare the payload");
  if (object.payload.size != mesh.payload.size || object.payload.fields.size() != mesh.payload.fields.size())
    throw std::runtime_error("the object and mesh payloads are different structs");
  for (std::size_t i = 0; i < object.payload.fields.size(); ++i)
    if (object.payload.fields[i].type != mesh.payload.fields[i].type ||
        object.payload.fields[i].offset != mesh.payload.fields[i].offset)
      throw std::runtime_error("the object and mesh payloads are different structs");
  if (object.payload.readOnly || !mesh.payload.readOnly)
    throw std::runtime_error("the object entry writes the payload and the mesh entry reads it");
}

std::vector<std::string> absentOptionalResources(const Stage &stage, const std::vector<FunctionConstantValue> &values) {
  // A constant is absent when the host set it to false; its source index is what the values carry.
  const auto absent = [&](const std::optional<Condition> &condition) {
    if (!condition) return false;
    for (const auto &declared : stage.functionConstants)
      if (declared.name == condition->constant)
        for (const auto &value : values)
          if (value.index == declared.index) return value.bytes[0] == std::byte{0};
    return false;
  };
  std::vector<std::string> names;
  for (const auto &buffer : stage.buffers) if (absent(buffer.condition)) names.push_back(buffer.name);
  for (const auto &texture : stage.textures) if (absent(texture.condition)) names.push_back(texture.name);
  for (const auto &sampler : stage.samplers) if (absent(sampler.condition)) names.push_back(sampler.name);
  return names;
}

namespace {
// The numeric class of a format, which is what has to match the fragment output's type: a float
// attachment takes a float output, an unsigned one an unsigned output, a signed one a signed
// output. Normalized and sRGB formats are float attachments, because the shader writes floats.
char formatClass(VkFormat format) {
  switch (format) {
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_R32_SFLOAT: return 'f';
    case VK_FORMAT_R32_UINT: return 'u';
    case VK_FORMAT_R32_SINT: return 'i';
    default: return '?';
  }
}
}  // namespace

void checkAttachments(VkPhysicalDevice physical, const Stage &fragment,
                      const std::vector<AttachmentDescription> &attachments) {
  for (std::size_t i = 0; i < attachments.size(); ++i) {
    const auto &described = attachments[i];
    const std::string where = "attachment " + std::to_string(described.index);
    for (std::size_t j = 0; j < i; ++j)
      if (attachments[j].index == described.index) throw std::runtime_error(where + " is described twice");
    const auto found = std::find(fragment.colorOutputs.begin(), fragment.colorOutputs.end(), described.index);
    if (found == fragment.colorOutputs.end())
      throw std::runtime_error(where + " is described but the fragment entry does not write it");
    const char wanted = formatClass(described.format);
    if (wanted == '?') throw std::runtime_error(where + " uses a format this profile does not admit");
    const std::size_t which = std::size_t(found - fragment.colorOutputs.begin());
    const std::string &type = which < fragment.colorOutputTypes.size() ? fragment.colorOutputTypes[which] : std::string();
    // An older reflection names no type; it cannot be checked, and saying so beats guessing.
    if (!type.empty() && type[0] != wanted)
      throw std::runtime_error(where + " has a format whose numeric class does not match the output type '" + type + "'");
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(physical, described.format, &properties);
    if (!(properties.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT))
      throw std::runtime_error(where + " uses a format this device does not advertise as a color attachment");
  }
}

void checkPerPrimitiveMatch(const Stage &mesh, const Stage &fragment) {
  const auto perPrimitive = [](const Stage &stage, std::uint32_t location) {
    return std::find(stage.perPrimitiveVaryings.begin(), stage.perPrimitiveVaryings.end(), location) !=
           stage.perPrimitiveVaryings.end();
  };
  for (const auto location : fragment.varyingLocations) {
    // A location the mesh entry does not write at all is the stage interface's own business; what
    // matters here is that the two stages agree about the ones it does.
    const bool written = std::find(mesh.varyingLocations.begin(), mesh.varyingLocations.end(), location) !=
                         mesh.varyingLocations.end() || perPrimitive(mesh, location);
    if (!written) continue;
    if (perPrimitive(mesh, location) && !perPrimitive(fragment, location))
      throw std::runtime_error("the mesh entry writes location " + std::to_string(location) +
                               " once per primitive and the fragment entry reads it as an ordinary varying "
                               "(compile the fragment with --mesh-entry)");
    if (!perPrimitive(mesh, location) && perPrimitive(fragment, location))
      throw std::runtime_error("the fragment entry reads location " + std::to_string(location) +
                               " as per-primitive data and the mesh entry writes it once per vertex");
  }
}

bool supportsDualSourceBlending(VkPhysicalDevice physical, std::uint32_t attachments) {
  VkPhysicalDeviceFeatures features{};
  vkGetPhysicalDeviceFeatures(physical, &features);
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(physical, &properties);
  return features.dualSrcBlend == VK_TRUE && attachments <= properties.limits.maxFragmentDualSrcAttachments;
}

std::vector<std::array<float, 2>> standardSamplePositions(VkSampleCountFlagBits samples) {
  // Vulkan's "Standard sample locations" table, as normalized offsets inside the pixel.
  switch (samples) {
  case VK_SAMPLE_COUNT_1_BIT:
    return {{{0.5f, 0.5f}}};
  case VK_SAMPLE_COUNT_2_BIT:
    return {{{0.75f, 0.75f}}, {{0.25f, 0.25f}}};
  case VK_SAMPLE_COUNT_4_BIT:
    return {{{0.375f, 0.125f}}, {{0.875f, 0.375f}}, {{0.125f, 0.625f}}, {{0.625f, 0.875f}}};
  case VK_SAMPLE_COUNT_8_BIT:
    return {{{0.5625f, 0.3125f}}, {{0.4375f, 0.6875f}}, {{0.8125f, 0.5625f}}, {{0.3125f, 0.1875f}},
            {{0.1875f, 0.8125f}}, {{0.0625f, 0.4375f}}, {{0.6875f, 0.9375f}}, {{0.9375f, 0.0625f}}};
  case VK_SAMPLE_COUNT_16_BIT:
    return {{{0.5625f, 0.5625f}}, {{0.4375f, 0.3125f}}, {{0.3125f, 0.625f}}, {{0.75f, 0.4375f}},
            {{0.1875f, 0.375f}}, {{0.625f, 0.8125f}}, {{0.8125f, 0.6875f}}, {{0.6875f, 0.1875f}},
            {{0.375f, 0.875f}}, {{0.5f, 0.0625f}}, {{0.25f, 0.125f}}, {{0.125f, 0.75f}},
            {{0.0f, 0.5f}}, {{0.9375f, 0.25f}}, {{0.875f, 0.9375f}}, {{0.0625f, 0.0f}}};
  default:
    throw std::runtime_error("no standard sample locations are defined for this sample count");
  }
}

std::uint32_t textureChannelMask(VkFormat format) {
  // Every format the runner's --image table and the demo's uploads can bind, by channel count. A
  // format outside the table is refused rather than guessed: a wrong mask reads a wrong border.
  switch (format) {
  case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R8G8B8A8_SINT:
  case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB: case VK_FORMAT_A8B8G8R8_UNORM_PACK32: case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
  case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32: case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_A2R10G10B10_UINT_PACK32:
  case VK_FORMAT_R4G4B4A4_UNORM_PACK16: case VK_FORMAT_B4G4R4A4_UNORM_PACK16: case VK_FORMAT_R5G5B5A1_UNORM_PACK16: case VK_FORMAT_B5G5R5A1_UNORM_PACK16:
  case VK_FORMAT_A1R5G5B5_UNORM_PACK16:
  case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16B16A16_SNORM: case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UINT:
  case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT:
  case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: case VK_FORMAT_BC2_UNORM_BLOCK: case VK_FORMAT_BC2_SRGB_BLOCK:
  case VK_FORMAT_BC3_UNORM_BLOCK: case VK_FORMAT_BC3_SRGB_BLOCK: case VK_FORMAT_BC7_UNORM_BLOCK: case VK_FORMAT_BC7_SRGB_BLOCK:
  case VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK: case VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK: case VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK:
  case VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK:
  case VK_FORMAT_ASTC_4x4_UNORM_BLOCK: case VK_FORMAT_ASTC_4x4_SRGB_BLOCK: case VK_FORMAT_ASTC_5x4_UNORM_BLOCK: case VK_FORMAT_ASTC_5x4_SRGB_BLOCK:
  case VK_FORMAT_ASTC_5x5_UNORM_BLOCK: case VK_FORMAT_ASTC_5x5_SRGB_BLOCK: case VK_FORMAT_ASTC_6x5_UNORM_BLOCK: case VK_FORMAT_ASTC_6x5_SRGB_BLOCK:
  case VK_FORMAT_ASTC_6x6_UNORM_BLOCK: case VK_FORMAT_ASTC_6x6_SRGB_BLOCK: case VK_FORMAT_ASTC_8x5_UNORM_BLOCK: case VK_FORMAT_ASTC_8x5_SRGB_BLOCK:
  case VK_FORMAT_ASTC_8x6_UNORM_BLOCK: case VK_FORMAT_ASTC_8x6_SRGB_BLOCK: case VK_FORMAT_ASTC_8x8_UNORM_BLOCK: case VK_FORMAT_ASTC_8x8_SRGB_BLOCK:
  case VK_FORMAT_ASTC_10x5_UNORM_BLOCK: case VK_FORMAT_ASTC_10x5_SRGB_BLOCK: case VK_FORMAT_ASTC_10x6_UNORM_BLOCK: case VK_FORMAT_ASTC_10x6_SRGB_BLOCK:
  case VK_FORMAT_ASTC_10x8_UNORM_BLOCK: case VK_FORMAT_ASTC_10x8_SRGB_BLOCK: case VK_FORMAT_ASTC_10x10_UNORM_BLOCK: case VK_FORMAT_ASTC_10x10_SRGB_BLOCK:
  case VK_FORMAT_ASTC_12x10_UNORM_BLOCK: case VK_FORMAT_ASTC_12x10_SRGB_BLOCK: case VK_FORMAT_ASTC_12x12_UNORM_BLOCK: case VK_FORMAT_ASTC_12x12_SRGB_BLOCK:
    return 15;
  case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: case VK_FORMAT_R5G6B5_UNORM_PACK16: case VK_FORMAT_B5G6R5_UNORM_PACK16:
  case VK_FORMAT_R8G8B8_UNORM: case VK_FORMAT_R8G8B8_SRGB: case VK_FORMAT_R16G16B16_SFLOAT: case VK_FORMAT_R32G32B32_SFLOAT:
  case VK_FORMAT_BC1_RGB_UNORM_BLOCK: case VK_FORMAT_BC1_RGB_SRGB_BLOCK: case VK_FORMAT_BC6H_UFLOAT_BLOCK: case VK_FORMAT_BC6H_SFLOAT_BLOCK:
  case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK: case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:
    return 7;
  case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R8G8_SNORM: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R16G16_UNORM:
  case VK_FORMAT_R16G16_SNORM: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R32G32_SFLOAT:
  case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32_SINT:
  case VK_FORMAT_BC5_UNORM_BLOCK: case VK_FORMAT_BC5_SNORM_BLOCK: case VK_FORMAT_EAC_R11G11_UNORM_BLOCK: case VK_FORMAT_EAC_R11G11_SNORM_BLOCK:
    return 3;
  case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_SNORM: case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT: case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16_SNORM:
  case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT:
  case VK_FORMAT_D16_UNORM: case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_X8_D24_UNORM_PACK32: case VK_FORMAT_D24_UNORM_S8_UINT: case VK_FORMAT_D32_SFLOAT_S8_UINT:
  case VK_FORMAT_BC4_UNORM_BLOCK: case VK_FORMAT_BC4_SNORM_BLOCK: case VK_FORMAT_EAC_R11_UNORM_BLOCK: case VK_FORMAT_EAC_R11_SNORM_BLOCK:
    return 1;
  default:
    throw std::runtime_error("the channel count of texture format " + std::to_string(format) + " is not known to the host");
  }
}

void specializeTextureChannels(Specialization &specialization, const Stage &stage,
                               const std::function<std::optional<VkFormat>(std::uint32_t set, std::uint32_t binding)> &format) {
  for (const auto &texture : stage.softwareBorderChannels) {
    const auto bound = format(texture.set, texture.binding);
    const std::uint32_t mask = bound ? textureChannelMask(*bound) : 15u;
    // Replace an entry a previous call made, so binding twice does not send the constant twice.
    const auto existing = std::find_if(specialization.entries.begin(), specialization.entries.end(),
                                       [&](const VkSpecializationMapEntry &entry) { return entry.constantID == texture.specId; });
    if (existing != specialization.entries.end()) {
      std::memcpy(specialization.data.data() + existing->offset, &mask, sizeof mask);
      continue;
    }
    specialization.entries.push_back({texture.specId, static_cast<std::uint32_t>(specialization.data.size()), sizeof mask});
    const auto *bytes = reinterpret_cast<const std::byte *>(&mask);
    specialization.data.insert(specialization.data.end(), bytes, bytes + sizeof mask);
  }
}

std::array<float, 3> clampedLodBounds(const VkSamplerCreateInfo &info) {
  if (info.compareEnable) throw std::runtime_error("a comparison sampler cannot supply clamped LOD bounds");
  if (info.unnormalizedCoordinates) throw std::runtime_error("an unnormalized sampler cannot supply clamped LOD bounds");
  for (const auto *next = static_cast<const VkBaseInStructure *>(info.pNext); next; next = next->pNext)
    if (next->sType == VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO &&
        reinterpret_cast<const VkSamplerReductionModeCreateInfo *>(next)->reductionMode != VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE)
      throw std::runtime_error("a min/max reduction sampler cannot supply clamped LOD bounds");
  return {info.minLod, info.maxLod, info.anisotropyEnable ? info.maxAnisotropy : 1.0f};
}

void checkLodQuerySampler(const VkSamplerCreateInfo &info) {
  if (info.compareEnable) throw std::runtime_error("an unclamped LOD query does not admit a comparison sampler");
  if (info.unnormalizedCoordinates) throw std::runtime_error("an unclamped LOD query does not admit an unnormalized sampler");
  if (info.anisotropyEnable && info.maxAnisotropy > 1.0f) throw std::runtime_error("an unclamped LOD query does not admit an anisotropic sampler");
  for (const auto *next = static_cast<const VkBaseInStructure *>(info.pNext); next; next = next->pNext)
    if (next->sType == VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO &&
        reinterpret_cast<const VkSamplerReductionModeCreateInfo *>(next)->reductionMode != VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE)
      throw std::runtime_error("an unclamped LOD query does not admit a min/max reduction sampler");
}

FunctionConstantValue parseFunctionConstant(const Stage &stage, const std::string &text) {
  const auto colon = text.find(':');
  if (colon == std::string::npos || colon == 0 || colon + 1 == text.size())
    throw std::runtime_error("a function constant is INDEX:VALUE, for example 0:1.5");
  const auto index = std::stoul(text.substr(0, colon));
  const auto spelling = text.substr(colon + 1);
  const auto *declared = static_cast<const Stage::FunctionConstant *>(nullptr);
  for (const auto &constant : stage.functionConstants) if (constant.index == index) declared = &constant;
  if (!declared) throw std::runtime_error("the entry declares no function constant with index " + std::to_string(index));
  FunctionConstantValue value{static_cast<std::uint32_t>(index), {}};
  const auto badValue = [&] {
    return std::runtime_error("function constant " + declared->name + " takes a " + declared->type + " value, not '" + spelling + "'");
  };
  // One value per lane, comma separated for a vector.
  std::vector<std::string> parts;
  for (std::size_t at = 0;;) {
    const auto comma = spelling.find(',', at);
    parts.push_back(spelling.substr(at, comma == std::string::npos ? std::string::npos : comma - at));
    if (comma == std::string::npos) break;
    at = comma + 1;
  }
  if (parts.size() != declared->lanes) throw badValue();
  const auto element = declared->type.substr(0, declared->type.find('x'));
  for (const auto &part : parts) {
    std::size_t used = 0;
    std::uint32_t word = 0;
    try {
      if (element == "f32") {
        const float parsed = std::stof(part, &used);
        std::memcpy(&word, &parsed, 4);
      } else if (element == "bool") {
        // A VkBool32: the shader's OpSpecConstantTrue/False takes a full word.
        word = part == "true" ? 1u : part == "false" ? 0u : static_cast<std::uint32_t>(std::stoul(part, &used, 0) != 0);
        if (part == "true" || part == "false") used = part.size();
      } else if (element == "i32") {
        const std::int32_t parsed = static_cast<std::int32_t>(std::stol(part, &used, 0));
        std::memcpy(&word, &parsed, 4);
      } else {
        word = static_cast<std::uint32_t>(std::stoul(part, &used, 0));
      }
    } catch (const std::invalid_argument &) {
      throw badValue();
    } catch (const std::out_of_range &) {
      throw badValue();
    }
    if (used != part.size()) throw badValue();
    const auto *bytes = reinterpret_cast<const std::byte *>(&word);
    value.bytes.insert(value.bytes.end(), bytes, bytes + 4);
  }
  return value;
}

Specialization functionConstantSpecialization(const Stage &stage, const std::vector<FunctionConstantValue> &values) {
  Specialization specialization;
  for (const auto &constant : stage.functionConstants) {
    const auto *supplied = static_cast<const FunctionConstantValue *>(nullptr);
    for (const auto &value : values) if (value.index == constant.index) supplied = &value;
    if (!supplied && constant.required)
      throw std::runtime_error("function constant " + constant.name + " (index " + std::to_string(constant.index) +
                               ") is read by " + stage.entry + " and needs a value");
    // The flag first, then the value, so a stage with no supplied values still answers every query.
    const std::uint32_t defined = supplied ? 1u : 0u;
    specialization.entries.push_back({constant.definedSpecId, static_cast<std::uint32_t>(specialization.data.size()), 4});
    const auto *definedBytes = reinterpret_cast<const std::byte *>(&defined);
    specialization.data.insert(specialization.data.end(), definedBytes, definedBytes + 4);
    if (!supplied) continue;
    if (supplied->bytes.size() != 4 * constant.lanes)
      throw std::runtime_error("function constant " + constant.name + " takes " + std::to_string(constant.lanes) + " lane values");
    // A scalar at its spec id; a vector one entry per lane at the lane ids.
    for (std::uint32_t lane = 0; lane < constant.lanes; ++lane) {
      const auto id = constant.lanes == 1 ? constant.specId : constant.laneSpecIds.at(lane);
      specialization.entries.push_back({id, static_cast<std::uint32_t>(specialization.data.size()), 4});
      specialization.data.insert(specialization.data.end(), supplied->bytes.begin() + 4 * lane, supplied->bytes.begin() + 4 * lane + 4);
    }
  }
  return specialization;
}

std::array<VkSpecializationMapEntry, 3> workgroupSizeSpecialization() {
  return {VkSpecializationMapEntry{0, 0, 4}, VkSpecializationMapEntry{1, 4, 4}, VkSpecializationMapEntry{2, 8, 4}};
}

} // namespace m2v::host
