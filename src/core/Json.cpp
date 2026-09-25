#include "core/Json.h"

#include "core/Log.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace basalt {
namespace {

class Parser {
public:
  explicit Parser(std::string_view source) : text(source) {}

  Json document() {
    Json value = parseValue(0);
    skipSpace();
    if (position != text.size()) fail("trailing characters");
    return value;
  }

private:
  // Deep enough for any reflection document, shallow enough to never exhaust the stack.
  static constexpr int maximumDepth = 256;

  std::string_view text;
  std::size_t position = 0;

  [[noreturn]] void fail(const std::string &what) const {
    throw Error("JSON: " + what + " at offset " + std::to_string(position));
  }

  void skipSpace() {
    while (position < text.size() &&
           (text[position] == ' ' || text[position] == '\t' || text[position] == '\n' || text[position] == '\r'))
      ++position;
  }

  char peek() {
    skipSpace();
    if (position >= text.size()) fail("unexpected end");
    return text[position];
  }

  void expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "'");
    ++position;
  }

  bool literal(std::string_view word) {
    if (text.substr(position, word.size()) != word) return false;
    position += word.size();
    return true;
  }

  Json parseValue(int depth) {
    if (depth > maximumDepth) fail("nesting too deep");
    Json value;
    const char c = peek();
    if (c == '{') {
      value.kind = Json::Kind::Object;
      ++position;
      if (peek() == '}') {
        ++position;
        return value;
      }
      for (;;) {
        if (peek() != '"') fail("expected a member name");
        std::string key = parseString();
        expect(':');
        value.object.emplace_back(std::move(key), parseValue(depth + 1));
        if (peek() == ',') {
          ++position;
          continue;
        }
        expect('}');
        return value;
      }
    }
    if (c == '[') {
      value.kind = Json::Kind::Array;
      ++position;
      if (peek() == ']') {
        ++position;
        return value;
      }
      for (;;) {
        value.array.push_back(parseValue(depth + 1));
        if (peek() == ',') {
          ++position;
          continue;
        }
        expect(']');
        return value;
      }
    }
    if (c == '"') {
      value.kind = Json::Kind::String;
      value.string = parseString();
      return value;
    }
    if (literal("true")) {
      value.kind = Json::Kind::Boolean;
      value.boolean = true;
      return value;
    }
    if (literal("false")) {
      value.kind = Json::Kind::Boolean;
      return value;
    }
    if (literal("null")) return value;
    value.kind = Json::Kind::Number;
    value.number = parseNumber();
    return value;
  }

  double parseNumber() {
    const std::size_t start = position;
    if (position < text.size() && text[position] == '-') ++position;
    auto digits = [&] {
      const std::size_t first = position;
      while (position < text.size() && text[position] >= '0' && text[position] <= '9') ++position;
      return position > first;
    };
    if (!digits()) fail("expected a value");
    if (position < text.size() && text[position] == '.') {
      ++position;
      if (!digits()) fail("expected digits after '.'");
    }
    if (position < text.size() && (text[position] == 'e' || text[position] == 'E')) {
      ++position;
      if (position < text.size() && (text[position] == '+' || text[position] == '-')) ++position;
      if (!digits()) fail("expected an exponent");
    }
    const std::string number(text.substr(start, position - start));
    return std::strtod(number.c_str(), nullptr);
  }

  static void appendUtf8(std::string &out, std::uint32_t code) {
    if (code < 0x80) {
      out += static_cast<char>(code);
    } else if (code < 0x800) {
      out += static_cast<char>(0xC0 | (code >> 6));
      out += static_cast<char>(0x80 | (code & 0x3F));
    } else if (code < 0x10000) {
      out += static_cast<char>(0xE0 | (code >> 12));
      out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (code & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (code >> 18));
      out += static_cast<char>(0x80 | ((code >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (code & 0x3F));
    }
  }

  std::uint32_t parseHex4() {
    if (position + 4 > text.size()) fail("truncated \\u escape");
    std::uint32_t code = 0;
    for (int i = 0; i < 4; ++i) {
      const char h = text[position++];
      code <<= 4;
      if (h >= '0' && h <= '9') code |= static_cast<std::uint32_t>(h - '0');
      else if (h >= 'a' && h <= 'f') code |= static_cast<std::uint32_t>(h - 'a' + 10);
      else if (h >= 'A' && h <= 'F') code |= static_cast<std::uint32_t>(h - 'A' + 10);
      else fail("bad \\u escape");
    }
    return code;
  }

  std::string parseString() {
    expect('"');
    std::string out;
    for (;;) {
      if (position >= text.size()) fail("unterminated string");
      const char c = text[position++];
      if (c == '"') return out;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (position >= text.size()) fail("unterminated escape");
      const char e = text[position++];
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
        std::uint32_t code = parseHex4();
        if (code >= 0xD800 && code < 0xDC00) {
          if (!literal("\\u")) fail("unpaired surrogate");
          const std::uint32_t low = parseHex4();
          if (low < 0xDC00 || low >= 0xE000) fail("unpaired surrogate");
          code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
        }
        appendUtf8(out, code);
        break;
      }
      default: fail("bad escape");
      }
    }
  }
};

} // namespace

Json Json::parse(std::string_view text) { return Parser(text).document(); }

Json Json::parseFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw Error("cannot open " + path);
  std::stringstream contents;
  contents << file.rdbuf();
  try {
    return parse(contents.str());
  } catch (const Error &error) {
    throw Error(path + ": " + error.what());
  }
}

const Json *Json::find(std::string_view key) const {
  if (kind != Kind::Object) return nullptr;
  for (const auto &[name, value] : object)
    if (name == key) return &value;
  return nullptr;
}

const Json &Json::at(std::string_view key) const {
  const Json *value = find(key);
  if (!value) throw Error("JSON: missing member " + std::string(key));
  return *value;
}

const std::string &Json::text(std::string_view key) const {
  const Json &value = at(key);
  if (value.kind != Kind::String) throw Error("JSON: member " + std::string(key) + " is not a string");
  return value.string;
}

std::uint64_t Json::integer(std::string_view key, std::uint64_t maximum) const {
  const Json &value = at(key);
  if (value.kind != Kind::Number || value.number < 0 || value.number != std::floor(value.number) ||
      value.number > static_cast<double>(maximum))
    throw Error("JSON: member " + std::string(key) + " is not an integer in 0.." + std::to_string(maximum));
  return static_cast<std::uint64_t>(value.number);
}

const std::vector<Json> &Json::list(std::string_view key) const {
  static const std::vector<Json> empty;
  const Json *value = find(key);
  if (!value) return empty;
  if (value->kind != Kind::Array) throw Error("JSON: member " + std::string(key) + " is not an array");
  return value->array;
}

} // namespace basalt
