// A small JSON reader for the shader compiler's reflection output.
#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace basalt {

struct Json {
  enum class Kind { Null, Boolean, Number, String, Array, Object };
  Kind kind = Kind::Null;
  bool boolean = false;
  double number = 0;
  std::string string;
  std::vector<Json> array;
  std::vector<std::pair<std::string, Json>> object;

  static Json parse(std::string_view text);
  static Json parseFile(const std::string &path);

  // Objects: the member, or null when absent.
  const Json *find(std::string_view key) const;
  // Objects: the member; throws when absent, naming the key.
  const Json &at(std::string_view key) const;
  // A string member; throws when absent or not a string.
  const std::string &text(std::string_view key) const;
  // A non-negative integral number member, at most maximum; throws otherwise.
  std::uint64_t integer(std::string_view key, std::uint64_t maximum) const;
  // An array member; an absent one reads as empty.
  const std::vector<Json> &list(std::string_view key) const;
};

} // namespace basalt
