// Logging and the error type; every failure names the operation and, for Vulkan, the result.
#pragma once
#include <vulkan/vulkan.h>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace basalt {

class Error : public std::runtime_error {
public:
  explicit Error(const std::string &what) : std::runtime_error(what) {}
};

struct LogEntry {
  enum class Level { Info, Warning, ErrorLevel };
  Level level;
  std::string text;
};
const std::vector<LogEntry> &logEntries();
void openLogFile(const std::string &path);
void logRaw(LogEntry::Level level, std::string text);

template <class... Args>
void logInfo(std::format_string<Args...> format, Args &&...args) {
  logRaw(LogEntry::Level::Info, std::format(format, std::forward<Args>(args)...));
}
template <class... Args>
void logWarning(std::format_string<Args...> format, Args &&...args) {
  logRaw(LogEntry::Level::Warning, std::format(format, std::forward<Args>(args)...));
}
template <class... Args>
void logError(std::format_string<Args...> format, Args &&...args) {
  logRaw(LogEntry::Level::ErrorLevel, std::format(format, std::forward<Args>(args)...));
}

const char *resultName(VkResult result);

inline void check(VkResult result, std::string_view what) {
  if (result != VK_SUCCESS)
    throw Error(std::format("{} failed: {}", what, resultName(result)));
}

} // namespace basalt
