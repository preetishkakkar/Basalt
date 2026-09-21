#include "core/Log.h"

#include <windows.h>

#include <fstream>
#include <mutex>

namespace basalt {
namespace {
std::mutex gMutex;
std::vector<LogEntry> gEntries;
std::ofstream gFile;
} // namespace

void openLogFile(const std::string &path) {
  std::lock_guard<std::mutex> lock(gMutex);
  gFile.open(path, std::ios::trunc);
}

const std::vector<LogEntry> &logEntries() { return gEntries; }

void logRaw(LogEntry::Level level, std::string text) {
  const char *tag = level == LogEntry::Level::Info      ? "info"
                    : level == LogEntry::Level::Warning ? "warning"
                                                        : "error";
  const std::string line = std::format("[basalt] {}: {}\n", tag, text);
  OutputDebugStringA(line.c_str());
  std::lock_guard<std::mutex> lock(gMutex);
  if (gFile.is_open()) {
    gFile << line;
    gFile.flush();
  }
  if (gEntries.size() > 4096) gEntries.erase(gEntries.begin(), gEntries.begin() + 2048);
  gEntries.push_back({level, std::move(text)});
}

const char *resultName(VkResult result) {
  switch (result) {
  case VK_SUCCESS: return "VK_SUCCESS";
  case VK_NOT_READY: return "VK_NOT_READY";
  case VK_TIMEOUT: return "VK_TIMEOUT";
  case VK_INCOMPLETE: return "VK_INCOMPLETE";
  case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
  case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
  case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
  case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
  case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
  case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
  case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
  case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
  case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
  case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
  case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
  case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
  case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
  case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
  case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
  case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
  case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
  case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
  case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
  default: return "VkResult";
  }
}

} // namespace basalt
