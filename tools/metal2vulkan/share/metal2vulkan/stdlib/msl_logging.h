#pragma once
#include "msl_prelude.h"
// Independently authored shader logging surface (MSL 3.2 os_log). Each call
// site records its format string in reflection and, when logging is enabled,
// appends a record of 32-bit argument words to the bounded log buffer bound
// at buffer index 31; see docs/LOGGING.md. Without --enable-logging the
// calls compile to nothing, as Apple's compiler does without its flag.
namespace metal {
struct os_log {
  template <typename... Args> void log(const char *format, Args... args) const __attribute__((annotate("msl.log:default")));
  template <typename... Args> void log_debug(const char *format, Args... args) const __attribute__((annotate("msl.log:debug")));
  template <typename... Args> void log_info(const char *format, Args... args) const __attribute__((annotate("msl.log:info")));
  template <typename... Args> void log_error(const char *format, Args... args) const __attribute__((annotate("msl.log:error")));
  template <typename... Args> void log_fault(const char *format, Args... args) const __attribute__((annotate("msl.log:fault")));
};
constexpr os_log os_log_default{};
}
