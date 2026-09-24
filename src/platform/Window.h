// Win32 window and polled input; ImGui sees the same messages through its own backend.
#pragma once
#include <windows.h>

#include <array>
#include <cstdint>
#include <string>

namespace basalt {

struct InputState {
  std::array<bool, 256> keyDown{};
  bool mouseDown[3]{};
  float mouseX = 0, mouseY = 0;
  float mouseDeltaX = 0, mouseDeltaY = 0;
  float wheelDelta = 0;
  bool keyPressed(int virtualKey) const {
    return virtualKey >= 0 && virtualKey < 256 && keyDown[static_cast<std::size_t>(virtualKey)];
  }
};

class Window {
public:
  Window(const std::string &title, std::uint32_t width, std::uint32_t height, bool visible = true);
  ~Window();
  Window(const Window &) = delete;
  Window &operator=(const Window &) = delete;

  bool pumpMessages();
  void endFrame();
  void waitForMessage();

  HWND handle() const { return window; }
  HINSTANCE moduleHandle() const { return instance; }
  std::uint32_t width() const { return clientWidth; }
  std::uint32_t height() const { return clientHeight; }
  bool minimized() const { return clientWidth == 0 || clientHeight == 0; }
  // True for exactly one frame after the client area changed size.
  bool consumeResized();
  const InputState &input() const { return state; }
  void setTitle(const std::string &title);
  void captureMouse(bool capture);
  bool mouseCaptured() const { return captured; }

private:
  static LRESULT CALLBACK windowProc(HWND window, UINT message, WPARAM w, LPARAM l);
  LRESULT handleMessage(HWND window, UINT message, WPARAM w, LPARAM l);

  HINSTANCE instance = nullptr;
  HWND window = nullptr;
  std::uint32_t clientWidth = 0, clientHeight = 0;
  bool closed = false;
  bool resized = false;
  bool captured = false;
  POINT capturedOrigin{};
  InputState state;
};

} // namespace basalt
