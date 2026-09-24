#include "platform/Window.h"

#include "core/Log.h"

#include <imgui_impl_win32.h>
#include <windowsx.h>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND window, UINT message, WPARAM w,
                                                             LPARAM l);

namespace basalt {
namespace {
constexpr wchar_t kClassName[] = L"BasaltWindowClass";
}

Window::Window(const std::string &title, std::uint32_t width, std::uint32_t height, bool visible) {
  instance = GetModuleHandleW(nullptr);

  WNDCLASSEXW windowClass{sizeof(WNDCLASSEXW)};
  windowClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
  windowClass.lpfnWndProc = &Window::windowProc;
  windowClass.hInstance = instance;
  windowClass.hCursor = LoadCursorW(nullptr, IDC_ARROW);
  windowClass.hbrBackground = nullptr;
  windowClass.lpszClassName = kClassName;
  if (!RegisterClassExW(&windowClass) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS)
    throw Error("RegisterClassExW failed");

  RECT rect{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
  AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
  const std::wstring wide(title.begin(), title.end());
  window = CreateWindowExW(0, kClassName, wide.c_str(), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT,
                           CW_USEDEFAULT, rect.right - rect.left, rect.bottom - rect.top, nullptr,
                           nullptr, instance, this);
  if (!window) throw Error("CreateWindowExW failed");

  RECT client{};
  GetClientRect(window, &client);
  clientWidth = static_cast<std::uint32_t>(client.right - client.left);
  clientHeight = static_cast<std::uint32_t>(client.bottom - client.top);

  ShowWindow(window, visible ? SW_SHOWNORMAL : SW_HIDE);
  if (visible) UpdateWindow(window);
}

Window::~Window() {
  if (window) DestroyWindow(window);
}

LRESULT CALLBACK Window::windowProc(HWND window, UINT message, WPARAM w, LPARAM l) {
  if (message == WM_NCCREATE) {
    auto *create = reinterpret_cast<CREATESTRUCTW *>(l);
    SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(create->lpCreateParams));
  }
  auto *self = reinterpret_cast<Window *>(GetWindowLongPtrW(window, GWLP_USERDATA));
  if (self) return self->handleMessage(window, message, w, l);
  return DefWindowProcW(window, message, w, l);
}

LRESULT Window::handleMessage(HWND hwnd, UINT message, WPARAM w, LPARAM l) {
  // ImGui sees the message first, except while the mouse is captured.
  if (ImGui_ImplWin32_WndProcHandler(hwnd, message, w, l) && !captured) return 1;

  switch (message) {
  case WM_CLOSE:
    closed = true;
    return 0;
  case WM_DESTROY:
    closed = true;
    PostQuitMessage(0);
    return 0;
  case WM_SIZE: {
    const std::uint32_t newWidth = LOWORD(l), newHeight = HIWORD(l);
    if (newWidth != clientWidth || newHeight != clientHeight) {
      clientWidth = newWidth;
      clientHeight = newHeight;
      resized = true;
    }
    return 0;
  }
  case WM_KEYDOWN:
  case WM_SYSKEYDOWN:
    if (w < 256) state.keyDown[w] = true;
    if (w == VK_ESCAPE && captured) captureMouse(false);
    if (message == WM_SYSKEYDOWN) break; // The system still wants Alt combinations.
    return 0;
  case WM_KEYUP:
  case WM_SYSKEYUP:
    if (w < 256) state.keyDown[w] = false;
    if (message == WM_SYSKEYUP) break;
    return 0;
  case WM_LBUTTONDOWN: state.mouseDown[0] = true; return 0;
  case WM_LBUTTONUP: state.mouseDown[0] = false; return 0;
  case WM_RBUTTONDOWN: state.mouseDown[1] = true; return 0;
  case WM_RBUTTONUP: state.mouseDown[1] = false; return 0;
  case WM_MBUTTONDOWN: state.mouseDown[2] = true; return 0;
  case WM_MBUTTONUP: state.mouseDown[2] = false; return 0;
  case WM_MOUSEWHEEL:
    state.wheelDelta += static_cast<float>(GET_WHEEL_DELTA_WPARAM(w)) / WHEEL_DELTA;
    return 0;
  case WM_MOUSEMOVE: {
    const float x = static_cast<float>(GET_X_LPARAM(l)), y = static_cast<float>(GET_Y_LPARAM(l));
    // While captured the pump measures from the pinned cursor; this would count it twice.
    if (!captured) {
      state.mouseDeltaX += x - state.mouseX;
      state.mouseDeltaY += y - state.mouseY;
    }
    state.mouseX = x;
    state.mouseY = y;
    return 0;
  }
  case WM_KILLFOCUS:
    state.keyDown.fill(false);
    for (bool &button : state.mouseDown) button = false;
    if (captured) captureMouse(false);
    return 0;
  default:
    break;
  }
  return DefWindowProcW(hwnd, message, w, l);
}

bool Window::pumpMessages() {
  MSG message{};
  while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
    TranslateMessage(&message);
    DispatchMessageW(&message);
    if (message.message == WM_QUIT) closed = true;
  }
  // The cursor is pinned, so deltas keep coming however far the camera turns.
  if (captured && !closed) {
    POINT centre = capturedOrigin;
    ClientToScreen(window, &centre);
    POINT current{};
    GetCursorPos(&current);
    state.mouseDeltaX += static_cast<float>(current.x - centre.x);
    state.mouseDeltaY += static_cast<float>(current.y - centre.y);
    SetCursorPos(centre.x, centre.y);
  }
  return !closed;
}

void Window::waitForMessage() { WaitMessage(); }

void Window::endFrame() {
  state.mouseDeltaX = 0;
  state.mouseDeltaY = 0;
  state.wheelDelta = 0;
}

bool Window::consumeResized() {
  const bool was = resized;
  resized = false;
  return was;
}

void Window::setTitle(const std::string &title) {
  const std::wstring wide(title.begin(), title.end());
  SetWindowTextW(window, wide.c_str());
}

void Window::captureMouse(bool capture) {
  if (capture == captured) return;
  captured = capture;
  if (capture) {
    RECT client{};
    GetClientRect(window, &client);
    capturedOrigin = {(client.right - client.left) / 2, (client.bottom - client.top) / 2};
    POINT screen = capturedOrigin;
    ClientToScreen(window, &screen);
    SetCursorPos(screen.x, screen.y);
    SetCapture(window);
    while (ShowCursor(FALSE) >= 0) {
    }
  } else {
    ReleaseCapture();
    while (ShowCursor(TRUE) < 0) {
    }
  }
  state.mouseDeltaX = 0;
  state.mouseDeltaY = 0;
}

} // namespace basalt
