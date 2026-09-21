#include "scene/Camera.h"

#include "platform/Window.h"

#include <algorithm>

namespace basalt {

void Camera::frame(const Aabb &bounds) {
  if (!bounds.valid()) return;
  sceneRadius = std::max(bounds.radius(), 1e-3f);
  // Far enough for the bounding sphere to fit the vertical field of view, with margin.
  distance = sceneRadius / std::tan(fieldOfView * 0.5f) * 1.4f;
  nearPlane = std::max(distance * 0.001f, sceneRadius * 0.001f);
  flySpeed = sceneRadius * 0.9f;
  // While flying `target` is the eye, which goes where the orbit would put it.
  target = flying ? bounds.center() + forwardVector() * distance : bounds.center();
}

Vec3 Camera::forwardVector() const {
  return {std::cos(pitch) * std::sin(yaw), std::sin(pitch), std::cos(pitch) * std::cos(yaw)};
}

Vec3 Camera::position() const {
  if (flying) return target;
  return target + forwardVector() * distance;
}

void Camera::setFlying(bool fly) {
  if (fly == flying) return;
  // `target` is the orbit centre in one mode and the eye in the other; the eye must not move.
  if (fly) target = target + forwardVector() * distance;
  else target = target - forwardVector() * distance;
  flying = fly;
}

Mat4 Camera::viewMatrix() const {
  const Vec3 eye = position();
  const Vec3 look = flying ? eye - forwardVector() : target;
  return lookAt(eye, look, {0, 1, 0});
}

Mat4 Camera::projectionMatrix(float aspect) const {
  return perspectiveReverseZ(fieldOfView, aspect, nearPlane);
}

void Camera::update(const InputState &input, float deltaSeconds, bool uiWantsMouse,
                    bool mouseCaptured) {
  const float limit = radians(89.0f);

  if (mouseCaptured) {
    yaw -= input.mouseDeltaX * orbitSpeed;
    pitch = std::clamp(pitch - input.mouseDeltaY * orbitSpeed, -limit, limit);

    const Vec3 forward = -forwardVector();
    const Vec3 right = normalize(cross(forward, {0, 1, 0}));
    float speed = flySpeed * deltaSeconds;
    if (input.keyPressed(VK_SHIFT)) speed *= 4.0f;
    if (input.keyPressed(VK_CONTROL)) speed *= 0.25f;

    Vec3 movement{0, 0, 0};
    if (input.keyPressed('W')) movement += forward;
    if (input.keyPressed('S')) movement -= forward;
    if (input.keyPressed('D')) movement += right;
    if (input.keyPressed('A')) movement -= right;
    if (input.keyPressed('E')) movement += Vec3{0, 1, 0};
    if (input.keyPressed('Q')) movement -= Vec3{0, 1, 0};
    if (dot(movement, movement) > 0.0f) target += normalize(movement) * speed;
    if (input.wheelDelta != 0.0f) flySpeed = std::max(0.01f, flySpeed * (1.0f + input.wheelDelta * 0.1f));
    return;
  }

  if (uiWantsMouse) return;

  if (input.mouseDown[0]) {
    yaw -= input.mouseDeltaX * orbitSpeed;
    pitch = std::clamp(pitch + input.mouseDeltaY * orbitSpeed, -limit, limit);
  }
  if (input.mouseDown[2]) {
    // Pan scaled by distance, so it feels the same at any range.
    const Vec3 forward = -forwardVector();
    const Vec3 right = normalize(cross(forward, {0, 1, 0}));
    const Vec3 up = cross(right, forward);
    const float scale = distance * panSpeed * 0.0015f;
    target += right * (-input.mouseDeltaX * scale) + up * (input.mouseDeltaY * scale);
  }
  if (input.wheelDelta != 0.0f) {
    distance = std::clamp(distance * (1.0f - input.wheelDelta * zoomSpeed), sceneRadius * 0.01f,
                          sceneRadius * 100.0f);
    nearPlane = std::max(distance * 0.001f, sceneRadius * 0.0005f);
  }
}

} // namespace basalt
