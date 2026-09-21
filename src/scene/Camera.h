// Orbit camera with a fly mode on the right mouse button.
#pragma once
#include "core/Math.h"

namespace basalt {

struct InputState;

class Camera {
public:
  void frame(const Aabb &bounds);
  void update(const InputState &input, float deltaSeconds, bool uiWantsMouse, bool mouseCaptured);
  // Converts `target` between orbit centre and eye, so the eye does not move.
  void setFlying(bool fly);

  Mat4 viewMatrix() const;
  Mat4 projectionMatrix(float aspect) const;
  Vec3 position() const;

  Vec3 target{0, 0, 0};
  float distance = 5.0f;
  float yaw = radians(45.0f);
  float pitch = radians(20.0f);
  float fieldOfView = radians(50.0f);
  float nearPlane = 0.05f;
  float orbitSpeed = 0.006f;
  float panSpeed = 1.0f;
  float zoomSpeed = 0.12f;
  float flySpeed = 3.0f;
  bool flying = false;
  float sceneRadius = 1.0f;

private:
  Vec3 forwardVector() const;
};

} // namespace basalt
