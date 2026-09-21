# Assets

Sample models are not committed to this repository, because they are large and
carry their own licences.

Anything that glTF 2.0 describes works. The models the screenshots use come
from the Khronos sample assets:

  https://github.com/KhronosGroup/glTF-Sample-Assets

Put a `.gltf` or `.glb` here (with its `.bin` and textures beside it, for the
non-binary form) and open it from the interface, or pass it on the command line:

  basalt.exe assets/DamagedHelmet.glb

An environment is a Radiance `.hdr` panorama in equirectangular projection.
Pass one on the command line or open it from the Sun and sky panel; with none,
the renderer bakes its lighting from a procedural sky that follows the sun.
