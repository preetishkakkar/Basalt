# Third-party notices

Basalt bundles the following components. Each is used under its own licence, and
each licence text ships with the component under `third_party/`.

| Component | Version | Use | Licence |
|---|---|---|---|
| [Dear ImGui](https://github.com/ocornut/imgui) | master, vendored | The user interface, and its Win32 platform backend. The Vulkan rendering backend is **not** used: Basalt draws ImGui through its own Metal shaders. | MIT (`third_party/imgui/LICENSE.txt`) |
| [cgltf](https://github.com/jkuhlmann/cgltf) | master, vendored | glTF 2.0 and GLB parsing. | MIT (`third_party/cgltf/LICENSE`) |
| [stb_image, stb_image_write](https://github.com/nothings/stb) | 2.x, vendored | Decoding model textures and Radiance panoramas; writing screenshots. | Public domain or MIT, at your option (stated in the headers) |
| [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | master, vendored | Device memory allocation. | MIT (`third_party/vma/LICENSE.txt`) |
| Vulkan headers and loader | From the installed Vulkan SDK | The graphics API. Not redistributed here. | Apache 2.0 (see the SDK) |


Basalt's own source is under the Apache License 2.0; see `LICENSE`.

`tools/metal2vulkan/` holds the `msl2spirv` compiler, its owned Metal standard
library and the host reflection library. The compiler executable is not in this
repository: the build fetches it from the release named `msl2spirv-<version>`.
It is a separate project, distributed under its own terms.

No Apple headers, libraries or tools are included. The Metal standard library
under `tools/metal2vulkan/share/metal2vulkan/stdlib/` is independently authored:
it declares the Metal surface the compiler accepts and contains no Apple code.

Sample models are not committed to this repository. The screenshots in `docs/`
were rendered from the [Khronos glTF Sample
Assets](https://github.com/KhronosGroup/glTF-Sample-Assets), which carry their
own licences.
