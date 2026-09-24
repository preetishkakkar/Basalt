# Third-party notices

Basalt bundles the following components. Each is used under its own licence, and
each licence text ships with the component under `third_party/`.

| Component | Version | Use | Licence |
|---|---|---|---|
| [Dear ImGui](https://github.com/ocornut/imgui) | 1.93.0 WIP (19297), vendored | The user interface, and its Win32 platform backend. The Vulkan rendering backend is **not** used: Basalt draws ImGui through its own Metal shaders. | MIT (`third_party/imgui/LICENSE.txt`) |
| [cgltf](https://github.com/jkuhlmann/cgltf) | master, vendored | glTF 2.0 and GLB parsing. | MIT (`third_party/cgltf/LICENSE`) |
| [stb_image, stb_image_write](https://github.com/nothings/stb) | stb_image 2.30, stb_image_write 1.16, vendored | Decoding model textures and Radiance panoramas; writing screenshots. | Public domain or MIT, at your option (stated in the headers) |
| [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | 3.4.0, vendored | Device memory allocation. | MIT (`third_party/vma/LICENSE.txt`) |
| Vulkan headers and loader | From the installed Vulkan SDK | The graphics API. Not redistributed here. | Apache 2.0 (see the SDK) |
| [Intel Embree](https://github.com/RenderKit/embree) | 4.4.1, prebuilt, fetched at configure | The CPU path tracer's alternative intersector (`embree4.dll`, `tbbmalloc.dll`). Optional: `-DBASALT_WITH_EMBREE=OFF`. | Apache 2.0 (`third_party/licenses/Embree.txt`) |
| [Intel Open Image Denoise](https://github.com/RenderKit/oidn) | 2.5.1, prebuilt, fetched at configure | Denoising the path tracer's image, on the CPU or a CUDA GPU. Optional: `-DBASALT_WITH_OIDN=OFF`. | Apache 2.0 (`third_party/licenses/OpenImageDenoise.txt`) |
| [oneAPI Threading Building Blocks](https://github.com/uxlfoundation/oneTBB) | As shipped with Open Image Denoise 2.5.1 | Threads for Embree and the denoiser (`tbb12.dll`). | Apache 2.0; its own third parties in `third_party/licenses/oneTBB.txt` |


Basalt's own source is under the Apache License 2.0; see `LICENSE`.

The OpenEXR files Basalt writes are produced by its own code (`src/pt/ImageFile.cpp`),
following the published OpenEXR file layout; no OpenEXR library is linked or bundled.
The prebuilt downloads and the compiler are checked against the SHA-256 pins in
`cmake/prebuilt.cmake` and `cmake/msl2spirv.cmake`.

`tools/metal2vulkan/` holds the `msl2spirv` compiler, its owned Metal standard
library and the host reflection library. The compiler's own third parties (LLVM and
SPIRV-Tools, both Apache 2.0 with LLVM exceptions / Apache 2.0) have their licence texts in
`third_party/licenses/LLVM.txt` and `third_party/licenses/SPIRV-Tools.txt`. The compiler executable is not in this
repository: the build fetches it from the release named `msl2spirv-<version>`.
It is a separate project, distributed under its own terms.

No Apple headers, libraries or tools are included. The Metal standard library
under `tools/metal2vulkan/share/metal2vulkan/stdlib/` is independently authored:
it declares the Metal surface the compiler accepts and contains no Apple code.

Sample models are not committed to this repository. The screenshots in `docs/`
were rendered from the [Khronos glTF Sample
Assets](https://github.com/KhronosGroup/glTF-Sample-Assets), which carry their
own licences.
