# Basalt

A Vulkan renderer whose shaders are written in **[Slang](https://shader-slang.org)**.

Every shader — the PBR forward pass, the shadow pass, the sky, the reflection
resolve, the bloom chain, the tone mapper, the image-based lighting bake, the
ray queries, the path tracers, the BVH builders and even the user interface —
is a Slang module compiled to Vulkan SPIR-V at build time by `slangc`, which the
first configure fetches for you. The same modules are compiled to C++ for the
CPU path tracer, so the CPU and the GPU run one implementation of the light
transport.

![The Damaged Helmet rendered by Basalt](docs/helmet.png)

## What it does

- **glTF 2.0 and GLB** loading: nodes, meshes, metallic-roughness materials,
  textures in the right colour space, punctual lights, and tangents generated
  where a model does not ship them.
- **Physically based shading**: Cook-Torrance GGX with height-correlated Smith
  visibility, multiple-scattering energy compensation, specular and horizon
  occlusion, normal mapping, occlusion, emissive, alpha masking and blending,
  and double-sided materials drawn and shadowed from both sides.
- **Image-based lighting** from a Radiance `.hdr` panorama or from a
  procedural sky: an environment cube and a roughness-prefiltered specular
  chain baked on the GPU, and the diffuse irradiance projected onto nine
  spherical harmonic coefficients on the host, which costs the shaders no
  texture at all. Loading a panorama moves its sun (or moon) out of the image
  into the analytic sun, energy and all, so the rasteriser's shadows and the
  path tracers light with the same sun; a panorama without one lights alone.
- **Reflections** resolved in a compute pass from a small G-buffer the forward
  pass writes beside its colour: the prefiltered environment alone, a
  screen-space march with binary refinement that falls back to it, or a ray
  query through the scene that shades what it hits with the hit's own base
  colour, metallic-roughness, emissive and normal maps, the sun through a
  shadow ray, the punctual lights and the environment. Rough surfaces read a
  blurred mip of the result while the view moves; once it holds still, each
  traced frame samples a different direction of the GGX lobe and the frames
  average into the glossy reflection itself.
- **Shadows** from four cascaded shadow maps, stabilised by fitting a sphere to
  each slice and snapping it to whole texels and filtered through a hardware
  comparison sampler, or from rays traced towards a sun disc of adjustable
  size, which gives contact-hard, distance-soft penumbrae with no bias to tune.
  Every traced ray tests alpha-masked candidates against their texture, so a
  leaf shadows as a leaf and not as its quad.
- **Ambient occlusion** from the material's map, or traced over the hemisphere:
  contact occlusion over a short radius, or sky visibility, the share of the
  environment each point sees, as the path tracers light it.
- **Punctual lights** from the file, each casting a traced shadow, so a light
  behind a wall does not light through it. They are culled into a grid of
  cells every frame, tiles across the screen by slices in depth, so a pixel
  walks only the lights that reach it. A count of test lights scatters point
  lights through the scene, since almost no glTF file carries any.
- **Temporal antialiasing**: every frame is rendered through a different
  sub-pixel offset and blended with the one before it, found again through the
  camera's motion and clamped to its neighbourhood. While the view is still
  the frames are averaged exactly instead, so the stochastic terms (shadow
  cones, occlusion rays, sampled reflections, the march's jitter) converge
  rather than settling at a floor, and any change starts over.
- **Post processing**: bloom through a downsample and tent-filter upsample
  chain, ACES or Reinhard tone mapping, exposure, vignette, grain, sharpening,
  and a choice of no antialiasing, FXAA or temporal.
- **A user interface** built with Dear ImGui, drawn through the engine's own
  shaders rather than the library's bundled SPIR-V.
- **A path tracer** beside the rasteriser, switched with F5 or the interface,
  on the CPU and on the GPU in three ways: inline ray queries, its own BVH in
  compute (SAH or GPU LBVH, binary or quantized BVH4/BVH8), and a full Vulkan
  ray-tracing pipeline. Each runs as a megakernel or as bounce-synchronous
  wavefront stages, and a hybrid mode continues rasterised primaries with
  traced paths. It is unidirectional, with next-event estimation of the
  environment, the sun, punctual lights and emissive meshes, multiple
  importance sampling and Russian roulette. It uses one BSDF for the glTF
  material with transmission, IOR, thin and closed dielectrics and clearcoat.
  Options add thin-lens depth of field, ray-cone texture filtering and ReSTIR DI
  at the primary vertex. Temporal reconstruction and Intel Open Image Denoise
  clean up the display without touching the raw image. The light transport is
  written once, as Slang modules generic over the intersector and the texture
  reads, and compiled both to SPIR-V and to C++, so every backend runs the same
  code and they agree to within tested tolerances.
  Captures write linear PFM or OpenEXR with JSON metadata; `basalt --help`
  lists every option.
- **Debug views** for base colour, normals, metallic, roughness, occlusion,
  the shadow term, cascade assignment, emissive, texture coordinates, the
  reflection weight, the resolved reflection and the trace's confidence. They
  bypass tone mapping, so what you see is the number.

The ray-traced options need `VK_KHR_acceleration_structure`, `VK_KHR_ray_query`
and a device that allows 128 sampled images in one stage, which is what the hit
texture table costs. On a device without them the engine loads the rasterised
shader pair and the interface greys those options out; everything else is the
same. Set `BASALT_NO_RAY_TRACING=1` to force that path on a device that could
trace.

## Building

Windows only for now. You need Visual Studio 2022 with the C++ toolset, the
Vulkan SDK, and CMake 3.24 or newer.

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config RelWithDebInfo --parallel
./build/RelWithDebInfo/basalt.exe
```

That works from an ordinary PowerShell prompt, because the Visual Studio
generator finds the compiler by itself. Ninja builds faster, but it has to be
installed and it expects the compiler already on `PATH`, so run it from the
**x64 Native Tools Command Prompt for VS 2022** rather than a plain shell:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel
./build/basalt.exe
```

The build compiles every entry point in [`shaders/slang/`](shaders/slang) with
`slangc` and validates each module with `spirv-val` before it links the engine,
so a shader that does not compile or validate fails the build. It also compiles
the path tracer's modules to C++ for the CPU tracer.

Slang is not in the repository. The first configure downloads its release
(v2026.18.2) into `.deps/`, checks it against a pinned SHA-256, and never asks
again; point `-DBASALT_SLANG_ROOT=<path>` at an unpacked release to use your own.
[`cmake/slang.cmake`](cmake/slang.cmake) has the rest.

The first configure also fetches Intel Embree and Intel Open Image Denoise,
prebuilt and checked against pinned hashes, into `.deps/`. Both are optional:
without them, or with `-DBASALT_WITH_EMBREE=OFF` and `-DBASALT_WITH_OIDN=OFF`,
the path tracer uses its own BVH and does not denoise.

## Running

```
basalt [model.gltf|model.glb] [environment.hdr] [options]
basalt --help
```

A few common runs:

```powershell
# Look around a model; F5 switches to the path tracer.
basalt DamagedHelmet.glb

# A converged GPU path-traced reference: 1024 samples per pixel, raw linear OpenEXR.
basalt DamagedHelmet.glb sky.hdr --renderer gpu --spp 1024 --pfm reference.exr --no-ui

# The same on the CPU, headless, with no Vulkan at all.
basalt-pt-cli DamagedHelmet.glb --environment sky.hdr --spp 1024 --output reference.exr

# ReSTIR DI and depth of field on the GPU's own BVH, wavefront execution.
basalt scene.gltf --renderer gpu-bvh --path-execution wavefront --di-estimator restir --aperture 0.05
```

`basalt --help` lists every option. An output path ending in `.exr` writes
OpenEXR, any other PFM, and each capture writes a JSON metadata file beside it.
`ctest --test-dir build` runs the test suite; a test that needs a capability the
device lacks reports itself skipped.

F1 hides and shows the interface; F5 switches between the rasteriser and the
path tracer; F12 saves a screenshot next to the executable, named by the time,
without leaving.

Orbit with the left mouse button, pan with the middle, zoom with the wheel.
Hold the right mouse button to fly, then use W, A, S, D, Q and E; Escape or
releasing the button gives the cursor back.

Sample models are not committed. Any glTF 2.0 file works; the
[Khronos sample assets](https://github.com/KhronosGroup/glTF-Sample-Assets) are
a good place to start.

## Architecture

**Shaders.** Every shader lives in [`shaders/slang/`](shaders/slang). The entry points are in
`entries/`; the code they share is organised as Slang modules, `basalt/` for the rasteriser
(material, shading, random numbers, the traced shadow and reflection rays) and `pt/` for the path
tracers (camera and path state, ray cones, BVH traversal, surfaces, the BSDF, lights, the path
loop, ReSTIR DI, the temporal filter, the BVH builders' records). An entry imports the modules
it needs, and variants are generics rather than preprocessor copies: the forward pass is one
function with and without traced rays, and every GPU path tracer is one pixel function over the
tracer it is given.

**Bindings.** Slang assigns every binding and the engine reads them back. Each compiled entry
point comes with slangc's reflection JSON, and `src/gpu/` builds descriptor set layouts, pipeline
layouts, push-constant ranges and vertex input state from it rather than from constants written
twice. A file's loose globals share one set, each `ParameterBlock` is a set of its own (the
rasteriser's per-frame view, per-material and scene data; the path tracers' scene and traced
structure), and an entry point's `uniform` parameters are its push constants. `DescriptorWriter`
takes resources by the name the shader gave them (`"geometry.materials"`, `"maps.table"`), so
renaming one moves its descriptor with no C++ change. What an entry reads comes from its module's
own decorations, so variants of one file share host code and a binding no stage reads is left
out of the layout.

**Path tracers.** The path loop is `ptTracePath<Tracer, Maps, ...>` in `pt/pt_integrator.slang`,
generic over an `IPtTracer` and an `IPtMaps`. The GPU gives it its texture table as the maps and
one of four tracers: the binary or quantized BVH4/BVH8 software BVH, ray queries, or the full ray
pipeline's `TraceRay`. The megakernel entries run a pixel's samples in one thread; the wavefront
entries split each bounce into intersect, shade and shadow stages over queues in device memory,
and the ReSTIR DI passes resample the primary vertex's lights before either. The CPU tracer
compiles the module `pt/pt_cpu.slang` to C++ (`slangc -target cpp`) into the `basalt-pt`
library and calls its exports; the host keeps only what it owns, its copies of the textures and
its own intersectors (Embree and an AVX2 BVH8 traversal), which the generated code calls back
through. The constants the host shares with the shaders are generated from the modules too.

**BVH builders.** The software BVH is built by default on the GPU with a binned SAH: the CPU
builder's tree, which traces fastest on every scene measured, built top-down one level per
dispatch (`--bvh-builder gpu-sah`; on the CPU where the device lacks subgroups of 32 or more).
The others (`--bvh-builder`) are the CPU builder itself, a serial LBVH kernel, a parallel LBVH
(Karras) over a GPU radix sort, Apetrei's single-pass LBVH, HIPRT's batched LBVH, PLOC, PLOC++
(PLOC with each iteration one fused dispatch) and H-PLOC (PLOC merges at the nodes of an LBVH
climb), all publishing the same node layout, collapsed on the GPU to quantized BVH4/BVH8 and
refitted in place for animated geometry (the SAH trees are rebuilt instead). The three parallel
LBVHs publish the serial builder's tree byte for byte, PLOC++
PLOC's and the GPU SAH the CPU builder's, which the tests check. Early split clipping
(`--bvh-split-clipping`, Ernst and Greiner) feeds any of them references to the parts of large
triangles instead of the triangles, clipped on the GPU for the GPU builders (the host's bytes, in
double precision; on the host where the device has no 64-bit floats); those trees are rebuilt
when animated.
The binary layout is traversed on the GPU with a stack by default, or (`--bvh-traversal`) with
Aila and Laine's while-while or speculative while-while loops or Laine's restart trail, which all
find the same hits. A tree deeper than 64 levels, top and bottom together (PLOC's reach that on
large scenes), is traversed by kernels with a 96-entry stack; the restart trail keeps no stack.

## Layout

| Path | What is in it |
|---|---|
| `shaders/slang/` | Every shader, in Slang: `entries/` holds the entry points, `basalt/` the rasteriser's modules and `pt/` the path tracers'. |
| `src/core/` | Maths, logging, the error type, the JSON reader for shader reflection. |
| `src/gpu/` | Vulkan context, swapchain, buffers and images, shader reflection, pipeline creation, descriptors, staging uploads. |
| `src/scene/` | glTF loading, the scene representation, the camera. |
| `src/render/` | The frame: shadow, forward, sky, reflection, bloom and post passes; the GPU path tracers' passes and reconstruction; the acceleration structures and the GPU BVH builders; the IBL bake; the ImGui backend. |
| `src/pt/` | The CPU side: the generated path tracer's host view (`Shared.h`, `Tracing.h`), the BVH builders and wide layouts, the environment and albedo tables, the CPU tracer, capture metadata and image files. |
| `tests/` | The CTest manifest's programs and scripts: CPU transport tests, GPU oracles and image gates per backend, lifecycle and CLI rejection tests, and generated test scenes in `tests/data/`. |
| `tools/` | The image comparison script, the benchmark harness, `fetch_scenes.py` (the larger benchmark scenes, from McGuire's archive) and `shader-stats`. |
| `cmake/` | Fetching Slang and the prebuilt Embree and Open Image Denoise, and generating the CPU tracer's C++ and constants. |
| `docs/` | The screenshot above. |

## Known limitations

- The rasteriser's ambient light is unoccluded unless traced sky visibility is
  on, and even then it misses light bounced between surfaces, which the path
  tracers include. With the procedural sky it also counts the sky's painted
  disc in its ambient light, about ten per cent too bright.
- The rasteriser draws the base layer of transmissive and clearcoated
  materials only; the path tracers render both layers.

- Windows only. Nothing in the engine is Windows-specific except the window and
  the file dialogs.
- No skinning, morph targets or animation; node transforms are static.
- Transparent surfaces are sorted per primitive, not per triangle, and cast no
  shadows.
- `KHR_materials_specular_glossiness` is approximated, not implemented.
- The serial GPU BVH builder (`--bvh-builder gpu-serial`) is one GPU thread, a correctness
  baseline. It submits its bottom levels in batches so that no submission outlasts the driver's
  timeout, but it rejects a mesh of more than 262,144 triangles and takes seconds on a large
  scene (about 14 s for the Godot Bistro's 2.8 million triangles on an RTX 4090).
- In the rasteriser's traced effects, blended surfaces are left out of the
  acceleration structure, so those rays pass through them; the path tracers
  test blended and masked surfaces in their any-hit.
- A hit shaded by a reflection ray takes the punctual lights unshadowed, and a
  reflection of a reflection is the environment.
- A cell of the light grid holds at most sixty-four lights. Past that the rest
  are dropped by index, so the cell keeps whichever come first rather than the
  nearest, and a light with no falloff reaches every cell. Both only bite where
  that many lights genuinely overlap.
- In the rasteriser, punctual lights are shadowed only on a ray tracing device;
  the rasterised path lights through walls, and the interface says so by
  greying the control. The path tracers shadow them on every device.
- Once a still view has converged, the traced reflection is the average of one
  lobe sample per frame, kept with the cosine as its probability; the frames it
  is not kept read the prefiltered environment instead. Both estimate the same
  integral, so the result is consistent, but it is not a pure path trace.
- The temporal history is reprojected from depth, which is right for a static
  scene but would smear a moving object, so the reprojection will need per
  object motion once anything animates.

## Licence

Basalt is licensed under the Apache License 2.0; see [LICENSE](LICENSE).

The vendored libraries under `third_party/` keep their own licences, and
[NOTICES.md](NOTICES.md) lists them with the fetched dependencies, Slang among
them.
