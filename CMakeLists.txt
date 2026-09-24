cmake_minimum_required(VERSION 3.24)
project(basalt LANGUAGES CXX)

# Basalt: a Vulkan renderer whose shaders are written in Metal Shading Language
# and compiled to SPIR-V by msl2spirv (tools/metal2vulkan). Windows only.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT WIN32)
  message(FATAL_ERROR "Basalt targets Windows only for now.")
endif()

execute_process(COMMAND git rev-parse --short=12 HEAD WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_VARIABLE BASALT_GIT_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
execute_process(COMMAND git status --porcelain WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_VARIABLE BASALT_GIT_STATUS ERROR_QUIET)
if(NOT BASALT_GIT_REVISION)
  set(BASALT_GIT_REVISION unknown)
endif()
if(BASALT_GIT_STATUS)
  set(BASALT_GIT_DIRTY 1)
else()
  set(BASALT_GIT_DIRTY 0)
endif()

# A genuinely GPU-independent configuration: no Vulkan discovery, shader compiler,
# window system or renderer targets. It builds the shared CPU integrator, glTF loader,
# command-line renderer and analytic tests.
option(BASALT_CPU_ONLY "Build only the headless CPU path tracer and tests" OFF)
if(BASALT_CPU_ONLY)
  include(cmake/prebuilt.cmake)
  add_library(basalt-asset-deps STATIC src/asset_deps.cpp)
  target_include_directories(basalt-asset-deps PUBLIC third_party/cgltf third_party/stb)
  if(MSVC)
    target_compile_options(basalt-asset-deps PRIVATE /W0)
  endif()

  add_library(basalt-pt STATIC
    src/pt/Tracing.cpp src/pt/Bvh.cpp src/pt/BvhVariants.cpp src/pt/BvhAvx2.cpp src/pt/Tables.cpp src/pt/CpuTracer.cpp
    src/pt/Embree.cpp src/pt/Denoiser.cpp src/pt/Gltf.cpp src/pt/CaptureMetadata.cpp src/pt/WavefrontPlan.cpp
    src/pt/ImageFile.cpp)
  target_include_directories(basalt-pt PUBLIC src shaders)
  target_include_directories(basalt-pt PRIVATE third_party/cgltf third_party/stb)
  target_compile_definitions(basalt-pt PUBLIC NOMINMAX)
  target_link_libraries(basalt-pt PUBLIC basalt-asset-deps)
  if(BASALT_WITH_EMBREE)
    target_link_libraries(basalt-pt PUBLIC basalt::embree)
    target_compile_definitions(basalt-pt PUBLIC BASALT_WITH_EMBREE=1)
  endif()
  if(BASALT_WITH_OIDN)
    target_link_libraries(basalt-pt PUBLIC basalt::oidn)
    target_compile_definitions(basalt-pt PUBLIC BASALT_WITH_OIDN=1)
  endif()
  if(MSVC)
    target_compile_options(basalt-pt PRIVATE /W4 /permissive-)
    set_source_files_properties(src/pt/BvhAvx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  endif()

  add_executable(basalt-pt-cli src/pt/main.cpp)
  target_link_libraries(basalt-pt-cli PRIVATE basalt-pt)
  target_compile_definitions(basalt-pt-cli PRIVATE
    BASALT_GIT_REVISION="${BASALT_GIT_REVISION}" BASALT_GIT_DIRTY=${BASALT_GIT_DIRTY})
  basalt_copy_runtime_dlls(basalt-pt-cli)
  add_executable(pt-tests tests/pt_tests.cpp)
  target_link_libraries(pt-tests PRIVATE basalt-pt)
  basalt_copy_runtime_dlls(pt-tests)
  if(MSVC)
    target_compile_options(basalt-pt-cli PRIVATE /W4 /permissive-)
    target_compile_options(pt-tests PRIVATE /W4 /permissive-)
  endif()
  enable_testing()
  add_executable(temporal-motion-tests tests/temporal_motion_tests.cpp)
  target_link_libraries(temporal-motion-tests PRIVATE basalt-pt)
  target_compile_definitions(temporal-motion-tests PRIVATE BASALT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
  basalt_copy_runtime_dlls(temporal-motion-tests)
  add_test(NAME temporal-motion-tests COMMAND temporal-motion-tests)
  add_test(NAME temporal-embree-tests COMMAND temporal-motion-tests --embree)
  set_tests_properties(temporal-embree-tests PROPERTIES SKIP_RETURN_CODE 77)
  add_test(NAME temporal-v7-motion-tests COMMAND temporal-motion-tests --v7)
  add_test(NAME temporal-v7-embree-tests COMMAND temporal-motion-tests --v7 --embree)
  set_tests_properties(temporal-v7-embree-tests PROPERTIES SKIP_RETURN_CODE 77)
  add_test(NAME pt-tests COMMAND pt-tests)
  add_test(NAME pt-headless-smoke COMMAND basalt-pt-cli
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/materials.gltf"
    --output "${CMAKE_CURRENT_BINARY_DIR}/pt-headless-smoke.pfm"
    --width 16 --height 16 --spp 1 --bounces 2 --threads 2)
  add_test(NAME pt-headless-restir-smoke COMMAND basalt-pt-cli
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/manylights.gltf"
    --output "${CMAKE_CURRENT_BINARY_DIR}/pt-headless-restir-smoke.pfm"
    --width 32 --height 18 --spp 4 --bounces 2 --threads 2 --di-estimator restir)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME pt-exr-captures COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/exr_captures.py"
      --pt-cli "$<TARGET_FILE:basalt-pt-cli>" --scene "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/materials.gltf"
      --output-dir "${CMAKE_CURRENT_BINARY_DIR}/exr-captures")
  endif()
  add_test(NAME pt-headless-invalid-backend COMMAND basalt-pt-cli
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/materials.gltf" --intersector invalid)
  set_tests_properties(pt-headless-invalid-backend PROPERTIES WILL_FAIL TRUE)
  add_test(NAME pt-headless-load-failure COMMAND basalt-pt-cli
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/does-not-exist.gltf")
  set_tests_properties(pt-headless-load-failure PROPERTIES WILL_FAIL TRUE)
  if(NOT BASALT_WITH_EMBREE)
    add_test(NAME pt-headless-unavailable-embree COMMAND basalt-pt-cli
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/materials.gltf" --intersector embree --width 8 --height 8 --spp 1)
    set_tests_properties(pt-headless-unavailable-embree PROPERTIES WILL_FAIL TRUE)
  endif()
  if(NOT BASALT_WITH_OIDN)
    add_test(NAME pt-headless-unavailable-oidn COMMAND basalt-pt-cli
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/materials.gltf"
      --output "${CMAKE_CURRENT_BINARY_DIR}/pt-raw.pfm"
      --denoised-output "${CMAKE_CURRENT_BINARY_DIR}/pt-denoised.pfm" --width 8 --height 8 --spp 1)
    set_tests_properties(pt-headless-unavailable-oidn PROPERTIES WILL_FAIL TRUE)
  endif()
  return()
endif()

find_package(Vulkan REQUIRED)

# The Metal shader compiler, fetched on the first configure.
include(cmake/msl2spirv.cmake)
basalt_fetch_msl2spirv()

set(BASALT_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${BASALT_SHADER_DIR}")
set(BASALT_SHADER_OUTPUTS "")

# basalt_shader(<source.metal> ENTRY <name> [PAIR <fragment entry>] [OPTIONS ...])
#
# Compiles one entry point to <binary>/shaders/<name>.spv beside its reflection
# JSON. PAIR asks the compiler to check the vertex entry's outputs against that
# fragment entry's inputs, so an interface mismatch fails the build rather than
# the frame. The depfile makes edits to shaders/common.metal rebuild dependents.
function(basalt_shader source)
  cmake_parse_arguments(SH "" "ENTRY;PAIR" "OPTIONS" ${ARGN})
  set(spv "${BASALT_SHADER_DIR}/${SH_ENTRY}.spv")
  set(json "${BASALT_SHADER_DIR}/${SH_ENTRY}.json")
  set(pair "")
  if(SH_PAIR)
    set(pair --check-fragment-entry ${SH_PAIR})
  endif()
  add_custom_command(
    OUTPUT "${spv}" "${json}"
    COMMAND "${BASALT_MSL2SPIRV}"
            "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
            --stdlib "${BASALT_MSL_STDLIB}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            --entry ${SH_ENTRY} ${pair} ${SH_OPTIONS}
            -o "${spv}" --emit-reflection "${json}"
            --depfile "${spv}.d"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${source}" "${BASALT_MSL2SPIRV}"
    DEPFILE "${spv}.d"
    COMMENT "msl2spirv ${source} [${SH_ENTRY}]"
    VERBATIM)
  set(BASALT_SHADER_OUTPUTS ${BASALT_SHADER_OUTPUTS} "${spv}" "${json}" PARENT_SCOPE)
endfunction()

basalt_shader(shaders/forward.metal ENTRY forward_vertex PAIR forward_fragment)
basalt_shader(shaders/forward.metal ENTRY forward_fragment)
basalt_shader(shaders/forward_rt.metal ENTRY forward_fragment_rt)
basalt_shader(shaders/shadow.metal  ENTRY shadow_vertex PAIR shadow_fragment)
basalt_shader(shaders/shadow.metal  ENTRY shadow_fragment)
basalt_shader(shaders/sky.metal     ENTRY sky_vertex PAIR sky_fragment)
basalt_shader(shaders/sky.metal     ENTRY sky_fragment)
basalt_shader(shaders/post.metal    ENTRY post_vertex PAIR post_fragment)
basalt_shader(shaders/post.metal    ENTRY post_fragment)
basalt_shader(shaders/reflect.metal ENTRY resolve_reflections OPTIONS --local-size 8 --local-size-y 8)
basalt_shader(shaders/reflect.metal ENTRY composite_reflections OPTIONS --local-size 8 --local-size-y 8)
basalt_shader(shaders/reflect_rt.metal ENTRY resolve_reflections_rt OPTIONS --local-size 8 --local-size-y 8)
basalt_shader(shaders/taa.metal     ENTRY resolve_temporal OPTIONS --local-size 8 --local-size-y 8)
basalt_shader(shaders/cluster.metal ENTRY cull_lights OPTIONS --local-size 64)
basalt_shader(shaders/ui.metal      ENTRY ui_vertex PAIR ui_fragment)
basalt_shader(shaders/ui.metal      ENTRY ui_fragment)
basalt_shader(shaders/ibl.metal     ENTRY equirect_to_cube OPTIONS --local-size 8 --local-size-y 8)
basalt_shader(shaders/ibl.metal     ENTRY prefilter_specular OPTIONS --local-size 8 --local-size-y 8)
basalt_shader(shaders/bloom.metal   ENTRY bloom_downsample OPTIONS --local-size 8 --local-size-y 8)
basalt_shader(shaders/bloom.metal   ENTRY bloom_upsample OPTIONS --local-size 8 --local-size-y 8)
basalt_shader(shaders/path_trace.metal ENTRY path_trace OPTIONS --local-size 8 --local-size-y 8)
basalt_shader(shaders/path_trace_wide.metal ENTRY path_trace_wide OPTIONS --local-size 8 --local-size-y 8)
basalt_shader(shaders/path_trace_wide_emissive.metal ENTRY path_trace_wide_emissive OPTIONS --local-size 8 --local-size-y 8)
basalt_shader(shaders/path_trace_emissive.metal ENTRY path_trace_emissive OPTIONS --local-size 8 --local-size-y 8)
basalt_shader(shaders/path_trace_rt.metal ENTRY path_trace_rt OPTIONS --local-size 8 --local-size-y 8)
basalt_shader(shaders/path_trace_pipeline.metal ENTRY path_trace_pipeline)
basalt_shader(shaders/path_trace_pipeline.metal ENTRY path_pipeline_miss)
basalt_shader(shaders/path_trace_pipeline.metal ENTRY path_pipeline_closest)
basalt_shader(shaders/path_trace_pipeline.metal ENTRY path_pipeline_alpha)
basalt_shader(shaders/path_trace_hybrid_rt.metal ENTRY path_trace_hybrid_rt OPTIONS --local-size 8 --local-size-y 8)
basalt_shader(shaders/path_wavefront.metal ENTRY path_wavefront_shade OPTIONS --local-size 64)
basalt_shader(shaders/path_wavefront_atomic.metal ENTRY path_wavefront_shade_atomic OPTIONS --local-size 64)
basalt_shader(shaders/path_wavefront_fused_rt.metal ENTRY path_wavefront_fused_rt OPTIONS --local-size 64)
basalt_shader(shaders/path_wavefront_fused.metal ENTRY path_wavefront_fused OPTIONS --local-size 64)
basalt_shader(shaders/path_wavefront_fused_wide.metal ENTRY path_wavefront_fused_wide OPTIONS --local-size 64)
basalt_shader(shaders/path_wavefront_intersect.metal ENTRY path_wavefront_intersect OPTIONS --local-size 64)
basalt_shader(shaders/path_wavefront_intersect_wide.metal ENTRY path_wavefront_intersect_wide OPTIONS --local-size 64)
basalt_shader(shaders/path_wavefront_intersect_rt.metal ENTRY path_wavefront_intersect_rt OPTIONS --local-size 64)
basalt_shader(shaders/path_wavefront_intersect_pipeline.metal ENTRY path_wavefront_intersect_pipeline)
basalt_shader(shaders/path_wavefront_intersect_pipeline.metal ENTRY path_wave_pipeline_miss)
basalt_shader(shaders/path_wavefront_intersect_pipeline.metal ENTRY path_wave_pipeline_closest)
basalt_shader(shaders/path_wavefront_intersect_pipeline.metal ENTRY path_wave_pipeline_alpha)
basalt_shader(shaders/path_wavefront_shadow.metal ENTRY path_wavefront_shadow OPTIONS --local-size 64)
basalt_shader(shaders/path_wavefront_shadow_wide.metal ENTRY path_wavefront_shadow_wide OPTIONS --local-size 64)
basalt_shader(shaders/path_wavefront_shadow_rt.metal ENTRY path_wavefront_shadow_rt OPTIONS --local-size 64)
basalt_shader(shaders/path_wavefront_shadow_pipeline.metal ENTRY path_wavefront_shadow_pipeline)
basalt_shader(shaders/path_wavefront_shadow_pipeline.metal ENTRY path_wave_shadow_pipeline_miss)
basalt_shader(shaders/path_wavefront_shadow_pipeline.metal ENTRY path_wave_shadow_pipeline_closest)
basalt_shader(shaders/path_wavefront_shadow_pipeline.metal ENTRY path_wave_shadow_pipeline_alpha)
basalt_shader(shaders/path_wavefront_resolve.metal ENTRY path_wavefront_resolve OPTIONS --local-size 128)
basalt_shader(shaders/path_restir.metal ENTRY path_restir_initial OPTIONS --local-size 64)
basalt_shader(shaders/path_restir.metal ENTRY path_restir_spatial OPTIONS --local-size 64)
basalt_shader(shaders/path_export_guides.metal ENTRY path_export_guides OPTIONS --local-size 8 --local-size-y 8)
basalt_shader(shaders/path_compare.metal ENTRY path_compare OPTIONS --local-size 8 --local-size-y 8)
basalt_shader(shaders/ray_query_oracle.metal ENTRY ray_query_oracle OPTIONS --local-size 64)
basalt_shader(shaders/ray_pipeline_oracle.metal ENTRY pipeline_oracle_generate)
basalt_shader(shaders/ray_pipeline_oracle.metal ENTRY pipeline_oracle_miss)
basalt_shader(shaders/ray_pipeline_oracle.metal ENTRY pipeline_oracle_closest)
basalt_shader(shaders/ray_pipeline_oracle.metal ENTRY pipeline_oracle_alpha)
basalt_shader(shaders/bvh_oracle.metal ENTRY bvh_oracle OPTIONS --local-size 64)
basalt_shader(shaders/bvh_wide_oracle.metal ENTRY bvh_wide_oracle OPTIONS --local-size 64)
basalt_shader(shaders/raycone_oracle.metal ENTRY raycone_oracle OPTIONS --local-size 64)
basalt_shader(shaders/wavefront_enqueue.metal ENTRY wavefront_enqueue OPTIONS --local-size 64)
basalt_shader(shaders/wavefront_trace.metal ENTRY wavefront_trace OPTIONS --local-size 64)
basalt_shader(shaders/bvh_build.metal ENTRY bvh_build OPTIONS --local-size 1)
basalt_shader(shaders/path_temporal.metal ENTRY path_temporal OPTIONS --local-size 8 --local-size-y 8)
basalt_shader(shaders/path_temporal.metal ENTRY path_atrous OPTIONS --local-size 8 --local-size-y 8)
basalt_shader(shaders/path_temporal.metal ENTRY path_reconstruct OPTIONS --local-size 8 --local-size-y 8)

add_custom_target(basalt-shaders ALL DEPENDS ${BASALT_SHADER_OUTPUTS})

# Third-party sources, all permissive. See NOTICES.md.
add_library(basalt-asset-deps STATIC src/asset_deps.cpp)
target_include_directories(basalt-asset-deps PUBLIC third_party/cgltf third_party/stb)
if(MSVC)
  target_compile_options(basalt-asset-deps PRIVATE /W0)
endif()

add_library(basalt-deps STATIC
  third_party/imgui/imgui.cpp
  third_party/imgui/imgui_draw.cpp
  third_party/imgui/imgui_tables.cpp
  third_party/imgui/imgui_widgets.cpp
  third_party/imgui/imgui_demo.cpp
  third_party/imgui/backends/imgui_impl_win32.cpp
  src/deps.cpp)
target_include_directories(basalt-deps PUBLIC
  third_party/imgui
  third_party/imgui/backends
  third_party/cgltf
  third_party/stb
  third_party/vma/include)
target_link_libraries(basalt-deps PUBLIC Vulkan::Vulkan)
target_link_libraries(basalt-deps PUBLIC basalt-asset-deps)
if(MSVC)
  target_compile_options(basalt-deps PRIVATE /W0)
endif()

# The Metal2Vulkan host library: reflection reading, descriptor layouts and
# sampler descriptions, built from the copy under tools/ so the engine has one
# toolchain and one configuration.
add_library(m2v-host STATIC tools/metal2vulkan/src/m2v_host.cpp)
target_include_directories(m2v-host PUBLIC tools/metal2vulkan/src)
target_link_libraries(m2v-host PUBLIC Vulkan::Vulkan)
if(MSVC)
  target_compile_options(m2v-host PRIVATE /W0)
endif()

# The path tracer's CPU backend: the shared shader headers compiled as C++. No Vulkan.
include(cmake/prebuilt.cmake)
add_library(basalt-pt STATIC
  src/pt/Tracing.cpp
  src/pt/Bvh.cpp
  src/pt/BvhVariants.cpp
  src/pt/BvhAvx2.cpp
  src/pt/Tables.cpp
  src/pt/CpuTracer.cpp
  src/pt/Embree.cpp
  src/pt/Denoiser.cpp
  src/pt/Gltf.cpp
  src/pt/CaptureMetadata.cpp
  src/pt/ImageFile.cpp
  src/pt/WavefrontPlan.cpp)
target_include_directories(basalt-pt PUBLIC src shaders)
target_include_directories(basalt-pt PRIVATE third_party/cgltf third_party/stb)
target_compile_definitions(basalt-pt PUBLIC NOMINMAX)
target_link_libraries(basalt-pt PUBLIC basalt-asset-deps)
if(BASALT_WITH_EMBREE)
  target_link_libraries(basalt-pt PUBLIC basalt::embree)
  target_compile_definitions(basalt-pt PUBLIC BASALT_WITH_EMBREE=1)
endif()
if(BASALT_WITH_OIDN)
  target_link_libraries(basalt-pt PUBLIC basalt::oidn)
  target_compile_definitions(basalt-pt PUBLIC BASALT_WITH_OIDN=1)
endif()
if(MSVC)
  target_compile_options(basalt-pt PRIVATE /W4 /permissive-)
  set_source_files_properties(src/pt/BvhAvx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
endif()

add_executable(basalt-pt-cli src/pt/main.cpp)
target_link_libraries(basalt-pt-cli PRIVATE basalt-pt)
target_compile_definitions(basalt-pt-cli PRIVATE
  BASALT_GIT_REVISION="${BASALT_GIT_REVISION}" BASALT_GIT_DIRTY=${BASALT_GIT_DIRTY})
basalt_copy_runtime_dlls(basalt-pt-cli)
if(MSVC)
  target_compile_options(basalt-pt-cli PRIVATE /W4 /permissive-)
endif()

add_executable(basalt WIN32
  src/main.cpp
  src/core/Log.cpp
  src/core/Math.cpp
  src/platform/Window.cpp
  src/gpu/Context.cpp
  src/gpu/Swapchain.cpp
  src/gpu/Resources.cpp
  src/gpu/Shader.cpp
  src/gpu/Pipeline.cpp
  src/gpu/RayPipeline.cpp
  src/gpu/Descriptors.cpp
  src/gpu/Uploader.cpp
  src/scene/Gltf.cpp
  src/scene/Scene.cpp
  src/scene/Camera.cpp
  src/render/Renderer.cpp
  src/render/GpuBvhBuilder.cpp
  src/render/GpuProfiler.cpp
  src/render/PathReconstruction.cpp
  src/render/Environment.cpp
  src/render/RayTracing.cpp
  src/render/TraceScene.cpp
  src/render/UiPass.cpp)
# shaders/ for the headers the path tracer compiles as both MSL and C++.
target_include_directories(basalt PRIVATE src shaders)
target_link_libraries(basalt PRIVATE basalt-deps basalt-pt m2v-host Vulkan::Vulkan comdlg32 shell32)
add_dependencies(basalt basalt-shaders)
basalt_copy_runtime_dlls(basalt)
target_compile_definitions(basalt PRIVATE
  UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN
  BASALT_GIT_REVISION="${BASALT_GIT_REVISION}" BASALT_GIT_DIRTY=${BASALT_GIT_DIRTY}
  BASALT_SHADER_DIR="${BASALT_SHADER_DIR}"
  BASALT_MSL2SPIRV_VERSION="${BASALT_MSL2SPIRV_VERSION}"
  BASALT_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets")
if(MSVC)
  target_compile_options(basalt PRIVATE /W4 /permissive- /wd4324)
  set_property(TARGET basalt PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

enable_testing()
add_executable(temporal-motion-tests tests/temporal_motion_tests.cpp)
target_link_libraries(temporal-motion-tests PRIVATE basalt-pt)
target_compile_definitions(temporal-motion-tests PRIVATE BASALT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
basalt_copy_runtime_dlls(temporal-motion-tests)
add_test(NAME temporal-motion-tests COMMAND temporal-motion-tests)
add_test(NAME temporal-embree-tests COMMAND temporal-motion-tests --embree)
set_tests_properties(temporal-embree-tests PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME temporal-v7-motion-tests COMMAND temporal-motion-tests --v7)
add_test(NAME temporal-v7-embree-tests COMMAND temporal-motion-tests --v7 --embree)
set_tests_properties(temporal-v7-embree-tests PROPERTIES SKIP_RETURN_CODE 77)
add_executable(pt-tests tests/pt_tests.cpp)
target_link_libraries(pt-tests PRIVATE basalt-pt)
basalt_copy_runtime_dlls(pt-tests)
if(MSVC)
  target_compile_options(pt-tests PRIVATE /W4 /permissive-)
endif()
add_test(NAME pt-tests COMMAND pt-tests)
add_executable(gpu-ray-tests
  tests/gpu_ray_tests.cpp
  src/core/Log.cpp
  src/core/Math.cpp
  src/platform/Window.cpp
  src/gpu/Context.cpp
  src/gpu/Resources.cpp
  src/gpu/Shader.cpp
  src/gpu/Pipeline.cpp
  src/gpu/RayPipeline.cpp
  src/gpu/Descriptors.cpp
  src/gpu/Uploader.cpp
  src/scene/Scene.cpp
  src/render/RayTracing.cpp
  src/render/GpuBvhBuilder.cpp
  src/render/TraceScene.cpp)
target_include_directories(gpu-ray-tests PRIVATE src shaders)
target_link_libraries(gpu-ray-tests PRIVATE basalt-deps basalt-pt m2v-host Vulkan::Vulkan)
add_dependencies(gpu-ray-tests basalt-shaders)
target_compile_definitions(gpu-ray-tests PRIVATE
  UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN BASALT_SHADER_DIR="${BASALT_SHADER_DIR}")
if(MSVC)
  target_compile_options(gpu-ray-tests PRIVATE /W4 /permissive- /wd4324)
endif()
add_test(NAME gpu-ray-tests COMMAND gpu-ray-tests)
add_test(NAME gpu-ray-pipeline-tests COMMAND gpu-ray-tests --pipeline)
set_tests_properties(gpu-ray-pipeline-tests PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME gpu-ray-pipeline-disabled-tests COMMAND gpu-ray-tests --pipeline)
set_tests_properties(gpu-ray-pipeline-disabled-tests PROPERTIES SKIP_RETURN_CODE 77
  ENVIRONMENT "BASALT_NO_RAY_PIPELINE=1")
add_test(NAME gpu-bvh-tests COMMAND gpu-ray-tests --software)
add_test(NAME gpu-bvh-build-tests COMMAND gpu-ray-tests --gpu-builder)
add_test(NAME gpu-bvh4-tests COMMAND gpu-ray-tests --wide4)
add_test(NAME gpu-bvh8-tests COMMAND gpu-ray-tests --wide8)
add_test(NAME gpu-wavefront-tests COMMAND gpu-ray-tests --wavefront)
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND AND NOT BASALT_CPU_ONLY)
  add_test(NAME gpu-wavefront-render-tests COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/wavefront_conformance.py"
    --exe "$<TARGET_FILE:basalt>" --scene "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/materials.gltf"
    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/wavefront-conformance")
  set_tests_properties(gpu-wavefront-render-tests PROPERTIES SKIP_RETURN_CODE 77)
  add_test(NAME gpu-wavefront-rt-render-tests COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/wavefront_conformance.py"
    --exe "$<TARGET_FILE:basalt>" --scene "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/materials.gltf"
    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/wavefront-conformance" --hardware)
  set_tests_properties(gpu-wavefront-rt-render-tests PROPERTIES SKIP_RETURN_CODE 77)
  add_test(NAME gpu-ray-pipeline-render-tests COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/pipeline_conformance.py"
    --exe "$<TARGET_FILE:basalt>" --scene "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/materials.gltf"
    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/pipeline-conformance")
  set_tests_properties(gpu-ray-pipeline-render-tests PROPERTIES SKIP_RETURN_CODE 77)
  add_test(NAME gpu-sidedness-tests COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/sidedness_conformance.py"
    --exe "$<TARGET_FILE:basalt>" --scene "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/mirrored.gltf"
    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/sidedness")
  set_tests_properties(gpu-sidedness-tests PROPERTIES SKIP_RETURN_CODE 77)
  add_test(NAME gpu-v7-render-tests COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/v7_conformance.py"
    --exe "$<TARGET_FILE:basalt>" --scene "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/v7_materials.gltf"
    --raycone-scene "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/raycone.gltf"
    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/v7-conformance")
  set_tests_properties(gpu-v7-render-tests PROPERTIES SKIP_RETURN_CODE 77)
  # ReSTIR DI against NEE at convergence: two fixtures, every backend, four seeds.
  add_test(NAME gpu-restir-render-tests COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/restir_conformance.py"
    --exe "$<TARGET_FILE:basalt>" --data "${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/restir-conformance")
  set_tests_properties(gpu-restir-render-tests PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 3600)
  add_test(NAME cli-help COMMAND basalt --help)
  set_tests_properties(cli-help PROPERTIES PASS_REGULAR_EXPRESSION "--di-estimator nee")
  # OpenEXR captures equal PFM bit for bit (headless CPU and the application's GPU outputs).
  add_test(NAME exr-capture-tests COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/exr_captures.py"
    --pt-cli "$<TARGET_FILE:basalt-pt-cli>" --exe "$<TARGET_FILE:basalt>"
    --scene "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/materials.gltf"
    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/exr-captures")
  set_tests_properties(exr-capture-tests PROPERTIES SKIP_RETURN_CODE 77)
  add_test(NAME gpu-ray-pipeline-unavailable-renderer COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_capture_error.py"
    --exe "$<TARGET_FILE:basalt>" --output-dir "${CMAKE_CURRENT_BINARY_DIR}/negative-cli"
    --name pipeline-unavailable --expect "requires VK_KHR_ray_tracing_pipeline" --
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/materials.gltf" --renderer gpu-pipeline
    --spp 1 --no-ui --no-vsync)
  set_tests_properties(gpu-ray-pipeline-unavailable-renderer PROPERTIES
    ENVIRONMENT "BASALT_NO_RAY_PIPELINE=1")
  # Negative CLI cases: each must exit nonzero with its own reason and write no capture.
  function(basalt_negative_cli name expect)
    add_test(NAME cli-reject-${name} COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_capture_error.py"
      --exe "$<TARGET_FILE:basalt>" --output-dir "${CMAKE_CURRENT_BINARY_DIR}/negative-cli"
      --name ${name} --expect "${expect}" --
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/materials.gltf" --spp 1 --no-ui --no-vsync ${ARGN})
  endfunction()
  basalt_negative_cli(unknown-argument "unknown or incomplete argument" --rendrer gpu)
  basalt_negative_cli(missing-value "unknown or incomplete argument" --renderer gpu --seed)
  basalt_negative_cli(unknown-renderer "unknown renderer" --renderer photon)
  basalt_negative_cli(zero-bounces "--bounces needs an integer" --renderer gpu-bvh --bounces 0)
  basalt_negative_cli(malformed-spf "--spf needs an integer" --renderer gpu-bvh --spf 4x)
  basalt_negative_cli(gpu-intersector "--intersector requires --renderer cpu" --renderer gpu-bvh --intersector embree)
  basalt_negative_cli(gpu-builder-wide "quantized wide BVHs currently require" --renderer gpu-bvh --bvh-builder gpu --bvh-width bvh8)
  basalt_negative_cli(cpu-wavefront "--path-execution wavefront requires" --renderer cpu --path-execution wavefront)
  basalt_negative_cli(cpu-guides "--albedo-pfm and --normal-pfm require a GPU path tracer" --renderer cpu --albedo-pfm "${CMAKE_CURRENT_BINARY_DIR}/negative-cli/a.pfm")
  basalt_negative_cli(missing-scene "does-not-exist" --renderer gpu-bvh "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/does-not-exist.gltf")
  basalt_negative_cli(aperture-hybrid "--aperture needs traced primary rays" --renderer hybrid --aperture 0.1)
  basalt_negative_cli(aperture-raster "--aperture and --focus-distance require a path tracer" --renderer raster --aperture 0.1)
  basalt_negative_cli(aperture-negative "--aperture needs a non-negative number" --renderer gpu-bvh --aperture -1)
  basalt_negative_cli(restir-hybrid "--di-estimator restir needs traced primary rays" --renderer hybrid --di-estimator restir)
  basalt_negative_cli(restir-raster "--di-estimator restir requires a path tracer" --renderer raster --di-estimator restir)
  basalt_negative_cli(restir-spf "--spf must be 1" --renderer gpu-bvh --di-estimator restir --spf 4)
  basalt_negative_cli(restir-reuse-without "--restir-reuse and --restir-candidates require" --renderer gpu-bvh --restir-reuse both)
  basalt_negative_cli(restir-unknown "unknown direct-light estimator" --renderer gpu-bvh --di-estimator ris)
  add_test(NAME gpu-wavefront-batching-tests COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/wavefront_batching.py"
    --exe "$<TARGET_FILE:basalt>" --scene "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/materials.gltf"
    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/wavefront-batching")
  set_tests_properties(gpu-wavefront-batching-tests PROPERTIES SKIP_RETURN_CODE 77)
  add_test(NAME gpu-lifecycle-transition-tests COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/lifecycle_transitions.py"
    --exe "$<TARGET_FILE:basalt>" --scene "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/materials.gltf"
    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/lifecycle")
  set_tests_properties(gpu-lifecycle-transition-tests PROPERTIES SKIP_RETURN_CODE 77)
endif()
add_executable(gpu-temporal-tests tests/gpu_temporal_tests.cpp
  src/core/Log.cpp src/platform/Window.cpp src/gpu/Context.cpp src/gpu/Resources.cpp
  src/gpu/Shader.cpp src/gpu/Pipeline.cpp src/gpu/Descriptors.cpp src/gpu/Uploader.cpp
  src/render/PathReconstruction.cpp)
target_include_directories(gpu-temporal-tests PRIVATE src shaders)
target_link_libraries(gpu-temporal-tests PRIVATE basalt-deps m2v-host Vulkan::Vulkan)
add_dependencies(gpu-temporal-tests basalt-shaders)
target_compile_definitions(gpu-temporal-tests PRIVATE UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN
  BASALT_SHADER_DIR="${BASALT_SHADER_DIR}")
if(MSVC)
  target_compile_options(gpu-temporal-tests PRIVATE /W4 /permissive- /wd4324)
endif()
add_test(NAME gpu-temporal-tests COMMAND gpu-temporal-tests)
# The ray-cone level of detail, bitwise on the CPU and the GPU.
add_executable(gpu-raycone-tests tests/gpu_raycone_tests.cpp
  src/core/Log.cpp src/platform/Window.cpp src/gpu/Context.cpp src/gpu/Resources.cpp
  src/gpu/Shader.cpp src/gpu/Pipeline.cpp src/gpu/Descriptors.cpp src/gpu/Uploader.cpp)
target_include_directories(gpu-raycone-tests PRIVATE src shaders)
target_link_libraries(gpu-raycone-tests PRIVATE basalt-deps basalt-pt m2v-host Vulkan::Vulkan)
add_dependencies(gpu-raycone-tests basalt-shaders)
basalt_copy_runtime_dlls(gpu-raycone-tests)
target_compile_definitions(gpu-raycone-tests PRIVATE UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN
  BASALT_SHADER_DIR="${BASALT_SHADER_DIR}")
if(MSVC)
  target_compile_options(gpu-raycone-tests PRIVATE /W4 /permissive- /wd4324)
endif()
add_test(NAME gpu-raycone-tests COMMAND gpu-raycone-tests)
# Driver statistics (registers, spills) of the compute pipelines; a diagnostic, not a test.
add_executable(shader-stats tools/shader_stats.cpp
  src/core/Log.cpp src/core/Math.cpp src/platform/Window.cpp src/gpu/Context.cpp src/gpu/Resources.cpp
  src/gpu/Shader.cpp src/gpu/Pipeline.cpp src/gpu/Descriptors.cpp)
target_include_directories(shader-stats PRIVATE src shaders)
target_link_libraries(shader-stats PRIVATE basalt-deps m2v-host Vulkan::Vulkan)
add_dependencies(shader-stats basalt-shaders)
target_compile_definitions(shader-stats PRIVATE
  UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN BASALT_SHADER_DIR="${BASALT_SHADER_DIR}")

add_executable(gpu-motion-tests tests/gpu_motion_tests.cpp
  src/core/Log.cpp src/core/Math.cpp src/platform/Window.cpp src/gpu/Context.cpp src/gpu/Resources.cpp
  src/gpu/Shader.cpp src/gpu/Pipeline.cpp src/gpu/Descriptors.cpp src/gpu/Uploader.cpp
  src/scene/Scene.cpp src/render/RayTracing.cpp src/render/GpuBvhBuilder.cpp src/render/PathReconstruction.cpp)
target_include_directories(gpu-motion-tests PRIVATE src shaders)
target_link_libraries(gpu-motion-tests PRIVATE basalt-deps basalt-pt m2v-host Vulkan::Vulkan)
add_dependencies(gpu-motion-tests basalt-shaders)
basalt_copy_runtime_dlls(gpu-motion-tests)
target_compile_definitions(gpu-motion-tests PRIVATE UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN
  BASALT_SHADER_DIR="${BASALT_SHADER_DIR}" BASALT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
if(MSVC)
  target_compile_options(gpu-motion-tests PRIVATE /W4 /permissive- /wd4324)
endif()
add_test(NAME gpu-motion-tests COMMAND gpu-motion-tests)
add_test(NAME gpu-bvh-motion-tests COMMAND gpu-motion-tests --software)
add_test(NAME gpu-bvh-build-motion-tests COMMAND gpu-motion-tests --gpu-builder)
add_test(NAME gpu-v7-motion-tests COMMAND gpu-motion-tests --v7)
add_test(NAME gpu-bvh-v7-motion-tests COMMAND gpu-motion-tests --v7 --software)
add_test(NAME gpu-emissive-tests COMMAND gpu-motion-tests --emissive)
add_test(NAME gpu-bvh-emissive-tests COMMAND gpu-motion-tests --emissive-software)
add_test(NAME gpu-bvh-build-emissive-tests COMMAND gpu-motion-tests --emissive-gpu-builder)
set_tests_properties(gpu-motion-tests PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(gpu-emissive-tests PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(gpu-bvh-motion-tests PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(gpu-bvh-build-motion-tests PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(gpu-v7-motion-tests PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(gpu-bvh-v7-motion-tests PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(gpu-bvh-emissive-tests PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(gpu-bvh-build-emissive-tests PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(gpu-ray-tests PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(gpu-bvh-tests PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(gpu-bvh-build-tests PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(gpu-bvh4-tests PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(gpu-bvh8-tests PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(gpu-wavefront-tests PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME pt-headless-smoke COMMAND basalt-pt-cli
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/materials.gltf"
  --output "${CMAKE_CURRENT_BINARY_DIR}/pt-headless-smoke.pfm"
  --width 16 --height 16 --spp 1 --bounces 2 --threads 2)
add_test(NAME pt-headless-restir-smoke COMMAND basalt-pt-cli
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/manylights.gltf"
  --output "${CMAKE_CURRENT_BINARY_DIR}/pt-headless-restir-smoke.pfm"
  --width 32 --height 18 --spp 4 --bounces 2 --threads 2 --di-estimator restir)
add_test(NAME pt-headless-invalid-backend COMMAND basalt-pt-cli
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/materials.gltf" --intersector invalid)
set_tests_properties(pt-headless-invalid-backend PROPERTIES WILL_FAIL TRUE)
add_test(NAME pt-headless-load-failure COMMAND basalt-pt-cli
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/does-not-exist.gltf")
set_tests_properties(pt-headless-load-failure PROPERTIES WILL_FAIL TRUE)
