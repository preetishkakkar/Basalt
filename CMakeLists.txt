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

find_package(Vulkan REQUIRED)

# ---------------------------------------------------------------------------
# The Metal shader compiler, fetched on the first configure.
# ---------------------------------------------------------------------------
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

add_custom_target(basalt-shaders ALL DEPENDS ${BASALT_SHADER_OUTPUTS})

# ---------------------------------------------------------------------------
# Third-party sources, all permissive. See NOTICES.md.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# The engine.
# ---------------------------------------------------------------------------
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
  src/gpu/Descriptors.cpp
  src/gpu/Uploader.cpp
  src/scene/Gltf.cpp
  src/scene/Scene.cpp
  src/scene/Camera.cpp
  src/render/Renderer.cpp
  src/render/Environment.cpp
  src/render/RayTracing.cpp
  src/render/UiPass.cpp)
target_include_directories(basalt PRIVATE src)
target_link_libraries(basalt PRIVATE basalt-deps m2v-host Vulkan::Vulkan comdlg32 shell32)
add_dependencies(basalt basalt-shaders)
target_compile_definitions(basalt PRIVATE
  UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN
  BASALT_SHADER_DIR="${BASALT_SHADER_DIR}"
  BASALT_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets")
if(MSVC)
  target_compile_options(basalt PRIVATE /W4 /permissive- /wd4324)
  set_property(TARGET basalt PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif()
