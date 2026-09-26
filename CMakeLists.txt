cmake_minimum_required(VERSION 3.24)
project(basalt LANGUAGES CXX)

# Basalt: a Vulkan renderer whose shaders are Slang modules, compiled to SPIR-V by slangc and,
# for the CPU path tracer, to C++. Windows only.

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

# The Slang shader compiler (SPIR-V for the GPU, C++ for the CPU tracer's shared modules).
include(cmake/slang.cmake)

# A genuinely GPU-independent configuration: no Vulkan discovery, SPIR-V shaders,
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
    src/pt/ImageFile.cpp src/pt/EnvironmentSun.cpp src/pt/SlangCpu.cpp)
  basalt_slang_cpu(basalt-pt)
  target_include_directories(basalt-pt PUBLIC src)
  target_compile_definitions(basalt-pt PUBLIC SLANG_PRELUDE_NAMESPACE=slangcpp)
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
    # The generated C++ is small static functions; RelWithDebInfo's /Ob1 would inline none.
    set_source_files_properties(src/pt/SlangCpu.cpp PROPERTIES COMPILE_OPTIONS "/Ob2")
  endif()

  add_executable(basalt-pt-cli src/pt/main.cpp)
  target_link_libraries(basalt-pt-cli PRIVATE basalt-pt)
  target_compile_definitions(basalt-pt-cli PRIVATE
    BASALT_GIT_REVISION="${BASALT_GIT_REVISION}" BASALT_GIT_DIRTY=${BASALT_GIT_DIRTY}
    BASALT_SLANG_VERSION="${BASALT_SLANG_VERSION}")
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
    add_test(NAME pt-environment-sun COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/environment_sun.py"
      "$<TARGET_FILE:basalt-pt-cli>" "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/raycone.gltf"
      "${CMAKE_CURRENT_BINARY_DIR}/environment-sun")
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

set(BASALT_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${BASALT_SHADER_DIR}")
set(BASALT_SHADER_OUTPUTS "")

# basalt_slang_shader(<source.slang> ENTRY <name> STAGE <stage> [CAPABILITIES ...])
#
# Compiles one Slang entry point to <binary>/shaders/<name>.spv beside slangc's reflection JSON,
# then validates the module (slangc does not by default). Slang assigns every binding and the
# host builds its layouts from the JSON. CAPABILITIES extend the SPIR-V 1.6 profile, e.g.
# spvRayQueryKHR. The depfile makes edits to imported modules rebuild their dependents.
find_program(BASALT_SPIRV_VAL spirv-val HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED)
set(BASALT_SLANG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders/slang")
function(basalt_slang_shader source)
  cmake_parse_arguments(SH "" "ENTRY;STAGE" "CAPABILITIES" ${ARGN})
  set(spv "${BASALT_SHADER_DIR}/${SH_ENTRY}.spv")
  set(json "${BASALT_SHADER_DIR}/${SH_ENTRY}.json")
  set(profile spirv_1_6)
  foreach(capability IN LISTS SH_CAPABILITIES)
    string(APPEND profile "+${capability}")
  endforeach()
  # -fvk-use-entrypoint-name keeps the entry's own name in the SPIR-V (not "main"); Shader passes
  # it as pName. 41012 ("profile implicitly upgraded") names capabilities the module does not use.
  add_custom_command(
    OUTPUT "${spv}" "${json}"
    COMMAND "${BASALT_SLANGC}" "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
            -target spirv -profile ${profile} -std 2026 -O1
            -fp-mode precise -matrix-layout-column-major -default-image-format-unknown
            -fvk-use-entrypoint-name -warnings-as-errors all -warnings-disable 41012
            -I "${BASALT_SLANG_DIR}" -I "${BASALT_SLANG_DIR}/basalt" -I "${BASALT_SLANG_DIR}/pt"
            -entry ${SH_ENTRY} -stage ${SH_STAGE}
            -o "${spv}" -reflection-json "${json}" -depfile "${spv}.d"
    COMMAND "${BASALT_SPIRV_VAL}" --target-env vulkan1.3 "${spv}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${source}" "${BASALT_SLANGC}"
    DEPFILE "${spv}.d"
    COMMENT "slangc ${source} [${SH_ENTRY}]"
    VERBATIM)
  set(BASALT_SHADER_OUTPUTS ${BASALT_SHADER_OUTPUTS} "${spv}" "${json}" PARENT_SCOPE)
endfunction()

set(BASALT_SLANG_ENTRIES shaders/slang/entries)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/forward.slang ENTRY forward_vertex STAGE vertex)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/forward.slang ENTRY forward_fragment STAGE fragment)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/forward.slang ENTRY forward_fragment_rt STAGE fragment CAPABILITIES spvRayQueryKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/shadow.slang ENTRY shadow_vertex STAGE vertex)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/shadow.slang ENTRY shadow_fragment STAGE fragment)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/sky.slang ENTRY sky_vertex STAGE vertex)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/sky.slang ENTRY sky_fragment STAGE fragment)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/post.slang ENTRY post_vertex STAGE vertex)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/post.slang ENTRY post_fragment STAGE fragment)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bloom.slang ENTRY bloom_downsample STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bloom.slang ENTRY bloom_upsample STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/reflect.slang ENTRY resolve_reflections STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/reflect.slang ENTRY composite_reflections STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/reflect.slang ENTRY resolve_reflections_rt STAGE compute CAPABILITIES spvRayQueryKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/taa.slang ENTRY resolve_temporal STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/cluster.slang ENTRY cull_lights STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/ui.slang ENTRY ui_vertex STAGE vertex)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/ui.slang ENTRY ui_fragment STAGE fragment)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/ibl.slang ENTRY equirect_to_cube STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/ibl.slang ENTRY prefilter_specular STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_trace STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_trace_wide STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_trace_wide_emissive STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_trace_wide_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_trace_wide_emissive_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_trace_emissive STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_trace_rt STAGE compute CAPABILITIES spvRayQueryKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_trace_pipeline STAGE raygeneration CAPABILITIES spvRayTracingKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_pipeline_miss STAGE miss CAPABILITIES spvRayTracingKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_pipeline_closest STAGE closesthit CAPABILITIES spvRayTracingKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_pipeline_alpha STAGE anyhit CAPABILITIES spvRayTracingKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_trace_hybrid_rt STAGE compute CAPABILITIES spvRayQueryKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_shade STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_shade_atomic STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_fused STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_fused_wide STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_fused_wide_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_intersect STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_intersect_wide STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_intersect_wide_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_shadow STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_shadow_wide STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_shadow_wide_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_resolve STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_fused_rt STAGE compute CAPABILITIES spvRayQueryKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_intersect_rt STAGE compute CAPABILITIES spvRayQueryKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_shadow_rt STAGE compute CAPABILITIES spvRayQueryKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_intersect_pipeline STAGE raygeneration CAPABILITIES spvRayTracingKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wave_pipeline_miss STAGE miss CAPABILITIES spvRayTracingKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wave_pipeline_closest STAGE closesthit CAPABILITIES spvRayTracingKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wave_pipeline_alpha STAGE anyhit CAPABILITIES spvRayTracingKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_shadow_pipeline STAGE raygeneration CAPABILITIES spvRayTracingKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wave_shadow_pipeline_miss STAGE miss CAPABILITIES spvRayTracingKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wave_shadow_pipeline_alpha STAGE anyhit CAPABILITIES spvRayTracingKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_restir.slang ENTRY path_restir_initial STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_restir.slang ENTRY path_restir_spatial STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_export_guides.slang ENTRY path_export_guides STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_compare.slang ENTRY path_compare STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/oracles.slang ENTRY ray_query_oracle STAGE compute CAPABILITIES spvRayQueryKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/oracles.slang ENTRY pipeline_oracle_generate STAGE raygeneration CAPABILITIES spvRayTracingKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/oracles.slang ENTRY pipeline_oracle_miss STAGE miss CAPABILITIES spvRayTracingKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/oracles.slang ENTRY pipeline_oracle_closest STAGE closesthit CAPABILITIES spvRayTracingKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/oracles.slang ENTRY pipeline_oracle_alpha STAGE anyhit CAPABILITIES spvRayTracingKHR)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/oracles.slang ENTRY bvh_oracle STAGE compute)
# The binary tree's other traversals (--bvh-traversal).
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_trace_while_while STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_trace_emissive_while_while STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_intersect_while_while STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_fused_while_while STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_shadow_while_while STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/oracles.slang ENTRY bvh_oracle_while_while STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_trace_speculative STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_trace_emissive_speculative STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_intersect_speculative STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_fused_speculative STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_shadow_speculative STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/oracles.slang ENTRY bvh_oracle_speculative STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_trace_restart_trail STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_trace_emissive_restart_trail STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_intersect_restart_trail STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_fused_restart_trail STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_shadow_restart_trail STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/oracles.slang ENTRY bvh_oracle_restart_trail STAGE compute)
# The binary deep traversal stack (kBvhStackDeep).
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_trace_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_trace_emissive_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_intersect_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_fused_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_shadow_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/oracles.slang ENTRY bvh_oracle_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_trace_while_while_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_trace_emissive_while_while_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_intersect_while_while_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_fused_while_while_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_shadow_while_while_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/oracles.slang ENTRY bvh_oracle_while_while_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_trace_speculative_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_trace.slang ENTRY path_trace_emissive_speculative_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_intersect_speculative_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_fused_speculative_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_wavefront.slang ENTRY path_wavefront_shadow_speculative_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/oracles.slang ENTRY bvh_oracle_speculative_deep STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/oracles.slang ENTRY bvh_wide_oracle STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/raycone_oracle.slang ENTRY raycone_oracle STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/oracles.slang ENTRY wavefront_enqueue STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/oracles.slang ENTRY wavefront_trace STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_build.slang ENTRY bvh_build STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_cost.slang ENTRY bvh_cost STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_cost.slang ENTRY bvh_wide_cost STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/animate.slang ENTRY animate_vertices STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_collapse.slang ENTRY bvh_collapse_gather STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_collapse.slang ENTRY bvh_collapse_size STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_collapse.slang ENTRY bvh_collapse_roots STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_collapse.slang ENTRY bvh_collapse_emit STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_collapse.slang ENTRY bvh_collapse_emit_args STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_collapse.slang ENTRY bvh_collapse_rows STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_ploc.slang ENTRY bvh_ploc_init STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_ploc.slang ENTRY bvh_ploc_single STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_ploc.slang ENTRY bvh_ploc_neighbours STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_ploc.slang ENTRY bvh_ploc_scan_sums STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_ploc.slang ENTRY bvh_ploc_compact STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_ploc.slang ENTRY bvh_ploc_finish STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_ploc.slang ENTRY bvh_ploc_fused STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_ploc.slang ENTRY bvh_hploc STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_apetrei.slang ENTRY bvh_apetrei_single STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_apetrei.slang ENTRY bvh_apetrei_climb STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_apetrei.slang ENTRY bvh_apetrei_levels STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_apetrei.slang ENTRY bvh_apetrei_scatter STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_batched.slang ENTRY bvh_batched_build STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_sah.slang ENTRY bvh_sah_level STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_sah.slang ENTRY bvh_sah_sizes STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_sah.slang ENTRY bvh_sah_bases STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_sah.slang ENTRY bvh_sah_emit STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_sah.slang ENTRY bvh_sah_large_bounds STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_sah.slang ENTRY bvh_sah_large_bins STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_sah.slang ENTRY bvh_sah_large_scatter STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_clip.slang ENTRY bvh_clip_area_blocks STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_clip.slang ENTRY bvh_clip_sums STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_clip.slang ENTRY bvh_clip_count STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_clip.slang ENTRY bvh_clip_scan STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_clip.slang ENTRY bvh_clip_instances STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_clip.slang ENTRY bvh_clip_emit STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_lbvh.slang ENTRY bvh_lbvh_extents STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_lbvh.slang ENTRY bvh_lbvh_instances STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_lbvh.slang ENTRY bvh_lbvh_morton STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_lbvh.slang ENTRY bvh_lbvh_topology STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_lbvh.slang ENTRY bvh_lbvh_seed STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_lbvh.slang ENTRY bvh_lbvh_fit STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_lbvh.slang ENTRY bvh_lbvh_number STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_lbvh.slang ENTRY bvh_lbvh_emit STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_lbvh.slang ENTRY bvh_lbvh_finish STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_sort.slang ENTRY bvh_sort_histogram STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_sort.slang ENTRY bvh_sort_reduce STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_sort.slang ENTRY bvh_sort_blocks STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_sort.slang ENTRY bvh_sort_apply STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/bvh_sort.slang ENTRY bvh_sort_scatter STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_temporal.slang ENTRY path_temporal STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_temporal.slang ENTRY path_atrous STAGE compute)
basalt_slang_shader(${BASALT_SLANG_ENTRIES}/path_temporal.slang ENTRY path_reconstruct STAGE compute)

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

# The path tracer's CPU backend: the shared Slang modules generated as C++ (pt_cpu). No Vulkan.
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
  src/pt/EnvironmentSun.cpp
  src/pt/WavefrontPlan.cpp
  src/pt/SlangCpu.cpp)
basalt_slang_cpu(basalt-pt)
target_include_directories(basalt-pt PUBLIC src)
target_compile_definitions(basalt-pt PUBLIC SLANG_PRELUDE_NAMESPACE=slangcpp)
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
  # The generated C++ is small static functions; RelWithDebInfo's /Ob1 would inline none.
  set_source_files_properties(src/pt/SlangCpu.cpp PROPERTIES COMPILE_OPTIONS "/Ob2")
endif()

add_executable(basalt-pt-cli src/pt/main.cpp)
target_link_libraries(basalt-pt-cli PRIVATE basalt-pt)
target_compile_definitions(basalt-pt-cli PRIVATE
  BASALT_GIT_REVISION="${BASALT_GIT_REVISION}" BASALT_GIT_DIRTY=${BASALT_GIT_DIRTY}
  BASALT_SLANG_VERSION="${BASALT_SLANG_VERSION}")
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
  src/gpu/Shader.cpp src/gpu/SlangReflection.cpp src/core/Json.cpp
  src/gpu/Pipeline.cpp
  src/gpu/RayPipeline.cpp
  src/gpu/Descriptors.cpp
  src/gpu/Uploader.cpp
  src/scene/Gltf.cpp
  src/scene/Scene.cpp
  src/scene/Camera.cpp
  src/render/Renderer.cpp
  src/render/GpuBvhBuilder.cpp
  src/render/GpuLbvhBuilder.cpp
  src/render/GpuBvhCollapse.cpp
  src/render/GpuProfiler.cpp
  src/render/PathReconstruction.cpp
  src/render/Environment.cpp
  src/render/RayTracing.cpp
  src/render/TraceScene.cpp
  src/render/UiPass.cpp)
target_include_directories(basalt PRIVATE src)
target_link_libraries(basalt PRIVATE basalt-deps basalt-pt Vulkan::Vulkan comdlg32 shell32)
add_dependencies(basalt basalt-shaders)
basalt_copy_runtime_dlls(basalt)
target_compile_definitions(basalt PRIVATE
  UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN
  BASALT_GIT_REVISION="${BASALT_GIT_REVISION}" BASALT_GIT_DIRTY=${BASALT_GIT_DIRTY}
  BASALT_SHADER_DIR="${BASALT_SHADER_DIR}"
  BASALT_SLANG_VERSION="${BASALT_SLANG_VERSION}"
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
  src/gpu/Shader.cpp src/gpu/SlangReflection.cpp src/core/Json.cpp
  src/gpu/Pipeline.cpp
  src/gpu/RayPipeline.cpp
  src/gpu/Descriptors.cpp
  src/gpu/Uploader.cpp
  src/scene/Scene.cpp
  src/render/RayTracing.cpp
  src/render/GpuBvhBuilder.cpp
  src/render/GpuLbvhBuilder.cpp
  src/render/GpuBvhCollapse.cpp
  src/render/TraceScene.cpp)
target_include_directories(gpu-ray-tests PRIVATE src)
target_link_libraries(gpu-ray-tests PRIVATE basalt-deps basalt-pt Vulkan::Vulkan)
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
add_test(NAME gpu-bvh-lbvh-tests COMMAND gpu-ray-tests --gpu-lbvh)
add_test(NAME gpu-bvh4-tests COMMAND gpu-ray-tests --wide4)
add_test(NAME gpu-bvh8-tests COMMAND gpu-ray-tests --wide8)
add_test(NAME gpu-bvh-lbvh-bvh4-tests COMMAND gpu-ray-tests --gpu-lbvh-wide4)
add_test(NAME gpu-bvh-lbvh-bvh8-tests COMMAND gpu-ray-tests --gpu-lbvh-wide8)
add_test(NAME gpu-bvh-apetrei-tests COMMAND gpu-ray-tests --gpu-apetrei)
add_test(NAME gpu-bvh-apetrei-bvh8-tests COMMAND gpu-ray-tests --gpu-apetrei-wide8)
add_test(NAME gpu-bvh-batched-tests COMMAND gpu-ray-tests --gpu-batched)
add_test(NAME gpu-bvh-batched-bvh8-tests COMMAND gpu-ray-tests --gpu-batched-wide8)
add_test(NAME gpu-bvh-ploc-tests COMMAND gpu-ray-tests --gpu-ploc)
add_test(NAME gpu-bvh-ploc-bvh8-tests COMMAND gpu-ray-tests --gpu-ploc-wide8)
add_test(NAME gpu-bvh-plocpp-tests COMMAND gpu-ray-tests --gpu-plocpp)
add_test(NAME gpu-bvh-hploc-tests COMMAND gpu-ray-tests --gpu-hploc)
add_test(NAME gpu-bvh-hploc-bvh8-tests COMMAND gpu-ray-tests --gpu-hploc-wide8)
add_test(NAME gpu-bvh-sah-tests COMMAND gpu-ray-tests --gpu-sah)
add_test(NAME gpu-bvh-sah-bvh8-tests COMMAND gpu-ray-tests --gpu-sah-wide8)
# The binary layout's other traversals, on the CPU builder's tree and the parallel LBVH's.
foreach(traversal while-while speculative restart-trail)
  add_test(NAME gpu-bvh-traversal-${traversal}-tests COMMAND gpu-ray-tests --software --traversal ${traversal})
  add_test(NAME gpu-bvh-lbvh-traversal-${traversal}-tests COMMAND gpu-ray-tests --gpu-lbvh --traversal ${traversal})
endforeach()
# The deep-stack kernels (trees deeper than 64), on the PLOC tree.
add_test(NAME gpu-bvh-deep-tests COMMAND gpu-ray-tests --gpu-ploc --deep)
foreach(traversal while-while speculative)
  add_test(NAME gpu-bvh-deep-${traversal}-tests COMMAND gpu-ray-tests --gpu-ploc --traversal ${traversal} --deep)
endforeach()
# Early split clipping: every builder over the same references.
foreach(mode builder lbvh apetrei batched ploc plocpp hploc sah lbvh-wide8 sah-wide8)
  add_test(NAME gpu-bvh-split-${mode}-tests COMMAND gpu-ray-tests --gpu-${mode} --split-clipping)
endforeach()
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
  add_test(NAME gpu-bvh-lbvh-identity-tests COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/bvh_builder_identity.py"
    --exe "$<TARGET_FILE:basalt>" --data "${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/bvh-builder-identity")
  set_tests_properties(gpu-bvh-lbvh-identity-tests PROPERTIES SKIP_RETURN_CODE 77)
  add_test(NAME gpu-bvh-ploc-render-tests COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/bvh_ploc_conformance.py"
    --exe "$<TARGET_FILE:basalt>" --data "${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/bvh-ploc-conformance")
  set_tests_properties(gpu-bvh-ploc-render-tests PROPERTIES SKIP_RETURN_CODE 77)
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
  # The rasteriser's traced occlusion modes, read through the occlusion debug view.
  add_test(NAME gpu-raster-occlusion-tests COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/raster_occlusion.py"
    --exe "$<TARGET_FILE:basalt>" --output-dir "${CMAKE_CURRENT_BINARY_DIR}/raster-occlusion")
  set_tests_properties(gpu-raster-occlusion-tests PROPERTIES SKIP_RETURN_CODE 77)
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
  # An .hdr's sun moved into the analytic disc lights the scene as it did from the image.
  add_test(NAME pt-environment-sun COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/environment_sun.py"
    "$<TARGET_FILE:basalt-pt-cli>" "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/raycone.gltf"
    "${CMAKE_CURRENT_BINARY_DIR}/environment-sun")
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
  basalt_negative_cli(unknown-ao-mode "--ao takes 0" --renderer raster --ao 3)
  basalt_negative_cli(unknown-environment-sun "unknown environment sun mode" --environment-sun always)
  basalt_negative_cli(zero-bounces "--bounces needs an integer" --renderer gpu-bvh --bounces 0)
  basalt_negative_cli(malformed-spf "--spf needs an integer" --renderer gpu-bvh --spf 4x)
  basalt_negative_cli(gpu-intersector "--intersector requires --renderer cpu" --renderer gpu-bvh --intersector embree)
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
  basalt_negative_cli(animate-renderer "--animate requires --renderer gpu-bvh" --renderer gpu --animate both)
  basalt_negative_cli(animate-temporal "--animate cannot be combined with --path-temporal" --renderer gpu-bvh --animate instances --path-temporal)
  basalt_negative_cli(unknown-animation "unknown animation" --renderer gpu-bvh --animate swirl)
  basalt_negative_cli(unknown-bvh-update "unknown BVH update" --renderer gpu-bvh --bvh-update sometimes)
  basalt_negative_cli(unknown-split-clipping "unknown BVH split clipping" --renderer gpu-bvh --bvh-split-clipping -1)
  basalt_negative_cli(split-clipping-renderer "--bvh-split-clipping requires --renderer gpu-bvh" --renderer gpu --bvh-split-clipping on)
  basalt_negative_cli(unknown-bvh-traversal "unknown BVH traversal" --renderer gpu-bvh --bvh-traversal if-if)
  basalt_negative_cli(bvh-traversal-wide "--bvh-traversal applies to the binary layout" --renderer gpu-bvh --bvh-width bvh8 --bvh-traversal restart-trail)
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
  src/gpu/Shader.cpp src/gpu/SlangReflection.cpp src/core/Json.cpp src/gpu/Pipeline.cpp src/gpu/Descriptors.cpp src/gpu/Uploader.cpp
  src/render/PathReconstruction.cpp)
target_include_directories(gpu-temporal-tests PRIVATE src)
target_link_libraries(gpu-temporal-tests PRIVATE basalt-pt basalt-deps Vulkan::Vulkan)
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
  src/gpu/Shader.cpp src/gpu/SlangReflection.cpp src/core/Json.cpp src/gpu/Pipeline.cpp src/gpu/Descriptors.cpp src/gpu/Uploader.cpp)
target_include_directories(gpu-raycone-tests PRIVATE src)
target_link_libraries(gpu-raycone-tests PRIVATE basalt-deps basalt-pt Vulkan::Vulkan)
add_dependencies(gpu-raycone-tests basalt-shaders)
basalt_copy_runtime_dlls(gpu-raycone-tests)
target_compile_definitions(gpu-raycone-tests PRIVATE UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN
  BASALT_SHADER_DIR="${BASALT_SHADER_DIR}")
if(MSVC)
  target_compile_options(gpu-raycone-tests PRIVATE /W4 /permissive- /wd4324)
endif()
add_test(NAME gpu-raycone-tests COMMAND gpu-raycone-tests)
# Driver statistics (registers, spills) of the compute and ray pipelines; a diagnostic, not a test.
add_executable(shader-stats tools/shader_stats.cpp
  src/core/Log.cpp src/core/Math.cpp src/platform/Window.cpp src/gpu/Context.cpp src/gpu/Resources.cpp
  src/gpu/Shader.cpp src/gpu/SlangReflection.cpp src/core/Json.cpp src/gpu/Pipeline.cpp src/gpu/Descriptors.cpp
  src/gpu/RayPipeline.cpp)
target_include_directories(shader-stats PRIVATE src)
target_link_libraries(shader-stats PRIVATE basalt-deps Vulkan::Vulkan)
add_dependencies(shader-stats basalt-shaders)
target_compile_definitions(shader-stats PRIVATE
  UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN BASALT_SHADER_DIR="${BASALT_SHADER_DIR}")

add_executable(gpu-motion-tests tests/gpu_motion_tests.cpp
  src/core/Log.cpp src/core/Math.cpp src/platform/Window.cpp src/gpu/Context.cpp src/gpu/Resources.cpp
  src/gpu/Shader.cpp src/gpu/SlangReflection.cpp src/core/Json.cpp src/gpu/Pipeline.cpp src/gpu/Descriptors.cpp src/gpu/Uploader.cpp
  src/scene/Scene.cpp src/render/RayTracing.cpp src/render/GpuBvhBuilder.cpp src/render/GpuLbvhBuilder.cpp src/render/GpuBvhCollapse.cpp src/render/PathReconstruction.cpp)
target_include_directories(gpu-motion-tests PRIVATE src)
target_link_libraries(gpu-motion-tests PRIVATE basalt-deps basalt-pt Vulkan::Vulkan)
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
