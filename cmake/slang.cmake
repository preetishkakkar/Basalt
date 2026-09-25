# The Slang shader compiler (https://github.com/shader-slang/slang), fetched once into .deps/
# (ignored) and checked against a pinned hash. slangc compiles the shaders to SPIR-V and the
# shared path-tracer modules to C++ for the CPU tracer; the C++ it generates needs the prelude
# headers in its include/ folder. Nothing from Slang ships with Basalt.

set(BASALT_SLANG_VERSION "2026.18.2" CACHE STRING "Version of Slang to fetch")
set(BASALT_SLANG_URL
    "https://github.com/shader-slang/slang/releases/download/v${BASALT_SLANG_VERSION}/slang-${BASALT_SLANG_VERSION}-windows-x86_64.zip"
    CACHE STRING "Release asset holding slangc")
set(BASALT_SLANG_SHA256 "747602aec6b3623658d55fea87492d71828e15d16802d7941204fde418ceee8e"
    CACHE STRING "SHA-256 of that asset; the download is rejected without a match")
set(BASALT_SLANG_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.deps/slang-${BASALT_SLANG_VERSION}"
    CACHE PATH "An unpacked Slang release (bin/slangc.exe, include/)")

function(basalt_fetch_slang)
  set(marker "${BASALT_SLANG_ROOT}/.basalt-unpacked")
  if(NOT EXISTS "${marker}")
    set(archive "${BASALT_SLANG_ROOT}.zip")
    get_filename_component(parent "${BASALT_SLANG_ROOT}" DIRECTORY)
    file(MAKE_DIRECTORY "${parent}")
    if(NOT EXISTS "${archive}")
      message(STATUS "Fetching Slang ${BASALT_SLANG_VERSION} (63 MB, once)")
      file(DOWNLOAD "${BASALT_SLANG_URL}" "${archive}.part" STATUS status TLS_VERIFY ON)
      list(GET status 0 code)
      if(NOT code EQUAL 0)
        list(GET status 1 reason)
        file(REMOVE "${archive}.part")
        message(FATAL_ERROR "Fetching Slang failed: ${reason}. Download ${BASALT_SLANG_URL} and unpack "
                            "it to ${BASALT_SLANG_ROOT}, or point -DBASALT_SLANG_ROOT at a copy.")
      endif()
      file(RENAME "${archive}.part" "${archive}")
    endif()
    file(SHA256 "${archive}" actual)
    if(NOT actual STREQUAL BASALT_SLANG_SHA256)
      file(REMOVE "${archive}")
      message(FATAL_ERROR "${archive} is not the expected build (SHA-256 ${actual}, "
                          "expected ${BASALT_SLANG_SHA256}).")
    endif()
    # The archive has no top-level folder: bin/, include/, lib/ ... unpack beside a marker.
    file(REMOVE_RECURSE "${BASALT_SLANG_ROOT}")
    file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${BASALT_SLANG_ROOT}")
    file(TOUCH "${marker}")
  endif()
  if(NOT EXISTS "${BASALT_SLANG_ROOT}/bin/slangc.exe")
    message(FATAL_ERROR "${BASALT_SLANG_ROOT} holds no bin/slangc.exe")
  endif()
endfunction()

basalt_fetch_slang()
set(BASALT_SLANGC "${BASALT_SLANG_ROOT}/bin/slangc.exe" CACHE FILEPATH "slangc executable" FORCE)
set(BASALT_SLANG_INCLUDE "${BASALT_SLANG_ROOT}/include")

# The CPU tracer's shared code, generated into <build>/generated for `target`: the module
# pt_cpu as C++ (pt_cpu.gen.cpp, which src/pt/SlangCpu.cpp compiles, and its header
# pt_cpu.gen.hpp) and the modules' public constants as pt_constants.gen.h.
function(basalt_slang_cpu target)
  set(dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
  set(slang "${CMAKE_CURRENT_SOURCE_DIR}/shaders/slang")
  file(MAKE_DIRECTORY "${dir}")
  # 41012 ("profile implicitly upgraded") names capabilities the module does not use.
  set(options -std 2026 -fp-mode precise -warnings-as-errors all -warnings-disable 41012
              -I "${slang}" -I "${slang}/basalt" -I "${slang}/pt")
  foreach(kind cpp hpp)
    set(output "${dir}/pt_cpu.gen.${kind}")
    add_custom_command(
      OUTPUT "${output}"
      COMMAND "${BASALT_SLANGC}" "${slang}/pt/pt_cpu.slang" -target ${kind} ${options}
              -o "${output}" -depfile "${output}.d"
      DEPENDS "${slang}/pt/pt_cpu.slang" "${BASALT_SLANGC}"
      DEPFILE "${output}.d"
      COMMENT "slangc pt_cpu.slang [${kind}]"
      VERBATIM)
  endforeach()
  set(modules basalt/basalt_types.slang basalt/basalt_shading.slang basalt/basalt_common.slang pt/pt_path.slang pt/pt_raycone.slang
              pt/pt_bsdf.slang pt/pt_bvh.slang pt/pt_wide_bvh.slang pt/pt_restir.slang pt/pt_wavefront.slang
              pt/pt_bvh_build.slang pt/pt_exact_float.slang)
  list(TRANSFORM modules PREPEND "${slang}/")
  list(JOIN modules "|" moduleArgument)
  add_custom_command(
    OUTPUT "${dir}/pt_constants.gen.h"
    COMMAND "${CMAKE_COMMAND}" "-DOUTPUT=${dir}/pt_constants.gen.h" "-DMODULES=${moduleArgument}"
            -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/slang_constants.cmake"
    DEPENDS ${modules} "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/slang_constants.cmake"
    COMMENT "Slang constants for the host"
    VERBATIM)
  set(generated "${dir}/pt_cpu.gen.cpp" "${dir}/pt_cpu.gen.hpp" "${dir}/pt_constants.gen.h")
  # The generated C++ is compiled only through src/pt/SlangCpu.cpp's #include, inside namespace pt;
  # listed for dependencies, never built alone.
  set_source_files_properties(${generated} PROPERTIES HEADER_FILE_ONLY ON)
  target_sources(${target} PRIVATE ${generated})
  target_include_directories(${target} PUBLIC "${dir}" "${BASALT_SLANG_INCLUDE}")
endfunction()
