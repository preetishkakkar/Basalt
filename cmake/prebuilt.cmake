# Optional prebuilt libraries for the path tracer, fetched once into .deps/ (ignored) and
# checked against pinned hashes: Intel Embree, an alternative CPU intersector, and Intel
# Open Image Denoise. Each is optional: a failed fetch warns and builds without it.

option(BASALT_WITH_EMBREE "Fetch Intel Embree for the CPU path tracer" ON)
option(BASALT_WITH_OIDN "Fetch Intel Open Image Denoise for the path tracer" ON)

set(BASALT_DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.deps")

# Leaves <out> set to the unpacked folder, or empty when the archive could not be had.
function(basalt_fetch_prebuilt name url sha256 out)
  set(archive "${BASALT_DEPS_DIR}/${name}.zip")
  set(folder "${BASALT_DEPS_DIR}/${name}")
  set(${out} "" PARENT_SCOPE)
  if(EXISTS "${folder}/.basalt-unpacked")
    set(${out} "${folder}" PARENT_SCOPE)
    return()
  endif()
  file(MAKE_DIRECTORY "${BASALT_DEPS_DIR}")
  if(NOT EXISTS "${archive}")
    message(STATUS "Fetching ${name}")
    file(DOWNLOAD "${url}" "${archive}.part" STATUS status TLS_VERIFY ON)
    list(GET status 0 code)
    if(NOT code EQUAL 0)
      list(GET status 1 reason)
      file(REMOVE "${archive}.part")
      message(WARNING "Could not fetch ${url}: ${reason}. Building without ${name}.")
      return()
    endif()
    file(RENAME "${archive}.part" "${archive}")
  endif()
  file(SHA256 "${archive}" actual)
  if(NOT actual STREQUAL sha256)
    file(REMOVE "${archive}")
    message(WARNING "${archive} is not the expected build (SHA-256 ${actual}, expected ${sha256}). "
                    "Building without ${name}.")
    return()
  endif()
  file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${BASALT_DEPS_DIR}")
  file(TOUCH "${folder}/.basalt-unpacked")
  set(${out} "${folder}" PARENT_SCOPE)
endfunction()

set(BASALT_RUNTIME_DLLS "")

if(BASALT_WITH_EMBREE)
  basalt_fetch_prebuilt(embree-4.4.1.x64.windows
    "https://github.com/RenderKit/embree/releases/download/v4.4.1/embree-4.4.1.x64.windows.zip"
    "5aa1fa3161a720f9c610b06c652b80f421cb2b1fe8221fdf92b74157336fdf8f" EMBREE_ROOT)
  if(EMBREE_ROOT)
    add_library(basalt::embree SHARED IMPORTED)
    set_target_properties(basalt::embree PROPERTIES
      IMPORTED_IMPLIB "${EMBREE_ROOT}/lib/embree4.lib"
      IMPORTED_LOCATION "${EMBREE_ROOT}/bin/embree4.dll"
      INTERFACE_INCLUDE_DIRECTORIES "${EMBREE_ROOT}/include")
    list(APPEND BASALT_RUNTIME_DLLS "${EMBREE_ROOT}/bin/embree4.dll" "${EMBREE_ROOT}/bin/tbbmalloc.dll")
    set(BASALT_TBB_DLL "${EMBREE_ROOT}/bin/tbb12.dll")
  else()
    set(BASALT_WITH_EMBREE OFF)
  endif()
endif()

if(BASALT_WITH_OIDN)
  basalt_fetch_prebuilt(oidn-2.5.1.x64.windows
    "https://github.com/RenderKit/oidn/releases/download/v2.5.1/oidn-2.5.1.x64.windows.zip"
    "f11f91bc072a5e3a564515724cb72ab8fcfbc445c84a84c197ff9b16cc01396f" OIDN_ROOT)
  if(OIDN_ROOT)
    add_library(basalt::oidn SHARED IMPORTED)
    set_target_properties(basalt::oidn PROPERTIES
      IMPORTED_IMPLIB "${OIDN_ROOT}/lib/OpenImageDenoise.lib"
      IMPORTED_LOCATION "${OIDN_ROOT}/bin/OpenImageDenoise.dll"
      INTERFACE_INCLUDE_DIRECTORIES "${OIDN_ROOT}/include")
    # The CPU device everywhere, the CUDA device where there is an NVIDIA GPU; the rest are left behind.
    list(APPEND BASALT_RUNTIME_DLLS
      "${OIDN_ROOT}/bin/OpenImageDenoise.dll" "${OIDN_ROOT}/bin/OpenImageDenoise_core.dll"
      "${OIDN_ROOT}/bin/OpenImageDenoise_device_cpu.dll" "${OIDN_ROOT}/bin/OpenImageDenoise_device_cuda.dll")
    # Both ship oneTBB 12; OIDN's is the newer build and serves both.
    set(BASALT_TBB_DLL "${OIDN_ROOT}/bin/tbb12.dll")
  else()
    set(BASALT_WITH_OIDN OFF)
  endif()
endif()

if(BASALT_TBB_DLL)
  list(APPEND BASALT_RUNTIME_DLLS "${BASALT_TBB_DLL}")
endif()

function(basalt_copy_runtime_dlls target)
  if(BASALT_RUNTIME_DLLS)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${BASALT_RUNTIME_DLLS} "$<TARGET_FILE_DIR:${target}>"
      VERBATIM)
  endif()
endfunction()
