# Locates the shader compiler. It is not committed: it is a 46 MB binary, so it
# ships as a release asset and is fetched once into tools/metal2vulkan/bin/,
# where the rest of the bundle already lives. A copy already in the tree is used
# as it stands, so a locally built compiler is never overwritten.

set(BASALT_MSL2SPIRV_VERSION "1.0.0"
    CACHE STRING "Version of msl2spirv to fetch")
set(BASALT_MSL2SPIRV_URL
    "https://github.com/preetishkakkar/Basalt/releases/download/msl2spirv-${BASALT_MSL2SPIRV_VERSION}/msl2spirv-${BASALT_MSL2SPIRV_VERSION}-windows-x64.zip"
    CACHE STRING "Release asset holding msl2spirv.exe")
set(BASALT_MSL2SPIRV_SHA256 "f8fff3537e8a475c4c5741336da0b352ff0bb395a1185d31be1e3e467b86196d"
    CACHE STRING "SHA-256 of that asset; the download is rejected without a match")
option(BASALT_DOWNLOAD_MSL2SPIRV "Fetch msl2spirv when the tree has no copy" ON)

set(BASALT_MSL2SPIRV "${CMAKE_CURRENT_SOURCE_DIR}/tools/metal2vulkan/bin/msl2spirv.exe"
    CACHE FILEPATH "msl2spirv executable that compiles Metal shaders to SPIR-V")
set(BASALT_MSL_STDLIB "${CMAKE_CURRENT_SOURCE_DIR}/tools/metal2vulkan/share/metal2vulkan/stdlib"
    CACHE PATH "The owned Metal standard library msl2spirv parses against")

function(basalt_fetch_msl2spirv)
  if(EXISTS "${BASALT_MSL2SPIRV}")
    return()
  endif()

  set(manual
      "Either download ${BASALT_MSL2SPIRV_URL} and unpack msl2spirv.exe to ${BASALT_MSL2SPIRV}, or point -DBASALT_MSL2SPIRV=<path> at a copy you already have. Then configure again.")

  if(NOT BASALT_DOWNLOAD_MSL2SPIRV)
    message(FATAL_ERROR "msl2spirv is not at ${BASALT_MSL2SPIRV} and fetching is off.\n${manual}")
  endif()
  if(BASALT_MSL2SPIRV_SHA256 STREQUAL "")
    message(FATAL_ERROR "BASALT_MSL2SPIRV_SHA256 is empty, so a download cannot be checked.\n${manual}")
  endif()

  set(staging "${CMAKE_CURRENT_BINARY_DIR}/msl2spirv-download")
  set(archive "${staging}/msl2spirv-${BASALT_MSL2SPIRV_VERSION}.zip")
  file(REMOVE_RECURSE "${staging}")
  file(MAKE_DIRECTORY "${staging}")

  message(STATUS "Fetching msl2spirv ${BASALT_MSL2SPIRV_VERSION} (46 MB, once)")
  # The hash is checked below rather than by EXPECTED_HASH, which aborts inside
  # file(DOWNLOAD) before this can say what to do about it.
  file(DOWNLOAD "${BASALT_MSL2SPIRV_URL}" "${archive}"
       STATUS status TLS_VERIFY ON SHOW_PROGRESS)
  list(GET status 0 code)
  list(GET status 1 reason)
  if(NOT code EQUAL 0)
    file(REMOVE_RECURSE "${staging}")
    message(FATAL_ERROR "Fetching msl2spirv failed: ${reason}.\n${manual}")
  endif()

  file(SHA256 "${archive}" actual)
  if(NOT actual STREQUAL BASALT_MSL2SPIRV_SHA256)
    file(REMOVE_RECURSE "${staging}")
    message(FATAL_ERROR
            "The asset at ${BASALT_MSL2SPIRV_URL} is not the expected build:\n"
            "  expected SHA-256 ${BASALT_MSL2SPIRV_SHA256}\n"
            "  received SHA-256 ${actual}\n"
            "Nothing was installed.\n${manual}")
  endif()

  file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${staging}/unpacked")
  file(GLOB_RECURSE unpacked "${staging}/unpacked/msl2spirv.exe")
  if(NOT unpacked)
    file(REMOVE_RECURSE "${staging}")
    message(FATAL_ERROR "The asset at ${BASALT_MSL2SPIRV_URL} holds no msl2spirv.exe.\n${manual}")
  endif()
  list(GET unpacked 0 executable)

  # Into place only once it is whole, so an interrupted configure leaves nothing
  # half-written for the next one to trust.
  get_filename_component(destination "${BASALT_MSL2SPIRV}" DIRECTORY)
  file(MAKE_DIRECTORY "${destination}")
  file(COPY_FILE "${executable}" "${BASALT_MSL2SPIRV}")
  file(REMOVE_RECURSE "${staging}")
  message(STATUS "msl2spirv is at ${BASALT_MSL2SPIRV}")
endfunction()
