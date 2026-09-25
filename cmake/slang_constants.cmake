# Script mode: writes the public constants of Slang modules as a C++ header, so the host and the
# shaders read one definition. Every `public static const <uint|int|float> <name> = <literal>;`
# of the listed modules becomes `inline constexpr` in namespace pt.
#   cmake -DOUTPUT=<header> -DMODULES=<a.slang|b.slang> -P slang_constants.cmake
if(NOT OUTPUT OR NOT MODULES)
  message(FATAL_ERROR "slang_constants.cmake needs OUTPUT and MODULES")
endif()
string(REPLACE "|" ";" MODULES "${MODULES}")

set(body "")
foreach(module IN LISTS MODULES)
  file(STRINGS "${module}" lines REGEX "^public static const ")
  get_filename_component(name "${module}" NAME)
  string(APPEND body "\n// ${name}\n")
  foreach(line IN LISTS lines)
    if(NOT line MATCHES "^public static const (uint|int|float) ([A-Za-z_][A-Za-z0-9_]*) = ([^;]+);")
      message(FATAL_ERROR "${name}: cannot translate '${line}'")
    endif()
    set(type "${CMAKE_MATCH_1}")
    if(type STREQUAL "uint")
      set(type "std::uint32_t")
    elseif(type STREQUAL "int")
      set(type "std::int32_t")
    endif()
    string(APPEND body "inline constexpr ${type} ${CMAKE_MATCH_2} = ${CMAKE_MATCH_3};\n")
  endforeach()
endforeach()

set(text "// Generated from the Slang modules' public constants by cmake/slang_constants.cmake.\n#pragma once\n#include <cstdint>\n\nnamespace pt {\n${body}\n} // namespace pt\n")
# Rewritten only when it changes, so dependents rebuild only then.
if(EXISTS "${OUTPUT}")
  file(READ "${OUTPUT}" previous)
  if(previous STREQUAL text)
    return()
  endif()
endif()
file(WRITE "${OUTPUT}" "${text}")
