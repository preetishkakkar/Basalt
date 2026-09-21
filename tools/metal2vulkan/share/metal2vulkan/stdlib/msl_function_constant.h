#pragma once
#include "msl_prelude.h"
// Independently authored declaration for the function-constant query (MSL
// section 5.8). The argument names a program-scope [[function_constant(n)]]
// variable; the compiler resolves the name and never evaluates the value, so
// there is no body here and no overload set over the scalar types.
namespace metal {
template <typename T>
bool is_function_constant_defined(T value) __attribute__((annotate("msl.math:is_function_constant_defined")));
}
