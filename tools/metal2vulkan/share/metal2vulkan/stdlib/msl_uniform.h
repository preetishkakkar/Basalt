#pragma once
#include "msl_prelude.h"
// Independently authored uniform<T> (MSL 2.14): a value promised to be the
// same for every thread of a draw or dispatch. The compiler represents it as
// its underlying arithmetic or vector type and verifies the promise at every
// assignment (docs/LANGUAGE.md); nothing here has a body.
namespace metal {
template <typename T> struct uniform {
  uniform() = default;
  uniform(const uniform &) = default;
  uniform(T value);
  operator T() const;
  uniform &operator=(const uniform &) = default;
  uniform &operator=(T value);
  // Increments and compound assignments (statements only).
  uniform &operator++();
  uniform operator++(int);
  uniform &operator--();
  uniform operator--(int);
  uniform &operator+=(T value);
  uniform &operator-=(T value);
  uniform &operator*=(T value);
  uniform &operator/=(T value);
  uniform &operator%=(T value);
  uniform &operator&=(T value);
  uniform &operator|=(T value);
  uniform &operator^=(T value);
  uniform &operator<<=(T value);
  uniform &operator>>=(T value);
};
}
