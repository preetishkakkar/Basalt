#pragma once
// Independently authored subset of the MSL visible function table interface (MSL 5.1.4,
// docs/INDIRECT_CALLS.md). A visible_function_table<R(Args...)> entry parameter is a read-only buffer
// of one uint per slot, the index of the [[visible]] function the host placed there; table[i] is a
// function pointer, which this compiler represents as that index, and a call through it is a finite
// switch over the [[visible]] functions of the pointer's type. Nothing here has a body.
namespace metal {
template <typename Signature> struct visible_function_table;
template <typename R, typename... Args> struct visible_function_table<R(Args...)> {
  typedef R (*function_pointer)(Args...);
  function_pointer operator[](uint index) const;
  uint size() const;
  // A table inside an argument buffer lives in the buffer's address space (MSL 2.13).
  function_pointer operator[](uint index) const constant;
  function_pointer operator[](uint index) const device;
  uint size() const constant;
  uint size() const device;
};
}
