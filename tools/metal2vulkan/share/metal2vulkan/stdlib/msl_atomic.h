#pragma once
#include "msl_prelude.h"
#include "msl_sync.h"
// Independently authored declarations. Clang AtomicType preserves the distinction
// from ordinary integers; only explicit supported operations reach shader IR.
// Every operation has the Metal 4.1 overload with trailing mem_flags naming the
// address spaces its memory order applies to.
namespace metal {
using atomic_int = _Atomic(int);
using atomic_uint = _Atomic(uint);
#define M2V_ATOMIC_RMW(NAME, A, T, SPACE) \
  T NAME(volatile SPACE A* object, T value, memory_order order) __attribute__((annotate("msl.atomic:" #NAME))); \
  T NAME(volatile SPACE A* object, T value, memory_order order, mem_flags flags) __attribute__((annotate("msl.atomic:" #NAME)));
#define M2V_ATOMIC(A, T, SPACE) \
  T atomic_load_explicit(const volatile SPACE A* object, memory_order order) __attribute__((annotate("msl.atomic:atomic_load_explicit"))); \
  T atomic_load_explicit(const volatile SPACE A* object, memory_order order, mem_flags flags) __attribute__((annotate("msl.atomic:atomic_load_explicit"))); \
  void atomic_store_explicit(volatile SPACE A* object, T value, memory_order order) __attribute__((annotate("msl.atomic:atomic_store_explicit"))); \
  void atomic_store_explicit(volatile SPACE A* object, T value, memory_order order, mem_flags flags) __attribute__((annotate("msl.atomic:atomic_store_explicit"))); \
  M2V_ATOMIC_RMW(atomic_exchange_explicit, A, T, SPACE) \
  M2V_ATOMIC_RMW(atomic_fetch_add_explicit, A, T, SPACE) \
  M2V_ATOMIC_RMW(atomic_fetch_sub_explicit, A, T, SPACE) \
  M2V_ATOMIC_RMW(atomic_fetch_min_explicit, A, T, SPACE) \
  M2V_ATOMIC_RMW(atomic_fetch_max_explicit, A, T, SPACE) \
  M2V_ATOMIC_RMW(atomic_fetch_and_explicit, A, T, SPACE) \
  M2V_ATOMIC_RMW(atomic_fetch_or_explicit, A, T, SPACE) \
  M2V_ATOMIC_RMW(atomic_fetch_xor_explicit, A, T, SPACE) \
  bool atomic_compare_exchange_weak_explicit(volatile SPACE A* object, thread T* expected, T desired, memory_order success, memory_order failure) \
    __attribute__((annotate("msl.atomic:atomic_compare_exchange_weak_explicit"))); \
  bool atomic_compare_exchange_weak_explicit(volatile SPACE A* object, thread T* expected, T desired, memory_order success, memory_order failure, mem_flags flags) \
    __attribute__((annotate("msl.atomic:atomic_compare_exchange_weak_explicit")));
M2V_ATOMIC(atomic_int, int, device)
M2V_ATOMIC(atomic_uint, uint, device)
M2V_ATOMIC(atomic_int, int, threadgroup)
M2V_ATOMIC(atomic_uint, uint, threadgroup)
#undef M2V_ATOMIC
// atomic_bool (MSL 2.4): a four-byte word holding 0 or 1, with load, store,
// exchange and compare-exchange in device and threadgroup storage.
using atomic_bool = _Atomic(bool);
#define M2V_ATOMIC_BOOL(SPACE) \
  bool atomic_load_explicit(const volatile SPACE atomic_bool* object, memory_order order) __attribute__((annotate("msl.atomic:atomic_load_explicit"))); \
  bool atomic_load_explicit(const volatile SPACE atomic_bool* object, memory_order order, mem_flags flags) __attribute__((annotate("msl.atomic:atomic_load_explicit"))); \
  void atomic_store_explicit(volatile SPACE atomic_bool* object, bool value, memory_order order) __attribute__((annotate("msl.atomic:atomic_store_explicit"))); \
  void atomic_store_explicit(volatile SPACE atomic_bool* object, bool value, memory_order order, mem_flags flags) __attribute__((annotate("msl.atomic:atomic_store_explicit"))); \
  M2V_ATOMIC_RMW(atomic_exchange_explicit, atomic_bool, bool, SPACE) \
  bool atomic_compare_exchange_weak_explicit(volatile SPACE atomic_bool* object, thread bool* expected, bool desired, memory_order success, memory_order failure) \
    __attribute__((annotate("msl.atomic:atomic_compare_exchange_weak_explicit"))); \
  bool atomic_compare_exchange_weak_explicit(volatile SPACE atomic_bool* object, thread bool* expected, bool desired, memory_order success, memory_order failure, mem_flags flags) \
    __attribute__((annotate("msl.atomic:atomic_compare_exchange_weak_explicit")));
M2V_ATOMIC_BOOL(device)
M2V_ATOMIC_BOOL(threadgroup)
#undef M2V_ATOMIC_BOOL
// 64-bit: relaxed min/max on device memory returning nothing (MSL 2.4).
using atomic_ulong = _Atomic(ulong);
void atomic_max_explicit(volatile device atomic_ulong* object, ulong value, memory_order order) __attribute__((annotate("msl.atomic:atomic_max_explicit")));
void atomic_min_explicit(volatile device atomic_ulong* object, ulong value, memory_order order) __attribute__((annotate("msl.atomic:atomic_min_explicit")));
// Float: load, store, exchange, compare-exchange, add and subtract on device memory.
using atomic_float = _Atomic(float);
float atomic_load_explicit(const volatile device atomic_float* object, memory_order order) __attribute__((annotate("msl.atomic:atomic_load_explicit")));
float atomic_load_explicit(const volatile device atomic_float* object, memory_order order, mem_flags flags) __attribute__((annotate("msl.atomic:atomic_load_explicit")));
void atomic_store_explicit(volatile device atomic_float* object, float value, memory_order order) __attribute__((annotate("msl.atomic:atomic_store_explicit")));
void atomic_store_explicit(volatile device atomic_float* object, float value, memory_order order, mem_flags flags) __attribute__((annotate("msl.atomic:atomic_store_explicit")));
M2V_ATOMIC_RMW(atomic_exchange_explicit, atomic_float, float, device)
M2V_ATOMIC_RMW(atomic_fetch_add_explicit, atomic_float, float, device)
M2V_ATOMIC_RMW(atomic_fetch_sub_explicit, atomic_float, float, device)
bool atomic_compare_exchange_weak_explicit(volatile device atomic_float* object, thread float* expected, float desired, memory_order success, memory_order failure)
  __attribute__((annotate("msl.atomic:atomic_compare_exchange_weak_explicit")));
bool atomic_compare_exchange_weak_explicit(volatile device atomic_float* object, thread float* expected, float desired, memory_order success, memory_order failure, mem_flags flags)
  __attribute__((annotate("msl.atomic:atomic_compare_exchange_weak_explicit")));
#undef M2V_ATOMIC_RMW
}
