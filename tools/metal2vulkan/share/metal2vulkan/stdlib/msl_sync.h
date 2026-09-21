#pragma once
// Independently authored subset of the Metal synchronization interface: memory
// orders, thread scopes, memory flags, barriers and fences. Values follow the
// MSL enumeration order; the compiler maps them to SPIR-V memory semantics and
// scopes as docs/MEMORY_MODEL.md records.
namespace metal {
enum memory_order { memory_order_relaxed = 0, memory_order_acquire = 2,
                    memory_order_release = 3, memory_order_acq_rel = 4, memory_order_seq_cst = 5 };
enum thread_scope { thread_scope_thread = 0, thread_scope_simdgroup = 1, thread_scope_threadgroup = 2, thread_scope_device = 3 };
enum class mem_flags { mem_none = 0, mem_device = 1, mem_threadgroup = 2, mem_texture = 4, mem_threadgroup_imageblock = 8, mem_object_data = 16 };
constexpr mem_flags operator|(mem_flags a, mem_flags b) { return mem_flags(int(a) | int(b)); }
void threadgroup_barrier(mem_flags flags, memory_order order = memory_order_seq_cst, thread_scope scope = thread_scope_threadgroup)
  __attribute__((annotate("msl.sync:threadgroup_barrier")));
void simdgroup_barrier(mem_flags flags, memory_order order = memory_order_seq_cst, thread_scope scope = thread_scope_simdgroup)
  __attribute__((annotate("msl.sync:simdgroup_barrier")));
void atomic_thread_fence(mem_flags flags, memory_order order, thread_scope scope = thread_scope_device)
  __attribute__((annotate("msl.atomic:atomic_thread_fence")));
}
