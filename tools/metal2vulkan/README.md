# msl2spirv

The Metal shader compiler Basalt builds with: **MSL source in, Vulkan SPIR-V and
a reflection JSON out**. 

This folder contains

| Path | What it is |
|---|---|
| `bin/msl2spirv.exe` | The compiler. Not committed: the first CMake configure fetches it from the release named `msl2spirv-<version>` and checks it against the SHA-256 pinned in [`cmake/msl2spirv.cmake`](../../cmake/msl2spirv.cmake). [`package_msl2spirv.ps1`](../package_msl2spirv.ps1) builds that asset and prints the hash to pin. |
| `share/metal2vulkan/stdlib/` | The owned Metal standard library it parses against. Required at run time. Independently authored: no Apple code. |
| `src/m2v_host.h`, `src/m2v_host.cpp` | The host reflection library, compiled into Basalt. It reads the reflection JSON and builds the Vulkan descriptor layouts from it. |

## Usage

```
msl2spirv input.metal [more.metal ...] --entry NAME -o output.spv
```

Several inputs link into one program, included in order; every definition must be
unique. Math is always strict: no contraction, no reassociation. Apple `.air`
and `.metallib` packaging, `-ffast-math` and platform-version options are
rejected rather than ignored.

## Options

**Selecting and emitting**

| Option | Effect |
|---|---|
| `--entry NAME` | The entry point to compile. Required. |
| `-o PATH` | The SPIR-V module to write. Required. |
| `--emit-reflection PATH` | The JSON binding and dispatch contract. Basalt reads this for every shader. |
| `--emit-ir PATH` | The typed MSL IR, for inspection. |
| `--emit-spirv-ir PATH` | The module as MLIR SPIR-V dialect, before serialization. |
| `--depfile PATH` | A Makefile-style dependency file listing every source and header read. |
| `-I DIR` | Another include directory for the source. |
| `--stdlib DIR` | The owned standard library. Defaults to `M2V_STDLIB_DIR`, then the installed copy, then the source tree. |
| `-std=metalX.Y` | `metal3.0`, `metal3.1`, `metal3.2`, `metal4.0` or `metal4.1` (default). |
| `--target-env vulkan1.3` | The only supported target. |
| `--module-cache DIR` | Reuse the prelude as a precompiled header cached here, keyed by the standard library digest and the standard. |
| `--list-features` | Print the embedded feature registry as JSON and exit. This is the authoritative list of what the compiler accepts and what it rejects. |
| `--version`, `--help` | The version, or the usage text this table came from. |

**Compute**

| Option | Effect |
|---|---|
| `--local-size N` | Workgroup size in X, 1 to 128. Default 64. |
| `--local-size-y N` | Workgroup size in Y, 1 to 128. Default 1. |
| `--local-size-z N` | Workgroup size in Z, 1 to 64. Default 1. Total threads stay within 128. |
| `--exact-grid` | Dispatch-plan mode: the workgroup size becomes a specialization constant and the grid builtins come from a push-constant block, so the host can launch exact grids the way Metal's `dispatchThreads` does. |
| `--threadgroup-capacity INDEX:BYTES` | The allocation behind a `[[threadgroup(INDEX)]]` pointer parameter. Default 4096; total threadgroup memory stays within 16 KiB. |
| `--stage-in-layout FILE` | The stage-input layout for a kernel `[[stage_in]]` struct, fetched per thread at `thread_position_in_grid.x`. |

**Bindings and resources**

| Option | Effect |
|---|---|
| `--argument-capacity NAME=N` | The element count, 1 to 256, of a pointer-to-argument-buffer parameter, whose resource members become descriptor arrays. |
| `--nullable NAME,...` | Buffer pointers and single textures the host may leave unbound. It binds valid placeholders and reports residency: reads return zero and ordinary writes are dropped. A Metal2Vulkan extension. |
| `--device-addresses` | Pointer members, pointer-to-integer round trips and dereferenced addresses become 64-bit buffer device addresses; the host fills the address table the reflection describes and enables `bufferDeviceAddress`. |
| `--enable-logging` | `os_log` calls write the bounded log buffer at buffer index 31. `-fmetal-enable-logging` is an alias. |

**Graphics**

| Option | Effect |
|---|---|
| `--check-fragment-entry NAME` | Check the selected vertex entry against a fragment entry in the same source at build time. Basalt passes this for every vertex and fragment pair, so a mismatched interface fails the build rather than a frame. |
| `--mesh-entry NAME` | Compile the selected fragment against a mesh entry in the same source. |
| `--emit-tessellation-vertex PATH` | The generated vertex stage of a tessellation pipeline. |
| `--emit-tessellation-control PATH` | The generated tessellation control stage. |
| `--tessellation-partition MODE` | `integer` (default), `fractional_odd` or `fractional_even`. |
| `--tessellation-winding W` | `clockwise` (default) or `counter_clockwise`. |

**Numerics and debugging**

| Option | Effect |
|---|---|
| `--float-controls none\|preserve` | Preserve NaN, infinity and signed zero. Needs the Vulkan float-controls property. |
| `-fmetal-rtz-fp-conversion` | Round float-to-float conversions toward zero, as Metal 4.1 does. The default rounds to nearest even. |
| `--debug-info` | Emit source line mapping as `OpString` and `OpLine`. `-g` and `-gline-tables-only` are aliases. |


## How Basalt calls it

[`CMakeLists.txt`](../../CMakeLists.txt) wraps one invocation per entry in a
`basalt_shader` function, which writes the module, the reflection JSON and a
depfile into `build/shaders/` and passes `--check-fragment-entry` for each
vertex and fragment pair. A shader that does not compile, or a vertex whose
outputs do not match its fragment's inputs, fails the build.


