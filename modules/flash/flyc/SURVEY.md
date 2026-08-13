# Survey: what does a FlyDSL → hsaco build actually depend on?

Goal: mimic the Triton hsaco build — a minimal venv (or a set, for alt-venvs) created during
the build — while staying compatible with venv-based TheRock.

All results below are measured in this container against flydsl 0.3.1 and
`rocm-sdk 7.14` in `/home/xinyazha/.venvs/nogpu`, cross-compiling the real
`flash_attn_func_gfx1201_aiw` kernel for gfx1201 on a non-gfx1201 host with no GPU access.
Probe script: `/tmp/probe_fmha.py`.

## Answers, up front

| Question | Answer |
|---|---|
| Can kernels be built with **no ROCm at all**? | **Only to ISA (`.s`), not to hsaco.** flydsl bundles the full LLVM AMDGPU backend, but not a linker. |
| If ROCm is needed, what is the **minimal component**? | **One file: `<ROCM_PATH>/llvm/bin/ld.lld`.** Nothing else — no HIP, no headers, no device bitcode, no `rocm_sdk` Python package. |
| Can flydsl live in venv1 and LLVM in venv2? | **Yes.** The coupling is a filesystem path (`$ROCM_PATH`), not an import. Verified across two real venvs. |
| Self-contained linker footprint | **143 MB** (`libLLVM.so` 140 MB + `lld` 8.2 MB + 2 wrappers + 1 MB sysdeps), vs 1.4 GB for the whole `rocm-sdk-core` wheel. |
| Is the result hermetic? | **Yes.** Byte-identical hsaco across all three configurations (sha256 `d537bb2ee33b0c121c865a9e…`). |

## 1. Building without ROCm

**flydsl reaches ISA entirely on its own.** With `ROCM_PATH` unset and the ROCm wheels
untouched, `gpu-module-to-binary{format=isa}` produced complete gfx1201 assembly for the real
kernel — 58,582 characters of it:

```
$ env -u ROCM_PATH python probe_fmha.py isa fmha
ROCM_PATH=<unset>
RESULT fmha/isa: OK 58582 bytes magic=b'\t.am'
```

So the guess in the question is half right. flydsl does *not* ship "only MLIR": its
`libFlyPythonCAPI.so.24.0git` statically contains LLVM 24.0git including the **AMDGPU target**,
which is what turns LLVM IR into gfx1201 ISA. Confirmed by `ldd`: that library has **no**
HIP/ROCm/HSA dependency at all. Only `libfly_jit_runtime.so` needs `libamdhip64.so.7`, and that
is the *execution engine*, never touched under `COMPILE_ONLY=1` — it is in fact already unresolved
in this venv (`libamdhip64.so.7 => not found`) while compilation succeeds.

What flydsl does **not** ship is a linker. MLIR's ROCDL serializer assembles the ISA in-process
with its own MC layer, then shells out to `ld.lld` to produce the HSA code object. That single
`execv` is the entire ROCm dependency:

```
$ env -u ROCM_PATH python probe_fmha.py binary trivial
RESULT trivial/binary: FAIL error: lld invocation failed
```

There is no MLIR option to link in-process — flydsl's MLIR build exports no `lld::lldMain`
symbols (checked with `nm -D` across all of `_mlir_libs/*.so`).

## 2. Minimal required components

**A four-entry directory is sufficient.** Building a tree containing nothing but a symlink to
`ld.lld`:

```
/tmp/minirocm_a/llvm/bin/ld.lld -> …/_rocm_sdk_core/lib/llvm/bin/ld.lld
```

…the real FMHA kernel links:

```
$ ROCM_PATH=/tmp/minirocm_a python probe_fmha.py binary fmha
RESULT fmha/binary: OK 13240 bytes magic=b'\x7fELF'
```

Not required, despite appearing in most ROCm-toolchain documentation:

- **device bitcode** (`<ROCM_PATH>/amdgcn/bitcode/*.bc`, 3.3 MB) — this kernel resolves every
  math op to native instructions and references no `ocml_*`/`ockl_*` symbol. *Caveat: unverified
  for kernels that do.* At 3.3 MB it is cheap insurance and I'd ship it anyway.
- **HIP runtime** (`libamdhip64`), **HSA runtime**, **comgr** — runtime only.
- **clang / the assembler** — MLIR assembles in-process. Notably the ROCm `clang` (LLVM 23)
  *cannot* assemble flydsl's ISA (LLVM 24.0git): it rejects `.amdhsa_exception_fp_denorm_src`
  and friends as unknown directives. The two LLVMs are only compatible at the *object* level,
  which is all the lld handoff needs.
- **headers, CMake config, `rocm_sdk` Python package** — `grep -rn rocm_sdk flydsl/` returns
  nothing; flydsl has zero awareness of the ROCm wheels.

### The self-contained 143 MB subset

`ld.lld` is a 26 KB `execv` wrapper that resolves `/proc/self/exe` and re-execs the real 8.2 MB
`lld` beside it, which in turn pulls `libLLVM.so.23.0git` via an `$ORIGIN`-relative RPATH.
Preserving that relative structure gives a standalone tree:

```
143M total
 140445585  llvm/lib/libLLVM.so.23.0git
   8181784  llvm/bin/lld
    899921  rocm_sysdeps/lib/librocm_sysdeps_zstd.so.1
    142857  rocm_sysdeps/lib/librocm_sysdeps_z.so.1
     26360  llvm/bin/ld.lld
```

```
$ ROCM_PATH=/tmp/minirocm_sc python probe_fmha.py binary fmha
RESULT fmha/binary: OK 13240 bytes magic=b'\x7fELF'
```

143 MB against 1.4 GB for `_rocm_sdk_core` and 2.6 GB for `_rocm_sdk_devel` (which is an
unexpanded tar requiring `rocm-sdk init`, and whose `rocm-sdk path --root` therefore points
somewhere useless for us).

**Any sufficiently recent `ld.lld` should do.** MLIR's invocation is generic ELF linking
(`-shared`, no AMDGPU-specific flags), so a stock upstream lld is expected to work and AMD's
fork is not special. *Unverified* — this container has no non-ROCm LLVM to test against.

## 3. venv1 (flydsl) + venv2 (LLVM)

Works, because there is no import relationship to break — only `$ROCM_PATH`.

Built `/tmp/venv1` containing exactly `flydsl`, `flydsl.libs`, `numpy` and **no** `rocm_sdk`
(`find_spec('rocm_sdk')` → `None`), then:

```
# venv1 + standalone linker tree
$ ROCM_PATH=/tmp/minirocm_sc /tmp/venv1/bin/python probe_fmha.py binary fmha
RESULT fmha/binary: OK 13240 bytes

# venv1 + LLVM living inside a second venv's site-packages
$ ROCM_PATH=/tmp/venv2/lib/python3.13/site-packages/_rocm_sdk_core/lib \
    /tmp/venv1/bin/python probe_fmha.py binary fmha
RESULT fmha/binary: OK 13240 bytes
```

All three configurations — full venv, minimal tree, cross-venv — emit a **byte-identical**
code object (sha256 `d537bb2ee33b0c121c865a9e…`). The linker location does not leak into the
artifact.

### Consequence for alt-venvs

AOTriton's `AOTRITON_ALT_TRITON_WHEEL_CONFIG_FILE` creates one venv per arch to pin different
Triton wheels. The FlyDSL equivalent is cheaper: since the linker is reached by *path* rather
than by import, **it does not need duplicating per alt-venv**. Each alt-venv carries only
flydsl + numpy (~300 MB) and every one of them points at a single shared `$ROCM_PATH`.

## 4. Recommended shape

**Build venv contents:** `flydsl` (280 MB, bundles its own LLVM 24.0git) + `numpy`
(already in `requirements.txt`). No torch — required, see `CMakeLists.txt:142`. No `rocm_sdk`
package, no HIP.

**Linker resolution**, in the driver, in order:

1. `$ROCM_PATH` if already set and `$ROCM_PATH/llvm/bin/ld.lld` exists — lets a TheRock or
   `/opt/rocm` user override, and validates rather than trusts.
2. `importlib.util.find_spec("_rocm_sdk_core")` → `<parent>/lib`. This is the TheRock-compatible
   path and works with a plain `pip install rocm-sdk-core` into either the build venv or any
   other venv. Verified:
   `/…/site-packages/_rocm_sdk_core/lib` → `llvm/bin/ld.lld` exists.
   Do **not** use `rocm-sdk path --root`; it returns the unexpanded `_rocm_sdk_devel` tree.
3. `/opt/rocm`.
4. Otherwise raise, listing candidates tried — the native failure is
   `error: lld invocation failed` with no lld output, which is unactionable.

**A new CMake knob is implied:** `AOTRITON_FLYDSL_ROCM_PATH` (default: auto-detect as above),
so a build can point at a shared or pre-staged linker tree instead of installing a 1.4 GB wheel
into the build venv.

## Open items

- Device bitcode requirement is unverified for kernels that reference `ocml`/`ockl`. Ship the
  3.3 MB `amdgcn/bitcode` directory regardless.
- Generic (non-AMD) `ld.lld` compatibility is argued, not measured.
- Whether the 143 MB subset should be produced by the build (extracted from the wheel) or just
  documented, with `rocm-sdk-core` installed whole, is a packaging decision not made here.
