# MFEM on XLA-GPU: paper58t reproduction bundle

This bundle reproduces the MFEM performance table: native CUDA vs. XLA-GPU (raised through Enzyme-JAX
and run by Reactant's linkable runtime) over every MFEM benchmark suite. The run is `paper58t`,
2026-11-04, with 107 cases. **The XLA/native geomean is 0.750; XLA is at or below native on 71
cases.** The per-case table is in `results/paper58t_cases.md`.

The scripts are copied verbatim from the working directory they ran in, so they still contain its
absolute paths (`/home/wmoses/ejwork`, `/home/wmoses/git/...`, `/mnt3/wmoses/git/mfem`). Each path
is mapped to its source below.

## Machine

- 2x NVIDIA GeForce RTX 5090 (sm_120), driver 575.57.08, CUDA 12.9. The run used GPU 0 only;
  another user's jobs held GPU 1.
- The machine is shared. `gpu_wait_idle.sh` waits for an idle GPU before every run, and
  `mfem3_bench_pair.sh` waits for a 1-minute load below 25. The 1-minute load during the run was
  3.2-10.9 (`results/paper58t/*/summary.txt`).

## Sources

Every component is pinned to a pushed commit. Anything that existed only as a local branch or as
uncommitted edits was pushed as a `repro/mfem-xla-paper58*` snapshot branch.

| Component | Repository / branch | Commit | Built into |
|---|---|---|---|
| MFEM | [wsmoses/mfem `repro/mfem-xla-paper58`](https://github.com/wsmoses/mfem/tree/repro/mfem-xla-paper58) (= `xla-paper-2026-10` 93b234fdc + 3 working-tree edits) | 020f93a6d | both builds (`mfem_builds/native`, `mfem_builds/xla2sync`) |
| Clang / LLVM | llvm/llvm-project | b08e005995ed | `clang++` for both builds |
| ClangEnzyme (native build only) | EnzymeAD/Enzyme `mss` | 3189c448b | `-DEnzyme_DIR` of the native build. None of the suites here use AD. |
| Raising plugin | [EnzymeAD/Reactant `repro/mfem-xla-paper58-plugin`](https://github.com/EnzymeAD/Reactant/tree/repro/mfem-xla-paper58-plugin) (= EnzymeAD/Reactant#114 + EnzymeAD/Reactant#115) | 7e2d0d27 | `libClangReactant_b08e.so` |
| Raising library (compile time) | [EnzymeAD/Enzyme-JAX `repro/mfem-xla-paper58-libraise`](https://github.com/EnzymeAD/Enzyme-JAX/tree/repro/mfem-xla-paper58-libraise) | 4a2868384 | `libRaise_best4sync.so` |
| Runtime: Enzyme-JAX side | [EnzymeAD/Enzyme-JAX `repro/mfem-xla-paper58-runtime`](https://github.com/EnzymeAD/Enzyme-JAX/tree/repro/mfem-xla-paper58-runtime) | 8a4a62d90 | `--override_repository=enzyme_ad=` of the runtime build |
| Runtime: Reactant.jl side | [EnzymeAD/Reactant.jl `repro/mfem-xla-paper58`](https://github.com/EnzymeAD/Reactant.jl/tree/repro/mfem-xla-paper58) | c1f975451 | `libReactantExtra.so` (`rt/paper58`) |
| JAX / XLA | JAX 3e6335a8b546 (pinned by the runtime branch) with the XLA patches listed below | — | the runtime |
| Google Benchmark | google/benchmark v1.9.5 | — | both builds |

### What the snapshot branches contain beyond main

**Compile-time raising library** (Enzyme-JAX
[compare](https://github.com/EnzymeAD/Enzyme-JAX/compare/main...repro/mfem-xla-paper58-libraise)).
The base is main 73603ba52 (2026-10-11), plus these Enzyme-JAX PRs:

- #3422, #3425, #3427, #3428, #3431 and #3438 (AffineCFG parallelization of the MFEM kernels).
- #3444 (scatter-scratch forwarding; closed, superseded).
- A local fix to the #3422 combine (f4a83ad8f).
- #3641 (`cudaDeviceSynchronize` → `reactantXLASync`). Without it the XLA benchmarks would time
  enqueue rates instead of kernel time. #3641 was merged, then reverted in #3667; the draft #3668
  carries it.

**Runtime, Enzyme-JAX side** (compare: [main...repro/mfem-xla-paper58-runtime](https://github.com/EnzymeAD/Enzyme-JAX/compare/main...repro/mfem-xla-paper58-runtime)).
The base is main 1ba9ab86a (2026-11-01). The exec pipeline gets its pattern list from this tree
(`enzymexlaGetTransformPassesList`). The numbers below are Enzyme-JAX PRs.

- Gather/scatter and index rewrites: #3490, #3617, #3626, #3758, #3759 (merged), #3760, #3761,
  #3762, #3765, #3767, #3771, #3772, #3773, #3774, #3778, #3780, #3782 (plus a local amend:
  columns through views), #3784, #3785.
- Layout and reductions: #3624, #3627, #3680, #3699 (element-major), #3788, #3790, #3792, and the
  concat_reduce shared-gather gate (#3801).
- Barrier patterns: #3634 reread_barrier, and #3538 concat_operand_barrier with an amend that leaves
  gathers unbarriered.
- Element chunking: #3809, including 68a4d7aae (uneven unrolled chunks).
- XLA patches (`patches/xla_*.patch`, applied in the `third_party/xla/workspace.bzl` list):
  small-transpose coalescing (#3727 v3), multi-output loop fusions not unrolled (#3786), and 64-bit
  outputs unrolled to 128-bit accesses (#3793).
  - Two more patches are gated on env variables that are unset in these runs, so they are inert:
    `XLA_LOOP_TPB` and `XLA_UNROLL_NONDIV`.
  - One local experiment is inert for the same reason: `ENZYME_RMD_MAX_ROW`.
- The last commit (8a4a62d90) rewrites `//:patches` labels to `@enzyme_ad//:patches`.
  `--override_repository` skips Reactant.jl's patch_cmds, so those labels are rewritten by hand.

**Runtime, Reactant.jl side** (compare: [main...repro/mfem-xla-paper58](https://github.com/EnzymeAD/Reactant.jl/compare/main...repro/mfem-xla-paper58)).
The base is main 561a5afa7 (2026-11-01). The branch adds:

- The linkable-runtime stack: EnzymeAD/Reactant.jl#3425, EnzymeAD/Reactant.jl#3426, EnzymeAD/Reactant.jl#3427, EnzymeAD/Reactant.jl#3428, EnzymeAD/Reactant.jl#3429, EnzymeAD/Reactant.jl#3432, EnzymeAD/Reactant.jl#3435 and EnzymeAD/Reactant.jl#3473.
- Three local commits: the retry round and the specialization chunks.
- The last commit, c1f975451, is the uncommitted `deps/ReactantExtra/API.cpp` this runtime was
  built with. It carries:
  - EnzymeAD/Reactant.jl#3474 (layout round), EnzymeAD/Reactant.jl#3495 (`reactantXLASync`), EnzymeAD/Reactant.jl#3639, EnzymeAD/Reactant.jl#3642, EnzymeAD/Reactant.jl#3646, EnzymeAD/Reactant.jl#3647 (CUDA graph for
    executables of three or more kernels) and EnzymeAD/Reactant.jl#3648.
  - Three changes since split out as their own PRs: EnzymeAD/Reactant.jl#3696 (a
    specialization's module loses its stale barriers before its passes run again),
    EnzymeAD/Reactant.jl#3697 (the exec-time form tuner, `REACTANT_EXEC_TUNE=emajor,nobarrier,chunk`:
    each module compiles its candidate forms, times them in blocks of 6 calls, and keeps the
    fastest) and EnzymeAD/Reactant.jl#3695 (cache-miss compiles on their own 64 MB-stack thread,
    which avoids glibc heap fragmentation slowing later calls).
  - Opt-in experiments and debugging switches that were unset in the paper runs
    (`REACTANT_EXEC_RELAYOUT`, `CONCAT_BARRIER`, `PREUNROLL`, `EMAJOR`, the `*_DROP` / `NO*` /
    `KEEP_BARRIERS` / `REPLACE` switches). These live only in this snapshot.

The snapshot branches are open as draft PRs so the diffs can be read: Enzyme-JAX #3854 (runtime)
and #3855 (libRaise), Reactant.jl #3694, wsmoses/mfem #1, and this bundle as
EnzymeAD/Enzyme-GPU-Tests #21.

## Build

1. **MFEM build directories.** `scripts/mfem3_configure.sh native` and `xla` give CMake's options.
   The configured caches are `config/CMakeCache.{native,xla2sync}.txt`.
   - The XLA build passes `-DEnzyme_DIR` pointing at `config/shim-xla2sync.EnzymeConfig.cmake`.
     That file adds the plugin flags (`-reactant-backend xla-gpu`, the libRaise path, libdevice)
     and links `libReactantExtra_stable.so`.
   - The binaries use `RUNPATH`, so the runtime picked at run time is whichever
     `LD_LIBRARY_PATH` names.
2. **Compile.**
   - `scripts/build_xla2sync.sh` builds the XLA side (`ninja mfem benchmarks`) through
     `scripts/mfem3_launch.sh`, which compiles the TUs in `config/mfem3_xla_fallback.txt` without
     the plugin. Those are three host TUs (mem_manager, densemat) and vecconvection_pa_grad.
   - The raising environment is `REACTANT_LOWER_INVOKE=0 REACTANT_SPECIALIZE_INDEX_STRIDES=1
     REACTANT_SPECIALIZE_INT_SCALARS=1 REACTANT_UNROLL_MAX_TRIP_COUNT=0
     REACTANT_UNROLL_MAX_NESTED_OPS=4096`.
   - The native build is `scripts/mfem3_build2.sh native`.
3. **assembly_levels with x initialized.** `scripts/build_x0.sh <build>` builds
   `bench_assembly_levels_x0` from `patches/bench_assembly_levels_x0.patch`. The stock benchmark
   never initializes `x`, which FormLinearSystem reads Dirichlet values from, so CG sometimes hits
   NaN.
4. **Runtime.** `scripts/build_paper58.sh` runs a bazel build of `//:libReactantExtra.so` in
   Reactant.jl `deps/ReactantExtra`, with:
   - `--config=cuda12` and `HERMETIC_CUDA_COMPUTE_CAPABILITIES=sm_60,sm_70,sm_80,sm_90,sm_100,compute_120`;
   - `--override_repository=enzyme_ad=<checkout of repro/mfem-xla-paper58-runtime>`.

## Run

`scripts/bench_paper58t.sh`. For each line of `config/paper_cases_gpu{0,1}.txt` (a binary plus a
case filter), `scripts/mfem3_bench_pair.sh` runs native and then XLA back to back on the same GPU.
Pairing the two puts both columns of host-bound cases under the same machine state.

- Flags: `--benchmark_repetitions=3 --benchmark_min_warmup_time=1`, with
  `--benchmark_context=device=cuda`.
- Environment: `REACTANT_EXEC_DOT_FORMATION=0 REACTANT_EXEC_UNROLL=8`, plus `config/xla2p58c.env`
  for the XLA column:

  ```
  LD_LIBRARY_PATH=<rt/paper58>
  REACTANT_EXEC_TUNE=emajor,nobarrier,chunk
  REACTANT_EXEC_PERMREDUCE=1 REACTANT_EXEC_SHORTROWS=1 REACTANT_EXEC_SHORTPAD=1
  ```
- `scripts/paper_table.py paper58t xla2p58c` writes `results/paper58t.{json,tex}`.
  `scripts/cmp_median.py` computes the geomean of per-case median ratios.

## Results (`results/`)

| suite | cases | XLA/native |
|---|---:|---:|
| assembly_levels | 28 | 0.74 |
| ceed | 12 | 0.94 |
| dg_amr | 4 | 0.91 |
| elasticity | 4 | 1.01 |
| nlvc | 12 | 0.81 |
| tmop | 32 | 0.57 |
| vector | 12 | 1.01 |
| virtuals | 3 | 0.96 |
| **all** | **107** | **0.750** |

The worst XLA/native ratios are:

| Case | XLA/native |
|---|---:|
| nlvc AssembleGradDiagonal2d/4 | 1.38 |
| nlvc AddMultPA2d/4 | 1.36 |
| tmop AssembleGradDiagonalPA/0/4 | 1.28 |
| elasticity Gradient3D/3/60/0 | 1.24 |
| assembly_levels BK5PARTIAL/4 | 1.23 |
| ceed BP5/2 | 1.18 |

The raw Google Benchmark JSON and stdout for both columns are in `results/paper58t/{native,xla2p58c}`.

## Known caveats

- **Short runs.** The sweep sets no `--benchmark_min_time`, so a case whose form tuner compiles
  several forms gets fewer timed iterations. tmop AddMultGradPA/0/4 reads 1.15 here but 0.998 with
  `--benchmark_min_time=8s`.
- **Unexplained late-suite slowdowns.** In two of five earlier assembly_levels runs of this runtime,
  XLA cases late in the suite ran 1.3-2.9× slow inside one process. Every candidate form runs at
  103-114 µs in isolation. The run here was clean: assembly_levels came in at 1.002 of a clean
  baseline run.
