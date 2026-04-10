# FactoriaX Profiling: Session Log

## Objective

Determine whether the FactoriaX JAX simulation is compute-bound or
memory-bandwidth-bound on the RTX 3050 6GB. This determines the optimization
strategy: kernel optimization (compute) vs data movement reduction (memory).

## Metric

Primary: **steps/second** at batch=1024 on 32x32 map, single player.
Secondary: per-operation wall-clock breakdown.

## Hardware

- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU (compute 8.6)
- Peak FP32: ~4.5 TFLOPS
- Memory bandwidth: ~192 GB/s
- Ridge point: ~23.4 FLOPs/byte

## Baseline (2026-04-10)

### Throughput Scaling

| Batch | Steps/s | Per-env (us) | Scaling vs batch=1 |
|------:|--------:|-------------:|--------------------:|
| 1 | 44 | 22,894 | 1.0x |
| 4 | 206 | 19,401 | 4.7x |
| 16 | 890 | 17,980 | 20.2x |
| 64 | 2,944 | 21,736 | 66.9x |
| 256 | 9,157 | 27,955 | 208x |
| 1024 | 16,608 | 61,658 | 377x |
| 4096 | 17,893 | 228,919 | 407x |
| 8192 | 17,990 | 455,361 | 409x |

Throughput plateaus at ~18K steps/s around batch=4096. From 1→64 the scaling
is near-linear (amortizing kernel launch overhead). From 256→1024 it's
sublinear (GPU filling). From 4096→8192 it's flat (saturated).

### Sub-operation Breakdown (batch=1024)

| Operation | Time (us) | % of step_env |
|-----------|----------:|--------:|
| update_biters | 16,806 | 29.3% |
| run_arms | 13,299 | 23.2% |
| run_assemblers | 8,858 | 15.4% |
| run_conveyor_belts | 7,744 | 13.5% |
| handle_player_action | 7,188 | 12.5% |
| push_miner_output | 6,599 | 11.5% |
| run_miners | 4,739 | 8.3% |
| refuel_machines | 2,582 | 4.5% |
| update_scent_field | 1,857 | 3.2% |
| update_crafting | 1,645 | 2.9% |
| core_game_conditions | 1,122 | 2.0% |
| global_array | 909 | 1.6% |
| achievement_reward | 39 | 0.1% |

Sum of parts (~127%) exceeds step_env (100%) because individually JIT-compiled
functions miss XLA fusion opportunities that the monolithic step_env gets.

### Roofline Classification

XLA `cost_analysis()` returned 0 FLOPs and 0 bytes for every operation.
This is a known limitation: operations composed of many `lax.cond`, `lax.scan`,
and scatter-gather primitives don't report aggregate costs through the XLA
cost model.

## Diagnosis: Launch-overhead + Memory-bandwidth Bound

**Evidence:**

1. **Scaling shape**: Near-linear 1→64, sublinear 256→1024, flat 4096→8192.
   This is textbook memory/launch saturation. Compute-bound workloads scale
   with batch until ALU utilization maxes out, then drop sharply. This gradual
   flatten is bandwidth saturation.

2. **Per-env time at plateau**: ~56 us at batch=8192. Total state is ~65 KB
   per env. At 192 GB/s, reading 65 KB takes ~0.34 us. We're 165x slower than
   the theoretical bandwidth floor, suggesting massive kernel launch overhead.

3. **XLA cost model returns zeros**: The operations are composed of many tiny
   fused kernels rather than a few large ones. Each kernel launch has ~5-10 us
   overhead on this GPU, and there are likely dozens of kernels per step.

4. **Top operations are branch-heavy**: `update_biters` (lax.cond gated,
   3 sub-phases), `run_arms` (two-phase pick/deposit with per-tile branching),
   `run_assemblers` (recipe gating + research checks). These produce many small
   kernels that serialize on the GPU.

**Conclusion**: Optimizing individual hot functions will show near-zero
improvement unless we reduce the number of kernel launches. The strategy should
focus on:
- Fusing sequential operations to reduce kernel count
- Reducing `lax.cond` usage (replace with masked arithmetic where possible)
- Reducing scatter-gather operations per step
- Increasing arithmetic density per kernel

## Literature Research (2026-04-10)

### Codebase Audit: Fusion Barriers

Counted all fusion barriers in `factoriax/`:
- **26 `lax.cond` calls** — 11 in the action dispatch chain alone (game_logic.py:758-828)
- **5 `lax.scan` calls** — crafting, biters, repair
- **8 `fori_loop` calls** — all in jax_renderer.py (rendering path, not step)
- **74 `.at[]` scatter operations** — throughout machines, game_logic, biters

Each `lax.cond` is both a **fusion barrier** (prevents XLA from merging kernels
across the boundary) and a **GPU-CPU sync point** (predicate must be read back
to CPU to decide which branch to execute). Under `vmap`, `lax.cond` is
automatically converted to `lax.select` internally, meaning you already pay
for computing both branches without getting the fusion benefit.

### Key Finding: lax.cond Is the Primary Bottleneck

JAX core developers confirmed in issue #7934 that `lax.cond` requires a
GPU-to-CPU roundtrip to evaluate the predicate. Benchmarks show ~20% speedup
switching a single `lax.cond` to `lax.select` on large arrays. The effect
compounds: the 11-cond action dispatch chain produces 20-30+ kernel launches
per step.

`lax.scan` has the same problem on GPU: each iteration corresponds to a
separate kernel launch (confirmed in JAX discussions #16106, #10233). XLA GPU
executes dynamic control flow on the CPU.

### Craftax: The Gold Standard (ICML 2024)

Craftax (arXiv:2402.16801) is the closest existing system to FactoriaX: a
complex tile-based game with inventory, crafting, entities, all in JAX. They
achieve 250x speedup over CPU Crafter using these patterns:

1. **`lax.select` everywhere, never `lax.cond`.** Both branches are
   precomputed, then selected. This is element-wise and fully fusible.
2. **Arithmetic masking.** Boolean conditions multiplied into values:
   `amount = base_amount * is_valid * has_space`. Single fused kernel.
3. **`tree_map` with `lax.select` for bulk state updates.** Instead of
   conditionally calling a state-modifying function, compute the new state
   unconditionally and select old vs new per-field.
4. **Fixed-size arrays with masking** for dynamic entity counts.

### XLA Fusion Paper (arXiv:2301.13062)

Snider & Liang studied operator fusion in XLA on a Cart-pole simulation with
a similar bottleneck profile (tiny elementwise kernels, launch-overhead bound).

Key results:
- Removing a single `jnp.concatenate` fusion barrier: **3.41x speedup**
- Unrolling a scan loop by 10x: **3.5x speedup**
- Combined: **10.56x total speedup**

The mechanism: with full fusion, intermediate values stay in GPU registers
instead of being written to global memory between kernels.

### PureJaxRL / JaxMARL

PureJaxRL compiles the entire training loop (env step + policy + loss +
gradient) into a single XLA program via `lax.scan` over timesteps. Performance:
CartPole 1000x faster than CPU gym, MinAtar 250x faster. The architecture
validates that env step quality is critical — every unnecessary kernel launch
inside `step` is multiplied by (num_envs x num_timesteps x num_epochs).

### Sources

- JAX Issue #7934: lax.cond vs lax.select efficiency
- JAX Discussion #12281: lax.cond vs jnp.where
- JAX Discussion #16106: lax.scan performance on GPU
- JAX Discussion #10233: lax.scan slow on GPU
- Craftax (arXiv:2402.16801, ICML 2024 Spotlight)
- Craftax GitHub: MichaelTMatthews/Craftax
- Operator Fusion in XLA (arXiv:2301.13062)
- XLA GPU Architecture: openxla.org/xla/gpu_architecture
- PureJaxRL: luchris429/purejaxrl
- JaxMARL (arXiv:2311.10090)
- Pgx (arXiv:2303.17503)
- EconoJax (arXiv:2410.22165)
- JAX GPU Performance Tips: docs.jax.dev/en/latest/gpu_performance_tips.html
