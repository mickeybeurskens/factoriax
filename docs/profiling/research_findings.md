# JAX Performance Research Findings

Research on optimizing JAX-based game simulation throughput, focused on the
FactoriaX step function. Conducted 2026-04-10.

## The Core Problem

FactoriaX is **launch-overhead bound** on GPU. The step function chains 14
sub-operations, each containing `lax.cond` branches and `lax.scan` loops that
XLA compiles into dozens of tiny GPU kernels. Each kernel launch costs ~5-10 us,
and the overhead dominates actual computation. Throughput plateaus at ~18K
steps/s regardless of batch size beyond 4096.

Measured: 26 `lax.cond` calls, 5 `lax.scan` calls, 74 scatter operations in
the step path.

## Finding 1: lax.cond Creates GPU-CPU Sync Points

**Source:** JAX Issue #7934, confirmed by JAX core developer.

On GPU, `lax.cond(pred, true_fn, false_fn)` reads the predicate back to CPU
to decide which branch to execute. This is a `cuStreamSynchronize` + `MemcpyD2H`
per call. It also prevents XLA from fusing operations across the boundary.

Under `vmap` (batched environments), `lax.cond` is silently converted to
`lax.select` internally, computing both branches. You pay for both branches
without getting the fusion benefit.

**Fix:** Replace with `jnp.where(pred, true_val, false_val)` which is
element-wise, fully fusible, and never leaves the GPU.

**Benchmark (from JAX issues):** Single cond→select swap: ~20% faster.
Conditional scatter: `jnp.where` at ~45.5us vs `fori_loop` with `lax.cond`
at ~541ms (1000x difference from kernel launch overhead).

**Links:**
- https://github.com/jax-ml/jax/issues/7934
- https://github.com/jax-ml/jax/discussions/12281
- https://github.com/jax-ml/jax/issues/19972

## Finding 2: Craftax Pattern (ICML 2024)

**Source:** arXiv:2402.16801, MichaelTMatthews/Craftax on GitHub.

Craftax is a Crafter-like game in JAX, the closest prior art to FactoriaX.
Tile map, inventory, crafting, entities. Achieves 250x over CPU Crafter.

Their optimization patterns:

1. **`lax.select` everywhere, never `lax.cond`.** Both branches precomputed,
   result selected. Fully fusible.

2. **Arithmetic masking.** Boolean conditions multiplied into values:
   ```python
   coal_amount = random_amount * (ore_id == 0) * is_looting_ore
   ```
   Compiles to a single fused elementwise kernel.

3. **`tree_map` + `lax.select` for bulk state updates.** Instead of:
   ```python
   state = lax.cond(pred, update_fn, identity, state)
   ```
   They do:
   ```python
   new_state = update_fn(state)
   state = jax.tree.map(
       lambda n, o: lax.select(pred, n, o), new_state, state
   )
   ```
   This computes the update unconditionally, then selects per-field. The
   `tree_map` + `select` is element-wise and fusible.

4. **Fixed-size arrays with masking** for variable entity counts.

**Key insight:** Read their game_logic.py for concrete examples of every
pattern. It maps almost 1:1 to FactoriaX's structure.

**Link:** https://github.com/MichaelTMatthews/Craftax

## Finding 3: XLA Fusion Mechanics

**Source:** arXiv:2301.13062 (Snider & Liang, "Operator Fusion in XLA")

XLA has four fusion strategies: instruction fusion, fusion merger, multi-output
fusion, and horizontal fusion. A fusion always compiles to exactly one GPU
kernel, and no intermediate data materializes in HBM within a fusion.

**What blocks fusion (hard barriers):**
- `lax.cond` / `lax.switch` — GPU-CPU sync, minimum 2-3 kernel boundary
- `lax.scan` / `fori_loop` iteration boundaries — each iteration is a kernel
- Custom library calls (cuBLAS, cuDNN) — opaque to XLA

**What blocks fusion (soft barriers):**
- Scatter operations (`.at[].add()`) — can accept fused inputs but limit
  downstream fusion
- Reduce operations — same as scatter
- "Expensive" operations (convolution, sort, all-reduce)

**Their case study** (Cart-pole simulation, similar bottleneck profile):
- Removing a `jnp.concatenate` fusion barrier: **3.41x speedup**
- Unrolling a scan loop by 10x: **3.5x speedup**
- Combined: **10.56x total**

The mechanism: with full fusion, values stay in GPU registers instead of
being written to global memory between kernels.

**Link:** https://arxiv.org/abs/2301.13062

## Finding 4: lax.scan on GPU = Kernel Per Iteration

**Source:** JAX Discussion #16106, #10233, Issue #16611.

On GPU, each `lax.scan` iteration corresponds to a kernel launch. XLA GPU
executes dynamic control flow on the CPU. A JAX maintainer stated:
"loops have high overhead on GPU because each iteration corresponds to a
kernel launch."

**Fix options:**
- `lax.scan(f, init, xs, unroll=N)` — inlines N iterations per kernel
- `unroll=True` — fully unrolls (may increase compile time)
- Python for-loop inside `@jit` — fully unrolled at trace time, allows
  cross-iteration fusion

**Benchmarks (from JAX issues):**
- Unrolled Python loop: 84.9ms (11.15s compile)
- `lax.scan`: 202ms (2.58s compile) — 2.4x slower execution

For small, known iteration counts (3-32), full unrolling is strongly preferred.

**Links:**
- https://github.com/jax-ml/jax/discussions/16106
- https://github.com/jax-ml/jax/discussions/10233
- https://github.com/jax-ml/jax/issues/16611

## Finding 5: PureJaxRL Architecture

**Source:** PureJaxRL (luchris429/purejaxrl), JaxMARL (arXiv:2311.10090).

The "Anakin" architecture compiles the entire training loop (env.step + policy
+ loss + gradient) into a single XLA program via `lax.scan` over timesteps.
No CPU-GPU transfers during training.

Performance: CartPole 1000x faster than gym, JaxMARL QMIX 21,500x throughput
improvement per agent. Training 512 agents on a single A40 in ~9 hours.

**Implication for FactoriaX:** The env step function is the innermost loop of
the entire training pipeline. Every unnecessary kernel launch in `step` is
multiplied by (num_envs x timesteps x epochs). Even a 2x improvement in step
throughput halves total training time.

**Links:**
- https://github.com/luchris429/purejaxrl
- https://chrislu.page/blog/meta-disco/
- https://arxiv.org/abs/2311.10090

## Finding 6: XLA Debugging Tools

**How to count kernels:**
```python
compiled = jax.jit(fn).lower(*args).compile()
hlo_text = compiled.as_text()
# Count 'fusion {' blocks — each is one GPU kernel
```

**How to dump full HLO for inspection:**
```bash
XLA_FLAGS="--xla_dump_to=/tmp/xladump --xla_dump_hlo_as_text" python script.py
```

**XLA flags to try (zero code change):**
- `--xla_gpu_graph_min_graph_size=2` — capture more ops into CUDA graphs
- `--xla_gpu_command_buffer_unroll_loops=true` — unroll loops in command buffers

**Visual profiling:**
```bash
nsys profile -o profile python scripts/profile_step.py --decompose
```

## Other Relevant Projects

| Project | Paper | Key Technique |
|---------|-------|---------------|
| Craftax | arXiv:2402.16801 | Branchless game logic, lax.select everywhere |
| Pgx | arXiv:2303.17503 | Frozen dataclass state, pure functions |
| EconoJax | arXiv:2410.22165 | 1D vector state, 750x over CPU |
| Gymnax | RobertTLange/gymnax | Reference JAX env implementations |
| Brax | google/brax | Physics sim, scan-based stepping |

## Recommended Reading Order

1. **Craftax source code** (game_logic.py) — concrete examples of every pattern
2. **XLA fusion paper** (arXiv:2301.13062) — understanding what blocks fusion
3. **JAX Issue #7934** — the lax.cond problem, explained by core devs
4. **PureJaxRL blog post** — why env step speed matters for training
