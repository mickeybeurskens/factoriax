# Optimization Ideas Queue

Ranked by expected impact based on profiling data and literature research.
Updated 2026-04-10 after reviewing Craftax, XLA fusion paper, and JAX issues.

## Critical Priority (Launch-overhead Elimination)

### 1. Replace all lax.cond with jnp.where / lax.select

The single highest-impact change. There are 26 `lax.cond` calls across the
codebase. Each one is a fusion barrier AND a GPU-CPU sync point. Under vmap
(batched envs), `lax.cond` is already silently converted to `lax.select`,
so you compute both branches regardless, but without the fusion benefit.

The Craftax pattern: compute both branches unconditionally, select with
`jnp.where`. For state updates, use `tree_map` with `lax.select`:
```python
# Before (fusion barrier):
state = lax.cond(pred, do_update, lambda s: s, state)

# After (fusible):
new_state = do_update(state)
state = jax.tree.map(lambda n, o: jnp.where(pred, n, o), new_state, state)
```

For conditional scatter: `state.at[idx].add(jnp.where(pred, val, 0))`

**Targets (by impact):**
- `_handle_player_action` in game_logic.py — 11 cascading conds (12.5% of step)
- `update_biters` in biters.py — 1 cond gating all biter logic (29% of step)
- `deposit_item`, `withdraw_item` in game_logic.py — 1 cond each
- `place_machine`, `pickup_machine` in placement.py — 1 cond each
- Remaining conds in crafting.py, game_logic.py

**Evidence:** JAX Issue #7934 (confirmed by core devs), Craftax source code,
XLA fusion paper. Benchmark: single cond→select swap = ~20% speedup. 26
cascading conds = potentially 3-5x.

**Expected impact:** 2-5x throughput improvement (literature: XLA fusion paper
showed 10.56x on similar profile, Craftax achieved 250x over CPU baseline).

### 2. Unroll lax.scan loops with small known counts

On GPU, each `lax.scan` iteration is a kernel launch (confirmed JAX Discussion
#16106). There are 5 scan calls in the step path:

- `update_crafting`: scan over num_players (1-4 iterations)
- `crafting.py`: scan over MAX_RECIPE_INPUTS (3 iterations)
- `crafting.py`: scan over recipe check (small)
- `biters.py`: scan over max_biters (32 iterations)
- `game_logic.py`: scan in repair logic

For small counts, set `unroll=True` or replace with Python for-loops inside
`@jit` (fully unrolled at trace time). The XLA fusion paper showed 3.5x
speedup from scan unrolling alone.

**Expected impact:** 1.5-3x on affected operations.

## High Priority (Fusion Opportunities)

### 3. Fuse machine update phases

The 6 machine phases (refuel, assemblers, miners, push, belts, arms) run
sequentially but are separate functions. After eliminating lax.cond barriers
within each function, the main remaining fusion barriers between them are
scatter operations. Consider restructuring to reduce intermediate state
materializations.

**Target**: `update_all_machines` in machines.py (~76% of step combined)
**Expected impact**: 20-40% after lax.cond elimination

### 4. Reduce scatter-gather operations

74 `.at[]` scatter operations across the codebase. Each is a soft fusion
barrier. Where possible:
- Batch multiple scatter updates into one (accumulate deltas, apply once)
- Use `unique_indices=True` where indices don't overlap
- Combine related field updates

**Target**: machines.py, game_logic.py, biters.py
**Expected impact**: 10-20% after other optimizations

## Medium Priority

### 5. XLA compiler flags for CUDA graph capture

Try these flags to let XLA batch kernel launches:
- `--xla_gpu_graph_min_graph_size=2` (default 5): capture more into CUDA graphs
- `--xla_gpu_command_buffer_unroll_loops=true`: unroll loop commands in command buffers

These are zero-code-change experiments. Run profiling script with
`XLA_FLAGS="..."` env var.

**Expected impact**: ~5-15% from CUDA graph amortization

### 6. Dump HLO to count fusion blocks

Before and after each optimization, inspect the compiled HLO:
```python
compiled = jax.jit(fn).lower(*args).compile()
print(compiled.as_text())  # Count 'fusion { ... }' blocks = kernel count
```
This is how you measure whether a change actually reduced kernel count.

### 7. Profile with Nsight Systems

Visualize kernel launch gaps and GPU utilization:
```bash
nsys profile -o factoriax_profile python scripts/profile_step.py --decompose
```
Confirms diagnosis and identifies remaining hotspots after code changes.

## Low Priority

### 8. Smaller dtypes for grid arrays

Most machine arrays could use int8/int16. Halves bandwidth for those fields.
Only matters at very high batch sizes where bandwidth is the true bottleneck.

### 9. Fuse observation into step

`global_array` is 1.6% of step time. Could share data with step computation.
Low priority unless observation becomes more expensive.

### 10. Pallas custom kernels

Custom Triton kernels for fused gather-compute-scatter patterns. Last resort
after architectural changes (items 1-4). Benchmarks show Pallas matches JIT
for elementwise operations, only wins for complex memory patterns.
