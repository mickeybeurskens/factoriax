"""Reproducible profiling script for FactoriaX environment stepping.

Four profiling layers:
    --throughput   Wall-clock scaling curve across batch sizes
    --decompose    Sub-operation timing breakdown
    --roofline     XLA cost analysis for compute vs memory classification
    --trace        Perfetto trace for visual kernel inspection
    --all          Run all layers

Usage:
    python scripts/profile_step.py --all
    python scripts/profile_step.py --throughput --map-size 32
    python scripts/profile_step.py --decompose --batch 1024
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

from factoriax.achievements import core_game_conditions
from factoriax.constants import NUM_ACTIONS
from factoriax.crafting import update_crafting
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.game_logic import _handle_player_action
from factoriax.machines import (
    push_miner_output,
    refuel_machines,
    run_arms,
    run_assemblers,
    run_conveyor_belts,
    run_miners,
)
from factoriax.observations import global_array
from factoriax.rewards import achievement_reward
from factoriax.state import EnvParams, EnvState

logger = logging.getLogger(__name__)

# RTX 3050 6GB Laptop GPU specs.
GPU_PEAK_TFLOPS = 4.5
GPU_BW_GBS = 192.0
RIDGE_POINT = GPU_PEAK_TFLOPS * 1e12 / (GPU_BW_GBS * 1e9)

WARMUP_ITERS = 5
TIMED_ITERS = 50

THROUGHPUT_BATCHES = [1, 4, 16, 64, 256, 1024, 4096, 8192]

PROFILING_DIR = Path("docs/profiling")
TRACE_DIR = PROFILING_DIR / "traces"


@dataclass
class OpSpec:
    """Specification for a profilable sub-operation.

    Attributes:
        name: Human-readable operation name.
        build_fn: Callable that takes (states, params, rng) and returns
            a tuple of (jitted_fn, args) ready to call and benchmark.
    """

    name: str
    build_fn: object  # Callable[[EnvState, EnvParams, jax.Array], tuple]


def _make_batched_states(
    n: int, params: EnvParams
) -> tuple[jax.Array, EnvState]:
    """Create n parallel environment states via vmapped reset.

    Args:
        n: Number of parallel environments.
        params: Environment parameters.

    Returns:
        Tuple of (rng_keys, batched_states).
    """
    env = FactoriaXEnv()
    keys = jax.random.split(jax.random.key(42), n)
    _, states = jax.vmap(env.reset_env, in_axes=(0, None))(keys, params)
    return keys, states


def _benchmark_fn(
    fn: object,
    args: tuple,
    warmup: int = WARMUP_ITERS,
    iters: int = TIMED_ITERS,
) -> float:
    """Benchmark a JIT-compiled function with proper synchronization.

    Args:
        fn: JIT-compiled callable.
        args: Arguments to pass.
        warmup: Number of warmup iterations.
        iters: Number of timed iterations.

    Returns:
        Median time per call in seconds.
    """
    for _ in range(warmup):
        result = fn(*args)
        jax.block_until_ready(jax.tree.leaves(result))

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        result = fn(*args)
        jax.block_until_ready(jax.tree.leaves(result))
        times.append(time.perf_counter() - t0)

    times.sort()
    return times[len(times) // 2]


def _build_op_specs(
    params: EnvParams,
) -> list[tuple[str, object, object]]:
    """Build (name, vmapped_fn, in_axes) specs for all sub-operations.

    Each entry returns a function that accepts batched arguments and
    can be JIT-compiled and benchmarked.

    Args:
        params: Environment parameters.

    Returns:
        List of (name, raw_fn, in_axes_for_vmap) tuples. The caller
        is responsible for constructing the correct args for each.
    """
    max_asm = params.max_assembler_stack_size
    return [
        (
            "handle_player_action",
            lambda s, a, p: _handle_player_action(s, a, p),
            "custom_player_action",
        ),
        ("update_crafting", update_crafting, "state_only"),
        ("refuel_machines", refuel_machines, "state_params"),
        ("run_assemblers", run_assemblers, "state_params"),
        ("run_miners", run_miners, "state_params"),
        (
            "push_miner_output",
            lambda s: push_miner_output(s, max_asm),
            "state_only",
        ),
        ("run_conveyor_belts", run_conveyor_belts, "state_only"),
        (
            "run_arms",
            lambda s: run_arms(s, max_asm),
            "state_only",
        ),
        ("core_game_conditions", core_game_conditions, "state_only"),
        (
            "achievement_reward",
            lambda prev, new, p: achievement_reward(prev, new, p),
            "custom_reward",
        ),
        (
            "global_array",
            lambda s, p, idx: global_array(s, p, idx),
            "custom_obs",
        ),
        (
            "step_env [full]",
            None,
            "custom_step_env",
        ),
    ]


def _build_jitted_op(
    name: str,
    fn: object,
    kind: str,
    states: EnvState,
    params: EnvParams,
    n_envs: int,
) -> tuple[object, tuple]:
    """Build a JIT-compiled, vmapped operation with its arguments.

    Args:
        name: Operation name (for logging).
        fn: Raw function to wrap.
        kind: Vmap pattern identifier.
        states: Batched environment states.
        params: Environment parameters.
        n_envs: Batch size.

    Returns:
        Tuple of (jitted_fn, args).
    """
    rng = jax.random.key(99)
    rngs = jax.random.split(rng, n_envs)
    actions = jax.random.randint(
        jax.random.key(7), (n_envs,), 0, NUM_ACTIONS
    )
    player_idxs = jnp.zeros(n_envs, dtype=jnp.int32)

    if kind == "state_only":
        jitted = jax.jit(jax.vmap(fn))
        args = (states,)
    elif kind == "state_params":
        jitted = jax.jit(jax.vmap(fn, in_axes=(0, None)))
        args = (states, params)
    elif kind == "state_params_rng":
        jitted = jax.jit(jax.vmap(fn, in_axes=(0, None, 0)))
        args = (states, params, rngs)
    elif kind == "custom_player_action":
        jitted = jax.jit(jax.vmap(fn, in_axes=(0, 0, 0)))
        args = (states, actions, player_idxs)
    elif kind == "custom_reward":
        jitted = jax.jit(jax.vmap(fn, in_axes=(0, 0, None)))
        args = (states, states, params)
    elif kind == "custom_obs":
        jitted = jax.jit(jax.vmap(fn, in_axes=(0, None, 0)))
        args = (states, params, player_idxs)
    elif kind == "custom_step_env":
        env = FactoriaXEnv()
        jitted = jax.jit(jax.vmap(env.step_env, in_axes=(0, 0, 0, None)))
        args = (rngs, states, actions, params)
    else:
        msg = f"Unknown op kind: {kind}"
        raise ValueError(msg)

    return jitted, args


# ── Layer 1: Throughput curve ───────────────────────────────────────


def run_throughput(params: EnvParams) -> list[dict]:
    """Measure wall-clock throughput across batch sizes.

    Args:
        params: Environment parameters.

    Returns:
        List of result dicts per batch size.
    """
    print("\n" + "=" * 70)
    print("LAYER 1: Throughput Scaling Curve")
    print("=" * 70)

    env = FactoriaXEnv()
    results = []

    for n_envs in THROUGHPUT_BATCHES:
        try:
            _, states = _make_batched_states(n_envs, params)
            vmap_step = jax.jit(
                jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
            )

            rng = jax.random.key(77)
            actions = jnp.zeros(n_envs, dtype=jnp.int32)
            step_keys = jax.random.split(rng, n_envs)

            # Warmup.
            for _ in range(WARMUP_ITERS):
                _, states, _, _, _ = vmap_step(
                    step_keys, states, actions, params
                )
            jax.block_until_ready(jax.tree.leaves(states))

            # Timed run.
            rng_t = jax.random.key(88)
            t0 = time.perf_counter()
            for i in range(TIMED_ITERS):
                rng_t, k_a, k_s = jax.random.split(rng_t, 3)
                actions = jax.random.randint(
                    k_a, (n_envs,), 0, NUM_ACTIONS
                )
                step_keys = jax.random.split(k_s, n_envs)
                _, states, _, _, _ = vmap_step(
                    step_keys, states, actions, params
                )
            jax.block_until_ready(jax.tree.leaves(states))
            elapsed = time.perf_counter() - t0

            total_steps = n_envs * TIMED_ITERS
            sps = total_steps / elapsed
            per_env_us = elapsed / TIMED_ITERS * 1e6

            row = {
                "batch": n_envs,
                "steps_per_sec": sps,
                "per_env_us": per_env_us,
            }
            results.append(row)
            print(
                f"  batch={n_envs:>5d}  "
                f"{sps:>10.0f} steps/s  "
                f"{per_env_us:>8.0f} us/step"
            )
        except Exception as e:
            print(f"  batch={n_envs:>5d}  FAILED: {e}")
            break

    return results


# ── Layer 2: Sub-operation decomposition ────────────────────────────


def run_decompose(
    params: EnvParams, batch: int
) -> list[dict]:
    """Time each sub-operation independently at a fixed batch size.

    Args:
        params: Environment parameters.
        batch: Number of parallel environments.

    Returns:
        List of result dicts per operation.
    """
    print("\n" + "=" * 70)
    print(f"LAYER 2: Sub-operation Decomposition (batch={batch})")
    print("=" * 70)

    _, states = _make_batched_states(batch, params)
    ops = _build_op_specs(params)
    results = []

    for name, fn, kind in ops:
        jitted, args = _build_jitted_op(
            name, fn, kind, states, params, batch
        )
        median_s = _benchmark_fn(jitted, args)
        results.append({
            "name": name,
            "median_us": median_s * 1e6,
        })

    # Sort by time descending.
    results.sort(key=lambda r: r["median_us"], reverse=True)

    # Find full step_env time for percentage calculation.
    full_time = next(
        (r["median_us"] for r in results if "full" in r["name"]),
        1.0,
    )

    print(f"\n  {'Operation':<30s} {'Time (us)':>10s} {'% of full':>10s}")
    print(f"  {'-' * 30} {'-' * 10} {'-' * 10}")
    for r in results:
        pct = r["median_us"] / full_time * 100
        r["pct_of_full"] = pct
        print(
            f"  {r['name']:<30s} "
            f"{r['median_us']:>10.1f} "
            f"{pct:>9.1f}%"
        )

    return results


# ── Layer 3: Roofline classification ────────────────────────────────


def run_roofline(
    params: EnvParams, batch: int
) -> list[dict]:
    """Classify each sub-operation as compute or memory bound.

    Uses XLA cost_analysis() to get theoretical FLOPs and bytes,
    then compares arithmetic intensity against the GPU ridge point.

    Args:
        params: Environment parameters.
        batch: Number of parallel environments.

    Returns:
        List of result dicts per operation.
    """
    print("\n" + "=" * 70)
    print(f"LAYER 3: Roofline Classification (batch={batch})")
    print(f"  GPU: {GPU_PEAK_TFLOPS} TFLOPS, {GPU_BW_GBS} GB/s")
    print(f"  Ridge point: {RIDGE_POINT:.1f} FLOPs/byte")
    print("=" * 70)

    _, states = _make_batched_states(batch, params)
    ops = _build_op_specs(params)
    results = []

    for name, fn, kind in ops:
        jitted, args = _build_jitted_op(
            name, fn, kind, states, params, batch
        )

        # Time it.
        median_s = _benchmark_fn(jitted, args)

        # Cost analysis via lowered compilation.
        try:
            lowered = jitted.lower(*args)
            compiled = lowered.compile()
            cost_list = compiled.cost_analysis()
            cost = cost_list[0] if cost_list else {}
        except Exception:
            cost = {}

        flops = cost.get("flops", 0)
        bytes_accessed = cost.get("bytes accessed", 0)

        if bytes_accessed > 0:
            ai = flops / bytes_accessed
        else:
            ai = float("inf") if flops > 0 else 0.0

        classification = "compute" if ai >= RIDGE_POINT else "memory"

        # Theoretical best time.
        if classification == "memory" and bytes_accessed > 0:
            theoretical_s = bytes_accessed / (GPU_BW_GBS * 1e9)
        elif flops > 0:
            theoretical_s = flops / (GPU_PEAK_TFLOPS * 1e12)
        else:
            theoretical_s = 0.0

        efficiency = (
            theoretical_s / median_s * 100 if median_s > 0 else 0.0
        )

        row = {
            "name": name,
            "flops": flops,
            "bytes": bytes_accessed,
            "ai": ai,
            "classification": classification,
            "measured_us": median_s * 1e6,
            "theoretical_us": theoretical_s * 1e6,
            "efficiency_pct": efficiency,
        }
        results.append(row)

    print(
        f"\n  {'Operation':<26s} {'FLOPs':>12s} {'Bytes':>12s} "
        f"{'AI':>8s} {'Class':>8s} {'Meas(us)':>9s} "
        f"{'Theo(us)':>9s} {'Eff%':>6s}"
    )
    print(f"  {'-' * 26} " + " ".join(["-" * w for w in [12, 12, 8, 8, 9, 9, 6]]))
    for r in results:
        ai_str = f"{r['ai']:.1f}" if r["ai"] != float("inf") else "inf"
        print(
            f"  {r['name']:<26s} "
            f"{r['flops']:>12,d} "
            f"{r['bytes']:>12,d} "
            f"{ai_str:>8s} "
            f"{r['classification']:>8s} "
            f"{r['measured_us']:>9.1f} "
            f"{r['theoretical_us']:>9.1f} "
            f"{r['efficiency_pct']:>5.1f}%"
        )

    return results


# ── Layer 4: Perfetto trace ─────────────────────────────────────────


def run_trace(
    params: EnvParams, batch: int, trace_steps: int = 100
) -> None:
    """Generate a Perfetto trace of a short rollout.

    Args:
        params: Environment parameters.
        batch: Number of parallel environments.
        trace_steps: Number of steps to trace.
    """
    print("\n" + "=" * 70)
    print(f"LAYER 4: Perfetto Trace (batch={batch}, steps={trace_steps})")
    print("=" * 70)

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = str(TRACE_DIR)

    env = FactoriaXEnv()
    _, states = _make_batched_states(batch, params)
    vmap_step = jax.jit(
        jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
    )

    # Warmup outside trace.
    rng = jax.random.key(77)
    actions = jnp.zeros(batch, dtype=jnp.int32)
    step_keys = jax.random.split(rng, batch)
    _, states, _, _, _ = vmap_step(step_keys, states, actions, params)
    jax.block_until_ready(jax.tree.leaves(states))

    print(f"  Writing trace to {trace_path}/")
    with jax.profiler.trace(trace_path):
        rng_t = jax.random.key(88)
        for _ in range(trace_steps):
            rng_t, k_a, k_s = jax.random.split(rng_t, 3)
            actions = jax.random.randint(
                k_a, (batch,), 0, NUM_ACTIONS
            )
            step_keys = jax.random.split(k_s, batch)
            _, states, _, _, _ = vmap_step(
                step_keys, states, actions, params
            )
        jax.block_until_ready(jax.tree.leaves(states))

    print("  Trace complete. Open in https://ui.perfetto.dev/")


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    """Run profiling layers based on CLI flags."""
    parser = argparse.ArgumentParser(
        description="Profile FactoriaX environment stepping.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all layers"
    )
    parser.add_argument(
        "--throughput", action="store_true", help="Throughput scaling"
    )
    parser.add_argument(
        "--decompose", action="store_true", help="Sub-op decomposition"
    )
    parser.add_argument(
        "--roofline", action="store_true", help="Roofline classification"
    )
    parser.add_argument(
        "--trace", action="store_true", help="Perfetto trace"
    )
    parser.add_argument(
        "--batch", type=int, default=1024,
        help="Batch size for decompose/roofline (default: 1024)",
    )
    parser.add_argument(
        "--map-size", type=int, default=32,
        help="Map width and height (default: 32)",
    )
    args = parser.parse_args()

    if not any([args.all, args.throughput, args.decompose,
                args.roofline, args.trace]):
        parser.print_help()
        return

    params = EnvParams(
        map_width=args.map_size,
        map_height=args.map_size,
        num_players=1,
    )

    print(f"Device: {jax.devices()[0]}")
    print(f"Map: {args.map_size}x{args.map_size}, Players: 1")

    all_results: dict = {}

    if args.all or args.throughput:
        all_results["throughput"] = run_throughput(params)

    if args.all or args.decompose:
        all_results["decompose"] = run_decompose(params, args.batch)

    if args.all or args.roofline:
        all_results["roofline"] = run_roofline(params, args.batch)

    if args.all or args.trace:
        run_trace(params, args.batch)

    # Write JSONL log.
    if all_results:
        PROFILING_DIR.mkdir(parents=True, exist_ok=True)
        jsonl_path = PROFILING_DIR / "autoresearch.jsonl"
        ts = int(time.time())
        with open(jsonl_path, "a") as f:
            f.write(json.dumps({
                "type": "config",
                "name": "Baseline profiling",
                "metricName": "steps_per_sec",
                "metricUnit": "steps/s",
                "bestDirection": "higher",
                "timestamp": ts,
                "device": str(jax.devices()[0]),
                "map_size": args.map_size,
                "batch": args.batch,
            }) + "\n")
            for layer, data in all_results.items():
                f.write(json.dumps({
                    "type": "result",
                    "layer": layer,
                    "data": data,
                    "timestamp": ts,
                }) + "\n")
        print(f"\nResults appended to {jsonl_path}")


if __name__ == "__main__":
    main()
