"""JAX parallel renderer benchmark for FactoriaX.

Compares rendering throughput across batch sizes for three approaches:
  1. Step only (no render) -- simulation ceiling.
  2. Step + map render -- map with terrain, machines, players.
  3. Step + full HUD -- map + 4-quadrant info panel.

A CPU step+render baseline (single env, NumPy renderer) is shown as a
dashed line for reference.

All rendering uses the pure-JAX renderer from factoriax.jax_renderer,
which compiles to XLA and vmaps across batched environment states.

Usage:
    uv run python scripts/jax_render_benchmark.py
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use("Agg")

from factoriax.constants import BlockType, Direction, ItemType, MachineType
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.jax_renderer import (
    JaxRenderer,
    render_hud,
    render_map,
)
from factoriax.renderer import render_pixels
from factoriax.state import EnvParams, EnvState

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAP_SIZE = 16
TILE_PX = 8
NUM_STEPS = 200
MAX_BATCH_POWER = 13  # Try up to 2^13 = 8192.


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def make_batched_envs(
    n: int, params: EnvParams
) -> tuple[jnp.ndarray, EnvState]:
    """Create N parallel environment states via vmapped reset.

    Args:
        n: Number of parallel environments.
        params: Shared environment parameters.

    Returns:
        Tuple of (rng_keys, batched_states).
    """
    env = FactoriaXEnv()
    keys = jax.random.split(jax.random.key(42), n)
    _, states = jax.vmap(env.reset_env, in_axes=(0, None))(keys, params)
    return keys, states


def extract_single_state(batched_state: EnvState, idx: int) -> EnvState:
    """Extract a single state from a batched pytree.

    Args:
        batched_state: Batched state with leading batch dimension.
        idx: Index to extract.

    Returns:
        Single-env EnvState.
    """
    return jax.tree.map(lambda x: x[idx], batched_state)


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


def _make_vmap_step(
    n_envs: int, params: EnvParams
) -> tuple[callable, EnvState, jnp.ndarray]:
    """Create vmapped step function, initial states, and warmup.

    Args:
        n_envs: Number of parallel environments.
        params: Environment parameters.

    Returns:
        Tuple of (vmap_step, states, rng).
    """
    env = FactoriaXEnv()
    _, states = make_batched_envs(n_envs, params)
    vmap_step = jax.jit(
        jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
    )
    rng = jax.random.key(99)
    rng, k_a, k_s = jax.random.split(rng, 3)
    actions = jax.random.randint(k_a, (n_envs,), 0, params.NUM_ACTIONS)
    step_keys = jax.random.split(k_s, n_envs)
    _, states, _, _, _ = vmap_step(step_keys, states, actions, params)
    jax.block_until_ready(jax.tree.leaves(states))
    return vmap_step, states, rng


def _step_loop(
    vmap_step: callable,
    states: EnvState,
    rng: jnp.ndarray,
    n_envs: int,
    params: EnvParams,
    num_steps: int,
    render_fn: object = None,
) -> float:
    """Run step loop (optionally with render), return wall-clock seconds.

    Args:
        vmap_step: Vmapped step function.
        states: Initial batched states.
        rng: RNG key.
        n_envs: Batch size.
        params: Environment parameters.
        num_steps: Steps to run.
        render_fn: Optional callable(states) for rendering.

    Returns:
        Wall-clock seconds.
    """
    t0 = time.perf_counter()
    for _ in range(num_steps):
        rng, k_a, k_s = jax.random.split(rng, 3)
        actions = jax.random.randint(k_a, (n_envs,), 0, params.NUM_ACTIONS)
        step_keys = jax.random.split(k_s, n_envs)
        _, states, _, _, _ = vmap_step(step_keys, states, actions, params)
        if render_fn is not None:
            result = render_fn(states)
    if render_fn is not None:
        jax.block_until_ready(result)
    else:
        jax.block_until_ready(jax.tree.leaves(states))
    t1 = time.perf_counter()
    return t1 - t0


def benchmark_step_only(
    n_envs: int, params: EnvParams, num_steps: int
) -> float:
    """Benchmark vmapped env stepping only (no rendering).

    Args:
        n_envs: Batch size.
        params: Environment parameters.
        num_steps: Steps to run.

    Returns:
        Wall-clock seconds (excluding warmup).
    """
    vmap_step, states, rng = _make_vmap_step(n_envs, params)
    return _step_loop(vmap_step, states, rng, n_envs, params, num_steps)


def benchmark_step_map(
    n_envs: int, params: EnvParams, num_steps: int,
    renderer: JaxRenderer,
) -> float:
    """Benchmark step + map render.

    Args:
        n_envs: Batch size.
        params: Environment parameters.
        num_steps: Steps to run.
        renderer: JaxRenderer instance.

    Returns:
        Wall-clock seconds (excluding warmup).
    """
    vmap_step, states, rng = _make_vmap_step(n_envs, params)
    vmap_render = renderer._vmap_render_map
    # Warmup render.
    result = vmap_render(states, renderer.block_atlas,
                         renderer.machine_atlas, renderer.player_sprite)
    jax.block_until_ready(result)

    def _render(s: EnvState) -> jnp.ndarray:
        return vmap_render(s, renderer.block_atlas,
                           renderer.machine_atlas, renderer.player_sprite)

    return _step_loop(vmap_step, states, rng, n_envs, params,
                      num_steps, _render)


def benchmark_step_hud(
    n_envs: int, params: EnvParams, num_steps: int,
    renderer: JaxRenderer,
) -> float:
    """Benchmark step + full HUD render.

    Args:
        n_envs: Batch size.
        params: Environment parameters.
        num_steps: Steps to run.
        renderer: JaxRenderer instance.

    Returns:
        Wall-clock seconds (excluding warmup).
    """
    vmap_step, states, rng = _make_vmap_step(n_envs, params)
    vmap_render = renderer._vmap_render_hud
    # Warmup render.
    result = vmap_render(
        states, renderer.block_atlas, renderer.machine_atlas,
        renderer.player_sprite, renderer.item_colors, renderer.digit_atlas,
    )
    jax.block_until_ready(result)

    def _render(s: EnvState) -> jnp.ndarray:
        return vmap_render(
            s, renderer.block_atlas, renderer.machine_atlas,
            renderer.player_sprite, renderer.item_colors,
            renderer.digit_atlas,
        )

    return _step_loop(vmap_step, states, rng, n_envs, params,
                      num_steps, _render)


def benchmark_cpu_step_render(
    params: EnvParams,
    num_steps: int,
    tile_px: int,
    log_fn: object = None,
) -> float:
    """Benchmark single-env step + NumPy render on the CPU backend.

    Args:
        params: Environment parameters.
        num_steps: Number of timesteps.
        tile_px: Tile pixel size for render_pixels.
        log_fn: Optional logging function for progress.

    Returns:
        Wall-clock seconds (excluding warmup).
    """
    _log = log_fn or (lambda *a, **kw: None)
    cpu_device = jax.devices("cpu")[0]
    env = FactoriaXEnv()

    with jax.default_device(cpu_device):
        key = jax.random.key(55)
        _, state = env.reset_env(key, params)
        step_fn = jax.jit(env.step_env, device=cpu_device)

        _log(" compiling on CPU...", end="", flush=True)
        rng = jax.random.key(56)
        rng, k_a, k_s = jax.random.split(rng, 3)
        action = jax.random.randint(k_a, (), 0, params.NUM_ACTIONS)
        _, state, _, _, _ = step_fn(k_s, state, action, params)
        jax.block_until_ready(jax.tree.leaves(state))
        render_pixels(state, block_pixel_size=tile_px)
        _log(" done.", end="", flush=True)

        t0 = time.perf_counter()
        for _ in range(num_steps):
            rng, k_a, k_s = jax.random.split(rng, 3)
            action = jax.random.randint(k_a, (), 0, params.NUM_ACTIONS)
            _, state, _, _, _ = step_fn(k_s, state, action, params)
            jax.block_until_ready(jax.tree.leaves(state))
            render_pixels(state, block_pixel_size=tile_px)
        t1 = time.perf_counter()

    return t1 - t0


# ---------------------------------------------------------------------------
# OOM-safe wrappers
# ---------------------------------------------------------------------------


def _try(fn: callable, *args: object) -> float | None:
    """Call fn(*args), returning None on OOM.

    Args:
        fn: Benchmark function.
        *args: Arguments to pass.

    Returns:
        Result or None on OOM.
    """
    try:
        return fn(*args)
    except (RuntimeError, jax.errors.JaxRuntimeError):
        return None


# ---------------------------------------------------------------------------
# Auto-scaling discovery
# ---------------------------------------------------------------------------


def discover_scaling(
    run_fn: callable,
    label: str,
    log_fn: callable,
) -> dict[int, float]:
    """Double batch size until OOM, recording fps at each step.

    Args:
        run_fn: Callable(n_envs) -> seconds | None.
        label: Label for log output.
        log_fn: Logging function.

    Returns:
        Dict of batch_size -> fps.
    """
    results: dict[int, float] = {}
    header = f"{'n_envs':>8} | {'time (s)':>10} | {'fps':>12}"
    log_fn(f"--- {label} ---")
    log_fn(header)
    log_fn("-" * len(header))

    for power in range(MAX_BATCH_POWER + 1):
        n = 1 << power
        elapsed = run_fn(n)
        if elapsed is None:
            log_fn(f"{n:>8} | {'OOM':>10} |")
            break
        total_frames = n * NUM_STEPS
        fps = total_frames / elapsed if elapsed > 0 else float("inf")
        results[n] = fps
        log_fn(f"{n:>8} | {elapsed:>10.3f} | {fps:>12,.0f}")

    return results


# ---------------------------------------------------------------------------
# Sample images
# ---------------------------------------------------------------------------


def save_sample_images(
    params: EnvParams,
    renderer: JaxRenderer,
    output_dir: Path,
) -> None:
    """Save sample rendered images for visual comparison.

    Args:
        params: Environment parameters.
        renderer: JaxRenderer instance.
        output_dir: Directory to write images into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # CPU render: single environment.
    _, single_states = make_batched_envs(1, params)
    state = extract_single_state(single_states, 0)
    cpu_img = render_pixels(state, block_pixel_size=renderer.tile_px)
    Image.fromarray(cpu_img[:, :, :3]).save(output_dir / "cpu_render.png")

    # JAX HUD render: batch of 4, stitched 2x2.
    _, batch_states = make_batched_envs(4, params)
    batch_imgs = np.array(renderer.vmap_render_hud(batch_states))
    top_row = np.concatenate([batch_imgs[0], batch_imgs[1]], axis=1)
    bot_row = np.concatenate([batch_imgs[2], batch_imgs[3]], axis=1)
    grid = np.concatenate([top_row, bot_row], axis=0)
    Image.fromarray(grid).save(output_dir / "jax_batch4.png")

    # Full HUD with items in inventory.
    from factoriax.levels import LevelBuilder, build_state

    level = (
        LevelBuilder(params.map_width, params.map_height)
        .fill_rect(2, 2, 3, 3, BlockType.IRON, resources=500)
        .fill_rect(10, 2, 3, 3, BlockType.COPPER, resources=300)
        .fill_rect(2, 10, 3, 3, BlockType.COAL, resources=200)
        .place_machine(6, 6, int(MachineType.MINER), int(Direction.DOWN))
        .set_player_position(6, 5)
        .build("hud_sample")
    )
    level.player_inventory = [
        (int(ItemType.IRON), 42),
        (int(ItemType.COPPER), 7),
        (int(ItemType.COAL), 1),
        (int(ItemType.MINER), 3),
        (int(ItemType.CHEST), 12),
    ]
    hud_state = build_state(level, params)
    hud_img = np.array(renderer.jit_render_hud(hud_state))
    Image.fromarray(hud_img).save(output_dir / "jax_full_hud.png")

    print(f"Sample images saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------


def save_throughput_chart(
    cpu_fps: float,
    step_render_fps: dict[int, float],
    step_render_hud_fps: dict[int, float],
    vector_fps: dict[int, float],
    output_dir: Path,
) -> None:
    """Save the throughput chart with three GPU series.

    Args:
        cpu_fps: CPU step+render throughput (fps), single env.
        step_render_fps: batch_size -> step + map render fps.
        step_render_hud_fps: batch_size -> step + full HUD fps.
        vector_fps: batch_size -> step-only fps.
        output_dir: Directory to write the chart into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_sizes = sorted(
        set(step_render_fps)
        & set(vector_fps)
        & set(step_render_hud_fps)
    )
    if not all_sizes:
        return

    labels = [str(n) for n in all_sizes]
    x = np.arange(len(labels))
    width = 0.25

    v_vals = np.array([vector_fps[n] for n in all_sizes])
    sr_vals = np.array([step_render_fps[n] for n in all_sizes])
    hud_vals = np.array([step_render_hud_fps[n] for n in all_sizes])

    fig, ax = plt.subplots(figsize=(14, 5))

    series = [
        (x - width, v_vals, "#5fbf5f", "Step only"),
        (x, sr_vals, "#5f8fd4", "Step + map render"),
        (x + width, hud_vals, "#d4a05f", "Step + full HUD"),
    ]
    for pos, vals, color, label in series:
        ax.bar(
            pos, vals, width, color=color,
            edgecolor="white", linewidth=0.5, label=label,
        )

    ax.axhline(
        cpu_fps, color="#d45f5f", linestyle="--", linewidth=1.5,
        label=f"CPU step+render ({cpu_fps:,.0f} fps)",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Batch size (envs)", fontsize=12)
    ax.set_ylabel("Steps per second", fontsize=12)
    ax.set_title(
        f"Throughput: step vs map render vs full HUD  "
        f"({MAP_SIZE}x{MAP_SIZE} map, {TILE_PX}px tiles, {NUM_STEPS} steps)",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    path = output_dir / "throughput_step_render.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Throughput chart saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full benchmark suite."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path("scripts/batch_results") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "benchmark.log"
    log_lines: list[str] = []

    def log(msg: str = "", end: str = "\n", flush: bool = False) -> None:
        """Print and log.

        Args:
            msg: Message.
            end: Line ending.
            flush: Flush stdout.
        """
        print(msg, end=end, flush=flush)
        log_lines.append(msg + end)

    renderer = JaxRenderer(tile_px=TILE_PX)

    log("FactoriaX JAX Renderer Benchmark")
    log("=" * 60)
    log(f"Timestamp: {timestamp}")
    log(f"Map: {MAP_SIZE}x{MAP_SIZE}, Tile: {TILE_PX}px")
    log(f"Steps: {NUM_STEPS}, Max batch: 2^{MAX_BATCH_POWER}")
    log(f"JAX devices: {jax.devices()}")
    log(f"Output: {output_dir}")
    log()

    params = EnvParams(
        map_width=MAP_SIZE,
        map_height=MAP_SIZE,
        num_players=1,
        max_timesteps=NUM_STEPS,
        nest_probability=0.0,
        max_biters=1,
    )

    save_sample_images(params, renderer, output_dir)

    # ---- CPU baseline ----
    log("Measuring CPU step+render baseline (1 env)...", end="", flush=True)
    cpu_sr_time = benchmark_cpu_step_render(params, NUM_STEPS, TILE_PX, log)
    cpu_sr_fps = NUM_STEPS / cpu_sr_time
    log(f" {cpu_sr_fps:,.0f} fps ({cpu_sr_time:.3f}s)")
    log()

    # ---- Series 1: step + map render ----
    step_render_fps = discover_scaling(
        run_fn=lambda n: _try(benchmark_step_map, n, params, NUM_STEPS,
                              renderer),
        label="Step + Map Render",
        log_fn=log,
    )
    log()

    # ---- Series 2: step + full HUD render ----
    step_render_hud_fps = discover_scaling(
        run_fn=lambda n: _try(benchmark_step_hud, n, params, NUM_STEPS,
                              renderer),
        label="Step + Full HUD Render",
        log_fn=log,
    )
    log()

    # ---- Series 3: step only ----
    vector_fps = discover_scaling(
        run_fn=lambda n: _try(benchmark_step_only, n, params, NUM_STEPS),
        label="Step Only (no render)",
        log_fn=log,
    )

    # ---- Save outputs ----
    save_throughput_chart(
        cpu_sr_fps, step_render_fps, step_render_hud_fps,
        vector_fps, output_dir,
    )

    log()
    log(f"All results saved to {output_dir}/")
    log_path.write_text("".join(log_lines))


if __name__ == "__main__":
    main()
