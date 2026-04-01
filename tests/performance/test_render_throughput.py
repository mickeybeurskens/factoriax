"""Performance tests for rendering throughput.

Measures step+render throughput for map-only and full-HUD renderers
at various batch sizes.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import pytest

from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.jax_renderer import JaxRenderer, render_hud, render_map
from factoriax.state import EnvParams

from .conftest import NUM_STEPS, make_batched_envs

BATCH_SIZES = [1, 8, 64, 512]

# Minimum fps thresholds. Render adds minimal overhead on top of
# stepping, so these are slightly below the step-only thresholds.
MIN_MAP_FPS = {
    1: 50,
    8: 400,
    64: 3000,
    512: 8000,
}

MIN_HUD_FPS = {
    1: 30,
    8: 250,
    64: 2000,
    512: 5000,
}


def _run_step_render(
    n_envs: int,
    params: EnvParams,
    renderer: JaxRenderer,
    use_hud: bool,
) -> float:
    """Step + render for n_envs, return fps.

    Args:
        n_envs: Batch size.
        params: Environment parameters.
        renderer: JaxRenderer instance.
        use_hud: If True, render full HUD; else map only.

    Returns:
        Frames per second.
    """
    env = FactoriaXEnv()
    _, states = make_batched_envs(n_envs, params)
    vmap_step = jax.jit(
        jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
    )

    if use_hud:
        vmap_render = renderer._vmap_render_hud
        render_args = (
            renderer.block_atlas, renderer.machine_atlas,
            renderer.player_sprite, renderer.item_colors,
            renderer.digit_atlas,
        )
    else:
        vmap_render = renderer._vmap_render_map
        render_args = (
            renderer.block_atlas, renderer.machine_atlas,
            renderer.player_sprite,
        )

    # Warmup.
    rng = jax.random.key(99)
    rng, k_a, k_s = jax.random.split(rng, 3)
    actions = jax.random.randint(k_a, (n_envs,), 0, params.NUM_ACTIONS)
    step_keys = jax.random.split(k_s, n_envs)
    _, states, _, _, _ = vmap_step(step_keys, states, actions, params)
    result = vmap_render(states, *render_args)
    jax.block_until_ready(result)

    # Timed run.
    t0 = time.perf_counter()
    for _ in range(NUM_STEPS):
        rng, k_a, k_s = jax.random.split(rng, 3)
        actions = jax.random.randint(
            k_a, (n_envs,), 0, params.NUM_ACTIONS
        )
        step_keys = jax.random.split(k_s, n_envs)
        _, states, _, _, _ = vmap_step(
            step_keys, states, actions, params
        )
        result = vmap_render(states, *render_args)
    jax.block_until_ready(result)
    elapsed = time.perf_counter() - t0

    return n_envs * NUM_STEPS / elapsed


@pytest.mark.parametrize("n_envs", BATCH_SIZES)
def test_map_render_throughput(
    n_envs: int, perf_params: EnvParams, renderer: JaxRenderer
) -> None:
    """Step + map render throughput must exceed minimum fps.

    Args:
        n_envs: Batch size.
        perf_params: Shared environment parameters.
        renderer: Shared JaxRenderer.
    """
    fps = _run_step_render(n_envs, perf_params, renderer, use_hud=False)
    threshold = MIN_MAP_FPS.get(n_envs, 50)
    assert fps >= threshold, (
        f"Map render {fps:.0f} fps < {threshold} fps at batch={n_envs}"
    )


@pytest.mark.parametrize("n_envs", BATCH_SIZES)
def test_hud_render_throughput(
    n_envs: int, perf_params: EnvParams, renderer: JaxRenderer
) -> None:
    """Step + HUD render throughput must exceed minimum fps.

    Args:
        n_envs: Batch size.
        perf_params: Shared environment parameters.
        renderer: Shared JaxRenderer.
    """
    fps = _run_step_render(n_envs, perf_params, renderer, use_hud=True)
    threshold = MIN_HUD_FPS.get(n_envs, 30)
    assert fps >= threshold, (
        f"HUD render {fps:.0f} fps < {threshold} fps at batch={n_envs}"
    )
