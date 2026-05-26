"""Build a level with ``LevelBuilder``, save it, load it, and reset on it.

Round-trips a small hand-built level through JSON so researchers can
script-author levels and check them into the repo.

Run::

    uv run python examples/custom_level.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax

import factoriax
from factoriax import (
    BlockType,
    Direction,
    Level,
    LevelBuilder,
    build_state,
    load_level,
    save_level,
)
from factoriax.constants import Machine


def build_demo_level() -> Level:
    """Construct a small level with two ore patches and a placed miner."""
    return (
        LevelBuilder(12, 12)
        .fill_rect(1, 1, 3, 3, BlockType.IRON, resources=200)
        .fill_rect(8, 1, 3, 3, BlockType.COPPER, resources=200)
        .place_machine(2, 4, int(Machine.MINER), direction=int(Direction.UP))
        .build("demo_two_patches")
    )


def main(steps: int = 4) -> None:
    """Author a level, persist it, load it back, and run a few steps.

    Args:
        steps: Number of environment steps to take after loading. Kept
            small so the example finishes quickly.
    """
    level = build_demo_level()
    print(f"built level '{level.name}' ({level.map_width}×{level.map_height})")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{level.name}.json"
        save_level(level, path)
        print(f"wrote {path.name} ({path.stat().st_size} bytes)")
        reloaded = load_level(path)
        assert reloaded.name == level.name
        assert reloaded.block_map.shape == level.block_map.shape

    env, params = factoriax.make(reloaded)
    state = build_state(reloaded, params.replace(num_players=1))
    rng = jax.random.PRNGKey(0)
    for _ in range(steps):
        rng, key = jax.random.split(rng)
        _, state, _, _, _ = env.step_env(key, state, 0, params)

    print(f"after {steps} steps on '{reloaded.name}': timestep={int(state.timestep)}")


if __name__ == "__main__":
    main()
