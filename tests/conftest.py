"""Shared test fixtures and utilities.

============================================================================
Standard env fixtures across the test suite — pick one before you make your own.
============================================================================

JIT cache thrash is the single biggest driver of test wall time on this
project. Every fresh ``FactoriaXEnv(...)`` + ``jax.jit(env.step_env)``
combination triggers a ~7s XLA compile. The catalogue below lists the
fixtures that already exist; consume them in preference to building your
own env. If you must build your own, add a comment naming the property
you assert that prevents you from using a standard.

Catalogue (env, wrapper, shape -> fixture name @ file):

- 8x8 1p plain ``FactoriaXEnv()``
    -> ``canonical_env_8x8_1p`` @ this file
       returns ``(env, params, jit_step_fn, state)``
- 8x8 1p with ``ScienceTallyWrapper``
    -> ``tally_env`` @ ``tests/test_science_tally_wrapper.py``
- 10x10 1p inside ``ScenarioRunner``
    -> ``runner`` @ ``tests/scenarios/conftest.py``
       (multi-entry cache; new ``blocked_actions`` configs compile once)
- 5x5 1p per skill level
    -> ``run_scripted(level_idx, policy)`` @
       ``tests/scenarios/skills/test_skills_scripted_solves.py``
- 8x8 1p with custom ``achievement_fn``
    -> ``make_env(achievement_fn)`` @ ``tests/test_achievement_engine.py``

Rule of thumb when adding a new test:

1. If the test asserts something shape-independent (observation
   structure, machine logic, achievement latching, etc.), consume
   ``canonical_env_8x8_1p`` and call it done.
2. If the test wraps the env (custom achievement_fn, action mask,
   etc.) and the wrapper already has a fixture above, use it.
3. If the test asserts a *specific* shape's behaviour (obs space
   dimensions at 32x32, level builder validation, etc.), build your
   own env and add a one-line comment naming the assertion.
4. If the test introduces a new wrapper used by more than one
   assertion, add a module-scoped fixture for it next to the test —
   then add a row to this catalogue.

Background: ``SPEC_TEST_SUITE.md`` walks through Phase 2 (Tasks
2.1-2.11) which collapsed ~60% of wall time by hoisting these
fixtures from per-test construction to shared scope.
"""

# Force headless rendering for every test in the suite. Setting these
# BEFORE pygame is imported anywhere is what keeps test runs from
# popping a window on the developer's desktop. Any conftest that later
# calls pygame.display.init() will get the dummy driver, which is
# side-effect-free but still supports Surface.blit and font rendering.
import os
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Pin JAX to the CPU backend for unit tests. The suite is dominated by
# small ops (single-tile states, MAX_ACHIEVEMENTS-sized masks) where
# CUDA autotuning costs far more than the kernels themselves; on a GPU
# host the full suite is ~110s, on CPU it is ~70s. Benchmarks or
# scripts that legitimately need GPU can override this by exporting
# ``JAX_PLATFORMS`` before invoking pytest.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pygame  # noqa: E402
import pytest  # noqa: E402
from jax import random  # noqa: E402

from factoriax import EnvState  # noqa: E402
from factoriax.constants import (  # noqa: E402
    BLOCK_RESOURCE_DTYPE,
    MAX_ACHIEVEMENTS,
    NUM_ITEM_TYPES,
    NUM_SCIENCE_PACK_TYPES,
    Direction,
    Machine,
)
from factoriax.envs.factoriax_env import FactoriaXEnv  # noqa: E402
from factoriax.state import EnvParams  # noqa: E402

# Default entity capacity used by the test factory.
_TEST_MAX_MACHINES: int = 64


@pytest.fixture(scope="session", autouse=True)
def pygame_session() -> None:
    """Initialise pygame display + font once per test session.

    SDL drivers are pinned to ``dummy`` above so this is side-effect
    free. The display surface ``set_mode((800, 600))`` is what
    :class:`factoriax.ui.scaling.ScaledCanvas` reads via
    ``pygame.display.get_surface()``; tests in ``tests/test_scaling.py``
    used to do this in their own session fixture, which is now
    redundant.
    """
    pygame.display.init()
    pygame.display.set_mode((800, 600))
    pygame.font.init()


@pytest.fixture(scope="session")
def canonical_env_8x8_1p() -> tuple[FactoriaXEnv, EnvParams, Any, Any]:
    """Session-scoped 8x8 single-player env + JITted step + reset state.

    The load-bearing fixture for the Phase 2 rollout in
    ``SPEC_TEST_SUITE.md``. Tests that currently build their own
    :class:`FactoriaXEnv` + ``jax.jit(env.step_env)`` for the canonical
    shape switch to consuming this fixture, so the XLA compile of
    ``env.step_env`` happens exactly once per session instead of once
    per test.

    Returns:
        Tuple ``(env, params, jit_step_fn, initial_state)``. The state
        is the post-reset state at ``timestep == 0`` from
        ``random.PRNGKey(0)``; tests step *from* it without mutating
        it (JAX pytrees are immutable by construction).

    Consumers must NOT replace ``initial_state`` in-place — pass a
    different state forward locally if a test needs to step further.
    """
    env = FactoriaXEnv()
    params = EnvParams(map_width=8, map_height=8, num_players=1)
    _, initial_state = env.reset_env(random.PRNGKey(0), params)
    jit_step_fn = jax.jit(env.step_env)
    return env, params, jit_step_fn, initial_state


@pytest.fixture
def state_factory():
    """Factory for creating test states with sensible defaults.

    Returns a function that creates :class:`EnvState` objects. Only
    ``world_map`` is required; all other fields have sensible defaults.

    Machine state can be expressed in **grid form** for readability —
    pass ``machine_types``, ``machine_direction``, ``machine_power``,
    ``buffer_type``, ``buffer_count``, ``asm_in_type``, ``asm_in_count``,
    ``asm_out_type``, ``asm_out_count`` as ``(H, W)``-shaped arrays and
    the factory packs them into the engine's ``ent_*`` entity arrays
    plus a ``tile_entity`` lookup. This translation is a deliberate
    test ergonomic, not a compatibility shim: the engine itself reads
    only the entity arrays; tests just use the grid form so a single
    setup line can place a machine at ``(y, x)`` with a given facing
    and buffer state.

    Example:
        def test_something(state_factory):
            state = state_factory(
                world_map=jnp.array([[BlockType.COAL]]),
                block_resources=jnp.array([[50]]),
            )
    """

    def _create(
        world_map: jnp.ndarray,
        player_position: tuple[int, int] | None = None,
        player_positions: jnp.ndarray | None = None,
        player_direction: int | None = None,
        player_directions: jnp.ndarray | None = None,
        timestep: int = 0,
        player_inventory: jnp.ndarray | None = None,
        selected_player: int = 0,
        num_players: int = 1,
        block_resources: jnp.ndarray | None = None,
        machine_types: jnp.ndarray | None = None,
        machine_power: jnp.ndarray | None = None,
        machine_direction: jnp.ndarray | None = None,
        buffer_type: jnp.ndarray | None = None,
        buffer_count: jnp.ndarray | None = None,
        asm_in_type: jnp.ndarray | None = None,
        asm_in_count: jnp.ndarray | None = None,
        asm_out_type: jnp.ndarray | None = None,
        asm_out_count: jnp.ndarray | None = None,
        items_mined: jnp.ndarray | None = None,
        science_consumed_step: jnp.ndarray | None = None,
        max_machines: int = _TEST_MAX_MACHINES,
        # Catch-all for kwargs that older tests pass under retired
        # EnvState field names. Silently dropped; new tests should
        # not rely on this.
        **_kwargs: object,
    ) -> EnvState:
        """Create a test state with defaults for unspecified fields.

        Args:
            world_map: Block types array (required).
            player_position: Single player (x, y) position.
            player_positions: All player positions.
            player_direction: Single player direction.
            player_directions: All player directions.
            timestep: Current timestep.
            player_inventory: Item counts per player.
            selected_player: Currently selected player index.
            num_players: Number of players (for defaults).
            block_resources: Resources per tile.
            machine_types: Machine type per tile.
            machine_power: Power per machine (grid form; packed into
                ``ent_power``).
            machine_direction: Direction per machine (grid form; packed
                into ``ent_direction``).
            buffer_type: Buffer item type per tile (grid form; packed
                into ``ent_buf_type``).
            buffer_count: Buffer item count per tile (grid form; packed
                into ``ent_buf_count``).
            asm_in_type: Assembler input types (grid form; packed into
                ``ent_asm_in_type``).
            asm_in_count: Assembler input counts (grid form; packed
                into ``ent_asm_in_count``).
            asm_out_type: Assembler output type (grid form; packed into
                ``ent_asm_out_type``).
            asm_out_count: Assembler output count (grid form; packed
                into ``ent_asm_out_count``).
            items_mined: Lifetime mined counts.
            science_consumed_step: Per-step science pack consumption
                delta (from SCIENCE_LAB entities).
            max_machines: Entity array capacity.

        Returns:
            Configured EnvState for testing.
        """
        shape = world_map.shape
        mm = max_machines

        if player_positions is not None:
            positions = player_positions
            num_players = positions.shape[0]
        elif player_position is not None:
            if isinstance(player_position, tuple):
                positions = jnp.array(
                    [player_position],
                    dtype=jnp.int16,
                )
            else:
                positions = player_position.reshape(1, 2).astype(jnp.int16)
            num_players = 1
        else:
            positions = jnp.array([[0, 0]], dtype=jnp.int16)
            num_players = 1

        if player_directions is not None:
            directions = player_directions.astype(jnp.int8)
        elif player_direction is not None:
            directions = jnp.array(
                [player_direction],
                dtype=jnp.int8,
            )
        else:
            directions = jnp.full(
                num_players,
                Direction.DOWN,
                dtype=jnp.int8,
            )

        inv_shape = (num_players, NUM_ITEM_TYPES)

        mt_grid = (
            machine_types.astype(jnp.int8)
            if machine_types is not None
            else jnp.full(shape, Machine.NONE, dtype=jnp.int8)
        )

        # Build entity arrays from the grid-based arguments.
        mt_np = np.asarray(mt_grid)
        md_np = (
            np.asarray(machine_direction)
            if machine_direction is not None
            else np.zeros(shape, dtype=np.int8)
        )
        mp_np = (
            np.asarray(machine_power)
            if machine_power is not None
            else np.zeros(shape, dtype=np.int16)
        )
        bt_np = (
            np.asarray(buffer_type)
            if buffer_type is not None
            else np.zeros(shape, dtype=np.int8)
        )
        bc_np = (
            np.asarray(buffer_count)
            if buffer_count is not None
            else np.zeros(shape, dtype=np.int16)
        )
        ait_np = (
            np.asarray(asm_in_type)
            if asm_in_type is not None
            else np.zeros((*shape, 2), dtype=np.int8)
        )
        aic_np = (
            np.asarray(asm_in_count)
            if asm_in_count is not None
            else np.zeros((*shape, 2), dtype=np.int16)
        )
        aot_np = (
            np.asarray(asm_out_type)
            if asm_out_type is not None
            else np.zeros(shape, dtype=np.int8)
        )
        aoc_np = (
            np.asarray(asm_out_count)
            if asm_out_count is not None
            else np.zeros(shape, dtype=np.int16)
        )

        # Allocate entity arrays.
        ent_y = np.full(mm, -1, dtype=np.int16)
        ent_x = np.full(mm, -1, dtype=np.int16)
        ent_type = np.zeros(mm, dtype=np.int8)
        ent_dir = np.zeros(mm, dtype=np.int8)
        ent_power = np.zeros(mm, dtype=np.int16)
        ent_buf_type = np.zeros(mm, dtype=np.int8)
        ent_buf_count = np.zeros(mm, dtype=np.int16)
        ent_asm_in_type = np.zeros((mm, 2), dtype=np.int8)
        ent_asm_in_count = np.zeros((mm, 2), dtype=np.int16)
        ent_asm_out_type = np.zeros(mm, dtype=np.int8)
        ent_asm_out_count = np.zeros(mm, dtype=np.int16)
        ent_health = np.zeros(mm, dtype=np.int16)
        tile_ent = np.full(shape, -1, dtype=np.int16)

        idx = 0
        for y in range(shape[0]):
            for x in range(shape[1]):
                if int(mt_np[y, x]) != int(Machine.NONE) and idx < mm:
                    ent_y[idx] = y
                    ent_x[idx] = x
                    ent_type[idx] = mt_np[y, x]
                    ent_dir[idx] = md_np[y, x]
                    ent_power[idx] = mp_np[y, x]
                    ent_buf_type[idx] = bt_np[y, x]
                    ent_buf_count[idx] = bc_np[y, x]
                    ent_asm_in_type[idx] = ait_np[y, x]
                    ent_asm_in_count[idx] = aic_np[y, x]
                    ent_asm_out_type[idx] = aot_np[y, x]
                    ent_asm_out_count[idx] = aoc_np[y, x]
                    tile_ent[y, x] = idx
                    idx += 1

        return EnvState(
            map=world_map.astype(jnp.int8),
            block_resources=(
                block_resources
                if block_resources is not None
                else jnp.zeros(shape, dtype=BLOCK_RESOURCE_DTYPE)
            ),
            machine_types=mt_grid,
            tile_entity=jnp.array(tile_ent, dtype=jnp.int16),
            ent_y=jnp.array(ent_y, dtype=jnp.int16),
            ent_x=jnp.array(ent_x, dtype=jnp.int16),
            ent_type=jnp.array(ent_type, dtype=jnp.int8),
            ent_direction=jnp.array(ent_dir, dtype=jnp.int8),
            ent_power=jnp.array(ent_power, dtype=jnp.int16),
            ent_buf_type=jnp.array(ent_buf_type, dtype=jnp.int8),
            ent_buf_count=jnp.array(ent_buf_count, dtype=jnp.int16),
            ent_asm_in_type=jnp.array(ent_asm_in_type, dtype=jnp.int8),
            ent_asm_in_count=jnp.array(ent_asm_in_count, dtype=jnp.int16),
            ent_asm_out_type=jnp.array(ent_asm_out_type, dtype=jnp.int8),
            ent_asm_out_count=jnp.array(ent_asm_out_count, dtype=jnp.int16),
            ent_health=jnp.array(ent_health, dtype=jnp.int16),
            player_positions=positions.astype(jnp.int16),
            player_directions=directions,
            player_inventory=(
                player_inventory.astype(jnp.int16)
                if player_inventory is not None
                else jnp.zeros(inv_shape, dtype=jnp.int16)
            ),
            # Match env.reset_env's pytree shape: factoriax/levels.py:686-687
            # emits these as jnp.int32(0); a Python-int leaf here would
            # force jax.jit(env.step_env) to retrace whenever a test feeds
            # a state_factory state through the canonical_env_8x8_1p
            # fixture's shared step path.
            selected_player=jnp.int32(selected_player),
            timestep=jnp.int32(timestep),
            items_mined=(
                items_mined
                if items_mined is not None
                else jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
            ),
            science_consumed_step=(
                science_consumed_step
                if science_consumed_step is not None
                else jnp.zeros(NUM_SCIENCE_PACK_TYPES, dtype=jnp.int32)
            ),
            achievements_unlocked=jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_),
        )

    return _create
