"""Tests for the bootstrap-feedback primitives.

The advanced factory agent does *not* pre-mine every material it
needs. Instead, once a tier of automation is online, the agent walks
to its bus pallets and hand-crafts the next-tier machinery from the
materials the factory has already produced. This module covers the
two new goals that make that loop work:

- :class:`baselines.rocket.scripted.goals.WithdrawFromBusAt` — drain
  a specific known pallet tile.
- :class:`baselines.rocket.scripted.goals.CraftFromBus` — orchestrate
  one withdraw per recipe input, then hand-craft.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from baselines.rocket.scripted import goals, skills
from baselines.rocket.scripted.world_model import decode_observation
from factoriax.engine.constants import (
    Direction,
    ItemType,
    Machine,
)
from factoriax.engine.envs import FactoriaXEnv
from factoriax.engine.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.engine.levels import LevelBuilder, build_state
from factoriax.engine.observations import global_array
from factoriax.engine.state import EnvParams
from factoriax.scenarios.rocket import (
    ROCKET_BLOCKED_ACTIONS,
    rocket_conditions,
)

pytestmark = pytest.mark.slow

_MAP_SIZE = 16
_SPAWN = (8, 8)


_ASSEMBLER_TILE = (8, 6)

# Cache the JIT'd step function keyed by env-params shape. Without
# this, each test paid a fresh trace because every `_build_test_env`
# call constructed a new wrapper chain whose bound `step_env` is a
# new closure JAX can't dedupe. Sharing across tests drops the second
# JIT-compile run from ~4s to <100ms per test.
_JIT_STEP_CACHE: dict[tuple[int, int, int], object] = {}


def _shared_jit_step(env_params: EnvParams):
    """Memoized JIT-compiled ``step_env`` for the bootstrap test env."""
    key = (env_params.map_width, env_params.map_height, env_params.max_timesteps)
    fn = _JIT_STEP_CACHE.get(key)
    if fn is None:
        env = ActionMaskWrapper(
            FactoriaXEnv(achievement_fn=rocket_conditions),
            ROCKET_BLOCKED_ACTIONS,
        )
        fn = jax.jit(env.step_env)
        _JIT_STEP_CACHE[key] = fn
    return fn


def _build_test_env(
    pallet_seeds: dict[tuple[int, int], tuple[int, int]],
    starting_inventory: dict[int, int] | None = None,
    place_assembler: bool = True,
):
    """Construct a minimal env with hand-placed bus pallets.

    Args:
        pallet_seeds: Mapping ``(x, y) -> (item_type_id, count)`` for
            each pallet to place pre-loaded. Tiles must not collide
            with the player spawn.
        starting_inventory: Optional ``{item_id: count}`` to seed the
            player's pouch with (e.g. for testing CraftFromBus before
            its withdraw stage).
        place_assembler: When true, drop a single assembler at
            :data:`_ASSEMBLER_TILE` so :class:`CraftFromBus` (which
            routes recipes through a placed machine) has somewhere
            to push inputs. The rocket scenario masks every
            ``CRAFT_*`` action, so production must flow through a
            machine — hand-crafting isn't available.

    Returns:
        ``(jit_step_fn, EnvState, env_params)``.
    """
    builder = LevelBuilder(_MAP_SIZE, _MAP_SIZE)
    builder.set_player_position(*_SPAWN)
    if place_assembler:
        builder.place_machine(
            *_ASSEMBLER_TILE,
            int(Machine.ASSEMBLER),
            direction=int(Direction.DOWN),
        )
    for (x, y), (item_id, count) in pallet_seeds.items():
        if (x, y) == _SPAWN:
            raise ValueError(f"pallet seed at spawn {_SPAWN}")
        if place_assembler and (x, y) == _ASSEMBLER_TILE:
            raise ValueError(
                f"pallet seed collides with assembler at {_ASSEMBLER_TILE}",
            )
        builder.place_machine(
            x,
            y,
            int(Machine.PALLET),
            direction=int(Direction.DOWN),
        )
        builder.set_machine_inventory(x, y, int(item_id), int(count))

    level = builder.build("bootstrap_feedback_test")
    env_params = EnvParams(
        map_width=_MAP_SIZE,
        map_height=_MAP_SIZE,
        num_players=1,
        max_timesteps=400,
    )
    state = build_state(level, env_params)

    if starting_inventory:
        inv = np.asarray(state.player_inventory).copy()
        for item_id, count in starting_inventory.items():
            inv[0, int(item_id)] = count
        state = state.replace(player_inventory=jnp.asarray(inv))

    return _shared_jit_step(env_params), state, env_params


_JIT_OBS_CACHE: dict[tuple[int, int, int], object] = {}


def _jit_obs(env_params: EnvParams):
    key = (env_params.map_width, env_params.map_height, env_params.max_timesteps)
    fn = _JIT_OBS_CACHE.get(key)
    if fn is None:
        fn = jax.jit(lambda s: global_array(s, env_params, 0))
        _JIT_OBS_CACHE[key] = fn
    return fn


def _view(state, env_params: EnvParams):
    obs = np.asarray(_jit_obs(env_params)(state))
    return decode_observation(
        obs,
        map_height=env_params.map_height,
        map_width=env_params.map_width,
        max_timesteps=env_params.max_timesteps,
    )


def _rollout(state, goal, jit_step, env_params, max_steps: int = 200):
    """Drive the goal against the env until DONE/FAIL or step budget."""
    key = jax.random.PRNGKey(0)
    for _ in range(max_steps):
        view = _view(state, env_params)
        result, action = goal.step(view)
        if result is skills.Result.DONE:
            return state, "done"
        if result is skills.Result.FAIL:
            return state, "fail"
        assert action is not None
        key, sub = jax.random.split(key)
        _, state, _, _, _ = jit_step(sub, state, jnp.int32(int(action)), env_params)
    return state, "timeout"


# ---------------------------------------------------------------------------
# WithdrawFromBusAt
# ---------------------------------------------------------------------------


def test_withdraw_from_bus_at_drains_specific_pallet() -> None:
    """Goal walks to the named pallet tile and withdraws until count met."""
    pallet_tile = (10, 8)
    jit_step, state, env_params = _build_test_env(
        pallet_seeds={pallet_tile: (int(ItemType.IRON_PLATE), 20)},
    )
    goal = goals.WithdrawFromBusAt(pallet_tile, ItemType.IRON_PLATE, count=5)
    final_state, verdict = _rollout(state, goal, jit_step, env_params, max_steps=80)

    assert verdict == "done", f"got {verdict}"
    view = _view(final_state, env_params)
    assert view.player.held(ItemType.IRON_PLATE) >= 5


def test_withdraw_from_bus_at_fails_when_pallet_missing() -> None:
    """If the target tile has no pallet, the goal FAILs without retry."""
    jit_step, state, env_params = _build_test_env(pallet_seeds={})
    goal = goals.WithdrawFromBusAt((10, 8), ItemType.IRON_PLATE, count=1)
    _, verdict = _rollout(state, goal, jit_step, env_params, max_steps=20)

    assert verdict == "fail", f"got {verdict}"


def test_withdraw_from_bus_at_short_circuits_when_already_held() -> None:
    """Goal is DONE immediately when inventory already has enough."""
    pallet_tile = (10, 8)
    jit_step, state, env_params = _build_test_env(
        pallet_seeds={pallet_tile: (int(ItemType.WIRE), 10)},
        starting_inventory={int(ItemType.WIRE): 5},
    )
    goal = goals.WithdrawFromBusAt(pallet_tile, ItemType.WIRE, count=3)

    view = _view(state, env_params)
    result, action = goal.step(view)
    assert result is skills.Result.DONE
    assert action is None


# ---------------------------------------------------------------------------
# CraftFromBus
# ---------------------------------------------------------------------------


def test_craft_from_bus_makes_arms_from_two_bus_pallets() -> None:
    """ARM = COPPER_PLATE + WIRE. Crafts from two pre-loaded pallets."""
    copper_tile = (6, 8)
    wire_tile = (10, 8)
    jit_step, state, env_params = _build_test_env(
        pallet_seeds={
            copper_tile: (int(ItemType.COPPER_PLATE), 20),
            wire_tile: (int(ItemType.WIRE), 20),
        },
    )
    goal = goals.CraftFromBus(
        ItemType.ARM,
        count=2,
        bus_tiles={
            ItemType.COPPER_PLATE: copper_tile,
            ItemType.WIRE: wire_tile,
        },
    )
    final_state, verdict = _rollout(state, goal, jit_step, env_params, max_steps=200)

    assert verdict == "done", f"got {verdict}"
    view = _view(final_state, env_params)
    assert view.player.held(ItemType.ARM) >= 2


def test_craft_from_bus_rejects_missing_input_tile() -> None:
    """Constructor raises if a recipe input has no pallet mapping."""
    with pytest.raises(ValueError, match="missing bus tile"):
        goals.CraftFromBus(
            ItemType.ARM,
            count=1,
            bus_tiles={ItemType.COPPER_PLATE: (6, 8)},  # WIRE missing
        )


def test_craft_from_bus_rejects_unknown_recipe() -> None:
    """Constructor raises if no recipe produces the requested output."""
    with pytest.raises(ValueError, match="no recipe"):
        goals.CraftFromBus(
            ItemType.IRON_ORE,  # raw resource, never a recipe output
            count=1,
            bus_tiles={},
        )


# ---------------------------------------------------------------------------
# Bootstrap-feedback chain — Tier 1 plates -> Tier 2 WIRE -> ARM
# ---------------------------------------------------------------------------


def test_chained_craft_plates_to_wire_to_arm() -> None:
    """Two CraftFromBus calls in sequence cascade plates -> WIRE -> ARM.

    This is the bootstrap-feedback pattern in real use. The agent
    starts with raw plates on the bus (output of the Tier-1 smelter
    cells in a full factory) and crafts up two levels:

    1. ``CraftFromBus(WIRE, 2)`` — withdraws COPPER_PLATE + TIN_PLATE
       from their bus pallets and runs them through the assembler.
       The 2 WIRE land in the player's inventory.
    2. ``CraftFromBus(ARM, 2)`` — needs 2 COPPER_PLATE + 2 WIRE.
       The 2 WIRE from step 1 are still in the player's pouch, so
       :class:`WithdrawFromBusAt` short-circuits on the WIRE input.
       Only the COPPER_PLATE withdraw runs.

    The wire bus tile is just a placeholder — empty pallet, never
    drained — proving the chain does not depend on a populated
    intermediate bus to work. In a real factory the WIRE pallet
    would also be filled by a Tier-2 module's output arm; this
    test deliberately leaves it empty to isolate the
    inventory-shortcut path.
    """
    copper_tile = (6, 8)
    tin_tile = (10, 8)
    wire_tile = (8, 10)
    jit_step, state, env_params = _build_test_env(
        pallet_seeds={
            copper_tile: (int(ItemType.COPPER_PLATE), 20),
            tin_tile: (int(ItemType.TIN_PLATE), 20),
            wire_tile: (int(ItemType.WIRE), 0),
        },
    )

    wire_goal = goals.CraftFromBus(
        ItemType.WIRE,
        count=2,
        bus_tiles={
            ItemType.COPPER_PLATE: copper_tile,
            ItemType.TIN_PLATE: tin_tile,
        },
    )
    state, verdict = _rollout(state, wire_goal, jit_step, env_params, max_steps=200)
    assert verdict == "done", f"WIRE craft got {verdict}"
    view = _view(state, env_params)
    assert view.player.held(ItemType.WIRE) >= 2, (
        f"after WIRE craft, player has {view.player.held(ItemType.WIRE)} WIRE"
    )

    arm_goal = goals.CraftFromBus(
        ItemType.ARM,
        count=2,
        bus_tiles={
            ItemType.COPPER_PLATE: copper_tile,
            ItemType.WIRE: wire_tile,
        },
    )
    final_state, verdict = _rollout(
        state, arm_goal, jit_step, env_params, max_steps=200
    )
    assert verdict == "done", f"ARM craft got {verdict}"
    view = _view(final_state, env_params)
    assert view.player.held(ItemType.ARM) >= 2, (
        f"after ARM craft, player has {view.player.held(ItemType.ARM)} ARM"
    )
