"""End-to-end tests for the splitter-cell advanced-factory rocket agent.

Two tests:

- ``test_advanced_factory_goal_list_structural`` runs in milliseconds
  and just checks that ``build_advanced_factory_goals`` constructs
  cleanly with placement coordinates inside the 32x32 map AND emits
  the expected splitter / crossing counts (2 SPLITTERs — one each
  for iron and copper — and 1 CROSSING, the auto-inserted (21, 12)
  intersection between the copper coal trunk and the copper
  splitter's south output).
- ``test_advanced_factory_iron_cell_produces_plates`` is the load-
  bearing smoke test: runs the agent against the rocket benchmark
  env and asserts a PALLET ends up at (10, 10) (the iron cell's
  *manual stash*, north of the splitter) holding IRON_PLATE — the
  entire ore->furnace->arm->splitter->manual_stash chain.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from baselines.rocket.scripted.agent_advanced_factory import (
    build_advanced_factory_goals,
    make_advanced_factory_rocket_agent,
)
from baselines.rocket.scripted.goals import PlaceMachineAt
from factoriax.benchmarks.rocket import (
    NUM_ROCKET_ACHIEVEMENTS,
    ROCKET_ACHIEVEMENT_INFO,
    ROCKET_BLOCKED_ACTIONS,
    build_rocket_level,
    rocket_conditions,
)
from factoriax.constants import MAX_ACHIEVEMENTS, ItemType, MachineType
from factoriax.envs import FactoriaXEnv
from factoriax.envs.achievement_wrapper import AchievementState, AchievementWrapper
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.levels import build_state
from factoriax.observations import global_array
from factoriax.state import EnvParams

_MAP_SIZE = 32
_IRON_MANUAL_STASH_TILE = (10, 10)
_IRON_SPLITTER_TILE = (10, 11)

_EXPECTED_UNLOCKS: tuple[str, ...] = (
    "collect_iron",
    "collect_copper",
    "collect_tin",
    "collect_coal",
    "smelt_iron",
    "smelt_copper",
    "smelt_tin",
    "craft_wire",
    "craft_miner",
    "place_miner",
    "craft_furnace",
    "place_furnace",
    "automated_mining",
    "craft_belt",
    "place_belt",
    "craft_pallet",
    "place_pallet",
    "pallet_filled",
    "craft_arm",
    "place_arm",
    "first_assembly",
    "belt_network",
    "scaling_up",
    "industrialist",
)


def test_advanced_factory_goal_list_structural() -> None:
    """Goal list builds without errors and every placement is in-map.

    Also confirms the splitter-cell design: 4 SPLITTERs (one each
    for iron, copper, tin, silicon) and 0 CROSSINGs — the copper
    coal trunk is rerouted through row 13 so the splitter S output
    drops onto a sink PALLET at (21, 12) without any path crossing
    that tile.
    """
    goals = build_advanced_factory_goals()
    assert len(goals) > 0
    splitter_count = 0
    crossing_count = 0
    crossing_tiles: list[tuple[int, int]] = []
    for g in goals:
        if isinstance(g, PlaceMachineAt):
            x, y = g.target
            assert 0 <= x < _MAP_SIZE, f"x out of range for {g.name} target={g.target}"
            assert 0 <= y < _MAP_SIZE, f"y out of range for {g.name} target={g.target}"
            if g.machine_type == int(MachineType.SPLITTER):
                splitter_count += 1
            elif g.machine_type == int(MachineType.CROSSING):
                crossing_count += 1
                crossing_tiles.append(g.target)
    assert splitter_count == 4, (
        f"expected 1 SPLITTER per cell (iron + copper + tin + silicon = 4), "
        f"got {splitter_count}"
    )
    assert crossing_count == 0, (
        f"expected 0 CROSSINGs (copper coal trunk reroutes through row "
        f"13 so no path-collision occurs), got {crossing_count} at "
        f"{crossing_tiles}"
    )


@pytest.mark.slow
def test_advanced_factory_iron_cell_produces_plates() -> None:
    """Run the agent and verify the iron cell delivers plates to the
    manual stash through the new splitter design.

    The load-bearing assertion: at episode end, the entity at
    ``(10, 10)`` (the iron cell's *manual stash*, north of its
    splitter) is a PALLET whose buffer holds at least one
    IRON_PLATE. That tile is reachable only if (a) the coal trunk
    delivered coal to the buffer pallet at (8, 12) (via the rerouted
    network laid by ``_phase_b_belt_network``), (b) the furnace at
    (8, 11) auto-pulled both iron ore (from (8, 10)) and coal (from
    (8, 12)) and ran the IRON_PLATE recipe, (c) the arm at (9, 11)
    extracted the plate east into the splitter at (10, 11), and
    (d) the splitter fired its north output (manual stash) — which
    requires both the manual stash and the automation belt's
    downstream sink at (10, 13) to be receptive.
    """
    max_steps = 8000
    env_params = EnvParams(
        map_width=_MAP_SIZE,
        map_height=_MAP_SIZE,
        num_players=1,
        max_timesteps=max_steps,
    )
    level = build_rocket_level()
    env_state = build_state(level, env_params)
    state = AchievementState(
        env_state=env_state,
        achievements_unlocked=jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_),
    )
    env = ActionMaskWrapper(
        AchievementWrapper(FactoriaXEnv(), rocket_conditions),
        ROCKET_BLOCKED_ACTIONS,
    )
    jit_step = jax.jit(env.step_env)
    jit_obs = jax.jit(lambda s: global_array(s, env_params, 0))
    agent = make_advanced_factory_rocket_agent(env_params)

    key = jax.random.PRNGKey(0)
    last_state = state
    for _ in range(max_steps):
        obs = np.asarray(jit_obs(last_state.env_state))
        action = agent.act(obs)
        key, subkey = jax.random.split(key)
        _, last_state, _, done, _ = jit_step(
            subkey,
            last_state,
            jnp.int32(action),
            env_params,
        )
        if agent.is_done or bool(done):
            break

    env_state = last_state.env_state
    machine_types = np.asarray(env_state.machine_types)
    tile_entity = np.asarray(env_state.tile_entity)
    ent_buf_type = np.asarray(env_state.ent_buf_type)
    ent_buf_count = np.asarray(env_state.ent_buf_count)

    sx, sy = _IRON_SPLITTER_TILE
    assert int(machine_types[sy, sx]) == int(MachineType.SPLITTER), (
        f"iron splitter missing at {_IRON_SPLITTER_TILE}; "
        f"got machine_type={int(machine_types[sy, sx])}"
    )

    mx, my = _IRON_MANUAL_STASH_TILE
    assert int(machine_types[my, mx]) == int(MachineType.PALLET), (
        f"iron manual stash missing at {_IRON_MANUAL_STASH_TILE}; "
        f"got machine_type={int(machine_types[my, mx])}"
    )
    ent_idx = int(tile_entity[my, mx])
    assert ent_idx >= 0, "tile_entity has no entity at manual stash"
    assert int(ent_buf_count[ent_idx]) >= 1, (
        "iron manual stash at (10, 10) is empty — the splitter did not "
        "fire its north output. Check coal flow to (8, 12), furnace at "
        "(8, 11), arm at (9, 11), splitter at (10, 11), and the sink at "
        "(10, 13) (splitter requires both outputs receptive to fire)."
    )
    assert int(ent_buf_type[ent_idx]) == int(ItemType.IRON_PLATE), (
        f"manual stash holds wrong item: type={int(ent_buf_type[ent_idx])}"
    )

    mask = np.asarray(last_state.achievements_unlocked)[:NUM_ROCKET_ACHIEVEMENTS]
    ids = {info.id: i for i, info in enumerate(ROCKET_ACHIEVEMENT_INFO)}
    missing = [name for name in _EXPECTED_UNLOCKS if not mask[ids[name]]]
    assert not missing, f"Missing expected achievements: {missing}"
