"""End-to-end tests for the iteration-1 advanced-factory rocket agent.

Two tests:

- ``test_advanced_factory_goal_list_structural`` runs in milliseconds
  and just checks that ``build_advanced_factory_goals`` constructs
  cleanly with placement coordinates inside the 32x32 map.
- ``test_advanced_factory_iter1_iron_cell_produces_plates`` is the
  load-bearing smoke test: runs the agent against the rocket benchmark
  env (with the hand-craft action mask) and asserts that (a) a PALLET
  ends up at (10, 11) holding IRON_PLATE, proving the entire
  ore->furnace->arm->plate-pallet chain works, and (b) the 24 expected
  achievements all fire.
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
_PLATE_PALLET_TILE = (10, 11)

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
    """Goal list builds without errors and every placement is in-map."""
    goals = build_advanced_factory_goals()
    assert len(goals) > 0
    for g in goals:
        if isinstance(g, PlaceMachineAt):
            x, y = g.target
            assert 0 <= x < _MAP_SIZE, f"x out of range for {g.name} target={g.target}"
            assert 0 <= y < _MAP_SIZE, f"y out of range for {g.name} target={g.target}"


@pytest.mark.slow
def test_advanced_factory_iter1_iron_cell_produces_plates() -> None:
    """Run the agent and verify the iron cell ends up making plates.

    The load-bearing assertion: at episode end, the entity at
    ``(10, 11)`` is a PALLET whose buffer holds at least one
    IRON_PLATE. That tile is reachable only if (a) the coal trunk
    delivered coal to the buffer pallet at (8, 12), (b) the furnace
    at (8, 11) auto-pulled both iron ore (from (8, 10)) and coal
    (from (8, 12)) and ran the IRON_PLATE recipe, and (c) the arm at
    (9, 11) extracted the plate east into the plate pallet. Any link
    breakage anywhere in the chain leaves the plate pallet empty.
    """
    max_steps = 4000
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
    px, py = _PLATE_PALLET_TILE
    machine_types = np.asarray(env_state.machine_types)
    tile_entity = np.asarray(env_state.tile_entity)
    ent_buf_type = np.asarray(env_state.ent_buf_type)
    ent_buf_count = np.asarray(env_state.ent_buf_count)

    assert int(machine_types[py, px]) == int(MachineType.PALLET), (
        f"plate pallet missing at {_PLATE_PALLET_TILE}; "
        f"got machine_type={int(machine_types[py, px])}"
    )
    ent_idx = int(tile_entity[py, px])
    assert ent_idx >= 0, "tile_entity has no entity at plate pallet"
    assert int(ent_buf_count[ent_idx]) >= 1, (
        "plate pallet at (10, 11) is empty — iron cell did not produce. "
        "Check coal flow to (8, 12), furnace at (8, 11), and arm at (9, 11)."
    )
    assert int(ent_buf_type[ent_idx]) == int(ItemType.IRON_PLATE), (
        f"plate pallet holds wrong item: type={int(ent_buf_type[ent_idx])}"
    )

    mask = np.asarray(last_state.achievements_unlocked)[:NUM_ROCKET_ACHIEVEMENTS]
    ids = {info.id: i for i, info in enumerate(ROCKET_ACHIEVEMENT_INFO)}
    missing = [name for name in _EXPECTED_UNLOCKS if not mask[ids[name]]]
    assert not missing, f"Missing expected achievements: {missing}"
