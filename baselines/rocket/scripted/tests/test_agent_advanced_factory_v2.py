"""End-to-end tests for the v2 advanced-factory rocket agent.

Two layers:

- Structural (sub-second): the goal list constructs cleanly, has the
  shape M3 promises, and contains no out-of-map placements.
- Smoke (``@pytest.mark.slow``, full env rollout): the agent unlocks
  the M3 achievement floor — every collect_*, every smelt_*,
  craft_wire, place_furnace, place_assembler, first_assembly. No
  automation is built yet (M4+), so achievements requiring placed
  miners / belts / pallets stay locked.
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
from baselines.rocket.scripted.goals import (
    MineOre,
    PlaceMachineAt,
    ProduceInMachine,
    Wait,
)
from factoriax.benchmarks.rocket import (
    NUM_ROCKET_ACHIEVEMENTS,
    ROCKET_ACHIEVEMENT_INFO,
    ROCKET_BLOCKED_ACTIONS,
    ROCKET_RECIPE_BOOK,
    ROCKET_RECIPE_TABLE,
    build_rocket_level,
    rocket_conditions,
)
from factoriax.constants import MAX_ACHIEVEMENTS, ItemType
from factoriax.envs import FactoriaXEnv
from factoriax.envs.achievement_wrapper import AchievementState, AchievementWrapper
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.levels import build_state
from factoriax.observations import global_array
from factoriax.state import EnvParams

# M3 doesn't place machinery, so unlock floors don't depend on
# automation. Once M4 lands these expand.
_M3_EXPECTED_UNLOCKS: tuple[str, ...] = (
    "collect_iron",
    "collect_copper",
    "collect_tin",
    "collect_coal",
    "collect_silicon",
    "smelt_iron",
    "smelt_copper",
    "smelt_tin",
    "smelt_wafer",
    "craft_wire",
    "place_furnace",
    "place_assembler",
    "first_assembly",
)


# ---------------------------------------------------------------------------
# Structural — millisecond
# ---------------------------------------------------------------------------


def test_goal_list_constructs_under_default_book() -> None:
    """``build_advanced_factory_goals()`` returns a non-empty list."""
    goals = build_advanced_factory_goals()
    assert len(goals) > 0


def test_goal_list_constructs_under_rocket_book() -> None:
    """The agent works against the rocket benchmark's recipe overlay."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    assert len(goals) > 0


def test_m3_emits_no_place_machine_goals() -> None:
    """M3 is bootstrap-only: zero PlaceMachineAt entries until M4."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    placements = [g for g in goals if isinstance(g, PlaceMachineAt)]
    assert placements == [], (
        f"M3 must not place any machines; saw {len(placements)} placements"
    )


def test_m3_mines_every_ore_type() -> None:
    """The starter targets resolve to a MineOre for each of the 5 ores."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    mined = {g.item_type: g.count for g in goals if isinstance(g, MineOre)}
    expected = {
        int(ItemType.IRON_ORE),
        int(ItemType.COPPER_ORE),
        int(ItemType.TIN_ORE),
        int(ItemType.SILICON),
        int(ItemType.COAL),
    }
    assert set(mined.keys()) >= expected, (
        f"M3 must mine all five ore leaves; missing {expected - set(mined.keys())}"
    )
    for item, qty in mined.items():
        assert qty > 0, f"MineOre({ItemType(item).name}) has zero qty"


def test_m3_smelts_each_plate_type() -> None:
    """The smelt phase covers iron, copper, tin, and wafer."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    produced_outputs: set[int] = set()
    for g in goals:
        if isinstance(g, ProduceInMachine):
            produced_outputs.add(g.output_item)
    expected_plates = {
        int(ItemType.IRON_PLATE),
        int(ItemType.COPPER_PLATE),
        int(ItemType.TIN_PLATE),
        int(ItemType.WAFER),
    }
    assert produced_outputs >= expected_plates, (
        f"M3 must smelt every plate; missing {expected_plates - produced_outputs}"
    )


def test_m3_ends_with_a_settling_wait() -> None:
    """The final goal is a ``Wait`` so achievement state stabilises."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    assert isinstance(goals[-1], Wait)


def test_recipe_overlay_changes_bom_quantities() -> None:
    """A recipe rebalance flows into the mine quantities (flexibility check).

    Doubling WIRE's COPPER_PLATE input via :class:`RecipeBalance`
    must roughly double the COPPER_ORE mining target — proves the
    BOM math still adapts to balance overlays after the rewrite.
    """
    from factoriax.recipes import BASE_RECIPE_BOOK, RecipeBalance, RecipeOverride

    base_goals = build_advanced_factory_goals(book=BASE_RECIPE_BOOK, slack={})
    base_copper = next(
        g.count
        for g in base_goals
        if isinstance(g, MineOre) and g.item_type == int(ItemType.COPPER_ORE)
    )

    heavy_book = BASE_RECIPE_BOOK.with_balance(
        RecipeBalance(
            overrides=((int(ItemType.WIRE), RecipeOverride(input_counts=(4, 1))),)
        )
    )
    heavy_goals = build_advanced_factory_goals(book=heavy_book, slack={})
    heavy_copper = next(
        g.count
        for g in heavy_goals
        if isinstance(g, MineOre) and g.item_type == int(ItemType.COPPER_ORE)
    )

    assert heavy_copper > base_copper, (
        f"Balance overlay should increase COPPER_ORE demand; base="
        f"{base_copper}, heavy={heavy_copper}"
    )


# ---------------------------------------------------------------------------
# Slow smoke — full env rollout under the rocket benchmark
# ---------------------------------------------------------------------------


def _run_agent(max_steps: int, seed: int = 0):
    """Drive the v2 advanced-factory agent through the masked benchmark."""
    env_params = EnvParams(
        map_width=32,
        map_height=32,
        num_players=1,
        max_timesteps=max_steps,
        recipe_table=ROCKET_RECIPE_TABLE,
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
    agent = make_advanced_factory_rocket_agent(env_params, book=ROCKET_RECIPE_BOOK)

    unlock_timestep = np.full((NUM_ROCKET_ACHIEVEMENTS,), -1, dtype=np.int32)
    key = jax.random.PRNGKey(seed)

    for t in range(max_steps):
        obs = np.asarray(jit_obs(state.env_state))
        action = agent.act(obs)
        key, subkey = jax.random.split(key)
        _, state, _, done, _ = jit_step(
            subkey,
            state,
            jnp.int32(action),
            env_params,
        )
        mask = np.asarray(state.achievements_unlocked)[:NUM_ROCKET_ACHIEVEMENTS]
        newly = (unlock_timestep < 0) & mask
        unlock_timestep[newly] = t
        if agent.is_done or bool(done):
            break

    final_mask = np.asarray(state.achievements_unlocked)[:NUM_ROCKET_ACHIEVEMENTS]
    return final_mask, unlock_timestep


def _index_of(achievement_id: str) -> int:
    for i, info in enumerate(ROCKET_ACHIEVEMENT_INFO):
        if info.id == achievement_id:
            return i
    raise AssertionError(f"Unknown achievement id: {achievement_id}")


@pytest.mark.slow
def test_m3_unlocks_expected_floor() -> None:
    """The M3 bootstrap-only agent unlocks at least the documented set.

    Achievement floor: every collect_* / smelt_* + craft_wire +
    first_assembly + place_furnace + place_assembler. ~13
    achievements; later milestones extend this floor without
    touching the prefix.
    """
    mask, timing = _run_agent(max_steps=2000)
    missing = [name for name in _M3_EXPECTED_UNLOCKS if not bool(mask[_index_of(name)])]
    timing_str = ", ".join(
        f"{ROCKET_ACHIEVEMENT_INFO[i].id}@{timing[i]}"
        for i in range(NUM_ROCKET_ACHIEVEMENTS)
        if timing[i] >= 0
    )
    assert not missing, f"M3 floor missing {missing}.\nUnlocked: {timing_str}"


@pytest.mark.slow
def test_m3_does_not_place_machines_during_run() -> None:
    """M3 is bootstrap-only; only the pre-placed F+A should exist at end.

    Sanity-checks that the goal list really did not place anything,
    so the smoke test keeps tracking what the planner actually does
    instead of what the structural test only inspects in the goal
    list.
    """
    env_params = EnvParams(
        map_width=32,
        map_height=32,
        num_players=1,
        max_timesteps=2000,
        recipe_table=ROCKET_RECIPE_TABLE,
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
    agent = make_advanced_factory_rocket_agent(env_params, book=ROCKET_RECIPE_BOOK)

    key = jax.random.PRNGKey(0)
    for _ in range(2000):
        obs = np.asarray(jit_obs(state.env_state))
        action = agent.act(obs)
        key, subkey = jax.random.split(key)
        _, state, _, done, _ = jit_step(subkey, state, jnp.int32(action), env_params)
        if agent.is_done or bool(done):
            break

    machine_types = np.asarray(state.env_state.machine_types)
    placed_count = int((machine_types != 0).sum())
    # Pre-placed: 1 furnace at (15, 16), 1 assembler at (17, 16).
    assert placed_count == 2, (
        f"M3 must not place machines; observed {placed_count} placed "
        f"(expected exactly the 2 pre-placed F+A)"
    )
