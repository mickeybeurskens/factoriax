"""End-to-end tests for the v2 advanced-factory rocket agent (M5).

Two layers:

- Structural (sub-second): the goal list constructs cleanly, has the
  M5 shape (Phase 0 bootstrap + iron + copper smelter cells, each
  with ore feed + coal trunk + verify gates), and every placement
  lands inside the 32x32 map.
- Smoke (``@pytest.mark.slow``, full env rollout): the agent unlocks
  the M5 achievement floor and both plate-bus PALLETs accumulate
  their plate type.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from baselines.rocket.scripted.agent_advanced_factory import (
    _COPPER_PLATE_BUS_TILE,
    _IRON_PLATE_BUS_TILE,
    build_advanced_factory_goals,
    make_advanced_factory_rocket_agent,
)
from baselines.rocket.scripted.goals import (
    MineOre,
    PlaceMachineAt,
    PlaceMachineFromBackAt,
    ProduceInMachine,
    VerifyLayout,
    Wait,
    WaitUntil,
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
from factoriax.constants import MAX_ACHIEVEMENTS, Direction, ItemType, MachineType
from factoriax.envs import FactoriaXEnv
from factoriax.envs.achievement_wrapper import AchievementState, AchievementWrapper
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.levels import build_state
from factoriax.observations import global_array
from factoriax.state import EnvParams

_M5_EXPECTED_UNLOCKS: tuple[str, ...] = (
    # M3 floor — still hold.
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
    # M4 floor — iron cell automation.
    "craft_miner",
    "place_miner",
    "craft_belt",
    "place_belt",
    "craft_pallet",
    "place_pallet",
    "craft_arm",
    "place_arm",
    "craft_furnace",
    "automated_mining",
    "pallet_filled",
    "belt_network",
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


def test_m5_emits_iron_cell_placements() -> None:
    """Iron cell + ore feed + coal trunk land at the documented tiles."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    placements = [
        g for g in goals if isinstance(g, (PlaceMachineAt, PlaceMachineFromBackAt))
    ]
    by_tile = {g.target: (g.machine_type, g.facing) for g in placements}

    # Iron miner + 2 ore belts + ore_pallet on row 9.
    assert by_tile[(4, 9)][0] == int(MachineType.MINER)
    assert by_tile[(5, 9)][0] == int(MachineType.CONVEYOR_BELT)
    assert by_tile[(6, 9)][0] == int(MachineType.CONVEYOR_BELT)
    assert by_tile[(7, 9)][0] == int(MachineType.PALLET)
    # Smelter cell at (7, 10).
    assert by_tile[(7, 10)][0] == int(MachineType.FURNACE)
    assert by_tile[(8, 10)][0] == int(MachineType.ARM)
    assert by_tile[(9, 10)][0] == int(MachineType.PALLET)
    assert by_tile[(7, 11)][0] == int(MachineType.PALLET)  # coal_buffer
    # Coal trunk: 6 belts on row 11 + miner on the coal column at (0, 11).
    assert by_tile[(0, 11)][0] == int(MachineType.MINER)
    for x in range(1, 7):
        assert by_tile[(x, 11)][0] == int(MachineType.CONVEYOR_BELT)


def test_m5_emits_copper_cell_placements() -> None:
    """Copper cell + ore feed + coal trunk land at the documented tiles."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    placements = [
        g for g in goals if isinstance(g, (PlaceMachineAt, PlaceMachineFromBackAt))
    ]
    by_tile = {g.target: (g.machine_type, g.facing) for g in placements}

    # Copper miner + 2 ore belts + ore_pallet on row 12.
    assert by_tile[(4, 12)][0] == int(MachineType.MINER)
    assert by_tile[(5, 12)][0] == int(MachineType.CONVEYOR_BELT)
    assert by_tile[(6, 12)][0] == int(MachineType.CONVEYOR_BELT)
    assert by_tile[(7, 12)][0] == int(MachineType.PALLET)
    # Smelter cell at (7, 13).
    assert by_tile[(7, 13)][0] == int(MachineType.FURNACE)
    assert by_tile[(8, 13)][0] == int(MachineType.ARM)
    assert by_tile[(9, 13)][0] == int(MachineType.PALLET)
    assert by_tile[(7, 14)][0] == int(MachineType.PALLET)  # coal_buffer
    # Coal trunk: 6 belts on row 14 + miner on the coal column at (0, 14).
    assert by_tile[(0, 14)][0] == int(MachineType.MINER)
    for x in range(1, 7):
        assert by_tile[(x, 14)][0] == int(MachineType.CONVEYOR_BELT)


def test_m5_uses_place_from_back_for_coal_miners() -> None:
    """Each coal miner sits on the column at x=0; both go via from-back."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    miners_from_back = [
        g
        for g in goals
        if isinstance(g, PlaceMachineFromBackAt)
        and g.machine_type == int(MachineType.MINER)
    ]
    assert len(miners_from_back) == 2, (
        f"Expect one from-back miner per cell (iron+copper); got "
        f"{len(miners_from_back)}"
    )
    targets = {g.target for g in miners_from_back}
    assert targets == {(0, 11), (0, 14)}
    for g in miners_from_back:
        assert g.facing == int(Direction.RIGHT)


def test_m5_uses_place_from_back_for_ore_pallets() -> None:
    """Each cell's ore_pallet uses from-back so the previous cell's
    coal_buffer (which blocks the natural north stand tile for cells
    after iron) doesn't trip placement."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    pallets_from_back = [
        g
        for g in goals
        if isinstance(g, PlaceMachineFromBackAt)
        and g.machine_type == int(MachineType.PALLET)
    ]
    assert len(pallets_from_back) == 2, (
        f"Expect one from-back ore_pallet per cell (iron+copper); got "
        f"{len(pallets_from_back)}"
    )
    targets = {g.target for g in pallets_from_back}
    assert targets == {(7, 9), (7, 12)}
    for g in pallets_from_back:
        assert g.facing == int(Direction.DOWN)


def test_m5_all_placements_in_map_bounds() -> None:
    """Every Phase 1 placement target sits inside the 32x32 map."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    for g in goals:
        if isinstance(g, (PlaceMachineAt, PlaceMachineFromBackAt)):
            x, y = g.target
            assert 0 <= x < 32, f"{g.target} x out of bounds"
            assert 0 <= y < 32, f"{g.target} y out of bounds"


def test_m5_includes_per_cell_verify_layout_gates() -> None:
    """Each cell ends with its own VerifyLayout (one per cell, not one total)."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    verifies = [g for g in goals if isinstance(g, VerifyLayout)]
    assert len(verifies) == 2, (
        f"Expect one VerifyLayout per cell (iron, copper); got {len(verifies)}"
    )
    labels = {v.label for v in verifies}
    assert labels == {"phase 1.iron", "phase 1.copper"}


def test_m5_wait_precedes_verify() -> None:
    """The first WaitUntil precedes the first VerifyLayout."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    waits = [i for i, g in enumerate(goals) if isinstance(g, WaitUntil)]
    verifies = [i for i, g in enumerate(goals) if isinstance(g, VerifyLayout)]
    assert waits and verifies
    assert min(waits) < min(verifies), "WaitUntil should precede VerifyLayout"


def test_m5_mines_every_ore_type() -> None:
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
        f"M5 must mine all five ore leaves; missing {expected - set(mined.keys())}"
    )
    for item, qty in mined.items():
        assert qty > 0, f"MineOre({ItemType(item).name}) has zero qty"


def test_m5_smelts_each_plate_type() -> None:
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
        f"M5 must smelt every plate; missing {expected_plates - produced_outputs}"
    )


def test_m5_ends_with_a_settling_wait() -> None:
    """The final goal is a Wait so plates accumulate before episode end."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    assert isinstance(goals[-1], Wait)


def test_recipe_overlay_changes_bom_quantities() -> None:
    """Doubling WIRE's COPPER_PLATE input scales COPPER_ORE mining up."""
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
        f"Balance overlay should increase COPPER_ORE demand; "
        f"base={base_copper}, heavy={heavy_copper}"
    )


# ---------------------------------------------------------------------------
# Slow smoke — full env rollout under the rocket benchmark
# ---------------------------------------------------------------------------


def _run_agent(max_steps: int, seed: int = 0):
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
        _, state, _, done, _ = jit_step(subkey, state, jnp.int32(action), env_params)
        mask = np.asarray(state.achievements_unlocked)[:NUM_ROCKET_ACHIEVEMENTS]
        newly = (unlock_timestep < 0) & mask
        unlock_timestep[newly] = t
        if agent.is_done or bool(done):
            break

    final_mask = np.asarray(state.achievements_unlocked)[:NUM_ROCKET_ACHIEVEMENTS]
    return final_mask, unlock_timestep, state


def _index_of(achievement_id: str) -> int:
    for i, info in enumerate(ROCKET_ACHIEVEMENT_INFO):
        if info.id == achievement_id:
            return i
    raise AssertionError(f"Unknown achievement id: {achievement_id}")


def _assert_plate_bus_holds(state, tile: tuple[int, int], plate: ItemType) -> None:
    machine_types = np.asarray(state.env_state.machine_types)
    tile_entity = np.asarray(state.env_state.tile_entity)
    ent_buf_type = np.asarray(state.env_state.ent_buf_type)
    ent_buf_count = np.asarray(state.env_state.ent_buf_count)

    bx, by = tile
    assert int(machine_types[by, bx]) == int(MachineType.PALLET), (
        f"plate-bus tile {tile} should be PALLET; got "
        f"{MachineType(int(machine_types[by, bx])).name}"
    )
    ent_id = int(tile_entity[by, bx])
    assert ent_id >= 0, f"plate-bus entity not registered at {tile}"
    assert int(ent_buf_count[ent_id]) >= 1, (
        f"plate-bus PALLET at {tile} is empty; cell pipeline broken"
    )
    assert int(ent_buf_type[ent_id]) == int(plate), (
        f"plate-bus PALLET at {tile} holds "
        f"{ItemType(int(ent_buf_type[ent_id])).name}, expected {plate.name}"
    )


@pytest.mark.slow
def test_m5_unlocks_expected_floor() -> None:
    """The M5 agent unlocks the M3+M4 floor (iron + copper cells)."""
    mask, timing, _ = _run_agent(max_steps=8000)
    missing = [name for name in _M5_EXPECTED_UNLOCKS if not bool(mask[_index_of(name)])]
    timing_str = ", ".join(
        f"{ROCKET_ACHIEVEMENT_INFO[i].id}@{timing[i]}"
        for i in range(NUM_ROCKET_ACHIEVEMENTS)
        if timing[i] >= 0
    )
    assert not missing, f"M5 floor missing {missing}.\nUnlocked: {timing_str}"


@pytest.mark.slow
def test_m5_iron_plate_bus_accumulates_plates() -> None:
    """The iron smelter cell's plate-bus PALLET holds IRON_PLATE."""
    _, _, state = _run_agent(max_steps=8000)
    _assert_plate_bus_holds(state, _IRON_PLATE_BUS_TILE, ItemType.IRON_PLATE)


@pytest.mark.slow
def test_m5_copper_plate_bus_accumulates_plates() -> None:
    """The copper smelter cell's plate-bus PALLET holds COPPER_PLATE."""
    _, _, state = _run_agent(max_steps=8000)
    _assert_plate_bus_holds(state, _COPPER_PLATE_BUS_TILE, ItemType.COPPER_PLATE)
