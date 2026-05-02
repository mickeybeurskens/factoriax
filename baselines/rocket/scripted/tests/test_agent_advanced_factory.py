"""End-to-end tests for the advanced-factory rocket agent.

Two layers:

- Structural (sub-second): the goal list constructs cleanly, has
  the expected shape (Phase 0 bootstrap + four smelter cells +
  WIRE assembler + tin SPLITTER + copper/tin -> WIRE plate routes
  + per-stage verify gates), and every placement lands inside the
  32x32 map.
- Smoke (``@pytest.mark.slow``, full env rollout): the agent
  unlocks the achievement floor in :data:`_EXPECTED_UNLOCKS`,
  every terminal plate-bus PALLET accumulates its plate type,
  and the WIRE output pallet at (15, 14) holds WIRE.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from baselines.rocket.scripted.agent_advanced_factory import (
    _COPPER_EXTRACT_ARM_TILE,
    _FRAME_ASSEMBLER_TILE,
    _FRAME_INPUT_A_TILE,
    _FRAME_INPUT_B_TILE,
    _FRAME_OUTPUT_TILE,
    _IRON_EXTRACT_ARM_TILE,
    _IRON_PLATE_BUS_TILE,
    _IRON_TO_FRAME_BELTS,
    _SILICON_PLATE_BUS_TILE,
    _TIN_EXTRACT_ARM_TILE,
    _TIN_FRAME_TRUNK_HEAD_TILE,
    _TIN_TO_FRAME_BELTS,
    _WIRE_ASSEMBLER_TILE,
    _WIRE_INPUT_A_TILE,
    _WIRE_INPUT_B_TILE,
    _WIRE_OUTPUT_TILE,
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

_EXPECTED_UNLOCKS: tuple[str, ...] = (
    # Bootstrap + ore collection.
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
    # Smelter-cell automation.
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
    # WIRE assembler placed by the agent (not just the pre-placed
    # one) and crafted via plates.
    "craft_assembler",
    # FRAME assembler with iron + tin extractors routing around the
    # pre-placed F+A drain zone.
    "craft_frame",
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


_CELL_GEOMETRY: tuple[
    tuple[str, tuple[int, int], tuple[int, int], tuple[int, int]], ...
] = (
    # (label, ore_miner, furnace, coal_miner)
    ("iron", (4, 9), (7, 10), (0, 11)),
    ("copper", (4, 12), (7, 13), (0, 14)),
    ("tin", (4, 15), (7, 16), (0, 17)),
    ("silicon", (4, 18), (7, 19), (0, 20)),
)


def test_emits_each_cell_placements() -> None:
    """Every cell's miner/belts/ore_pallet/smelter/coal_trunk land
    at the documented tiles."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    placements = [
        g for g in goals if isinstance(g, (PlaceMachineAt, PlaceMachineFromBackAt))
    ]
    by_tile = {g.target: (g.machine_type, g.facing) for g in placements}

    for label, (mx, my), (fx, fy), (cx, cy) in _CELL_GEOMETRY:
        assert by_tile[(mx, my)][0] == int(MachineType.MINER), label
        assert by_tile[(mx + 1, my)][0] == int(MachineType.CONVEYOR_BELT), label
        assert by_tile[(mx + 2, my)][0] == int(MachineType.CONVEYOR_BELT), label
        # Ore feeder is now a belt facing DOWN (combiners pull only
        # from facing belts under the directional Phase 0).
        assert by_tile[(fx, my)][0] == int(MachineType.CONVEYOR_BELT), label
        assert by_tile[(fx, fy)][0] == int(MachineType.FURNACE), label
        assert by_tile[(fx + 1, fy)][0] == int(MachineType.ARM), label
        assert by_tile[(fx + 2, fy)][0] == int(MachineType.PALLET), label
        # Coal feeder south of the furnace is also a belt (UP-facing).
        assert by_tile[(fx, fy + 1)][0] == int(MachineType.CONVEYOR_BELT), label
        assert by_tile[(cx, cy)][0] == int(MachineType.MINER), label
        for x in range(1, 7):
            assert by_tile[(x, cy)][0] == int(MachineType.CONVEYOR_BELT), label


def test_emits_wire_cell_placements() -> None:
    """The WIRE assembler module + extractor arms + route belts land
    at the documented tiles."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    placements = [
        g for g in goals if isinstance(g, (PlaceMachineAt, PlaceMachineFromBackAt))
    ]
    by_tile = {g.target: (g.machine_type, g.facing) for g in placements}

    # WIRE assembler module: inputs are feeder belts; output is a pallet.
    assert by_tile[_WIRE_ASSEMBLER_TILE][0] == int(MachineType.ASSEMBLER)
    assert by_tile[_WIRE_INPUT_A_TILE][0] == int(MachineType.CONVEYOR_BELT)
    assert by_tile[_WIRE_INPUT_B_TILE][0] == int(MachineType.CONVEYOR_BELT)
    assert by_tile[_WIRE_OUTPUT_TILE][0] == int(MachineType.PALLET)
    assert by_tile[(14, 14)][0] == int(MachineType.ARM)  # output arm

    # Copper extractor arm + 2 belt route to input_a.
    assert by_tile[_COPPER_EXTRACT_ARM_TILE][0] == int(MachineType.ARM)
    assert by_tile[(11, 13)][0] == int(MachineType.CONVEYOR_BELT)
    assert by_tile[(12, 13)][0] == int(MachineType.CONVEYOR_BELT)

    # Tin extractor arm + SPLITTER + 2 belt L-route to input_b.
    # The splitter at (11, 16) lets the same extractor feed both
    # the WIRE route (via its UP output) and a future FRAME route
    # (via its DOWN output). The UP output drops onto (11, 15)
    # RIGHT then bends to (12, 15) UP into the WIRE input_b.
    assert by_tile[_TIN_EXTRACT_ARM_TILE][0] == int(MachineType.ARM)
    assert by_tile[(11, 16)][0] == int(MachineType.SPLITTER)
    assert by_tile[(11, 15)][0] == int(MachineType.CONVEYOR_BELT)
    assert by_tile[(12, 15)][0] == int(MachineType.CONVEYOR_BELT)


def test_uses_place_from_back_for_coal_miners() -> None:
    """Each cell's coal miner sits on the column at x=0; all go via from-back."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    miners_from_back = [
        g
        for g in goals
        if isinstance(g, PlaceMachineFromBackAt)
        and g.machine_type == int(MachineType.MINER)
    ]
    expected_targets = {(0, 11), (0, 14), (0, 17), (0, 20)}
    assert {g.target for g in miners_from_back} == expected_targets
    for g in miners_from_back:
        assert g.facing == int(Direction.RIGHT)


def test_uses_place_from_back_for_extractor_arms() -> None:
    """Every plate-bus extractor arm (iron, copper, tin) lands east
    of its bus PALLET; the natural west stand tile is the PALLET
    itself, so all three go via from-back."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    arms_from_back = [
        g
        for g in goals
        if isinstance(g, PlaceMachineFromBackAt)
        and g.machine_type == int(MachineType.ARM)
    ]
    assert {g.target for g in arms_from_back} == {
        _IRON_EXTRACT_ARM_TILE,
        _COPPER_EXTRACT_ARM_TILE,
        _TIN_EXTRACT_ARM_TILE,
    }
    for g in arms_from_back:
        assert g.facing == int(Direction.RIGHT)


def test_emits_frame_cell_placements() -> None:
    """The FRAME assembler module + iron extractor + iron / tin
    routes land at the documented tiles."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    placements = [
        g for g in goals if isinstance(g, (PlaceMachineAt, PlaceMachineFromBackAt))
    ]
    by_tile = {g.target: (g.machine_type, g.facing) for g in placements}

    # FRAME assembler module: inputs are feeder belts; output is a pallet.
    cx, cy = _FRAME_ASSEMBLER_TILE
    assert by_tile[_FRAME_ASSEMBLER_TILE][0] == int(MachineType.ASSEMBLER)
    assert by_tile[_FRAME_INPUT_A_TILE][0] == int(MachineType.CONVEYOR_BELT)
    assert by_tile[_FRAME_INPUT_B_TILE][0] == int(MachineType.CONVEYOR_BELT)
    assert by_tile[_FRAME_OUTPUT_TILE][0] == int(MachineType.PALLET)
    assert by_tile[(cx + 1, cy)][0] == int(MachineType.ARM)  # output arm

    # Iron extractor + every iron route belt.
    assert by_tile[_IRON_EXTRACT_ARM_TILE][0] == int(MachineType.ARM)
    for tile, _facing in _IRON_TO_FRAME_BELTS:
        assert by_tile[tile][0] == int(MachineType.CONVEYOR_BELT), tile

    # Tin route belts + the from-back trunk-head.
    for tile, _facing in _TIN_TO_FRAME_BELTS:
        assert by_tile[tile][0] == int(MachineType.CONVEYOR_BELT), tile
    assert by_tile[_TIN_FRAME_TRUNK_HEAD_TILE][0] == int(MachineType.CONVEYOR_BELT)


def test_frame_routes_avoid_pre_placed_drain_zone() -> None:
    """No FRAME-feed tile sits on a 4-neighbour of the pre-placed
    FURNACE (15, 16) or ASSEMBLER (17, 16). Both combiners auto-pull
    from any adjacent buffer regardless of recipe match, so a belt
    on those tiles would have its plate siphoned into a stalled
    input slot."""
    drain_zone: set[tuple[int, int]] = set()
    for cx, cy in ((15, 16), (17, 16)):
        drain_zone |= {(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)}

    feed_tiles = {tile for tile, _ in _IRON_TO_FRAME_BELTS}
    feed_tiles |= {tile for tile, _ in _TIN_TO_FRAME_BELTS}
    feed_tiles.add(_TIN_FRAME_TRUNK_HEAD_TILE)
    feed_tiles.add(_FRAME_INPUT_A_TILE)
    feed_tiles.add(_FRAME_INPUT_B_TILE)

    overlap = feed_tiles & drain_zone
    assert not overlap, f"FRAME feed tiles in F+A drain zone: {sorted(overlap)}"


def test_uses_place_from_back_for_ore_feeder_belts() -> None:
    """Each cell's ore-feeder belt uses from-back so the previous
    cell's coal-feeder belt doesn't trip placement (the natural
    north stand tile would be that prior belt for cells 2-4, and
    although belts are walkable it's simpler to use from-back
    uniformly across all four cells)."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    expected_targets = {(7, 9), (7, 12), (7, 15), (7, 18)}
    feeders_from_back = [
        g
        for g in goals
        if isinstance(g, PlaceMachineFromBackAt)
        and g.machine_type == int(MachineType.CONVEYOR_BELT)
        and g.target in expected_targets
    ]
    assert {g.target for g in feeders_from_back} == expected_targets
    for g in feeders_from_back:
        assert g.facing == int(Direction.DOWN)


def test_all_placements_in_map_bounds() -> None:
    """Every placement target sits inside the 32x32 map."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    for g in goals:
        if isinstance(g, (PlaceMachineAt, PlaceMachineFromBackAt)):
            x, y = g.target
            assert 0 <= x < 32, f"{g.target} x out of bounds"
            assert 0 <= y < 32, f"{g.target} y out of bounds"


def test_includes_per_stage_verify_layout_gates() -> None:
    """Each smelter cell + the WIRE stage end with their own
    VerifyLayout (6 total: 4 smelter + 1 wire + 1 frame)."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    verifies = [g for g in goals if isinstance(g, VerifyLayout)]
    expected_labels = {
        "phase 1.iron",
        "phase 1.copper",
        "phase 1.tin",
        "phase 1.silicon",
        "phase 2.wire",
        "phase 3.frame",
    }
    assert {v.label for v in verifies} == expected_labels


def test_wait_precedes_verify() -> None:
    """The first WaitUntil precedes the first VerifyLayout."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    waits = [i for i, g in enumerate(goals) if isinstance(g, WaitUntil)]
    verifies = [i for i, g in enumerate(goals) if isinstance(g, VerifyLayout)]
    assert waits and verifies
    assert min(waits) < min(verifies), "WaitUntil should precede VerifyLayout"


def test_mines_every_ore_type() -> None:
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
        f"Goals must mine all five ore leaves; missing {expected - set(mined.keys())}"
    )
    for item, qty in mined.items():
        assert qty > 0, f"MineOre({ItemType(item).name}) has zero qty"


def test_smelts_each_plate_type() -> None:
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
        f"Goals must smelt every plate; missing {expected_plates - produced_outputs}"
    )


def test_ends_with_a_settling_wait() -> None:
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


def _assert_pallet_holds(state, tile: tuple[int, int], item: ItemType) -> None:
    machine_types = np.asarray(state.env_state.machine_types)
    tile_entity = np.asarray(state.env_state.tile_entity)
    ent_buf_type = np.asarray(state.env_state.ent_buf_type)
    ent_buf_count = np.asarray(state.env_state.ent_buf_count)

    bx, by = tile
    assert int(machine_types[by, bx]) == int(MachineType.PALLET), (
        f"tile {tile} should be PALLET; got "
        f"{MachineType(int(machine_types[by, bx])).name}"
    )
    ent_id = int(tile_entity[by, bx])
    assert ent_id >= 0, f"entity not registered at {tile}"
    assert int(ent_buf_count[ent_id]) >= 1, (
        f"PALLET at {tile} is empty; expected {item.name}"
    )
    assert int(ent_buf_type[ent_id]) == int(item), (
        f"PALLET at {tile} holds "
        f"{ItemType(int(ent_buf_type[ent_id])).name}, expected {item.name}"
    )


@pytest.mark.slow
def test_unlocks_expected_floor() -> None:
    """The agent unlocks every achievement in :data:`_EXPECTED_UNLOCKS`."""
    mask, timing, _ = _run_agent(max_steps=8000)
    missing = [name for name in _EXPECTED_UNLOCKS if not bool(mask[_index_of(name)])]
    timing_str = ", ".join(
        f"{ROCKET_ACHIEVEMENT_INFO[i].id}@{timing[i]}"
        for i in range(NUM_ROCKET_ACHIEVEMENTS)
        if timing[i] >= 0
    )
    assert not missing, f"Achievement floor missing {missing}.\nUnlocked: {timing_str}"


@pytest.mark.slow
def test_terminal_plate_buses_accumulate_plates() -> None:
    """Iron and silicon plate-bus PALLETs are still terminal (no
    extractor placed yet) and accumulate their plate type."""
    _, _, state = _run_agent(max_steps=8000)
    _assert_pallet_holds(state, _IRON_PLATE_BUS_TILE, ItemType.IRON_PLATE)
    _assert_pallet_holds(state, _SILICON_PLATE_BUS_TILE, ItemType.WAFER)


@pytest.mark.slow
def test_wire_output_pallet_accumulates_wire() -> None:
    """The WIRE assembler's output pallet at (15, 14) holds WIRE.

    Load-bearing for the inter-smelter route: copper plates extracted
    from (9, 13) and tin plates extracted from (9, 16) both reach the
    WIRE assembler's input pallets and an arm pushes the assembled
    WIRE into (15, 14). The copper and tin plate-buses themselves
    are *transient* (drained by extractor arms), so they are not
    checked directly — a non-empty WIRE output proves both feeds
    worked end-to-end.
    """
    _, _, state = _run_agent(max_steps=8000)
    _assert_pallet_holds(state, _WIRE_OUTPUT_TILE, ItemType.WIRE)


@pytest.mark.slow
def test_frame_output_pallet_accumulates_frame() -> None:
    """The FRAME assembler's output pallet at (21, 22) holds FRAME.

    Load-bearing for the F+A drain bypass: iron plates extracted
    from (9, 10) run east along row 10 to col 19, then south down
    col 19 — the first fully-clear column east of the pre-placed
    FURNACE (15, 16) + ASSEMBLER (17, 16). Tin plates come from
    the existing splitter's DOWN output. Both arrive at the FRAME
    assembler at (19, 22); a non-empty output pallet proves both
    routes survived the drain zone.
    """
    _, _, state = _run_agent(max_steps=8000)
    _assert_pallet_holds(state, _FRAME_OUTPUT_TILE, ItemType.FRAME)
