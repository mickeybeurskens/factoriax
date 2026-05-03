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
    _BUS_LEAVES_PLATES,
    _CIRCUIT_ASSEMBLER_TILE,
    _CIRCUIT_EXTRACT_ARM_TILE,
    _CIRCUIT_INPUT_A_TILE,
    _CIRCUIT_INPUT_B_TILE,
    _CIRCUIT_IRON_CROSSING_TILE,
    _CIRCUIT_OUTPUT_TILE,
    _CIRCUIT_TO_SENSOR_BELTS,
    _CIRCUIT_TO_SENSOR_MID_BELTS,
    _CIRCUIT_TO_SENSOR_TAIL_BELTS,
    _CIRCUIT_WIRE_CROSSING_TILE,
    _COPPER_EXTRACT_ARM_TILE,
    _COPPER_SPLITTER_TILE,
    _COPPER_TO_CIRCUIT_BELTS,
    _FRAME_ASSEMBLER_TILE,
    _FRAME_EXTRACT_ARM_TILE,
    _FRAME_INPUT_A_TILE,
    _FRAME_INPUT_B_TILE,
    _FRAME_OUTPUT_TILE,
    _IRON_EXTRACT_ARM_TILE,
    _IRON_TO_FRAME_BELTS,
    _MOTOR_ASSEMBLER_TILE,
    _MOTOR_INPUT_A_TILE,
    _MOTOR_INPUT_B_TILE,
    _MOTOR_OUTPUT_TILE,
    _SENSOR_ASSEMBLER_TILE,
    _SENSOR_INPUT_A_TILE,
    _SENSOR_INPUT_B_TILE,
    _SENSOR_OUTPUT_TILE,
    _TIN_EXTRACT_ARM_TILE,
    _TIN_TO_FRAME_BELTS,
    _WAFER_CROSSING_TILE,
    _WAFER_EXTRACT_ARM_TILE,
    _WAFER_POST_CROSSING_BELTS,
    _WAFER_PRE_CROSSING_BELTS,
    _WIRE_ASSEMBLER_TILE,
    _WIRE_COPPER_CROSSING_TILE,
    _WIRE_EXTRACT_ARM_TILE,
    _WIRE_INPUT_A_TILE,
    _WIRE_INPUT_B_TILE,
    _WIRE_IRON_CROSSING_TILE,
    _WIRE_OUTPUT_TILE,
    _WIRE_SPLITTER_TILE,
    _WIRE_TO_MOTOR_BELTS,
    _WIRE_TO_MOTOR_MID_BELTS,
    _WIRE_TO_MOTOR_TAIL_BELTS,
    _WIRE_TO_SENSOR_BELTS,
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
    # CIRCUIT assembler fed by the copper splitter at (13, 12) and
    # the wafer extractor at (9, 18); the wafer route crosses the tin
    # trunk via a CROSSING at (11, 17).
    "craft_circuit",
    # MOTOR assembler at (24, 22) fed by a FRAME extractor at (22, 22)
    # and a WIRE extractor at (16, 14); the wire route crosses the
    # iron-to-FRAME col-19 trunk via a CROSSING at (19, 14).
    "craft_motor",
    # SENSOR assembler at (28, 18) fed by a CIRCUIT extractor at
    # (18, 19) and a WIRE splitter at (24, 12); the CIRCUIT route
    # crosses the iron and wire trunks via two CROSSINGs at (19, 20)
    # and (24, 20).
    "craft_sensor",
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
    """The WIRE assembler module + extractor arms + both inter-cell
    splitters + route belts land at the documented tiles."""
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

    # Copper extractor arm + pre-splitter L-route + horizontal SPLITTER
    # at (13, 12). The splitter's DOWN output lands on the WIRE
    # input_a feeder at (13, 13); its UP output is consumed by Phase
    # 3.CIRCUIT off (13, 11).
    assert by_tile[_COPPER_EXTRACT_ARM_TILE][0] == int(MachineType.ARM)
    assert by_tile[(11, 13)] == (int(MachineType.CONVEYOR_BELT), int(Direction.UP))
    assert by_tile[(11, 12)] == (int(MachineType.CONVEYOR_BELT), int(Direction.RIGHT))
    assert by_tile[(12, 12)] == (int(MachineType.CONVEYOR_BELT), int(Direction.RIGHT))
    assert by_tile[_COPPER_SPLITTER_TILE] == (
        int(MachineType.SPLITTER),
        int(Direction.RIGHT),
    )

    # Tin extractor arm + SPLITTER + 2 belt L-route to input_b.
    # The splitter at (11, 16) lets the same extractor feed both
    # the WIRE route (via its UP output) and the FRAME route (via
    # its DOWN output). The UP output drops onto (11, 15) RIGHT
    # then bends to (12, 15) UP into the WIRE input_b.
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
    """Every plate-bus / output-pallet extractor arm lands adjacent to
    a non-walkable PALLET; the natural back-side stand tile is the
    PALLET itself, so each arm goes via from-back."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    arms_from_back = [
        g
        for g in goals
        if isinstance(g, PlaceMachineFromBackAt)
        and g.machine_type == int(MachineType.ARM)
    ]
    by_target = {g.target: g.facing for g in arms_from_back}
    assert by_target == {
        _IRON_EXTRACT_ARM_TILE: int(Direction.RIGHT),
        _COPPER_EXTRACT_ARM_TILE: int(Direction.RIGHT),
        _TIN_EXTRACT_ARM_TILE: int(Direction.RIGHT),
        _WAFER_EXTRACT_ARM_TILE: int(Direction.UP),
        _WIRE_EXTRACT_ARM_TILE: int(Direction.UP),
        _FRAME_EXTRACT_ARM_TILE: int(Direction.RIGHT),
        _CIRCUIT_EXTRACT_ARM_TILE: int(Direction.DOWN),
    }


def test_emits_circuit_cell_placements() -> None:
    """The CIRCUIT assembler module + wafer extractor + copper-tap +
    wafer route + CROSSING land at the documented tiles."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    placements = [
        g for g in goals if isinstance(g, (PlaceMachineAt, PlaceMachineFromBackAt))
    ]
    by_tile = {g.target: (g.machine_type, g.facing) for g in placements}

    # CIRCUIT module: inputs are feeder belts; output is a pallet.
    cx, cy = _CIRCUIT_ASSEMBLER_TILE
    assert by_tile[_CIRCUIT_ASSEMBLER_TILE][0] == int(MachineType.ASSEMBLER)
    assert by_tile[_CIRCUIT_INPUT_A_TILE][0] == int(MachineType.CONVEYOR_BELT)
    assert by_tile[_CIRCUIT_INPUT_B_TILE][0] == int(MachineType.CONVEYOR_BELT)
    assert by_tile[_CIRCUIT_OUTPUT_TILE][0] == int(MachineType.PALLET)
    assert by_tile[(cx + 1, cy)][0] == int(MachineType.ARM)  # output arm

    # Wafer extractor (from-back, facing UP from (9, 18)).
    assert by_tile[_WAFER_EXTRACT_ARM_TILE] == (
        int(MachineType.ARM),
        int(Direction.UP),
    )

    # CROSSING shares (11, 17) with the tin -> FRAME trunk; dir=1
    # is N->S vertical + W->E horizontal.
    assert by_tile[_WAFER_CROSSING_TILE] == (int(MachineType.CROSSING), 1)

    # Every copper-to-CIRCUIT and wafer-route belt landed.
    for tile, _facing in _COPPER_TO_CIRCUIT_BELTS:
        assert by_tile[tile][0] == int(MachineType.CONVEYOR_BELT), tile
    for tile, _facing in _WAFER_PRE_CROSSING_BELTS:
        assert by_tile[tile][0] == int(MachineType.CONVEYOR_BELT), tile
    for tile, _facing in _WAFER_POST_CROSSING_BELTS:
        assert by_tile[tile][0] == int(MachineType.CONVEYOR_BELT), tile


def test_emits_motor_cell_placements() -> None:
    """The MOTOR assembler module + WIRE / FRAME extractor arms +
    WIRE route + two CROSSINGs land at the documented tiles."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    placements = [
        g for g in goals if isinstance(g, (PlaceMachineAt, PlaceMachineFromBackAt))
    ]
    by_tile = {g.target: (g.machine_type, g.facing) for g in placements}

    # MOTOR module: inputs are feeder belts; output is a pallet.
    cx, cy = _MOTOR_ASSEMBLER_TILE
    assert by_tile[_MOTOR_ASSEMBLER_TILE][0] == int(MachineType.ASSEMBLER)
    assert by_tile[_MOTOR_INPUT_A_TILE][0] == int(MachineType.CONVEYOR_BELT)
    assert by_tile[_MOTOR_INPUT_B_TILE][0] == int(MachineType.CONVEYOR_BELT)
    assert by_tile[_MOTOR_OUTPUT_TILE][0] == int(MachineType.PALLET)
    assert by_tile[(cx + 1, cy)][0] == int(MachineType.ARM)  # output arm

    # Extractor arms.
    assert by_tile[_WIRE_EXTRACT_ARM_TILE] == (
        int(MachineType.ARM),
        int(Direction.UP),
    )
    assert by_tile[_FRAME_EXTRACT_ARM_TILE] == (
        int(MachineType.ARM),
        int(Direction.RIGHT),
    )

    # Both CROSSINGs share their tile with an upstream cell's trunk;
    # dir=1 = N->S vertical (copper or iron) + W->E horizontal (wire).
    assert by_tile[_WIRE_COPPER_CROSSING_TILE] == (int(MachineType.CROSSING), 1)
    assert by_tile[_WIRE_IRON_CROSSING_TILE] == (int(MachineType.CROSSING), 1)

    # Every wire-route belt landed.
    for tile, _facing in _WIRE_TO_MOTOR_BELTS:
        assert by_tile[tile][0] == int(MachineType.CONVEYOR_BELT), tile
    for tile, _facing in _WIRE_TO_MOTOR_MID_BELTS:
        assert by_tile[tile][0] == int(MachineType.CONVEYOR_BELT), tile
    for tile, _facing in _WIRE_TO_MOTOR_TAIL_BELTS:
        assert by_tile[tile][0] == int(MachineType.CONVEYOR_BELT), tile


def test_circuit_and_iron_belts_skip_motor_crossing_tiles() -> None:
    """The (16, 12) and (19, 12) tiles are CROSSINGs placed by
    Phase 3.MOTOR, not BELTs — verifies CIRCUIT and FRAME phases
    drop those tiles from their respective belt lists."""
    copper_tiles = {tile for tile, _ in _COPPER_TO_CIRCUIT_BELTS}
    iron_tiles = {tile for tile, _ in _IRON_TO_FRAME_BELTS}
    assert _WIRE_COPPER_CROSSING_TILE not in copper_tiles
    assert _WIRE_IRON_CROSSING_TILE not in iron_tiles


def test_iron_and_wire_belts_skip_sensor_crossing_tiles() -> None:
    """The (19, 20) and (24, 20) tiles are CROSSINGs placed by
    Phase 3.SENSOR, not BELTs — verifies FRAME and MOTOR phases
    drop those tiles from their respective belt lists. (24, 12)
    is a SPLITTER placed by SENSOR, also dropped from MOTOR."""
    iron_tiles = {tile for tile, _ in _IRON_TO_FRAME_BELTS}
    wire_motor_tiles = {tile for tile, _ in _WIRE_TO_MOTOR_BELTS}
    assert _CIRCUIT_IRON_CROSSING_TILE not in iron_tiles
    assert _CIRCUIT_WIRE_CROSSING_TILE not in wire_motor_tiles
    assert _WIRE_SPLITTER_TILE not in wire_motor_tiles


def test_emits_sensor_cell_placements() -> None:
    """The SENSOR assembler module + WIRE splitter + CIRCUIT
    extractor + WIRE / CIRCUIT routes + 2 CROSSINGs land at the
    documented tiles."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    placements = [
        g for g in goals if isinstance(g, (PlaceMachineAt, PlaceMachineFromBackAt))
    ]
    by_tile = {g.target: (g.machine_type, g.facing) for g in placements}

    # SENSOR module: inputs are feeder belts; output is a pallet.
    cx, cy = _SENSOR_ASSEMBLER_TILE
    assert by_tile[_SENSOR_ASSEMBLER_TILE][0] == int(MachineType.ASSEMBLER)
    assert by_tile[_SENSOR_INPUT_A_TILE][0] == int(MachineType.CONVEYOR_BELT)
    assert by_tile[_SENSOR_INPUT_B_TILE][0] == int(MachineType.CONVEYOR_BELT)
    assert by_tile[_SENSOR_OUTPUT_TILE][0] == int(MachineType.PALLET)
    assert by_tile[(cx + 1, cy)][0] == int(MachineType.ARM)  # output arm

    # CIRCUIT extractor (from-back, facing DOWN).
    assert by_tile[_CIRCUIT_EXTRACT_ARM_TILE] == (
        int(MachineType.ARM),
        int(Direction.DOWN),
    )

    # WIRE splitter inline on the WIRE -> MOTOR trunk.
    assert by_tile[_WIRE_SPLITTER_TILE] == (
        int(MachineType.SPLITTER),
        int(Direction.RIGHT),
    )

    # Two CROSSINGs on the CIRCUIT route across iron and wire trunks.
    assert by_tile[_CIRCUIT_IRON_CROSSING_TILE] == (int(MachineType.CROSSING), 1)
    assert by_tile[_CIRCUIT_WIRE_CROSSING_TILE] == (int(MachineType.CROSSING), 1)

    for tile, _facing in _WIRE_TO_SENSOR_BELTS:
        assert by_tile[tile][0] == int(MachineType.CONVEYOR_BELT), tile
    for tile, _facing in _CIRCUIT_TO_SENSOR_BELTS:
        assert by_tile[tile][0] == int(MachineType.CONVEYOR_BELT), tile
    for tile, _facing in _CIRCUIT_TO_SENSOR_MID_BELTS:
        assert by_tile[tile][0] == int(MachineType.CONVEYOR_BELT), tile
    for tile, _facing in _CIRCUIT_TO_SENSOR_TAIL_BELTS:
        assert by_tile[tile][0] == int(MachineType.CONVEYOR_BELT), tile


def test_emits_frame_cell_placements() -> None:
    """The FRAME assembler module + iron extractor + iron / tin
    routes land at the documented tiles. The tin trunk head at
    (11, 17) is *not* a FRAME-phase placement — Phase 3.CIRCUIT
    drops a CROSSING there instead."""
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

    # Tin route belts (the trunk head at (11, 17) is a CROSSING,
    # placed by Phase 3.CIRCUIT, not a BELT).
    for tile, _facing in _TIN_TO_FRAME_BELTS:
        assert by_tile[tile][0] == int(MachineType.CONVEYOR_BELT), tile
    assert by_tile[_WAFER_CROSSING_TILE][0] == int(MachineType.CROSSING)


def test_frame_routes_avoid_pre_placed_drain_zone() -> None:
    """No FRAME or CIRCUIT plate-feed tile sits on a 4-neighbour of
    the pre-placed FURNACE (15, 16) or ASSEMBLER (17, 16) *as a belt
    facing toward those combiners*. The directional Phase 0 pull
    only siphons from belts that face the combiner; belts facing
    parallel to or perpendicular-away from the combiner are safe.
    The tile (16, 16) is in the drain zone but our copper-to-CIRCUIT
    belt there faces DOWN (not LEFT/RIGHT toward F or A), so no
    siphon."""
    drain_zone: set[tuple[int, int]] = set()
    for cx, cy in ((15, 16), (17, 16)):
        drain_zone |= {(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)}

    # Tiles where the belt's facing matters: a belt facing INTO a
    # combiner gets drained. Map (tile, facing) and assert no entry
    # faces the wrong way.
    feeds: list[tuple[tuple[int, int], int]] = []
    feeds.extend(_IRON_TO_FRAME_BELTS)
    feeds.extend(_TIN_TO_FRAME_BELTS)
    feeds.extend(_COPPER_TO_CIRCUIT_BELTS)
    feeds.extend(_WAFER_PRE_CROSSING_BELTS)
    feeds.extend(_WAFER_POST_CROSSING_BELTS)
    feeds.extend(_WIRE_TO_MOTOR_BELTS)
    feeds.extend(_WIRE_TO_MOTOR_MID_BELTS)
    feeds.extend(_WIRE_TO_MOTOR_TAIL_BELTS)
    feeds.extend(_WIRE_TO_SENSOR_BELTS)
    feeds.extend(_CIRCUIT_TO_SENSOR_BELTS)
    feeds.extend(_CIRCUIT_TO_SENSOR_MID_BELTS)
    feeds.extend(_CIRCUIT_TO_SENSOR_TAIL_BELTS)
    feeds.append((_FRAME_INPUT_A_TILE, int(Direction.DOWN)))
    feeds.append((_FRAME_INPUT_B_TILE, int(Direction.RIGHT)))
    feeds.append((_CIRCUIT_INPUT_A_TILE, int(Direction.DOWN)))
    feeds.append((_CIRCUIT_INPUT_B_TILE, int(Direction.RIGHT)))
    feeds.append((_MOTOR_INPUT_A_TILE, int(Direction.DOWN)))
    feeds.append((_MOTOR_INPUT_B_TILE, int(Direction.RIGHT)))
    feeds.append((_SENSOR_INPUT_A_TILE, int(Direction.DOWN)))
    feeds.append((_SENSOR_INPUT_B_TILE, int(Direction.RIGHT)))

    # For each combiner, list the {neighbour_tile: bad_facing} where
    # bad_facing is the direction a belt at neighbour_tile would face
    # to be drained.
    bad: dict[tuple[int, int], int] = {}
    for cx, cy in ((15, 16), (17, 16)):
        bad[(cx - 1, cy)] = int(
            Direction.RIGHT
        )  # west neighbour facing east -> drained
        bad[(cx + 1, cy)] = int(Direction.LEFT)  # east -> west
        bad[(cx, cy - 1)] = int(Direction.DOWN)  # north -> south
        bad[(cx, cy + 1)] = int(Direction.UP)  # south -> north

    drained: list[tuple[tuple[int, int], int]] = []
    for tile, facing in feeds:
        if tile in bad and bad[tile] == facing:
            drained.append((tile, facing))
    assert not drained, f"plate-feed belts that would be drained: {drained}"

    # Sanity: feed tiles in the drain zone exist but with safe facings.
    feed_tiles = {tile for tile, _ in feeds}
    overlap = feed_tiles & drain_zone
    if overlap:
        for tile in overlap:
            facing = next(f for t, f in feeds if t == tile)
            assert bad.get(tile) != facing, tile


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
    """Each smelter cell + each later stage end with their own
    VerifyLayout (9 total: 4 smelter + 1 wire + 1 frame + 1 circuit
    + 1 motor + 1 sensor)."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    verifies = [g for g in goals if isinstance(g, VerifyLayout)]
    expected_labels = {
        "phase 1.iron",
        "phase 1.copper",
        "phase 1.tin",
        "phase 1.silicon",
        "phase 2.wire",
        "phase 3.frame",
        "phase 3.circuit",
        "phase 3.motor",
        "phase 3.sensor",
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


def test_phase_0a_smelts_iron_copper_tin_only() -> None:
    """Phase 0a hand-smelts the three plates Phase 1 cells consume.

    WAFER is *not* hand-smelted: the silicon cell built in Phase 1
    smelts wafer automatically, and Phase 0b withdraws any wafer the
    later assembler crafts (CIRCUIT) need from the silicon cell's
    plate-bus.
    """
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    produced_outputs: set[int] = set()
    for g in goals:
        if isinstance(g, ProduceInMachine):
            produced_outputs.add(g.output_item)
    expected_plates = {
        int(ItemType.IRON_PLATE),
        int(ItemType.COPPER_PLATE),
        int(ItemType.TIN_PLATE),
    }
    assert produced_outputs >= expected_plates, (
        f"Goals must smelt iron / copper / tin; missing "
        f"{expected_plates - produced_outputs}"
    )
    assert int(ItemType.WAFER) not in produced_outputs, (
        "WAFER should not appear in the bootstrap smelt set — the silicon "
        "cell handles it"
    )


def test_ends_with_a_settling_wait() -> None:
    """The final goal is a Wait so plates accumulate before episode end."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    assert isinstance(goals[-1], Wait)


def test_bus_pull_chunk_counts_are_cumulative() -> None:
    """Bus-pull chunks must aim for *cumulative* held counts, not per-chunk.

    ``ProduceInMachine.step`` (goals.py) exits as soon as
    ``view.player.held(output_item) >= self.count``. The player doesn't
    drop placed items until Phase 2/3, so held counts grow monotonically
    across chunks. If chunk N's ``count`` is per-chunk (e.g. 25), the
    player already holds 25 from chunk N-1, so chunk N returns DONE
    immediately and produces nothing. Result: only the first chunk_size
    of any item ever gets crafted, and Phase 2/3 placements starve.

    Forces a target requiring multiple chunks (60 > _BUS_PULL_CHUNK=25)
    and asserts the resulting ProduceInMachine goals have strictly
    increasing counts across chunks.
    """
    from baselines.rocket.scripted.agent_advanced_factory import (
        _BUS_PULL_CHUNK,
        _bus_pull_phase,
    )

    qty = _BUS_PULL_CHUNK * 2 + 10  # 60 with the current chunk constant
    targets = {int(ItemType.CONVEYOR_BELT): qty}
    phase_goals = _bus_pull_phase(
        targets, _BUS_LEAVES_PLATES, ROCKET_RECIPE_BOOK, slack={}
    )
    belt_produce = [
        g
        for g in phase_goals
        if isinstance(g, ProduceInMachine)
        and g.output_item == int(ItemType.CONVEYOR_BELT)
    ]
    assert len(belt_produce) >= 3, (
        f"qty={qty} > _BUS_PULL_CHUNK={_BUS_PULL_CHUNK} should yield "
        f">=3 ProduceInMachine chunks; got {len(belt_produce)}"
    )
    counts = [g.count for g in belt_produce]
    assert all(b > a for a, b in zip(counts, counts[1:])), (
        f"Bus-pull ProduceInMachine counts {counts} aren't strictly "
        f"increasing; later chunks will no-op because the player's held "
        f"count from earlier chunks already satisfies the per-chunk count."
    )
    assert counts[-1] >= qty, (
        f"Final chunk count {counts[-1]} < total target {qty}; the "
        f"cumulative target must reach the requested qty by the last chunk."
    )


def test_bus_pull_uses_finished_wire_when_available() -> None:
    """When WIRE is in ``available_buses``, MINER crafts pull finished WIRE.

    With WIRE listed as a bus leaf, the BOM walk treats it as raw and
    the chunk loop emits a ``WithdrawFromBusAt`` against the WIRE
    output pallet. The COPPER_PLATE / TIN_PLATE inputs the WIRE recipe
    would consume should not appear as MineOre goals (no copper/tin
    ore demand) because WIRE itself is now a leaf.
    """
    from baselines.rocket.scripted.agent_advanced_factory import (
        _BUS_PALLETS,
        _bus_pull_phase,
    )

    targets = {int(ItemType.MINER): 5}
    available = frozenset(_BUS_LEAVES_PLATES | {int(ItemType.WIRE)})
    phase_goals = _bus_pull_phase(targets, available, ROCKET_RECIPE_BOOK, slack={})

    from baselines.rocket.scripted.goals import WithdrawFromBusAt

    wire_pulls = [
        g
        for g in phase_goals
        if isinstance(g, WithdrawFromBusAt)
        and g.item_type == int(ItemType.WIRE)
        and g.tile == _BUS_PALLETS[int(ItemType.WIRE)]
    ]
    assert wire_pulls, "expected at least one WithdrawFromBusAt(WIRE) chunk"

    copper_mines = [
        g
        for g in phase_goals
        if isinstance(g, MineOre) and g.item_type == int(ItemType.COPPER_ORE)
    ]
    tin_mines = [
        g
        for g in phase_goals
        if isinstance(g, MineOre) and g.item_type == int(ItemType.TIN_ORE)
    ]
    assert not copper_mines, (
        "WIRE listed as a bus leaf should suppress COPPER_ORE mining"
    )
    assert not tin_mines, "WIRE listed as a bus leaf should suppress TIN_ORE mining"


def test_bus_pull_uses_bus_frame_and_circuit_for_assemblers() -> None:
    """With FRAME+CIRCUIT bus leaves, ASSEMBLER crafts pull finished items."""
    from baselines.rocket.scripted.agent_advanced_factory import (
        _BUS_PALLETS,
        _bus_pull_phase,
    )
    from baselines.rocket.scripted.goals import WithdrawFromBusAt

    targets = {int(ItemType.ASSEMBLER): 2}
    available = frozenset(
        _BUS_LEAVES_PLATES
        | {int(ItemType.WIRE), int(ItemType.FRAME), int(ItemType.CIRCUIT)}
    )
    phase_goals = _bus_pull_phase(targets, available, ROCKET_RECIPE_BOOK, slack={})

    frame_pulls = [
        g
        for g in phase_goals
        if isinstance(g, WithdrawFromBusAt)
        and g.item_type == int(ItemType.FRAME)
        and g.tile == _BUS_PALLETS[int(ItemType.FRAME)]
    ]
    circuit_pulls = [
        g
        for g in phase_goals
        if isinstance(g, WithdrawFromBusAt)
        and g.item_type == int(ItemType.CIRCUIT)
        and g.tile == _BUS_PALLETS[int(ItemType.CIRCUIT)]
    ]
    assert frame_pulls, "expected WithdrawFromBusAt(FRAME) for ASSEMBLER recipe"
    assert circuit_pulls, "expected WithdrawFromBusAt(CIRCUIT) for ASSEMBLER recipe"

    # FRAME's recipe inputs (IRON_PLATE+TIN_PLATE) shouldn't appear in
    # the schedule as crafted items because FRAME is a leaf.
    produced = {g.output_item for g in phase_goals if isinstance(g, ProduceInMachine)}
    assert int(ItemType.FRAME) not in produced
    assert int(ItemType.CIRCUIT) not in produced


def test_bus_pull_falls_back_to_plates_without_frame_circuit() -> None:
    """Without FRAME/CIRCUIT in available_buses, the ASSEMBLER walk
    recurses to plates and the chunk loop crafts FRAME and CIRCUIT
    at the pre-placed assembler.
    """
    from baselines.rocket.scripted.agent_advanced_factory import _bus_pull_phase
    from baselines.rocket.scripted.goals import WithdrawFromBusAt

    targets = {int(ItemType.ASSEMBLER): 2}
    available = frozenset(_BUS_LEAVES_PLATES | {int(ItemType.WIRE)})
    phase_goals = _bus_pull_phase(targets, available, ROCKET_RECIPE_BOOK, slack={})

    produced = {g.output_item for g in phase_goals if isinstance(g, ProduceInMachine)}
    assert int(ItemType.FRAME) in produced, (
        "FRAME not a bus leaf -> ASSEMBLER schedule must craft FRAME"
    )
    assert int(ItemType.CIRCUIT) in produced

    pull_items = {g.item_type for g in phase_goals if isinstance(g, WithdrawFromBusAt)}
    assert int(ItemType.FRAME) not in pull_items
    assert int(ItemType.CIRCUIT) not in pull_items


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


@pytest.mark.slow
def test_circuit_output_pallet_accumulates_circuit() -> None:
    """The CIRCUIT assembler's output pallet at (18, 18) holds CIRCUIT.

    Load-bearing for the inter-cell splitter + CROSSING design:
    copper from the WIRE-bound splitter at (13, 12) UP-output runs
    east on row 11 to col 16, then south down col 16 to the CIRCUIT
    input_a feeder at (16, 17). Wafer from the silicon plate-bus
    at (9, 19) is extracted UP, crosses the tin -> FRAME trunk via
    a CROSSING at (11, 17), and lands at the CIRCUIT input_b feeder
    at (15, 18). A non-empty output pallet proves both inputs reach
    the assembler concurrently.
    """
    _, _, state = _run_agent(max_steps=8000)
    _assert_pallet_holds(state, _CIRCUIT_OUTPUT_TILE, ItemType.CIRCUIT)


@pytest.mark.slow
def test_motor_output_pallet_accumulates_motor() -> None:
    """The MOTOR assembler's output pallet at (26, 22) holds MOTOR.

    Load-bearing for the first cell-chain bridge: FRAME extractor at
    (22, 22) RIGHT pushes directly onto the MOTOR input_b feeder at
    (23, 22), and a WIRE extractor at (16, 14) RIGHT routes wire east
    along row 14, through a CROSSING at (19, 14) (vertical lane =
    iron DOWN unchanged, horizontal lane = wire RIGHT), and south
    down col 24 to the input_a feeder at (24, 21). A non-empty MOTOR
    pallet proves both upstream cells (FRAME + WIRE) feed it under
    the new CROSSING design.
    """
    _, _, state = _run_agent(max_steps=8000)
    _assert_pallet_holds(state, _MOTOR_OUTPUT_TILE, ItemType.MOTOR)


@pytest.mark.slow
def test_sensor_output_pallet_accumulates_sensor() -> None:
    """The SENSOR assembler's output pallet at (30, 18) holds SENSOR.

    Load-bearing for the WIRE splitter design: a SPLITTER at (24, 12)
    fans wire DOWN to MOTOR (existing route) and UP to a new SENSOR-
    bound corridor on row 11 + col 28 south. CIRCUIT comes from the
    CIRCUIT output PALLET via an extractor at (18, 19) DOWN; the
    route runs east on row 20 through two CROSSINGs (iron at
    (19, 20), wire at (24, 20)) and lands on the SENSOR input_b
    feeder at (27, 18). A non-empty SENSOR pallet proves the WIRE
    fan-out and the dual-CROSSING CIRCUIT route both work.
    """
    _, _, state = _run_agent(max_steps=8000)
    _assert_pallet_holds(state, _SENSOR_OUTPUT_TILE, ItemType.SENSOR)
