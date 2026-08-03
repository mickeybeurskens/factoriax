"""Tests for the rocket scenario.

Covers the achievement catalogue, per-condition correctness, reward
semantics, and a small end-to-end run through the scenario runner.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.engine.constants import (
    MAX_ACHIEVEMENTS,
    NUM_ITEM_TYPES,
    Action,
    BlockType,
    Direction,
    ItemType,
    Machine,
)
from factoriax.engine.envs.rocket import (
    MAX_ROCKET_SCORE,
    ROCKET_ACHIEVEMENT_WEIGHTS,
    ROCKET_ACHIEVEMENTS,
    ROCKET_BLOCKED_ACTIONS,
    build_rocket_level,
    rocket_conditions,
    rocket_reward,
)
from factoriax.engine.state import EnvParams, EnvState

# ---------------------------------------------------------------------------
# Catalogue + construction
# ---------------------------------------------------------------------------


def test_catalogue_size_and_weights() -> None:
    """The catalogue has 38 entries and tiered weights summing to 140."""
    assert len(ROCKET_ACHIEVEMENTS) == 38
    assert ROCKET_ACHIEVEMENT_WEIGHTS.shape == (MAX_ACHIEVEMENTS,)
    weights_np = np.asarray(ROCKET_ACHIEVEMENT_WEIGHTS)
    assert float(weights_np.sum()) == pytest.approx(MAX_ROCKET_SCORE)
    assert MAX_ROCKET_SCORE == 140
    # Tier shape: 10 × 1, 11 × 3, 13 × 5, 4 × 8.
    assert list(weights_np[:10]) == [1.0] * 10
    assert list(weights_np[10:21]) == [3.0] * 11
    assert list(weights_np[21:34]) == [5.0] * 13
    assert list(weights_np[34:38]) == [8.0] * 4
    assert float(weights_np[38:].sum()) == 0.0


def test_all_ids_unique() -> None:
    """No duplicate achievement ids."""
    ids = [a.id for a in ROCKET_ACHIEVEMENTS]
    assert len(set(ids)) == len(ids)


def test_rocket_factory_builds_env() -> None:
    """The registry factory yields the rocket env at the advertised params."""
    from factoriax.make import env_from_name

    env, params = env_from_name("Rocket-v1")
    assert params.max_timesteps == 8000
    assert env.map_width == 32
    assert env.map_height == 32
    assert env.num_players == 1


def test_build_rocket_level_has_all_ore_types() -> None:
    """All five ore block types are present on the level map."""
    level = build_rocket_level()
    present = set(int(x) for x in np.unique(np.asarray(level.block_map)))
    for expected in (
        BlockType.IRON,
        BlockType.COPPER,
        BlockType.COAL,
        BlockType.TIN,
        BlockType.SILICON,
    ):
        assert int(expected) in present


def test_build_rocket_level_preplaces_furnace_and_assembler() -> None:
    """A furnace and assembler are pre-placed adjacent to spawn."""
    level = build_rocket_level()
    mt = np.asarray(level.machine_types)
    # Spawn at map center (16, 16). Furnace immediately west, assembler east.
    assert int(mt[16, 15]) == int(Machine.FURNACE)
    assert int(mt[16, 17]) == int(Machine.ASSEMBLER)


def test_rocket_blocks_all_craft_actions() -> None:
    """The rocket hand-craft mask is exactly the CRAFT_* family."""
    from factoriax.engine.constants import CRAFT_ITEMS

    # The whole craft family is blocked (one action per non-resource item),
    # including the machine crafts the old hand-numbered range leaked.
    assert len(ROCKET_BLOCKED_ACTIONS) == len(CRAFT_ITEMS)
    assert {a for a in Action if a.name.startswith("CRAFT_")} == {
        Action(v) for v in ROCKET_BLOCKED_ACTIONS
    }
    assert int(Action.CRAFT_IRON_PLATE) in ROCKET_BLOCKED_ACTIONS
    assert int(Action.CRAFT_ROCKET) in ROCKET_BLOCKED_ACTIONS
    # Movement / mining / placement actions must NOT be blocked.
    for allowed in (
        Action.NOOP,
        Action.MINE,
        Action.UP,
        Action.PLACE_MINER,
        Action.WITHDRAW,
        Action.DEPOSIT_COAL,
    ):
        assert int(allowed) not in ROCKET_BLOCKED_ACTIONS


# ``test_action_mask_wrapper_noops_blocked_actions`` (~4.1s, unique
# 32x32 step compile) was removed in the replacement-for-speedup pass.
# The wrapper's blocked->NOOP rewrite is covered by stub-based unit
# tests in tests/test_action_mask_wrapper.py, and the fact that
# ROCKET_BLOCKED_ACTIONS contains CRAFT_IRON_PLATE is asserted in
# test_rocket_benchmark_exposes_blocked_actions above. Chained, those
# cover the same property at sub-millisecond cost.


# ---------------------------------------------------------------------------
# Per-condition unit tests
# ---------------------------------------------------------------------------


def _index_of(achievement_id: str) -> int:
    for i, info in enumerate(ROCKET_ACHIEVEMENTS):
        if info.id == achievement_id:
            return i
    raise AssertionError(f"Unknown achievement id: {achievement_id}")


def _inv(**items: int) -> jnp.ndarray:
    """Single-player inventory with the given items."""
    arr = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
    name_to_type = {m.name: int(m) for m in ItemType}
    for name, count in items.items():
        arr = arr.at[0, name_to_type[name]].set(count)
    return arr


def _empty_map() -> jnp.ndarray:
    """Minimal 1x1 empty map for states that do not need terrain."""
    return jnp.array([[BlockType.DIRT]], dtype=jnp.int32)


# Item-holding achievements: id -> inventory kwargs that unlock it.
_ITEM_ACHIEVEMENT_CASES: list[tuple[str, dict[str, int]]] = [
    ("collect_iron", {"IRON_ORE": 1}),
    ("collect_copper", {"COPPER_ORE": 1}),
    ("collect_tin", {"TIN_ORE": 1}),
    ("collect_coal", {"COAL": 1}),
    ("collect_silicon", {"SILICON": 1}),
    ("smelt_iron", {"IRON_PLATE": 1}),
    ("smelt_copper", {"COPPER_PLATE": 1}),
    ("smelt_tin", {"TIN_PLATE": 1}),
    ("smelt_wafer", {"WAFER": 1}),
    ("craft_wire", {"WIRE": 1}),
    ("craft_circuit", {"CIRCUIT": 1}),
    ("craft_frame", {"FRAME": 1}),
    ("craft_motor", {"MOTOR": 1}),
    ("craft_sensor", {"SENSOR": 1}),
    ("craft_miner", {"MINER": 1}),
    ("craft_furnace", {"FURNACE": 1}),
    ("craft_belt", {"CONVEYOR_BELT": 1}),
    ("craft_pallet", {"PALLET": 1}),
    ("craft_arm", {"ARM": 1}),
    ("craft_assembler", {"ASSEMBLER": 1}),
    ("craft_rocket", {"ROCKET": 1}),
]


@pytest.mark.parametrize("achievement_id,inventory", _ITEM_ACHIEVEMENT_CASES)
def test_item_holding_conditions(
    achievement_id: str,
    inventory: dict[str, int],
    state_factory,
) -> None:
    """Holding a specific item flips the matching slot and nothing else."""
    state = state_factory(
        world_map=_empty_map(),
        player_inventory=_inv(**inventory),
    )
    mask = rocket_conditions(state)
    idx = _index_of(achievement_id)
    assert bool(mask[idx]), f"{achievement_id} should unlock"
    # Every other item-holding achievement stays False.
    for other_id, _ in _ITEM_ACHIEVEMENT_CASES:
        if other_id == achievement_id:
            continue
        assert not bool(mask[_index_of(other_id)]), (
            f"{other_id} wrongly unlocked alongside {achievement_id}"
        )


# Placement / count achievements: id -> (machine_types grid, expected idx).
@pytest.mark.parametrize(
    "achievement_id,mt_grid",
    [
        ("place_miner", jnp.array([[Machine.MINER]], dtype=jnp.int32)),
        ("place_furnace", jnp.array([[Machine.FURNACE]], dtype=jnp.int32)),
        (
            "place_belt",
            jnp.array([[Machine.CONVEYOR_BELT]], dtype=jnp.int32),
        ),
        ("place_pallet", jnp.array([[Machine.PALLET]], dtype=jnp.int32)),
        ("place_arm", jnp.array([[Machine.ARM]], dtype=jnp.int32)),
        (
            "place_assembler",
            jnp.array([[Machine.ASSEMBLER]], dtype=jnp.int32),
        ),
        ("place_rocket", jnp.array([[Machine.ROCKET]], dtype=jnp.int32)),
    ],
)
def test_single_placement_conditions(
    achievement_id: str, mt_grid: jnp.ndarray, state_factory
) -> None:
    """Placing a single machine flips the matching slot."""
    # Use a map matching the machine_types shape.
    shape = mt_grid.shape
    world = jnp.full(shape, BlockType.DIRT, dtype=jnp.int32)
    state = state_factory(world_map=world, machine_types=mt_grid)
    mask = rocket_conditions(state)
    assert bool(mask[_index_of(achievement_id)])


def test_belt_network_requires_five(state_factory) -> None:
    """belt_network fires at 5 belts, not at 4."""
    # 1x4 strip of belts, below threshold.
    belts_4 = jnp.full((1, 4), Machine.CONVEYOR_BELT, dtype=jnp.int32)
    world_4 = jnp.full((1, 4), BlockType.DIRT, dtype=jnp.int32)
    state_4 = state_factory(world_map=world_4, machine_types=belts_4)
    assert not bool(rocket_conditions(state_4)[_index_of("belt_network")])
    # 1x5 strip, at threshold.
    belts_5 = jnp.full((1, 5), Machine.CONVEYOR_BELT, dtype=jnp.int32)
    world_5 = jnp.full((1, 5), BlockType.DIRT, dtype=jnp.int32)
    state_5 = state_factory(world_map=world_5, machine_types=belts_5)
    assert bool(rocket_conditions(state_5)[_index_of("belt_network")])


def test_scaling_up_requires_three_miners(state_factory) -> None:
    """scaling_up fires at 3 miners, not at 2."""
    miners_2 = jnp.full((1, 2), Machine.MINER, dtype=jnp.int32)
    world_2 = jnp.full((1, 2), BlockType.DIRT, dtype=jnp.int32)
    state_2 = state_factory(world_map=world_2, machine_types=miners_2)
    assert not bool(rocket_conditions(state_2)[_index_of("scaling_up")])
    miners_3 = jnp.full((1, 3), Machine.MINER, dtype=jnp.int32)
    world_3 = jnp.full((1, 3), BlockType.DIRT, dtype=jnp.int32)
    state_3 = state_factory(world_map=world_3, machine_types=miners_3)
    assert bool(rocket_conditions(state_3)[_index_of("scaling_up")])


def test_industrialist_requires_ten_machines(state_factory) -> None:
    """industrialist fires at 10+ machines of any type."""
    # 2x5 mixed grid: 10 machines.
    grid = jnp.array(
        [
            [Machine.MINER] * 5,
            [Machine.PALLET] * 5,
        ],
        dtype=jnp.int32,
    )
    world = jnp.full((2, 5), BlockType.DIRT, dtype=jnp.int32)
    state = state_factory(world_map=world, machine_types=grid)
    assert bool(rocket_conditions(state)[_index_of("industrialist")])


def test_automated_mining_requires_buffered_ore(state_factory) -> None:
    """automated_mining fires when a placed miner has buffer items."""
    world = jnp.array([[BlockType.DIRT]], dtype=jnp.int32)
    mt = jnp.array([[Machine.MINER]], dtype=jnp.int32)
    # Empty buffer. It must NOT fire.
    state_empty = state_factory(world_map=world, machine_types=mt)
    assert not bool(rocket_conditions(state_empty)[_index_of("automated_mining")])
    # Non-empty buffer. It must fire.
    state_full = state_factory(
        world_map=world,
        machine_types=mt,
        buffer_type=jnp.array([[ItemType.IRON_ORE]], dtype=jnp.int8),
        buffer_count=jnp.array([[5]], dtype=jnp.int16),
    )
    assert bool(rocket_conditions(state_full)[_index_of("automated_mining")])


def test_pallet_filled_requires_buffered_item(state_factory) -> None:
    """pallet_filled fires when a placed pallet has items."""
    world = jnp.array([[BlockType.DIRT]], dtype=jnp.int32)
    mt = jnp.array([[Machine.PALLET]], dtype=jnp.int32)
    state = state_factory(
        world_map=world,
        machine_types=mt,
        buffer_type=jnp.array([[ItemType.IRON_PLATE]], dtype=jnp.int8),
        buffer_count=jnp.array([[3]], dtype=jnp.int16),
    )
    assert bool(rocket_conditions(state)[_index_of("pallet_filled")])


def test_first_assembly_requires_assembler_output(state_factory) -> None:
    """first_assembly fires when a placed assembler has output items."""
    world = jnp.array([[BlockType.DIRT]], dtype=jnp.int32)
    mt = jnp.array([[Machine.ASSEMBLER]], dtype=jnp.int32)
    state = state_factory(
        world_map=world,
        machine_types=mt,
        asm_out_type=jnp.array([[ItemType.WIRE]], dtype=jnp.int8),
        asm_out_count=jnp.array([[2]], dtype=jnp.int16),
    )
    assert bool(rocket_conditions(state)[_index_of("first_assembly")])


# ---------------------------------------------------------------------------
# Reward semantics
# ---------------------------------------------------------------------------


def _ach_state_with(mask_indices: list[int]) -> EnvState:
    """Build an EnvState with specific achievement slots latched."""
    # The other state fields do not matter for achievement_reward. It
    # reads only ``achievements_unlocked``.
    unlocked = jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_)
    for i in mask_indices:
        unlocked = unlocked.at[i].set(True)
    return _dummy_env_state().replace(achievements_unlocked=unlocked)


def _dummy_env_state() -> EnvState:
    """Minimal EnvState. The reward function ignores its contents."""
    shape = (1, 1)
    return EnvState(
        map=jnp.zeros(shape, dtype=jnp.int8),
        block_resources=jnp.zeros(shape, dtype=jnp.int16),
        machine_types=jnp.zeros(shape, dtype=jnp.int8),
        tile_entity=jnp.full(shape, -1, dtype=jnp.int16),
        ent_y=jnp.full((1,), -1, dtype=jnp.int16),
        ent_x=jnp.full((1,), -1, dtype=jnp.int16),
        ent_type=jnp.zeros((1,), dtype=jnp.int8),
        ent_direction=jnp.zeros((1,), dtype=jnp.int8),
        ent_power=jnp.zeros((1,), dtype=jnp.int16),
        ent_buf_type=jnp.zeros((1,), dtype=jnp.int8),
        ent_buf_count=jnp.zeros((1,), dtype=jnp.int16),
        ent_asm_in_type=jnp.zeros((1, 2), dtype=jnp.int8),
        ent_asm_in_count=jnp.zeros((1, 2), dtype=jnp.int16),
        ent_asm_out_type=jnp.zeros((1,), dtype=jnp.int8),
        ent_asm_out_count=jnp.zeros((1,), dtype=jnp.int16),
        ent_health=jnp.zeros((1,), dtype=jnp.int16),
        player_positions=jnp.zeros((1, 2), dtype=jnp.int16),
        player_directions=jnp.zeros((1,), dtype=jnp.int8),
        player_inventory=jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int16),
        selected_player=jnp.int32(0),
        timestep=jnp.int32(0),
        items_mined=jnp.zeros((NUM_ITEM_TYPES,), dtype=jnp.int32),
        science_consumed_step=jnp.zeros((2,), dtype=jnp.int32),
        achievements_unlocked=jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_),
    )


def test_rocket_reward_basic_unlock() -> None:
    """Unlocking one Basic-tier slot yields +1 reward."""
    prev = _ach_state_with([])
    new = _ach_state_with([_index_of("collect_iron")])
    r = float(rocket_reward(prev, new, EnvParams()))
    assert r == pytest.approx(1.0)


def test_rocket_reward_capstone_unlock() -> None:
    """Unlocking the capstone yields +8 reward."""
    prev = _ach_state_with([])
    new = _ach_state_with([_index_of("place_rocket")])
    r = float(rocket_reward(prev, new, EnvParams()))
    assert r == pytest.approx(8.0)


def test_rocket_reward_no_unlock_is_zero() -> None:
    """Already-latched slots produce no repeat reward."""
    idx = _index_of("collect_coal")
    prev = _ach_state_with([idx])
    new = _ach_state_with([idx])
    r = float(rocket_reward(prev, new, EnvParams()))
    assert r == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# End-to-end: runner populates achievements_unlocked
# ---------------------------------------------------------------------------


# ``test_runner_populates_achievements_end_to_end`` (~9s) was removed
# in the replacement-for-speedup pass. The properties it covered are
# decomposed across cheap unit tests:
#
#   - tests/scenarios/test_runner.py::TestAchievementsAccessor:
#     ``_achievements`` returns numpy of state.achievements_unlocked
#     when a fn is bound, None otherwise.
#   - ``test_rocket_benchmark_exposes_blocked_actions`` above:
#     RocketScenario advertises blocked_actions correctly.
#   - The ``RocketScenario.achievement_fn`` attribute is checked
#     indirectly via the catalogue tests + the runner's _ensure_env
#     pickup logic (TestResolveBlocked + TestBuildEnv in
#     test_runner.py).
#   - ``TestRunnerExecution::test_aggregate_equals_scenario_score``
#     covers aggregate_score finiteness via the shared noop_result.
#
# Chained, those cover the same wiring at sub-second cost.


# -------------------------------------------------------------------------
# The v2 map layout
# -------------------------------------------------------------------------


_MAP_SIZE = 32
_PATCH_SIZE = 2
_PATCHES: list[tuple[int, int, BlockType]] = [
    (3, 9, BlockType.IRON),
    (3, 12, BlockType.COPPER),
    (3, 15, BlockType.TIN),
    (3, 18, BlockType.SILICON),
    (3, 21, BlockType.LIMESTONE),
]


def test_coal_column_fills_full_left_edge() -> None:
    """Every tile on x=0 carries BlockType.COAL with non-zero resources."""
    level = build_rocket_level()
    block_map = np.asarray(level.block_map)
    resources = np.asarray(level.block_resources)
    for y in range(_MAP_SIZE):
        assert int(block_map[y, 0]) == int(BlockType.COAL), f"(0, {y}) is not COAL"
        assert int(resources[y, 0]) > 0, f"(0, {y}) has zero coal resource"


def test_ore_patches_at_expected_v2_coords() -> None:
    """Each 2x2 patch lands on cols 3-4 at the documented top-left row."""
    level = build_rocket_level()
    block_map = np.asarray(level.block_map)
    for top_x, top_y, block in _PATCHES:
        for dx in range(_PATCH_SIZE):
            for dy in range(_PATCH_SIZE):
                tile = (top_x + dx, top_y + dy)
                assert int(block_map[tile[1], tile[0]]) == int(block), (
                    f"{block.name} patch missing at {tile}"
                )


def test_dirt_buffer_between_coal_and_ores() -> None:
    """Cols 1-2 across every patch row are dirt. This is the 2-tile gap."""
    level = build_rocket_level()
    block_map = np.asarray(level.block_map)
    for _, top_y, _ in _PATCHES:
        for dy in range(_PATCH_SIZE):
            for x in (1, 2):
                tile = (x, top_y + dy)
                assert int(block_map[tile[1], tile[0]]) == int(BlockType.DIRT), (
                    f"{tile} should be DIRT, got "
                    f"{BlockType(int(block_map[tile[1], tile[0]])).name}"
                )


def test_pre_placed_furnace_and_assembler_unchanged() -> None:
    """Furnace at (15, 16), assembler at (17, 16), the same as v1."""
    level = build_rocket_level()
    machine_types = np.asarray(level.machine_types)
    machine_directions = np.asarray(level.machine_directions)
    assert int(machine_types[16, 15]) == int(Machine.FURNACE)
    assert int(machine_types[16, 17]) == int(Machine.ASSEMBLER)
    assert int(machine_directions[16, 15]) == int(Direction.DOWN)
    assert int(machine_directions[16, 17]) == int(Direction.DOWN)


def test_player_spawns_at_centre() -> None:
    """Spawn stays at (16, 16) so existing skills' navigation works."""
    level = build_rocket_level()
    spawn = np.asarray(level.player_positions)
    assert spawn.shape[0] >= 1
    assert int(spawn[0, 0]) == 16
    assert int(spawn[0, 1]) == 16


def test_factory_zone_east_of_patches_is_dirt() -> None:
    """Cols 5-31, every row, are all dirt. This is the factory build area.

    Only exception: row 16, cols 15+17 (pre-placed F+A).
    """
    level = build_rocket_level()
    block_map = np.asarray(level.block_map)
    machine_types = np.asarray(level.machine_types)
    for y in range(_MAP_SIZE):
        for x in range(5, _MAP_SIZE):
            block = int(block_map[y, x])
            mt = int(machine_types[y, x])
            assert block == int(BlockType.DIRT), (
                f"({x}, {y}) factory zone should be DIRT, got "
                f"{BlockType(block).name} (machine={Machine(mt).name})"
            )


def test_no_overlap_between_coal_column_and_ore_patches() -> None:
    """The 2-tile dirt gap at cols 1-2 separates them. This is a sanity check."""
    coal_tiles: set[tuple[int, int]] = {(0, y) for y in range(_MAP_SIZE)}
    ore_tiles: set[tuple[int, int]] = set()
    for top_x, top_y, _ in _PATCHES:
        for dx in range(_PATCH_SIZE):
            for dy in range(_PATCH_SIZE):
                ore_tiles.add((top_x + dx, top_y + dy))
    assert coal_tiles.isdisjoint(ore_tiles)
