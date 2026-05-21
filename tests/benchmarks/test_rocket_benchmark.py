"""Tests for the rocket benchmark.

Covers the achievement catalogue, per-condition correctness, reward
semantics, and a small end-to-end run through the benchmark runner.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.benchmarks import (
    MAX_ROCKET_SCORE,
    ROCKET_ACHIEVEMENT_INFO,
    ROCKET_ACHIEVEMENT_WEIGHTS,
    LevelResult,
    RocketBenchmark,
    build_rocket_level,
    rocket_conditions,
    rocket_reward,
)
from factoriax.constants import (
    MAX_ACHIEVEMENTS,
    NUM_ITEM_TYPES,
    Action,
    BlockType,
    ItemType,
    MachineType,
)
from factoriax.state import EnvParams, EnvState

# ---------------------------------------------------------------------------
# Catalogue + construction
# ---------------------------------------------------------------------------


def test_catalogue_size_and_weights() -> None:
    """The catalogue has 38 entries and tiered weights summing to 140."""
    assert len(ROCKET_ACHIEVEMENT_INFO) == 38
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
    ids = [info.id for info in ROCKET_ACHIEVEMENT_INFO]
    assert len(set(ids)) == len(ids)


def test_benchmark_construction() -> None:
    """RocketBenchmark has one level at the advertised dimensions."""
    bench = RocketBenchmark()
    assert bench.name == "rocket"
    assert bench.num_players == 1
    levels = bench.levels()
    assert len(levels) == 1
    params = levels[0].env_params
    assert params.max_timesteps == 8000
    assert params.map_width == 32
    assert params.map_height == 32
    assert params.num_players == 1


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
    assert int(mt[16, 15]) == int(MachineType.FURNACE)
    assert int(mt[16, 17]) == int(MachineType.ASSEMBLER)


def test_rocket_benchmark_exposes_blocked_actions() -> None:
    """The benchmark advertises every CRAFT_* action as blocked."""
    from factoriax.benchmarks.rocket import ROCKET_BLOCKED_ACTIONS

    bench = RocketBenchmark()
    # All 18 CRAFT_* actions (IRON_PLATE .. ROCKET) must be blocked.
    assert len(bench.blocked_actions) == 18
    assert Action.CRAFT_IRON_PLATE in bench.blocked_actions
    assert Action.CRAFT_ROCKET in bench.blocked_actions
    assert Action.CRAFT_BASIC_SCIENCE in bench.blocked_actions
    # Movement / mining / placement actions must NOT be blocked.
    for allowed in (
        Action.NOOP,
        Action.MINE,
        Action.UP,
        Action.PLACE_MINER,
        Action.WITHDRAW,
        Action.DEPOSIT_COAL,
    ):
        assert int(allowed) not in bench.blocked_actions
    assert bench.blocked_actions == ROCKET_BLOCKED_ACTIONS


def test_action_mask_wrapper_noops_blocked_actions() -> None:
    """ActionMaskWrapper silently converts blocked actions to NOOP."""
    from factoriax.benchmarks.rocket import ROCKET_BLOCKED_ACTIONS
    from factoriax.envs import FactoriaXEnv
    from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
    from factoriax.levels import build_state

    env = ActionMaskWrapper(FactoriaXEnv(), ROCKET_BLOCKED_ACTIONS)
    level = build_rocket_level()
    params = EnvParams(map_width=32, map_height=32, num_players=1, max_timesteps=10)
    state = build_state(level, params)
    key = jax.random.PRNGKey(0)

    # Emitting a masked CRAFT_IRON_PLATE should behave exactly like NOOP
    # — inventory unchanged.
    inv_before = np.asarray(state.player_inventory[0])
    _, new_state, _, _, _ = env.step_env(
        key,
        state,
        jnp.int32(int(Action.CRAFT_IRON_PLATE)),
        params,
    )
    inv_after = np.asarray(new_state.player_inventory[0])
    np.testing.assert_array_equal(inv_before, inv_after)


# ---------------------------------------------------------------------------
# Per-condition unit tests
# ---------------------------------------------------------------------------


def _index_of(achievement_id: str) -> int:
    for i, info in enumerate(ROCKET_ACHIEVEMENT_INFO):
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
    """Minimal 1x1 empty map for states that don't need terrain."""
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
    # Every other item-holding achievement should remain False.
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
        ("place_miner", jnp.array([[MachineType.MINER]], dtype=jnp.int32)),
        ("place_furnace", jnp.array([[MachineType.FURNACE]], dtype=jnp.int32)),
        (
            "place_belt",
            jnp.array([[MachineType.CONVEYOR_BELT]], dtype=jnp.int32),
        ),
        ("place_pallet", jnp.array([[MachineType.PALLET]], dtype=jnp.int32)),
        ("place_arm", jnp.array([[MachineType.ARM]], dtype=jnp.int32)),
        (
            "place_assembler",
            jnp.array([[MachineType.ASSEMBLER]], dtype=jnp.int32),
        ),
        ("place_rocket", jnp.array([[MachineType.ROCKET]], dtype=jnp.int32)),
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
    # 1x4 strip of belts — below threshold.
    belts_4 = jnp.full((1, 4), MachineType.CONVEYOR_BELT, dtype=jnp.int32)
    world_4 = jnp.full((1, 4), BlockType.DIRT, dtype=jnp.int32)
    state_4 = state_factory(world_map=world_4, machine_types=belts_4)
    assert not bool(rocket_conditions(state_4)[_index_of("belt_network")])
    # 1x5 strip — at threshold.
    belts_5 = jnp.full((1, 5), MachineType.CONVEYOR_BELT, dtype=jnp.int32)
    world_5 = jnp.full((1, 5), BlockType.DIRT, dtype=jnp.int32)
    state_5 = state_factory(world_map=world_5, machine_types=belts_5)
    assert bool(rocket_conditions(state_5)[_index_of("belt_network")])


def test_scaling_up_requires_three_miners(state_factory) -> None:
    """scaling_up fires at 3 miners, not at 2."""
    miners_2 = jnp.full((1, 2), MachineType.MINER, dtype=jnp.int32)
    world_2 = jnp.full((1, 2), BlockType.DIRT, dtype=jnp.int32)
    state_2 = state_factory(world_map=world_2, machine_types=miners_2)
    assert not bool(rocket_conditions(state_2)[_index_of("scaling_up")])
    miners_3 = jnp.full((1, 3), MachineType.MINER, dtype=jnp.int32)
    world_3 = jnp.full((1, 3), BlockType.DIRT, dtype=jnp.int32)
    state_3 = state_factory(world_map=world_3, machine_types=miners_3)
    assert bool(rocket_conditions(state_3)[_index_of("scaling_up")])


def test_industrialist_requires_ten_machines(state_factory) -> None:
    """industrialist fires at 10+ machines of any type."""
    # 2x5 mixed grid: 10 machines.
    grid = jnp.array(
        [
            [MachineType.MINER] * 5,
            [MachineType.PALLET] * 5,
        ],
        dtype=jnp.int32,
    )
    world = jnp.full((2, 5), BlockType.DIRT, dtype=jnp.int32)
    state = state_factory(world_map=world, machine_types=grid)
    assert bool(rocket_conditions(state)[_index_of("industrialist")])


def test_automated_mining_requires_buffered_ore(state_factory) -> None:
    """automated_mining fires when a placed miner has buffer items."""
    world = jnp.array([[BlockType.DIRT]], dtype=jnp.int32)
    mt = jnp.array([[MachineType.MINER]], dtype=jnp.int32)
    # Empty buffer — should NOT fire.
    state_empty = state_factory(world_map=world, machine_types=mt)
    assert not bool(rocket_conditions(state_empty)[_index_of("automated_mining")])
    # Non-empty buffer — should fire.
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
    mt = jnp.array([[MachineType.PALLET]], dtype=jnp.int32)
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
    mt = jnp.array([[MachineType.ASSEMBLER]], dtype=jnp.int32)
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
    # The other state fields don't matter for achievement_reward — it
    # reads only ``achievements_unlocked``.
    unlocked = jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_)
    for i in mask_indices:
        unlocked = unlocked.at[i].set(True)
    return _dummy_env_state().replace(achievements_unlocked=unlocked)


def _dummy_env_state() -> EnvState:
    """Minimal EnvState — the reward function ignores its contents."""
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


def test_score_reads_achievement_mask() -> None:
    """score() weights each slot in the result mask correctly."""
    bench = RocketBenchmark()
    mask = np.zeros(MAX_ACHIEVEMENTS, dtype=bool)
    mask[_index_of("collect_iron")] = True  # +1
    mask[_index_of("place_rocket")] = True  # +8
    result = LevelResult(
        level_name="rocket_v1",
        items_mined={"coal": 0, "iron": 0, "copper": 0},
        weighted_score=0.0,
        timesteps_used=0,
        actions=np.zeros((0,), dtype=np.int32),
        achievements_unlocked=mask,
    )
    assert bench.score([result]) == pytest.approx(9.0)


def test_score_without_mask_returns_zero() -> None:
    """Defensive: missing achievements mask scores as zero."""
    bench = RocketBenchmark()
    result = LevelResult(
        level_name="rocket_v1",
        items_mined={"coal": 0, "iron": 0, "copper": 0},
        weighted_score=0.0,
        timesteps_used=0,
        actions=np.zeros((0,), dtype=np.int32),
        achievements_unlocked=None,
    )
    assert bench.score([result]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# End-to-end: runner populates achievements_unlocked
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_runner_populates_achievements_end_to_end(runner) -> None:
    """A random policy run surfaces a non-None achievements mask.

    Uses the shared session-scoped ``runner`` fixture (from
    ``tests/benchmarks/conftest.py``) so the BenchmarkRunner
    constructor's no-achievement-fn compile is reused. The runner's
    multi-entry cache (Task 2.10) absorbs the rocket_conditions
    achievement_fn config without trashing the no-fn entry. Only one
    new compile this test pays for: rocket_conditions + ROCKET_BLOCKED.
    """

    def random_policy(obs: jax.Array) -> jax.Array:
        # Constant NOOP — simplest possible policy. The point of this
        # test is that the runner wires up the wrapper, not that the
        # random policy unlocks anything.
        return jnp.int32(0)

    bench = RocketBenchmark()
    # Override max_timesteps so JIT compile stays cheap.
    short = bench.levels()[0]
    short_params = short.env_params.replace(max_timesteps=20)

    class _ShortRocket(RocketBenchmark):
        def levels(self) -> list:
            return [
                short.__class__(
                    name=short.name,
                    description=short.description,
                    level=short.level,
                    env_params=short_params,
                )
            ]

    result = runner.run(_ShortRocket(), [random_policy])
    assert result.level_results[0].achievements_unlocked is not None
    assert result.level_results[0].achievements_unlocked.shape == (MAX_ACHIEVEMENTS,)
    # Aggregate score is a finite non-negative number.
    assert result.aggregate_score >= 0.0
    assert np.isfinite(result.aggregate_score)
