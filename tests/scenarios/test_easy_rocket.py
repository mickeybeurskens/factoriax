"""Unit tests for the easy-rocket scenario."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.engine.constants import (
    MAX_ACHIEVEMENTS,
    NUM_ITEM_TYPES,
    BlockType,
    ItemType,
    Machine,
)
from factoriax.engine.levels import Level
from factoriax.engine.recipes import RecipeBook, RecipeTable
from factoriax.engine.scenarios.easy_rocket import (
    EASY_ROCKET_ACHIEVEMENT_NAMES,
    EASY_ROCKET_ACHIEVEMENT_WEIGHTS,
    EASY_ROCKET_RECIPE_BOOK,
    EASY_ROCKET_RECIPE_TABLE,
    MAX_EASY_ROCKET_SCORE,
    NUM_EASY_ROCKET_ACHIEVEMENTS,
    build_easy_rocket_level,
    easy_rocket_conditions,
    easy_rocket_reward,
)
from factoriax.engine.state import EnvParams

_SPAWN: tuple[int, int] = (8, 8)
_FORBID_RADIUS: int = 1
_MAP_SIZE: int = 16

_ORE_BLOCKS: frozenset[int] = frozenset(
    {
        int(BlockType.IRON),
        int(BlockType.COPPER),
        int(BlockType.TIN),
        int(BlockType.SILICON),
        int(BlockType.COAL),
        int(BlockType.LIMESTONE),
    }
)


def _levels_equal(a: Level, b: Level) -> bool:
    if (a.name, a.map_width, a.map_height) != (b.name, b.map_width, b.map_height):
        return False
    for field in (
        "block_map",
        "block_resources",
        "machine_types",
        "machine_directions",
        "machine_inventory",
        "machine_selected_recipe",
    ):
        av = getattr(a, field)
        bv = getattr(b, field)
        if (av is None) != (bv is None):
            return False
        if av is not None and not np.array_equal(av, bv):
            return False
    return a.player_positions == b.player_positions


_EXPECTED_OUTPUTS: frozenset[int] = frozenset(
    {
        int(ItemType.ASSEMBLER),
        int(ItemType.CONVEYOR_BELT),
        int(ItemType.SPLITTER),
        int(ItemType.CROSSING),
        int(ItemType.ARM),
        int(ItemType.PALLET),
        int(ItemType.MINER),
        int(ItemType.HULL),
        int(ItemType.ENGINE_UNIT),
        int(ItemType.ROCKET),
    }
)


def test_recipe_book_constructs() -> None:
    assert isinstance(EASY_ROCKET_RECIPE_BOOK, RecipeBook)
    assert isinstance(EASY_ROCKET_RECIPE_TABLE, RecipeTable)

    actual_outputs = frozenset(r.output for r in EASY_ROCKET_RECIPE_BOOK.recipes)
    assert actual_outputs == _EXPECTED_OUTPUTS
    assert len(EASY_ROCKET_RECIPE_BOOK.recipes) == 10


def test_recipe_book_rocket_takes_hull_and_engine_unit() -> None:
    rocket_recipes = [
        r for r in EASY_ROCKET_RECIPE_BOOK.recipes if r.output == int(ItemType.ROCKET)
    ]
    assert len(rocket_recipes) == 1
    rocket = rocket_recipes[0]

    input_items = {item for item, _ in rocket.inputs}
    assert input_items == {int(ItemType.HULL), int(ItemType.ENGINE_UNIT)}

    forbidden = {int(ItemType.AVIONICS), int(ItemType.ROCKET_CORE)}

    output_items = {r.output for r in EASY_ROCKET_RECIPE_BOOK.recipes}
    assert not (output_items & forbidden), (
        "AVIONICS / ROCKET_CORE must not be outputs in the easy-rocket book"
    )

    for recipe in EASY_ROCKET_RECIPE_BOOK.recipes:
        input_set = {item for item, _ in recipe.inputs}
        assert not (input_set & forbidden), (
            f"Recipe {recipe.name!r} references forbidden input {forbidden & input_set}"
        )


def test_build_level_dimensions() -> None:
    level = build_easy_rocket_level(jax.random.PRNGKey(0))
    assert level.map_width == _MAP_SIZE
    assert level.map_height == _MAP_SIZE
    assert level.player_positions == [_SPAWN]
    assert level.machine_types is None


def test_build_level_determinism() -> None:
    key = jax.random.PRNGKey(42)
    assert _levels_equal(build_easy_rocket_level(key), build_easy_rocket_level(key))


def test_build_level_keys_vary() -> None:
    base = build_easy_rocket_level(jax.random.PRNGKey(0))
    found_difference = False
    for seed in (1, 2, 3, 4, 5):
        other = build_easy_rocket_level(jax.random.PRNGKey(seed))
        if not np.array_equal(base.block_map, other.block_map):
            found_difference = True
            break
    assert found_difference


def test_build_level_has_all_ore_types() -> None:
    level = build_easy_rocket_level(jax.random.PRNGKey(7))
    present = {int(b) for b in np.unique(level.block_map).tolist()}
    assert _ORE_BLOCKS.issubset(present)


def test_build_level_patches_avoid_spawn() -> None:
    sx, sy = _SPAWN
    for seed in (0, 1, 2, 3, 4):
        level = build_easy_rocket_level(jax.random.PRNGKey(seed))
        for ty in range(sy - _FORBID_RADIUS, sy + _FORBID_RADIUS + 1):
            for tx in range(sx - _FORBID_RADIUS, sx + _FORBID_RADIUS + 1):
                block = int(level.block_map[ty, tx])
                assert block not in _ORE_BLOCKS, (
                    f"Patch overlaps spawn zone at ({tx}, {ty}); seed={seed}"
                )


# Achievement indices in the condition mask, derived from the engine's
# name list so tests track future bit additions / reorderings without a
# manual update.
def _ach_index(name: str) -> int:
    return EASY_ROCKET_ACHIEVEMENT_NAMES.index(name)


_A_MINE_1_ORE = _ach_index("mine_1_ore")
_A_MINE_3_ORE = _ach_index("mine_3_ore")
_A_PROSPECTOR = _ach_index("prospector")
_A_PLACE_MINER = _ach_index("place_miner")
_A_PLACE_3_MINERS = _ach_index("place_3_miners")
_A_METAL_IN_MOTION = _ach_index("metal_in_motion")
_A_AUTOMATED_MINING = _ach_index("automated_mining")
_A_ORE_FIELDS = _ach_index("ore_fields")
_A_MINING_MASTER = _ach_index("mining_master")
_A_ONE_PALLET = _ach_index("one_pallet")
_A_THREE_PALLETS = _ach_index("three_pallets")
_A_STACKED = _ach_index("stacked")
_A_ASSEMBLER_ONLINE = _ach_index("assembler_online")
_A_ASSEMBLER_AND_ARM = _ach_index("assembler_and_arm")
_A_ONE_BELT = _ach_index("one_belt")
_A_THREE_BELTS = _ach_index("three_belts")
_A_HULL_PRODUCED = _ach_index("hull_production")
_A_TWO_ASSEMBLERS = _ach_index("two_assemblers")
_A_AN_ARM_AND_A_LEG = _ach_index("an_arm_and_a_leg")
_A_SIX_BELTS = _ach_index("six_belts")
_A_ENGINE_PRODUCED = _ach_index("engine_production")
_A_AUTO_BOTS = _ach_index("auto_bots")
_A_BELT_SPAGHETTI = _ach_index("belt_spaghetti")
_A_ROCKET_PRODUCED = _ach_index("rocket_assembled")
_A_LIFTOFF = _ach_index("liftoff")


def _inv(**items: int) -> jnp.ndarray:
    arr = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
    name_to_type = {m.name: int(m) for m in ItemType}
    for name, count in items.items():
        arr = arr.at[0, name_to_type[name]].set(count)
    return arr


def _dirt_map() -> jnp.ndarray:
    return jnp.array([[BlockType.DIRT]], dtype=jnp.int32)


def test_conditions_returns_full_shape(state_factory) -> None:
    mask = easy_rocket_conditions(state_factory(world_map=_dirt_map()))
    assert mask.shape == (MAX_ACHIEVEMENTS,)
    assert mask.dtype == jnp.bool_


def test_conditions_empty_state_all_false(state_factory) -> None:
    mask = easy_rocket_conditions(state_factory(world_map=_dirt_map()))
    assert not bool(jnp.any(mask))


def test_conditions_mine_1_ore_any_ore_unlocks(state_factory) -> None:
    state = state_factory(world_map=_dirt_map(), player_inventory=_inv(IRON_ORE=1))
    mask = easy_rocket_conditions(state)
    assert bool(mask[_A_MINE_1_ORE])
    assert not bool(mask[_A_PROSPECTOR])


def test_conditions_mine_1_of_each_needs_all_six(state_factory) -> None:
    five = state_factory(
        world_map=_dirt_map(),
        player_inventory=_inv(IRON_ORE=1, COPPER_ORE=1, TIN_ORE=1, SILICON=1, COAL=1),
    )
    assert not bool(easy_rocket_conditions(five)[_A_PROSPECTOR])

    six = state_factory(
        world_map=_dirt_map(),
        player_inventory=_inv(
            IRON_ORE=1, COPPER_ORE=1, TIN_ORE=1, SILICON=1, COAL=1, LIMESTONE=1
        ),
    )
    assert bool(easy_rocket_conditions(six)[_A_PROSPECTOR])


_ALL_ORE_BLOCKS: tuple[int, ...] = (
    int(BlockType.IRON),
    int(BlockType.COPPER),
    int(BlockType.TIN),
    int(BlockType.SILICON),
    int(BlockType.COAL),
    int(BlockType.LIMESTONE),
)


def _miner_row(state_factory, ore_blocks, buffer_counts):
    """A one-row map of miners over ``ore_blocks`` with given output buffers."""
    world = jnp.array([list(ore_blocks)], dtype=jnp.int32)
    mt = jnp.array([[Machine.MINER] * len(ore_blocks)], dtype=jnp.int32)
    bc = jnp.array([list(buffer_counts)], dtype=jnp.int16)
    return state_factory(world_map=world, machine_types=mt, buffer_count=bc)


def _assembler(
    state_factory,
    *,
    in_types: tuple[int, int] = (0, 0),
    in_counts: tuple[int, int] = (0, 0),
    out_type: int = 0,
    out_count: int = 0,
    buf_type: int = 0,
    buf_count: int = 0,
):
    """A single placed assembler with the given input / output / buffer slots."""
    return state_factory(
        world_map=_dirt_map(),
        machine_types=jnp.array([[Machine.ASSEMBLER]], dtype=jnp.int32),
        asm_in_type=jnp.array([[list(in_types)]], dtype=jnp.int8),
        asm_in_count=jnp.array([[list(in_counts)]], dtype=jnp.int16),
        asm_out_type=jnp.array([[out_type]], dtype=jnp.int8),
        asm_out_count=jnp.array([[out_count]], dtype=jnp.int16),
        buffer_type=jnp.array([[buf_type]], dtype=jnp.int8),
        buffer_count=jnp.array([[buf_count]], dtype=jnp.int16),
    )


def test_place_miner_unlocks_on_placed_machine(state_factory) -> None:
    state = state_factory(
        world_map=_dirt_map(),
        machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
    )
    assert bool(easy_rocket_conditions(state)[_A_PLACE_MINER])


def test_place_miner_does_not_fire_from_inventory(state_factory) -> None:
    # Inventory miner alone (no placed machine) must not flip the
    # placement-checking bit; this pins the inventory-vs-placement split.
    state = state_factory(world_map=_dirt_map(), player_inventory=_inv(MINER=1))
    assert not bool(easy_rocket_conditions(state)[_A_PLACE_MINER])


def test_automated_mining_needs_producing_miner(state_factory) -> None:
    blocks = (int(BlockType.IRON),)
    # Placed on ore but empty buffer -> not yet producing.
    idle = _miner_row(state_factory, blocks, (0,))
    assert not bool(easy_rocket_conditions(idle)[_A_AUTOMATED_MINING])
    # Output buffer non-empty -> producing.
    producing = _miner_row(state_factory, blocks, (3,))
    assert bool(easy_rocket_conditions(producing)[_A_AUTOMATED_MINING])


def test_ore_fields_needs_three_distinct_producing(state_factory) -> None:
    blocks = (int(BlockType.IRON), int(BlockType.COPPER), int(BlockType.TIN))
    assert bool(
        easy_rocket_conditions(_miner_row(state_factory, blocks, (1, 1, 1)))[
            _A_ORE_FIELDS
        ]
    )
    # Three distinct blocks but none producing.
    assert not bool(
        easy_rocket_conditions(_miner_row(state_factory, blocks, (0, 0, 0)))[
            _A_ORE_FIELDS
        ]
    )
    # Only two producing.
    assert not bool(
        easy_rocket_conditions(_miner_row(state_factory, blocks, (1, 1, 0)))[
            _A_ORE_FIELDS
        ]
    )
    # Three producing miners but all on the same ore block.
    same = (int(BlockType.IRON),) * 3
    assert not bool(
        easy_rocket_conditions(_miner_row(state_factory, same, (1, 1, 1)))[
            _A_ORE_FIELDS
        ]
    )


def test_mining_master_needs_all_six_producing(state_factory) -> None:
    all_producing = _miner_row(state_factory, _ALL_ORE_BLOCKS, (1,) * 6)
    assert bool(easy_rocket_conditions(all_producing)[_A_MINING_MASTER])
    # Five producing, the sixth idle -> not complete.
    five = _miner_row(state_factory, _ALL_ORE_BLOCKS, (1, 1, 1, 1, 1, 0))
    assert not bool(easy_rocket_conditions(five)[_A_MINING_MASTER])


def test_assembler_online_counts_placed_assembler(state_factory) -> None:
    placed = state_factory(
        world_map=_dirt_map(),
        machine_types=jnp.array([[Machine.ASSEMBLER]], dtype=jnp.int32),
    )
    assert bool(easy_rocket_conditions(placed)[_A_ASSEMBLER_ONLINE])
    empty = state_factory(world_map=_dirt_map())
    assert not bool(easy_rocket_conditions(empty)[_A_ASSEMBLER_ONLINE])


@pytest.mark.parametrize(
    "item,produced_idx",
    [
        (ItemType.HULL, _A_HULL_PRODUCED),
        (ItemType.ENGINE_UNIT, _A_ENGINE_PRODUCED),
        (ItemType.ROCKET, _A_ROCKET_PRODUCED),
    ],
)
def test_assembler_produced_via_output_slot(
    state_factory, item, produced_idx: int
) -> None:
    state = _assembler(state_factory, out_type=int(item), out_count=1)
    assert bool(easy_rocket_conditions(state)[produced_idx])


@pytest.mark.parametrize(
    "item,produced_idx",
    [
        (ItemType.HULL, _A_HULL_PRODUCED),
        (ItemType.ENGINE_UNIT, _A_ENGINE_PRODUCED),
        (ItemType.ROCKET, _A_ROCKET_PRODUCED),
    ],
)
def test_assembler_produced_via_buffer_drain(
    state_factory, item, produced_idx: int
) -> None:
    # The engine drains a finished output into the buffer on the next tick.
    state = _assembler(state_factory, buf_type=int(item), buf_count=1)
    assert bool(easy_rocket_conditions(state)[produced_idx])


def test_liftoff_needs_placed_rocket(state_factory) -> None:
    placed = state_factory(
        world_map=_dirt_map(),
        machine_types=jnp.array([[Machine.ROCKET]], dtype=jnp.int32),
    )
    assert bool(easy_rocket_conditions(placed)[_A_LIFTOFF])


def test_inventory_does_not_unlock_machine_bits(state_factory) -> None:
    # Hand-held section items must not flip any machine-sourced bit; only
    # populated machine buffers (or placed machines) do. This pins the
    # hand-vs-machine invariant the production curriculum relies on.
    inv = _inv(
        MINER=1,
        ASSEMBLER=1,
        ROCKET=1,
        HULL=200,
        ENGINE_UNIT=200,
        IRON_ORE=50,
        COPPER_ORE=50,
        TIN_ORE=50,
        SILICON=50,
        COAL=50,
        LIMESTONE=50,
    )
    mask = easy_rocket_conditions(
        state_factory(world_map=_dirt_map(), player_inventory=inv)
    )
    machine_sourced = (
        _A_PLACE_MINER,
        _A_PLACE_3_MINERS,
        _A_METAL_IN_MOTION,
        _A_AUTOMATED_MINING,
        _A_ORE_FIELDS,
        _A_MINING_MASTER,
        _A_ONE_PALLET,
        _A_THREE_PALLETS,
        _A_STACKED,
        _A_ASSEMBLER_ONLINE,
        _A_ASSEMBLER_AND_ARM,
        _A_ONE_BELT,
        _A_THREE_BELTS,
        _A_HULL_PRODUCED,
        _A_TWO_ASSEMBLERS,
        _A_AN_ARM_AND_A_LEG,
        _A_SIX_BELTS,
        _A_ENGINE_PRODUCED,
        _A_AUTO_BOTS,
        _A_BELT_SPAGHETTI,
        _A_ROCKET_PRODUCED,
        _A_LIFTOFF,
    )
    for idx in machine_sourced:
        assert not bool(mask[idx]), f"bit {idx} wrongly flipped from player inventory"
    # The inventory-sourced bits, by contrast, do fire from inventory alone.
    assert bool(mask[_A_MINE_1_ORE])
    assert bool(mask[_A_MINE_3_ORE])
    assert bool(mask[_A_PROSPECTOR])


def test_weights_and_max_score() -> None:
    assert EASY_ROCKET_ACHIEVEMENT_WEIGHTS.shape == (MAX_ACHIEVEMENTS,)
    assert EASY_ROCKET_ACHIEVEMENT_WEIGHTS.dtype == jnp.float32
    head = EASY_ROCKET_ACHIEVEMENT_WEIGHTS[:NUM_EASY_ROCKET_ACHIEVEMENTS]
    tail = EASY_ROCKET_ACHIEVEMENT_WEIGHTS[NUM_EASY_ROCKET_ACHIEVEMENTS:]
    assert bool(jnp.all(head == 1.0))
    assert bool(jnp.all(tail == 0.0))
    assert MAX_EASY_ROCKET_SCORE == float(NUM_EASY_ROCKET_ACHIEVEMENTS)
    assert float(jnp.sum(EASY_ROCKET_ACHIEVEMENT_WEIGHTS)) == MAX_EASY_ROCKET_SCORE


def _ach_state(state_factory, mask_indices: list[int]):
    base = state_factory(world_map=_dirt_map())
    mask = jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_)
    if mask_indices:
        mask = mask.at[jnp.array(mask_indices)].set(True)
    return base.replace(achievements_unlocked=mask)


def test_reward_single_unlock_returns_weight(state_factory) -> None:
    prev = _ach_state(state_factory, [])
    new = _ach_state(state_factory, [_A_PLACE_MINER])
    reward = easy_rocket_reward(prev, new, EnvParams())
    assert float(reward) == 1.0


def test_reward_no_unlock_is_zero(state_factory) -> None:
    same = _ach_state(state_factory, [_A_PLACE_MINER])
    reward = easy_rocket_reward(same, same, EnvParams())
    assert float(reward) == 0.0


def test_reward_only_counts_newly_unlocked(state_factory) -> None:
    # Already-unlocked bits do not re-fire reward.
    prev = _ach_state(state_factory, [_A_PLACE_MINER])
    new = _ach_state(state_factory, [_A_PLACE_MINER, _A_ASSEMBLER_ONLINE])
    reward = easy_rocket_reward(prev, new, EnvParams())
    assert float(reward) == 1.0


def test_easy_rocket_factory_builds_env() -> None:
    """The registry factory yields a steppable env at the scenario's params."""
    import factoriax

    env, params = factoriax.make("EasyRocket-v1")
    assert params.map_width == _MAP_SIZE and params.map_height == _MAP_SIZE
    assert params.max_timesteps == 2000
    assert params.recipe_table is EASY_ROCKET_RECIPE_TABLE
    assert MAX_EASY_ROCKET_SCORE == float(NUM_EASY_ROCKET_ACHIEVEMENTS)
