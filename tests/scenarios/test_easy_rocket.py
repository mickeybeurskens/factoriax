"""Unit tests for the easy-rocket scenario."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.constants import (
    MAX_ACHIEVEMENTS,
    NUM_ITEM_TYPES,
    BlockType,
    ItemType,
    MachineType,
)
from factoriax.levels import Level
from factoriax.recipes import RecipeBook, RecipeTable
from factoriax.scenarios.core import LevelResult, Scenario
from factoriax.scenarios.easy_rocket import (
    EASY_ROCKET_ACHIEVEMENT_WEIGHTS,
    EASY_ROCKET_RECIPE_BOOK,
    EASY_ROCKET_RECIPE_TABLE,
    MAX_EASY_ROCKET_SCORE,
    NUM_EASY_ROCKET_ACHIEVEMENTS,
    EasyRocketScenario,
    build_easy_rocket_level,
    easy_rocket_conditions,
    easy_rocket_reward,
)
from factoriax.state import EnvParams

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
    assert len(EASY_ROCKET_RECIPE_BOOK.recipes) == 8


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


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_build_level_patches_avoid_spawn(seed: int) -> None:
    level = build_easy_rocket_level(jax.random.PRNGKey(seed))
    sx, sy = _SPAWN
    for ty in range(sy - _FORBID_RADIUS, sy + _FORBID_RADIUS + 1):
        for tx in range(sx - _FORBID_RADIUS, sx + _FORBID_RADIUS + 1):
            block = int(level.block_map[ty, tx])
            assert block not in _ORE_BLOCKS, (
                f"Patch overlaps spawn zone at ({tx}, {ty}); seed={seed}"
            )


# Achievement indices in the condition mask, mirroring the spec order.
_A_MINE_1_ORE = 0
_A_MINE_1_OF_EACH = 1
_A_MINE_10_OF_EACH = 2
_A_CRAFT_MINER = 3
_A_CRAFT_ASSEMBLER = 4
_A_CRAFT_BELT = 5
_A_MINER_ON_ORE = 6
_A_FEED_BELT_WITH_MINER = 7
_A_THREE_ORE_TYPES = 8
_A_FEED_ASSEMBLER_BELT = 9
_A_FEED_ASSEMBLER_TWO_BELTS = 10
_A_CONNECT_TWO_ASSEMBLERS = 11
_A_PLACE_ROCKET = 12

_GRAPH_GATED_INDICES = (
    _A_FEED_BELT_WITH_MINER,
    _A_FEED_ASSEMBLER_BELT,
    _A_FEED_ASSEMBLER_TWO_BELTS,
    _A_CONNECT_TWO_ASSEMBLERS,
)


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
    assert not bool(mask[_A_MINE_1_OF_EACH])


def test_conditions_mine_1_of_each_needs_all_six(state_factory) -> None:
    five = state_factory(
        world_map=_dirt_map(),
        player_inventory=_inv(IRON_ORE=1, COPPER_ORE=1, TIN_ORE=1, SILICON=1, COAL=1),
    )
    assert not bool(easy_rocket_conditions(five)[_A_MINE_1_OF_EACH])

    six = state_factory(
        world_map=_dirt_map(),
        player_inventory=_inv(
            IRON_ORE=1, COPPER_ORE=1, TIN_ORE=1, SILICON=1, COAL=1, LIMESTONE=1
        ),
    )
    assert bool(easy_rocket_conditions(six)[_A_MINE_1_OF_EACH])


def test_conditions_mine_10_of_each_uses_miner_craft_ores(state_factory) -> None:
    nine = state_factory(
        world_map=_dirt_map(),
        player_inventory=_inv(IRON_ORE=10, COPPER_ORE=10, TIN_ORE=10, COAL=9),
    )
    assert not bool(easy_rocket_conditions(nine)[_A_MINE_10_OF_EACH])

    ten = state_factory(
        world_map=_dirt_map(),
        player_inventory=_inv(IRON_ORE=10, COPPER_ORE=10, TIN_ORE=10, COAL=10),
    )
    assert bool(easy_rocket_conditions(ten)[_A_MINE_10_OF_EACH])

    # Silicon is not on the miner-craft path; absence does not block #3.
    no_silicon_but_others_ok = state_factory(
        world_map=_dirt_map(),
        player_inventory=_inv(IRON_ORE=10, COPPER_ORE=10, TIN_ORE=10, COAL=10),
    )
    assert bool(easy_rocket_conditions(no_silicon_but_others_ok)[_A_MINE_10_OF_EACH])


@pytest.mark.parametrize(
    "item_name,achievement_idx",
    [
        ("MINER", _A_CRAFT_MINER),
        ("ASSEMBLER", _A_CRAFT_ASSEMBLER),
        ("CONVEYOR_BELT", _A_CRAFT_BELT),
    ],
)
def test_conditions_inventory_machine_unlock(
    state_factory, item_name: str, achievement_idx: int
) -> None:
    inv = _inv(**{item_name: 1})
    state = state_factory(world_map=_dirt_map(), player_inventory=inv)
    assert bool(easy_rocket_conditions(state)[achievement_idx])


def test_conditions_miner_on_ore_unlocks(state_factory) -> None:
    world = jnp.array([[BlockType.IRON]], dtype=jnp.int32)
    mt = jnp.array([[MachineType.MINER]], dtype=jnp.int32)
    state = state_factory(world_map=world, machine_types=mt)
    assert bool(easy_rocket_conditions(state)[_A_MINER_ON_ORE])


def test_conditions_miner_on_dirt_does_not_unlock(state_factory) -> None:
    world = jnp.array([[BlockType.DIRT]], dtype=jnp.int32)
    mt = jnp.array([[MachineType.MINER]], dtype=jnp.int32)
    state = state_factory(world_map=world, machine_types=mt)
    assert not bool(easy_rocket_conditions(state)[_A_MINER_ON_ORE])


def test_conditions_three_ore_types_under_miners(state_factory) -> None:
    world_three = jnp.array(
        [[BlockType.IRON, BlockType.COPPER, BlockType.TIN]],
        dtype=jnp.int32,
    )
    mt_three = jnp.array(
        [[MachineType.MINER, MachineType.MINER, MachineType.MINER]],
        dtype=jnp.int32,
    )
    state_three = state_factory(world_map=world_three, machine_types=mt_three)
    assert bool(easy_rocket_conditions(state_three)[_A_THREE_ORE_TYPES])

    world_two = jnp.array(
        [[BlockType.IRON, BlockType.COPPER]],
        dtype=jnp.int32,
    )
    mt_two = jnp.array(
        [[MachineType.MINER, MachineType.MINER]],
        dtype=jnp.int32,
    )
    state_two = state_factory(world_map=world_two, machine_types=mt_two)
    assert not bool(easy_rocket_conditions(state_two)[_A_THREE_ORE_TYPES])

    # Three miners on the same ore type still does not unlock.
    world_same = jnp.array(
        [[BlockType.IRON, BlockType.IRON, BlockType.IRON]], dtype=jnp.int32
    )
    state_same = state_factory(world_map=world_same, machine_types=mt_three)
    assert not bool(easy_rocket_conditions(state_same)[_A_THREE_ORE_TYPES])


def test_conditions_rocket_placed(state_factory) -> None:
    world = jnp.array([[BlockType.DIRT]], dtype=jnp.int32)
    mt = jnp.array([[MachineType.ROCKET]], dtype=jnp.int32)
    state = state_factory(world_map=world, machine_types=mt)
    assert bool(easy_rocket_conditions(state)[_A_PLACE_ROCKET])


def test_conditions_graph_stubs_always_false(state_factory) -> None:
    # A setup that "looks like" several graph-gated achievements could unlock:
    # a miner adjacent to a belt carrying ore, plus a placed assembler.
    world = jnp.array(
        [[BlockType.IRON, BlockType.DIRT, BlockType.DIRT]], dtype=jnp.int32
    )
    mt = jnp.array(
        [
            [
                MachineType.MINER,
                MachineType.CONVEYOR_BELT,
                MachineType.ASSEMBLER,
            ]
        ],
        dtype=jnp.int32,
    )
    state = state_factory(
        world_map=world,
        machine_types=mt,
        buffer_type=jnp.array([[0, int(ItemType.IRON_ORE), 0]], dtype=jnp.int8),
        buffer_count=jnp.array([[0, 5, 0]], dtype=jnp.int16),
    )
    mask = easy_rocket_conditions(state)
    for idx in _GRAPH_GATED_INDICES:
        assert not bool(mask[idx]), f"graph-gated achievement #{idx} should stay False"


def test_max_easy_rocket_score_is_13() -> None:
    assert MAX_EASY_ROCKET_SCORE == 13.0
    assert float(jnp.sum(EASY_ROCKET_ACHIEVEMENT_WEIGHTS)) == MAX_EASY_ROCKET_SCORE


def test_weights_shape_and_values() -> None:
    assert EASY_ROCKET_ACHIEVEMENT_WEIGHTS.shape == (MAX_ACHIEVEMENTS,)
    assert EASY_ROCKET_ACHIEVEMENT_WEIGHTS.dtype == jnp.float32
    # First 13 slots are 1.0, the rest are 0.0.
    head = EASY_ROCKET_ACHIEVEMENT_WEIGHTS[:NUM_EASY_ROCKET_ACHIEVEMENTS]
    tail = EASY_ROCKET_ACHIEVEMENT_WEIGHTS[NUM_EASY_ROCKET_ACHIEVEMENTS:]
    assert bool(jnp.all(head == 1.0))
    assert bool(jnp.all(tail == 0.0))


def _ach_state(state_factory, mask_indices: list[int]):
    base = state_factory(world_map=_dirt_map())
    mask = jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_)
    if mask_indices:
        mask = mask.at[jnp.array(mask_indices)].set(True)
    return base.replace(achievements_unlocked=mask)


def test_reward_single_unlock_returns_weight(state_factory) -> None:
    prev = _ach_state(state_factory, [])
    new = _ach_state(state_factory, [_A_CRAFT_MINER])
    reward = easy_rocket_reward(prev, new, EnvParams())
    assert float(reward) == 1.0


def test_reward_no_unlock_is_zero(state_factory) -> None:
    same = _ach_state(state_factory, [_A_CRAFT_MINER])
    reward = easy_rocket_reward(same, same, EnvParams())
    assert float(reward) == 0.0


def test_reward_only_counts_newly_unlocked(state_factory) -> None:
    # Already-unlocked bits do not re-fire reward.
    prev = _ach_state(state_factory, [_A_CRAFT_MINER])
    new = _ach_state(state_factory, [_A_CRAFT_MINER, _A_CRAFT_ASSEMBLER])
    reward = easy_rocket_reward(prev, new, EnvParams())
    assert float(reward) == 1.0


def test_scenario_implements_protocol() -> None:
    scenario = EasyRocketScenario()
    assert isinstance(scenario, Scenario)
    assert scenario.name == "easy_rocket"
    assert scenario.num_players == 1
    assert EasyRocketScenario.blocked_actions == frozenset()


def test_scenario_levels_shape() -> None:
    [scenario_level] = EasyRocketScenario().levels()
    params = scenario_level.env_params
    assert params.max_timesteps == 2000
    assert params.map_width == _MAP_SIZE
    assert params.map_height == _MAP_SIZE
    assert params.num_players == 1
    assert params.recipe_table is EASY_ROCKET_RECIPE_TABLE
    assert scenario_level.level.player_positions == [_SPAWN]


def test_scenario_seed_changes_layout() -> None:
    a = EasyRocketScenario(seed=0).levels()[0].level
    b = EasyRocketScenario(seed=1).levels()[0].level
    assert not np.array_equal(a.block_map, b.block_map)


def _level_result_with_mask(mask_indices: list[int]) -> LevelResult:
    mask = np.zeros(MAX_ACHIEVEMENTS, dtype=bool)
    for i in mask_indices:
        mask[i] = True
    return LevelResult(
        level_name="easy_rocket_v1",
        items_mined={},
        weighted_score=0.0,
        timesteps_used=0,
        actions=np.zeros((0,), dtype=np.int32),
        achievements_unlocked=mask,
    )


def test_score_returns_max_with_full_mask() -> None:
    result = _level_result_with_mask(list(range(NUM_EASY_ROCKET_ACHIEVEMENTS)))
    assert EasyRocketScenario().score([result]) == MAX_EASY_ROCKET_SCORE


def test_score_returns_zero_with_no_results() -> None:
    assert EasyRocketScenario().score([]) == 0.0


def test_score_returns_zero_when_mask_missing() -> None:
    result = LevelResult(
        level_name="easy_rocket_v1",
        items_mined={},
        weighted_score=0.0,
        timesteps_used=0,
        actions=np.zeros((0,), dtype=np.int32),
        achievements_unlocked=None,
    )
    assert EasyRocketScenario().score([result]) == 0.0


def test_observation_shape_local_radius_5(state_factory) -> None:
    from factoriax.observations import local_array

    state = state_factory(
        world_map=jnp.full((_MAP_SIZE, _MAP_SIZE), BlockType.DIRT, dtype=jnp.int32),
        player_position=_SPAWN,
    )
    params = EasyRocketScenario().levels()[0].env_params
    obs = local_array(state, params, 0, radius=5)
    assert obs.ndim == 1
    assert obs.dtype == jnp.float32
    # Spatial block: 10 channels × (2r+1)^2 = 10 × 121.
    assert obs.shape[0] > 10 * 121


def test_easy_rocket_public_exports() -> None:
    import factoriax.scenarios as scenarios

    assert scenarios.EasyRocketScenario is EasyRocketScenario
    assert scenarios.EASY_ROCKET_RECIPE_BOOK is EASY_ROCKET_RECIPE_BOOK
    assert scenarios.EASY_ROCKET_RECIPE_TABLE is EASY_ROCKET_RECIPE_TABLE
    assert scenarios.EASY_ROCKET_ACHIEVEMENT_WEIGHTS is EASY_ROCKET_ACHIEVEMENT_WEIGHTS
    assert scenarios.MAX_EASY_ROCKET_SCORE == MAX_EASY_ROCKET_SCORE
    assert scenarios.build_easy_rocket_level is build_easy_rocket_level
    assert scenarios.easy_rocket_conditions is easy_rocket_conditions
    assert scenarios.easy_rocket_reward is easy_rocket_reward
