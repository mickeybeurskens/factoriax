"""Tests for science packs, research, and tech gating."""

import jax.numpy as jnp
from jax import random

from factoriax.constants import (
    NUM_TECHNOLOGIES,
    RESEARCH_COST,
    Action,
    ItemType,
    MachineType,
)
from factoriax.game_logic import apply_research, factoriax_step
from factoriax.machines import run_assemblers
from factoriax.state import EnvParams
from factoriax.world_gen import generate_world


def _make_assembler_state(
    recipe: int,
    slot0_item: int = 0,
    slot0_count: int = 0,
    slot1_item: int = 0,
    slot1_count: int = 0,
    research_unlocked: jnp.ndarray | None = None,
) -> object:
    """Create a state with an assembler at (0,0) and given inputs.

    Args:
        recipe: Assembler recipe index.
        slot0_item: Item in input slot 0.
        slot0_count: Count in input slot 0.
        slot1_item: Item in input slot 1.
        slot1_count: Count in input slot 1.
        research_unlocked: Boolean array of unlocked techs.

    Returns:
        Configured EnvState.
    """
    rng = random.PRNGKey(0)
    params = EnvParams(map_width=4, map_height=4, num_players=1)
    state = generate_world(rng, params)

    if research_unlocked is None:
        research_unlocked = jnp.zeros(NUM_TECHNOLOGIES, dtype=jnp.bool_)

    state = state.replace(
        machine_types=state.machine_types.at[0, 0].set(MachineType.ASSEMBLER),
        machine_selected_recipe=state.machine_selected_recipe.at[0, 0].set(
            recipe
        ),
        research_unlocked=research_unlocked,
    )
    inv_items = state.machine_inventory_items
    inv_counts = state.machine_inventory_counts
    inv_items = inv_items.at[0, 0, 0].set(slot0_item)
    inv_counts = inv_counts.at[0, 0, 0].set(slot0_count)
    inv_items = inv_items.at[0, 0, 1].set(slot1_item)
    inv_counts = inv_counts.at[0, 0, 1].set(slot1_count)
    return state.replace(
        machine_inventory_items=inv_items,
        machine_inventory_counts=inv_counts,
    )


class TestSciencePackRecipes:
    """Assemblers should produce science packs from correct inputs."""

    def test_basic_science_pack_crafts(self) -> None:
        """Basic science pack: 1 iron + 1 copper -> starts craft."""
        unlocked = jnp.ones(NUM_TECHNOLOGIES, dtype=jnp.bool_)
        state = _make_assembler_state(
            recipe=3,
            slot0_item=int(ItemType.IRON),
            slot0_count=5,
            slot1_item=int(ItemType.COPPER),
            slot1_count=5,
            research_unlocked=unlocked,
        )
        new = run_assemblers(state)
        assert int(new.machine_power[0, 0]) == 4
        assert int(new.machine_inventory_counts[0, 0, 0]) == 4
        assert int(new.machine_inventory_counts[0, 0, 1]) == 4

    def test_fuel_science_pack_crafts(self) -> None:
        """Fuel science pack: 1 iron + 1 coal -> starts craft."""
        unlocked = jnp.ones(NUM_TECHNOLOGIES, dtype=jnp.bool_)
        state = _make_assembler_state(
            recipe=4,
            slot0_item=int(ItemType.IRON),
            slot0_count=5,
            slot1_item=int(ItemType.COAL),
            slot1_count=5,
            research_unlocked=unlocked,
        )
        new = run_assemblers(state)
        assert int(new.machine_power[0, 0]) == 4
        assert int(new.machine_inventory_counts[0, 0, 0]) == 4
        assert int(new.machine_inventory_counts[0, 0, 1]) == 4

    def test_advanced_science_pack_crafts(self) -> None:
        """Advanced science pack: 1 hull + 1 fuel pack -> starts craft."""
        unlocked = jnp.ones(NUM_TECHNOLOGIES, dtype=jnp.bool_)
        state = _make_assembler_state(
            recipe=5,
            slot0_item=int(ItemType.HULL),
            slot0_count=3,
            slot1_item=int(ItemType.FUEL_PACK),
            slot1_count=3,
            research_unlocked=unlocked,
        )
        new = run_assemblers(state)
        assert int(new.machine_power[0, 0]) == 8
        assert int(new.machine_inventory_counts[0, 0, 0]) == 2
        assert int(new.machine_inventory_counts[0, 0, 1]) == 2


class TestTechGating:
    """Assembler recipes gated by research should not start until unlocked."""

    def test_hull_blocked_without_research(self) -> None:
        """Hull recipe should not start when basic science is not researched."""
        state = _make_assembler_state(
            recipe=0,
            slot0_item=int(ItemType.IRON),
            slot0_count=10,
        )
        new = run_assemblers(state)
        assert int(new.machine_power[0, 0]) == 0
        assert int(new.machine_inventory_counts[0, 0, 0]) == 10

    def test_hull_starts_when_researched(self) -> None:
        """Hull recipe should start when basic science tech is unlocked."""
        unlocked = jnp.zeros(NUM_TECHNOLOGIES, dtype=jnp.bool_).at[0].set(
            True
        )
        state = _make_assembler_state(
            recipe=0,
            slot0_item=int(ItemType.IRON),
            slot0_count=10,
            research_unlocked=unlocked,
        )
        new = run_assemblers(state)
        assert int(new.machine_power[0, 0]) == 4
        assert int(new.machine_inventory_counts[0, 0, 0]) == 5

    def test_fuel_pack_blocked_without_research(self) -> None:
        """Fuel pack recipe should not start without fuel science research."""
        state = _make_assembler_state(
            recipe=1,
            slot0_item=int(ItemType.COPPER),
            slot0_count=10,
            slot1_item=int(ItemType.COAL),
            slot1_count=10,
        )
        new = run_assemblers(state)
        assert int(new.machine_power[0, 0]) == 0

    def test_fuel_pack_starts_when_researched(self) -> None:
        """Fuel pack recipe should start when fuel science tech is unlocked."""
        unlocked = jnp.zeros(NUM_TECHNOLOGIES, dtype=jnp.bool_).at[1].set(
            True
        )
        state = _make_assembler_state(
            recipe=1,
            slot0_item=int(ItemType.COPPER),
            slot0_count=10,
            slot1_item=int(ItemType.COAL),
            slot1_count=10,
            research_unlocked=unlocked,
        )
        new = run_assemblers(state)
        assert int(new.machine_power[0, 0]) == 6

    def test_science_packs_ungated(self) -> None:
        """Science pack recipes should work without any research."""
        state = _make_assembler_state(
            recipe=3,
            slot0_item=int(ItemType.IRON),
            slot0_count=5,
            slot1_item=int(ItemType.COPPER),
            slot1_count=5,
        )
        new = run_assemblers(state)
        assert int(new.machine_power[0, 0]) == 4


class TestResearchAction:
    """The RESEARCH action should consume science packs and unlock techs."""

    def test_consume_basic_science_pack(self, state_factory) -> None:
        """Consuming a basic science pack increments research progress."""
        inv_items = jnp.zeros((1, 10), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(int(ItemType.BASIC_SCIENCE_PACK))
        inv_counts = jnp.zeros((1, 10), dtype=jnp.int16)
        inv_counts = inv_counts.at[0, 0].set(5)

        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        new = apply_research(state, 0)
        assert int(new.research_progress[0]) == 1
        assert int(new.inventory_counts[0, 0]) == 4
        assert not bool(new.research_unlocked[0])

    def test_research_unlocks_at_threshold(self, state_factory) -> None:
        """Tech should unlock when progress reaches RESEARCH_COST."""
        inv_items = jnp.zeros((1, 10), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(int(ItemType.BASIC_SCIENCE_PACK))
        inv_counts = jnp.zeros((1, 10), dtype=jnp.int16)
        inv_counts = inv_counts.at[0, 0].set(RESEARCH_COST)

        progress = jnp.array([RESEARCH_COST - 1, 0], dtype=jnp.int32)

        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            research_progress=progress,
        )
        new = apply_research(state, 0)
        assert bool(new.research_unlocked[0])
        assert int(new.research_progress[0]) == RESEARCH_COST

    def test_no_consume_without_science_pack(self, state_factory) -> None:
        """RESEARCH with non-science item should be a no-op."""
        inv_items = jnp.zeros((1, 10), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(int(ItemType.IRON))
        inv_counts = jnp.zeros((1, 10), dtype=jnp.int16)
        inv_counts = inv_counts.at[0, 0].set(5)

        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        new = apply_research(state, 0)
        assert int(new.research_progress[0]) == 0
        assert int(new.inventory_counts[0, 0]) == 5

    def test_no_consume_when_already_unlocked(self, state_factory) -> None:
        """RESEARCH should not consume packs for an already-unlocked tech."""
        inv_items = jnp.zeros((1, 10), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(int(ItemType.BASIC_SCIENCE_PACK))
        inv_counts = jnp.zeros((1, 10), dtype=jnp.int16)
        inv_counts = inv_counts.at[0, 0].set(5)

        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
            research_unlocked=jnp.array([True, False], dtype=jnp.bool_),
        )
        new = apply_research(state, 0)
        assert int(new.inventory_counts[0, 0]) == 5

    def test_fuel_science_unlocks_tech_1(self, state_factory) -> None:
        """Fuel science packs should advance tech index 1."""
        inv_items = jnp.zeros((1, 10), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(int(ItemType.FUEL_SCIENCE_PACK))
        inv_counts = jnp.zeros((1, 10), dtype=jnp.int16)
        inv_counts = inv_counts.at[0, 0].set(1)

        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        new = apply_research(state, 0)
        assert int(new.research_progress[1]) == 1
        assert int(new.research_progress[0]) == 0

    def test_research_via_step(self, state_factory) -> None:
        """RESEARCH action through factoriax_step should work."""
        inv_items = jnp.zeros((1, 10), dtype=jnp.int32)
        inv_items = inv_items.at[0, 0].set(int(ItemType.BASIC_SCIENCE_PACK))
        inv_counts = jnp.zeros((1, 10), dtype=jnp.int16)
        inv_counts = inv_counts.at[0, 0].set(3)

        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
            inventory_items=inv_items,
            inventory_counts=inv_counts,
        )
        params = EnvParams(map_width=4, map_height=4, num_players=1)
        rng = random.PRNGKey(0)
        new = factoriax_step(rng, state, int(Action.RESEARCH), params)
        assert int(new.research_progress[0]) == 1
        assert int(new.inventory_counts[0, 0]) == 2


# Need BlockType for state_factory calls.
from factoriax.constants import BlockType  # noqa: E402
