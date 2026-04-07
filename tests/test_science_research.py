"""Tests for science packs, research, and tech gating.

Uses the pouch inventory model where player_inventory has shape
(P, NUM_ITEM_TYPES) and machine_inventory has shape (H, W, NUM_ITEM_TYPES).
"""

import jax.numpy as jnp
from jax import random

from factoriax.constants import (
    DEFAULT_MACHINE_MAX_HEALTH,
    MACHINE_INVENTORY_COUNT_DTYPE,
    NUM_ITEM_TYPES,
    NUM_TECHNOLOGIES,
    RESEARCH_COST,
    Action,
    BlockType,
    ItemType,
    MachineType,
)
from factoriax.game_logic import apply_research, factoriax_step
from factoriax.machines import run_assemblers
from factoriax.state import EnvParams
from factoriax.world_gen import generate_world


def _make_assembler_state(
    recipe: int,
    inv_entries: dict[int, int] | None = None,
    research_unlocked: jnp.ndarray | None = None,
) -> object:
    """Create a state with an assembler at (0,0) and given inputs.

    Args:
        recipe: Assembler recipe index.
        inv_entries: Mapping of item_type -> count for the assembler.
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
            recipe,
        ),
        machine_health=state.machine_health.at[0, 0].set(
            DEFAULT_MACHINE_MAX_HEALTH,
        ),
        research_unlocked=research_unlocked,
    )
    inv = state.machine_inventory.astype(jnp.int32)
    for item_type, count in (inv_entries or {}).items():
        inv = inv.at[0, 0, item_type].set(count)
    return state.replace(
        machine_inventory=inv.astype(MACHINE_INVENTORY_COUNT_DTYPE),
    )


class TestSciencePackRecipes:
    """Assemblers should produce science packs from correct inputs."""

    def test_basic_science_pack_crafts(self) -> None:
        """Basic science pack: 1 iron + 1 copper -> starts craft."""
        unlocked = jnp.ones(NUM_TECHNOLOGIES, dtype=jnp.bool_)
        state = _make_assembler_state(
            recipe=3,
            inv_entries={
                int(ItemType.IRON): 5,
                int(ItemType.COPPER): 5,
            },
            research_unlocked=unlocked,
        )
        new = run_assemblers(state)
        assert int(new.machine_power[0, 0]) == 4
        assert int(new.machine_inventory[0, 0, ItemType.IRON]) == 4
        assert int(new.machine_inventory[0, 0, ItemType.COPPER]) == 4

    def test_fuel_science_pack_crafts(self) -> None:
        """Fuel science pack: 1 iron + 1 coal -> starts craft."""
        unlocked = jnp.ones(NUM_TECHNOLOGIES, dtype=jnp.bool_)
        state = _make_assembler_state(
            recipe=4,
            inv_entries={
                int(ItemType.IRON): 5,
                int(ItemType.COAL): 5,
            },
            research_unlocked=unlocked,
        )
        new = run_assemblers(state)
        assert int(new.machine_power[0, 0]) == 4
        assert int(new.machine_inventory[0, 0, ItemType.IRON]) == 4
        assert int(new.machine_inventory[0, 0, ItemType.COAL]) == 4

    def test_advanced_science_pack_crafts(self) -> None:
        """Advanced science pack: 1 hull + 1 fuel pack -> starts craft."""
        unlocked = jnp.ones(NUM_TECHNOLOGIES, dtype=jnp.bool_)
        state = _make_assembler_state(
            recipe=5,
            inv_entries={
                int(ItemType.HULL): 3,
                int(ItemType.FUEL_PACK): 3,
            },
            research_unlocked=unlocked,
        )
        new = run_assemblers(state)
        assert int(new.machine_power[0, 0]) == 8
        assert int(new.machine_inventory[0, 0, ItemType.HULL]) == 2
        assert int(new.machine_inventory[0, 0, ItemType.FUEL_PACK]) == 2


class TestTechGating:
    """Assembler recipes gated by research should not start until unlocked."""

    def test_hull_blocked_without_research(self) -> None:
        """Hull recipe should not start when basic science is not researched."""
        state = _make_assembler_state(
            recipe=0,
            inv_entries={int(ItemType.IRON): 10},
        )
        new = run_assemblers(state)
        assert int(new.machine_power[0, 0]) == 0
        assert int(new.machine_inventory[0, 0, ItemType.IRON]) == 10

    def test_hull_starts_when_researched(self) -> None:
        """Hull recipe should start when basic science tech is unlocked."""
        unlocked = (
            jnp.zeros(NUM_TECHNOLOGIES, dtype=jnp.bool_)
            .at[0]
            .set(
                True,
            )
        )
        state = _make_assembler_state(
            recipe=0,
            inv_entries={int(ItemType.IRON): 10},
            research_unlocked=unlocked,
        )
        new = run_assemblers(state)
        assert int(new.machine_power[0, 0]) == 4
        assert int(new.machine_inventory[0, 0, ItemType.IRON]) == 5

    def test_fuel_pack_blocked_without_research(self) -> None:
        """Fuel pack recipe should not start without fuel science research."""
        state = _make_assembler_state(
            recipe=1,
            inv_entries={
                int(ItemType.COPPER): 10,
                int(ItemType.COAL): 10,
            },
        )
        new = run_assemblers(state)
        assert int(new.machine_power[0, 0]) == 0

    def test_fuel_pack_starts_when_researched(self) -> None:
        """Fuel pack recipe should start when fuel science tech is unlocked."""
        unlocked = (
            jnp.zeros(NUM_TECHNOLOGIES, dtype=jnp.bool_)
            .at[1]
            .set(
                True,
            )
        )
        state = _make_assembler_state(
            recipe=1,
            inv_entries={
                int(ItemType.COPPER): 10,
                int(ItemType.COAL): 10,
            },
            research_unlocked=unlocked,
        )
        new = run_assemblers(state)
        assert int(new.machine_power[0, 0]) == 6

    def test_science_packs_ungated(self) -> None:
        """Science pack recipes should work without any research."""
        state = _make_assembler_state(
            recipe=3,
            inv_entries={
                int(ItemType.IRON): 5,
                int(ItemType.COPPER): 5,
            },
        )
        new = run_assemblers(state)
        assert int(new.machine_power[0, 0]) == 4


class TestResearchAction:
    """Research actions should consume science packs and unlock techs."""

    def test_consume_basic_science_pack(self, state_factory) -> None:
        """Consuming a basic science pack increments research progress."""
        p_inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        p_inv = p_inv.at[0, ItemType.BASIC_SCIENCE_PACK].set(5)

        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
            player_inventory=p_inv,
        )
        new = apply_research(state, 0, ItemType.BASIC_SCIENCE_PACK)
        assert int(new.research_progress[0]) == 1
        assert int(new.player_inventory[0, ItemType.BASIC_SCIENCE_PACK]) == 4
        assert not bool(new.research_unlocked[0])

    def test_research_unlocks_at_threshold(self, state_factory) -> None:
        """Tech should unlock when progress reaches RESEARCH_COST."""
        p_inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        p_inv = p_inv.at[0, ItemType.BASIC_SCIENCE_PACK].set(RESEARCH_COST)

        progress = jnp.array([RESEARCH_COST - 1, 0], dtype=jnp.int32)

        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
            player_inventory=p_inv,
            research_progress=progress,
        )
        new = apply_research(state, 0, ItemType.BASIC_SCIENCE_PACK)
        assert bool(new.research_unlocked[0])
        assert int(new.research_progress[0]) == RESEARCH_COST

    def test_no_consume_without_science_pack(self, state_factory) -> None:
        """Research with zero science packs should be a no-op."""
        p_inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)

        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
            player_inventory=p_inv,
        )
        new = apply_research(state, 0, ItemType.BASIC_SCIENCE_PACK)
        assert int(new.research_progress[0]) == 0

    def test_no_consume_when_already_unlocked(self, state_factory) -> None:
        """Research should not consume packs for an already-unlocked tech."""
        p_inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        p_inv = p_inv.at[0, ItemType.BASIC_SCIENCE_PACK].set(5)

        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
            player_inventory=p_inv,
            research_unlocked=jnp.array([True, False], dtype=jnp.bool_),
        )
        new = apply_research(state, 0, ItemType.BASIC_SCIENCE_PACK)
        assert int(new.player_inventory[0, ItemType.BASIC_SCIENCE_PACK]) == 5

    def test_fuel_science_unlocks_tech_1(self, state_factory) -> None:
        """Fuel science packs should advance tech index 1."""
        p_inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        p_inv = p_inv.at[0, ItemType.FUEL_SCIENCE_PACK].set(1)

        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
            player_inventory=p_inv,
        )
        new = apply_research(state, 0, ItemType.FUEL_SCIENCE_PACK)
        assert int(new.research_progress[1]) == 1
        assert int(new.research_progress[0]) == 0

    def test_research_via_step(self, state_factory) -> None:
        """RESEARCH_BASIC action through factoriax_step should work."""
        p_inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        p_inv = p_inv.at[0, ItemType.BASIC_SCIENCE_PACK].set(3)

        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
            player_inventory=p_inv,
        )
        params = EnvParams(map_width=4, map_height=4, num_players=1)
        rng = random.PRNGKey(0)
        new = factoriax_step(
            rng,
            state,
            int(Action.RESEARCH_BASIC),
            params,
        )
        assert int(new.research_progress[0]) == 1
        assert int(new.player_inventory[0, ItemType.BASIC_SCIENCE_PACK]) == 2
