"""Tests for :mod:`factoriax.engine.crafting`.

``engine/step.py`` imports ``craft_recipe``, so the step tests already execute
every line of this module. They do not assert its contract. This file drives
the three functions directly and pins what the docstrings promise, above all
the two refusals that a step-level test never reaches: a recipe row of ``-1``,
and an output that would pass the player stack limit.

A refusal is all or nothing. The inventory after one must equal the inventory
before it, so a caller can issue a craft action at any time.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from factoriax.engine.constants import NUM_ITEM_TYPES, BlockType, ItemType
from factoriax.engine.crafting import (
    can_afford_recipe,
    count_item_in_inventory,
    craft_recipe,
)
from factoriax.engine.recipes import BASE_RECIPES
from factoriax.engine.state import EnvParams
from factoriax.engine.tables import PLAYER_MAX_STACK

#: Row 0 of the shipped table: one IRON_ORE plus one COAL makes one IRON_PLATE.
_IRON_PLATE_ROW = 0
_OUT = int(ItemType.IRON_PLATE)
_IN_A = int(ItemType.IRON_ORE)
_IN_B = int(ItemType.COAL)


@pytest.fixture
def params() -> EnvParams:
    """Return parameters carrying the shipped recipe table."""
    return EnvParams()


def _inventory(**items: int) -> jnp.ndarray:
    """Build a one-player inventory from item name to count."""
    inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int16)
    for name, count in items.items():
        inv = inv.at[0, int(getattr(ItemType, name))].set(count)
    return inv


def _state(state_factory, **items: int):
    """Build a one-tile state whose player carries ``items``."""
    return state_factory(
        world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32),
        player_inventory=_inventory(**items),
    )


class TestCountItemInInventory:
    """``count_item_in_inventory`` reads one cell of the inventory."""

    def test_reports_the_count_the_player_holds(self, state_factory) -> None:
        """A held item reports its count."""
        state = _state(state_factory, COAL=7)
        assert int(count_item_in_inventory(state, 0, _IN_B)) == 7

    def test_reports_zero_for_an_item_the_player_lacks(self, state_factory) -> None:
        """An item the player does not hold reads as zero, and does not raise."""
        state = _state(state_factory, COAL=7)
        assert int(count_item_in_inventory(state, 0, _IN_A)) == 0


class TestCanAffordRecipe:
    """``can_afford_recipe`` tests the inputs, and only the inputs."""

    def test_true_when_every_input_is_present(self, state_factory, params) -> None:
        """A player holding both inputs can afford the recipe."""
        state = _state(state_factory, IRON_ORE=1, COAL=1)
        assert bool(can_afford_recipe(state, params, 0, _IRON_PLATE_ROW))

    def test_false_when_one_input_is_short(self, state_factory, params) -> None:
        """Holding one input and not the other is not enough."""
        state = _state(state_factory, IRON_ORE=1)
        assert not bool(can_afford_recipe(state, params, 0, _IRON_PLATE_ROW))

    def test_true_with_a_surplus(self, state_factory, params) -> None:
        """More than the recipe needs still affords it."""
        state = _state(state_factory, IRON_ORE=50, COAL=50)
        assert bool(can_afford_recipe(state, params, 0, _IRON_PLATE_ROW))

    def test_ignores_whether_the_output_has_room(self, state_factory, params) -> None:
        """A full output stack does not make a recipe unaffordable.

        The docstring separates the two questions. ``craft_recipe`` asks about
        space, so a recipe the player can afford can still refuse to craft.
        """
        cap = int(PLAYER_MAX_STACK[_OUT])
        state = _state(state_factory, IRON_ORE=1, COAL=1, IRON_PLATE=cap)
        assert bool(can_afford_recipe(state, params, 0, _IRON_PLATE_ROW))


class TestCraftRecipe:
    """``craft_recipe`` charges the inputs and credits the output, or neither."""

    def test_consumes_the_inputs_and_adds_the_output(
        self, state_factory, params
    ) -> None:
        """One craft spends each input and yields ``output_count``."""
        state = _state(state_factory, IRON_ORE=3, COAL=3)
        new = craft_recipe(state, params, 0, _IRON_PLATE_ROW)

        assert int(new.player_inventory[0, _IN_A]) == 2
        assert int(new.player_inventory[0, _IN_B]) == 2
        assert (
            int(new.player_inventory[0, _OUT])
            == BASE_RECIPES[_IRON_PLATE_ROW].output_count
        )

    def test_a_recipe_row_of_minus_one_does_nothing(
        self, state_factory, params
    ) -> None:
        """``-1`` is normal, not an error, and leaves the inventory alone.

        A ``CRAFT_`` action names an item. A scenario book does not have to
        produce every item the action space can name, so the lookup returns
        ``-1`` and the craft gates off.
        """
        state = _state(state_factory, IRON_ORE=3, COAL=3)
        new = craft_recipe(state, params, 0, -1)

        assert jnp.array_equal(new.player_inventory, state.player_inventory)

    def test_an_unaffordable_craft_spends_nothing(self, state_factory, params) -> None:
        """A refused craft must not take the inputs it could reach."""
        state = _state(state_factory, IRON_ORE=1)
        new = craft_recipe(state, params, 0, _IRON_PLATE_ROW)

        assert jnp.array_equal(new.player_inventory, state.player_inventory)

    def test_a_full_output_stack_refuses_the_whole_craft(
        self, state_factory, params
    ) -> None:
        """No room for the output means the inputs stay put.

        This is the refusal a step-level test never reaches, because filling a
        stack to its cap through gameplay takes a long episode.
        """
        cap = int(PLAYER_MAX_STACK[_OUT])
        state = _state(state_factory, IRON_ORE=5, COAL=5, IRON_PLATE=cap)
        new = craft_recipe(state, params, 0, _IRON_PLATE_ROW)

        assert jnp.array_equal(new.player_inventory, state.player_inventory)

    def test_crafting_into_the_last_slot_is_allowed(
        self, state_factory, params
    ) -> None:
        """A craft that exactly reaches the cap is not a refusal."""
        cap = int(PLAYER_MAX_STACK[_OUT])
        yield_count = BASE_RECIPES[_IRON_PLATE_ROW].output_count
        state = _state(state_factory, IRON_ORE=5, COAL=5, IRON_PLATE=cap - yield_count)
        new = craft_recipe(state, params, 0, _IRON_PLATE_ROW)

        assert int(new.player_inventory[0, _OUT]) == cap
        assert int(new.player_inventory[0, _IN_A]) == 4

    def test_leaves_every_other_state_field_alone(self, state_factory, params) -> None:
        """A craft writes ``player_inventory`` and nothing else."""
        state = _state(state_factory, IRON_ORE=3, COAL=3)
        new = craft_recipe(state, params, 0, _IRON_PLATE_ROW)

        assert jnp.array_equal(new.map, state.map)
        assert jnp.array_equal(new.player_positions, state.player_positions)
        assert int(new.timestep) == int(state.timestep)

    def test_takes_no_time(self, state_factory, params) -> None:
        """A hand craft finishes in the step that starts it.

        The module never reads ``Recipe.ticks``. The same recipe costs an
        assembler that delay. ``ISSUES.md`` records the asymmetry.
        """
        assert BASE_RECIPES[_IRON_PLATE_ROW].ticks > 0

        state = _state(state_factory, IRON_ORE=1, COAL=1)
        new = craft_recipe(state, params, 0, _IRON_PLATE_ROW)

        assert int(new.player_inventory[0, _OUT]) > 0
        assert int(new.timestep) == int(state.timestep)
