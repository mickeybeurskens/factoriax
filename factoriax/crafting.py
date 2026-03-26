"""Crafting system for the FactoriaX environment."""

import jax
import jax.numpy as jnp
from jax import lax

from factoriax.constants import (
    MAX_STACK_SIZE,
    NUM_INVENTORY_SLOTS,
    ItemType,
)
from factoriax.inventory import add_items_to_slots
from factoriax.recipes import (
    MAX_RECIPE_INPUTS,
    RECIPE_INPUT_COUNTS,
    RECIPE_INPUT_ITEMS,
    RECIPE_OUTPUTS,
    RECIPE_TICKS,
)
from factoriax.state import EnvState


def count_item_in_inventory(
    state: EnvState, player_idx: int | jax.Array, item_type: int | jax.Array
) -> jax.Array:
    """Count total quantity of an item type in a player's inventory.

    Args:
        state: Current environment state
        player_idx: Index of the player
        item_type: Item type to count

    Returns:
        Total count of the item across all inventory slots
    """
    items = state.inventory_items[player_idx]
    counts = state.inventory_counts[player_idx]
    mask = items == item_type
    return jnp.sum(jnp.where(mask, counts, 0))


def can_afford_recipe(
    state: EnvState, player_idx: int | jax.Array, recipe_idx: int | jax.Array
) -> jax.Array:
    """Check if a player can afford to craft a recipe.

    Args:
        state: Current environment state
        player_idx: Index of the player
        recipe_idx: Index of the recipe in RECIPES list

    Returns:
        Boolean indicating if player has all required materials
    """
    input_items = RECIPE_INPUT_ITEMS[recipe_idx]
    input_counts = RECIPE_INPUT_COUNTS[recipe_idx]

    def check_input(carry: jax.Array, input_idx: int) -> tuple[jax.Array, None]:
        can_afford = carry
        item_type = input_items[input_idx]
        required = input_counts[input_idx]
        have = count_item_in_inventory(state, player_idx, item_type)
        is_valid_input = item_type != ItemType.EMPTY
        can_afford = can_afford & (jnp.where(is_valid_input, have >= required, True))
        return can_afford, None

    can_afford, _ = lax.scan(
        check_input, jnp.bool_(True), jnp.arange(MAX_RECIPE_INPUTS)
    )
    return can_afford


def remove_item_from_inventory(
    state: EnvState,
    player_idx: int | jax.Array,
    item_type: int | jax.Array,
    amount: int | jax.Array,
) -> EnvState:
    """Remove a quantity of an item from a player's inventory.

    Removes items from slots containing the item type until the required
    amount is removed. May span multiple slots if needed.

    Args:
        state: Current environment state
        player_idx: Index of the player
        item_type: Item type to remove
        amount: Quantity to remove

    Returns:
        Updated state with items removed from inventory
    """
    items = state.inventory_items[player_idx]
    counts = state.inventory_counts[player_idx]
    remaining = amount

    def remove_from_slot(carry: tuple, slot_idx: int) -> tuple:
        items_arr, counts_arr, remaining_amt = carry
        is_match = items_arr[slot_idx] == item_type
        slot_count = counts_arr[slot_idx]
        to_remove = jnp.minimum(remaining_amt, slot_count)
        to_remove = jnp.where(is_match, to_remove, 0)

        new_count = slot_count - to_remove
        new_remaining = remaining_amt - to_remove

        counts_arr = counts_arr.at[slot_idx].set(new_count)
        items_arr = jnp.where(
            new_count == 0,
            items_arr.at[slot_idx].set(ItemType.EMPTY),
            items_arr,
        )

        return (items_arr, counts_arr, new_remaining), None

    (new_items, new_counts, _), _ = lax.scan(
        remove_from_slot,
        (items, counts, remaining),
        jnp.arange(NUM_INVENTORY_SLOTS),
    )

    return state.replace(
        inventory_items=state.inventory_items.at[player_idx].set(new_items),
        inventory_counts=state.inventory_counts.at[player_idx].set(new_counts),
    )


def consume_recipe_materials(
    state: EnvState, player_idx: int | jax.Array, recipe_idx: int | jax.Array
) -> EnvState:
    """Consume materials required for a recipe from player's inventory.

    Args:
        state: Current environment state
        player_idx: Index of the player
        recipe_idx: Index of the recipe in RECIPES list

    Returns:
        Updated state with materials removed from inventory
    """
    input_items = RECIPE_INPUT_ITEMS[recipe_idx]
    input_counts = RECIPE_INPUT_COUNTS[recipe_idx]

    def consume_input(state: EnvState, input_idx: int) -> tuple[EnvState, None]:
        item_type = input_items[input_idx]
        required = input_counts[input_idx]
        is_valid_input = item_type != ItemType.EMPTY
        state = lax.cond(
            is_valid_input,
            lambda s: remove_item_from_inventory(s, player_idx, item_type, required),
            lambda s: s,
            state,
        )
        return state, None

    state, _ = lax.scan(consume_input, state, jnp.arange(MAX_RECIPE_INPUTS))
    return state


def add_item_to_inventory(
    state: EnvState,
    player_idx: int | jax.Array,
    item_type: int | jax.Array,
    amount: int | jax.Array,
) -> EnvState:
    """Add items to a player's inventory.

    First tries to stack with existing items of the same type, then uses
    empty slots. Items that don't fit are lost.

    Args:
        state: Current environment state
        player_idx: Index of the player
        item_type: Item type to add
        amount: Quantity to add

    Returns:
        Updated state with items added to inventory
    """
    items = state.inventory_items[player_idx]
    counts = state.inventory_counts[player_idx]

    new_items, new_counts, _ = add_items_to_slots(
        items, counts, item_type, amount, MAX_STACK_SIZE
    )

    return state.replace(
        inventory_items=state.inventory_items.at[player_idx].set(new_items),
        inventory_counts=state.inventory_counts.at[player_idx].set(new_counts),
    )


def start_crafting(
    state: EnvState,
    player_idx: int | jax.Array,
    recipe_idx: int | jax.Array,
) -> EnvState:
    """Craft a specific recipe for a player.

    Checks affordability and that no craft is already in progress. When
    ``RECIPE_TICKS`` is 0 (the default), crafting is instant: materials
    are consumed and the output appears in the same step. When ticks > 0,
    materials are consumed immediately but the output is deferred until
    ``update_crafting`` counts the progress down to zero.

    Args:
        state: Current environment state.
        player_idx: Index of the player.
        recipe_idx: Index of the recipe to craft.

    Returns:
        Updated state with crafting started (or unchanged if invalid).
    """
    is_crafting = state.craft_progress[player_idx] > 0
    can_afford = can_afford_recipe(state, player_idx, recipe_idx)
    should_start = can_afford & ~is_crafting

    craft_ticks = RECIPE_TICKS[recipe_idx]
    is_instant = craft_ticks == 0

    def do_craft(s: EnvState) -> EnvState:
        s = consume_recipe_materials(s, player_idx, recipe_idx)
        output_item = RECIPE_OUTPUTS[recipe_idx]
        # Instant: add output now. Delayed: set progress countdown.
        s = lax.cond(
            is_instant,
            lambda s2: add_item_to_inventory(s2, player_idx, output_item, 1),
            lambda s2: s2.replace(
                craft_progress=s2.craft_progress.at[player_idx].set(
                    craft_ticks
                ),
                crafting_recipe=s2.crafting_recipe.at[player_idx].set(
                    recipe_idx
                ),
            ),
            s,
        )
        return s

    return lax.cond(should_start, do_craft, lambda s: s, state)


def update_crafting(state: EnvState) -> EnvState:
    """Progress all active crafts and complete finished ones.

    Called each game step. Decrements craft_progress for all players with
    active crafts. When a craft completes (progress reaches 0), reads
    the recipe from ``crafting_recipe`` and adds the output item to the
    player's inventory.

    Args:
        state: Current environment state.

    Returns:
        Updated state with crafting progressed.
    """

    def update_player_craft(
        state: EnvState, player_idx: int
    ) -> tuple[EnvState, None]:
        progress = state.craft_progress[player_idx]
        is_crafting = progress > 0

        new_progress = jnp.maximum(0, progress - 1)
        just_finished = is_crafting & (new_progress == 0)

        state = state.replace(
            craft_progress=state.craft_progress.at[player_idx].set(
                new_progress
            )
        )

        recipe_idx = state.crafting_recipe[player_idx]
        output_item = RECIPE_OUTPUTS[recipe_idx]

        state = lax.cond(
            just_finished,
            lambda s: add_item_to_inventory(
                s, player_idx, output_item, 1
            ),
            lambda s: s,
            state,
        )

        return state, None

    num_players = state.player_positions.shape[0]
    state, _ = lax.scan(
        update_player_craft, state, jnp.arange(num_players)
    )

    return state


def cycle_slot(
    state: EnvState,
    player_idx: int | jax.Array,
    direction: int | jax.Array,
) -> EnvState:
    """Cycle the selected inventory slot for a player.

    Args:
        state: Current environment state.
        player_idx: Index of the player.
        direction: 1 for next slot, -1 for previous slot.

    Returns:
        Updated state with new selected slot.
    """
    current_slot = state.selected_slots[player_idx]
    new_slot = (current_slot + direction) % NUM_INVENTORY_SLOTS

    return state.replace(
        selected_slots=state.selected_slots.at[player_idx].set(new_slot)
    )
