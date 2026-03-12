"""Crafting system for the FactoriaX environment."""

import jax
import jax.numpy as jnp
from jax import lax

from factoriax.constants import (
    MAX_RECIPE_INPUTS,
    MAX_STACK_SIZE,
    NUM_INVENTORY_SLOTS,
    NUM_RECIPES,
    RECIPE_INPUT_COUNTS,
    RECIPE_INPUT_ITEMS,
    RECIPE_OUTPUTS,
    RECIPE_TICKS,
    ItemType,
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

    can_afford, _ = lax.scan(check_input, jnp.bool_(True), jnp.arange(MAX_RECIPE_INPUTS))
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
    remaining = amount

    def add_to_slot(carry: tuple, slot_idx: int) -> tuple:
        items_arr, counts_arr, remaining_amt = carry
        slot_item = items_arr[slot_idx]
        slot_count = counts_arr[slot_idx]

        can_stack = (slot_item == item_type) & (slot_count < MAX_STACK_SIZE)
        is_empty = slot_item == ItemType.EMPTY

        space = jnp.where(can_stack, MAX_STACK_SIZE - slot_count, 0)
        space = jnp.where(is_empty, MAX_STACK_SIZE, space)

        to_add = jnp.minimum(remaining_amt, space)

        new_count = slot_count + to_add
        new_remaining = remaining_amt - to_add

        items_arr = jnp.where(
            (is_empty & (to_add > 0)),
            items_arr.at[slot_idx].set(item_type),
            items_arr,
        )
        counts_arr = counts_arr.at[slot_idx].set(new_count)

        return (items_arr, counts_arr, new_remaining), None

    (new_items, new_counts, _), _ = lax.scan(
        add_to_slot,
        (items, counts, remaining),
        jnp.arange(NUM_INVENTORY_SLOTS),
    )

    return state.replace(
        inventory_items=state.inventory_items.at[player_idx].set(new_items),
        inventory_counts=state.inventory_counts.at[player_idx].set(new_counts),
    )


def start_crafting(state: EnvState, player_idx: int | jax.Array) -> EnvState:
    """Start crafting the currently selected recipe for a player.

    Checks if the player can afford the recipe and is not already crafting.
    If valid, consumes materials and sets the craft timer.

    Args:
        state: Current environment state
        player_idx: Index of the player

    Returns:
        Updated state with crafting started (or unchanged if invalid)
    """
    recipe_idx = state.selected_recipes[player_idx]
    is_crafting = state.craft_progress[player_idx] > 0
    can_afford = can_afford_recipe(state, player_idx, recipe_idx)

    should_start = can_afford & ~is_crafting

    craft_ticks = RECIPE_TICKS[recipe_idx]

    new_state = lax.cond(
        should_start,
        lambda s: consume_recipe_materials(s, player_idx, recipe_idx).replace(
            craft_progress=s.craft_progress.at[player_idx].set(craft_ticks)
        ),
        lambda s: s,
        state,
    )

    return new_state


def update_crafting(state: EnvState) -> EnvState:
    """Progress all active crafts and complete finished ones.

    Called each game step. Decrements craft_progress for all players with
    active crafts. When a craft completes (progress reaches 0), adds the
    output item to the player's inventory.

    Args:
        state: Current environment state

    Returns:
        Updated state with crafting progressed
    """

    def update_player_craft(state: EnvState, player_idx: int) -> tuple[EnvState, None]:
        progress = state.craft_progress[player_idx]
        is_crafting = progress > 0

        new_progress = jnp.maximum(0, progress - 1)
        just_finished = is_crafting & (new_progress == 0)

        state = state.replace(
            craft_progress=state.craft_progress.at[player_idx].set(new_progress)
        )

        recipe_idx = state.selected_recipes[player_idx]
        output_item = RECIPE_OUTPUTS[recipe_idx]

        state = lax.cond(
            just_finished,
            lambda s: add_item_to_inventory(s, player_idx, output_item, 1),
            lambda s: s,
            state,
        )

        return state, None

    num_players = state.player_positions.shape[0]
    state, _ = lax.scan(update_player_craft, state, jnp.arange(num_players))

    return state


def cycle_slot(
    state: EnvState, player_idx: int | jax.Array, direction: int | jax.Array
) -> EnvState:
    """Cycle the selected inventory slot for a player.

    Args:
        state: Current environment state
        player_idx: Index of the player
        direction: 1 for next slot, -1 for previous slot

    Returns:
        Updated state with new selected slot
    """
    current_slot = state.selected_slots[player_idx]
    new_slot = (current_slot + direction) % NUM_INVENTORY_SLOTS

    return state.replace(
        selected_slots=state.selected_slots.at[player_idx].set(new_slot)
    )


def cycle_recipe(
    state: EnvState, player_idx: int | jax.Array, direction: int | jax.Array
) -> EnvState:
    """Cycle the selected recipe for a player.

    Args:
        state: Current environment state
        player_idx: Index of the player
        direction: 1 for next recipe, -1 for previous recipe

    Returns:
        Updated state with new selected recipe
    """
    current_recipe = state.selected_recipes[player_idx]
    new_recipe = (current_recipe + direction) % NUM_RECIPES

    return state.replace(
        selected_recipes=state.selected_recipes.at[player_idx].set(new_recipe)
    )
