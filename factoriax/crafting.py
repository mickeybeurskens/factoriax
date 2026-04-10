"""Crafting system for the FactoriaX environment.

Player crafting is instant: check if the player has the required inputs,
consume them, produce the output. Uses the same recipe table as assemblers.
"""

import jax
import jax.numpy as jnp

from factoriax.constants import PLAYER_MAX_STACK, ItemType
from factoriax.recipes import (
    MAX_RECIPE_INPUTS,
    RECIPE_INPUT_COUNTS,
    RECIPE_INPUT_ITEMS,
    RECIPE_OUTPUTS,
)
from factoriax.state import EnvState


def count_item_in_inventory(
    state: EnvState,
    player_idx: int | jax.Array,
    item_type: int | jax.Array,
) -> jax.Array:
    """Count how many of an item a player has.

    Args:
        state: Current environment state.
        player_idx: Index of the player.
        item_type: Item type to count.

    Returns:
        Item count.
    """
    return state.player_inventory[player_idx, item_type]


def can_afford_recipe(
    state: EnvState,
    player_idx: int | jax.Array,
    recipe_idx: int | jax.Array,
) -> jax.Array:
    """Check if a player can afford to craft a recipe.

    Args:
        state: Current environment state.
        player_idx: Index of the player.
        recipe_idx: Index of the recipe.

    Returns:
        Boolean indicating if player has all required materials.
    """
    input_items = RECIPE_INPUT_ITEMS[recipe_idx]
    input_counts = RECIPE_INPUT_COUNTS[recipe_idx]
    inv = state.player_inventory[player_idx]

    result = jnp.bool_(True)
    for i in range(MAX_RECIPE_INPUTS):
        item_type = input_items[i]
        required = input_counts[i]
        have = inv[item_type]
        is_valid = item_type != int(ItemType.EMPTY)
        result = result & jnp.where(is_valid, have >= required, True)
    return result


def craft_recipe(
    state: EnvState,
    player_idx: int | jax.Array,
    recipe_idx: int | jax.Array,
) -> EnvState:
    """Instantly craft a recipe for a player.

    Checks affordability, consumes inputs, produces output. Does nothing
    if the player can't afford it or has no space for the output.

    Args:
        state: Current environment state.
        player_idx: Index of the player.
        recipe_idx: Index of the recipe to craft.

    Returns:
        Updated state with recipe crafted (or unchanged).
    """
    can_craft = can_afford_recipe(state, player_idx, recipe_idx)

    output_item = RECIPE_OUTPUTS[recipe_idx]
    output_count = state.player_inventory[player_idx, output_item]
    output_max = PLAYER_MAX_STACK[output_item]
    has_space = output_count < output_max
    should_craft = can_craft & has_space

    # Consume inputs.
    inv = state.player_inventory[player_idx]
    for i in range(MAX_RECIPE_INPUTS):
        item_type = RECIPE_INPUT_ITEMS[recipe_idx, i]
        amount = RECIPE_INPUT_COUNTS[recipe_idx, i]
        is_valid = item_type != int(ItemType.EMPTY)
        inv = inv.at[item_type].add(
            jnp.where(should_craft & is_valid, -amount, 0).astype(jnp.int16),
        )

    # Produce output.
    inv = inv.at[output_item].add(
        jnp.where(should_craft, jnp.int16(1), jnp.int16(0)),
    )

    return state.replace(
        player_inventory=state.player_inventory.at[player_idx].set(inv),
    )
