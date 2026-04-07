"""Crafting system for the FactoriaX environment.

Uses the pouch inventory model: recipes check and modify per-type
counts directly, with no slot scanning or cursor management.
"""

import jax
import jax.numpy as jnp
from jax import lax

from factoriax.constants import PLAYER_MAX_STACK, ItemType
from factoriax.inventory import add_to_pouch, remove_from_pouch
from factoriax.recipes import (
    MAX_RECIPE_INPUTS,
    RECIPE_INPUT_COUNTS,
    RECIPE_INPUT_ITEMS,
    RECIPE_OUTPUTS,
    RECIPE_TICKS,
)
from factoriax.state import EnvState


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

    def check_input(
        carry: jax.Array, input_idx: jax.Array,
    ) -> tuple[jax.Array, None]:
        can_afford = carry
        item_type = input_items[input_idx]
        required = input_counts[input_idx]
        have = inv[item_type]
        is_valid = item_type != int(ItemType.EMPTY)
        can_afford = can_afford & jnp.where(
            is_valid, have >= required, True,
        )
        return can_afford, None

    result, _ = lax.scan(
        check_input, jnp.bool_(True), jnp.arange(MAX_RECIPE_INPUTS),
    )
    return result


def consume_recipe_materials(
    state: EnvState,
    player_idx: int | jax.Array,
    recipe_idx: int | jax.Array,
) -> EnvState:
    """Consume materials required for a recipe from player inventory.

    Args:
        state: Current environment state.
        player_idx: Index of the player.
        recipe_idx: Index of the recipe.

    Returns:
        Updated state with materials removed.
    """
    input_items = RECIPE_INPUT_ITEMS[recipe_idx]
    input_counts = RECIPE_INPUT_COUNTS[recipe_idx]

    def consume_input(
        inv: jax.Array, input_idx: jax.Array,
    ) -> tuple[jax.Array, None]:
        item_type = input_items[input_idx]
        required = input_counts[input_idx]
        is_valid = item_type != int(ItemType.EMPTY)
        new_inv, _ = remove_from_pouch(inv, item_type, required)
        inv = jnp.where(is_valid, new_inv, inv)
        return inv, None

    inv = state.player_inventory[player_idx]
    inv, _ = lax.scan(consume_input, inv, jnp.arange(MAX_RECIPE_INPUTS))
    return state.replace(
        player_inventory=state.player_inventory.at[player_idx].set(inv),
    )


def add_item_to_player(
    state: EnvState,
    player_idx: int | jax.Array,
    item_type: int | jax.Array,
    amount: int | jax.Array,
) -> EnvState:
    """Add items to a player's pouch inventory.

    Items that exceed PLAYER_MAX_STACK are lost.

    Args:
        state: Current environment state.
        player_idx: Index of the player.
        item_type: Item type to add.
        amount: Quantity to add.

    Returns:
        Updated state with items added.
    """
    inv = state.player_inventory[player_idx]
    new_inv, _ = add_to_pouch(inv, item_type, amount, PLAYER_MAX_STACK)
    return state.replace(
        player_inventory=state.player_inventory.at[player_idx].set(
            new_inv,
        ),
    )


def start_crafting(
    state: EnvState,
    player_idx: int | jax.Array,
    recipe_idx: int | jax.Array,
) -> EnvState:
    """Craft a specific recipe for a player.

    Checks affordability and that no craft is already in progress.
    When RECIPE_TICKS is 0, crafting is instant. When > 0, materials
    are consumed immediately but the output is deferred.

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
        s = lax.cond(
            is_instant,
            lambda s2: add_item_to_player(
                s2, player_idx, output_item, 1,
            ),
            lambda s2: s2.replace(
                craft_progress=s2.craft_progress.at[player_idx].set(
                    craft_ticks,
                ),
                crafting_recipe=s2.crafting_recipe.at[player_idx].set(
                    recipe_idx,
                ),
            ),
            s,
        )
        return s

    return lax.cond(should_start, do_craft, lambda s: s, state)


def update_crafting(state: EnvState) -> EnvState:
    """Progress all active crafts and complete finished ones.

    Called each game step. Decrements craft_progress for players with
    active crafts. When progress reaches 0, adds the output item.

    Args:
        state: Current environment state.

    Returns:
        Updated state with crafting progressed.
    """

    def update_player_craft(
        state: EnvState, player_idx: jax.Array,
    ) -> tuple[EnvState, None]:
        progress = state.craft_progress[player_idx]
        is_crafting = progress > 0

        new_progress = jnp.maximum(0, progress - 1)
        just_finished = is_crafting & (new_progress == 0)

        state = state.replace(
            craft_progress=state.craft_progress.at[player_idx].set(
                new_progress,
            ),
        )

        recipe_idx = state.crafting_recipe[player_idx]
        output_item = RECIPE_OUTPUTS[recipe_idx]

        state = lax.cond(
            just_finished,
            lambda s: add_item_to_player(
                s, player_idx, output_item, 1,
            ),
            lambda s: s,
            state,
        )

        return state, None

    num_players = state.player_positions.shape[0]
    state, _ = lax.scan(
        update_player_craft, state, jnp.arange(num_players),
    )
    return state


# Backward-compat aliases for Phase 2 UI code.
def count_item_in_inventory(
    state: EnvState,
    player_idx: int,
    item_type: int,
) -> jax.Array:
    """Deprecated. Read count directly from player_inventory."""
    return state.player_inventory[player_idx, item_type]


# Alias: old name -> new name.
add_item_to_inventory = add_item_to_player
