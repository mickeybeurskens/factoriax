"""Crafting system for the FactoriaX environment.

Player crafting is instant: check if the player has the required inputs,
consume them, produce the output. Uses the same recipe table as assemblers.

Recipe arrays flow in via :class:`~factoriax.state.EnvParams.recipe_table`
so balance overlays (Step 5+) can tune input/output counts without
rebaking the XLA graph.
"""

import jax
import jax.numpy as jnp

from factoriax.constants import ItemType
from factoriax.recipes import MAX_RECIPE_INPUTS
from factoriax.state import EnvParams, EnvState
from factoriax.tables import PLAYER_MAX_STACK


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
    params: EnvParams,
    player_idx: int | jax.Array,
    recipe_idx: int | jax.Array,
) -> jax.Array:
    """Check if a player can afford to craft a recipe.

    Args:
        state: Current environment state.
        params: Environment parameters (supplies the recipe table).
        player_idx: Index of the player.
        recipe_idx: Index of the recipe.

    Returns:
        Boolean indicating if player has all required materials.
    """
    table = params.recipe_table
    input_items = table.input_items[recipe_idx]
    input_counts = table.input_counts[recipe_idx]
    inv = state.player_inventory[player_idx]

    result = jnp.bool_(True)
    for i in range(MAX_RECIPE_INPUTS):
        item_type = input_items[i]
        required = input_counts[i]
        have = inv[item_type]
        is_valid = item_type != int(ItemType.EMPTY)
        result = result & jnp.where(is_valid, have >= required, True)
    return jnp.asarray(result)


def craft_recipe(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
    recipe_idx: int | jax.Array,
) -> EnvState:
    """Instantly craft a recipe for a player.

    Checks affordability, consumes inputs, produces output. Does nothing
    if the player can't afford it or has no space for the output.

    Args:
        state: Current environment state.
        params: Environment parameters (supplies the recipe table).
        player_idx: Index of the player.
        recipe_idx: Index of the recipe to craft.

    Returns:
        Updated state with recipe crafted (or unchanged).
    """
    table = params.recipe_table
    # ``recipe_idx`` is -1 when the active table has no recipe for the
    # requested item (e.g. a CRAFT_* action whose output isn't in this
    # scenario's table). Clamp the gather to a valid row and gate the craft
    # off ``valid`` so the read is safe and the action no-ops.
    valid = jnp.asarray(recipe_idx) >= 0
    recipe_idx = jnp.maximum(jnp.asarray(recipe_idx), 0)
    can_craft = can_afford_recipe(state, params, player_idx, recipe_idx)

    output_item = table.outputs[recipe_idx]
    output_count = state.player_inventory[player_idx, output_item]
    output_max = PLAYER_MAX_STACK[output_item]
    yield_count = table.output_counts[recipe_idx]
    has_space = output_count + yield_count <= output_max
    should_craft = valid & can_craft & has_space

    # Consume inputs.
    inv = state.player_inventory[player_idx]
    for i in range(MAX_RECIPE_INPUTS):
        item_type = table.input_items[recipe_idx, i]
        amount = table.input_counts[recipe_idx, i]
        is_valid = item_type != int(ItemType.EMPTY)
        inv = inv.at[item_type].add(
            jnp.where(should_craft & is_valid, -amount, 0).astype(jnp.int16),
        )

    # Produce output.
    inv = inv.at[output_item].add(
        jnp.where(should_craft, yield_count.astype(jnp.int16), jnp.int16(0)),
    )

    return state.replace(
        player_inventory=state.player_inventory.at[player_idx].set(inv),
    )
