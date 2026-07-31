"""Craft a recipe by hand, from the inventory of the player.

A player craft finishes in the step that starts it. The inputs leave the
inventory and the output arrives in the same call. This module therefore never
reads the ``ticks`` field of a recipe. An assembler that runs the same recipe
pays a cost that the player does not pay.

Recipes come from :attr:`~factoriax.engine.state.EnvParams.recipe_table`, not
from a module global. A scenario can therefore supply its own book, and a
balance overlay can change the counts with no new XLA graph.

Every function here is traceable and takes one player index. An action applies
to one player in each step, so no function here works on all players at the
same time.
"""

import jax
import jax.numpy as jnp

from factoriax.engine.constants import ItemType
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.tables import PLAYER_MAX_STACK


def count_item_in_inventory(
    state: EnvState,
    player_idx: int | jax.Array,
    item_type: int | jax.Array,
) -> jax.Array:
    """Count the items of one type that a player carries.

    Parameters
    ----------
    state
        State to read. The function does not modify it.
    player_idx
        Row of ``player_inventory`` to read. The function does not test the
        bounds. Under ``jit``, an index past the player count reads a clamped
        row and raises nothing.
    item_type
        ``ItemType`` value to count.

    Returns
    -------
    jax.Array
        Scalar int16 count. Zero means that the player holds none of this
        item. An item with no recipe and no source also reads as zero.
    """
    return state.player_inventory[player_idx, item_type]


def can_afford_recipe(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
    recipe_idx: int | jax.Array,
) -> jax.Array:
    """Test whether a player holds every input that one craft of a recipe needs.

    This function tests the inputs only. Space for the output is a separate
    question. :func:`craft_recipe` asks that question before it writes, so a
    recipe that the player can afford can still fail to craft.

    The function tests each of the two input slots of the recipe on its own. A
    slot with the padding value ``ItemType.EMPTY`` passes. This is correct only
    because a recipe names each input item one time at most.
    :class:`~factoriax.engine.recipes.RecipeBook` enforces that rule when it
    builds the table. With a repeated item, one slot alone can pass the test,
    and the craft then takes the count below zero.

    Parameters
    ----------
    state
        State to read. The function does not modify it.
    params
        Supplies ``recipe_table``. This function reads the book of the
        scenario, if the scenario has one.
    player_idx
        Player whose inventory the function reads.
    recipe_idx
        Row of the recipe table. The value must be a real row. This function
        does not accept ``-1``, but :func:`craft_recipe` does.

    Returns
    -------
    jax.Array
        Scalar bool. True when the player holds each named input in the
        required count or more.
    """
    table = params.recipe_table
    input_items = table.input_items[recipe_idx]
    input_counts = table.input_counts[recipe_idx]
    inv = state.player_inventory[player_idx]

    result = jnp.bool_(True)
    for i in range(input_items.shape[0]):
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
    """Craft one batch of a recipe into the inventory of a player, in this step.

    The function consumes the inputs of the recipe and adds ``output_count`` of
    its output. It refuses the whole craft, and never a part of it, in three
    cases:

    - The recipe row does not exist.
    - The player cannot afford the inputs.
    - The output passes the stack limit of the player.

    A refused craft leaves the inventory exactly as it was, so a caller can
    issue the action at any time.

    One call crafts one batch. A hand craft takes no time, so this function
    never reads ``ticks``. The player pays none of the delay that the same
    recipe costs an assembler.

    Parameters
    ----------
    state
        State to craft from.
    params
        Supplies ``recipe_table``, which fixes the inputs, output, and yield.
    player_idx
        Player whose inventory the function charges and credits.
    recipe_idx
        Row of the recipe table, or ``-1`` when the active table has no recipe
        for the item. The value ``-1`` is normal and is not an error. A
        ``CRAFT_`` action names an item, and the book of a scenario does not
        have to produce every item that the action space can name. The
        function clamps the gather to a real row and gates the craft off, so
        such a call does nothing.

    Returns
    -------
    EnvState
        A new state. Its ``player_inventory`` holds the result of the craft.
        After a refused craft that inventory is equal to the input inventory.
        Every other field passes through unchanged.
    """
    table = params.recipe_table
    # Clamp the gather to a valid row and gate the craft off ``valid``, so a
    # -1 index reads safely and the action does nothing.
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
    for i in range(table.input_items.shape[1]):
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
