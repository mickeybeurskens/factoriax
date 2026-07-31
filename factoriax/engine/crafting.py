"""Crafting a recipe by hand, out of the player's own inventory.

A player craft finishes in the step that starts it: the inputs leave the
inventory and the output arrives in the same call. A recipe's ``ticks`` is
therefore not read here, and an assembler running the same recipe pays a cost
the player does not.

Recipes come from :attr:`~factoriax.engine.state.EnvParams.recipe_table`
rather than from a module global, so a scenario can ship its own book and a
balance overlay can retune counts without rebaking the XLA graph.

Every function here is traceable and takes a player index rather than acting
on all players, because an action applies to one player per step.
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
    """Count how many of one item a player carries.

    Parameters
    ----------
    state
        State to read. Not modified.
    player_idx
        Row of ``player_inventory`` to read. Not bounds-checked, so an index
        past the player count silently reads a clamped row under ``jit``.
    item_type
        ``ItemType`` value to count.

    Returns
    -------
    jax.Array
        Scalar int16 count. Zero means the player holds none, which is also
        what an item with no recipe and no source reads as.
    """
    return state.player_inventory[player_idx, item_type]


def can_afford_recipe(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
    recipe_idx: int | jax.Array,
) -> jax.Array:
    """Check whether a player holds every input one craft of a recipe needs.

    Inputs only. Whether the output would fit is a separate question, which
    :func:`craft_recipe` asks before it commits, so an affordable recipe can
    still refuse to craft.

    Each of the recipe's two input slots is checked on its own, and a slot
    padded with ``ItemType.EMPTY`` passes. That is correct only because a
    recipe names each input item at most once, which
    :class:`~factoriax.engine.recipes.RecipeBook` enforces at construction: a
    repeated item would read as affordable on one slot's worth and then craft
    the count below zero.

    Parameters
    ----------
    state
        State to read. Not modified.
    params
        Supplies ``recipe_table``. A scenario's own book is honoured here.
    player_idx
        Which player's inventory to check.
    recipe_idx
        Row of the recipe table. Must be a real row; unlike
        :func:`craft_recipe` this does not accept ``-1``.

    Returns
    -------
    jax.Array
        Scalar bool. True when every named input is present in at least the
        required count.
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
    """Craft one batch of a recipe into a player's inventory, in this step.

    Consumes the recipe's inputs and adds ``output_count`` of its output. The
    craft is refused whole, never partially, when the recipe row does not
    exist, the player cannot afford it, or the output would pass the player's
    stack cap. A refused craft leaves the inventory exactly as it was, so a
    caller can issue the action unconditionally.

    One batch per call. Crafting takes no time, so ``ticks`` is not read and
    a player pays none of the delay the same recipe costs an assembler.

    Parameters
    ----------
    state
        State to craft from.
    params
        Supplies ``recipe_table``, which fixes the inputs, output, and yield.
    player_idx
        Which player's inventory to charge and credit.
    recipe_idx
        Row of the recipe table, or ``-1`` when the active table has no
        recipe for the requested item. ``-1`` is expected rather than an
        error: a ``CRAFT_`` action names an item, and a scenario's book need
        not produce every item the action space can name. The gather is
        clamped to a real row and the craft gated off, so the call is a
        no-op.

    Returns
    -------
    EnvState
        A new state whose ``player_inventory`` reflects the craft, or an
        equivalent state when the craft was refused. Every other field is
        carried through untouched.
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
