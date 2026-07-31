"""Score what changed in one step.

A reward function reads the state before a step and the state after it. It
returns one number that says how good the step was. It never changes the state.
Every function here takes ``(prev_state, new_state, params)`` and returns a
scalar float32. An environment can therefore take any of them, and
:func:`functools.partial` can bind the extra argument of the functions that
need one.

The functions are of two kinds, and the difference matters more than the names
suggest.

A **sparse** reward pays only when the target event happens: a player mines an
ore, or an achievement unlocks. It is honest and hard to learn from, because
most steps score zero.

A **dense** reward adds shaping terms. Those terms pay a small amount for a
position near something useful, or for a quantity that moves in the right
direction. A dense reward is easier to learn from and easier to get wrong,
because an agent optimises the terms that it receives, and not the goal behind
them. Two of the shaping terms here have that problem, and their own docstrings
say so.

A proximity term measures Manhattan distance from ``state.selected_player``,
and not from a player in an argument. With more than one player it therefore
always scores the selected player.

The rewards have no clamp, and several can go negative, because they are
differences between two states and a quantity can fall. Read the return
description of each function. Do not assume a floor of zero.
"""

import jax
import jax.numpy as jnp

from factoriax.engine.constants import (
    ItemType,
    Machine,
)
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.tables import MINEABLE_BLOCKS

# ---------------------------------------------------------------------------
# Proximity helper
# ---------------------------------------------------------------------------


def _proximity(state: EnvState, tile_mask: jax.Array) -> jax.Array:
    """Score how close the selected player is to the nearest matching tile.

    This is the shaping term that the dense rewards are built from. It turns a
    distance into a number that rises as the player comes closer. An agent
    therefore has a gradient to follow, and not a flat zero until it arrives.

    The distance is Manhattan, in tiles, from
    ``state.player_positions[state.selected_player]``. The function ignores the
    other players.

    Parameters
    ----------
    state
        State to read. The function reads ``map`` for the shape, and
        ``player_positions``.
    tile_mask
        Tiles that count as a target. Shape ``(map_h, map_w)``, bool.

    Returns
    -------
    jax.Array
        Scalar float32, ``1 / (1 + d)``. A player on a matching tile scores
        1.0, and the value falls with distance. A mask that is False
        everywhere does not score 0. The distance becomes ``map_h + map_w``,
        so the lowest value is ``1 / (1 + map_h + map_w)``. That is a small
        positive number, and it changes with the map size. A caller that
        compares rewards across map sizes must allow for this.
    """
    pos = state.player_positions[state.selected_player]
    px, py = pos[0], pos[1]
    map_h, map_w = state.map.shape
    grid_y, grid_x = jnp.meshgrid(jnp.arange(map_h), jnp.arange(map_w), indexing="ij")
    dist = jnp.abs(grid_x - px) + jnp.abs(grid_y - py)
    large = jnp.int32(map_h + map_w)
    min_dist = jnp.min(jnp.where(tile_mask, dist, large))
    return 1.0 / (1.0 + min_dist.astype(jnp.float32))


def achievement_reward(
    prev_state: EnvState,
    new_state: EnvState,
    params: EnvParams,
    weights: jax.Array,
) -> jax.Array:
    """Sparse reward for newly unlocked achievements.

    The function compares ``achievements_unlocked`` in the two ``EnvState``
    objects, and returns the weighted sum of the slots that the step satisfied.
    The achievements sit on :class:`~factoriax.engine.state.EnvState` itself,
    and the ``achievement_fn`` constructor argument of the environment latches
    them in every step.

    An achievement pays one time. The reward is the weighted count of the bits
    that are set now and were not set before. A bit that stays set therefore
    scores nothing on a later step. A bit that returns to False also scores
    nothing, and never a negative amount.

    Parameters
    ----------
    prev_state
        State just before the step.
    new_state
        State just after the step.
    params
        The function does not read this argument. It is present so that the
        signature matches the shared reward signature after
        :func:`functools.partial` binds ``weights``.
    weights
        Reward amount for each slot, shape ``(MAX_ACHIEVEMENTS,)``. Build it
        with :func:`factoriax.engine.achievements.achievement_weights`. This
        argument has no default, because a bit index means a different
        achievement in each scenario.

    Returns
    -------
    jax.Array
        Scalar float32. The value is zero on every step that unlocks nothing,
        which is almost every step. The value is never negative.
    """
    newly_unlocked = new_state.achievements_unlocked & ~prev_state.achievements_unlocked
    reward: jax.Array = jnp.sum(weights * newly_unlocked)
    return reward


def mining_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward: proximity to ore, plus a bonus for each ore mined.

    The function adds two terms:

    - **Proximity**: ``0.05 / (1 + d)``, where ``d`` is the Manhattan distance
      in tiles from the selected player to the nearest block in
      ``MINEABLE_BLOCKS``. The value is 0.05 at most, and never 0.
    - **Mining bonus**: 20.0 for each ore item that this step extracted. This
      is the rise in ``items_mined``, added over coal, iron ore, and copper
      ore.

    The scale of 400:1 between the two terms is deliberate. Proximity only
    separates steps that mine nothing. One ore is worth more than any distance
    to ore.

    The two terms disagree about the meaning of ore. Proximity targets every
    block in ``MINEABLE_BLOCKS``, which includes tin, silicon, and limestone.
    The bonus counts coal, iron ore, and copper ore only. An agent that follows
    the proximity term to a silicon patch and mines there gets no ore bonus.

    Parameters
    ----------
    prev_state
        State just before the step.
    new_state
        State just after the step.
    params
        The function does not read this argument. It is present for the shared
        reward signature.

    Returns
    -------
    jax.Array
        Scalar float32. The value is always positive, because the lowest
        proximity value is positive. It is about 0.05 or less on a step that
        mines nothing.
    """
    player_pos = new_state.player_positions[new_state.selected_player]
    px, py = player_pos[0], player_pos[1]
    map_h, map_w = new_state.map.shape

    grid_y, grid_x = jnp.meshgrid(jnp.arange(map_h), jnp.arange(map_w), indexing="ij")
    dist = jnp.abs(grid_x - px) + jnp.abs(grid_y - py)

    # Find the ore tiles: broadcast (H, W, 1) == (3,) -> (H, W, 3) -> (H, W)
    is_ore = jnp.any(new_state.map[..., None] == MINEABLE_BLOCKS, axis=-1)

    large = jnp.int32(map_h + map_w)
    ore_dist = jnp.where(is_ore, dist, large)
    min_dist = jnp.min(ore_dist)

    proximity: jax.Array = 0.05 / (1.0 + min_dist.astype(jnp.float32))

    ore_items = jnp.array(
        [ItemType.COAL, ItemType.IRON_ORE, ItemType.COPPER_ORE], dtype=jnp.int32
    )
    mined_delta = jnp.sum(
        new_state.items_mined[ore_items] - prev_state.items_mined[ore_items]
    )
    mining_bonus: jax.Array = 20.0 * mined_delta.astype(jnp.float32)

    return proximity + mining_bonus


def sparse_mining_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Sparse reward of 1.0 for each ore item mined during this step.

    The function adds the difference in ``items_mined`` between the two states,
    over coal, iron, and copper. The value is zero on every step that extracts
    nothing. This makes behaviour harder to shape, but the number is easy to
    read: one unit of reward for one unit of ore.

    The function counts coal, iron ore, and copper ore only. A player can also
    mine tin, silicon, and limestone, and those score nothing.

    Parameters
    ----------
    prev_state
        State just before the step.
    new_state
        State just after the step.
    params
        The function does not read this argument. It is present for the shared
        reward signature.

    Returns
    -------
    jax.Array
        Scalar float32, one unit for each ore item extracted. The value is zero
        on most steps.
    """
    ore_items = jnp.array(
        [ItemType.COAL, ItemType.IRON_ORE, ItemType.COPPER_ORE], dtype=jnp.int32
    )
    delta = jnp.sum(
        new_state.items_mined[ore_items] - prev_state.items_mined[ore_items]
    )
    reward: jax.Array = delta.astype(jnp.float32)
    return reward


def sparse_pallet_crafting_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Sparse reward of 1.0 for each pallet that a craft produced.

    The function finds a craft in two facts together: the inventory of the
    player gained pallets, and it lost iron in the same step. A move of pallets
    between the inventory and a machine changes the pallet count and consumes
    no iron. A cycle of place, pickup, deposit, or withdraw therefore pays
    nothing.

    Parameters
    ----------
    prev_state
        State just before the step.
    new_state
        State just after the step.
    params
        The function does not read this argument. It is present for the shared
        reward signature.

    Returns
    -------
    jax.Array
        Scalar float32, one unit for each pallet crafted. The value is zero
        when pallets appeared and the step spent no iron. That test is what
        refuses the reward for a move of items.
    """
    p = new_state.selected_player
    pallet_delta = (
        new_state.player_inventory[p, ItemType.PALLET]
        - prev_state.player_inventory[p, ItemType.PALLET]
    )
    iron_delta = (
        new_state.player_inventory[p, ItemType.IRON_ORE]
        - prev_state.player_inventory[p, ItemType.IRON_ORE]
    )

    # A craft consumes iron and produces pallets in the same step.
    is_craft = (pallet_delta > 0) & (iron_delta < 0)
    reward: jax.Array = jnp.where(is_craft, pallet_delta, 0).astype(jnp.float32)
    return reward


def sparse_miner_crafting_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Sparse reward of 1.0 for each miner that a craft produced.

    The function finds a craft in three facts together: the inventory of the
    player gained miners, and it lost both iron and copper in the same step. A
    move of miners between the inventory and the map, through place or pickup,
    consumes no resources. Those actions therefore pay nothing.

    Parameters
    ----------
    prev_state
        State just before the step.
    new_state
        State just after the step.
    params
        The function does not read this argument. It is present for the shared
        reward signature.

    Returns
    -------
    jax.Array
        Scalar float32, one unit for each miner crafted. The value is zero
        unless both iron and copper fell in the same step.
    """
    p = new_state.selected_player
    miner_delta = (
        new_state.player_inventory[p, ItemType.MINER]
        - prev_state.player_inventory[p, ItemType.MINER]
    )
    iron_delta = (
        new_state.player_inventory[p, ItemType.IRON_ORE]
        - prev_state.player_inventory[p, ItemType.IRON_ORE]
    )
    copper_delta = (
        new_state.player_inventory[p, ItemType.COPPER_ORE]
        - prev_state.player_inventory[p, ItemType.COPPER_ORE]
    )

    is_craft = (miner_delta > 0) & (iron_delta < 0) & (copper_delta < 0)
    reward: jax.Array = jnp.where(is_craft, miner_delta, 0).astype(jnp.float32)
    return reward


def miner_output_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Score the change in the ore that sits in the miner buffers.

    This function measures a level, not a flow. It is the rise in
    ``ent_buf_count``, added over the placed miners. It therefore pays for ore
    that stays in a miner, and not for ore that a miner produced.

    CAUTION: This reward pays for the opposite of automation. Use
    :func:`miner_throughput_reward` for a dense mining signal. A miner that
    pushes its output into a pallet or a belt ends the step as empty as it
    started, and scores 0. An idle miner that nothing empties scores 3, the
    default mining rate. On a one-tile coal patch the measured values are 0.0
    for a miner that faces a pallet, and 3.0 for the same miner that faces
    nothing. A step that empties a miner scores a negative amount, because the
    buffer falls.

    Parameters
    ----------
    prev_state
        State just before the step.
    new_state
        State just after the step.
    params
        The function does not read this argument. It is present for the shared
        reward signature.

    Returns
    -------
    jax.Array
        Scalar float32. The value is negative on a step where the miner
        buffers lose more than they gain.
    """
    is_miner = (new_state.ent_type == Machine.MINER) & (new_state.ent_y >= 0)
    prev_output = jnp.where(is_miner, prev_state.ent_buf_count, 0)
    new_output = jnp.where(is_miner, new_state.ent_buf_count, 0)
    delta = jnp.sum(new_output) - jnp.sum(prev_output)
    reward: jax.Array = delta.astype(jnp.float32)
    return reward


def miner_throughput_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Reward for the ore that placed miners take out of the ground each tick.

    The function measures the fall in ``block_resources`` on the tiles that
    hold a miner. It counts the extraction from the ground itself, and not a
    change in an output slot. A withdraw from the output slot, or no withdraw
    at all, therefore does not change the value.

    This function measures a flow, and :func:`miner_output_reward` measures a
    level. The value here does not depend on whether something empties the
    miner. The function clamps the fall on each tile at zero, so a tile with
    more resources than before cannot cancel a tile that a miner mined.

    Parameters
    ----------
    prev_state
        State just before the step.
    new_state
        State just after the step.
    params
        The function does not read this argument. It is present for the shared
        reward signature.

    Returns
    -------
    jax.Array
        Scalar float32, the ore taken out of the ground this step. Never
        negative.
    """
    is_miner = new_state.machine_types == Machine.MINER
    prev_res = prev_state.block_resources.astype(jnp.int32)
    new_res = new_state.block_resources.astype(jnp.int32)
    depleted = jnp.where(is_miner, prev_res - new_res, 0)
    reward: jax.Array = jnp.sum(jnp.maximum(depleted, 0)).astype(jnp.float32)
    return reward


def pallet_filling_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Score the change in how much is sitting in pallet buffers.

    The function pays for each item as it arrives and does not wait for a full
    stack, so one deposit scores at once.

    This function measures a level and not a flow, the same as
    :func:`miner_output_reward`. The route that an item took to reach a pallet
    does not matter, so a player deposit and an arm delivery score the same. A
    step that empties a pallet scores a negative amount.

    Parameters
    ----------
    prev_state
        State just before the step.
    new_state
        State just after the step.
    params
        The function does not read this argument. It is present for the shared
        reward signature.

    Returns
    -------
    jax.Array
        Scalar float32. The value is negative on a step where the pallets lose
        more than they gain.
    """
    is_pallet = (new_state.ent_type == Machine.PALLET) & (new_state.ent_y >= 0)
    prev_counts = jnp.where(is_pallet, prev_state.ent_buf_count, 0)
    new_counts = jnp.where(is_pallet, new_state.ent_buf_count, 0)
    delta = jnp.sum(new_counts) - jnp.sum(prev_counts)
    reward: jax.Array = delta.astype(jnp.float32)
    return reward


def player_inventory_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Reward for each item that the selected player gains.

    The function adds the rise in the item counts over all inventory slots. The
    value is positive when the step adds items, as a withdraw or a mine does.
    It is zero or negative when the step consumes items, as a craft, a deposit,
    or a placement does.

    Every item counts the same. A coal and an assembler are both worth 1, so
    this reward cannot say that some items are harder to get than others.

    Parameters
    ----------
    prev_state
        State just before the step.
    new_state
        State just after the step.
    params
        The function does not read this argument. It is present for the shared
        reward signature.

    Returns
    -------
    jax.Array
        Scalar float32. The value is negative when the player spends or stores
        more than it takes up. A craft, a placement, and a deposit all do this.
    """
    p = new_state.selected_player
    prev_total = jnp.sum(prev_state.player_inventory[p])
    new_total = jnp.sum(new_state.player_inventory[p])
    reward: jax.Array = (new_total - prev_total).astype(jnp.float32)
    return reward


# ---------------------------------------------------------------------------
# Dense reward functions for basic_skills scenario levels
# ---------------------------------------------------------------------------


def _ore_proximity(state: EnvState) -> jax.Array:
    """Score how close the selected player is to any mineable block.

    The function targets every block in ``MINEABLE_BLOCKS``. That set is wider
    than the three ores that the mining deltas count.

    Parameters
    ----------
    state
        State to read.

    Returns
    -------
    jax.Array
        Scalar float32 from :func:`_proximity`.
    """
    is_ore = jnp.any(state.map[..., None] == MINEABLE_BLOCKS, axis=-1)
    return _proximity(state, is_ore)


def _mining_delta(prev: EnvState, new: EnvState) -> jax.Array:
    """Count the ore taken out of the ground between two states.

    The function counts coal, iron ore, and copper ore only, the same set as
    :func:`sparse_mining_reward`.

    Parameters
    ----------
    prev
        State just before the step.
    new
        State just after the step.

    Returns
    -------
    jax.Array
        Scalar float32. Zero on a step that mines nothing.
    """
    ore = jnp.array(
        [ItemType.COAL, ItemType.IRON_ORE, ItemType.COPPER_ORE], dtype=jnp.int32
    )
    return jnp.sum(new.items_mined[ore] - prev.items_mined[ore]).astype(jnp.float32)


def _pallet_filling_delta(prev: EnvState, new: EnvState) -> jax.Array:
    """Measure the change in the contents of the pallets between two states.

    Parameters
    ----------
    prev
        State just before the step.
    new
        State just after the step.

    Returns
    -------
    jax.Array
        Scalar float32. The value is negative when the pallets lose more than
        they gain.
    """
    is_pallet = (new.ent_type == Machine.PALLET) & (new.ent_y >= 0)
    prev_c = jnp.sum(jnp.where(is_pallet, prev.ent_buf_count, 0))
    new_c = jnp.sum(jnp.where(is_pallet, new.ent_buf_count, 0))
    return (new_c - prev_c).astype(jnp.float32)


def _inventory_delta(prev: EnvState, new: EnvState) -> jax.Array:
    """Measure the change in the total item count of the selected player.

    Parameters
    ----------
    prev
        State just before the step.
    new
        State just after the step.

    Returns
    -------
    jax.Array
        Scalar float32. Negative when the player spends more than it gains.
    """
    p = new.selected_player
    return (
        jnp.sum(new.player_inventory[p]) - jnp.sum(prev.player_inventory[p])
    ).astype(jnp.float32)


def _item_count(state: EnvState, item: int) -> jax.Array:
    """Read the number of one item that player 0 carries.

    The function reads player 0 directly, and not ``selected_player``. The two
    agree only in a single-player scenario.

    Parameters
    ----------
    state
        State to read.
    item
        Item id to look up.

    Returns
    -------
    jax.Array
        Scalar int16 count.
    """
    return state.player_inventory[0, item]


def dense_craft_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward for crafting levels (craft_pallets, craft_miners).

    The function adds ore proximity, ore mined, and 10.0 for each placeable
    item gained. It counts miners, pallets, belts, and assemblers together.

    CAUTION: The craft term does not test that the step spent materials. It is
    the rise in the total placeable count, clamped at zero. A pickup of a
    machine from the map therefore pays the same 10.0 as a craft. Use
    :func:`sparse_pallet_crafting_reward` or
    :func:`sparse_miner_crafting_reward` when that difference matters.

    The craft term reads player 0, and not ``selected_player``.

    Parameters
    ----------
    prev_state
        State just before the step.
    new_state
        State just after the step.
    params
        The function does not read this argument. It is present for the shared
        reward signature.

    Returns
    -------
    jax.Array
        Scalar float32, always positive.
    """
    proximity = _ore_proximity(new_state)
    mining = _mining_delta(prev_state, new_state)

    # Find a craft: the total count of placeable items went up.
    placeables = jnp.array(
        [
            ItemType.MINER,
            ItemType.PALLET,
            ItemType.CONVEYOR_BELT,
            ItemType.ASSEMBLER,
        ],
        dtype=jnp.int32,
    )
    prev_count = jnp.sum(prev_state.player_inventory[0, placeables])
    new_count = jnp.sum(new_state.player_inventory[0, placeables])
    craft_delta = jnp.maximum(new_count - prev_count, 0)

    return proximity + mining + 10.0 * craft_delta.astype(jnp.float32)


def dense_fill_pallet_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward for the fill_pallet level.

    The function adds ore proximity, pallet proximity, ore mined, and 5.0 for
    each item that arrived in a pallet.

    Parameters
    ----------
    prev_state
        State just before the step.
    new_state
        State just after the step.
    params
        The function does not read this argument. It is present for the shared
        reward signature.

    Returns
    -------
    jax.Array
        Scalar float32. The value can go negative on a step that empties a
        pallet, because the filling term is a level and the lowest proximity
        value is small.
    """
    ore_prox = _ore_proximity(new_state)
    pallet_prox = _proximity(new_state, new_state.machine_types == Machine.PALLET)
    mining = _mining_delta(prev_state, new_state)
    filling = _pallet_filling_delta(prev_state, new_state)
    return ore_prox + pallet_prox + mining + 5.0 * filling


def dense_deploy_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward for the levels that ask a player to place a miner and run it.

    The levels ``deploy_miner``, ``place_and_fuel``, and ``mining_factory`` use
    this function. It adds ore proximity, ore mined, and 5.0 times
    :func:`miner_output_reward`.

    CAUTION: That last term has the fault that its own docstring describes. It
    pays for ore that stays in a miner. A miner connected to a belt therefore
    scores 0 on that term, and a miner that nothing empties scores 15. On the
    deployment levels this pays for a miner that a player places and leaves,
    which is what those levels ask for. It does not fit a level about the
    movement of the ore after that.

    Parameters
    ----------
    prev_state
        State just before the step.
    new_state
        State just after the step.
    params
        The function does not read this argument. It is present for the shared
        reward signature.

    Returns
    -------
    jax.Array
        Scalar float32. The value is negative on a step that empties the miners
        faster than the mining term and the proximity term add.
    """
    proximity = _ore_proximity(new_state)
    mining = _mining_delta(prev_state, new_state)
    output = miner_output_reward(prev_state, new_state, params)
    return proximity + mining + 5.0 * output


def dense_withdraw_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward for the withdraw_ore level.

    The function adds proximity to the nearest miner that still holds output,
    and 10.0 for each item that the player gained.

    CAUTION: The withdraw term counts every inventory gain, and not the
    withdraws. A mine into the inventory pays the same 10.0.

    Parameters
    ----------
    prev_state
        State just before the step.
    new_state
        State just after the step.
    params
        The function does not read this argument. It is present for the shared
        reward signature.

    Returns
    -------
    jax.Array
        Scalar float32, always positive. The gain term is clamped at zero, so a
        step that spends items costs nothing.
    """
    has_output = (
        (new_state.ent_type == Machine.MINER)
        & (new_state.ent_y >= 0)
        & (new_state.ent_buf_count > 0)
    )
    # Build a tile mask from the entity positions for the proximity term. Clip
    # the positions first. A free slot holds -1, which indexes the last row and
    # the last column instead of dropping out. Then use OR and not an
    # assignment, so the False of a free slot cannot overwrite the True of a
    # real machine on a tile that both reach.
    h, w = new_state.map.shape
    ey = jnp.clip(new_state.ent_y, 0, h - 1)
    ex = jnp.clip(new_state.ent_x, 0, w - 1)
    has_output_grid = jnp.zeros((h, w), dtype=jnp.bool_).at[ey, ex].max(has_output)
    proximity = _proximity(new_state, has_output_grid)
    inv_gain = _inventory_delta(prev_state, new_state)
    return proximity + 10.0 * jnp.maximum(inv_gain, 0.0)


def dense_deposit_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward for the deposit_into_pallets level.

    The function adds proximity to the nearest pallet, and 5.0 for each item
    that arrived in a pallet.

    Parameters
    ----------
    prev_state
        State just before the step.
    new_state
        State just after the step.
    params
        The function does not read this argument. It is present for the shared
        reward signature.

    Returns
    -------
    jax.Array
        Scalar float32. The value is negative on a step that empties a pallet.
    """
    pallet_prox = _proximity(new_state, new_state.machine_types == Machine.PALLET)
    filling = _pallet_filling_delta(prev_state, new_state)
    return pallet_prox + 5.0 * filling


def dense_pickup_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward for the pickup_machines level.

    The function adds proximity to the nearest machine, and 10.0 for each
    machine that left the map in this step.

    CAUTION: A placement costs nothing, because the count term is clamped at
    zero. An agent can therefore place a machine and pick it up again for 10.0
    every two steps, without end. The level that uses this reward is short, so
    the loop does not control the score. The term does not fit a longer
    episode.

    Parameters
    ----------
    prev_state
        State just before the step.
    new_state
        State just after the step.
    params
        The function does not read this argument. It is present for the shared
        reward signature.

    Returns
    -------
    jax.Array
        Scalar float32, always positive.
    """
    has_machine = new_state.machine_types != Machine.NONE
    proximity = _proximity(new_state, has_machine)
    prev_count = jnp.sum(prev_state.machine_types != Machine.NONE)
    new_count = jnp.sum(new_state.machine_types != Machine.NONE)
    picked_up = jnp.maximum(prev_count - new_count, 0)
    return proximity + 10.0 * picked_up.astype(jnp.float32)


def dense_belt_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward for the belt_line level.

    The function adds 5.0 for each belt placed, and 5.0 for each item that
    arrived in a pallet.

    There is no proximity term. The level asks the player to close a gap in a
    belt line, and no field in the state marks the tile of the gap. There is
    therefore no target to measure a distance to.

    CAUTION: A belt on any tile pays, and not only a belt that closes the gap.

    Parameters
    ----------
    prev_state
        State just before the step.
    new_state
        State just after the step.
    params
        The function does not read this argument. It is present for the shared
        reward signature.

    Returns
    -------
    jax.Array
        Scalar float32. The value is zero on a step that does neither, and
        negative on a step that empties a pallet.
    """
    prev_belts = jnp.sum(prev_state.machine_types == Machine.CONVEYOR_BELT)
    new_belts = jnp.sum(new_state.machine_types == Machine.CONVEYOR_BELT)
    belt_placed = jnp.maximum(new_belts - prev_belts, 0)
    filling = _pallet_filling_delta(prev_state, new_state)
    return 5.0 * belt_placed.astype(jnp.float32) + 5.0 * filling


def dense_assembler_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward for the assembler_production level.

    The function adds proximity to the nearest assembler, 2.0 for each item
    that arrived in an assembler input slot, and 10.0 for each tier-1 science
    pack that the player gained.

    Both delta terms are clamped at zero. An assembler that consumes its inputs
    to start a craft therefore costs nothing, although the input count falls.
    The pack count reads player 0, and not ``selected_player``.

    Parameters
    ----------
    prev_state
        State just before the step.
    new_state
        State just after the step.
    params
        The function does not read this argument. It is present for the shared
        reward signature.

    Returns
    -------
    jax.Array
        Scalar float32, always positive.
    """
    asm_prox = _proximity(new_state, new_state.machine_types == Machine.ASSEMBLER)
    # The deposited input is the total of the items in the assembler buffers.
    is_asm = (new_state.ent_type == Machine.ASSEMBLER) & (new_state.ent_y >= 0)
    prev_inputs = jnp.sum(
        jnp.where(
            is_asm[:, None],
            prev_state.ent_asm_in_count,
            0,
        )
    )
    new_inputs = jnp.sum(
        jnp.where(
            is_asm[:, None],
            new_state.ent_asm_in_count,
            0,
        )
    )
    input_delta = jnp.maximum(new_inputs - prev_inputs, 0).astype(jnp.float32)
    # Science packs that arrived in the inventory of the player.
    prev_packs = _item_count(prev_state, ItemType.TIER1_SCIENCE_PACK)
    new_packs = _item_count(new_state, ItemType.TIER1_SCIENCE_PACK)
    pack_delta = jnp.maximum(new_packs - prev_packs, 0).astype(jnp.float32)
    return asm_prox + 2.0 * input_delta + 10.0 * pack_delta
