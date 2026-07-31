"""Score what changed in one step.

A reward function reads the state before a step and the state after it, and
returns one number saying how good that step was. It never changes the
state. Every function here takes ``(prev_state, new_state, params)`` and
returns a scalar float32, so an environment can be handed any of them, and
:func:`functools.partial` can bind the extra argument the two-argument ones
need.

The functions split into two kinds, and the difference matters more than the
names suggest.

A **sparse** reward pays only when the thing you actually want happens: an
ore is mined, an achievement unlocks. It is honest and hard to learn from,
because most steps score zero.

A **dense** reward adds shaping terms that pay a little for being near
something useful or for a quantity moving the right way. It is easier to
learn from and easier to get wrong, because an agent optimises the terms it
is given rather than the goal behind them. Two of the shaping terms here
have that problem and say so in their own docstrings.

Proximity terms measure Manhattan distance from ``state.selected_player``,
not from a player passed in, so under multiple players they always score the
selected one.

Rewards are not clamped and several can go negative, because they are
differences between two states and a quantity can fall. Read each function's
return description rather than assuming a floor of zero.
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

    The shaping term the dense rewards are built from. It turns a distance
    into a number that rises as the player approaches, so an agent gets a
    gradient to follow instead of a flat zero until it arrives.

    Distance is Manhattan, in tiles, measured from
    ``state.player_positions[state.selected_player]``. Other players are
    ignored.

    Parameters
    ----------
    state
        State to read. Uses ``map`` for the shape and ``player_positions``.
    tile_mask
        Which tiles count as a target. Shape ``(map_h, map_w)``, bool.

    Returns
    -------
    jax.Array
        Scalar float32, ``1 / (1 + d)``. Standing on a matching tile scores
        1.0 and the value falls off with distance. An all-False mask does
        not score 0: the distance is replaced by ``map_h + map_w``, so the
        floor is ``1 / (1 + map_h + map_w)``, a small positive number that
        varies with map size. A caller comparing rewards across map sizes
        has to account for that.
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

    Compares ``achievements_unlocked`` between the two ``EnvState``
    instances and returns the weighted sum of newly satisfied slots.
    Achievements live on :class:`~factoriax.engine.state.EnvState`
    directly; the env's ``achievement_fn`` constructor argument latches
    them each step.

    An achievement pays once. The reward is the weighted count of bits that
    are set now and were not set before, so a bit that stays set scores
    nothing on later steps. A bit that somehow cleared scores nothing
    either, rather than a negative.

    Parameters
    ----------
    prev_state
        State immediately before the step.
    new_state
        State immediately after the step.
    params
        Unused. Present so this matches the shared reward signature once
        ``weights`` is bound with :func:`functools.partial`.
    weights
        Per-slot reward magnitudes, shape ``(MAX_ACHIEVEMENTS,)``. Build it
        with :func:`factoriax.engine.achievements.achievement_weights`.
        Required rather than defaulted, because a bit index means a
        different achievement in each scenario.

    Returns
    -------
    jax.Array
        Scalar float32. Zero on every step that unlocks nothing, which is
        almost all of them. Never negative.
    """
    newly_unlocked = new_state.achievements_unlocked & ~prev_state.achievements_unlocked
    reward: jax.Array = jnp.sum(weights * newly_unlocked)
    return reward


def mining_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward combining proximity to ore and a bonus for each ore mined.

    Two components are summed:

    - **Proximity**: ``0.05 / (1 + d)``, where ``d`` is the Manhattan
      distance in tiles from the selected player to the nearest block in
      ``MINEABLE_BLOCKS``. At most 0.05, and never 0.
    - **Mining bonus**: 20.0 per ore item extracted this step, the rise in
      ``items_mined`` summed over coal, iron ore, and copper ore.

    The two terms are scaled 400:1 on purpose. Proximity only breaks ties
    between steps that mine nothing; one ore outweighs any amount of
    standing near ore.

    The two terms disagree about what ore is. Proximity targets every block
    in ``MINEABLE_BLOCKS``, which includes tin, silicon, and limestone. The
    bonus counts only coal, iron ore, and copper ore. An agent led to a
    silicon patch by the proximity term and mining it there scores nothing
    for the ore.

    Parameters
    ----------
    prev_state
        State immediately before the step.
    new_state
        State immediately after the step.
    params
        Unused. Present for the shared reward signature.

    Returns
    -------
    jax.Array
        Scalar float32, always positive because the proximity floor is
        positive. Roughly 0.05 or less on a step that mines nothing.
    """
    player_pos = new_state.player_positions[new_state.selected_player]
    px, py = player_pos[0], player_pos[1]
    map_h, map_w = new_state.map.shape

    grid_y, grid_x = jnp.meshgrid(jnp.arange(map_h), jnp.arange(map_w), indexing="ij")
    dist = jnp.abs(grid_x - px) + jnp.abs(grid_y - py)

    # Identify ore tiles: broadcast (H, W, 1) == (3,) -> (H, W, 3) -> (H, W)
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

    Counts the total delta across coal, iron, and copper in ``items_mined``
    between the two states.  This signal is zero on every step where nothing
    is extracted, which makes it harder to shape behaviour but trivial to
    interpret: one unit of reward per one unit of ore.

    Counts coal, iron ore, and copper ore only. Tin, silicon, and limestone
    are mineable and score nothing.

    Parameters
    ----------
    prev_state
        State immediately before the step.
    new_state
        State immediately after the step.
    params
        Unused. Present for the shared reward signature.

    Returns
    -------
    jax.Array
        Scalar float32, one per ore item extracted. Zero on most steps.
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
    """Sparse reward of 1.0 for each pallet gained via crafting.

    Detects crafting by requiring that the player's inventory gained
    pallets *and* lost iron in the same step. Moving pallets between
    inventory and machines changes pallet count without consuming iron,
    so place/pickup/deposit/withdraw exploits yield zero reward.

    Parameters
    ----------
    prev_state
        State immediately before the step.
    new_state
        State immediately after the step.
    params
        Unused. Present for the shared reward signature.

    Returns
    -------
    jax.Array
        Scalar float32, one per pallet crafted. Zero when pallets appeared
        without iron being spent, which is how the move-shuffling exploits
        are refused.
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

    # Crafting consumes iron and produces pallets in the same step.
    is_craft = (pallet_delta > 0) & (iron_delta < 0)
    reward: jax.Array = jnp.where(is_craft, pallet_delta, 0).astype(jnp.float32)
    return reward


def sparse_miner_crafting_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Sparse reward of 1.0 for each miner gained via crafting.

    Detects crafting by requiring that the player's inventory gained
    miners *and* lost both iron and copper in the same step. Moving
    miners between inventory and the map via place/pickup does not
    consume resources, so those actions yield zero reward.

    Parameters
    ----------
    prev_state
        State immediately before the step.
    new_state
        State immediately after the step.
    params
        Unused. Present for the shared reward signature.

    Returns
    -------
    jax.Array
        Scalar float32, one per miner crafted. Zero unless iron and copper
        both fell in the same step.
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
    """Score the change in how much ore is sitting in miner buffers.

    This measures a level, not a flow. It is the rise in ``ent_buf_count``
    summed over placed miners, so it pays for ore accumulating in a miner
    and not for ore being produced.

    That makes it reward the opposite of automation, and a caller choosing
    a dense mining signal should usually prefer
    :func:`miner_throughput_reward`. A miner pushing its output into a
    pallet or a belt ends the step as empty as it started and scores 0. An
    idle miner nobody drains scores 3, the default mining rate. Measured on
    a one-tile coal patch: miner facing a pallet, 0.0; the same miner facing
    nothing, 3.0. Draining a miner scores negative, because the buffer
    falls.

    Parameters
    ----------
    prev_state
        State immediately before the step.
    new_state
        State immediately after the step.
    params
        Unused. Present for the shared reward signature.

    Returns
    -------
    jax.Array
        Scalar float32. Negative on any step where miner buffers net drain.
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
    """Reward for ore extracted from blocks by placed miners each tick.

    Measures the decrease in ``block_resources`` on tiles that have a
    miner. This counts actual extraction from the ground rather than
    output slot changes, so it is unaffected by the agent withdrawing
    from or ignoring the output slot.

    Unlike :func:`miner_output_reward` this measures a flow, so it is
    unaffected by whether anything drains the miner. Per-tile falls are
    clamped at zero, so a tile whose resources somehow rose cannot cancel
    out a tile that was mined.

    Parameters
    ----------
    prev_state
        State immediately before the step.
    new_state
        State immediately after the step.
    params
        Unused. Present for the shared reward signature.

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

    Pays per item as it arrives rather than waiting for a full stack, so a
    single deposit scores immediately.

    Like :func:`miner_output_reward` this is a level and not a flow. It does
    not care how an item reached a pallet, so a player deposit and an arm
    delivery score the same, and emptying a pallet scores negative.

    Parameters
    ----------
    prev_state
        State immediately before the step.
    new_state
        State immediately after the step.
    params
        Unused. Present for the shared reward signature.

    Returns
    -------
    jax.Array
        Scalar float32. Negative on any step where pallets net drain.
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
    """Reward for each item gained in the selected player's inventory.

    Counts the total increase in item counts across all inventory slots.
    Positive when items are added (withdraw, mine), zero or negative when
    items are consumed (craft, deposit, place).

    Every item counts the same. A coal and an assembler are both worth 1,
    so this cannot express that some items are harder to get than others.

    Parameters
    ----------
    prev_state
        State immediately before the step.
    new_state
        State immediately after the step.
    params
        Unused. Present for the shared reward signature.

    Returns
    -------
    jax.Array
        Scalar float32. Negative whenever the player spends or stores more
        than it picks up, which includes crafting, placing, and depositing.
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

    Targets every block in ``MINEABLE_BLOCKS``, which is wider than the
    three ores the mining deltas count.

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

    Coal, iron ore, and copper ore only, matching
    :func:`sparse_mining_reward`.

    Parameters
    ----------
    prev
        State immediately before the step.
    new
        State immediately after the step.

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
    """Measure the change in how much pallets hold between two states.

    Parameters
    ----------
    prev
        State immediately before the step.
    new
        State immediately after the step.

    Returns
    -------
    jax.Array
        Scalar float32. Negative when pallets net drain.
    """
    is_pallet = (new.ent_type == Machine.PALLET) & (new.ent_y >= 0)
    prev_c = jnp.sum(jnp.where(is_pallet, prev.ent_buf_count, 0))
    new_c = jnp.sum(jnp.where(is_pallet, new.ent_buf_count, 0))
    return (new_c - prev_c).astype(jnp.float32)


def _inventory_delta(prev: EnvState, new: EnvState) -> jax.Array:
    """Measure the change in the selected player's total item count.

    Parameters
    ----------
    prev
        State immediately before the step.
    new
        State immediately after the step.

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
    """Read how many of one item player 0 is carrying.

    Reads player 0 outright, not ``selected_player``. The two agree only in
    a single-player scenario.

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

    Sums ore proximity, ore mined, and 10.0 per placeable item gained,
    counting miners, pallets, belts, and assemblers together.

    The craft term does not check that materials were spent, despite what
    an earlier version of this text claimed. It is the rise in the total
    placeable count, clamped at zero, so picking a machine back up off the
    map pays the same 10.0 as crafting one. Use
    :func:`sparse_pallet_crafting_reward` or
    :func:`sparse_miner_crafting_reward` where that distinction matters.

    Reads player 0 rather than ``selected_player`` for the craft term.

    Parameters
    ----------
    prev_state
        State immediately before the step.
    new_state
        State immediately after the step.
    params
        Unused. Present for the shared reward signature.

    Returns
    -------
    jax.Array
        Scalar float32, always positive.
    """
    proximity = _ore_proximity(new_state)
    mining = _mining_delta(prev_state, new_state)

    # Detect any crafting: total placeable items increased.
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

    Sums ore proximity, pallet proximity, ore mined, and 5.0 per item that
    arrived in a pallet.

    Parameters
    ----------
    prev_state
        State immediately before the step.
    new_state
        State immediately after the step.
    params
        Unused. Present for the shared reward signature.

    Returns
    -------
    jax.Array
        Scalar float32. Can go negative on a step that empties a pallet,
        because the filling term is a level and the proximity floor is
        small.
    """
    ore_prox = _ore_proximity(new_state)
    pallet_prox = _proximity(new_state, new_state.machine_types == Machine.PALLET)
    mining = _mining_delta(prev_state, new_state)
    filling = _pallet_filling_delta(prev_state, new_state)
    return ore_prox + pallet_prox + mining + 5.0 * filling


def dense_deploy_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward for the levels that ask a miner to be placed and run.

    Used by ``deploy_miner``, ``place_and_fuel``, and ``mining_factory``.

    Sums ore proximity, ore mined, and 5.0 times
    :func:`miner_output_reward`.

    That last term carries the flaw described in its own docstring: it pays
    for ore piling up in a miner, so a miner wired into a belt scores 0 on
    it while an unattended one scores 15. On the deployment levels this
    rewards placing a miner and leaving it, which is what those levels ask
    for, but it does not extend to a level about moving the ore onward.

    Parameters
    ----------
    prev_state
        State immediately before the step.
    new_state
        State immediately after the step.
    params
        Unused. Present for the shared reward signature.

    Returns
    -------
    jax.Array
        Scalar float32. Negative on a step that drains miners faster than
        the mining and proximity terms make up.
    """
    proximity = _ore_proximity(new_state)
    mining = _mining_delta(prev_state, new_state)
    output = miner_output_reward(prev_state, new_state, params)
    return proximity + mining + 5.0 * output


def dense_withdraw_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward for the withdraw_ore level.

    Proximity to the nearest miner still holding output, plus 10.0 per item
    the player gained.

    The withdraw term counts any inventory gain, not withdrawals. Mining
    into the inventory pays the same 10.0.

    Parameters
    ----------
    prev_state
        State immediately before the step.
    new_state
        State immediately after the step.
    params
        Unused. Present for the shared reward signature.

    Returns
    -------
    jax.Array
        Scalar float32, always positive. The gain term is clamped at zero,
        so spending items costs nothing.
    """
    has_output = (
        (new_state.ent_type == Machine.MINER)
        & (new_state.ent_y >= 0)
        & (new_state.ent_buf_count > 0)
    )
    # Build a spatial mask from entity positions for proximity. Clip the
    # positions first: a free slot holds -1, which would index the last row
    # and column rather than being dropped. Then OR rather than assign, so a
    # free slot's False cannot overwrite a real machine's True on the tile
    # they collide on.
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

    Proximity to the nearest pallet plus 5.0 per item that arrived in one.

    Parameters
    ----------
    prev_state
        State immediately before the step.
    new_state
        State immediately after the step.
    params
        Unused. Present for the shared reward signature.

    Returns
    -------
    jax.Array
        Scalar float32. Negative on a step that empties a pallet.
    """
    pallet_prox = _proximity(new_state, new_state.machine_types == Machine.PALLET)
    filling = _pallet_filling_delta(prev_state, new_state)
    return pallet_prox + 5.0 * filling


def dense_pickup_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Dense reward for the pickup_machines level.

    Proximity to the nearest machine plus 10.0 per machine that left the
    map this step.

    Placing costs nothing, because the count term is clamped at zero. An
    agent can therefore place a machine and pick it straight back up for
    10.0 every two steps, indefinitely. The level this scores is short
    enough that the loop does not dominate, but the term does not
    generalise to a longer episode.

    Parameters
    ----------
    prev_state
        State immediately before the step.
    new_state
        State immediately after the step.
    params
        Unused. Present for the shared reward signature.

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

    5.0 per belt placed plus 5.0 per item that arrived in a pallet.

    No proximity term. The level asks the player to close a gap in a belt
    line, and nothing in the state marks which tile the gap is on, so there
    is no target to measure a distance to.

    Any belt placed anywhere pays, not only one that closes the gap.

    Parameters
    ----------
    prev_state
        State immediately before the step.
    new_state
        State immediately after the step.
    params
        Unused. Present for the shared reward signature.

    Returns
    -------
    jax.Array
        Scalar float32. Zero on a step that does neither, and negative on a
        step that empties a pallet.
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

    Proximity to the nearest assembler, 2.0 per item that arrived in an
    assembler input slot, and 10.0 per tier-1 science pack the player
    gained.

    Both delta terms are clamped at zero, so an assembler consuming its
    inputs to start a craft costs nothing even though the input count
    falls. Reads player 0 for the pack count, not ``selected_player``.

    Parameters
    ----------
    prev_state
        State immediately before the step.
    new_state
        State immediately after the step.
    params
        Unused. Present for the shared reward signature.

    Returns
    -------
    jax.Array
        Scalar float32, always positive.
    """
    asm_prox = _proximity(new_state, new_state.machine_types == Machine.ASSEMBLER)
    # Input deposited = total items in assembler entity buffers.
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
    # Science packs gained in player inventory.
    prev_packs = _item_count(prev_state, ItemType.TIER1_SCIENCE_PACK)
    new_packs = _item_count(new_state, ItemType.TIER1_SCIENCE_PACK)
    pack_delta = jnp.maximum(new_packs - prev_packs, 0).astype(jnp.float32)
    return asm_prox + 2.0 * input_delta + 10.0 * pack_delta
