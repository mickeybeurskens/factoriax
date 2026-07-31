"""The environment's state and its per-episode parameters.

:class:`EnvState` is everything a step reads and writes. :class:`EnvParams`
is the tuning that stays fixed while an episode runs. Both are
``flax.struct.PyTreeNode``, so their array fields are PyTree leaves and both
pass through ``jax.jit`` and ``jax.vmap`` without further registration.

Machine state is held as entity lists, not as one grid per attribute. Every
``ent_`` array is indexed by entity id and sized to a capacity fixed when the
level is built, so shapes stay static however much gets constructed during an
episode.
"""

from typing import ClassVar

import jax.numpy as jnp
from flax import struct

from factoriax.engine.constants import Action
from factoriax.engine.recipes import DEFAULT_RECIPE_TABLE, RecipeTable


class EnvState(struct.PyTreeNode):  # type: ignore[no-untyped-call]
    """Everything one environment step reads and writes.

    Machine state lives in the ``ent_`` arrays, indexed by entity id rather
    than by tile. A slot is free when ``ent_y < 0``: placement claims the
    first free one and pickup releases it. The capacity is fixed at level
    build, so a map with every slot taken refuses further placement instead of
    growing an array.

    Two fields deliberately hold the same fact. ``machine_types`` answers
    "is something on this tile" for movement and placement, while ``ent_type``
    carries the kind per entity for the machine tick. ``tile_entity`` joins
    them, and placement writes all three together.

    Shapes below use ``H`` and ``W`` for the map, ``E`` for the entity
    capacity, and ``P`` for the player count.

    Attributes
    ----------
    map
        Terrain, one :class:`~factoriax.engine.constants.BlockType` per tile.
        Shape ``(H, W)``, int8.
    block_resources
        Ore units left in each tile, decremented as it is mined. Shape
        ``(H, W)``, int16. Zero means exhausted. The value is meaningless on a
        tile that was never an ore block.
    machine_types
        :class:`~factoriax.engine.constants.Machine` kind occupying each tile,
        ``NONE`` where the tile is clear. Shape ``(H, W)``, int8.
    tile_entity
        Entity id occupying each tile, ``-1`` where none. Shape ``(H, W)``,
        int16. This is how a neighbour lookup gets from a tile to entity state.
    ent_y
        Tile row of each entity. Shape ``(E,)``, int16. Negative marks a free
        slot, and is the test the engine uses for "this entity is not placed".
    ent_x
        Tile column of each entity. Shape ``(E,)``, int16.
    ent_type
        Machine kind per entity. Shape ``(E,)``, int8.
    ent_direction
        Facing per entity, as a
        :class:`~factoriax.engine.constants.Direction` value, ``0`` when
        unset. Shape ``(E,)``, int8. A crossing packs both its axes into this
        one byte; ``factoriax.engine.tables.CROSSING_AXIS_DIRS`` unpacks it.
    ent_power
        Steps left in the machine's current craft, counting down. Shape
        ``(E,)``, int16. The craft finishes on the step this reaches 1, and 0
        means idle. Only assemblers and furnaces use it; every other kind
        holds 0.
    ent_buf_type
        Item in the general-purpose buffer slot. Shape ``(E,)``, int8. This is
        a miner's output, a pallet's storage, and a belt's contents.
    ent_buf_count
        Items in that buffer, capped per kind by
        ``factoriax.engine.tables.MACHINE_MAX_STACK``. Shape ``(E,)``, int16.
    ent_asm_in_type
        Item in each of the two input slots. Shape ``(E, 2)``, int8. Used by
        assemblers, furnaces, and science labs, and reused by a crossing to
        hold one buffer per axis.
        A crossing reuses the pair as one buffer per axis, indexed by
        ``CROSSING_VERT_SLOT`` and ``CROSSING_HORIZ_SLOT``.
    ent_asm_in_count
        Items in each input slot, aligned with ``ent_asm_in_type``. Shape
        ``(E, 2)``, int16.
    ent_asm_out_type
        Item in the finished-output slot. Shape ``(E,)``, int8. A completed
        craft parks here and is not overwritten, so the machine stalls until
        something withdraws it.
    ent_asm_out_count
        Items in the output slot. Shape ``(E,)``, int16.
    ent_health
        Hit points per entity. Shape ``(E,)``, int16. Placement sets
        ``MACHINE_MAX_HEALTH``, repair clamps back to it, and pickup is
        refused below it.
    player_positions
        ``(x, y)`` tile of each player. Shape ``(P, 2)``, int16.
    player_directions
        Facing of each player, a ``Direction`` value. Shape ``(P,)``, int8.
    player_inventory
        Items carried, per player and item id. Shape
        ``(P, NUM_ITEM_TYPES)``, int16. Column 0 is ``ItemType.EMPTY`` and
        stays zero. Per-item caps come from
        ``factoriax.engine.tables.PLAYER_MAX_STACK``.
    selected_player
        Index of the player an action applies to. Scalar int32. The
        single-agent step drives this one.
    timestep
        Steps taken since reset. Scalar int32. The episode ends when it
        reaches ``EnvParams.max_timesteps``.
    items_mined
        Running total mined per item id since reset, never reset mid-episode.
        Shape ``(NUM_ITEM_TYPES,)``, int32. Reward functions read the
        difference between two states rather than the total.
    science_consumed_step
        Science packs consumed during the last step only, indexed by position
        in :data:`~factoriax.engine.constants.SCIENCE_PACK_TYPES`. Shape
        ``(NUM_SCIENCE_PACK_TYPES,)``, int32. This one is a per-step delta,
        not a running total.
    achievements_unlocked
        One latched bit per achievement. Shape ``(MAX_ACHIEVEMENTS,)``, bool.
        A bit is OR-folded in when its condition first holds and stays set for
        the rest of the episode. A scenario with fewer achievements leaves the
        trailing bits False.

    Examples
    --------
    >>> import jax
    >>> from factoriax.engine.constants import NUM_ITEM_TYPES
    >>> from factoriax.make import env_from_name
    >>> env, params = env_from_name("EasyRocket-v1", auto_reset=False)
    >>> _, state = env.reset_env(jax.random.PRNGKey(0), params)
    >>> state.player_inventory.shape == (1, NUM_ITEM_TYPES)
    True
    """

    # Grid (terrain + spatial lookup)
    map: jnp.ndarray
    block_resources: jnp.ndarray
    machine_types: jnp.ndarray
    tile_entity: jnp.ndarray

    # Entity arrays (machine state)
    ent_y: jnp.ndarray
    ent_x: jnp.ndarray
    ent_type: jnp.ndarray
    ent_direction: jnp.ndarray
    ent_power: jnp.ndarray
    ent_buf_type: jnp.ndarray
    ent_buf_count: jnp.ndarray
    ent_asm_in_type: jnp.ndarray
    ent_asm_in_count: jnp.ndarray
    ent_asm_out_type: jnp.ndarray
    ent_asm_out_count: jnp.ndarray
    ent_health: jnp.ndarray

    # Player
    player_positions: jnp.ndarray
    player_directions: jnp.ndarray
    player_inventory: jnp.ndarray
    selected_player: int

    # Progress
    timestep: int
    items_mined: jnp.ndarray
    science_consumed_step: jnp.ndarray
    achievements_unlocked: jnp.ndarray


class EnvParams(struct.PyTreeNode):  # type: ignore[no-untyped-call]
    """Tuning that stays fixed for the length of an episode.

    Passed to every engine function alongside the state, so a scenario can
    change balance without editing engine code. The defaults describe the
    generated sandbox map; a scenario shipping its own level overrides the
    terrain fields, which then go unread.

    The six terrain probabilities apply only when a level is generated rather
    than loaded. Each is tested independently against its own smooth-noise
    field, so they are per-tile shares rather than bands and need not sum to
    1. Where two draws claim the same tile the later assignment wins, in the
    order silicon, tin, coal, copper, iron, water, so water displaces any ore
    it overlaps. A tile no draw claims stays plain ground.

    Attributes
    ----------
    max_timesteps
        Steps before the episode reports done. Compared against
        ``EnvState.timestep``.
    water_probability
        Share of generated tiles that become water, which blocks movement and
        placement.
    iron_probability
        Share of generated tiles that become iron ore.
    copper_probability
        Share of generated tiles that become copper ore.
    coal_probability
        Share of generated tiles that become coal.
    tin_probability
        Share of generated tiles that become tin ore.
    silicon_probability
        Share of generated tiles that become silicon.
    base_resources
        Ore units every mineable tile starts with, written into
        ``EnvState.block_resources``; a non-mineable tile gets 0. Must fit
        int16. Above
        :data:`~factoriax.engine.constants.BLOCK_MAX_RESOURCES` the tile
        still works, but it normalizes past 1.0 in the observation.
    miner_mining_rate
        Ore a placed miner extracts per step. Its output slot holds
        ``factoriax.engine.machines.MINER_OUTPUT_CAP``, so a miner fills up in
        one step and then idles until something drains it.
    player_mining_yield
        Ore one ``MINE`` action gives the player. Capped by what the tile has
        left and by the player's stack limit.
    recipe_table
        Recipes this episode runs, as a
        :class:`~factoriax.engine.recipes.RecipeTable`. Defaults to the
        engine's own book. Engine code sizes its recipe loops from this table
        rather than from the module-level default, so a scenario may ship a
        different recipe count.
    """

    max_timesteps: int = 1000
    water_probability: float = 0.1
    iron_probability: float = 0.12
    copper_probability: float = 0.12
    coal_probability: float = 0.12
    tin_probability: float = 0.10
    silicon_probability: float = 0.10
    base_resources: int = 1000
    miner_mining_rate: int = 3
    player_mining_yield: int = 1
    recipe_table: RecipeTable = DEFAULT_RECIPE_TABLE

    #: Size of the action space. A ``ClassVar``, so it is neither a PyTree
    #: leaf nor tunable per episode: the action space is fixed by the
    #: :class:`~factoriax.engine.constants.Action` enum.
    NUM_ACTIONS: ClassVar[int] = len(Action)
