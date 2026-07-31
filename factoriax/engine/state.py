"""State of the environment, and its per-episode parameters.

:class:`EnvState` is everything that a step reads and writes. :class:`EnvParams`
is the tuning that stays fixed while an episode runs. Both are
``flax.struct.PyTreeNode``, so their array fields are PyTree leaves. Both
therefore pass through ``jax.jit`` and ``jax.vmap`` with no further
registration.

The engine holds machine state as entity lists, and not as one grid for each
attribute. An entity id indexes every ``ent_`` array. The capacity is fixed when
the level is built, so the shapes stay static, whatever a player builds during
an episode.
"""

from typing import ClassVar

import jax.numpy as jnp
from flax import struct

from factoriax.engine.constants import Action
from factoriax.engine.recipes import DEFAULT_RECIPE_TABLE, RecipeTable


class EnvState(struct.PyTreeNode):  # type: ignore[no-untyped-call]
    """Everything one environment step reads and writes.

    Machine state lives in the ``ent_`` arrays. An entity id indexes them, not
    a tile. A slot is free when ``ent_y < 0``. Placement claims the first free
    slot, and pickup releases it. The capacity is fixed when the level is
    built. A map with every slot in use therefore refuses a new placement, and
    no array grows.

    Two fields hold the same fact, and this is deliberate. ``machine_types``
    answers "is something on this tile" for movement and placement.
    ``ent_type`` carries the kind of each entity for the machine tick.
    ``tile_entity`` joins the two, and placement writes all three together.

    The shapes below use ``H`` and ``W`` for the map, ``E`` for the entity
    capacity, and ``P`` for the player count.

    Attributes
    ----------
    map
        Terrain, one :class:`~factoriax.engine.constants.BlockType` per tile.
        Shape ``(H, W)``, int8.
    block_resources
        Ore units left in each tile. A mine action lowers the value. Shape
        ``(H, W)``, int16. Zero means that the tile is empty. The value has no
        meaning on a tile that was never an ore block.
    machine_types
        :class:`~factoriax.engine.constants.Machine` kind on each tile,
        ``NONE`` where the tile is clear. Shape ``(H, W)``, int8.
    tile_entity
        Entity id on each tile, ``-1`` where the tile holds no entity. Shape
        ``(H, W)``, int16. A neighbour lookup reads this array to go from a
        tile to entity state.
    ent_y
        Tile row of each entity. Shape ``(E,)``, int16. A negative value marks
        a free slot. This is the test that the engine uses for "this entity is
        not on the map".
    ent_x
        Tile column of each entity. Shape ``(E,)``, int16.
    ent_type
        Machine kind per entity. Shape ``(E,)``, int8.
    ent_direction
        Facing of each entity, as a
        :class:`~factoriax.engine.constants.Direction` value, ``0`` when
        nothing has set it. Shape ``(E,)``, int8. A crossing packs both its
        axes into this one byte, and
        ``factoriax.engine.tables.CROSSING_AXIS_DIRS`` unpacks them.
    ent_power
        Steps left in the current craft of the machine. The value counts down.
        Shape ``(E,)``, int16. The craft finishes on the step where the value
        reaches 1, and 0 means idle. Only assemblers and furnaces use this
        field. Every other kind holds 0.
    ent_buf_type
        Item in the general-purpose buffer slot. Shape ``(E,)``, int8. This
        slot holds the output of a miner, the storage of a pallet, and the
        contents of a belt.
    ent_buf_count
        Items in that buffer. Shape ``(E,)``, int16.
        ``factoriax.engine.tables.MACHINE_MAX_STACK`` gives the limit for each
        kind.
    ent_asm_in_type
        Item in each of the two input slots. Shape ``(E, 2)``, int8.
        Assemblers, furnaces, and science labs use these slots. A crossing uses
        the same pair as one buffer for each axis, at the indexes
        ``CROSSING_VERT_SLOT`` and ``CROSSING_HORIZ_SLOT``.
    ent_asm_in_count
        Items in each input slot, in the same order as ``ent_asm_in_type``.
        Shape ``(E, 2)``, int16.
    ent_asm_out_type
        Item in the finished-output slot. Shape ``(E,)``, int8. A finished
        craft stays here, and nothing writes over it. The machine therefore
        stops until something withdraws the item.
    ent_asm_out_count
        Items in the output slot. Shape ``(E,)``, int16.
    ent_health
        Hit points of each entity. Shape ``(E,)``, int16. Placement sets
        ``MACHINE_MAX_HEALTH``. A repair clamps back to that value, and the
        engine refuses a pickup below it.
    player_positions
        ``(x, y)`` tile of each player. Shape ``(P, 2)``, int16.
    player_directions
        Facing of each player, a ``Direction`` value. Shape ``(P,)``, int8.
    player_inventory
        Items that each player carries, by player and item id. Shape
        ``(P, NUM_ITEM_TYPES)``, int16. Column 0 is ``ItemType.EMPTY`` and
        stays zero. ``factoriax.engine.tables.PLAYER_MAX_STACK`` gives the
        limit for each item.
    selected_player
        Index of the player that an action applies to. Scalar int32. The
        single-agent step sets this field.
    timestep
        Steps since the last reset. Scalar int32. The episode ends when this
        value reaches ``EnvParams.max_timesteps``.
    items_mined
        Total mined for each item id since the last reset. Shape
        ``(NUM_ITEM_TYPES,)``, int32. Nothing clears this field during an
        episode. Reward functions read the difference between two states, not
        the total.
    science_consumed_step
        Science packs that the last step consumed, by position in
        :data:`~factoriax.engine.constants.SCIENCE_PACK_TYPES`. Shape
        ``(NUM_SCIENCE_PACK_TYPES,)``, int32. This field is a per-step delta,
        not a total.
    achievements_unlocked
        One latched bit for each achievement. Shape ``(MAX_ACHIEVEMENTS,)``,
        bool. The engine folds a bit in with OR when its condition first
        holds, and the bit stays set for the rest of the episode. A scenario
        with fewer achievements leaves the bits at the end False.

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

    Every engine function takes this object next to the state, so a scenario
    can change the balance and needs no change to engine code. The defaults
    describe the generated sandbox map. A scenario that brings its own level
    overrides the terrain fields, and nothing reads them.

    The six terrain probabilities apply only when the engine generates a level,
    and not when it loads one. The engine tests each probability against its
    own smooth-noise field. They are therefore per-tile shares and not bands,
    and their total does not have to be 1. If two draws claim the same tile,
    the later draw wins. The order is silicon, tin, coal, copper, iron, water,
    so water replaces any ore on the same tile. A tile that no draw claims
    stays plain ground.

    Attributes
    ----------
    max_timesteps
        Steps before the episode reports done. The engine compares this value
        against ``EnvState.timestep``.
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
        Ore units in every mineable tile at the start. The engine writes the
        value into ``EnvState.block_resources``, and a tile that no player can
        mine gets 0. The value must fit in int16. With more than
        :data:`~factoriax.engine.constants.BLOCK_MAX_RESOURCES` the tile still
        works, but it normalizes to more than 1.0 in the observation.
    miner_mining_rate
        Ore that a placed miner extracts in one step. Its output slot holds
        ``factoriax.engine.machines.MINER_OUTPUT_CAP``, so a miner reaches that
        limit in one step. It then stays idle until something empties the slot.
    player_mining_yield
        Ore that one ``MINE`` action gives to the player. The ore left in the
        tile and the stack limit of the player both limit this amount.
    recipe_table
        Recipes that this episode runs, as a
        :class:`~factoriax.engine.recipes.RecipeTable`. The default is the
        table of the engine itself. Engine code takes the size of its recipe
        loops from this table, and not from the module-level default. A
        scenario can therefore supply a different number of recipes.
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

    #: Size of the action space. This is a ``ClassVar``, so it is not a PyTree
    #: leaf and no episode can tune it. The
    #: :class:`~factoriax.engine.constants.Action` enum fixes the action space.
    NUM_ACTIONS: ClassVar[int] = len(Action)
