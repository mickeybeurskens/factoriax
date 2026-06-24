"""State dataclasses for the FactoriaX environment."""

from typing import ClassVar

import jax.numpy as jnp
from flax import struct

from factoriax.engine.constants import Action
from factoriax.engine.recipes import DEFAULT_RECIPE_TABLE, RecipeTable


class EnvState(struct.PyTreeNode):  # type: ignore[no-untyped-call]
    """Immutable environment state.

    Machine state uses entity lists: fixed-size arrays indexed by entity
    ID, not tile position. The ``tile_entity`` grid maps tile positions
    to entity indices for neighbor lookups. Inactive entities have
    ``ent_y < 0``.

    Terrain (``map``, ``block_resources``) and spatial lookup
    (``machine_types``, ``tile_entity``) remain on the grid.

    Example:
        >>> import jax
        >>> import factoriax
        >>> env, params = factoriax.make("EasyRocket-v1")
        >>> _, state = env.reset_env(jax.random.PRNGKey(0), params)
        >>> ni = factoriax.NUM_ITEM_TYPES
        >>> state.player_inventory.shape == (params.num_players, ni)
        True

    Attributes:
        map: Block types, shape ``(H, W)``, int8.
        block_resources: Ore remaining per tile, shape ``(H, W)``, int16.
        machine_types: Machine type per tile (walkability), shape ``(H, W)``, int8.
        tile_entity: Entity index at each tile (-1=none), shape ``(H, W)``, int16.
        ent_y: Entity row positions, shape ``(MAX_M,)``, int16. -1=inactive.
        ent_x: Entity column positions, shape ``(MAX_M,)``, int16.
        ent_type: Entity machine type, shape ``(MAX_M,)``, int8.
        ent_direction: Entity facing, shape ``(MAX_M,)``, int8.
        ent_power: Craft countdown for assemblers, shape ``(MAX_M,)``, int16.
        ent_buf_type: Buffer item type, shape ``(MAX_M,)``, int8.
        ent_buf_count: Buffer item count, shape ``(MAX_M,)``, int16.
        ent_asm_in_type: Assembler input types, shape ``(MAX_M, 2)``, int8.
        ent_asm_in_count: Assembler input counts, shape ``(MAX_M, 2)``, int16.
        ent_asm_out_type: Assembler output type, shape ``(MAX_M,)``, int8.
        ent_asm_out_count: Assembler output count, shape ``(MAX_M,)``, int16.
        ent_health: Per-entity machine health, shape ``(MAX_M,)``, int16.
            Initialized to ``MACHINE_MAX_HEALTH[ent_type]`` on placement;
            inactive slots hold ``0``. Only
            :func:`~factoriax.engine.placement.apply_repair` and
            :func:`~factoriax.engine.placement.pickup_machine` touch it, so
            wrappers can layer their own degradation/repair models.
        player_positions: (x, y) per player, shape ``(P, 2)``, int16.
        player_directions: Facing per player, shape ``(P,)``, int8.
        player_inventory: Item counts per player, shape ``(P, N)``, int16.
        selected_player: Active player index, scalar.
        timestep: Current step, scalar.
        items_mined: Lifetime mined per type, shape ``(N,)``, int32.
        science_consumed_step: Science packs consumed this step by
            SCIENCE_LAB entities, shape ``(NUM_SCIENCE_PACK_TYPES,)``, int32.
            Reset to zero each step; :class:`ScienceTallyWrapper` accumulates
            the running total.
        achievements_unlocked: Latched achievement flags, shape
            ``(MAX_ACHIEVEMENTS,)``, bool. Once a bit flips on it stays on
            for the rest of the episode.
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
    """Environment parameters.

    Example:
        >>> import factoriax
        >>> _, params = factoriax.make("EasyRocket-v1")
        >>> larger = params.replace(map_width=64, map_height=64)
        >>> larger.map_width, larger.map_height
        (64, 64)

    Attributes:
        max_timesteps: Maximum steps per episode.
        map_width: Grid width.
        map_height: Grid height.
        num_players: Number of players.
        max_machines: Maximum entity slots for machines; sizes the
            fixed-size entity arrays that all machine operations iterate
            over. Step cost scales with this value, not with the number of
            machines actually placed. ``0`` selects the auto default
            ``max(64, map_width * map_height // 4)``. Reduce for faster
            stepping on small, sparse maps; increase to allow more machines.
        water_probability: Tile water probability during generation.
        iron_probability: Iron ore probability.
        copper_probability: Copper ore probability.
        coal_probability: Coal probability.
        tin_probability: Tin ore probability.
        silicon_probability: Silicon probability.
        base_resources: Starting ore count per tile.
        miner_mining_rate: Ore extracted per tick by a placed Miner.
        player_mining_yield: Ore extracted per successful MINE action by
            the player. Defaults to 1.
        recipe_table: Per-recipe balance numbers (input/output counts,
            ticks) and identity arrays (machine type, output items) packed
            as a :class:`~factoriax.engine.recipes.RecipeTable`. Defaults to
            :data:`~factoriax.engine.recipes.DEFAULT_RECIPE_TABLE`. A PyTree
            leaf of fixed shape, so kernels read values without retracing and
            tuned balance overlays reuse the JIT cache.
    """

    max_timesteps: int = 1000
    map_width: int = 32
    map_height: int = 32
    num_players: int = 2
    max_machines: int = 0  # 0 = auto: max(64, map_area // 4)
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

    NUM_ACTIONS: ClassVar[int] = len(Action)

    def resolved_max_machines(self) -> int:
        """Return max_machines, resolving 0 to the auto default."""
        if self.max_machines > 0:
            return self.max_machines
        return max(64, self.map_width * self.map_height // 4)
