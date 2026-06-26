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
    
    Examples
    --------

    Parameters
    ----------

    Returns
    -------

    
    >>> import jax
        >>> import factoriax
        >>> env, params = factoriax.make("EasyRocket-v1")
        >>> _, state = env.reset_env(jax.random.PRNGKey(0), params)
        >>> ni = factoriax.NUM_ITEM_TYPES
        >>> state.player_inventory.shape == (params.num_players, ni)
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
    """Environment parameters."""

    max_timesteps: int = 1000
    num_players: int = 2
    max_machines: int = 0
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
