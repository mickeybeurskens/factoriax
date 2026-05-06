"""State dataclasses for the FactoriaX environment."""

from typing import ClassVar

import jax.numpy as jnp
from flax import struct

from factoriax.constants import Action
from factoriax.machine_config import DEFAULT_MACHINE_CONFIG, MachineConfig
from factoriax.recipes import DEFAULT_RECIPE_TABLE, RecipeTable


class EnvState(struct.PyTreeNode):  # type: ignore[no-untyped-call]
    """Immutable environment state.

    Machine state uses entity lists: fixed-size arrays indexed by entity
    ID, not tile position. The ``tile_entity`` grid maps tile positions
    to entity indices for neighbor lookups. Inactive entities have
    ``ent_y < 0``.

    Terrain (``map``, ``block_resources``) and spatial lookup
    (``machine_types``, ``tile_entity``) remain on the grid.

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
            Initialized to ``params.machine_config.max_health[ent_type]``
            on placement. Inactive slots hold ``0``. Read by
            :func:`~factoriax.placement.apply_repair` and
            :func:`~factoriax.placement.pickup_machine`; no other engine
            kernel reads or writes this field, so wrappers can layer
            arbitrary degradation/repair models on top without engine
            changes.
        player_positions: (x, y) per player, shape ``(P, 2)``, int16.
        player_directions: Facing per player, shape ``(P,)``, int8.
        player_inventory: Item counts per player, shape ``(P, N)``, int16.
        selected_player: Active player index, scalar.
        timestep: Current step, scalar.
        items_mined: Lifetime mined per type, shape ``(N,)``, int32.
        science_consumed_step: Per-step signal from SCIENCE_LAB entities,
            shape ``(NUM_SCIENCE_PACK_TYPES,)``, int32. Summed over every
            lab's input slots during each step, reset to zero at the next
            step. Read by :class:`ScienceTallyWrapper` to accumulate
            total research consumption without touching the engine.
        achievements_unlocked: Latched achievement flags, shape
            ``(MAX_ACHIEVEMENTS,)``, bool. Once a bit flips on it stays
            on for the rest of the episode. The engine evaluates the
            condition function bound at env-construction time inside
            :func:`~factoriax.game_logic.factoriax_step` and folds the
            result in with ``|``. Wrappers, observations, rewards, and
            benchmarks read this field directly — no wrapper needed.
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

    Attributes:
        max_timesteps: Maximum steps per episode.
        map_width: Grid width.
        map_height: Grid height.
        num_players: Number of players.
        max_machines: Maximum entity slots for machines. Controls the
            fixed-size entity arrays that all machine operations iterate
            over. Cost scales linearly with this value regardless of how
            many machines are actually placed. Default uses
            ``max(64, map_width * map_height // 4)`` which gives a 4x
            speedup over grid-based iteration while supporting up to 25%
            machine density. Reduce for faster stepping on small maps
            with few machines; increase if the agent needs to place more.
        water_probability: Tile water probability during generation.
        iron_probability: Iron ore probability.
        copper_probability: Copper ore probability.
        coal_probability: Coal probability.
        tin_probability: Tin ore probability.
        silicon_probability: Silicon probability.
        base_resources: Starting ore count per tile.
        miner_mining_rate: Ore extracted per tick.
        max_assembler_stack_size: Max items per assembler slot.
        recipe_table: Per-recipe balance numbers (input/output counts,
            ticks) and identity arrays (machine type, output items)
            packed as a :class:`~factoriax.recipes.RecipeTable`. Defaults
            to :data:`~factoriax.recipes.DEFAULT_RECIPE_TABLE`. Stored as
            a PyTree leaf so JIT'd kernels in :mod:`factoriax.machines`
            and :mod:`factoriax.crafting` can read recipe values from
            ``params.recipe_table.*`` without re-baking the XLA graph
            when the user constructs an :class:`EnvParams` with a tuned
            balance overlay (added in Step 5+). Shape is fixed by
            :data:`~factoriax.recipes.NUM_RECIPES` and
            :data:`~factoriax.recipes.MAX_RECIPE_INPUTS` so JIT cache
            reuse is preserved across overlays.
        machine_config: Per-machine-type tunable knobs (currently just
            ``max_stack``) packed as a
            :class:`~factoriax.machine_config.MachineConfig`. Defaults
            to :data:`~factoriax.machine_config.DEFAULT_MACHINE_CONFIG`.
            Engine kernels in :mod:`factoriax.machines` read from
            ``params.machine_config.max_stack`` when computing buffer
            caps; constructing an :class:`EnvParams` with overrides via
            ``DEFAULT_MACHINE_CONFIG.with_overrides({...})`` retunes
            those caps without rebuilding the JIT cache (the array
            shape is fixed by ``len(MachineType)``).
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
    max_assembler_stack_size: int = 1000
    recipe_table: RecipeTable = DEFAULT_RECIPE_TABLE
    machine_config: MachineConfig = DEFAULT_MACHINE_CONFIG

    NUM_ACTIONS: ClassVar[int] = len(Action)

    def resolved_max_machines(self) -> int:
        """Return max_machines, resolving 0 to the auto default.

        Returns:
            Concrete max_machines value.
        """
        if self.max_machines > 0:
            return self.max_machines
        return max(64, self.map_width * self.map_height // 4)
