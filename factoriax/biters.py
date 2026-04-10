"""Biter system: scent field, spawning, movement, and machine attack.

Biters are environmental threats that spawn from nest tiles, navigate
toward machines via a diffusing scent field, and damage adjacent machines.
Players cannot fight biters directly — they can only repair damaged
machines.

All functions are JAX-native and JIT-compatible. Biters are stored in a
fixed-capacity pool. Inactive slots have ``health == 0``. The pool size
is derived from the shape of ``state.biter_health``, not from params,
so array shapes remain static under JIT.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from factoriax.constants import BlockType, MachineType
from factoriax.state import EnvParams, EnvState

# Cardinal direction offsets: right, left, down, up.
_DX = jnp.array([1, -1, 0, 0], dtype=jnp.int32)
_DY = jnp.array([0, 0, 1, -1], dtype=jnp.int32)


def update_scent_field(state: EnvState, params: EnvParams) -> EnvState:
    """Update the scent field: emit from machines, then diffuse.

    Active machines (health > 0) emit scent at their tile. The field
    then diffuses via a 3x3 averaging stencil with decay. Water and
    out-of-bounds tiles block diffusion.

    Args:
        state: Current environment state.
        params: Environment parameters.

    Returns:
        State with updated scent_field.
    """
    # Emit: machines with health > 0 set scent to emission value.
    has_machine = (state.machine_types != MachineType.NONE) & (
        state.machine_health > 0
    )
    scent = jnp.where(has_machine, params.scent_emission, state.scent_field)

    # Diffuse: 3x3 average of neighbors, scaled by decay.
    padded = jnp.pad(scent, 1, mode="constant", constant_values=0.0)
    neighbors_sum = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    diffused = scent * params.scent_decay + neighbors_sum * (
        (1.0 - params.scent_decay) / 4.0
    )

    # Block diffusion on non-walkable tiles.
    walkable = ~jnp.isin(
        state.map, jnp.array([BlockType.WATER, BlockType.OUT_OF_BOUNDS])
    )
    diffused = jnp.where(walkable, diffused, 0.0)

    # Re-emit on machine tiles.
    diffused = jnp.where(has_machine, params.scent_emission, diffused)

    return state.replace(scent_field=diffused)


def spawn_biters(
    state: EnvState, params: EnvParams, rng: jax.Array
) -> EnvState:
    """Spawn biters from nest tiles into inactive pool slots.

    Each nest has a ``biter_spawn_rate`` probability of producing a biter
    per action tick. Biters spawn on a walkable tile adjacent to the nest.
    Spawning stops when the pool is full.

    Args:
        state: Current environment state.
        params: Environment parameters.
        rng: JAX random key.

    Returns:
        State with newly spawned biters.
    """
    h, w = state.map.shape
    max_b = state.biter_health.shape[0]  # static from array shape

    # Find nest positions (fixed-size output for JIT).
    is_nest = state.map == BlockType.NEST
    nest_ys, nest_xs = jnp.where(is_nest, size=h * w, fill_value=-1)
    nest_valid = (nest_ys >= 0) & (nest_xs >= 0)

    # Roll spawn probabilities.
    rng, spawn_key = jax.random.split(rng)
    spawn_rolls = jax.random.uniform(spawn_key, shape=(h * w,))
    should_spawn = nest_valid & (spawn_rolls < params.biter_spawn_rate)

    # Pick a random adjacent tile for each nest.
    rng, dir_key = jax.random.split(rng)
    dir_rolls = jax.random.randint(dir_key, shape=(h * w,), minval=0, maxval=4)
    spawn_x = nest_xs + _DX[dir_rolls]
    spawn_y = nest_ys + _DY[dir_rolls]

    spawn_x = jnp.clip(spawn_x, 0, w - 1)
    spawn_y = jnp.clip(spawn_y, 0, h - 1)

    # Check spawn tile is walkable.
    spawn_block = state.map[spawn_y, spawn_x]
    spawn_walkable = ~jnp.isin(
        spawn_block,
        jnp.array([BlockType.WATER, BlockType.OUT_OF_BOUNDS, BlockType.NEST]),
    )
    should_spawn = should_spawn & spawn_walkable

    # Assign spawns to inactive biter slots via scan.
    def assign_one(
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        spawn_idx: jax.Array,
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], None]:
        """Try to assign one spawn to the next free biter slot."""
        positions, health, next_slot = carry
        do_spawn = should_spawn[spawn_idx] & (next_slot < max_b)

        inactive = health == 0
        slot_indices = jnp.arange(max_b)
        available = inactive & (slot_indices >= next_slot)
        has_slot = jnp.any(available)
        slot = jnp.where(has_slot, jnp.argmax(available), max_b)

        can_assign = do_spawn & has_slot & (slot < max_b)

        new_pos = positions.at[slot, 0].set(
            jnp.where(can_assign, spawn_x[spawn_idx], positions[slot, 0])
        ).at[slot, 1].set(
            jnp.where(can_assign, spawn_y[spawn_idx], positions[slot, 1])
        )
        new_hp = health.at[slot].set(
            jnp.where(can_assign, params.biter_health_default, health[slot])
        )
        new_next = jnp.where(can_assign, slot + 1, next_slot)

        return (new_pos, new_hp, new_next), None

    (new_pos, new_health, _), _ = jax.lax.scan(
        assign_one,
        (state.biter_positions, state.biter_health, jnp.int32(0)),
        jnp.arange(h * w),
    )

    return state.replace(biter_positions=new_pos, biter_health=new_health)


def move_biters(
    state: EnvState, params: EnvParams, rng: jax.Array
) -> EnvState:
    """Move active biters toward higher scent concentrations.

    Each active biter reads the scent at its 4 cardinal neighbors and
    steps toward the highest value. If all neighbors have zero scent,
    the biter takes a random step.

    Args:
        state: Current environment state.
        params: Environment parameters.
        rng: JAX random key for random walk fallback.

    Returns:
        State with updated biter positions.
    """
    h, w = state.map.shape
    max_b = state.biter_health.shape[0]
    is_active = state.biter_health > 0
    bx = state.biter_positions[:, 0]
    by = state.biter_positions[:, 1]

    # Compute scent at 4 neighbors for each biter.
    padded_scent = jnp.pad(
        state.scent_field, 1, mode="constant", constant_values=0.0
    )
    # In padded coords, biter at (x, y) maps to (y+1, x+1).
    neighbor_scent = jnp.stack(
        [
            padded_scent[by + 1, bx + 2],  # right
            padded_scent[by + 1, bx],       # left
            padded_scent[by + 2, bx + 1],   # down
            padded_scent[by, bx + 1],        # up
        ],
        axis=-1,
    )

    # Check walkability of each neighbor.
    padded_map = jnp.pad(
        state.map, 1, mode="constant",
        constant_values=int(BlockType.OUT_OF_BOUNDS),
    )
    neighbor_block = jnp.stack(
        [
            padded_map[by + 1, bx + 2],
            padded_map[by + 1, bx],
            padded_map[by + 2, bx + 1],
            padded_map[by, bx + 1],
        ],
        axis=-1,
    )
    neighbor_walkable = ~jnp.isin(
        neighbor_block,
        jnp.array([BlockType.WATER, BlockType.OUT_OF_BOUNDS]),
    )

    masked_scent = jnp.where(neighbor_walkable, neighbor_scent, -1e6)

    has_scent = jnp.any(neighbor_scent > 0, axis=-1)
    best_dir = jnp.argmax(masked_scent, axis=-1)

    rng, walk_key = jax.random.split(rng)
    random_dir = jax.random.randint(
        walk_key, shape=(max_b,), minval=0, maxval=4
    )
    chosen_dir = jnp.where(has_scent, best_dir, random_dir)

    new_x = jnp.clip(bx + _DX[chosen_dir], 0, w - 1)
    new_y = jnp.clip(by + _DY[chosen_dir], 0, h - 1)

    final_x = jnp.where(is_active, new_x, bx)
    final_y = jnp.where(is_active, new_y, by)

    return state.replace(
        biter_positions=jnp.stack([final_x, final_y], axis=-1)
    )


def attack_machines(state: EnvState, params: EnvParams) -> EnvState:
    """Active biters damage adjacent machines.

    Each active biter checks its 4 cardinal neighbors for machines.
    If a machine is found, it takes ``biter_attack_damage`` HP of damage.
    Multiple biters can attack the same machine in one tick.

    Args:
        state: Current environment state.
        params: Environment parameters.

    Returns:
        State with reduced machine health.
    """
    h, w = state.map.shape
    is_active = state.biter_health > 0
    bx = state.biter_positions[:, 0]
    by = state.biter_positions[:, 1]

    damage = jnp.zeros((h, w), dtype=jnp.int32)

    for d in range(4):
        tx = jnp.clip(bx + _DX[d], 0, w - 1)
        ty = jnp.clip(by + _DY[d], 0, h - 1)
        has_machine = state.machine_types[ty, tx] != MachineType.NONE
        should_attack = is_active & has_machine
        damage = damage.at[ty, tx].add(
            jnp.where(should_attack, params.biter_attack_damage, 0)
        )

    new_health = jnp.maximum(state.machine_health - damage, 0)
    return state.replace(machine_health=new_health)


def update_biters(
    state: EnvState, params: EnvParams, rng: jax.Array
) -> EnvState:
    """Run the full biter update: spawn, move, attack.

    Only executes on biter action ticks (every ``biter_tick_interval``
    steps). On other ticks, this is a no-op.

    Args:
        state: Current environment state.
        params: Environment parameters.
        rng: JAX random key.

    Returns:
        Updated state.
    """
    is_biter_tick = (state.timestep % params.biter_tick_interval) == 0

    rng_local = jax.random.fold_in(rng, state.timestep)
    r1, r2 = jax.random.split(rng_local)
    updated = spawn_biters(state, params, r1)
    updated = move_biters(updated, params, r2)
    updated = attack_machines(updated, params)

    return jax.tree.map(
        lambda n, o: jnp.where(is_biter_tick, n, o), updated, state,
    )
