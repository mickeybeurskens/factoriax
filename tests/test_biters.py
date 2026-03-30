"""Tests for the biter system: scent field, spawning, movement, and attack."""

import jax
import jax.numpy as jnp

from factoriax.biters import (
    attack_machines,
    move_biters,
    spawn_biters,
    update_scent_field,
)
from factoriax.constants import (
    DEFAULT_MACHINE_MAX_HEALTH,
    DEFAULT_MAX_BITERS,
    BlockType,
    MachineType,
)
from factoriax.state import EnvParams


class TestScentField:
    """Scent field should emit from machines and diffuse."""

    def test_machine_emits_scent(self, state_factory) -> None:
        """A healthy machine should set scent at its tile."""
        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
            machine_types=jnp.full((4, 4), int(MachineType.NONE), dtype=jnp.int32).at[2, 2].set(int(MachineType.MINER)),
            machine_health=jnp.zeros((4, 4), dtype=jnp.int32).at[2, 2].set(DEFAULT_MACHINE_MAX_HEALTH),
        )
        params = EnvParams(map_width=4, map_height=4, num_players=1)
        new = update_scent_field(state, params)
        assert float(new.scent_field[2, 2]) == params.scent_emission

    def test_scent_diffuses_to_neighbors(self, state_factory) -> None:
        """After multiple updates, scent should spread to nearby tiles."""
        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
            machine_types=jnp.full((4, 4), int(MachineType.NONE), dtype=jnp.int32).at[2, 2].set(int(MachineType.MINER)),
            machine_health=jnp.zeros((4, 4), dtype=jnp.int32).at[2, 2].set(DEFAULT_MACHINE_MAX_HEALTH),
        )
        params = EnvParams(map_width=4, map_height=4, num_players=1)
        for _ in range(5):
            state = update_scent_field(state, params)
        # Neighbors should have some scent.
        assert float(state.scent_field[2, 1]) > 0
        assert float(state.scent_field[1, 2]) > 0

    def test_disabled_machine_no_scent(self, state_factory) -> None:
        """A machine at 0 HP should not emit scent."""
        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
            machine_types=jnp.full((4, 4), int(MachineType.NONE), dtype=jnp.int32).at[2, 2].set(int(MachineType.MINER)),
            machine_health=jnp.zeros((4, 4), dtype=jnp.int32),
        )
        params = EnvParams(map_width=4, map_height=4, num_players=1)
        new = update_scent_field(state, params)
        assert float(new.scent_field[2, 2]) == 0.0

    def test_water_blocks_diffusion(self, state_factory) -> None:
        """Scent should not diffuse through water tiles."""
        wm = jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32)
        wm = wm.at[2, 1].set(int(BlockType.WATER))  # Water between machine and target.
        state = state_factory(
            world_map=wm,
            machine_types=jnp.full((4, 4), int(MachineType.NONE), dtype=jnp.int32).at[2, 2].set(int(MachineType.MINER)),
            machine_health=jnp.zeros((4, 4), dtype=jnp.int32).at[2, 2].set(DEFAULT_MACHINE_MAX_HEALTH),
        )
        params = EnvParams(map_width=4, map_height=4, num_players=1)
        for _ in range(5):
            state = update_scent_field(state, params)
        # Water tile should have zero scent.
        assert float(state.scent_field[2, 1]) == 0.0


class TestSpawning:
    """Biters should spawn from nests into inactive pool slots."""

    def test_spawn_from_nest(self, state_factory) -> None:
        """With spawn_rate=1.0, a nest should always produce a biter."""
        wm = jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32)
        wm = wm.at[2, 2].set(int(BlockType.NEST))
        state = state_factory(world_map=wm)
        params = EnvParams(
            map_width=4, map_height=4, num_players=1,
            biter_spawn_rate=1.0,
        )
        rng = jax.random.PRNGKey(42)
        new = spawn_biters(state, params, rng)
        assert int(jnp.sum(new.biter_health > 0)) >= 1

    def test_no_spawn_rate_zero(self, state_factory) -> None:
        """With spawn_rate=0.0, no biters should spawn."""
        wm = jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32)
        wm = wm.at[2, 2].set(int(BlockType.NEST))
        state = state_factory(world_map=wm)
        params = EnvParams(
            map_width=4, map_height=4, num_players=1,
            biter_spawn_rate=0.0,
        )
        rng = jax.random.PRNGKey(0)
        new = spawn_biters(state, params, rng)
        assert int(jnp.sum(new.biter_health > 0)) == 0

    def test_no_spawn_without_nest(self, state_factory) -> None:
        """Without any nest tiles, no biters should spawn."""
        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
        )
        params = EnvParams(
            map_width=4, map_height=4, num_players=1,
            biter_spawn_rate=1.0,
        )
        rng = jax.random.PRNGKey(0)
        new = spawn_biters(state, params, rng)
        assert int(jnp.sum(new.biter_health > 0)) == 0


class TestMovement:
    """Biters should move toward scent."""

    def test_biter_moves_toward_scent(self, state_factory) -> None:
        """A biter should step toward higher scent values."""
        scent = jnp.zeros((4, 4), dtype=jnp.float32)
        scent = scent.at[2, 3].set(1.0)  # Scent to the right of biter.
        positions = jnp.zeros((DEFAULT_MAX_BITERS, 2), dtype=jnp.int32)
        positions = positions.at[0].set(jnp.array([2, 2]))  # Biter at (2, 2).
        health = jnp.zeros(DEFAULT_MAX_BITERS, dtype=jnp.int32)
        health = health.at[0].set(10)
        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
            biter_positions=positions,
            biter_health=health,
            scent_field=scent,
        )
        params = EnvParams(map_width=4, map_height=4, num_players=1)
        rng = jax.random.PRNGKey(0)
        new = move_biters(state, params, rng)
        # Biter should have moved toward scent (right: x+1).
        assert int(new.biter_positions[0, 0]) == 3

    def test_inactive_biter_stays(self, state_factory) -> None:
        """A biter with health=0 should not move."""
        positions = jnp.zeros((DEFAULT_MAX_BITERS, 2), dtype=jnp.int32)
        positions = positions.at[0].set(jnp.array([1, 1]))
        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
            biter_positions=positions,
            scent_field=jnp.ones((4, 4), dtype=jnp.float32),
        )
        params = EnvParams(map_width=4, map_height=4, num_players=1)
        rng = jax.random.PRNGKey(0)
        new = move_biters(state, params, rng)
        assert int(new.biter_positions[0, 0]) == 1
        assert int(new.biter_positions[0, 1]) == 1


class TestAttack:
    """Biters should damage adjacent machines."""

    def test_biter_damages_adjacent_machine(self, state_factory) -> None:
        """A biter next to a machine should reduce its health."""
        mt = jnp.full((4, 4), int(MachineType.NONE), dtype=jnp.int32)
        mt = mt.at[2, 2].set(int(MachineType.MINER))
        mh = jnp.zeros((4, 4), dtype=jnp.int32)
        mh = mh.at[2, 2].set(DEFAULT_MACHINE_MAX_HEALTH)
        positions = jnp.zeros((DEFAULT_MAX_BITERS, 2), dtype=jnp.int32)
        positions = positions.at[0].set(jnp.array([1, 2]))  # Adjacent left.
        health = jnp.zeros(DEFAULT_MAX_BITERS, dtype=jnp.int32)
        health = health.at[0].set(10)
        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
            machine_types=mt,
            machine_health=mh,
            biter_positions=positions,
            biter_health=health,
        )
        params = EnvParams(
            map_width=4, map_height=4, num_players=1,
            biter_attack_damage=15,
        )
        new = attack_machines(state, params)
        assert int(new.machine_health[2, 2]) == DEFAULT_MACHINE_MAX_HEALTH - 15

    def test_inactive_biter_no_attack(self, state_factory) -> None:
        """A biter with health=0 should not deal damage."""
        mt = jnp.full((4, 4), int(MachineType.NONE), dtype=jnp.int32)
        mt = mt.at[2, 2].set(int(MachineType.MINER))
        mh = jnp.zeros((4, 4), dtype=jnp.int32)
        mh = mh.at[2, 2].set(DEFAULT_MACHINE_MAX_HEALTH)
        positions = jnp.zeros((DEFAULT_MAX_BITERS, 2), dtype=jnp.int32)
        positions = positions.at[0].set(jnp.array([1, 2]))
        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
            machine_types=mt,
            machine_health=mh,
            biter_positions=positions,
        )
        params = EnvParams(
            map_width=4, map_height=4, num_players=1,
            biter_attack_damage=15,
        )
        new = attack_machines(state, params)
        assert int(new.machine_health[2, 2]) == DEFAULT_MACHINE_MAX_HEALTH

    def test_health_clamps_to_zero(self, state_factory) -> None:
        """Machine health should not go below zero."""
        mt = jnp.full((4, 4), int(MachineType.NONE), dtype=jnp.int32)
        mt = mt.at[2, 2].set(int(MachineType.MINER))
        mh = jnp.zeros((4, 4), dtype=jnp.int32)
        mh = mh.at[2, 2].set(3)  # Low health.
        positions = jnp.zeros((DEFAULT_MAX_BITERS, 2), dtype=jnp.int32)
        positions = positions.at[0].set(jnp.array([1, 2]))
        health = jnp.zeros(DEFAULT_MAX_BITERS, dtype=jnp.int32)
        health = health.at[0].set(10)
        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
            machine_types=mt,
            machine_health=mh,
            biter_positions=positions,
            biter_health=health,
        )
        params = EnvParams(
            map_width=4, map_height=4, num_players=1,
            biter_attack_damage=15,
        )
        new = attack_machines(state, params)
        assert int(new.machine_health[2, 2]) == 0
