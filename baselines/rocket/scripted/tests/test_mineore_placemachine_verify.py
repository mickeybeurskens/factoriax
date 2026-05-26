"""Tests for :meth:`MineOre.verify` and :meth:`PlaceMachine.verify`.

Both are sanity checks layered on top of step gates that already
match the same condition. The tests confirm:

- :class:`MineOre` carries ``verify_failure_action="ignore"`` so a
  divergence between step's "I'm done" and the inventory snapshot is
  logged but doesn't halt a long run.
- :class:`PlaceMachine` defaults to ``halt`` and tightens step's
  loose ``total_machines`` count gate to a per-machine-type count.
"""

from __future__ import annotations

import numpy as np
import pytest

from baselines.rocket.scripted.goals import MineOre, PlaceMachine
from baselines.rocket.scripted.world_model import PlayerScalars, WorldView
from factoriax.engine.constants import Direction, ItemType, Machine


def _view_with_inventory(item: int, count: int) -> WorldView:
    """3x3 view with the player holding ``count`` of ``item``."""
    zero = np.zeros((3, 3), dtype=np.int32)
    inv = np.zeros(64, dtype=np.int32)
    inv[item] = count
    return WorldView(
        block_type=zero,
        machine_type=zero,
        block_resources=zero,
        slot0_type=zero,
        slot0_count=zero,
        slot1_type=zero,
        slot1_count=zero,
        slot2_type=zero,
        slot2_count=zero,
        machine_direction=zero,
        buffer_type=zero,
        player=PlayerScalars(
            pos_x=0,
            pos_y=0,
            direction=int(Direction.DOWN),
            timestep=0,
            facing_machine_type=0,
            facing_buffer_type=0,
            facing_buffer_count=0,
            inventory=inv,
        ),
        walkable=np.zeros((3, 3), dtype=bool),
    )


def _view_with_machine(machine_type: int, x: int, y: int) -> WorldView:
    """3x3 view with one machine of ``machine_type`` at tile ``(x, y)``."""
    zero = np.zeros((3, 3), dtype=np.int32)
    machines = zero.copy()
    machines[y, x] = machine_type
    return WorldView(
        block_type=zero,
        machine_type=machines,
        block_resources=zero,
        slot0_type=zero,
        slot0_count=zero,
        slot1_type=zero,
        slot1_count=zero,
        slot2_type=zero,
        slot2_count=zero,
        machine_direction=zero,
        buffer_type=zero,
        player=PlayerScalars(
            pos_x=0,
            pos_y=0,
            direction=int(Direction.DOWN),
            timestep=0,
            facing_machine_type=0,
            facing_buffer_type=0,
            facing_buffer_count=0,
            inventory=np.zeros(64, dtype=np.int32),
        ),
        walkable=np.zeros((3, 3), dtype=bool),
    )


class TestMineOreVerify:
    def test_action_is_ignore(self) -> None:
        """Failures are logged but don't halt — action is "ignore"."""
        assert MineOre(ItemType.IRON_ORE, 5).verify_failure_action == "ignore"

    def test_passes_when_count_met(self) -> None:
        view = _view_with_inventory(int(ItemType.IRON_ORE), 5)
        assert MineOre(ItemType.IRON_ORE, 5).verify(view) is True

    def test_fails_when_count_short(self) -> None:
        view = _view_with_inventory(int(ItemType.IRON_ORE), 3)
        assert MineOre(ItemType.IRON_ORE, 5).verify(view) is False


class TestPlaceMachineVerify:
    def test_action_is_halt(self) -> None:
        # Use a dummy predicate; verify_failure_action is class-level.
        predicate = lambda _v: None  # noqa: E731
        assert PlaceMachine(Machine.MINER, predicate).verify_failure_action == "halt"

    def test_returns_false_before_step_runs(self) -> None:
        """Without a baseline (step never ran), verify must return False."""
        predicate = lambda _v: None  # noqa: E731
        goal = PlaceMachine(Machine.MINER, predicate)
        view = _view_with_machine(int(Machine.MINER), 0, 0)
        assert goal.verify(view) is False

    def test_passes_when_typed_count_grew(self) -> None:
        """Set baseline manually, then verify against a view with one more."""
        predicate = lambda _v: None  # noqa: E731
        goal = PlaceMachine(Machine.MINER, predicate)
        goal._start_typed_count = 0
        view = _view_with_machine(int(Machine.MINER), 1, 1)
        assert goal.verify(view) is True

    def test_fails_when_typed_count_unchanged(self) -> None:
        predicate = lambda _v: None  # noqa: E731
        goal = PlaceMachine(Machine.MINER, predicate)
        goal._start_typed_count = 1
        # View has only one MINER; 1 > 1 is False.
        view = _view_with_machine(int(Machine.MINER), 1, 1)
        assert goal.verify(view) is False

    def test_fails_when_other_type_grew(self) -> None:
        """Step's loose ``total_machines`` gate fires on any new
        machine, but verify is per-type, so a placement that landed
        as PALLET when MINER was requested still trips verify."""
        predicate = lambda _v: None  # noqa: E731
        goal = PlaceMachine(Machine.MINER, predicate)
        goal._start_typed_count = 0
        # Wrong type present.
        view = _view_with_machine(int(Machine.PALLET), 1, 1)
        assert goal.verify(view) is False


@pytest.mark.parametrize(
    "machine_type",
    [Machine.MINER, Machine.PALLET, Machine.ARM, Machine.FURNACE],
)
def test_place_machine_verify_works_for_all_types(
    machine_type: Machine,
) -> None:
    """Smoke test: verify across machine types."""
    predicate = lambda _v: None  # noqa: E731
    goal = PlaceMachine(machine_type, predicate)
    goal._start_typed_count = 0
    view = _view_with_machine(int(machine_type), 1, 1)
    assert goal.verify(view) is True
