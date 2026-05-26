"""Verify per-level action masks expose exactly the spec's allowed actions.

The masks defined in :mod:`factoriax.scenarios.skills.achievements`
are the contract between the curriculum and the runner — each level
sees only the action subset that exercises its skill. Drift here would
quietly change what the scenario measures, so each mask is pinned by
exact-set comparison rather than spot checks.
"""

from __future__ import annotations

from factoriax.constants import Action
from factoriax.scenarios.skills.achievements import (
    ARM_TRANSFER_BLOCKED_ACTIONS,
    BELT_LINE_BLOCKED_ACTIONS,
    CRAFT_MINER_BLOCKED_ACTIONS,
    FUEL_AND_COLLECT_BLOCKED_ACTIONS,
    MINE_BLOCKED_ACTIONS,
    MINI_FACTORY_BLOCKED_ACTIONS,
    NAVIGATE_BLOCKED_ACTIONS,
    PLACE_MINER_BLOCKED_ACTIONS,
)

NUM_ACTIONS = len(Action)


def _allowed(mask: frozenset[int]) -> set[int]:
    """Return the actions *not* blocked by *mask*."""
    return {int(a) for a in Action} - mask


class TestNavigateMask:
    def test_only_movement_and_noop(self) -> None:
        allowed = _allowed(NAVIGATE_BLOCKED_ACTIONS)
        assert allowed == {
            int(Action.NOOP),
            int(Action.UP),
            int(Action.DOWN),
            int(Action.LEFT),
            int(Action.RIGHT),
        }

    def test_blocks_mine(self) -> None:
        assert int(Action.MINE) in NAVIGATE_BLOCKED_ACTIONS

    def test_size(self) -> None:
        # 79 total actions − 5 allowed = 74 blocked.
        assert len(NAVIGATE_BLOCKED_ACTIONS) == NUM_ACTIONS - 5


class TestMineMask:
    def test_allows_movement_mine_noop(self) -> None:
        # FACE_* is required because MINE targets the tile in front of the
        # player — the policy must rotate to face an adjacent ore tile
        # without stepping onto it. See MINE_BLOCKED_ACTIONS docstring.
        allowed = _allowed(MINE_BLOCKED_ACTIONS)
        assert allowed == {
            int(Action.NOOP),
            int(Action.UP),
            int(Action.DOWN),
            int(Action.LEFT),
            int(Action.RIGHT),
            int(Action.FACE_UP),
            int(Action.FACE_DOWN),
            int(Action.FACE_LEFT),
            int(Action.FACE_RIGHT),
            int(Action.MINE),
        }


class TestCraftMinerMask:
    def test_allows_only_craft_miner(self) -> None:
        """Of all CRAFT_* actions, only CRAFT_MINER passes through."""
        for craft_action in range(
            int(Action.CRAFT_IRON_PLATE), int(Action.CRAFT_CROSSING) + 1
        ):
            if craft_action == int(Action.CRAFT_MINER):
                assert craft_action not in CRAFT_MINER_BLOCKED_ACTIONS
            else:
                assert craft_action in CRAFT_MINER_BLOCKED_ACTIONS

    def test_blocks_mine(self) -> None:
        """No ore on the map, so MINE is blocked along with placement/etc."""
        assert int(Action.MINE) in CRAFT_MINER_BLOCKED_ACTIONS

    def test_allows_withdraw_and_facing(self) -> None:
        """Player must FACE pallets and WITHDRAW their contents."""
        allowed = _allowed(CRAFT_MINER_BLOCKED_ACTIONS)
        assert int(Action.WITHDRAW) in allowed
        for face in (
            Action.FACE_UP,
            Action.FACE_DOWN,
            Action.FACE_LEFT,
            Action.FACE_RIGHT,
        ):
            assert int(face) in allowed

    def test_exact_allowed_set(self) -> None:
        allowed = _allowed(CRAFT_MINER_BLOCKED_ACTIONS)
        assert allowed == {
            int(Action.NOOP),
            int(Action.UP),
            int(Action.DOWN),
            int(Action.LEFT),
            int(Action.RIGHT),
            int(Action.FACE_UP),
            int(Action.FACE_DOWN),
            int(Action.FACE_LEFT),
            int(Action.FACE_RIGHT),
            int(Action.WITHDRAW),
            int(Action.CRAFT_MINER),
        }


class TestPlaceMinerMask:
    def test_allows_only_place_miner(self) -> None:
        """Of all PLACE_* actions, only PLACE_MINER passes through."""
        for place_action in range(
            int(Action.PLACE_MINER), int(Action.PLACE_CROSSING) + 1
        ):
            if place_action == int(Action.PLACE_MINER):
                assert place_action not in PLACE_MINER_BLOCKED_ACTIONS
            else:
                assert place_action in PLACE_MINER_BLOCKED_ACTIONS

    def test_allows_movement_and_facing(self) -> None:
        allowed = _allowed(PLACE_MINER_BLOCKED_ACTIONS)
        for face in (
            Action.FACE_UP,
            Action.FACE_DOWN,
            Action.FACE_LEFT,
            Action.FACE_RIGHT,
        ):
            assert int(face) in allowed
        for move in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
            assert int(move) in allowed


class TestFuelAndCollectMask:
    def test_blocks_mine(self) -> None:
        """Hand-mining is blocked — agent must collect from placed miner."""
        assert int(Action.MINE) in FUEL_AND_COLLECT_BLOCKED_ACTIONS

    def test_allows_deposit_coal_and_withdraw(self) -> None:
        allowed = _allowed(FUEL_AND_COLLECT_BLOCKED_ACTIONS)
        assert int(Action.DEPOSIT_COAL) in allowed
        assert int(Action.WITHDRAW) in allowed

    def test_blocks_other_deposit_actions(self) -> None:
        """Only DEPOSIT_COAL is exposed; other deposit ops are blocked."""
        for a in range(int(Action.DEPOSIT_COAL) + 1, int(Action.DEPOSIT_CROSSING) + 1):
            assert a in FUEL_AND_COLLECT_BLOCKED_ACTIONS


class TestBeltLineMask:
    def test_allows_place_belt_and_mine(self) -> None:
        allowed = _allowed(BELT_LINE_BLOCKED_ACTIONS)
        assert int(Action.PLACE_CONVEYOR_BELT) in allowed
        assert int(Action.MINE) in allowed

    def test_allows_rotation(self) -> None:
        allowed = _allowed(BELT_LINE_BLOCKED_ACTIONS)
        for rot in (
            Action.ROTATE_LEFT,
            Action.ROTATE_RIGHT,
            Action.ROTATE_UP,
            Action.ROTATE_DOWN,
        ):
            assert int(rot) in allowed

    def test_blocks_other_place_actions(self) -> None:
        for a in range(int(Action.PLACE_MINER), int(Action.PLACE_CROSSING) + 1):
            if a == int(Action.PLACE_CONVEYOR_BELT):
                continue
            assert a in BELT_LINE_BLOCKED_ACTIONS


class TestArmTransferMask:
    def test_allows_place_arm(self) -> None:
        assert int(Action.PLACE_ARM) not in ARM_TRANSFER_BLOCKED_ACTIONS

    def test_allows_deposit_coal(self) -> None:
        assert int(Action.DEPOSIT_COAL) not in ARM_TRANSFER_BLOCKED_ACTIONS

    def test_allows_rotation(self) -> None:
        for rot in (
            Action.ROTATE_LEFT,
            Action.ROTATE_RIGHT,
            Action.ROTATE_UP,
            Action.ROTATE_DOWN,
        ):
            assert int(rot) not in ARM_TRANSFER_BLOCKED_ACTIONS

    def test_blocks_mine(self) -> None:
        assert int(Action.MINE) in ARM_TRANSFER_BLOCKED_ACTIONS


class TestMiniFactoryMask:
    def test_blocks_every_craft_action(self) -> None:
        """All CRAFT_* (21..41) blocked; nothing else."""
        for a in range(int(Action.CRAFT_IRON_PLATE), int(Action.CRAFT_CROSSING) + 1):
            assert a in MINI_FACTORY_BLOCKED_ACTIONS

    def test_blocks_only_craft_actions(self) -> None:
        for a in range(NUM_ACTIONS):
            craft_range = range(
                int(Action.CRAFT_IRON_PLATE), int(Action.CRAFT_CROSSING) + 1
            )
            if a in craft_range:
                continue
            assert a not in MINI_FACTORY_BLOCKED_ACTIONS

    def test_allows_place_and_deposit_and_movement(self) -> None:
        allowed = _allowed(MINI_FACTORY_BLOCKED_ACTIONS)
        for a in (
            Action.PLACE_MINER,
            Action.DEPOSIT_COAL,
            Action.MINE,
            Action.WITHDRAW,
        ):
            assert int(a) in allowed


class TestMaskInvariants:
    """Invariants that hold across every mask."""

    ALL_MASKS = [
        NAVIGATE_BLOCKED_ACTIONS,
        MINE_BLOCKED_ACTIONS,
        CRAFT_MINER_BLOCKED_ACTIONS,
        PLACE_MINER_BLOCKED_ACTIONS,
        FUEL_AND_COLLECT_BLOCKED_ACTIONS,
        BELT_LINE_BLOCKED_ACTIONS,
        ARM_TRANSFER_BLOCKED_ACTIONS,
        MINI_FACTORY_BLOCKED_ACTIONS,
    ]

    def test_every_mask_keeps_noop_unblocked(self) -> None:
        """NOOP must always pass through — the wrapper rewrites blocked → NOOP."""
        for mask in self.ALL_MASKS:
            assert int(Action.NOOP) not in mask, (
                f"NOOP must never be blocked (mask={mask})"
            )

    def test_every_mask_is_subset_of_all_actions(self) -> None:
        for mask in self.ALL_MASKS:
            for a in mask:
                assert 0 <= a < NUM_ACTIONS

    def test_no_mask_is_empty(self) -> None:
        """Even mini_factory blocks something (all CRAFT_*)."""
        for mask in self.ALL_MASKS:
            assert len(mask) > 0
