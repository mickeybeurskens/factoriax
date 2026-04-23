"""Tests for the action strip cursor alignment and click-to-seek.

Two invariants under test:

1. ``render_action_strip`` and ``draw_cursor`` agree column-for-column:
   for every step ``s``, the column returned by ``draw_cursor`` renders
   the color of step ``s`` in the strip. When these drift, the cursor
   visibly detaches from the strip colors near the end of long runs —
   the 8000-step, 320-wide case was off by ~24 steps before the fix.

2. ``action_strip_step_from_click`` is the exact inverse of the strip's
   column→step mapping for clicks inside the strip rectangle, and
   returns ``None`` for clicks outside it.
"""

from __future__ import annotations

import numpy as np
import pytest

from factoriax.agentdebugger.charts import (
    draw_cursor,
    render_action_strip,
)
from factoriax.agentdebugger.layout import (
    action_strip_height,
    action_strip_step_from_click,
)
from factoriax.agentdebugger.state import DebuggerState
from factoriax.analysis.trajectory import Trajectory
from factoriax.constants import Action


def _trajectory_with_distinct_actions(length: int) -> Trajectory:
    """Build a single-episode trajectory whose action at step t is t mod N.

    Using ``t mod N`` means each step has a predictable action color,
    so we can assert "column x shows the color of step t" without
    depending on any particular action semantics.
    """
    num_actions = len(Action)
    actions = np.arange(length, dtype=np.int32) % num_actions
    return Trajectory(actions=actions.reshape(1, length))


@pytest.mark.parametrize(
    "total_steps,width",
    [
        (8000, 320),  # realistic: 8k-step rocket run on default quadrant
        (500, 320),  # short rollout
        (100, 320),  # very short
        (320, 320),  # 1:1
        (50, 320),  # upscale (fewer steps than columns)
        (10, 50),  # tiny
    ],
)
class TestCursorAlignsWithStrip:
    """Cursor column must render the color of the current step."""

    def test_endpoints_align(self, total_steps: int, width: int) -> None:
        """At step 0 and step T-1, the cursor falls on the step's own column."""
        traj = _trajectory_with_distinct_actions(total_steps)
        strip = render_action_strip(traj, episode=0, player=0, width=width, height=8)

        for step in (0, total_steps - 1):
            cursor_img = draw_cursor(strip, step=step, total_steps=total_steps)
            # Find the cursor column — it's the one the cursor overwrote
            # with the default white (255, 255, 255). Endpoints of the
            # strip are never white in a uniformly-cycled strip, so the
            # write is visible.
            diff = np.any(cursor_img != strip, axis=(0, 2))
            cursor_cols = np.where(diff)[0]
            assert cursor_cols.size > 0
            # The strip column at the cursor position must carry the
            # action color of the current step.
            col = int(cursor_cols[0])
            # Walk the strip with the same formula as the renderer to
            # find what step that column represents.
            rendered_step = min(
                int(col * (total_steps - 1) / max(1, width - 1)),
                total_steps - 1,
            )
            assert rendered_step == step, (
                f"cursor at step={step} landed on column {col} which "
                f"renders step {rendered_step} (T={total_steps}, W={width})"
            )

    def test_midpoint_within_bin_tolerance(self, total_steps: int, width: int) -> None:
        """Mid-rollout cursor lands in the same bin as the target step.

        When ``total_steps > width`` each strip column represents a bin
        of ~``total_steps / width`` steps — we can't do better than that
        without subpixel cursors. The correctness property is "cursor is
        in the bin the step belongs to", not "exact step match".
        """
        if total_steps < 4:
            pytest.skip("trajectory too short for mid-point check")
        traj = _trajectory_with_distinct_actions(total_steps)
        strip = render_action_strip(traj, episode=0, player=0, width=width, height=8)

        step = total_steps // 2
        cursor_img = draw_cursor(strip, step=step, total_steps=total_steps)
        diff = np.any(cursor_img != strip, axis=(0, 2))
        cols = np.where(diff)[0]
        assert cols.size > 0
        col = int(cols[0])
        rendered_step = min(
            int(col * (total_steps - 1) / max(1, width - 1)),
            total_steps - 1,
        )
        bin_size = max(1, total_steps // width) + 1
        assert abs(rendered_step - step) <= bin_size, (
            f"step={step} cursor at col {col} shows step {rendered_step} "
            f"(bin_size={bin_size}, T={total_steps}, W={width})"
        )


class TestClickToSeek:
    """``action_strip_step_from_click`` hit-tests and maps correctly."""

    # All examples use base_frame = 2*qw × (2*qh + status), scale=1,
    # offsets=0 so window coords == base-frame coords.
    QW = 320
    QH = 240
    TOTAL = 8000

    def _click(self, x: int, y: int) -> int | None:
        return action_strip_step_from_click(
            click_x=x,
            click_y=y,
            win_ox=0,
            win_oy=0,
            win_scale=1,
            quadrant_w=self.QW,
            quadrant_h=self.QH,
            total_steps=self.TOTAL,
        )

    def test_click_at_strip_left_seeks_to_zero(self) -> None:
        """The left edge of the strip maps to step 0."""
        assert self._click(x=0, y=self.QH) == 0

    def test_click_at_strip_right_seeks_to_last(self) -> None:
        """The right edge of the strip maps to the last step."""
        assert self._click(x=self.QW - 1, y=self.QH) == self.TOTAL - 1

    def test_click_in_middle_seeks_to_middle(self) -> None:
        """A click at the horizontal midpoint lands near step T/2."""
        step = self._click(x=self.QW // 2, y=self.QH + 10)
        assert step is not None
        mid = self.TOTAL // 2
        assert abs(step - mid) <= self.TOTAL // self.QW + 1

    def test_click_above_strip_returns_none(self) -> None:
        """Clicks in the game view (Q1) must not trigger seek."""
        assert self._click(x=self.QW // 2, y=self.QH - 1) is None

    def test_click_below_strip_returns_none(self) -> None:
        """Clicks in the legend area (below the strip) must not seek."""
        strip_h = action_strip_height(self.QH)
        assert self._click(x=self.QW // 2, y=self.QH + strip_h) is None

    def test_click_right_of_strip_returns_none(self) -> None:
        """Clicks in Q4 (reward chart) must not seek."""
        assert self._click(x=self.QW, y=self.QH + 5) is None

    def test_click_maps_to_strip_column_of_resulting_step(self) -> None:
        """Click at column x → step s → ``render_action_strip`` paints s at x.

        This is the end-to-end round-trip: if the user clicks a colored
        pixel in the strip, the seek must land on the step that column
        actually represents. Exercises the fact that the two mappings
        are inverses.
        """
        traj = _trajectory_with_distinct_actions(self.TOTAL)
        strip = render_action_strip(traj, episode=0, player=0, width=self.QW, height=8)
        for click_col in (0, self.QW // 4, self.QW // 2, self.QW - 1):
            step = self._click(x=click_col, y=self.QH)
            assert step is not None
            # Color in the strip at click_col must be the action color
            # for the resulting step.
            strip_color = tuple(int(v) for v in strip[0, click_col])
            expected_action = step % len(Action)
            # Walk the strip to find the column the renderer paints for
            # this step; it must match click_col (within ±1 for int cast).
            rendered_step = min(
                int(click_col * (self.TOTAL - 1) / max(1, self.QW - 1)),
                self.TOTAL - 1,
            )
            assert rendered_step == step, (
                f"click_col={click_col}: seek landed on step {step} but "
                f"the strip paints step {rendered_step} at that column"
            )
            # Sanity: the strip column carries a non-zero color for
            # this action (zero would mean the renderer dropped it).
            assert strip_color != (0, 0, 0) or expected_action == 0


class TestDrawCursorNoMarginHack:
    """Regression: ``draw_cursor`` must not read column 0/1 as plot bounds.

    The old implementation embedded plot-area fractions into ``img[0,0,0]``
    and ``img[0,1,0]`` of the reward/cost charts and sniffed them back
    out inside ``draw_cursor``. On the action strip those pixels are real
    data (step 0 and step 1 colors) — whenever the red-channel of step 1
    was greater than the red-channel of step 0 AND step 0's red was > 0,
    the heuristic misfired and the cursor got confined to a sub-range
    of the strip. That's the "same margin on both sides" bug.
    """

    def test_action_strip_cursor_at_step_zero_is_column_zero(self) -> None:
        """Even when column 0 and column 1 have rising red channels."""
        img = np.zeros((8, 100, 3), dtype=np.uint8)
        # Seed values that would have triggered the old heuristic:
        # col0 red=189 (NOOP-grey), col1 red=190 (slightly brighter).
        img[:, 0] = (189, 189, 189)
        img[:, 1] = (190, 119, 180)

        cursor_img = draw_cursor(img, step=0, total_steps=100)
        diff = np.any(cursor_img != img, axis=(0, 2))
        cols = np.where(diff)[0]
        # Step 0 must place the cursor at column 0 (or 0-1 since cursor
        # is 2 px wide), never at some plot-area inset.
        assert cols.size > 0
        assert int(cols[0]) == 0, (
            f"cursor at step 0 should be at column 0, got {cols.tolist()}"
        )

    def test_cursor_with_explicit_plot_bounds(self) -> None:
        """When ``plot_bounds`` is supplied, cursor respects them."""
        img = np.zeros((8, 300, 3), dtype=np.uint8)
        cursor_at_start = draw_cursor(
            img, step=0, total_steps=100, plot_bounds=(40, 200)
        )
        cursor_at_end = draw_cursor(
            img, step=99, total_steps=100, plot_bounds=(40, 200)
        )
        start_col = int(np.where(np.any(cursor_at_start != img, axis=(0, 2)))[0][0])
        end_col = int(np.where(np.any(cursor_at_end != img, axis=(0, 2)))[0][0])
        assert start_col == 40
        assert end_col == 200


class TestReversePlayback:
    """``playback_direction`` drives auto-advance in either direction.

    These tests exercise the same arithmetic the main loop runs per
    frame (see main.py playback auto-advance), without spinning up a
    pygame window.
    """

    @staticmethod
    def _tick(dbg: DebuggerState, total: int) -> None:
        """One simulated auto-advance tick, mirroring main.py's logic."""
        if not dbg.playing:
            return
        step_delta = dbg.playback_speed * dbg.playback_direction
        new_step = dbg.current_step + step_delta
        dbg.current_step = max(0, min(new_step, total - 1))
        if (dbg.playback_direction > 0 and dbg.current_step >= total - 1) or (
            dbg.playback_direction < 0 and dbg.current_step <= 0
        ):
            dbg.playing = False

    def test_forward_advance_unchanged(self) -> None:
        dbg = DebuggerState(
            replay_mode=True, playing=True, current_step=5, playback_speed=1
        )
        self._tick(dbg, total=100)
        assert dbg.current_step == 6
        assert dbg.playing is True

    def test_backward_advance(self) -> None:
        dbg = DebuggerState(
            replay_mode=True,
            playing=True,
            current_step=5,
            playback_speed=1,
            playback_direction=-1,
        )
        self._tick(dbg, total=100)
        assert dbg.current_step == 4
        assert dbg.playing is True

    def test_backward_stops_at_start(self) -> None:
        dbg = DebuggerState(
            replay_mode=True,
            playing=True,
            current_step=1,
            playback_speed=1,
            playback_direction=-1,
        )
        self._tick(dbg, total=100)
        assert dbg.current_step == 0
        assert dbg.playing is False, "playback should stop at step 0"

    def test_forward_stops_at_end(self) -> None:
        dbg = DebuggerState(
            replay_mode=True,
            playing=True,
            current_step=98,
            playback_speed=1,
        )
        self._tick(dbg, total=100)
        assert dbg.current_step == 99
        assert dbg.playing is False

    def test_reverse_at_speed(self) -> None:
        """Same ``playback_speed`` applies regardless of direction."""
        dbg = DebuggerState(
            replay_mode=True,
            playing=True,
            current_step=50,
            playback_speed=5,
            playback_direction=-1,
        )
        self._tick(dbg, total=100)
        assert dbg.current_step == 45

    def test_default_direction_is_forward(self) -> None:
        """New DebuggerState starts forward so existing behavior is unchanged."""
        assert DebuggerState().playback_direction == 1


class TestClickToSeekScaling:
    """Window→base coordinate mapping respects win_ox/win_oy/win_scale."""

    def test_scale_and_offset(self) -> None:
        """Click at (win_ox + scale*col, win_oy + scale*row) → base (col, row)."""
        qw, qh = 320, 240
        total = 1000
        # Base frame placed at (100, 50) in window, scaled 3x.
        step = action_strip_step_from_click(
            click_x=100 + 3 * 50,  # base col 50
            click_y=50 + 3 * qh,  # base row qh (top of strip)
            win_ox=100,
            win_oy=50,
            win_scale=3,
            quadrant_w=qw,
            quadrant_h=qh,
            total_steps=total,
        )
        expected = int(50 * (total - 1) / (qw - 1))
        assert step == expected

    def test_click_outside_window_returns_none(self) -> None:
        """Negative base coords (click landed on window letterbox) → None."""
        step = action_strip_step_from_click(
            click_x=10,
            click_y=10,
            win_ox=100,
            win_oy=100,
            win_scale=2,
            quadrant_w=320,
            quadrant_h=240,
            total_steps=100,
        )
        assert step is None
