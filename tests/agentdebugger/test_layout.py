"""Tests for agentdebugger layout rendering."""

import numpy as np
import pytest

from factoriax.agentdebugger.layout import (
    STATUS_BAR_HEIGHT,
    compute_debugger_dimensions,
    render_debugger_frame,
)
from factoriax.agentdebugger.main import Debugger


class TestComputeDebuggerDimensions:
    """Tests for compute_debugger_dimensions."""

    def test_default_dimensions(self) -> None:
        """Default quadrants produce expected base size."""
        w, h = compute_debugger_dimensions()
        assert w == 200 * 2
        assert h == 150 * 2 + STATUS_BAR_HEIGHT

    def test_custom_quadrants(self) -> None:
        """Custom quadrant sizes scale correctly."""
        w, h = compute_debugger_dimensions(300, 200)
        assert w == 600
        assert h == 400 + STATUS_BAR_HEIGHT


@pytest.mark.slow
class TestRenderDebuggerFrame:
    """Tests for render_debugger_frame."""

    def test_output_shape(self, debugger: Debugger) -> None:
        """Frame has the expected dimensions."""
        qw, qh = 320, 240
        bw, bh = compute_debugger_dimensions(qw, qh)
        frame = render_debugger_frame(
            debugger._dbg,
            debugger._states,
            debugger._rewards,
            debugger._actions,
            debugger._costs,
            debugger._constraint_names,
            bw,
            bh,
            qw,
            qh,
            has_reward=True,
            has_cost=True,
        )
        assert frame.shape == (bh, bw, 3)
        assert frame.dtype == np.uint8

    def test_not_all_black(self, debugger: Debugger) -> None:
        """Rendered frame contains visible content."""
        qw, qh = 320, 240
        bw, bh = compute_debugger_dimensions(qw, qh)
        frame = render_debugger_frame(
            debugger._dbg,
            debugger._states,
            debugger._rewards,
            debugger._actions,
            debugger._costs,
            debugger._constraint_names,
            bw,
            bh,
            qw,
            qh,
        )
        assert frame.sum() > 0

    def test_frame_after_steps(self, debugger: Debugger) -> None:
        """Frame renders correctly after accumulating steps."""
        for _ in range(3):
            debugger._execute_step(0)
        qw, qh = 320, 240
        bw, bh = compute_debugger_dimensions(qw, qh)
        frame = render_debugger_frame(
            debugger._dbg,
            debugger._states,
            debugger._rewards,
            debugger._actions,
            debugger._costs,
            debugger._constraint_names,
            bw,
            bh,
            qw,
            qh,
            has_reward=True,
            has_cost=True,
        )
        assert frame.shape == (bh, bw, 3)
