"""Tests for :mod:`factoriax.analysis.curriculum_strip`.

The strip draws one cell for each achievement bit, left to right in
bit order, in phase groups. A dashed line marks where hand crafting
stops and automation becomes necessary.

These tests assert on cell placement directly. The horizontal spacing
alone carries the phase groups. Cells inside a phase sit one small gap
apart, and a phase change opens a wider gap. :func:`render` gets a
smoke test only. It writes an image. An assertion on an image needs a
committed reference file, and every deliberate style change makes that
file invalid. The assertions therefore stop at "a non-empty file
appeared".

Positions and sizes are matplotlib data units, with the aspect ratio
locked to 1. A distance along x therefore equals the same distance
along y. The package ``conftest`` selects the Agg backend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from factoriax.analysis.curriculum_strip import (
    AchievementSpec,
    StripLayout,
    _cell_x_positions,
    render,
)


def _curriculum() -> list[AchievementSpec]:
    """Return four bits in three phases, with one hand-craft boundary.

    Bits 0 and 1 are hand craftable and share a phase. The list
    therefore covers a step inside a phase, two phase changes, and the
    switch to automation between bits 1 and 2.
    """
    return [
        AchievementSpec(bit=0, name="Mine", phase="Bootstrap", hand_craftable=True),
        AchievementSpec(bit=1, name="Smelt", phase="Bootstrap", hand_craftable=True),
        AchievementSpec(bit=2, name="Auto", phase="Automation", hand_craftable=False),
        AchievementSpec(bit=3, name="Launch", phase="Endgame", hand_craftable=False),
    ]


class TestCellXPositions:
    """Horizontal placement of the cells."""

    def test_first_cell_starts_at_the_origin(self) -> None:
        """Start the strip at x of 0 and grow to the right."""
        xs = _cell_x_positions(_curriculum(), StripLayout())
        assert xs[0] == 0.0

    def test_cells_in_one_phase_are_a_plain_gap_apart(self) -> None:
        """Use a stride of ``cell_w + gap`` inside a phase."""
        layout = StripLayout()
        xs = _cell_x_positions(_curriculum(), layout)
        assert xs[1] - xs[0] == pytest.approx(layout.cell_w + layout.gap)

    def test_a_phase_change_widens_the_stride(self) -> None:
        """Use a stride of ``cell_w + phase_gap`` at a phase boundary.

        The wider space is the only separator between one phase group
        and the next. The phase name over the group is a label, not a
        boundary.
        """
        layout = StripLayout()
        xs = _cell_x_positions(_curriculum(), layout)
        wide = pytest.approx(layout.cell_w + layout.phase_gap)
        assert xs[2] - xs[1] == wide
        assert xs[3] - xs[2] == wide

    def test_positions_are_strictly_increasing(self) -> None:
        """Keep every cell clear of the cell before it."""
        xs = _cell_x_positions(_curriculum(), StripLayout())
        assert all(b > a for a, b in zip(xs, xs[1:], strict=False))


class TestRender:
    """The top-level entry point."""

    def test_writes_png(self, tmp_path: Path) -> None:
        """Render the strip and return the path of the new file."""
        out = tmp_path / "strip.png"
        # The white fill runs the dark-label branch, and the
        # near-black fill runs the light one. "Endgame" is absent from
        # the palette, so the fallback color runs too.
        palette = {"Bootstrap": "#ffffff", "Automation": "#102030"}

        result = render(_curriculum(), out, phase_palette=palette)

        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_accepts_a_string_path_and_svg_extension(self, tmp_path: Path) -> None:
        """Take the output format from the file extension.

        ``out_path`` accepts a string as well as a
        :class:`~pathlib.Path`. A custom layout also reaches the
        geometry code.
        """
        out = tmp_path / "strip.svg"
        palette = {
            "Bootstrap": "#cc0000",
            "Automation": "#00aa00",
            "Endgame": "#0000cc",
        }

        render(
            _curriculum(),
            str(out),
            phase_palette=palette,
            layout=StripLayout(cell_w=2.0),
        )

        assert out.exists()

    def test_without_automation_skips_the_boundary(self, tmp_path: Path) -> None:
        """Draw no dashed line for a curriculum without automation."""
        specs = [
            AchievementSpec(bit=0, name="A", phase="P1", hand_craftable=True),
            AchievementSpec(bit=1, name="B", phase="P1", hand_craftable=True),
        ]
        out = tmp_path / "no_boundary.png"

        render(specs, out, phase_palette={"P1": "#777777"})

        assert out.exists()
