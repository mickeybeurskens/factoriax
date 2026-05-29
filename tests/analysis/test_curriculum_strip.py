"""Tests for :mod:`factoriax.analysis.curriculum_strip`.

Synthetic achievement sequences are rendered to a temporary PNG/SVG; the
assertions check the output is written and the public dataclasses behave.
The package conftest forces the Agg backend.
"""

from __future__ import annotations

from pathlib import Path

from factoriax.analysis.curriculum_strip import (
    AchievementSpec,
    StripLayout,
    render,
)


def _curriculum() -> list[AchievementSpec]:
    """Four bits across three phases with a hand -> automation boundary."""
    return [
        AchievementSpec(bit=0, name="Mine", phase="Bootstrap", hand_craftable=True),
        AchievementSpec(bit=1, name="Smelt", phase="Bootstrap", hand_craftable=True),
        AchievementSpec(bit=2, name="Auto", phase="Automation", hand_craftable=False),
        AchievementSpec(bit=3, name="Launch", phase="Endgame", hand_craftable=False),
    ]


def test_render_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "strip.png"
    # White fill exercises the dark-label branch and the dark fill the light
    # branch; "Endgame" is absent so the fallback-colour path also runs.
    palette = {"Bootstrap": "#ffffff", "Automation": "#102030"}

    result = render(_curriculum(), out, phase_palette=palette)

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_custom_layout_and_svg(tmp_path: Path) -> None:
    out = tmp_path / "strip.svg"
    palette = {"Bootstrap": "#cc0000", "Automation": "#00aa00", "Endgame": "#0000cc"}

    render(
        _curriculum(),
        str(out),
        phase_palette=palette,
        layout=StripLayout(cell_w=2.0),
    )

    assert out.exists()


def test_render_without_automation_skips_boundary(tmp_path: Path) -> None:
    # Every bit is hand-craftable, so there is no boundary line to draw.
    specs = [
        AchievementSpec(bit=0, name="A", phase="P1", hand_craftable=True),
        AchievementSpec(bit=1, name="B", phase="P1", hand_craftable=True),
    ]
    out = tmp_path / "no_boundary.png"

    render(specs, out, phase_palette={"P1": "#777777"})

    assert out.exists()
