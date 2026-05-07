"""Verify that the committed sprite atlas matches what build_atlas.py produces.

The atlas at ``factoriax/assets/atlas.png`` (and its sidecar
``atlas.json``) is the source of truth for sprite gathers in the JAX
renderer. Hand-edits to either file will silently desync from the
build script, so this test re-runs the script into a tempdir and
asserts byte-equivalence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v3 as iio
import orjson

# Make scripts/ importable for the build_atlas function.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from build_atlas import build_atlas  # noqa: E402

_COMMITTED_PNG = _REPO / "factoriax" / "assets" / "atlas.png"
_COMMITTED_JSON = _REPO / "factoriax" / "assets" / "atlas.json"


def test_atlas_png_matches_committed(tmp_path: Path) -> None:
    """Regenerated atlas.png is byte-identical to the committed copy."""
    out_png = tmp_path / "atlas.png"
    out_json = tmp_path / "atlas.json"
    build_atlas(out_png, out_json)

    assert out_png.read_bytes() == _COMMITTED_PNG.read_bytes(), (
        "atlas.png is stale. Re-run scripts/build_atlas.py after editing "
        "the procedural sprite code."
    )


def test_atlas_json_matches_committed(tmp_path: Path) -> None:
    """Regenerated atlas.json is byte-identical to the committed copy."""
    out_png = tmp_path / "atlas.png"
    out_json = tmp_path / "atlas.json"
    build_atlas(out_png, out_json)

    assert out_json.read_bytes() == _COMMITTED_JSON.read_bytes(), (
        "atlas.json is stale. Re-run scripts/build_atlas.py."
    )


def test_atlas_json_declares_expected_layout() -> None:
    """The committed atlas.json declares the layout from atlas.layout.md."""
    payload = orjson.loads(_COMMITTED_JSON.read_bytes())
    assert payload["cell_px"] == 32
    assert payload["rows"] == 5
    assert payload["cols"] == 33
    assert set(payload["categories"]) == {
        "blocks",
        "machines",
        "items",
        "misc",
        "digits",
    }


def test_atlas_png_has_expected_shape() -> None:
    """The committed atlas.png has the dimensions implied by the layout."""
    img = iio.imread(_COMMITTED_PNG)
    assert img.shape == (160, 1056, 3), (
        f"Atlas PNG shape {img.shape} doesn't match (160, 1056, 3). "
        "Layout says 5 rows × 33 cols × 32 px."
    )
    assert img.dtype.name == "uint8"
