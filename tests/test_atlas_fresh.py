"""Verify the committed sprite atlas matches what build_atlas.py produces.

The atlas at ``factoriax/assets/atlas.png`` (and its sidecar
``atlas.json``) is the source of truth for sprite gathers in the JAX
renderer. Hand-edits to either file will silently desync from the
build script, so this test re-runs the script into a tempdir and
asserts byte-equivalence.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import orjson

from factoriax.assets.build_atlas import build_atlas

_REPO = Path(__file__).resolve().parent.parent
_COMMITTED_PNG = _REPO / "factoriax" / "assets" / "atlas.png"
_COMMITTED_JSON = _REPO / "factoriax" / "assets" / "atlas.json"


def test_atlas_png_matches_committed(tmp_path: Path) -> None:
    """Regenerated atlas.png is byte-identical to the committed copy."""
    out_png = tmp_path / "atlas.png"
    out_json = tmp_path / "atlas.json"
    build_atlas(out_png, out_json)

    assert out_png.read_bytes() == _COMMITTED_PNG.read_bytes(), (
        "atlas.png is stale. Re-run factoriax/assets/build_atlas.py after editing "
        "the procedural sprite code."
    )


def test_atlas_json_matches_committed(tmp_path: Path) -> None:
    """Regenerated atlas.json is byte-identical to the committed copy."""
    out_png = tmp_path / "atlas.png"
    out_json = tmp_path / "atlas.json"
    build_atlas(out_png, out_json)

    assert out_json.read_bytes() == _COMMITTED_JSON.read_bytes(), (
        "atlas.json is stale. Re-run factoriax/assets/build_atlas.py."
    )


def test_atlas_json_declares_expected_layout() -> None:
    """The committed atlas.json declares the layout from atlas.layout.md."""
    payload = orjson.loads(_COMMITTED_JSON.read_bytes())
    assert payload["cell_px"] == 32
    assert payload["rows"] == 8
    assert payload["cols"] == 34
    assert payload["direction_axis"] == ["LEFT", "RIGHT", "UP", "DOWN"]
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
    assert img.shape == (256, 1088, 4), (
        f"Atlas PNG shape {img.shape} doesn't match (256, 1088, 4). "
        "Layout says 8 rows × 34 cols × 32 px, RGBA."
    )
    assert img.dtype.name == "uint8"


def test_atlas_alpha_channel_is_non_trivial() -> None:
    """Some atlas cells must use alpha < 255 so render_map's blend works.

    Block cells stay opaque (terrain is always the ground truth) but
    machine and player cells carry transparent regions so placed
    objects read as overlays on terrain rather than as solid tiles.
    """
    img = iio.imread(_COMMITTED_PNG)
    transparent_pixels = (img[..., 3] < 255).sum()
    assert transparent_pixels > 0, (
        "Atlas has no transparent pixels — alpha compositing in "
        "render_map will be a no-op. Re-run factoriax/assets/build_atlas.py."
    )
