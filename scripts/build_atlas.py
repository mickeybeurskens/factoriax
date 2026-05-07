"""Build the FactoriaX sprite atlas from the procedural drawing code.

Generates ``factoriax/assets/atlas.png`` and the sidecar
``factoriax/assets/atlas.json`` per the layout documented in
``factoriax/assets/atlas.layout.md``. The renderer (post-Item-4 of
SPEC.md) reads from this atlas instead of building per-category
atlases at JIT-init time.

v1 is a placeholder: every sprite is captured from the current
procedural code in ``factoriax/jax_renderer.py``. Future iterations
swap the PNG for hand-drawn art with the same layout — see
``atlas.layout.md`` for the contract.

Usage::

    uv run python scripts/build_atlas.py
    uv run python scripts/build_atlas.py --out custom_dir/

The output is hash-stable across runs; CI verifies this via
``tests/test_atlas_fresh.py``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import orjson

from factoriax.constants import (
    ITEM_COLORS,
    NUM_ITEM_TYPES,
    BlockType,
    ItemType,
    MachineType,
)
from factoriax.jax_renderer import build_digit_atlas

logger = logging.getLogger(__name__)

CELL_PX: int = 32
NUM_ROWS: int = 5
NUM_COLS: int = 33  # max(num_block, num_machine, num_item, ...) — driven by ItemType
MISSING_COLOR: tuple[int, int, int] = (255, 0, 255)  # magenta — flags gaps


# Row indices — keep in sync with atlas.layout.md.
ROW_BLOCKS: int = 0
ROW_MACHINES: int = 1
ROW_ITEMS: int = 2
ROW_MISC: int = 3
ROW_DIGITS: int = 4


# Source-of-truth color tables. Mirrors the dictionaries in
# factoriax.jax_renderer.build_block_atlas and build_machine_atlas.
# Keep these in sync; v2 (atlas swap to hand-drawn art) replaces the
# functions and the dicts together.
_BLOCK_COLORS: dict[BlockType, tuple[int, int, int]] = {
    BlockType.INVALID: (139, 90, 43),
    BlockType.OUT_OF_BOUNDS: (30, 30, 30),
    BlockType.DIRT: (139, 90, 43),
    BlockType.WATER: (64, 164, 223),
    BlockType.IRON: (192, 192, 192),
    BlockType.COPPER: (184, 115, 51),
    BlockType.COAL: (54, 54, 54),
    BlockType.NEST: (90, 40, 60),
}

_MACHINE_COLORS: dict[MachineType, tuple[int, int, int]] = {
    MachineType.NONE: (0, 0, 0),
    MachineType.MINER: (0, 200, 0),
    MachineType.PALLET: (170, 170, 175),
    MachineType.ASSEMBLER: (160, 80, 200),
    MachineType.CONVEYOR_BELT: (220, 180, 50),
    MachineType.ROCKET: (240, 240, 240),
    MachineType.SCIENCE_LAB: (76, 29, 149),
}


def _solid_cell(rgb: tuple[int, int, int]) -> np.ndarray:
    """Return a 32x32 RGB uint8 cell filled with *rgb*."""
    cell = np.empty((CELL_PX, CELL_PX, 3), dtype=np.uint8)
    cell[:, :] = rgb
    return cell


def _circle_cell(
    rgb: tuple[int, int, int],
    bg: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Return a 32x32 RGB uint8 cell with a filled circle on *bg*.

    Mirrors ``factoriax.jax_renderer._circle_tile`` at the atlas's
    base resolution.
    """
    cell = np.full((CELL_PX, CELL_PX, 3), bg, dtype=np.uint8)
    center = CELL_PX / 2.0
    radius_sq = (CELL_PX / 3.0) ** 2
    ys, xs = np.mgrid[:CELL_PX, :CELL_PX]
    dist_sq = (xs - center + 0.5) ** 2 + (ys - center + 0.5) ** 2
    cell[dist_sq <= radius_sq] = rgb
    return cell


def _digit_cell(digit: int, digit_atlas: np.ndarray) -> np.ndarray:
    """Render a single digit at the cell's top-left.

    The procedural digit atlas is 3 wide × 5 tall bool mask. We blit
    it as white-on-black at the top-left of a 32x32 cell so the rest
    of the cell is unused space.
    """
    cell = np.zeros((CELL_PX, CELL_PX, 3), dtype=np.uint8)
    glyph = digit_atlas[digit]  # shape (5, 3) bool
    cell[: glyph.shape[0], : glyph.shape[1]] = np.where(
        glyph[:, :, None],
        np.array([255, 255, 255], dtype=np.uint8),
        np.array([0, 0, 0], dtype=np.uint8),
    )
    return cell


def _ordered_enum_names(enum_cls: type) -> list[str]:
    """Return enum member names sorted by integer value."""
    return [m.name for m in sorted(enum_cls, key=int)]


def _build_atlas_array() -> np.ndarray:
    """Construct the (160, 1056, 3) uint8 atlas image.

    Cells beyond a category's defined enum values are filled with
    MISSING_COLOR so future enum extensions produce a visible
    artifact rather than silent zeros.
    """
    height = NUM_ROWS * CELL_PX
    width = NUM_COLS * CELL_PX
    atlas = np.empty((height, width, 3), dtype=np.uint8)
    atlas[:, :] = MISSING_COLOR

    def _put(row: int, col: int, cell: np.ndarray) -> None:
        y0, x0 = row * CELL_PX, col * CELL_PX
        atlas[y0 : y0 + CELL_PX, x0 : x0 + CELL_PX] = cell

    # Row 0: blocks.
    for block in sorted(BlockType, key=int):
        rgb = _BLOCK_COLORS.get(block)
        if rgb is None:
            continue  # leave magenta — flags the gap
        _put(ROW_BLOCKS, int(block), _solid_cell(rgb))

    # Row 1: machines.
    for machine in sorted(MachineType, key=int):
        rgb = _MACHINE_COLORS.get(machine)
        if rgb is None:
            continue
        _put(ROW_MACHINES, int(machine), _solid_cell(rgb))

    # Row 2: items. ITEM_COLORS may not cover every ItemType; missing
    # entries stay magenta.
    for item in sorted(ItemType, key=int):
        rgb = ITEM_COLORS.get(item)
        if rgb is None:
            continue
        _put(ROW_ITEMS, int(item), _solid_cell(rgb))

    # Row 3: misc.
    _put(ROW_MISC, 0, _circle_cell((255, 100, 100)))  # player
    # col 1 (biter) intentionally left magenta until a real sprite lands

    # Row 4: digits.
    digit_atlas = np.asarray(build_digit_atlas())
    for digit in range(10):
        _put(ROW_DIGITS, digit, _digit_cell(digit, digit_atlas))

    return atlas


def _build_atlas_json() -> dict:
    """Construct the sidecar JSON describing the atlas layout."""
    return {
        "cell_px": CELL_PX,
        "rows": NUM_ROWS,
        "cols": NUM_COLS,
        "categories": {
            "blocks": {
                "row": ROW_BLOCKS,
                "names": _ordered_enum_names(BlockType),
                "missing": "magenta",
            },
            "machines": {
                "row": ROW_MACHINES,
                "names": _ordered_enum_names(MachineType),
                "missing": "magenta",
            },
            "items": {
                "row": ROW_ITEMS,
                "names": _ordered_enum_names(ItemType),
                "missing": "magenta",
            },
            "misc": {
                "row": ROW_MISC,
                "names": ["player", "biter"],
            },
            "digits": {
                "row": ROW_DIGITS,
                "names": [str(d) for d in range(10)],
            },
        },
    }


def build_atlas(out_png: Path, out_json: Path) -> None:
    """Build atlas.png and atlas.json into the given paths.

    The function is deterministic: running it twice into the same
    directory produces byte-identical files. CI verifies this via
    ``tests/test_atlas_fresh.py``.

    Args:
        out_png: Path to write the atlas PNG.
        out_json: Path to write the sidecar JSON.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    atlas = _build_atlas_array()
    iio.imwrite(out_png, atlas, extension=".png")

    payload = _build_atlas_json()
    out_json.write_bytes(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    )

    # Sanity-check the items row matches NUM_ITEM_TYPES — this catches
    # an enum extension landing without a matching atlas regen.
    assert len(payload["categories"]["items"]["names"]) == NUM_ITEM_TYPES


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "factoriax" / "assets",
        help="Output directory for atlas.png and atlas.json.",
    )
    args = parser.parse_args()
    out_png = args.out / "atlas.png"
    out_json = args.out / "atlas.json"
    build_atlas(out_png, out_json)
    logger.info("Wrote %s and %s", out_png, out_json)
    print(f"Wrote {out_png} ({out_png.stat().st_size} bytes)")
    print(f"Wrote {out_json} ({out_json.stat().st_size} bytes)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
