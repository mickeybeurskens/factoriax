# Sprite Atlas Layout

Source of truth for `factoriax/assets/atlas.png` and its sidecar
`factoriax/assets/atlas.json`. Both are regenerated from
`scripts/build_atlas.py`; CI verifies they match this layout via
`tests/test_atlas_fresh.py`.

## Grid

The atlas is a uniform grid of 32×32 px RGB cells, packed
**row-major by category, column-by-enum-value**. Indexing a sprite
is two integers: `(row, col)`. The renderer reads them via
`atlas.json`, which the `JaxRenderer` slurps at construction time.

Cell size is fixed at 32×32. Display tile size (`tile_px`) is a
renderer-construction parameter; downsampling from 32×32 to the
display size happens at gather time using nearest-neighbour.
Upsampling (rare; for "zoomed-in" views) uses the same path.

The atlas image dimensions are `(num_rows × 32, num_cols × 32, 3)`
uint8 with no alpha. Transparency is not used in v1 — every cell
is a fully-opaque RGB sprite. Layered compositing (machine on top
of terrain) happens in the renderer via masks derived from
`state.machine_types`, not via per-pixel alpha in the atlas.

## Rows

Each row corresponds to one enum category. Cells beyond the enum's
defined values are filled with the row's "missing" color (magenta,
`(255, 0, 255)`) so a future enum extension produces a visible
artifact rather than silent zeros.

| Row | Category    | Source enum                      | Defined cells | Notes |
| --: | ----------- | -------------------------------- | ------------: | ----- |
|   0 | blocks      | `factoriax.constants.BlockType`  | 11            | Solid colors per build_block_atlas (v1). |
|   1 | machines    | `factoriax.constants.MachineType`| 11            | Solid colors per build_machine_atlas (v1). |
|   2 | items       | `factoriax.constants.ItemType`   | 33            | Solid colors from `ITEM_COLORS`. |
|   3 | misc        | (manually enumerated)            | 2             | col 0 = player sprite; col 1 = biter sprite. |
|   4 | digits      | digits 0-9                       | 10            | Each cell is 32×32; the 3×5 glyph is rendered at the cell's top-left, padded to 32×32 with zeros. |

Width of the atlas is `max(num_cells_per_row) = 33` (driven by
`ItemType`). Height is `num_rows = 5`. Atlas image:
`(5 × 32, 33 × 32, 3) = (160, 1056, 3)` uint8.

## Sidecar JSON shape

`factoriax/assets/atlas.json` is a single object:

```json
{
  "cell_px": 32,
  "rows": 5,
  "cols": 33,
  "categories": {
    "blocks":   {"row": 0, "names": ["INVALID", "OUT_OF_BOUNDS", ...], "missing": "magenta"},
    "machines": {"row": 1, "names": ["NONE", "MINER", ...], "missing": "magenta"},
    "items":    {"row": 2, "names": ["EMPTY", "COAL", ...], "missing": "magenta"},
    "misc":     {"row": 3, "names": ["player", "biter"]},
    "digits":   {"row": 4, "names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]}
  }
}
```

Names within each category appear in enum-value order (so
`names[i]` lives at `(row, i)`). A renderer that wants the
`MINER` machine sprite reads
`(blocks/machines.row, names.index("MINER"))` — for v1 the renderer
hardcodes the row constants rather than parsing the JSON, but the
JSON is canonical for tooling and tests.

## Determinism

The build script must produce byte-identical output across runs:

- Sort enum members by integer value before iterating.
- Use `np.full(..., dtype=np.uint8)` for backing arrays — no
  uninitialised memory.
- Encode the PNG with `imageio.imwrite(..., compress_level=6)`
  (default) and a fixed metadata block.
- Encode the JSON via `orjson.dumps(payload, option=orjson.OPT_INDENT_2)`
  and a sorted-keys flag at the top level.

`tests/test_atlas_fresh.py` regenerates both files in a tempdir and
asserts byte-equivalence with the committed copies.

## v2 extensions (not in this spec)

- **Direction overlays**: 4 cells (one per `Direction`) with arrows
  drawn over a transparent background. Belt orientation,
  miner/assembler facing.
- **Animation frames**: extra columns per cell for sprite cycling
  (e.g. conveyor flow). Renderer reads
  `(row, base_col + state.timestep % num_frames)`.
- **Status overlays**: separate rows for "machine has output",
  "machine starved of input", etc. — visual debugging aids.
- **Glyph atlas**: a richer text atlas to phase out the pygame text
  overlay. Out of scope for v1 by spec decision.

These are listed so the layout doesn't silently invalidate them.
Adding a row at the bottom is non-breaking; reordering existing
rows would be breaking. Don't reorder.
