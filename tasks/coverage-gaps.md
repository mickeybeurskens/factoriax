# Coverage gaps

Every module in `factoriax/` that has no mirror entry under `tests/`. A
module with an entry is not listed: the entry is the test.

Three verdicts:

- **gap** means write a test. Nothing drives the module.
- **covered** means a test drives it, but through another module. The
  missing piece is a mirror entry, not coverage.
- **waived** means do not write one, for the reason given.

Measured with `uv run pytest -q --cov` on 2026-08-03, at 1411 tests.

| module | lines | coverage | verdict | note |
|---|---:|---:|---|---|
| `playground/editor/toolbar.py` | 788 | 0% | gap | nothing reaches it |
| `playground/ui/panels.py` | 761 | 27% | gap | nothing reaches it |
| `playground/editor/inventory_panel.py` | 390 | 0% | gap | nothing reaches it |
| `analysis/eval.py` | 299 | 21% | gap | nothing reaches it |
| `playground/menu/main_menu.py` | 219 | 0% | gap | nothing reaches it |
| `playground/ui/forms.py` | 162 | 58% | gap | nothing reaches it |
| `playground/ui/window.py` | 83 | 25% | gap | nothing reaches it |
| `engine/envs/miner_curriculum.py` | 368 | 94% | covered | tests/integration/scenarios/test_miner_curriculum.py |
| `analysis/categories.py` | 252 | 100% | covered | tests/analysis/test_recipe_graph.py, through item_palette |
| `engine/envs/science_tiers.py` | 238 | 100% | covered | tests/integration/scenarios/test_science_tiers.py |
| `engine/envs/common.py` | 217 | 99% | covered | the scenario tests, which all build through it |
| `engine/crafting.py` | 182 | 100% | covered | the step tests, through craft_recipe |
| `playground/ui/compositing.py` | 154 | 100% | covered | the play and editor render tests |
| `playground/ui/primitives.py` | 100 | 93% | covered | the play render tests |
| `make.py` | 98 | 100% | covered | tests/integration/scenarios/, through env_from_name |
| `playground/ui/fonts.py` | 91 | 90% | covered | the play render tests |
| `engine/envs/mining.py` | 87 | 100% | covered | tests/integration/scenarios/test_mining.py |
| `playground/play/play_state.py` | 65 | 100% | covered | tests/contracts/test_no_direct_state_edits.py |
| `playground/ui/labels.py` | 20 | 100% | covered | the play render tests |
| `playground/editor/main.py` | 1680 | 0% | waived | entry point for the editor |
| `playground/play/main.py` | 632 | 0% | waived | entry point for the play UI |
| `analysis/video.py` | 238 | 37% | waived | shells out to ffmpeg |
| `playground/app.py` | 67 | 0% | waived | entry point that dispatches to the two above |

## What to do first

7 modules are true gaps. Two of them are large and drawn:
`playground/editor/toolbar.py` at 788 lines and 0 percent, and
`playground/ui/panels.py` at 761 lines and 27 percent. A test for either
costs a display fixture and an output-contract assertion, which is the shape
the files under `tests/playground/ui/` already use.

`playground/editor/inventory_panel.py` and `playground/menu/main_menu.py`
are the same shape and smaller. `analysis/eval.py` at 21 percent is the one
gap outside the playground.

The twelve **covered** rows need a mirror entry, not a test. Four of the
`engine/envs` modules have scenario tests under
`tests/integration/scenarios/`, which is where an end-to-end rollout
belongs. Their absence from `tests/engine/envs/` is correct and deliberate.

