# Documentation Pass: Module Checklist

Systematic docstring pass over every module in `factoriax/`.

## Process rules

1. Work through the checklist below in order, one module at a time.
2. After each module is edited, stop. The reviewer reads the file.
3. Iterate on that module until the reviewer greenlights it.
4. Only then tick its box and move to the next module.
5. Do not edit ahead. One module in flight at any time.
6. Docstring format for this project: NumPy style.

## Tooling

`ruff` enforces the format via pydocstyle (`select = ["D", ...]`, `convention = "numpy"`).
Check a module before handing it over:

```sh
uv run ruff check factoriax/<path>.py
```

Sphinx has `napoleon_google_docstring = False`, so a Google-style block fails
visibly instead of rendering silently.

There is no CI in this repository, so the linter is local-only until one exists.

### Backlog at the start of the pass

A previous automated pass left malformed docstrings across the tree: each
parameter is repeated with the type appended and no description, and most
`Returns` sections are empty stubs. Counts when the pass started:

| Check | Count |
|---|---|
| `D` violations | 669 |
| `W293` (blank line with whitespace, mostly inside the broken docstrings) | 2470 |
| Files with duplicated `name : type :` stubs | 56 |

Pre-existing non-docstring violations to leave alone unless a module under
edit owns them: 16 `F821`, 7 `E501`, 6 `F401`, 5 `I001`, 1 `F841`.

The per-module job is therefore mostly deleting generated noise and writing a
real contract, not editing prose.

## Documentation checklist

Apply this to every module in the list.

### Before you write

- [ ] Identify what the reader already knows and what they need to know after reading. The gap between those two states is the scope of the docstring (Losh: find state 1, find state 2, pick one idea that moves the reader closer, repeat)
- [ ] Classify the content. A docstring is reference material. Narrative, motivation, and onboarding belong in a tutorial or explanation page (mixing the four types causes most documentation defects)
- [ ] Confirm the function has a describable contract. If the behavior is hard to state, fix the design first (antirez: behavior that emerges at random from complexity cannot be documented, so return to the code)

### Summary line

- [ ] Use imperative mood: "Return the user", not "Returns the user"
- [ ] Make it self-contained. It renders in IDE tooltips, `help()` output, and API indexes without surrounding context
- [ ] One line, terminated with a period, on the same line as the opening quotes
- [ ] Describe the operation, not the object. "Parse a config file" over "A parser for config files"
- [ ] Do not restate the signature. `get_user(id)` documented as "Get the user by id" adds no information

### Body content

- [ ] Insert a blank line between summary and body
- [ ] Specify the contract, not the implementation. State what callers can rely on (completeness and correctness come first; the docstring is a contract)
- [ ] Define behavior at the boundaries: `None`, empty collections, zero, negative values, out-of-range input
- [ ] Record the reason for non-obvious behavior: the constraint, the upstream requirement, the deliberate tradeoff
- [ ] Explain domain concepts the reader may not have. If the function assumes trigonometry, a wire protocol, or a statistical model, state the assumption (antirez classifies these as teaching comments)
- [ ] Declare units, coordinate system, timezone, encoding, and numeric precision. Type annotations do not carry these
- [ ] State side effects: argument mutation, I/O, global state changes, thread safety, idempotency

### Parameters, returns, exceptions

- [ ] Document every parameter with its meaning. The annotation supplies the type; the prose supplies the semantics
- [ ] Document the return value including what an empty, zero, or `None` result signifies
- [ ] List raised exceptions and the conditions that trigger each one
- [ ] Do not duplicate information that already exists in the signature. Duplicated facts drift out of sync

### Examples

- [ ] Derive examples from real call sites rather than placeholder identifiers (Evans, "Write good examples by starting with real code")
- [ ] Make examples executable as written, with imports and setup included
- [ ] Use doctest format where possible so the example is verified by the test suite
- [ ] Weigh the maintenance cost. Every example is a second artifact to keep current

### Exclusions

- [ ] Move algorithm descriptions and implementation notes to a comment placed below the docstring, outside the string the caller sees
- [ ] Omit line-by-line restatement of the code (readable from the source itself)
- [ ] Omit tutorial sequences, design philosophy, and single-case problem solutions (four purposes in one text serves none of them)
- [ ] Omit claims with a short shelf life that no test will catch when they expire

### Style mechanics

- [ ] Select one format (Google, NumPy, or reST) and apply it across the project
- [ ] Use one term per concept. Do not vary vocabulary for readability
- [ ] Prefer plain language. Use jargon only when it carries meaning the plain word cannot
- [ ] Keep sentences short. Read the docstring aloud to surface awkward constructions
- [ ] Remove "simply", "just", "obviously", and "trivially"
- [ ] Document private helpers. An undocumented `_parse_response` costs the next reader hours
- [ ] Remove all em dashes and LLM language in favor of clear technical language. Do not convolute.

### Project level

- [ ] Store documentation in the same repository as the code so one commit contains the change, the tests, and the docs
- [ ] Run a docstring linter in CI to check for missing docstrings and undocumented parameters
- [ ] Add documentation unit tests that assert every plugin hook, config option, or public class has a matching documented section. Mark existing gaps `xfail` to adopt this without a full backfill
- [ ] Flag missing or stale docs during code review
- [ ] Include docstrings in the definition of done

### Coverage check

- [ ] Verify a new user can locate the function without knowing its name in advance. If not, add a guide above the API reference (docstrings provide no structure beyond the namespace they sit in)
- [ ] Verify you have not treated complete docstring coverage as complete documentation (docstrings work well once the reader already knows the project)

## Modules

- [ ] `factoriax/__init__.py`
- [ ] `factoriax/make.py`

### `factoriax/engine/`

- [ ] `engine/__init__.py`
- [ ] `engine/achievements.py`
- [ ] `engine/actions.py`
- [ ] `engine/belts.py`
- [ ] `engine/constants.py`
- [ ] `engine/crafting.py`
- [ ] `engine/game_logic.py`
- [ ] `engine/jax_renderer.py`
- [ ] `engine/levels.py`
- [ ] `engine/machine_spec.py`
- [ ] `engine/machines.py`
- [ ] `engine/observations.py`
- [ ] `engine/placement.py`
- [ ] `engine/recipes.py`
- [ ] `engine/recipes_io.py`
- [ ] `engine/rewards.py`
- [ ] `engine/state.py`
- [ ] `engine/tables.py`

#### `factoriax/engine/envs/`

- [ ] `engine/envs/__init__.py`
- [ ] `engine/envs/base.py`
- [ ] `engine/envs/common.py`
- [ ] `engine/envs/easy_rocket.py`
- [ ] `engine/envs/miner_curriculum.py`
- [ ] `engine/envs/mining.py`
- [ ] `engine/envs/registry.py`
- [ ] `engine/envs/rocket.py`
- [ ] `engine/envs/science_tiers.py`
- [ ] `engine/envs/wrappers.py`

### `factoriax/analysis/`

- [ ] `analysis/__init__.py`
- [ ] `analysis/actions.py`
- [ ] `analysis/build_progression.py`
- [ ] `analysis/categories.py`
- [ ] `analysis/curriculum_strip.py`
- [ ] `analysis/eval.py`
- [ ] `analysis/graph_layout.py`
- [ ] `analysis/inventory.py`
- [ ] `analysis/milestones.py`
- [ ] `analysis/multiagent.py`
- [ ] `analysis/recipe_graph.py`
- [ ] `analysis/recorder.py`
- [ ] `analysis/state.py`
- [ ] `analysis/trajectory.py`
- [ ] `analysis/utils.py`
- [ ] `analysis/video.py`

### `factoriax/assets/`

- [ ] `assets/__init__.py`
- [ ] `assets/build_atlas.py`

### `factoriax/playground/`

- [ ] `playground/__init__.py`
- [ ] `playground/__main__.py`
- [ ] `playground/app.py`
- [ ] `playground/config.py`

#### `factoriax/playground/editor/`

- [ ] `playground/editor/__init__.py`
- [ ] `playground/editor/__main__.py`
- [ ] `playground/editor/canvas.py`
- [ ] `playground/editor/dialogs.py`
- [ ] `playground/editor/inventory_panel.py`
- [ ] `playground/editor/main.py`
- [ ] `playground/editor/slot_display.py`
- [ ] `playground/editor/state.py`
- [ ] `playground/editor/toolbar.py`

#### `factoriax/playground/menu/`

- [ ] `playground/menu/__init__.py`
- [ ] `playground/menu/controls_menu.py`
- [ ] `playground/menu/main_menu.py`

#### `factoriax/playground/play/`

- [ ] `playground/play/__init__.py`
- [ ] `playground/play/__main__.py`
- [ ] `playground/play/achievements.py`
- [ ] `playground/play/game_ui.py`
- [ ] `playground/play/launch_screen.py`
- [ ] `playground/play/main.py`
- [ ] `playground/play/play_state.py`
- [ ] `playground/play/transfer.py`
- [ ] `playground/play/ui.py`

#### `factoriax/playground/ui/`

- [ ] `playground/ui/__init__.py`
- [ ] `playground/ui/compositing.py`
- [ ] `playground/ui/fonts.py`
- [ ] `playground/ui/forms.py`
- [ ] `playground/ui/icons.py`
- [ ] `playground/ui/labels.py`
- [ ] `playground/ui/panels.py`
- [ ] `playground/ui/primitives.py`
- [ ] `playground/ui/scaling.py`
- [ ] `playground/ui/theme.py`
- [ ] `playground/ui/window.py`
