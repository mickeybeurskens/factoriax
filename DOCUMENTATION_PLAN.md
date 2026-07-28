# Documentation Pass: Module Checklist

Systematic docstring pass over every module in `factoriax/`.

## Process rules

1. Work through the **Modules** list at the bottom in the order given, one
   module at a time.
2. After each module is edited, stop. The reviewer reads the file.
3. Iterate on that module until the reviewer greenlights it.
4. Only then tick its box and move to the next module.
5. Do not edit ahead. One module in flight at any time.
6. Docstring format for this project: NumPy style.

The **Rubric** section is grading criteria, not trackable state. It applies to
every module, so it carries no checkboxes. The only boxes in this document are
the module list and the one-time tasks.

## Tooling

`ruff` enforces the format via pydocstyle (`select = ["D", ...]`,
`convention = "numpy"`). Check a module before handing it over:

```sh
uv run ruff check factoriax/<path>.py
```

Sphinx has `napoleon_google_docstring = False`, so a Google-style block fails
visibly instead of rendering silently.

There is no CI in this repository and none is planned. The linter is run by
hand, per module, as part of step 2 above.

Doctest collection is not enabled either: `pyproject.toml` sets
`testpaths = ["tests"]` with no `--doctest-modules`. Examples in docstrings are
inert text that nothing verifies. Write them anyway where they earn their
keep, but treat each one as an unverified artifact that can rot.

### Backlog at the start of the pass

A previous automated pass left malformed docstrings across the tree: each
parameter is repeated with the type appended and no description, and most
`Returns` sections are empty stubs. Counts when the pass started:

| Check | Count |
|---|---|
| `D` violations | 669 |
| `W293` (blank line with whitespace, mostly inside the broken docstrings) | 2470 |
| Files with duplicated `name : type :` stubs | 56 |
| Modules to cover | 84 |

Pre-existing non-docstring violations to leave alone unless a module under
edit owns them: 16 `F821`, 7 `E501`, 6 `F401`, 5 `I001`, 1 `F841`.

The per-module job is therefore mostly deleting generated noise and writing a
real contract, not editing prose.

## Rubric

Applies to every module. Not a tick list.

### Before you write

1. Identify what the reader already knows and what they need to know after
   reading. The gap between those two states is the scope of the docstring
   (Losh: find state 1, find state 2, pick one idea that moves the reader
   closer, repeat)
2. Classify the content. A docstring is reference material. Narrative,
   motivation, and onboarding belong in a tutorial or explanation page (mixing
   the four types causes most documentation defects)
3. Confirm the function has a describable contract. If the behavior is hard to
   state, fix the design first (antirez: behavior that emerges at random from
   complexity cannot be documented, so return to the code)

### Summary line

4. Use imperative mood: "Return the user", not "Returns the user"
5. Make it self-contained. It renders in IDE tooltips, `help()` output, and API
   indexes without surrounding context
6. One line, terminated with a period, on the same line as the opening quotes
7. Describe the operation, not the object. "Parse a config file" over "A parser
   for config files"
8. Do not restate the signature. `get_user(id)` documented as "Get the user by
   id" adds no information

### Body content

9. Insert a blank line between summary and body
10. Specify the contract, not the implementation. State what callers can rely
    on (completeness and correctness come first; the docstring is a contract)
11. Define behavior at the boundaries: `None`, empty collections, zero,
    negative values, out-of-range input
12. Record the reason for non-obvious behavior: the constraint, the upstream
    requirement, the deliberate tradeoff
13. Explain domain concepts the reader may not have. If the function assumes
    trigonometry, a wire protocol, or a statistical model, state the assumption
    (antirez classifies these as teaching comments)
14. Declare units, coordinate system, timezone, encoding, and numeric
    precision. Type annotations do not carry these
15. State side effects: argument mutation, I/O, global state changes, thread
    safety, idempotency

### Parameters, returns, exceptions

16. Document every parameter with its meaning. The annotation supplies the
    type; the prose supplies the semantics
17. Document the return value including what an empty, zero, or `None` result
    signifies
18. List raised exceptions and the conditions that trigger each one
19. Do not duplicate information that already exists in the signature.
    Duplicated facts drift out of sync

### Examples

20. Derive examples from real call sites rather than placeholder identifiers
    (Evans, "Write good examples by starting with real code")
21. Make examples executable as written, with imports and setup included
22. Weigh the maintenance cost. Nothing runs these examples, so every one is a
    second artifact to keep current by hand

### Exclusions

23. Move algorithm descriptions and implementation notes to a comment placed
    below the docstring, outside the string the caller sees
24. Omit line-by-line restatement of the code (readable from the source itself)
25. Omit tutorial sequences, design philosophy, and single-case problem
    solutions (four purposes in one text serves none of them)
26. Omit claims with a short shelf life that no test will catch when they expire

### Style mechanics

27. Use one term per concept. Do not vary vocabulary for readability
28. Prefer plain language. Use jargon only when it carries meaning the plain
    word cannot
29. Keep sentences short. Read the docstring aloud to surface awkward
    constructions
30. Remove "simply", "just", "obviously", and "trivially"
31. Document private helpers. An undocumented `_parse_response` costs the next
    reader hours
32. Remove all em dashes and LLM language in favor of clear technical language.
    Do not convolute.

## One-time tasks

Repo-wide, done once. Not per module.

- [ ] Add documentation unit tests asserting that every public class and config
      option has a matching documented section. Mark existing gaps `xfail` to
      adopt this without a full backfill
- [ ] Verify a new user can locate a function without knowing its name in
      advance. If not, add a guide above the API reference in `docs/api/`
      (docstrings provide no structure beyond the namespace they sit in)
- [ ] Verify complete docstring coverage has not been mistaken for complete
      documentation (docstrings work well only once the reader already knows
      the project)
- [ ] Confirm `uv run ruff check factoriax` reports zero `D` violations

Standing practice, no box: keep docs in the same commit as the code and the
tests, flag stale docs in review, and treat docstrings as part of done.

## Modules

Ordered by import dependency, computed from the AST: a module appears only
after everything it imports from its own package. Within a dependency layer the
order is alphabetical. `__init__.py` comes last in each package because a
package docstring summarises modules that should already be documented.

### `factoriax/engine/`

- [ ] `factoriax/engine/constants.py`
- [ ] `factoriax/engine/actions.py`
- [ ] `factoriax/engine/belts.py`
- [ ] `factoriax/engine/machine_spec.py`
- [ ] `factoriax/engine/recipes.py`
- [ ] `factoriax/engine/tables.py`
- [ ] `factoriax/engine/recipes_io.py`
- [ ] `factoriax/engine/state.py`
- [ ] `factoriax/engine/achievements.py`
- [ ] `factoriax/engine/crafting.py`
- [ ] `factoriax/engine/jax_renderer.py`
- [ ] `factoriax/engine/levels.py`
- [ ] `factoriax/engine/machines.py`
- [ ] `factoriax/engine/placement.py`
- [ ] `factoriax/engine/rewards.py`
- [ ] `factoriax/engine/game_logic.py`
- [ ] `factoriax/engine/observations.py`
- [ ] `factoriax/engine/__init__.py`

#### `factoriax/engine/envs/`

- [ ] `factoriax/engine/envs/base.py`
- [ ] `factoriax/engine/envs/common.py`
- [ ] `factoriax/engine/envs/easy_rocket.py`
- [ ] `factoriax/engine/envs/mining.py`
- [ ] `factoriax/engine/envs/science_tiers.py`
- [ ] `factoriax/engine/envs/wrappers.py`
- [ ] `factoriax/engine/envs/miner_curriculum.py`
- [ ] `factoriax/engine/envs/rocket.py`
- [ ] `factoriax/engine/envs/registry.py`
- [ ] `factoriax/engine/envs/__init__.py`

### `factoriax/analysis/`

- [ ] `factoriax/analysis/actions.py`
- [ ] `factoriax/analysis/build_progression.py`
- [ ] `factoriax/analysis/categories.py`
- [ ] `factoriax/analysis/curriculum_strip.py`
- [ ] `factoriax/analysis/graph_layout.py`
- [ ] `factoriax/analysis/inventory.py`
- [ ] `factoriax/analysis/milestones.py`
- [ ] `factoriax/analysis/multiagent.py`
- [ ] `factoriax/analysis/recorder.py`
- [ ] `factoriax/analysis/state.py`
- [ ] `factoriax/analysis/trajectory.py`
- [ ] `factoriax/analysis/utils.py`
- [ ] `factoriax/analysis/eval.py`
- [ ] `factoriax/analysis/recipe_graph.py`
- [ ] `factoriax/analysis/video.py`
- [ ] `factoriax/analysis/__init__.py`

### `factoriax/assets/`

- [ ] `factoriax/assets/build_atlas.py`
- [ ] `factoriax/assets/__init__.py`

### `factoriax/playground/ui/`

- [ ] `factoriax/playground/ui/fonts.py`
- [ ] `factoriax/playground/ui/labels.py`
- [ ] `factoriax/playground/ui/scaling.py`
- [ ] `factoriax/playground/ui/theme.py`
- [ ] `factoriax/playground/ui/window.py`
- [ ] `factoriax/playground/ui/compositing.py`
- [ ] `factoriax/playground/ui/forms.py`
- [ ] `factoriax/playground/ui/icons.py`
- [ ] `factoriax/playground/ui/primitives.py`
- [ ] `factoriax/playground/ui/panels.py`
- [ ] `factoriax/playground/ui/__init__.py`

### `factoriax/playground/editor/`

- [ ] `factoriax/playground/editor/slot_display.py`
- [ ] `factoriax/playground/editor/state.py`
- [ ] `factoriax/playground/editor/canvas.py`
- [ ] `factoriax/playground/editor/dialogs.py`
- [ ] `factoriax/playground/editor/inventory_panel.py`
- [ ] `factoriax/playground/editor/toolbar.py`
- [ ] `factoriax/playground/editor/main.py`
- [ ] `factoriax/playground/editor/__main__.py`
- [ ] `factoriax/playground/editor/__init__.py`

### `factoriax/playground/menu/`

- [ ] `factoriax/playground/menu/controls_menu.py`
- [ ] `factoriax/playground/menu/main_menu.py`
- [ ] `factoriax/playground/menu/__init__.py`

### `factoriax/playground/play/`

- [ ] `factoriax/playground/play/achievements.py`
- [ ] `factoriax/playground/play/launch_screen.py`
- [ ] `factoriax/playground/play/play_state.py`
- [ ] `factoriax/playground/play/transfer.py`
- [ ] `factoriax/playground/play/ui.py`
- [ ] `factoriax/playground/play/game_ui.py`
- [ ] `factoriax/playground/play/main.py`
- [ ] `factoriax/playground/play/__main__.py`
- [ ] `factoriax/playground/play/__init__.py`

### `factoriax/playground/`

- [ ] `factoriax/playground/config.py`
- [ ] `factoriax/playground/app.py`
- [ ] `factoriax/playground/__main__.py`
- [ ] `factoriax/playground/__init__.py`

### `factoriax/` (root)

- [ ] `factoriax/make.py`
- [ ] `factoriax/__init__.py`
