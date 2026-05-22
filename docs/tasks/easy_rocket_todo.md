# Todo: Easy Rocket Scenario

Tracking checklist for `docs/tasks/easy_rocket_plan.md`. Mark each
sub-task when its acceptance criteria are met. Stop and review at the
pre-merge checkpoint.

Sizes: S (1-2 files, ~30-90 min), M (3-5 files, ~1-3 hours).

Slices 1, 2, 3 are independent and can run in any order or in
parallel. Slices 4, 5, 6 are sequential.

## Slice 1 — Recipe book

- [x] 1.1 Define `EASY_ROCKET_RECIPE_BOOK` with exactly the eight
      user-owned recipes: `MINER`, `ASSEMBLER`, `CONVEYOR_BELT`,
      `SPLITTER`, `CROSSING`, `HULL`, `ENGINE_UNIT` copied from
      `BASE_RECIPES`, plus a rewired
      `Recipe(output=ROCKET, inputs=((HULL, 1), (ENGINE_UNIT, 1)),
      ticks=50)`. Project to `EASY_ROCKET_RECIPE_TABLE`. Smelts and
      intermediates are intentionally absent — the user fills them
      in post-merge. Tests: book constructs with exactly 8 outputs;
      ROCKET inputs are (HULL, ENGINE_UNIT); AVIONICS and
      ROCKET_CORE absent everywhere. (S)

## Slice 2 — Level builder

- [x] 2.1 Implement `build_easy_rocket_level(key)`: 16x16 map, spawn
      at (8, 8), six 2x2 ore patches (iron, copper, tin, silicon,
      coal, limestone) placed by uniform sampling with rejection,
      `forbid_radius=1` ring around spawn. Tests: dimensions,
      determinism, key variation, invariants (all ores present, no
      overlaps). (M)

## Slice 3 — Shared condition helpers — DROPPED

Skipped in favour of inlining the two helpers easy-rocket actually needs
(`_holds_item`, `_count_machines`) directly in `easy_rocket.py`. The other
two rocket-scenario helpers (`_any_entity_buf_nonempty`,
`_any_assembler_has_output`) belong to the graph-gated achievements
that ship as always-False stubs in v1, so they have no use here.
Block-under-miner logic lives inline in the condition function.

## Slice 4 — Condition function

- [x] 4.1 Implement `easy_rocket_conditions(state)`: 13 achievements,
      v1 ones wired (indices 0, 1, 2, 3, 4, 5, 6, 8, 12), graph-gated
      ones stubbed to always-False (indices 7, 9, 10, 11). Tests:
      positive + negative fixture per v1 achievement; one test asserts
      graph-gated stubs always read False even under fixtures designed
      to look like they should unlock. (M)

## Slice 5 — Reward and scoring

- [x] 5.1 Add `EASY_ROCKET_ACHIEVEMENT_WEIGHTS` (all 1.0),
      `MAX_EASY_ROCKET_SCORE = 13.0`, and `easy_rocket_reward` (thin
      `achievement_reward` wrapper). Tests: single-unlock reward
      equals weight, no-unlock reward is zero. (S)

## Slice 6 — Scenario class, exports, observation shape

- [x] 6.1 Implement `EasyRocketScenario(seed=0)` with `levels()`,
      `score_level`, `score`, and class attrs (`name`, `num_players`,
      `achievement_fn`, `blocked_actions`). Tests: construction,
      protocol conformance, `env_params` fields, seed varies layout,
      score returns `MAX_EASY_ROCKET_SCORE` for all-True mask. (M)
- [x] 6.2 Add observation-shape test: `local_array(state, params, 0,
      radius=5)` on a hand-built 16x16 state returns the documented
      shape and dtype. No env stepping. (S)
- [x] 6.3 Export new public symbols from
      `factoriax/scenarios/__init__.py` (alphabetised). Test: each
      symbol imports at the top level. (S)

## Pre-merge checkpoint

- [ ] `uv run ruff check factoriax/scenarios/ tests/scenarios/` green.
- [ ] `uv run ruff format --check factoriax/scenarios/ tests/scenarios/`
      green.
- [ ] `uv run mypy factoriax/scenarios/easy_rocket.py
      factoriax/scenarios/_common.py factoriax/scenarios/rocket.py`
      green.
- [ ] `uv run pytest tests/scenarios/test_easy_rocket.py
      tests/scenarios/test_common.py
      tests/scenarios/test_rocket_scenario.py -q` green.
- [ ] `uv run pytest tests/scenarios/test_easy_rocket.py -q
      --durations=10` total wall-clock under 1 second.
- [ ] Manual smoke: `uv run python -m factoriax` reaches the
      easy-rocket scenario from the menu and steps a few times
      without crashing.
- [ ] **Review with human** before merge.
