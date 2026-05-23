# Implementation Plan: PPO Baseline for the Easy Rocket Bench

This plan implements `docs/specs/2026_easy_rocket_ppo.md`. The spec
defines *what* and *why*; this document defines *how* — the order of
work, the file-level scope of each step, the verification gate
between steps, and the risks. Sub-tasks are tracked in
`tasks/todo_easy_rocket_ppo.md`.

## Overview

The spec is intentionally narrow: ship a PPO training script for the
easy rocket scenario that reuses every shared building block in
`baselines/ppo/` and the reset pattern from
`baselines/rocket/train_ppo.py`. v1 trains on a single fixed layout;
procgen-per-reset is parked as v2.

Three phases, all vertical slices — each one ends with the codebase
in a working, demonstrable state:

- **Phase 1 — MVP end-to-end.** The script runs and a smoke test
  asserts one iteration trains without NaNs. No artifacts, no
  wandb, no plots. The point is to land the rollout/GAE/update
  loop wired correctly to easy rocket's reward and env factory.
- **Phase 2 — Artifacts.** Add the model save, eval-rollout video,
  per-achievement plots, and W&B upload — near-copy from the
  rocket script. After this phase a short run produces a watchable
  mp4 and a loadable msgpack under `runs/easy_rocket_ppo/`.
- **Phase 3 — Pilot validation.** Run the default config on one
  GPU, verify the spec's success criterion (≥6/9 mean achievements
  in ≤10 min), and lock in defaults. Up to one bounded tweak if
  the first run misses, per [[feedback_beat_target_or_move_on]];
  otherwise ship the partial win and move on.

Each phase has a single checkpoint that requires human review
before the next phase starts.

## Architecture (decisions locked in `docs/specs/2026_easy_rocket_ppo.md`)

- **Reuse, do not re-implement.** `ActorCritic`, `PPOConfig`,
  `add_ppo_args`, `compute_gae`, `RunningStats`, `normalize_obs`,
  `update_running_stats` all come from `baselines/ppo`. The new
  script defines `Config`, the env factory, the fused
  `train_step`, the Python-side loop, and (in Phase 2) artifact
  helpers.
- **Reset is single-fixed-state for v1.** Call
  `build_easy_rocket_level(PRNGKey(seed))` once on host, build the
  state, broadcast across `num_envs`, swap back on dones inside
  the JIT'd scan. No host callback per terminating env.
- **Observation is global.** 16x16 map; flat global obs vector
  is small enough that the local-window wrapper isn't worth it.
  `obs="global"` in `factoriax.make(...)`.
- **No action mask.** Easy rocket scenario sets
  `blocked_actions = frozenset()`. The script honors that by
  passing `blocked_actions=()`.
- **Reward is `easy_rocket_reward` verbatim.** 4 of 13
  achievements are belt-network stubs that always read False; the
  reachable ceiling per episode is 9. The training log surfaces
  both numbers so a plateau at 9 reads as expected, not a bug.

## Dependency Graph

```
                   ┌────────────────────────────────────┐
                   │ baselines/ppo/* (already exists)   │
                   │   ActorCritic, PPOConfig,          │
                   │   compute_gae, RunningStats,       │
                   │   add_ppo_args, normalize_obs      │
                   └────────────────┬───────────────────┘
                                    │ import
                                    ▼
   ┌─────────────────────────────────────────────────────────┐
   │ Phase 1: baselines/easy_rocket/__init__.py              │
   │          baselines/easy_rocket/train_ppo.py             │
   │          tests/baselines/easy_rocket/test_train_smoke.py│
   └────────────────────┬────────────────────────────────────┘
                        │ depends on
                        ▼
   ┌─────────────────────────────────────────────────────────┐
   │ Phase 2: artifact helpers in train_ppo.py               │
   │          (_resolve_out_dir, _save_final_model,          │
   │           _render_eval_episode, _finalize_artifacts)    │
   │          CLI flags: --out-dir, --no-save-{model,video}, │
   │                     --video-fps, --use-wandb already in │
   │                     add_ppo_args                        │
   └────────────────────┬────────────────────────────────────┘
                        │ depends on
                        ▼
   ┌─────────────────────────────────────────────────────────┐
   │ Phase 3: pilot run + defaults validation                │
   │          (no new code unless tweaks needed)             │
   └─────────────────────────────────────────────────────────┘
```

The script also reads from `factoriax/`:

- `factoriax.make(...)` — env factory.
- `factoriax.levels.build_state` — fixed-state materialization.
- `factoriax.scenarios.easy_rocket.{build_easy_rocket_level,
  easy_rocket_reward, easy_rocket_conditions,
  EASY_ROCKET_ACHIEVEMENT_WEIGHTS, NUM_EASY_ROCKET_ACHIEVEMENTS,
  MAX_EASY_ROCKET_SCORE, EASY_ROCKET_RECIPE_TABLE}`.
- `factoriax.constants.{NUM_ACTIONS, Action, MAX_ACHIEVEMENTS}`.
- Phase 2 only: `factoriax.analysis.{eval.EvalRollout,
  eval.generate_eval_plots, video.compose_frame_with_inventory,
  video.write_video}`.

No file outside `baselines/easy_rocket/` and
`tests/baselines/easy_rocket/` is edited in any phase.

## Phase 1 — MVP end-to-end

### Item 1.1 — Skeleton + training loop (M)

**Scope:** Two files, ~250 lines total.

- `baselines/easy_rocket/__init__.py` — empty package marker with
  a one-line docstring.
- `baselines/easy_rocket/train_ppo.py`:
  - `Config` dataclass: embeds `PPOConfig`, plus `max_timesteps`,
    `anneal_lr`, `out_dir`, `save_final_model`, `save_final_video`,
    `video_fps` (the latter four are placeholders Phase 2 uses;
    declare them now so the Phase 2 diff is small).
  - `_make_env_and_state(config) -> (env, state0, env_params)`:
    builds the fixed level via
    `build_easy_rocket_level(PRNGKey(seed))`, calls
    `factoriax.make(level, obs="global",
    achievement_fn=easy_rocket_conditions)`, replaces
    `env_params.max_timesteps` with `config.max_timesteps`, builds
    the initial state via `build_state(level, env_params)`.
  - `train(config) -> dict[str, float]`: same shape as the rocket
    script — broadcast initial state, vmap env step + obs, fused
    JIT `train_step` (rollout → GAE → PPO update), Python-side
    log loop. Uses `rocket_reward`'s easy_rocket sibling
    `easy_rocket_reward` directly. Logs `iter`, `step`, `sps`,
    `mean_ret`, `mean_ach/9`, `best_ach`, `entropy`, top-3
    actions, same format as the rocket script's log line. **No
    wandb logging or final-artifact rendering in this phase.**
  - `main()`: `add_ppo_args(parser)` + `parser.set_defaults(
    num_envs=256, total_steps=2_000_000, log_interval=1,
    wandb_project="factoriax_easy_rocket")` + a `--max-timesteps`
    and `--no-anneal-lr` flag. Build `Config`, call `train()`.

**Acceptance:**

- File-level: both files exist, ruff clean, mypy clean.
- Runtime: `uv run python -m baselines.easy_rocket.train_ppo \
  --num-envs 4 --rollout-steps 4 --total-steps 16` runs end-to-end
  without error or NaN and prints at least one log line.
- Throughput: after the first iteration's JIT compile, subsequent
  iterations do not recompile (single compile cache entry visible
  via JAX's logging or simply via consistent wall-time per iter).

**Verification:**

- `uv run ruff check baselines/easy_rocket`
- `uv run mypy baselines/easy_rocket`
- Manual smoke run above.

### Item 1.2 — Smoke test (S)

**Scope:** Two files.

- `tests/baselines/easy_rocket/__init__.py` — empty.
- `tests/baselines/easy_rocket/test_train_smoke.py`:
  - Build a tiny `Config(num_envs=4, rollout_steps=8,
    total_steps=32, hidden_dims=(32,), num_minibatches=2,
    update_epochs=1, normalize_obs=True, use_wandb=False,
    save_final_model=False, save_final_video=False)`.
  - Call `train(config)`.
  - Assert returned dict contains keys `mean_ep_return`,
    `mean_ep_achievements`, `best_achievements_ever`,
    `max_possible_score`, `sps`.
  - Assert no NaN in any returned float.
  - Assert `max_possible_score == MAX_EASY_ROCKET_SCORE`.
  - Mark `@pytest.mark.slow` if wall time exceeds a few seconds
    on CPU JIT.

**Acceptance:**

- `uv run pytest tests/baselines/easy_rocket -q` passes locally.
- Total runtime under 60 seconds on CPU; if it exceeds that, the
  slow marker keeps it out of the default `pytest -m "not slow"`
  loop.

**Verification:**

- `uv run pytest tests/baselines/easy_rocket -q`
- `uv run pytest -q -m "not slow"` — full default suite still
  green (no regression in existing tests).

### Checkpoint P1

- All of: ruff + mypy + pytest on the new files green.
- Manual run on the developer's box (CPU is fine; tiny config).
- Coverage floor (37%) not regressed — the new script and test
  add covered lines, so this should be neutral-or-better.
- Review with human before starting Phase 2.

## Phase 2 — Artifacts

### Item 2.1 — Model save (S)

**Scope:** Edit `baselines/easy_rocket/train_ppo.py`.

- Add `_resolve_out_dir(config) -> Path` — default
  `runs/easy_rocket_ppo/{wandb_run_name or "default"}/`.
- Add `_save_final_model(path, params, obs_stats, config)` —
  msgpack of params + obs_stats, sibling `.config.json` of the
  dataclass. Identical contract to the rocket script.

**Acceptance:**

- Calling `train()` with `save_final_model=True` writes
  `final_model.msgpack` and `final_model.config.json` under
  `_resolve_out_dir()`.
- File sizes are nonzero; msgpack roundtrips through
  `flax.serialization.from_bytes` to an equivalent PyTree
  structure.

**Verification:**

- Short run + manual check that the files exist.
- Add a one-line assertion in the smoke test (when
  `save_final_model=True` for a single sub-test) that the
  expected files exist.

### Item 2.2 — Eval rollout video + plots (M)

**Scope:** Edit `baselines/easy_rocket/train_ppo.py`.

- Add `_render_eval_episode(config, env, env_params,
  initial_state, network, params, obs_stats) -> EvalRollout`:
  Python-side per-step loop, JIT'd `env.step_env` and
  `network.apply`, deterministic-ish via PRNG, collects frames
  through `compose_frame_with_inventory(state)`, logs actions
  and per-step achievement masks. Stops at `done`. Same shape as
  the rocket reference.
- Add `_finalize_artifacts(...)` — orchestrates model save +
  eval rollout + `write_video` to mp4 + `generate_eval_plots`,
  with W&B Artifact + inline upload gated on
  `wandb_run is not None`.
- Wire `_finalize_artifacts` into `train()` after the iteration
  loop, in a try/except that logs (but does not raise) so a
  failed artifact pass never erases a successful training run.

**Acceptance:**

- Short run (`--num-envs 16 --total-steps 5000
  --rollout-steps 64`) produces under
  `runs/easy_rocket_ppo/default/`:
  - `final_model.msgpack` (>1 KB)
  - `final_model.config.json` (valid JSON, matches `Config`)
  - `final_rollout.mp4` (>1 KB, opens in `ffprobe` cleanly)
  - At least one `plots/*.png` (>1 KB)
- Artifact generation runs strictly after the training loop —
  the iteration log still ends cleanly even when artifacts fail
  (manually verified by temporarily breaking `_save_final_model`
  and confirming the run reports done before the traceback).

**Verification:**

- Short run + `ls -lh runs/easy_rocket_ppo/default/`.
- `ffprobe runs/easy_rocket_ppo/default/final_rollout.mp4`
  returns a valid video stream.

### Item 2.3 — Artifact CLI flags (S)

**Scope:** Edit `baselines/easy_rocket/train_ppo.py`.

- Add `--out-dir`, `--no-save-model`, `--no-save-video`,
  `--video-fps` to `main()`. Same names and semantics as the
  rocket script.
- Wire them into `Config` construction.

**Acceptance:**

- `--no-save-video --no-save-model` skips both artifacts but
  still runs training.
- `--out-dir /tmp/easy_rocket_test` writes there.
- `python -m baselines.easy_rocket.train_ppo --help` shows the
  new flags.

**Verification:**

- Manual run with each flag combination.

### Checkpoint P2

- Short run produces all expected artifacts.
- Smoke test still green.
- Lint + mypy clean.
- W&B path verified (one short `--use-wandb` run hits the
  factoriax_easy_rocket project, model + video + plots appear as
  artifacts and inline images on the run page).
- Review with human before starting Phase 3.

## Phase 3 — Pilot validation

### Item 3.1 — Pilot run (M)

**Scope:** No code edits unless 3.2 fires. Execution only.

- `uv run python -m baselines.easy_rocket.train_ppo --use-wandb`
  with the spec's default config (num_envs=256,
  total_steps=2_000_000, max_timesteps=2000, log_interval=1).
- One GPU, no other GPU jobs running
  ([[feedback_wait_before_running]]).
- Capture: wall time, final `mean_ep_return`,
  `mean_ep_achievements`, `best_achievements_ever`, end-of-run
  `sps`, the per-achievement unlock table from the training log,
  and the W&B run URL (or the local `runs/` directory).

**Acceptance:**

- Wall time ≤ 10 minutes.
- Last 100 episodes mean_ep_achievements ≥ 6.0 (out of 9
  reachable).
- No NaN in the loss curves at any iteration.
- `train_step` compiles exactly once (visible in W&B sps curve —
  a sharp early dip then flat).

**Verification:**

- Read the W&B run page, copy the numbers.
- Watch `final_rollout.mp4` to confirm the policy is actually
  doing the rocket chain, not exploiting a reward bug
  ([[feedback_video_for_anomalies]]).

### Item 3.2 — Bounded tweak if pilot misses (S)

**Scope:** At most one of: `learning_rate`,
`entropy_coef`, `hidden_dims`, `num_envs`. Only if 3.1's
acceptance bar is missed.

- Per [[feedback_beat_target_or_move_on]]: try one focused
  change, re-run, log result. If still short, **ship the partial
  win** and document why in this plan's notes section.

**Acceptance:**

- If the tweak hits ≥6/9, lock it in via Item 3.3.
- If not, write a short "Notes" entry below explaining what was
  tried, the resulting numbers, and why the partial win is
  acceptable.

### Item 3.3 — Lock in defaults (S)

**Scope:** Possibly edit `baselines/easy_rocket/train_ppo.py`
defaults and `docs/specs/2026_easy_rocket_ppo.md`'s "Default"
table to match what 3.1/3.2 settled on.

**Acceptance:**

- The spec's listed defaults equal the `parser.set_defaults` and
  `PPOConfig` defaults in the script.
- A fresh `python -m baselines.easy_rocket.train_ppo` (no flags)
  reproduces the pilot's headline number within a sane run-to-run
  variance.

### Checkpoint P3

- Pilot numbers reported to human.
- Spec defaults match script defaults.
- Final model + video archived under
  `runs/easy_rocket_ppo/pilot-v1/` (or similar) for future
  comparison.
- Sign-off.

## Phase 4 — Parked

Captured here so they don't get lost. Each becomes its own
sub-spec when a trigger fires.

- **4.1 Extract `_finalize_artifacts` to
  `baselines/ppo/artifacts.py`.** Trigger: Phase 2 lands and the
  duplication with `baselines/rocket/train_ppo.py` is now real
  (~150 lines). One-shot extraction; both call sites switch to
  the shared module.
- **4.2 Per-achievement unlock logging during training.** Cheap
  (sum over the running peak mask), gives the W&B run page a
  per-achievement learning curve instead of one aggregate
  number. Useful for spotting which achievement plateaus.
- **4.3 Procgen-per-reset (v2 of the bench).** Trigger: pilot
  saturates at 9/9 quickly enough that the bench stops being
  informative. v2 adds a pre-built pool of K layouts gathered by
  random index on reset (see Open Question 1 in the spec).
- **4.4 Curriculum / fixed→procgen schedule.** Only if 4.3 lands
  and procgen-from-scratch is too hard to credit-assign.

## Risks

- **A — JIT recompilation.** `train_step` compiles once if every
  closed-over shape is constant for the run. The
  `set_defaults`/CLI path means `num_envs`, `rollout_steps`,
  `num_minibatches` are fixed at script entry; nothing inside
  the rollout changes shape. Mitigation: smoke test asserts
  consistent per-iter wall time after iter 1.
- **B — NaN explosion early.** Easy rocket gives zero reward for
  many ticks before the first achievement; `RunningStats` can
  drift before the first non-trivial obs lands. Mitigation:
  obs is clipped to ±10 inside `normalize_obs`. Watchpoint, not
  a known issue.
- **C — 9/9 ceiling hit too fast.** If the policy memorizes the
  fixed layout in <500k steps, the bench is too easy. That's
  the trigger for 4.3 (procgen v2), not a bug.
- **D — VRAM at num_envs=256.** GPU has ~6 GB
  ([[gpu_memory_ceiling]]); the rocket script at num_envs=512
  is close to the wall. Easy rocket's obs is smaller, but if
  OOM, drop num_envs to 128 before any algorithmic change.
- **E — Pre-commit coverage floor.** Floor is 37%. The new
  script + smoke test add coverage; should be neutral-or-better.
  If the smoke test runs `@pytest.mark.slow` it won't lift the
  floor — keep it in the default suite if it stays under a few
  seconds.

## Notes

(Empty. Populated as needed during Phase 3 to record any
partial-win documentation per [[feedback_beat_target_or_move_on]].)
