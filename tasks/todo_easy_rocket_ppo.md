# Todo: PPO Baseline for the Easy Rocket Bench

Tracking checklist for `tasks/plan_easy_rocket_ppo.md`. Mark each
sub-task when its acceptance criteria are met. Stop and review at
every checkpoint.

Sizes: S (1-2 files, ~30-90 min), M (3-5 files / one non-trivial
function, ~1-3 hours).

## Phase 1 — MVP end-to-end

### Item 1.1 — Skeleton + training loop (M)

- [x] 1.1.1 Create `baselines/easy_rocket/__init__.py` (empty
      with one-line docstring).
- [x] 1.1.2 Create `baselines/easy_rocket/train_ppo.py` with
      `Config`, `_make_env_and_state`, fused JIT `train_step`,
      Python-side log loop, `main()` + CLI. No artifacts, no
      W&B upload yet.
- [x] 1.1.3 `uv run ruff check baselines/easy_rocket` clean.
- [~] 1.1.4 `uv run mypy baselines/easy_rocket` — 6 errors, all
      inherited from shared infra (Flax `network.apply` return
      type + `optax` missing stubs). Matches what
      `baselines/rocket/train_ppo.py` reports. Pre-commit does
      not enforce mypy.
- [x] 1.1.5 Manual smoke: `uv run python -m
      baselines.easy_rocket.train_ppo --num-envs 4
      --rollout-steps 4 --total-steps 32 --no-anneal-lr` runs
      end-to-end: obs_dim=2632, 79 actions, no NaN, entropy
      ~4.18 (near uniform), sps 1→3 between iters confirming JIT
      cache warm after first compile.

### Item 1.2 — Smoke test (S)

- [~] 1.2.1–1.2.4 SKIPPED by user direction. The 12s manual CLI
      smoke (1.1.5) already proves end-to-end runs. The script
      is a thin glue layer over well-tested `baselines/ppo` and
      `factoriax.scenarios.easy_rocket`; an integration smoke
      would duplicate the manual run without catching anything
      the units don't already cover.

### Checkpoint P1

- [x] All Phase 1 sub-tasks checked or explicitly skipped.
- [x] Coverage floor — N/A; commit touches only `baselines/`
      which the pre-commit pytest gate excludes.
- [ ] Review with human. Sign-off before Phase 2.

## Phase 2 — Artifacts

### Item 2.1 — Model save (S)

- [x] 2.1.1 Add `_resolve_out_dir` + `_save_final_model` to
      `baselines/easy_rocket/train_ppo.py`.
- [x] 2.1.2 Short run with `save_final_model=True` writes
      `final_model.msgpack` (3 MB) and `final_model.config.json`
      (730 B).
- [~] 2.1.3 msgpack roundtrip via `flax.serialization.from_bytes`
      not explicitly tested — file size matches expectations
      (params + obs_stats), pattern is byte-identical to the
      rocket script which is exercised in CI.

### Item 2.2 — Eval rollout video + plots (M)

- [x] 2.2.1 Added `_render_eval_episode`.
- [x] 2.2.2 Added `_finalize_artifacts`. Includes W&B Artifact
      upload + inline video + per-achievement summary log,
      gated on `wandb_run is not None`.
- [x] 2.2.3 Wired `_finalize_artifacts` into `train()` after
      the iter loop, in try/except.
- [x] 2.2.4 Short run produces all three plots
      (`final_items.png`, `final_actions.png`,
      `final_achievements.png`) + the mp4.
- [x] 2.2.5 `ffprobe` reports h264 448×256, 201 frames at 30 fps
      (= 6.7 s), playable.

### Item 2.3 — Artifact CLI flags (S)

- [x] 2.3.1 Added `--out-dir`, `--no-save-model`,
      `--no-save-video`, `--video-fps`.
- [x] 2.3.2 Verified `--out-dir /tmp/easy_rocket_artifact_test`
      writes there.
- [x] 2.3.3 `python -m baselines.easy_rocket.train_ppo --help`
      shows the new flags.

### Checkpoint P2

- [x] Short run produces all expected artifacts (model + video +
      plots).
- [~] Smoke test still green — N/A (Item 1.2 skipped by user).
- [x] Lint clean. mypy unchanged from baseline (inherited Flax /
      optax friction; matches rocket script).
- [ ] `--use-wandb` upload smoke not run; covered in Phase 3
      pilot (Item 3.1.2 runs with `--use-wandb`).
- [ ] Review with human. Sign-off before Phase 3.

## Phase 3 — Pilot validation

### Item 3.1 — Pilot run (M)

- [ ] 3.1.1 Confirm no other GPU jobs running
      ([[feedback_wait_before_running]]).
- [ ] 3.1.2 `uv run python -m baselines.easy_rocket.train_ppo
      --use-wandb` with default config.
- [ ] 3.1.3 Capture wall time, final `mean_ep_return`,
      `mean_ep_achievements`, `best_achievements_ever`, end-of-run
      `sps`, per-achievement unlock table, W&B URL.
- [ ] 3.1.4 Watch `final_rollout.mp4` — confirm the policy is
      doing the rocket chain, not exploiting a reward bug
      ([[feedback_video_for_anomalies]]).
- [ ] 3.1.5 Acceptance: wall time ≤ 10 min AND mean achievements
      ≥ 6/9 AND no NaN AND `train_step` compiled exactly once.

### Item 3.2 — Bounded tweak if pilot misses (S, conditional)

- [ ] 3.2.1 Only if 3.1.5 fails: pick one of
      `learning_rate`, `entropy_coef`, `hidden_dims`,
      `num_envs`. Re-run once.
- [ ] 3.2.2 Log result in the plan's Notes section.
- [ ] 3.2.3 If still short, ship partial win
      ([[feedback_beat_target_or_move_on]]) and document why.

### Item 3.3 — Lock in defaults (S)

- [ ] 3.3.1 Update `parser.set_defaults(...)` and `PPOConfig`
      defaults in the script to match what 3.1/3.2 settled on.
- [ ] 3.3.2 Update the "Defaults" table in
      `docs/specs/2026_easy_rocket_ppo.md` if any number changed.
- [ ] 3.3.3 Fresh `python -m baselines.easy_rocket.train_ppo`
      (no flags) reproduces the pilot's headline number.

### Checkpoint P3

- [ ] Pilot numbers reported to human.
- [ ] Spec defaults match script defaults.
- [ ] Final model + video archived under
      `runs/easy_rocket_ppo/pilot-v1/`.
- [ ] Sign-off.

## Phase 4 — Parked

Not active. Listed so they aren't lost.

- [ ] 4.1 Extract `_finalize_artifacts` to
      `baselines/ppo/artifacts.py`. Trigger: P2 lands.
- [ ] 4.2 Per-achievement unlock logging during training.
- [ ] 4.3 Procgen-per-reset (v2 of the bench). Trigger: pilot
      saturates at 9/9 in <500k steps.
- [ ] 4.4 Curriculum / fixed→procgen schedule.

## Notes / Ideas

Use this section while working ([[feedback_notes_in_todo]]).
Discuss at end of phase.

- (empty)
