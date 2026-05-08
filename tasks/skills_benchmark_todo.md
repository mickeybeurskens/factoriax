# Todo: Skills Challenge Benchmark

Tracking checklist for `tasks/skills_benchmark_plan.md`. Mark each
sub-task when its acceptance criteria are met. Stop and review at every
checkpoint.

Sizes: S (1-2 files, ~30-90 min), M (3-5 files, ~1-3 hours).

## Phase F — Foundation

- [ ] **F.1** Extend `BenchmarkLevel` with optional `blocked_actions`; runner honours per-level mask. (S)
- [ ] **F.2** Skills package skeleton: `__init__.py`, `achievements.py`, `levels.py`, `reward.py`. (M)
- [ ] **F.3** Per-level action mask constants — eight `frozenset[int]` derived from `Action` enum. (S)
- [ ] **Checkpoint F:** rocket tests still green, skills skeleton imports cleanly, mypy + ruff clean, review with user.

## Phase L — Curriculum (vertical slices, in order)

Each slice: achievement bit + condition, level builder, scripted policy,
tests. Bit index = curriculum index − 1. Achievement bits 0..7.

- [ ] **L.1** `navigate` — fixed-corner goal, varying spawn (M)
- [ ] **L.2** `mine` — ore-rich 5x5, scripted BFS-to-mine (M)
- [ ] **L.3** `craft_miner` — pre-placed materials, mask only `CRAFT_MINER` (M)
- [ ] **L.4** `place_miner` — 5 miners in inventory, ore patch off-centre (M)
- [ ] **Checkpoint L1** (after L.4): scripted aggregate ≥ 0.9 on first 4 levels, random ≤ 0.1, JIT recompile cost reasonable, review with user.
- [ ] **L.5** `fuel_and_collect` — pre-placed miner, MINE blocked, deposit-then-withdraw (M)
- [ ] **L.6** `belt_line` — 12-wide corridor, item must reach right-edge belt (M)
- [ ] **L.7** `arm_transfer` — pre-placed miner+chest, agent places arm (M)
- [ ] **L.8** `mini_factory` — full chain on 16x16, scripted ≤ 200 LOC (M-L)
- [ ] **Checkpoint L2** (after L.8): full curriculum scripted ≥ 0.9 aggregate, random ≤ 0.1, no leftover imports of old skills code, review with user.

## Phase T — Training & integration

Interactive phase. **Always ask the user before kicking off any training
run.** Every run logs to wandb project `factoriax_skills_benchmark` with
mandatory tags: `skills`, `<phase>`, `<level_name>`, `<git_sha_short>`,
`<role>`. See plan for full conventions.

- [ ] **T.1** Rewrite `baselines/skills/train_ppo.py` driven by `SkillsBenchmark` with mandatory wandb logging. (M)
- [ ] **T.1.a** Smoke run: train `navigate` only — ask user first; share wandb URL; wait for sign-off.
- [ ] **T.1.b** Smoke run: train `mining` — same gate.
- [ ] **T.1.c** Full curriculum sweep `--all` (only after user approves T.1.a/T.1.b).
- [ ] **T.2** Random-policy floor — pytest test (`@pytest.mark.slow`) + `scripts/run_skills_random_floor.py` CLI logging to wandb tagged `floor`/`random`. (S)
- [ ] **T.3** Throughput A/B harness `scripts/bench_skills_runner.py` — logs `baseline`/`candidate` wandb runs grouped by SHA tag. (S)
- [ ] **Checkpoint T:** user has reviewed PPO runs on wandb (at least navigate + mining), full curriculum sweep complete, random floor green, A/B baseline captured. Final user sign-off before docs phase.

## Phase D — Documentation & cutover

- [ ] **D.1** Delete `factoriax/benchmarks/skills/{mining,place_miner}.py` and any leftover old-shape imports. (S)
- [ ] **D.2** `docs/skills_benchmark.md` — user-facing benchmark doc, structure + commands only (no baseline results / scores / training curves — deferred to later). Link from `docs/getting-started.md`. (S)
- [ ] **D.3** `docs/cookbook.md` — "Build your own skills challenge" walkthrough using navigate as worked example. (M)
- [ ] **D.4** Public API + stability manifest update; regenerate `docs/api-reference.md`. (S)
- [ ] **Checkpoint D — Final:** all SPEC success criteria checked off, full pytest green incl. `-m slow`, no orphan imports, final user review.

## Parallelization notes

- L.1 must land first (creates scaffolding). L.2–L.8 can be drafted in
  parallel but should land in curriculum order to keep achievement bit
  indexing sequential.
- T.1 / T.2 / T.3 are independent — any order after L.8.
- D.2 and D.3 can be drafted in parallel.

## Decisions locked (from spec review)

- 8-level curriculum, T1–T4. No research, no repair.
- One `SkillsBenchmark` mirroring `RocketBenchmark` shape.
- Per-level action mask via optional `BenchmarkLevel.blocked_actions`.
- Layout-varying seed, default `0` canonical for evaluation.
- Time-weighted score: `solved * (max - used + 1) / max`.
- Hand-crafting masked on levels 5–8; pre-placed machinery rocket-style.
- Cutover at end (Phase D), not coexistence.
- Tiny scripted policies, one file per level; mini_factory hard-capped at 200 LOC.
