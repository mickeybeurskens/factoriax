# Spec: PPO Baseline for the Easy Rocket Bench

## Objective

Stand up a PPO training script targeted at the `easy_rocket` scenario
defined in `factoriax/scenarios/easy_rocket.py`. The easy rocket bench
is a deliberately small variant of the rocket task — a 16x16 procgen
map with six 2x2 ore patches, an 8-recipe book, 13 achievements (4 of
which are belt-network stubs that always read False today), and a
2000-tick episode budget. It exists so we can iterate on RL algorithms
in minutes per run instead of hours.

The PPO infrastructure under `baselines/ppo/` (network, GAE, loss,
normalization, CLI) and the rocket reference script at
`baselines/rocket/train_ppo.py` already exist. This spec adds a thin
training entry point at `baselines/easy_rocket/train_ppo.py` that
reuses that shared infrastructure and tunes only the bits that need
to change for the small-map, short-horizon shape of easy rocket. No
new PPO algorithmic surface is introduced.

The v1 scope is deliberately narrow: train on a **single fixed
layout** sampled from `build_easy_rocket_level(seed=0)` and
broadcast across all parallel envs, matching the rocket reference
script's reset pattern. Procgen-per-reset is a known follow-up
(parked in Open Questions) but adds engineering — easy rocket's
level builder uses Python-level rejection sampling that cannot run
inside the JIT'd rollout, so any procgen scheme needs a host-side
level pool or an alternate builder. Punting that to v2 lets v1 land
as a near-mechanical reuse of the rocket script.

### User-facing summary

- `python -m baselines.easy_rocket.train_ppo` runs a pilot in a few
  minutes on one GPU and prints achievement counts as it learns.
- Same CLI shape as `baselines.rocket.train_ppo` (`add_ppo_args`
  contract), with easy-rocket-specific defaults (smaller map, smaller
  network, shorter horizon, no action mask).
- Final artifacts (`final_model.msgpack`, `final_rollout.mp4`,
  per-achievement unlock plots) land in
  `runs/easy_rocket_ppo/{wandb_run_name or "default"}/`. wandb upload
  on `--use-wandb`.

### Success criteria

- `baselines/easy_rocket/train_ppo.py` exists and runs end-to-end on
  one consumer GPU within the configured budget, with no NaNs, no
  shape errors, and no JIT recompilations after the first iteration.
- A pilot run at the default config (defined below) finishes in
  under 10 minutes on a single GPU and reaches a mean episode
  achievement count of at least 6 out of the 9 reachable
  achievements over the last 100 episodes. (6/9 is the soft pass
  bar; the hard bar is "trains without diverging.")
- The training script reuses every shared building block in
  `baselines/ppo/` — `ActorCritic`, `PPOConfig`, `add_ppo_args`,
  `compute_gae`, `make_update_fn`/`update`, `RunningStats`, etc. No
  reimplementation of PPO math.
- All parallel envs train on the same fixed layout
  (`build_easy_rocket_level(jax.random.PRNGKey(config.seed))`),
  broadcast into device memory once at startup. Reset swaps each
  terminated env back to that same initial state, identical to
  `baselines/rocket/train_ppo.py`'s pattern.
- The four stub achievements are left unchanged in
  `EASY_ROCKET_ACHIEVEMENT_WEIGHTS`; reward is exactly
  `easy_rocket_reward` as published by the scenario. The training
  log makes the 9/13 ceiling visible so a reader doesn't mistake
  "stuck at 9" for a policy bug.
- A smoke test under `tests/baselines/easy_rocket/` exercises one
  training iteration in well under 60 seconds on CPU JIT, with
  `num_envs=4`, `rollout_steps=8`, `total_steps=32`, asserts no NaN
  in params or metrics, and asserts params change between iterations.

## Tech Stack

- Python 3.11+, JAX (GPU-accelerated, single device assumed), Flax
  for the actor-critic module, Optax for the optimizer.
- Existing `factoriax.make()` factory for the wrapped env stack
  (`factoriax/__init__.py:44`). `obs='global'` is the default for
  easy rocket since the 16x16 map flattens to a manageable input.
- Existing `baselines/ppo/*` shared infrastructure. No new files
  under `baselines/ppo/`.
- `wandb` for optional run logging; `imageio[ffmpeg]` for the final
  rollout video; `flax.serialization` for model checkpoints.
- Test framework: pytest. Lint/format: ruff. Type checking: mypy.
- Package management: `uv`.

## Commands

```
Train (pilot):     uv run python -m baselines.easy_rocket.train_ppo
Train (wandb):     uv run python -m baselines.easy_rocket.train_ppo --use-wandb
Test (smoke):      uv run pytest tests/baselines/easy_rocket -q
Lint:              uv run ruff check baselines/easy_rocket --fix
Format:            uv run ruff format baselines/easy_rocket
Typecheck:         uv run mypy baselines/easy_rocket
```

The training script's `--help` is the source of truth for every
hyperparameter; this spec lists only the easy-rocket-specific
defaults.

## Project Structure

```
baselines/easy_rocket/
  __init__.py              -> empty (package marker)
  train_ppo.py             -> main training script; defines Config,
                              train(), main(); CLI entry point.

tests/baselines/easy_rocket/
  __init__.py
  test_train_smoke.py      -> one-iteration smoke; CPU-JIT under 60s.

runs/easy_rocket_ppo/{run_name}/
  final_model.msgpack
  final_model.config.json
  final_rollout.mp4
  plots/*.png
```

No edits to existing files except:

- `baselines/__init__.py` — already present; no change.
- `baselines/ppo/*` — no change.
- `factoriax/scenarios/easy_rocket.py` — no change. The training
  script reads `easy_rocket_reward`, `easy_rocket_conditions`,
  `EASY_ROCKET_ACHIEVEMENT_WEIGHTS`, `NUM_EASY_ROCKET_ACHIEVEMENTS`,
  `MAX_EASY_ROCKET_SCORE`, `build_easy_rocket_level`,
  `EASY_ROCKET_RECIPE_TABLE` from this module verbatim.

## Code Style

Project conventions from `CLAUDE.md` apply: PEP 8, ruff-formatted
lines at 88 chars, type hints on every function, snake_case
functions, PascalCase classes, UPPER_CASE constants, no emoji.

Composition over inheritance — `Config` embeds a `PPOConfig`
instance rather than subclassing, matching `baselines/rocket`.

```python
@dataclasses.dataclass
class Config:
    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    max_timesteps: int = 2000
    anneal_lr: bool = True
    out_dir: str | None = None
    save_final_model: bool = True
    save_final_video: bool = True
    video_fps: int = 30
```

Easy-rocket-specific PPO defaults (set via
`parser.set_defaults(...)` so any explicit CLI flag still wins):

```
num_envs        = 256       # 16x16 map => obs vector small enough
rollout_steps   = 128
total_steps     = 2_000_000 # ~60 iters at the defaults above
hidden_dims     = (256, 256)
learning_rate   = 3.0e-4
gamma           = 0.99
gae_lambda      = 0.95
clip_eps        = 0.2
value_coef      = 0.5
entropy_coef    = 0.01
update_epochs   = 4
num_minibatches = 8
max_grad_norm   = 0.5
normalize_obs   = True
obs             = "global"  # window > map; flat global is cheaper
wandb_project   = "factoriax_easy_rocket"
log_interval    = 1
```

These mirror the rocket script's structure but with `num_envs`
halved (smaller obs), `total_steps` cut to 2M (short horizon, fast
credit assignment), and `obs="global"` instead of local-window-7.

## Reset Mechanics

V1 trains on a single fixed layout. At startup the script calls
`build_easy_rocket_level(jax.random.PRNGKey(config.ppo.seed))` once
on the host, materializes the corresponding `EnvState` via
`factoriax.levels.build_state`, and broadcasts it across the
`num_envs` axis into device memory:

```python
def _broadcast(x):
    a = jnp.asarray(x)
    return jnp.broadcast_to(a[None], (num_envs,) + a.shape)

fixed_states = jax.tree_util.tree_map(_broadcast, initial_state)
```

Reset inside the JIT'd rollout reuses the rocket script's pattern:
on a terminating env, swap the next state back to the fixed
initial state.

```python
def _where(reset_arr, cur_arr):
    mask = dones.reshape((-1,) + (1,) * (cur_arr.ndim - 1))
    return jnp.where(mask, reset_arr, cur_arr)

next_states = jax.tree_util.tree_map(_where, fixed_states, next_states)
```

This keeps `train_step` fully JIT'd and incurs no host roundtrip
on reset. The policy will memorize ore placement at this seed; that
is expected for v1. A procgen-per-reset follow-up is parked in
Open Questions.

## Reward and Action Space

- **Reward**: `easy_rocket_reward(prev_state, new_state, params)` —
  Craftax-style sparse +weight on each newly unlocked achievement,
  weights from `EASY_ROCKET_ACHIEVEMENT_WEIGHTS`. The four belt-
  network stub conditions read False, so 4 of 13 weights never
  contribute. The achievable ceiling per episode is therefore
  9 (out of `MAX_EASY_ROCKET_SCORE = 13.0`); the training log
  reports both the raw count and the 9-ceiling explicitly.
- **Action space**: 1D categorical over `NUM_ACTIONS`. Easy rocket
  sets `blocked_actions = frozenset()` — no masking. The training
  script honors this by passing `blocked_actions=()` to
  `factoriax.make()`, which short-circuits the
  `ActionMaskWrapper` entirely.
- **Observation**: `obs="global"`. With NUM_SPATIAL_CHANNELS=10
  and a 16x16 map, the spatial part is 2560 floats; plus
  `NUM_PLAYER_SCALARS` (afford bits + inventory counts + facing
  context), total obs dim is comfortably under 3000. Online
  observation normalization via `RunningStats` from
  `baselines.ppo.normalization`.
- **Episode horizon**: `max_timesteps=2000`, matching the
  scenario's published budget.

## Training Loop Shape

The script reuses the fused-step pattern from the rocket reference
(rollout + GAE + PPO update all inside a single `jax.jit`'d
function), with the level pool plumbed in as a closure variable so
the JIT cache sees a fixed pool shape:

```
train_step(params, opt_state, obs_stats, env_states, obs, rng):
  for _ in rollout_steps:                # jax.lax.scan
    sample actions, step vmap'd env, compute reward
    on done: where(dones, fixed_states, next_states)
  bootstrap last value
  compute_gae(traj.reward, traj.value, traj.done, last_value, ...)
  for _ in update_epochs:                # jax.lax.scan
    for each minibatch: PPO loss + Adam step
```

The Python-side loop only does logging, deque accounting, and the
post-training artifact rendering — same shape as the rocket script.

## Testing Strategy

- **Smoke test** under `tests/baselines/easy_rocket/test_train_smoke.py`:
  - Build a tiny `Config` (`num_envs=4`, `rollout_steps=8`,
    `total_steps=32`, `hidden_dims=(32,)`, `num_minibatches=2`,
    `update_epochs=1`, `use_wandb=False`, `save_final_model=False`,
    `save_final_video=False`).
  - Run `train(config)` and assert (a) no NaNs in returned metrics,
    (b) returned `mean_ep_return >= 0`, (c) training completes
    inside the test's timeout. Marked `@pytest.mark.slow` if it
    exceeds a few seconds on CI's CPU runner.
- **Performance**: This spec is not a perf change; no A/B benchmark
  is required ([[feedback_test_perf_game_logic]] is scoped to game-
  logic edits). The training script's own steps-per-second is
  logged every `log_interval` iterations and visible in the run
  output for tracking.

## Boundaries

- **Always:**
  - Reuse `baselines/ppo/*` building blocks; do not re-implement
    PPO math in the training script.
  - Keep the rollout fully JIT'd — `train_step` must compile once
    and never recompile within a run. Reset stays in device memory
    via `where(dones, fixed_states, next_states)`; no host
    callback per terminating env.
  - Run pytest, ruff, mypy before each commit.
  - Log both the raw achievement count and the 9-achievement
    reachable ceiling, so the reader can see how close the policy
    is to the wall the stubs impose.
  - State changes only via JAX state-management functions, not
    from play/UI code ([[feedback_state_management]]).
  - Commit each perf-impacting change separately with a benchmark
    between them ([[feedback_incremental_perf]]); for non-perf
    work, normal commits.
  - Final eval render uses
    `factoriax.analysis.video.compose_frame_with_inventory` and
    `write_video`, identical to the rocket script's path.
  - Save artifacts under `runs/easy_rocket_ppo/{run_name}/`;
    `--out-dir` overrides.
- **Ask first:**
  - Reward shaping beyond `easy_rocket_reward` — adding dense
    intermediate signals, scaling stub weights to zero, or any
    change to `EASY_ROCKET_ACHIEVEMENT_WEIGHTS`.
  - Changes to the easy_rocket scenario itself
    (`factoriax/scenarios/easy_rocket.py`). The PPO script is a
    consumer; if the bench needs to change, that's a separate
    spec.
  - Switching from MLP `ActorCritic` to `VisionActorCritic` or
    introducing a recurrent policy.
  - Adding action masking. The scenario explicitly sets
    `blocked_actions = frozenset()` — overriding that in the
    training script silently would invalidate score comparisons.
  - Introducing any reset randomization (procgen pool, fresh
    layout per reset, multi-seed broadcast). v1 is single-seed by
    design; see Open Question 1.
- **Never:**
  - Hardcode or print W&B API keys or any other secret.
  - Run the training script while another GPU job is in progress
    ([[feedback_wait_before_running]]).
  - Use `pgrep`/`kill -0` to poll long-running training; rely on
    the script's own logging and the file timestamps
    ([[feedback_no_kill_polling.md]]).
  - Add a fallback path that builds a level on the host inside the
    JIT'd train step ([[feedback_no_ad_hoc_fallbacks]]).
  - Commit debug prints, commented-out tests, or any artifact
    under `runs/`.

## Open Questions

1. **Procgen-per-reset (v2).** v1 trains on a single fixed seed
   and the policy is allowed to memorize ore placement. The
   natural v2 is a pre-built pool of K layouts (call
   `build_easy_rocket_level` K times on host, stack into a
   `(K, ...)` PyTree, gather by random index on reset) which
   keeps the rollout JIT'd at the cost of K times one initial
   state in VRAM. Decide on this — and on K — after v1's pilot
   numbers land; if the policy hits the 9/9 ceiling quickly, the
   bench needs procgen to stay informative.
2. **Should `_finalize_artifacts` be extracted into
   `baselines/ppo/artifacts.py`?** Right now the rocket script
   owns ~150 lines of artifact code (model save + video render +
   plots + W&B upload). Easy rocket needs the same. The clean move
   is to extract it once both call sites exist; out of scope for
   the initial landing but worth raising before duplicating.
3. **Periodic eval vs end-of-training eval only.** The rocket
   script only evals at the end. For a fast pilot bench, a 1-episode
   eval every N iterations would give a less noisy learning curve
   in W&B than the per-rollout running-mean. Costs ~2k env steps
   per eval. Park unless the noise actually obscures comparisons.
4. **Single-GPU only?** The rocket script targets one device. If
   we ever want multi-device, every `vmap` over `num_envs` would
   become a `pmap`/sharded compute pattern. Park; easy rocket is
   small enough that one GPU is more than enough.
5. **Whether to log per-achievement unlock rates during training
   (not just at the end).** The rocket script does the per-
   achievement breakdown only in its final summary. Doing it per-
   log-interval is cheap (a sum over the running peak mask) and
   helps spot which achievement the policy is plateauing on. Lean
   toward yes for easy rocket since the 9-achievement ceiling
   makes it easy to see exactly which one is stuck.
