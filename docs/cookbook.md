# FactoriaX Cookbook

Short recipes for the moves a research project actually makes.
Each recipe is one paragraph plus a pointer to a runnable file
under [`examples/`](../examples) that you can copy verbatim and
adapt. Read [`docs/getting-started.md`](getting-started.md) first
if you haven't.

## Compose the canonical wrapper stack

`factoriax.make()` is the only factory you need for the standard
shapes. Each option is opt-in: omit it and the corresponding
wrapper isn't included. Composition order, outermost-to-innermost,
is `AutoReset → ActionMask → LocalObservation → FactoriaXEnv`.

→ [`examples/wrapped_env.py`](../examples/wrapped_env.py)

## Train (or evaluate) a random policy

The simplest possible "training loop" — uniform-random actions, a
JIT'd step, mean reward over a budget. Useful as a smoke test
that the env, observation, and reward plumbing are wired up before
you plug in a real learner.

→ [`examples/train_random_policy.py`](../examples/train_random_policy.py)

## Author a level in code and save it as JSON

`LevelBuilder` is fluent: `.fill_rect(...)`, `.place_machine(...)`,
`.set_player_position(...)`, `.build(name)`. `save_level(level, path)`
writes JSON; `load_level(path)` reads it back. The Editor produces
the same JSON shape, so hand-built and editor-built levels are
interchangeable.

→ [`examples/custom_level.py`](../examples/custom_level.py)

## Write a custom reward function

Reward functions in FactoriaX are pure: they take
`(prev_state, new_state, params)` and return a `jax.Array` scalar.
That makes them composable inside `jax.lax.scan` rollouts and
trivial to swap. Use `state.items_mined`, `state.player_inventory`,
`state.achievements_unlocked`, etc. to score whatever your
research question rewards.

→ [`examples/custom_reward.py`](../examples/custom_reward.py)

## Evaluate across many seeds in parallel

JAX's `vmap` over a batch of PRNG keys is the canonical batched-
rollout shape. Inside each rollout, `jax.lax.scan` keeps the step
loop on-device. The example reports mean and standard deviation of
cumulative reward across 16 parallel envs.

→ [`examples/batched_evaluation.py`](../examples/batched_evaluation.py)

## Track achievements as engine state

`FactoriaXEnv(achievement_fn=...)` binds a condition function
into the env constructor. Every step the engine evaluates it and
OR-folds the result into `state.achievements_unlocked` (a bool
array of shape `(MAX_ACHIEVEMENTS,)`). Once a bit flips on, it
stays on for the rest of the episode. Read it from anywhere —
observations, rewards, analysis, benchmarks — without a wrapper.

→ Pattern: see `factoriax.benchmarks.rocket.rocket_conditions`
for a worked function across 38 achievements;
[`baselines/rocket/train_ppo.py`](../baselines/rocket/train_ppo.py)
is the end-to-end run.

## Use a built-in level

`factoriax.LEVELS` is a dict of name → `Level`. Pass the name to
`factoriax.make("name")` and the env binds to that level — params
are derived from the level's `map_width` × `map_height` so
`reset_env` Just Works. `factoriax.get_level("name")` returns the
`Level` object directly if you need to inspect it.

→ Snippet in `docs/getting-started.md`; full call in
[`examples/wrapped_env.py`](../examples/wrapped_env.py).

## Mask actions

`factoriax.make(blocked_actions=(int(Action.MINE),))` wraps the env
in `ActionMaskWrapper`. Listed actions are silently rewritten to
`NOOP` before the inner env sees them, so a policy that emits a
blocked action just no-ops that step. The rocket benchmark uses
this to forbid hand-crafting and force production through machines.

→ [`baselines/rocket/train_ppo.py`](../baselines/rocket/train_ppo.py)
sets `blocked_actions=ROCKET_BLOCKED_ACTIONS` on construction.

## Switch to a local observation window

For policies that don't need the full map, `obs="local"` swaps in
`LocalObservationWrapper`. Window side length is `2 * radius + 1`;
`radius=7` (the default) gives a 15×15 window plus per-player
scalars. Fixed-shape obs keeps your network's first FC layer
constant as the map grows.

→ [`examples/wrapped_env.py`](../examples/wrapped_env.py) shows
construction; the full PPO setup is in
[`baselines/rocket/train_ppo.py`](../baselines/rocket/train_ppo.py).

## Render an RGB frame

`factoriax.rgb(state, block_pixel_size=8)` returns a uint8 RGB
array drawn through the same JAX renderer the play view uses.
Useful for vision-mode observations, debugging frames, or video
writers.

→ The renderer reads its sprites from
[`factoriax/assets/atlas.png`](../factoriax/assets/atlas.png);
the layout is documented in
[`factoriax/assets/atlas.layout.md`](../factoriax/assets/atlas.layout.md).
Swapping in new art is a drop-in PNG replacement.

## Auto-reset for `jax.lax.scan` training loops

`factoriax.make(auto_reset=True)` wraps the env in
`AutoResetWrapper`, which caches the initial state at reset time
and restores it whenever `done=True` — without a Python-side
branch, so the full step still compiles to one XLA graph. Required
for PureJaxRL-style PPO, where every iteration is a single scan.

→ [`examples/wrapped_env.py`](../examples/wrapped_env.py).
