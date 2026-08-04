# Factoriax

A grid world for reinforcement learning research, in the style of Factorio,
written in JAX.

The state is a set of JAX arrays. `step` and `reset` compile to one XLA graph,
so `jax.vmap` runs a batch of environments on the GPU, with no Python in the
inner loop.

```{toctree}
:maxdepth: 1
:hidden:
:caption: Home

Home <self>
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Start here

start-here/index
start-here/getting_started
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Training models using Factoriax

training/index
training/train_ppo_mining
training/record_and_replay_a_rollout
training/build_a_custom_scenario
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Understanding Factoriax

understanding/index
understanding/state_and_step
understanding/observation_space
understanding/action_design
understanding/scenarios_and_curriculum
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: API documentation

api/index
```

## Build an environment

```python
import jax

from factoriax.make import env_from_name

env, params = env_from_name("EasyRocket-v1")
obs, state = env.reset_env(jax.random.PRNGKey(0), params)
```

Five scenarios carry an id: `Mining-v1`, `MinerBootstrap-v1`,
`ScienceTiers-v1`, `EasyRocket-v1`, and `Rocket-v1`.

## Where to go next

Pick a section by what you want to do, not by what you want to read.

- {doc}`start-here/index`: new to Factoriax. Follow one path to a working
  rollout.
- {doc}`training/index`: you know Factoriax and have a specific training
  task, such as running PPO or recording a rollout.
- {doc}`understanding/index`: you want the reasoning behind a design choice,
  such as the observation profiles or the action space.
- {doc}`api/index`: you know what you want and need the exact signature.
  Sphinx builds this section from the docstrings.

## Run the playground

The playground is the human interface. It holds the game and the level editor.

```bash
uv run python -m factoriax.playground
```
