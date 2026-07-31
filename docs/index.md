# FactoriaX

A grid world for reinforcement learning research, in the style of Factorio,
written in JAX.

The state is a set of JAX arrays. `step` and `reset` compile to one XLA graph,
so `jax.vmap` runs a batch of environments on the GPU, with no Python in the
inner loop.

```{toctree}
:maxdepth: 2
:hidden:

guides/getting_started
guides/ppo_training_example
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

- {doc}`guides/getting_started` — the environment, the observation, the action
  space, and a random rollout.
- {doc}`guides/ppo_training_example` — train a PPO agent on a FactoriaX task.
- {doc}`api/index` — the reference for every public module. Sphinx builds it
  from the docstrings.

## Run the playground

The playground is the human interface. It holds the game and the level editor.

```bash
uv run python -m factoriax.playground
```
