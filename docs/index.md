# Factoriax

A Factorio-inspired grid-world RL environment, written in JAX.
State lives in JAX arrays; `step` and `reset` compile to a single
XLA graph, so `jax.vmap` over a batch of envs runs on GPU without
Python in the inner loop.

```{toctree}
:maxdepth: 2
:hidden:

guides/getting_started
guides/ppo_training_example
api/index
```

## Where to go next

- {doc}`guides/getting_started` — Start here for a more in depth overview of Factoriax and its functionalities.
- {doc}`guides/ppo_training_example` — Train a PPO algorithm on a Factoriax navigation task.
- {doc}`api/index` — auto-generated reference for every public module.
