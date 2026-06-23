# FactoriaX

A Factorio-inspired grid-world RL environment, written in JAX.
State lives in JAX arrays; `step` and `reset` compile to a single
XLA graph, so `jax.vmap` over a batch of envs runs on GPU without
Python in the inner loop.

```{toctree}
:maxdepth: 2
:hidden:

quickstart
guides/action-design
api/index
```

## Where to go next

- {doc}`quickstart` — install and open the launcher.
- {doc}`guides/action-design` — the three properties the action set
  is built around.
- {doc}`api/index` — auto-generated reference for every public module.
