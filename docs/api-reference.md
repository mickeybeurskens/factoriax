# API Reference

Auto-generated from `factoriax.__all__`. Do not edit by hand — run `uv run python scripts/generate_api_reference.py` to regenerate.

Stability tiers come from `factoriax/_stability.py`. **Stable** symbols carry the usual backward-compat guarantees; **Experimental** may change between minor versions; **Internal** is exported only for tooling.

| Tier | Name | Kind | Signature | Summary |
| --- | --- | --- | --- | --- |
| Stable | `Action` | class |  | Player actions using compound action design. |
| Stable | `ActionMaskWrapper` | class |  | Replace blocked actions with :data:`Action.NOOP`. |
| Stable | `AutoResetWrapper` | class |  | Gymnax wrapper providing cached-state auto-reset. |
| Stable | `BlockType` | class |  | Block types in the environment grid. |
| Stable | `Direction` | class |  | Compass facing directions for players and machines. |
| Stable | `EnvParams` | class |  | Environment parameters. |
| Stable | `EnvState` | class |  | Immutable environment state. |
| Stable | `FactoriaXEnv` | class |  | FactoriaX JAX-based grid environment. |
| Stable | `ItemType` | class |  | Item types that can be stored in inventory. |
| Stable | `LEVELS` | constant |  |  |
| Stable | `Level` | class |  | Serializable description of an initial world state. |
| Stable | `LevelBuilder` | class |  | Fluent builder for constructing :class:`Level` objects programmatically. |
| Stable | `NUM_ITEM_TYPES` | constant |  |  |
| Stable | `build_state` | function | `(level: 'Level', params: 'EnvParams') -> 'EnvState'` | Construct a JAX :class:`~factoriax.engine.state.EnvState` from a :class:`Level`. |
| Stable | `generate_state` | function | `(rng: 'jax.Array', params: 'EnvParams') -> 'EnvState'` | Generate a procedural world state from a random key. |
| Stable | `get_level` | function | `(name: 'str') -> 'Level'` | Look up a built-in level by name. |
| Stable | `load_level` | function | `(path: 'Path') -> 'Level'` | Deserialize a :class:`Level` from a JSON file written by :func:`save_level`. |
| Stable | `make` | function | `(env_id: str, *, obs: str \| None = None, obs_radius: int \| None = None, auto_reset: bool = False, resample: bool \| None = None) -> tuple[typing.Any, factoriax.engine.state.EnvParams]` | Build a registered scenario env. |
| Stable | `save_level` | function | `(level: 'Level', path: 'Path') -> 'None'` | Serialize a :class:`Level` to a JSON file using orjson. |
| Experimental | `OBSERVATIONS` | constant |  |  |
| Experimental | `ScienceTallyWrapper` | class |  | Accumulate per-step science pack consumption into a running total. |
| Experimental | `global_superficial` | function | `(state: 'EnvState', params: 'EnvParams', player_idx: 'int \| jax.Array') -> 'jax.Array'` | Full-map superficial observation for one player. |
| Experimental | `global_x_ray` | function | `(state: 'EnvState', params: 'EnvParams', player_idx: 'int \| jax.Array') -> 'jax.Array'` | Full-map x_ray observation for one player. |
| Experimental | `local_superficial` | function | `(state: 'EnvState', params: 'EnvParams', player_idx: 'int \| jax.Array', radius: 'int' = 10) -> 'jax.Array'` | Local windowed superficial observation centered on one player. |
| Experimental | `local_x_ray` | function | `(state: 'EnvState', params: 'EnvParams', player_idx: 'int \| jax.Array', radius: 'int' = 10) -> 'jax.Array'` | Local windowed x_ray observation centered on one player. |
| Experimental | `mining_reward` | function | `(prev_state: factoriax.engine.state.EnvState, new_state: factoriax.engine.state.EnvState, params: factoriax.engine.state.EnvParams) -> jax.jaxlib._jax.Array` | Dense reward combining proximity to ore and a bonus for each ore mined. |
| Experimental | `rgb` | function | `(state: 'EnvState', block_pixel_size: 'int' = 32) -> 'np.ndarray'` | Render the full map as an RGB image. |
