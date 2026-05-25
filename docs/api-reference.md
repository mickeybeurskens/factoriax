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
| Stable | `LocalObservationWrapper` | class |  | Replace the inner env's full-map obs with a local radius-R window. |
| Stable | `NUM_ITEM_TYPES` | constant |  |  |
| Stable | `build_state` | function | `(level: 'Level', params: 'EnvParams') -> 'EnvState'` | Construct a JAX :class:`~factoriax.state.EnvState` from a :class:`Level`. |
| Stable | `generate_state` | function | `(rng: 'jax.Array', params: 'EnvParams') -> 'EnvState'` | Generate a procedural world state from a random key. |
| Stable | `get_level` | function | `(name: 'str') -> 'Level'` | Look up a built-in level by name. |
| Stable | `load_level` | function | `(path: 'Path') -> 'Level'` | Deserialize a :class:`Level` from a JSON file written by :func:`save_level`. |
| Stable | `make` | function | `(level: factoriax.levels.Level \| str \| None = None, *, obs: Literal['global', 'local'] = 'global', obs_radius: int = 7, achievement_fn: collections.abc.Callable[[factoriax.state.EnvState], jax.jaxlib._jax.Array] \| None = None, auto_reset: bool = False, blocked_actions: collections.abc.Iterable[int] = ()) -> tuple[typing.Any, factoriax.state.EnvParams]` | Build a FactoriaX environment with the canonical wrapper stack. |
| Stable | `save_level` | function | `(level: 'Level', path: 'Path') -> 'None'` | Serialize a :class:`Level` to a JSON file using orjson. |
| Experimental | `ScienceTallyWrapper` | class |  | Accumulate per-step science pack consumption into a running total. |
| Experimental | `global_array` | function | `(state: 'EnvState', params: 'EnvParams', player_idx: 'int \| jax.Array') -> 'jax.Array'` | Full-map flat observation for one player. |
| Experimental | `local_array` | function | `(state: 'EnvState', params: 'EnvParams', player_idx: 'int \| jax.Array', radius: 'int' = 10) -> 'jax.Array'` | Local windowed observation centered on one player. |
| Experimental | `mining_reward` | function | `(prev_state: factoriax.state.EnvState, new_state: factoriax.state.EnvState, params: factoriax.state.EnvParams) -> jax.jaxlib._jax.Array` | Dense reward combining proximity to ore and a bonus for each ore mined. |
| Experimental | `rgb` | function | `(state: 'EnvState', block_pixel_size: 'int' = 32) -> 'np.ndarray'` | Render the full map as an RGB image. |
