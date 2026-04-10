"""BenchmarkRunner: executes policies against a Benchmark.

The runner owns all simulation concerns — environment stepping, observation
extraction, PRNG management, trajectory collection. A ``Benchmark`` tells
the runner what levels to run and how to score results; the runner tells
nobody anything, it just executes.

This separation means benchmarks can be defined, tested, and reasoned about
without any knowledge of JAX internals, JIT compilation, or PRNG splits.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from factoriax.benchmarks.core import (
    Benchmark,
    BenchmarkLevel,
    BenchmarkResult,
    LevelResult,
    Policy,
)
from factoriax.constants import ItemType
from factoriax.envs import FactoriaXEnv
from factoriax.levels import build_state
from factoriax.observations import global_array
from factoriax.state import EnvParams, EnvState

logger = logging.getLogger(__name__)

# Type for a JIT-compiled step function.
_StepFn = Callable[
    [jax.Array, EnvState, int | jax.Array, EnvParams],
    tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[str, jax.Array]],
]


def _split3(
    keys: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Split a batch of PRNG keys into three batches.

    Args:
        keys: Array of shape ``(N, 2)``.

    Returns:
        Three arrays of shape ``(N, 2)``.
    """
    keys = jax.vmap(lambda k: jax.random.split(k, 3))(keys)
    return keys[:, 0], keys[:, 1], keys[:, 2]


class BenchmarkRunner:
    """Executes a list of policies against a ``Benchmark`` and returns results.

    The runner JIT-compiles ``env.step_env`` at construction time and reuses
    it across all levels. JAX will retrace for levels with different map
    shapes, which is expected — the tracing cost is amortised over 200 steps.

    The PRNG key is seeded from ``self.seed`` at the start of each ``run()``
    call, making results reproducible for the same seed and policy.

    For multi-agent benchmarks: each tick, all players act in sequence and
    each call to ``step_env`` increments ``state.timestep``. Set
    ``max_timesteps`` in ``EnvParams`` accordingly for multi-agent levels.

    Attributes:
        seed: Random seed used to initialise the PRNG key for each run.
    """

    def __init__(self, seed: int = 0) -> None:
        """Initialise the runner.

        Args:
            seed: Random seed for reproducible episode steps. The key is
                re-seeded from this value at the start of every ``run()``
                call, not shared across calls.
        """
        self._default_env = FactoriaXEnv()
        self._default_jit_step = jax.jit(self._default_env.step_env)
        self._env_cache: dict[int, tuple[FactoriaXEnv, _StepFn]] = {}
        self.seed = seed

    def _get_env_and_step(
        self,
        bench_level: BenchmarkLevel,
    ) -> tuple[FactoriaXEnv, _StepFn]:
        """Return the env and JIT-compiled step for a level.

        Uses the default env when no custom achievement function is set.
        Caches envs by achievement function identity to avoid re-JITing.

        Args:
            bench_level: Level that may carry a custom achievement_fn.

        Returns:
            ``(env, jit_step)`` pair.
        """
        if bench_level.achievement_fn is None:
            return self._default_env, self._default_jit_step
        fn_id = id(bench_level.achievement_fn)
        if fn_id not in self._env_cache:
            env = FactoriaXEnv(achievement_fn=bench_level.achievement_fn)
            self._env_cache[fn_id] = (env, jax.jit(env.step_env))
        return self._env_cache[fn_id]

    def run(
        self,
        benchmark: Benchmark,
        policies: list[Policy],
        obs_fn: Callable[[EnvState, EnvParams, int], jax.Array] | None = None,
        constraint_fn: Callable[[EnvState, EnvState, EnvParams], jax.Array]
        | None = None,
    ) -> BenchmarkResult:
        """Run policies through all benchmark levels and return aggregated results.

        Args:
            benchmark: Benchmark to evaluate against.
            policies: One policy per player, indexed by player index. Each
                policy must accept a JAX float32 observation array and return
                a JAX integer action scalar.
            obs_fn: Observation extraction function with signature
                ``(state, env_params, player_idx) -> obs_array``. Defaults to
                ``global_array``. Pass a custom function to match the
                observation space used during training (e.g. ``local_array``
                with a fixed radius for policies trained with local obs).
            constraint_fn: Optional constraint cost function with signature
                ``(prev_state, new_state, params) -> jax.Array`` returning
                a cost vector of shape ``(K,)``.  When provided, the per-step
                costs are recorded in ``LevelResult.constraint_costs``.

        Returns:
            ``BenchmarkResult`` containing per-level results and the
            aggregate score computed by ``benchmark.score``.

        Raises:
            ValueError: If ``len(policies)`` does not equal
                ``benchmark.num_players``.
        """
        _obs_fn = obs_fn if obs_fn is not None else global_array
        if len(policies) != benchmark.num_players:
            noun = "policy" if benchmark.num_players == 1 else "policies"
            raise ValueError(
                f"Benchmark '{benchmark.name}' requires {benchmark.num_players} "
                f"{noun}, got {len(policies)}."
            )

        rng = jax.random.PRNGKey(self.seed)
        level_results: list[LevelResult] = []

        for bench_level in benchmark.levels():
            rng, subkey = jax.random.split(rng)
            result = self._run_level(
                benchmark,
                bench_level,
                policies,
                subkey,
                _obs_fn,
                constraint_fn,
            )
            level_results.append(result)
            logger.info(
                "Level '%s': score=%.1f  mined=%s  steps=%d",
                bench_level.name,
                result.weighted_score,
                result.items_mined,
                result.timesteps_used,
            )

        aggregate = benchmark.score(level_results)
        logger.info(
            "Benchmark '%s' complete: aggregate_score=%.3f",
            benchmark.name,
            aggregate,
        )
        return BenchmarkResult(
            benchmark_name=benchmark.name,
            level_results=level_results,
            aggregate_score=aggregate,
        )

    def run_batched(
        self,
        benchmark: Benchmark,
        policy_fn: Callable[[jax.Array, jax.Array], jax.Array],
        seeds: list[int],
        obs_fn: Callable[[EnvState, EnvParams, int], jax.Array] | None = None,
    ) -> list[BenchmarkResult]:
        """Run a policy across multiple seeds in parallel using vmap.

        Much faster than calling :meth:`run` in a loop because all seeds
        share a single ``jax.lax.scan`` per level, keeping computation
        on-device.

        Only supports single-player benchmarks. The policy must be a
        pure function compatible with JAX tracing.

        Args:
            benchmark: Benchmark to evaluate against.
            policy_fn: Pure policy function with signature
                ``(obs, rng_key) -> action``. The runner manages PRNG
                splitting. ``obs`` is a float32 array and ``action`` is
                an integer scalar. The function must be JIT-traceable.
            seeds: List of integer seeds, one per parallel evaluation.
            obs_fn: Observation extraction function. Defaults to
                ``global_array``.

        Returns:
            List of ``BenchmarkResult``, one per seed, in the same
            order as ``seeds``.

        Raises:
            ValueError: If the benchmark requires more than one player.
        """
        if benchmark.num_players != 1:
            raise ValueError(
                "run_batched only supports single-player benchmarks, "
                f"got num_players={benchmark.num_players}."
            )
        _obs_fn = obs_fn if obs_fn is not None else global_array
        num_seeds = len(seeds)
        levels = benchmark.levels()

        # Per-level batched results: level_name -> (final_states, actions, timesteps)
        level_data: list[
            tuple[BenchmarkLevel, EnvState, np.ndarray, np.ndarray]
        ] = []

        for bench_level in levels:
            params = bench_level.env_params
            state0 = build_state(bench_level.level, params)
            max_steps = params.max_timesteps

            # Broadcast initial state to (num_seeds, ...).
            states = jax.tree.map(
                lambda x: jnp.broadcast_to(
                    jnp.asarray(x)[None], (num_seeds,) + jnp.asarray(x).shape,
                ),
                state0,
            )

            rngs = jax.vmap(jax.random.PRNGKey)(jnp.array(seeds))

            level_env, _ = self._get_env_and_step(bench_level)
            vmap_step = jax.vmap(
                level_env.step_env, in_axes=(0, 0, 0, None),
            )

            def _obs_single(s: EnvState) -> jax.Array:
                return _obs_fn(s, params, 0)

            vmap_obs = jax.vmap(_obs_single)
            vmap_policy = jax.vmap(policy_fn, in_axes=(0, 0))

            @jax.jit
            def _scan(
                states: EnvState, rngs: jax.Array,
            ) -> tuple[EnvState, jax.Array, jax.Array]:
                def step(
                    carry: tuple[EnvState, jax.Array, jax.Array, jax.Array],
                    _: None,
                ) -> tuple[
                    tuple[EnvState, jax.Array, jax.Array, jax.Array],
                    jax.Array,
                ]:
                    st, rn, dones, t_used = carry
                    obs = vmap_obs(st)
                    rn, act_keys, step_keys = _split3(rn)
                    actions = vmap_policy(obs, act_keys)

                    _obs, next_st, _rew, step_done, _info = vmap_step(
                        step_keys, st, actions, params,
                    )

                    # Freeze states that are already done.
                    next_st = jax.tree.map(
                        lambda o, n: jnp.where(
                            dones.reshape((-1,) + (1,) * (n.ndim - 1)),
                            o, n,
                        ),
                        st, next_st,
                    )
                    new_dones = dones | step_done
                    t_used = t_used + (~dones).astype(jnp.int32)
                    recorded = jnp.where(dones, 0, actions)
                    return (next_st, rn, new_dones, t_used), recorded

                init_dones = jnp.zeros(num_seeds, dtype=bool)
                init_t = jnp.zeros(num_seeds, dtype=jnp.int32)
                (final_st, _, _, t_used), all_actions = jax.lax.scan(
                    step, (states, rngs, init_dones, init_t),
                    None, length=max_steps,
                )
                # all_actions: (T, N)
                return final_st, all_actions, t_used

            final_states, all_actions, timesteps_used = _scan(states, rngs)

            # Transfer to CPU once.
            all_actions_np = np.asarray(all_actions)  # (T, N)
            timesteps_np = np.asarray(timesteps_used)  # (N,)
            level_data.append(
                (bench_level, final_states, all_actions_np, timesteps_np)
            )

        # Assemble per-seed BenchmarkResults.
        results: list[BenchmarkResult] = []
        for i in range(num_seeds):
            level_results: list[LevelResult] = []
            for bench_level, final_states, all_actions_np, timesteps_np in level_data:
                t_used = int(timesteps_np[i])
                fs_i = jax.tree.map(lambda x: x[i], final_states)
                items_mined = {
                    "coal": int(fs_i.items_mined[ItemType.COAL]),
                    "iron": int(fs_i.items_mined[ItemType.IRON_ORE]),
                    "copper": int(fs_i.items_mined[ItemType.COPPER_ORE]),
                }
                score = benchmark.score_level(bench_level, items_mined)
                level_results.append(LevelResult(
                    level_name=bench_level.name,
                    items_mined=items_mined,
                    weighted_score=score,
                    timesteps_used=t_used,
                    actions=all_actions_np[:t_used, i],
                    final_state=fs_i,
                ))
            agg = benchmark.score(level_results)
            results.append(BenchmarkResult(
                benchmark_name=benchmark.name,
                level_results=level_results,
                aggregate_score=agg,
            ))

        agg_scores = [r.aggregate_score for r in results]
        logger.info(
            "Benchmark '%s' batched (%d seeds): mean=%.3f  std=%.3f",
            benchmark.name, num_seeds,
            float(np.mean(agg_scores)), float(np.std(agg_scores)),
        )
        return results

    def _run_level(
        self,
        benchmark: Benchmark,
        bench_level: BenchmarkLevel,
        policies: list[Policy],
        rng: jax.Array,
        obs_fn: Callable[[EnvState, EnvParams, int], jax.Array],
        constraint_fn: Callable[[EnvState, EnvState, EnvParams], jax.Array]
        | None = None,
    ) -> LevelResult:
        """Execute one level and return the result.

        The level state is built deterministically from ``bench_level.level``
        and ``bench_level.env_params`` via ``build_state`` — no random key
        is needed for the reset. The PRNG key is used only for step_env.

        When *constraint_fn* is provided, it is evaluated once per tick
        (after all players have acted) and the per-step cost vectors are
        stored in ``LevelResult.constraint_costs``.

        Args:
            benchmark: Benchmark owning this level (provides per-level scoring).
            bench_level: Level to run.
            policies: Policies indexed by player index.
            rng: PRNG key for this level's episode steps.
            obs_fn: Observation extraction function.
            constraint_fn: Optional constraint cost function.

        Returns:
            ``LevelResult`` for this level.
        """
        params = bench_level.env_params
        state = build_state(bench_level.level, params)
        num_players = params.num_players
        _, jit_step = self._get_env_and_step(bench_level)

        actions_log: list[int] = []
        costs_log: list[np.ndarray] = []
        done = jnp.array(False)

        for _ in range(params.max_timesteps):
            prev_state = state
            for p in range(num_players):
                state_p = state.replace(selected_player=p)
                obs = obs_fn(state_p, params, p)
                action = policies[p](obs)

                rng, subkey = jax.random.split(rng)
                _, state, _, done, _ = jit_step(
                    subkey,
                    state_p,
                    action,
                    params,
                )

                if p == 0:
                    actions_log.append(int(action))

            if constraint_fn is not None:
                cost = constraint_fn(prev_state, state, params)
                costs_log.append(np.asarray(cost))

            if bool(done):
                break

        items_mined: dict[str, int] = {
            "coal": int(state.items_mined[ItemType.COAL]),
            "iron": int(state.items_mined[ItemType.IRON_ORE]),
            "copper": int(state.items_mined[ItemType.COPPER_ORE]),
        }
        weighted_score = benchmark.score_level(bench_level, items_mined)
        constraint_costs = np.stack(costs_log) if costs_log else None

        return LevelResult(
            level_name=bench_level.name,
            items_mined=items_mined,
            weighted_score=weighted_score,
            timesteps_used=len(actions_log),
            actions=np.array(actions_log, dtype=np.int32),
            constraint_costs=constraint_costs,
            final_state=state,
        )
