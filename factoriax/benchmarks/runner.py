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
from typing import Any

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
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.levels import build_state
from factoriax.observations import global_array
from factoriax.state import EnvParams, EnvState

# An achievement function maps an EnvState to a bool array of shape
# (MAX_ACHIEVEMENTS,). Benchmarks that want achievement tracking expose
# one via an ``achievement_fn`` attribute, or pass one to the runner.
AchievementFn = Callable[[EnvState], jax.Array]

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

    def __init__(
        self,
        seed: int = 0,
        achievement_fn: AchievementFn | None = None,
    ) -> None:
        """Initialise the runner.

        Args:
            seed: Random seed for reproducible episode steps. The key is
                re-seeded from this value at the start of every ``run()``
                call, not shared across calls.
            achievement_fn: Optional achievement condition function with
                signature ``(EnvState) -> bool[MAX_ACHIEVEMENTS]``. When
                supplied, the runner constructs ``FactoriaXEnv`` with
                this function and surfaces the latched unlock mask
                (read from ``state.achievements_unlocked``) on each
                :class:`LevelResult`. Benchmarks can also advertise
                their own function via an ``achievement_fn`` attribute;
                the explicit argument here takes precedence when both
                are set.
        """
        self._achievement_fn: AchievementFn | None = achievement_fn
        # Multi-entry cache keyed on (id(achievement_fn), blocked_actions).
        # A single-entry cache thrashes when consecutive levels use
        # different masks (each switch costs one full XLA compile of
        # step_env). The dict keeps every config compiled at most once
        # across the runner's lifetime.
        self._env_cache: dict[
            tuple[int | None, frozenset[int]], tuple[Any, Callable[..., Any]]
        ] = {}
        # ``_current_fn`` tracks the resolved achievement_fn for the
        # most recent ``_ensure_env`` call so ``_achievements`` and the
        # ``run()`` loop can read it without re-resolving. Not used for
        # cache keying — that's ``_env_cache``.
        self._current_fn: AchievementFn | None = achievement_fn
        self._env, self._jit_step = self._cached_env(achievement_fn, frozenset())
        self.seed = seed

    def _cached_env(
        self,
        achievement_fn: AchievementFn | None,
        blocked_actions: frozenset[int],
    ) -> tuple[Any, Callable[..., Any]]:
        """Return cached ``(env, jit_step)`` for this config, building if missing."""
        fn_key = id(achievement_fn) if achievement_fn is not None else None
        key = (fn_key, blocked_actions)
        if key not in self._env_cache:
            self._env_cache[key] = self._build_env(achievement_fn, blocked_actions)
        return self._env_cache[key]

    def _build_env(
        self,
        achievement_fn: AchievementFn | None,
        blocked_actions: frozenset[int],
    ) -> tuple[Any, Callable[..., Any]]:
        """Build the environment and its JIT-compiled step function.

        Layering, outermost to innermost:

        - :class:`ActionMaskWrapper` (if any actions are blocked) —
          rewrites masked actions to ``NOOP`` before the inner env
          sees them.
        - :class:`FactoriaXEnv` — core simulator. The achievement
          condition function is bound at construction time and
          evaluated inside ``step_env``; results live on
          ``state.achievements_unlocked``.

        Args:
            achievement_fn: Optional achievement condition function.
            blocked_actions: Set of action ints to mask out.

        Returns:
            Tuple of ``(env, jit_step)``.
        """
        env: Any = FactoriaXEnv(achievement_fn=achievement_fn)
        if blocked_actions:
            env = ActionMaskWrapper(env, tuple(blocked_actions))
        return env, jax.jit(env.step_env)

    def _resolve_blocked(
        self, benchmark: Benchmark, bench_level: BenchmarkLevel
    ) -> frozenset[int]:
        """Pick the effective ``blocked_actions`` mask for *bench_level*.

        Per-level ``BenchmarkLevel.blocked_actions`` wins when set —
        including when set to an empty ``frozenset()``, which means
        "this level explicitly has no mask". When the per-level field
        is ``None``, fall back to the benchmark's class-level
        ``blocked_actions`` attribute (used by ``RocketBenchmark`` to
        share a single mask across all its levels).

        Args:
            benchmark: Benchmark being run.
            bench_level: Level whose mask we're resolving.

        Returns:
            Frozen set of action ids to block. Empty set means no mask.
        """
        if bench_level.blocked_actions is not None:
            return frozenset(bench_level.blocked_actions)
        return frozenset(getattr(benchmark, "blocked_actions", ()) or ())

    def _ensure_env(
        self, benchmark: Benchmark, bench_level: BenchmarkLevel
    ) -> AchievementFn | None:
        """Rebuild the cached env + JIT step when the level's needs change.

        Resolves both the achievement function (runner override beats
        benchmark-advertised) and the per-level ``blocked_actions``
        mask. Rebuilds ``self._env`` / ``self._jit_step`` only when one
        of these differs from what is currently wrapped — important
        because each rebuild triggers a JAX trace + compile of
        :meth:`step_env` for that level's shape.

        Args:
            benchmark: Benchmark being run.
            bench_level: Level about to execute.

        Returns:
            Resolved achievement function, or ``None`` if neither side
            advertises one.
        """
        resolved = self._achievement_fn or getattr(benchmark, "achievement_fn", None)
        blocked = self._resolve_blocked(benchmark, bench_level)
        self._env, self._jit_step = self._cached_env(resolved, blocked)
        self._current_fn = resolved
        return resolved

    def _achievements(self, state: EnvState) -> np.ndarray | None:
        """Return the unlocked-achievement mask as a numpy array, or None.

        Returns ``None`` when no achievement_fn is bound to the runner;
        the engine still carries an all-False ``achievements_unlocked``
        field on every state, but reporting ``None`` preserves the
        contract that ``LevelResult.achievements_unlocked`` distinguishes
        "this benchmark tracks achievements" from "no tracking attempted".
        """
        if self._current_fn is None:
            return None
        return np.asarray(state.achievements_unlocked)

    @staticmethod
    def _select_player(state: EnvState, player_idx: int) -> EnvState:
        """Return a copy of ``state`` with ``selected_player`` set."""
        return state.replace(selected_player=player_idx)

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
            self._ensure_env(benchmark, bench_level)
            rng, subkey = jax.random.split(rng)
            result = self._run_level(
                benchmark,
                bench_level,
                policies,
                subkey,
                _obs_fn,
                constraint_fn,
                self._current_fn,
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
        level_data: list[tuple[BenchmarkLevel, EnvState, np.ndarray, np.ndarray]] = []

        for bench_level in levels:
            self._ensure_env(benchmark, bench_level)
            params = bench_level.env_params
            state0 = build_state(bench_level.level, params)
            max_steps = params.max_timesteps

            # Broadcast initial state to (num_seeds, ...).
            states = jax.tree.map(
                lambda x: jnp.broadcast_to(
                    jnp.asarray(x)[None],
                    (num_seeds,) + jnp.asarray(x).shape,
                ),
                state0,
            )

            rngs = jax.vmap(jax.random.PRNGKey)(jnp.array(seeds))

            vmap_step = jax.vmap(
                self._env.step_env,
                in_axes=(0, 0, 0, None),
            )

            def _obs_single(s: EnvState) -> jax.Array:
                return _obs_fn(s, params, 0)

            vmap_obs = jax.vmap(_obs_single)
            vmap_policy = jax.vmap(policy_fn, in_axes=(0, 0))

            @jax.jit
            def _scan(
                states: EnvState,
                rngs: jax.Array,
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
                        step_keys,
                        st,
                        actions,
                        params,
                    )

                    # Freeze states that are already done.
                    next_st = jax.tree.map(
                        lambda o, n: jnp.where(
                            dones.reshape((-1,) + (1,) * (n.ndim - 1)),
                            o,
                            n,
                        ),
                        st,
                        next_st,
                    )
                    new_dones = dones | step_done
                    t_used = t_used + (~dones).astype(jnp.int32)
                    recorded = jnp.where(dones, 0, actions)
                    return (next_st, rn, new_dones, t_used), recorded

                init_dones = jnp.zeros(num_seeds, dtype=bool)
                init_t = jnp.zeros(num_seeds, dtype=jnp.int32)
                (final_st, _, _, t_used), all_actions = jax.lax.scan(
                    step,
                    (states, rngs, init_dones, init_t),
                    None,
                    length=max_steps,
                )
                # all_actions: (T, N)
                return final_st, all_actions, t_used

            final_states, all_actions, timesteps_used = _scan(states, rngs)

            # Transfer to CPU once.
            all_actions_np = np.asarray(all_actions)  # (T, N)
            timesteps_np = np.asarray(timesteps_used)  # (N,)
            level_data.append((bench_level, final_states, all_actions_np, timesteps_np))

        # Assemble per-seed BenchmarkResults.
        results: list[BenchmarkResult] = []
        for i in range(num_seeds):
            level_results: list[LevelResult] = []
            for bench_level, final_states, all_actions_np, timesteps_np in level_data:
                t_used = int(timesteps_np[i])
                state_i = jax.tree.map(lambda x: x[i], final_states)
                items_mined = {
                    "coal": int(state_i.items_mined[ItemType.COAL]),
                    "iron": int(state_i.items_mined[ItemType.IRON_ORE]),
                    "copper": int(state_i.items_mined[ItemType.COPPER_ORE]),
                }
                score = benchmark.score_level(bench_level, items_mined)
                level_results.append(
                    LevelResult(
                        level_name=bench_level.name,
                        items_mined=items_mined,
                        weighted_score=score,
                        timesteps_used=t_used,
                        actions=all_actions_np[:t_used, i],
                        final_state=state_i,
                        achievements_unlocked=self._achievements(state_i),
                    )
                )
            agg = benchmark.score(level_results)
            results.append(
                BenchmarkResult(
                    benchmark_name=benchmark.name,
                    level_results=level_results,
                    aggregate_score=agg,
                )
            )

        agg_scores = [r.aggregate_score for r in results]
        logger.info(
            "Benchmark '%s' batched (%d seeds): mean=%.3f  std=%.3f",
            benchmark.name,
            num_seeds,
            float(np.mean(agg_scores)),
            float(np.std(agg_scores)),
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
        achievement_fn: AchievementFn | None = None,
    ) -> LevelResult:
        """Execute one level and return the result.

        The level state is built deterministically from ``bench_level.level``
        and ``bench_level.env_params`` via ``build_state`` — no random key
        is needed for the reset. The PRNG key is used only for step_env.

        When *constraint_fn* is provided, it is evaluated once per tick
        (after all players have acted) and the per-step cost vectors are
        stored in ``LevelResult.constraint_costs``. When the runner has
        an achievement function bound (either explicitly or via the
        benchmark's ``achievement_fn`` attribute), the latched unlock
        mask is read directly off ``state.achievements_unlocked``
        and returned on the result.

        Args:
            benchmark: Benchmark owning this level (provides per-level scoring).
            bench_level: Level to run.
            policies: Policies indexed by player index.
            rng: PRNG key for this level's episode steps.
            obs_fn: Observation extraction function.
            constraint_fn: Optional constraint cost function.
            achievement_fn: Unused — kept in the signature for backward
                compat; the function itself is already baked into
                ``self._jit_step`` via the env constructor.

        Returns:
            ``LevelResult`` for this level.
        """
        del achievement_fn  # bound at env-construction time, not used here
        params = bench_level.env_params
        state: EnvState = build_state(bench_level.level, params)
        num_players = params.num_players
        jit_step = self._jit_step

        actions_log: list[int] = []
        costs_log: list[np.ndarray] = []
        done = jnp.array(False)

        for _ in range(params.max_timesteps):
            prev_state = state
            for p in range(num_players):
                state_p = self._select_player(state, p)
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
            achievements_unlocked=self._achievements(state),
        )
