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
        self._env = FactoriaXEnv()
        self._jit_step = jax.jit(self._env.step_env)
        self.seed = seed

    def run(
        self,
        benchmark: Benchmark,
        policies: list[Policy],
        obs_fn: Callable[[EnvState, EnvParams, int], jax.Array] | None = None,
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
            result = self._run_level(benchmark, bench_level, policies, subkey, _obs_fn)
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

    def _run_level(
        self,
        benchmark: Benchmark,
        bench_level: BenchmarkLevel,
        policies: list[Policy],
        rng: jax.Array,
        obs_fn: Callable[[EnvState, EnvParams, int], jax.Array],
    ) -> LevelResult:
        """Execute one level and return the result.

        The level state is built deterministically from ``bench_level.level``
        and ``bench_level.env_params`` via ``build_state`` — no random key
        is needed for the reset. The PRNG key is used only for step_env.

        Args:
            benchmark: Benchmark owning this level (provides per-level scoring).
            bench_level: Level to run.
            policies: Policies indexed by player index.
            rng: PRNG key for this level's episode steps.
            obs_fn: Observation extraction function.

        Returns:
            ``LevelResult`` for this level.
        """
        params = bench_level.env_params
        state = build_state(bench_level.level, params)
        num_players = params.num_players

        # Player-0 actions are logged for analysis. For multi-agent benchmarks
        # all players act sequentially each tick; analysis modules for those
        # benchmarks should interpret actions accordingly.
        actions_log: list[int] = []
        done = jnp.array(False)

        for _ in range(params.max_timesteps):
            for p in range(num_players):
                state_p = state.replace(selected_player=p)
                obs = obs_fn(state_p, params, p)
                action = policies[p](obs)

                rng, subkey = jax.random.split(rng)
                _, state, _, done, _ = self._jit_step(subkey, state_p, action, params)

                if p == 0:
                    actions_log.append(int(action))

            if bool(done):
                break

        items_mined: dict[str, int] = {
            "coal": int(state.items_mined[ItemType.COAL]),
            "iron": int(state.items_mined[ItemType.IRON]),
            "copper": int(state.items_mined[ItemType.COPPER]),
        }
        weighted_score = benchmark.score_level(bench_level, items_mined)

        return LevelResult(
            level_name=bench_level.name,
            items_mined=items_mined,
            weighted_score=weighted_score,
            timesteps_used=len(actions_log),
            actions=np.array(actions_log, dtype=np.int32),
        )
