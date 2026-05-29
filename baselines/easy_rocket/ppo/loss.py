"""PPO update function for the easy rocket baseline.

The ``make_update_fn`` factory builds a JIT-compiled update closure that
runs PPO epochs on a flattened trajectory batch. It is parameterised by
a ``PPOHyperParams`` protocol so any config dataclass that exposes the
required fields can be used directly.
"""

from __future__ import annotations

from typing import Any, Protocol

import jax
import jax.numpy as jnp
import optax

from baselines.easy_rocket.ppo.network import ActorCritic
from baselines.easy_rocket.ppo.normalization import RunningStats, normalize_obs


class PPOHyperParams(Protocol):
    """Structural protocol for the PPO hyperparameters needed by the update.

    Any dataclass exposing these fields satisfies this protocol, including
    ``PPOConfig`` and any domain config that embeds it.
    """

    clip_eps: float
    value_coef: float
    entropy_coef: float
    num_minibatches: int
    update_epochs: int
    normalize_obs: bool


def make_update_fn(
    network: ActorCritic,
    optimizer: optax.GradientTransformation,
    config: PPOHyperParams,
) -> Any:
    """Build a JIT-compiled PPO update function.

    The returned function runs ``config.update_epochs`` epochs of minibatch
    SGD on a flattened trajectory batch, using the standard clipped surrogate
    objective with entropy bonus.

    Args:
        network: Actor-critic Flax module.
        optimizer: Optax gradient transformation.
        config: Any object exposing the required PPO hyperparameters.

    Returns:
        ``update_fn(params, opt_state, obs_stats, flat_obs, flat_actions,
        flat_log_probs, flat_advantages, flat_returns, rng)`` returning
        ``(new_params, new_opt_state, metrics_dict, new_rng)``.
    """

    @jax.jit
    def update(
        params: Any,
        opt_state: optax.OptState,
        obs_stats: RunningStats,
        flat_obs: jax.Array,
        flat_actions: jax.Array,
        flat_log_probs: jax.Array,
        flat_advantages: jax.Array,
        flat_returns: jax.Array,
        rng: jax.Array,
    ) -> tuple[Any, optax.OptState, dict[str, jax.Array], jax.Array]:
        """Run PPO epochs on a flattened trajectory batch.

        Args:
            params: Current network parameters.
            opt_state: Current optimizer state.
            obs_stats: Running observation statistics.
            flat_obs: Shape ``(B, obs_dim)``.
            flat_actions: Shape ``(B,)``.
            flat_log_probs: Behavior log-probabilities, shape ``(B,)``.
            flat_advantages: GAE advantages, shape ``(B,)``.
            flat_returns: Value targets, shape ``(B,)``.
            rng: PRNG key.

        Returns:
            Tuple ``(new_params, new_opt_state, metrics_dict, new_rng)``.
        """
        batch_size = flat_obs.shape[0]
        mb_size = batch_size // config.num_minibatches

        def _loss(
            p: Any,
            obs: jax.Array,
            actions: jax.Array,
            old_lp: jax.Array,
            adv: jax.Array,
            rets: jax.Array,
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            norm = normalize_obs(obs_stats, obs) if config.normalize_obs else obs
            logits, values = network.apply(p, norm)
            log_probs_all = jax.nn.log_softmax(logits)
            lp = log_probs_all[jnp.arange(obs.shape[0]), actions]

            probs = jax.nn.softmax(logits)
            entropy = -(probs * log_probs_all).sum(axis=-1).mean()

            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = jnp.exp(lp - old_lp)
            pg_loss = -jnp.minimum(
                ratio * adv,
                jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv,
            ).mean()
            value_loss = 0.5 * ((values - rets) ** 2).mean()
            total = (
                pg_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            )
            return total, {
                "loss/total": total,
                "loss/policy": pg_loss,
                "loss/value": value_loss,
                "loss/entropy": entropy,
                "misc/approx_kl": ((ratio - 1.0) - jnp.log(ratio)).mean(),
                "misc/clip_frac": (
                    (jnp.abs(ratio - 1.0) > config.clip_eps).astype(jnp.float32)
                ).mean(),
            }

        def _minibatch_step(
            carry: tuple[Any, optax.OptState],
            mb: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> tuple[tuple[Any, optax.OptState], dict[str, jax.Array]]:
            p, os = carry
            (_, m), grads = jax.value_and_grad(_loss, has_aux=True)(p, *mb)
            updates, new_os = optimizer.update(grads, os, p)
            return (optax.apply_updates(p, updates), new_os), m

        def _epoch(
            carry: tuple[Any, optax.OptState, jax.Array],
            _: None,
        ) -> tuple[tuple[Any, optax.OptState, jax.Array], dict[str, jax.Array]]:
            p, os, epoch_rng = carry
            epoch_rng, key_perm = jax.random.split(epoch_rng)
            perm = jax.random.permutation(key_perm, batch_size)

            def _reshape(x: jax.Array) -> jax.Array:
                return x[perm].reshape((config.num_minibatches, mb_size) + x.shape[1:])

            mbs = (
                _reshape(flat_obs),
                _reshape(flat_actions),
                _reshape(flat_log_probs),
                _reshape(flat_advantages),
                _reshape(flat_returns),
            )
            (p, os), metrics = jax.lax.scan(_minibatch_step, (p, os), mbs)
            return (p, os, epoch_rng), metrics

        (params, opt_state, rng), metrics = jax.lax.scan(
            _epoch,
            (params, opt_state, rng),
            None,
            length=config.update_epochs,
        )
        metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
        return params, opt_state, metrics, rng

    return update
