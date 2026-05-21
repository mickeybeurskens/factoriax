"""Constraint functions for the FactoriaX environment.

Constraint functions measure safety violations at each step.  They
share the same ``(prev_state, new_state, params)`` input signature as
reward functions but return a *vector* of non-negative costs instead
of a scalar reward.  Zero means the constraint is satisfied; positive
values indicate the magnitude of the violation.

This separation is intentional.  Rewards and constraints live side by
side in the step loop, stored in different fields of the result, so
researchers can combine them however they like: as Lagrangian
penalties, as hard filters, as separate CMDP cost channels, or purely
as diagnostic metrics.  The scenario infrastructure never blends
them automatically.

Each constraint function has a companion ``*_names()`` function that
returns human-readable labels for each dimension, so analysis tooling
can label plots without hardcoding constraint indices.

All functions are pure JAX and fully JIT-compatible.  Bind extra
parameters with :func:`functools.partial` before passing to
:func:`jax.jit`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from factoriax.constants import ItemType
from factoriax.state import EnvParams, EnvState

_ORE_ITEMS = jnp.array(
    [ItemType.COAL, ItemType.IRON_ORE, ItemType.COPPER_ORE],
    dtype=jnp.int32,
)

# Pairs for pairwise balance: (coal, iron), (coal, copper), (iron, copper).
_PAIR_A = jnp.array([ItemType.COAL, ItemType.COAL, ItemType.IRON_ORE], dtype=jnp.int32)
_PAIR_B = jnp.array(
    [ItemType.IRON_ORE, ItemType.COPPER_ORE, ItemType.COPPER_ORE],
    dtype=jnp.int32,
)


def balance_cost(
    prev_state: EnvState,
    new_state: EnvState,
    params: EnvParams,
    threshold: float = 3.0,
) -> jax.Array:
    """Per-pair imbalance cost across the three ore types.

    For each pair ``(A, B)`` of ore types, the cost is
    ``max(0, max(a, b) / (min(a, b) + 1) - threshold)`` where ``a``
    and ``b`` are the lifetime mined counts.  The ``+1`` in the
    denominator avoids division by zero and makes the cost smooth when
    one type has not been mined yet.

    A cost of zero means the pair is within the allowed imbalance
    ratio.  Positive values grow linearly with the excess ratio.

    Args:
        prev_state: State immediately before the step (unused; present
            for interface uniformity).
        new_state: State immediately after the step.
        params: Environment parameters (unused).
        threshold: Maximum allowed ratio before cost becomes positive.

    Returns:
        Float32 array of shape ``(3,)``:
        ``[coal/iron, coal/copper, iron/copper]``.
    """
    counts_a = new_state.items_mined[_PAIR_A].astype(jnp.float32)
    counts_b = new_state.items_mined[_PAIR_B].astype(jnp.float32)
    ratio = jnp.maximum(counts_a, counts_b) / (jnp.minimum(counts_a, counts_b) + 1.0)
    cost: jax.Array = jnp.maximum(0.0, ratio - threshold)
    return cost


def balance_cost_names() -> list[str]:
    """Return human-readable labels for each dimension of :func:`balance_cost`.

    Returns:
        List of three strings.
    """
    return ["coal_iron_imbalance", "coal_copper_imbalance", "iron_copper_imbalance"]


def diversity_cost(
    prev_state: EnvState,
    new_state: EnvState,
    params: EnvParams,
) -> jax.Array:
    """Number of ore types not yet mined.

    Returns 0 when all three types (coal, iron, copper) have been
    mined at least once.  Returns 1, 2, or 3 otherwise.

    Args:
        prev_state: State immediately before the step (unused).
        new_state: State immediately after the step.
        params: Environment parameters (unused).

    Returns:
        Float32 array of shape ``(1,)``.
    """
    mined = new_state.items_mined[_ORE_ITEMS]
    unmined = jnp.sum(mined == 0).astype(jnp.float32)
    cost: jax.Array = jnp.array([unmined])
    return cost


def diversity_cost_names() -> list[str]:
    """Return human-readable labels for each dimension of :func:`diversity_cost`.

    Returns:
        List of one string.
    """
    return ["unmined_types"]
