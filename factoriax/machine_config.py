"""Machine configuration — per-machine-type tunable knobs.

Parallel structure to :mod:`factoriax.recipes`, but for machine
behavior rather than recipes. Currently exposes a single knob —
``max_stack``, the per-machine-type buffer cap — but the dataclass
shape is designed to grow (extra arrays for craft rate, distinct
type cap, etc.) without breaking the JIT cache.

Two layers:

- :class:`MachineConfigOverride` is a frozen dataclass holding the
  optional override fields a caller wants to apply for one machine
  type. Fields are ``None`` by default; only non-None fields take
  effect.
- :class:`MachineConfig` is the JAX-friendly :class:`PyTreeNode`
  that the engine reads from. Its arrays are indexed by machine
  type (length = ``len(MachineType)``), so the JIT cache survives
  any override (the shape is fixed; only the values change).

Default values come from :data:`~factoriax.machine_spec.MACHINE_MAX_STACK`
so the constant remains the single source of truth for the engine's
shipped stack caps. Override-aware overlays use
:meth:`MachineConfig.with_overrides` which mirrors
:meth:`factoriax.recipes.RecipeBook.with_balance`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from flax import struct

from factoriax.constants import MachineType
from factoriax.machine_spec import MACHINE_MAX_HEALTH, MACHINE_MAX_STACK


@dataclass(frozen=True)
class MachineConfigOverride:
    """Optional per-machine-type overrides.

    Each field is ``None`` by default; only non-None fields actually
    override the corresponding entry in :class:`MachineConfig`. Bundles
    the override knobs into one record so callers can extend
    :class:`MachineConfig` (more arrays) without changing every call
    site that builds an override dict.

    Attributes:
        max_stack: New per-machine buffer cap. Must be a non-negative
            int; the engine reads it as int16 so values must fit.
        max_health: New per-machine maximum health. Must be a
            non-negative int; the engine reads it as int16 so values
            must fit.
    """

    max_stack: int | None = None
    max_health: int | None = None


class MachineConfig(struct.PyTreeNode):  # type: ignore[no-untyped-call]
    """Per-machine-type tuning the engine consumes inside JIT'd kernels.

    Stored as a PyTree leaf on :class:`~factoriax.state.EnvParams` so
    JIT'd kernels in :mod:`factoriax.machines` can read
    ``params.machine_config.max_stack`` without re-baking the XLA
    graph when overrides change. Shape is fixed by ``len(MachineType)``
    so the JIT cache survives across different override sets.

    Attributes:
        max_stack: Per-machine buffer cap, shape
            ``(len(MachineType),)``, int16.
        max_health: Per-machine maximum health, shape
            ``(len(MachineType),)``, int16. Default mirrors
            :data:`~factoriax.machine_spec.MACHINE_MAX_HEALTH`;
            wrappers tune via :meth:`with_overrides`.
    """

    max_stack: jnp.ndarray
    max_health: jnp.ndarray

    @classmethod
    def default(cls) -> MachineConfig:
        """Construct the default config from shipped engine constants.

        Returns:
            :class:`MachineConfig` whose ``max_stack`` mirrors
            :data:`~factoriax.machine_spec.MACHINE_MAX_STACK` and whose
            ``max_health`` mirrors
            :data:`~factoriax.machine_spec.MACHINE_MAX_HEALTH`.
        """
        return cls(
            max_stack=jnp.asarray(MACHINE_MAX_STACK, dtype=jnp.int16),
            max_health=jnp.asarray(MACHINE_MAX_HEALTH, dtype=jnp.int16),
        )

    def with_overrides(
        self,
        overrides: dict[int, MachineConfigOverride],
    ) -> MachineConfig:
        """Apply a dict of per-machine-type overrides over this config.

        Caller specifies ``{machine_type: MachineConfigOverride(...)}``.
        Any override field set to ``None`` leaves the corresponding
        engine value untouched; only set fields take effect. The result
        is a new :class:`MachineConfig` with the same array shapes.

        Args:
            overrides: Mapping from machine type integer (e.g.
                ``int(MachineType.PALLET)``) to a
                :class:`MachineConfigOverride`. Empty or all-None
                overrides return an array equal to ``self``.

        Returns:
            New :class:`MachineConfig`.

        Raises:
            ValueError: If a key is out of range for the machine-type
                axis, or a ``max_stack`` value is negative.
        """
        if not overrides:
            return self
        n = self.max_stack.shape[0]
        max_stack = self.max_stack
        max_health = self.max_health
        for mt, ov in overrides.items():
            mt_int = int(mt)
            if mt_int < 0 or mt_int >= n:
                raise ValueError(
                    f"machine type {mt_int} out of range [0, {n}); valid "
                    f"types: {[m.name for m in MachineType]}"
                )
            if ov.max_stack is not None:
                if ov.max_stack < 0:
                    raise ValueError(
                        f"max_stack must be non-negative; got {ov.max_stack} "
                        f"for {MachineType(mt_int).name}"
                    )
                max_stack = max_stack.at[mt_int].set(jnp.int16(ov.max_stack))
            if ov.max_health is not None:
                if ov.max_health < 0:
                    raise ValueError(
                        f"max_health must be non-negative; got {ov.max_health} "
                        f"for {MachineType(mt_int).name}"
                    )
                max_health = max_health.at[mt_int].set(jnp.int16(ov.max_health))
        return MachineConfig(max_stack=max_stack, max_health=max_health)


#: Default :class:`MachineConfig` derived from
#: :data:`~factoriax.machine_spec.MACHINE_MAX_STACK`. Used as the default
#: ``machine_config`` value on :class:`~factoriax.state.EnvParams`.
DEFAULT_MACHINE_CONFIG: MachineConfig = MachineConfig.default()
