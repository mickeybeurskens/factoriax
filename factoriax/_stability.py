"""Stability tier manifest for symbols re-exported from :mod:`factoriax`.

Every name in :data:`factoriax.__all__` carries a tier so callers
know what backward-compatibility guarantees apply. The tiers are:

- ``"Stable"``: backward-compatible API. Removals or signature
  changes go through a deprecation cycle. Researchers can build on
  these without expecting churn.
- ``"Experimental"``: useful and exported, but the shape may change
  between minor versions. Suitable for active research code; not
  suitable for downstream packages that pin a version range.
- ``"Internal"``: exported for tooling (api-reference generator,
  tests) but not intended for end users. Treat as private.

The manifest is consumed by :mod:`scripts.generate_api_reference`,
which renders ``docs/api-reference.md`` with a tier column, and by
:mod:`tests.test_public_api`, which fails the build if a symbol is
exported without a tier.
"""

from __future__ import annotations

from typing import Literal

StabilityTier = Literal["Stable", "Experimental", "Internal"]

STABILITY: dict[str, StabilityTier] = {
    # --- engine: enums and constants (Stable; researchers reference
    # them when building levels and reading observation channels).
    "Action": "Stable",
    "BlockType": "Stable",
    "Direction": "Stable",
    "ItemType": "Stable",
    "NUM_ITEM_TYPES": "Stable",
    # --- engine: state + params PyTrees (Stable; the data contract
    # between the engine and everything that reads from it).
    "EnvParams": "Stable",
    "EnvState": "Stable",
    # --- engine: env class + canonical factory (Stable).
    "FactoriaXEnv": "Stable",
    "make": "Stable",
    # --- wrappers (Stable; small surface, well-tested,
    # composition order is documented on factoriax.make).
    "ActionMaskWrapper": "Stable",
    "AutoResetWrapper": "Stable",
    "ScienceTallyWrapper": "Experimental",
    # --- levels: builder + registry surface (Stable).
    "LEVELS": "Stable",
    "Level": "Stable",
    "LevelBuilder": "Stable",
    "build_state": "Stable",
    "generate_state": "Stable",
    "get_level": "Stable",
    "load_level": "Stable",
    "save_level": "Stable",
    # --- observations (Experimental; the schema is still settling
    # as new spatial channels land).
    "OBSERVATIONS": "Experimental",
    "global_superficial": "Experimental",
    "global_x_ray": "Experimental",
    "local_superficial": "Experimental",
    "local_x_ray": "Experimental",
    "rgb": "Experimental",
    # --- rewards (Experimental; new reward functions land regularly
    # and the signature may grow extra args).
    "mining_reward": "Experimental",
}
