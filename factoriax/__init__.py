"""FactoriaX: A JAX-based grid environment."""

from factoriax.constants import (
    MAX_STACK_SIZE,
    NUM_INVENTORY_SLOTS,
    Action,
    BlockType,
    ItemType,
)
from factoriax.envs.factoriax_env import FactoriaXEnv, make_factoriax_env
from factoriax.state import EnvParams, EnvState

__all__ = [
    "Action",
    "BlockType",
    "EnvParams",
    "EnvState",
    "FactoriaXEnv",
    "ItemType",
    "MAX_STACK_SIZE",
    "NUM_INVENTORY_SLOTS",
    "make_factoriax_env",
]
