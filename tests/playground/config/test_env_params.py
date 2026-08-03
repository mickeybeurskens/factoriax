"""Tests for the bridge from a player configuration to :class:`EnvParams`.

``config_to_env_params`` reads the tunable fields of a configuration and
returns the parameters the engine takes. ``env_params_to_dict`` runs the other
way, and a recording carries its result so a replay can restore the same
world.
"""

from __future__ import annotations

from factoriax.engine.state import EnvParams
from factoriax.playground.config import (
    PlayerConfig,
    config_to_env_params,
    env_params_to_dict,
)


class TestEnvParamsConversion:
    """Verify round-trip EnvParams <-> dict conversion."""

    def test_round_trip_defaults(self) -> None:
        """Default EnvParams survives a dict round trip."""
        params = EnvParams()
        d = env_params_to_dict(params)
        restored = config_to_env_params(PlayerConfig(env_params=d))
        for field_name in d:
            assert getattr(restored, field_name) == getattr(params, field_name)

    def test_custom_values(self) -> None:
        """A round trip keeps custom param values."""
        params = EnvParams(player_mining_yield=5)
        d = env_params_to_dict(params)
        assert d["player_mining_yield"] == 5

    def test_missing_fields_use_defaults(self) -> None:
        """Missing fields in the dict fall back to defaults."""
        config = PlayerConfig(env_params={"max_timesteps": 500})
        params = config_to_env_params(config)
        assert params.max_timesteps == 500
        assert params.player_mining_yield == EnvParams().player_mining_yield

    def test_player_mining_yield_default(self) -> None:
        """EnvParams exposes a default player_mining_yield of 1."""
        assert EnvParams().player_mining_yield == 1

    def test_player_mining_yield_in_dict(self) -> None:
        """player_mining_yield round-trips through env_params_to_dict."""
        d = env_params_to_dict(EnvParams())
        assert d["player_mining_yield"] == 1

    def test_player_mining_yield_custom_value(self) -> None:
        """A non-default player_mining_yield survives the round trip."""
        params = EnvParams(player_mining_yield=5)
        d = env_params_to_dict(params)
        restored = config_to_env_params(PlayerConfig(env_params=d))
        assert d["player_mining_yield"] == 5
        assert restored.player_mining_yield == 5
