"""Isolation tests for the throughput-JSON writer in train_ppo.

Exercises ``_write_throughput_json`` directly with synthetic timing samples
(no env stepping / JIT) to check it writes into the requested directory,
returns the written path, auto-suffixes the basename, and emits the schema
the paper-side consolidator expects.
"""

from __future__ import annotations

from pathlib import Path

import orjson

from baselines.easy_rocket.ppo.train_ppo import Config, _write_throughput_json


def test_writes_into_requested_dir_and_returns_path(tmp_path: Path) -> None:
    config = Config()
    steps_per_iter = config.ppo.num_envs * config.ppo.rollout_steps

    out = _write_throughput_json(
        output_path=str(tmp_path / "ppo_throughput.json"),
        scenario="EasyRocket-v1",
        config=config,
        steady_sps=12345.0,
        warmup_wall_samples=[0.5],
        iter_wall_samples=[0.1, 0.1, 0.1],
        steps_per_iter=steps_per_iter,
    )

    assert isinstance(out, Path)
    assert out.exists()
    assert out.parent == tmp_path  # written into the directory we asked for
    # Basename is auto-suffixed with gpu + num_envs so files never collide.
    assert out.name.startswith("ppo_throughput")
    assert f"envs{config.ppo.num_envs}" in out.name


def test_payload_schema(tmp_path: Path) -> None:
    config = Config()
    out = _write_throughput_json(
        output_path=str(tmp_path / "ppo_throughput.json"),
        scenario="EasyRocket-v1",
        config=config,
        steady_sps=999.0,
        warmup_wall_samples=[1.0],
        iter_wall_samples=[0.2, 0.2],
        steps_per_iter=config.ppo.num_envs * config.ppo.rollout_steps,
    )

    payload = orjson.loads(out.read_bytes())
    assert payload["scenario"] == "EasyRocket-v1"
    assert payload["num_envs"] == config.ppo.num_envs
    run = payload["runs"][0]
    assert run["steady_state_sps"] == 999.0
    assert run["measured_steps"] == 2 * config.ppo.num_envs * config.ppo.rollout_steps
