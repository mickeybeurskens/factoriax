# scripts/

Standalone command-line tools that live outside the package proper.
The default test runner doesn't collect from this directory; treat
each script as an independent program.

For a stated goal, pick the script below and read its module
docstring for full options.

## Performance & profiling

| Script | Purpose | Output target |
| --- | --- | --- |
| [`benchmark.py`](benchmark.py) | Comprehensive step-throughput sweep across map sizes, single env and vmapped batches. | Stdout table + optional wandb log. |
| [`post_commit_perf.py`](post_commit_perf.py) | A/B benchmark of the current branch vs. `HEAD~1`; flags regressions. | Stdout summary; wandb logs by default. |
| [`profiler.py`](profiler.py) | JAX profiler runs (throughput, memory, Perfetto traces) at multiple map sizes. | `runs/profiler/` traces + summary. |
| [`quick_profile_to_term.py`](quick_profile_to_term.py) | Fast terminal-only profile via `lax.scan`. | Stdout. |
| [`bench_rotate.py`](bench_rotate.py) | A/B before/after the action-rotate path. | Stdout. |
| [`bench_rocket_wrapper.py`](bench_rocket_wrapper.py) | Raw vs `achievement_fn`-bound env throughput on the rocket level. | Stdout. |
| [`jax_addition_benchmark.py`](jax_addition_benchmark.py) | Standalone JAX A/B (naive loop vs vmap vs scan) for orientation. | Stdout. |
| [`plot_benchmarks.py`](plot_benchmarks.py) | Render historical benchmark progression plots; upload to wandb. | `plots/benchmarks/` PNGs. |

## Build / asset tooling

| Script | Purpose | Output target |
| --- | --- | --- |
| [`build_atlas.py`](build_atlas.py) | Regenerate `factoriax/assets/atlas.png` + `atlas.json` from the procedural sprite code in `factoriax.ui.icons`. Output is byte-stable; CI verifies via `tests/test_atlas_fresh.py`. | `factoriax/assets/atlas.png`, `factoriax/assets/atlas.json`. |
| [`generate_api_reference.py`](generate_api_reference.py) | Walk `factoriax.__all__` and emit `docs/api-reference.md`. Deterministic; `tests/test_api_reference_fresh.py` enforces freshness. | `docs/api-reference.md`. |

## Rocket-benchmark agents and analysis

| Script | Purpose | Output target |
| --- | --- | --- |
| [`compare_rocket_agents.py`](compare_rocket_agents.py) | Run the scripted rocket agents and compare per-achievement timings. Save side-by-side videos. | `runs/compare_rocket/` artifacts. |
| [`record_scripted_rocket.py`](record_scripted_rocket.py) | Record a full scripted-rocket-agent run as an `agentdebugger` `.npz` trajectory. | `runs/scripted_rocket/` trajectory. |
| [`diagnose_advanced_factory.py`](diagnose_advanced_factory.py) | End-of-run diagnostic dump for the M4 advanced-factory scripted agent. | Stdout + per-step CSVs. |
| [`fault_advanced_factory.py`](fault_advanced_factory.py) | Fault analysis for the advanced-factory agent at a specific tick. | Stdout. |
| [`scout_phase3_layout.py`](scout_phase3_layout.py) | Reconnaissance dump for the Phase 3 rocket-factory layout. | Stdout. |

## Misc

| Script | Purpose | Output target |
| --- | --- | --- |
| [`test_agentdebugger.py`](test_agentdebugger.py) | Smoke test that the agent debugger boots, replays a trajectory, and saves a frame. | `runs/agentdebugger_smoke/`. |

## Conventions

- All scripts use [`uv`](https://docs.astral.sh/uv/) for dependency
  management. Run them as `uv run python scripts/<name>.py`.
- Scripts that produce artifacts default to a path under `runs/`;
  most accept `--out` to override.
- Scripts that touch GPUs are guarded by `nvidia-smi` checks where
  it makes sense; respect the convention that long-running training
  shouldn't be interrupted by ad-hoc benchmark runs.
