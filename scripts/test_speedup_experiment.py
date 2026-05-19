"""Run the four-variant Phase 1 experiment for ``SPEC_TEST_SUITE.md``.

Invokes ``pytest performance_experiments/jit_share/`` four times — one
per combination of JAX backend (CUDA, CPU) and fixture scope
(function, session) — and prints a markdown-ready summary table the
user can paste straight into the spec's "Phase 1 Results" section.

The variants:

| Variant | Backend | Fixture scope | -k filter                   |
| ------- | ------- | ------------- | --------------------------- |
| A1      | CUDA    | function      | TestBaselineFunctionScope   |
| A2      | CPU     | function      | TestBaselineFunctionScope   |
| B1      | CUDA    | session       | TestSharedSessionScope      |
| B2      | CPU     | session       | TestSharedSessionScope      |

The script also evaluates the gate inequalities from the spec:

- B2 must be at least 50% faster than A1 (validates the mechanism).
- B2 must be at least 30% faster than B1 (validates that CPU is part
  of the recipe).

The gate verdict is informational. The script exits non-zero only if
a variant has a failing test, never just because the gate failed —
that's a decision for the human reviewing Task 1.4's results.

Peak RSS is omitted from the output. ``/usr/bin/time -v`` is not
installed on the user's Arch box, and ``resource.getrusage`` reports
a running max across children that does not cleanly attribute to a
single subprocess. The gate is on wall time; RSS is informational and
not load-bearing.

Use ``--dry-run`` to see the four pytest invocations without running
them.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class Variant:
    """One row of the four-variant matrix."""

    label: str
    backend: str
    scope: str
    test_class: str


_VARIANTS: Final[list[Variant]] = [
    Variant("A1", "CUDA", "function", "TestBaselineFunctionScope"),
    Variant("A2", "CPU", "function", "TestBaselineFunctionScope"),
    Variant("B1", "CUDA", "session", "TestSharedSessionScope"),
    Variant("B2", "CPU", "session", "TestSharedSessionScope"),
]

_PYTEST_BASE: Final[list[str]] = [
    "uv",
    "run",
    "--quiet",
    "pytest",
    "performance_experiments/jit_share/",
    "-p",
    "no:cacheprovider",
    "--no-header",
    "--no-cov",
    "-q",
    "-o",
    "addopts=",
]

_TESTS_PER_VARIANT: Final[int] = 10


@dataclass
class VariantResult:
    """Captured outcome of one variant run."""

    variant: Variant
    wall_seconds: float
    passed: bool
    exit_code: int


def _build_command(variant: Variant) -> list[str]:
    """Construct the pytest invocation for a variant."""
    return [*_PYTEST_BASE, "-k", variant.test_class]


def _build_env(variant: Variant) -> dict[str, str]:
    """Construct the env-var overlay for a variant."""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cuda" if variant.backend == "CUDA" else "cpu"
    return env


def _run_variant(variant: Variant) -> VariantResult:
    """Run one variant under wall-clock measurement."""
    cmd = _build_command(variant)
    env = _build_env(variant)
    print(
        f"[{variant.label}] backend={variant.backend} scope={variant.scope} ...",
        flush=True,
    )
    start = time.monotonic()
    completed = subprocess.run(cmd, env=env, check=False)
    elapsed = time.monotonic() - start
    print(
        f"[{variant.label}] elapsed={elapsed:.2f}s exit={completed.returncode}",
        flush=True,
    )
    return VariantResult(
        variant=variant,
        wall_seconds=elapsed,
        passed=completed.returncode == 0,
        exit_code=completed.returncode,
    )


def _render_table(results: list[VariantResult]) -> str:
    """Format the four-row markdown table for the spec."""
    rows = [
        "| Variant | Backend | Fixture scope | Tests | Wall time (s) "
        "| s / test | Result |",
        "| ------- | ------- | ------------- | ----- | ------------- "
        "| -------- | ------ |",
    ]
    for r in results:
        per_test = r.wall_seconds / _TESTS_PER_VARIANT
        status = "passed" if r.passed else f"FAILED (exit {r.exit_code})"
        rows.append(
            f"| {r.variant.label} | {r.variant.backend} | "
            f"{r.variant.scope} | {_TESTS_PER_VARIANT} | "
            f"{r.wall_seconds:.2f} | {per_test:.3f} | {status} |"
        )
    return "\n".join(rows)


def _evaluate_gate(results: list[VariantResult]) -> tuple[str, bool]:
    """Compute the two gate inequalities and an overall verdict.

    Returns:
        Tuple ``(report_text, gate_passed)``. The text is the
        verdict block ready to paste into the spec; the boolean is
        ``True`` iff both inequalities hold.
    """
    by_label = {r.variant.label: r for r in results}
    a1 = by_label["A1"].wall_seconds
    b1 = by_label["B1"].wall_seconds
    b2 = by_label["B2"].wall_seconds

    def _pct_faster(slow: float, fast: float) -> float:
        return 100.0 * (slow - fast) / slow if slow > 0 else 0.0

    b2_vs_a1 = _pct_faster(a1, b2)
    b2_vs_b1 = _pct_faster(b1, b2)
    mechanism_ok = b2_vs_a1 >= 50.0
    cpu_ok = b2_vs_b1 >= 30.0

    lines = [
        "Gate evaluation:",
        f"- B2 vs A1: A1 = {a1:.2f}s, B2 = {b2:.2f}s -> "
        f"{b2_vs_a1:.1f}% faster (need >= 50%)  "
        f"{'PASS' if mechanism_ok else 'FAIL'}",
        f"- B2 vs B1: B1 = {b1:.2f}s, B2 = {b2:.2f}s -> "
        f"{b2_vs_b1:.1f}% faster (need >= 30%)  "
        f"{'PASS' if cpu_ok else 'FAIL'}",
        f"- Overall: {'PASS' if (mechanism_ok and cpu_ok) else 'FAIL'}",
    ]
    return "\n".join(lines), (mechanism_ok and cpu_ok)


def _dry_run() -> int:
    """Print the four pytest invocations without running them."""
    for variant in _VARIANTS:
        cmd = _build_command(variant)
        env_prefix = f"JAX_PLATFORMS={'cuda' if variant.backend == 'CUDA' else 'cpu'}"
        print(f"[{variant.label}] {env_prefix} {' '.join(cmd)}")
    return 0


def main() -> int:
    """Run the four variants and emit the results table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the four commands and exit without running them.",
    )
    args = parser.parse_args()

    if args.dry_run:
        return _dry_run()

    results: list[VariantResult] = []
    for variant in _VARIANTS:
        results.append(_run_variant(variant))

    print()
    print(_render_table(results))
    print()
    report, _ = _evaluate_gate(results)
    print(report)

    any_failed = any(not r.passed for r in results)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
