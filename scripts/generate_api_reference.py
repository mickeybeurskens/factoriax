"""Generate ``docs/api-reference.md`` from ``factoriax.__all__``.

Walks every symbol the package re-exports, pulls its signature via
``inspect`` and its summary line from ``__doc__``, and tags it with
the stability tier declared in :mod:`factoriax._stability`. Output
is deterministic — running the script twice into the same path
produces byte-identical files. CI verifies this via
``tests/test_api_reference_fresh.py``.

Usage::

    uv run python scripts/generate_api_reference.py
    uv run python scripts/generate_api_reference.py --out custom.md
"""

from __future__ import annotations

import argparse
import inspect
import logging
from pathlib import Path
from typing import Any

import factoriax
from factoriax._stability import STABILITY

logger = logging.getLogger(__name__)

_TIER_ORDER: dict[str, int] = {"Stable": 0, "Experimental": 1, "Internal": 2}


_GENERIC_DOC_PREFIXES: tuple[str, ...] = (
    "int(",
    "float(",
    "str(",
    "bool(",
    "dict(",
    "list(",
    "tuple(",
    "set(",
    "frozenset(",
    "Built-in",
    "bytes(",
)


def _summary_line(symbol: Any) -> str:
    """Return the docstring's first non-blank line, or ``""``.

    Filters out the auto-generated docstrings that ``dict``,
    ``int``, ``tuple`` etc. inherit when used as module-level
    constants — they're noise in the rendered table.
    """
    doc = inspect.getdoc(symbol) or ""
    for raw_line in doc.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(_GENERIC_DOC_PREFIXES):
            return ""
        return line
    return ""


def _signature(symbol: Any) -> str:
    """Return the symbol's signature as a markdown-safe string.

    Class signatures are suppressed: ``EnvParams``/``EnvState`` are
    PyTree dataclasses whose ``__init__`` defaults dump array
    contents into the rendered text; the docstring summary covers
    what callers need to know about the type.
    """
    if inspect.isclass(symbol):
        return ""
    try:
        sig = inspect.signature(symbol)
    except (TypeError, ValueError):
        return ""
    return str(sig).replace("|", "\\|")


def _kind(symbol: Any) -> str:
    """Classify a symbol for the api-reference table."""
    if inspect.isclass(symbol):
        return "class"
    if inspect.isfunction(symbol) or inspect.ismethod(symbol):
        return "function"
    if isinstance(symbol, (int, float, str, bool)):
        return "constant"
    if isinstance(symbol, (list, tuple, dict, frozenset, set)):
        return "constant"
    return "value"


def _sort_key(name: str) -> tuple[int, str]:
    """Order rows by stability tier first, then alphabetically."""
    tier = STABILITY.get(name, "Internal")
    return (_TIER_ORDER.get(tier, 99), name)


def _render_table(rows: list[tuple[str, str, str, str, str]]) -> list[str]:
    """Render a markdown table from ``(tier, name, kind, signature, summary)``."""
    out = [
        "| Tier | Name | Kind | Signature | Summary |",
        "| --- | --- | --- | --- | --- |",
    ]
    for tier, name, kind, sig, summary in rows:
        sig_md = f"`{sig}`" if sig else ""
        summary_md = summary.replace("|", "\\|")
        out.append(f"| {tier} | `{name}` | {kind} | {sig_md} | {summary_md} |")
    return out


def render_reference() -> str:
    """Render the api-reference markdown as a single string."""
    rows: list[tuple[str, str, str, str, str]] = []
    for name in sorted(factoriax.__all__, key=_sort_key):
        symbol = getattr(factoriax, name)
        tier = STABILITY.get(name, "Internal")
        rows.append(
            (
                tier,
                name,
                _kind(symbol),
                _signature(symbol),
                _summary_line(symbol),
            )
        )

    lines: list[str] = []
    lines.append("# API Reference")
    lines.append("")
    lines.append(
        "Auto-generated from `factoriax.__all__`. Do not edit by hand — run "
        "`uv run python scripts/generate_api_reference.py` to regenerate."
    )
    lines.append("")
    lines.append(
        "Stability tiers come from `factoriax/_stability.py`. **Stable** "
        "symbols carry the usual backward-compat guarantees; "
        "**Experimental** may change between minor versions; **Internal** "
        "is exported only for tooling."
    )
    lines.append("")
    lines.extend(_render_table(rows))
    lines.append("")
    return "\n".join(lines)


def write_reference(out_path: Path) -> None:
    """Write the api-reference markdown to *out_path*.

    Output is byte-stable across runs: the function pins the row
    order, signature formatting, and summary extraction so the
    freshness test passes against any commit of the same source.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_reference())


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "docs" / "api-reference.md",
        help="Output path for the markdown file.",
    )
    args = parser.parse_args()
    write_reference(args.out)
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
