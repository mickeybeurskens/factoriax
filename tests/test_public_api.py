"""Public API contract: every ``factoriax.__all__`` symbol is documented.

Spec: SPEC.md Phase B item 6 — *every symbol in ``__all__`` has a
docstring, a stability tier, and at least one test or example
reference.* This test enforces the first two; the example reference
lives in the cookbook recipes added later in Phase D.

When this test fails, the failure message names the offending symbol
and the rule it violated, so the fix is local — add a docstring, an
``Example:`` block, or a stability-manifest entry.
"""

from __future__ import annotations

import inspect
from typing import Any

import factoriax
from factoriax._stability import STABILITY, StabilityTier


def _is_documented_callable(symbol: Any) -> bool:
    """Return ``True`` when *symbol* is a function/class that owes an Example block.

    Constants (enums, ints, dict instances, dataclass instances, modules)
    do not need an ``Example:`` block — they are referenced inline. Only
    callables and the classes researchers instantiate by hand carry the
    burden of a worked-example docstring.
    """
    if inspect.isfunction(symbol) or inspect.ismethod(symbol):
        return True
    if inspect.isclass(symbol):
        return not _is_enum_class(symbol)
    return False


def _is_enum_class(symbol: type) -> bool:
    """Return ``True`` when *symbol* is an Enum subclass.

    Enums document their members through the member list, not through
    a worked usage example.
    """
    import enum  # noqa: PLC0415

    return isinstance(symbol, type) and issubclass(symbol, enum.Enum)


def test_all_symbols_resolve() -> None:
    """Every name in ``factoriax.__all__`` resolves to a real attribute."""
    for name in factoriax.__all__:
        assert hasattr(factoriax, name), (
            f"factoriax.__all__ lists {name!r} but it is not importable; "
            f"add it to factoriax/__init__.py or remove it from __all__."
        )


def test_all_symbols_have_docstrings() -> None:
    """Every name in ``factoriax.__all__`` has a non-empty docstring."""
    for name in factoriax.__all__:
        symbol = getattr(factoriax, name)
        if isinstance(symbol, (int, str, float, bool, dict, list, tuple)):
            # Module-level constants (e.g. MAX_STACK_SIZE, LEVELS) carry
            # their documentation in the module that defines them; the
            # `__all__` docstring rule applies to callables and classes.
            continue
        doc = inspect.getdoc(symbol)
        assert doc, f"factoriax.{name} has no docstring; add one to its definition."


def test_callables_carry_example_blocks() -> None:
    """Functions and instantiable classes include an ``Example:`` block.

    The contract: a researcher reading the api-reference should see at
    least one runnable example for each function / class they might use.
    Enums and frozen pure-data containers are exempt.
    """
    missing: list[str] = []
    for name in factoriax.__all__:
        symbol = getattr(factoriax, name)
        if not _is_documented_callable(symbol):
            continue
        doc = inspect.getdoc(symbol) or ""
        if "Example:" not in doc:
            missing.append(name)
    assert not missing, (
        "These public callables have no Example: block in their docstring: "
        f"{missing}. Add a runnable code example."
    )


def test_stability_manifest_covers_all() -> None:
    """The stability manifest has exactly one entry per ``__all__`` symbol."""
    public = set(factoriax.__all__)
    declared = set(STABILITY)
    extra = declared - public
    missing = public - declared
    assert not extra, (
        f"factoriax/_stability.py declares symbols absent from __all__: {sorted(extra)}"
    )
    assert not missing, (
        f"factoriax/_stability.py is missing tier entries for: {sorted(missing)}"
    )


def test_stability_manifest_uses_known_tiers() -> None:
    """Every manifest value is one of the documented tiers."""
    valid: set[StabilityTier] = {"Stable", "Experimental", "Internal"}
    for name, tier in STABILITY.items():
        assert tier in valid, (
            f"factoriax._stability.STABILITY[{name!r}] = {tier!r} is not a "
            f"recognized tier (one of {sorted(valid)})."
        )
