"""TOML loader for :class:`~factoriax.engine.recipes.RecipeBalance`.

Reads a balance overlay from a TOML file so users can tune game
balance without editing Python source. The loader:

1. Maps each TOML table key to an ``ItemType`` (case-insensitive
   match against the enum's member names).
2. Maps each table's keys to :class:`~factoriax.engine.recipes.RecipeOverride`
   fields (``input_counts``, ``output_count``, ``ticks``).
3. Builds a :class:`~factoriax.engine.recipes.RecipeBalance` with the parsed
   overrides; ``RecipeBalance``'s own ``__post_init__`` validates
   uniqueness of overridden outputs, and
   :meth:`~factoriax.engine.recipes.RecipeBook.with_balance` validates the
   per-recipe values when the overlay is applied.

Example TOML:

.. code-block:: toml

    [iron_plate]
    ticks = 1
    output_count = 2

    [wire]
    input_counts = [2, 1]

The loader is intentionally strict — unknown ItemType keys and
unknown override fields raise ``ValueError`` so a balance file with a
typo fails fast rather than silently doing nothing.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from factoriax.engine.constants import ItemType
from factoriax.engine.recipes import RecipeBalance, RecipeOverride

# Allowed override fields, mapped to their expected Python types
# in the parsed TOML.
_OVERRIDE_FIELDS: dict[str, type] = {
    "input_counts": list,
    "output_count": int,
    "ticks": int,
}


def _parse_override(name: str, raw: dict[str, Any]) -> RecipeOverride:
    """Parse one TOML table into a :class:`RecipeOverride`.

    Parameters
    ----------
    name :
        Recipe key (used in error messages).
    raw :
        Decoded TOML table.

    Any] :

    Returns
    -------
    The constructed
        class:`RecipeOverride`.

    Raises
    ------
    ValueError
        If ``raw`` contains an unknown field, the wrong
        type for a known field, or an empty ``input_counts`` list.
    """
    unknown = set(raw) - set(_OVERRIDE_FIELDS)
    if unknown:
        raise ValueError(
            f"Recipe override [{name}] has unknown field(s) "
            f"{sorted(unknown)}. Allowed fields: "
            f"{sorted(_OVERRIDE_FIELDS)}."
        )

    kwargs: dict[str, Any] = {}
    for field, expected in _OVERRIDE_FIELDS.items():
        if field not in raw:
            continue
        value = raw[field]
        if not isinstance(value, expected):
            raise ValueError(
                f"Recipe override [{name}].{field} expected "
                f"{expected.__name__}, got {type(value).__name__}."
            )
        if field == "input_counts":
            # The isinstance check above guarantees ``value`` is a list,
            # but ``expected`` is a runtime ``type`` and so cannot be
            # used to narrow. Assert keeps the static type accurate.
            assert isinstance(value, list)
            if not value:
                raise ValueError(
                    f"Recipe override [{name}].input_counts must be a "
                    f"non-empty list (use a count of 0 to neutralise "
                    f"a slot)."
                )
            for entry in value:
                if not isinstance(entry, int):
                    raise ValueError(
                        f"Recipe override [{name}].input_counts must "
                        f"contain integers; got {type(entry).__name__} "
                        f"({entry!r})."
                    )
            kwargs[field] = tuple(value)
        else:
            kwargs[field] = value
    return RecipeOverride(**kwargs)


def _resolve_item(name: str) -> int:
    """Resolve a TOML key into an ``ItemType`` integer (case-insensitive).

    Parameters
    ----------
    name :
        TOML table key (e.g. ``"iron_plate"``).

    Returns
    -------
    Integer value of the matching
        class:`ItemType` member.

    Raises
    ------
    ValueError
        If ``name`` does not match any ``ItemType``.
    """
    target = name.upper()
    for member in ItemType:
        if member.name == target:
            return int(member)
    valid = sorted(m.name.lower() for m in ItemType)
    raise ValueError(
        f"Unknown recipe key '{name}' — does not match any ItemType. "
        f"Valid keys (case-insensitive): {valid}."
    )


def load_balance_from_toml(path: str | Path) -> RecipeBalance:
    """Load a :class:`RecipeBalance` from a TOML file.

    The file's top-level tables are recipe keys (matched against
    :class:`ItemType` member names, case-insensitive). Each table's
    fields populate a :class:`RecipeOverride`.

    Parameters
    ----------
    path :
        Path to the TOML file.

    Returns
    -------
    Parsed
        class:`RecipeBalance`.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file contains an unknown recipe key, an
        unknown override field, or a value of the wrong type.
    tomllib.TOMLDecodeError
        If the TOML is malformed.
    """
    path = Path(path)
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    overrides: list[tuple[int, RecipeOverride]] = []
    for name, raw in data.items():
        if not isinstance(raw, dict):
            raise ValueError(
                f"Top-level entry '{name}' must be a TOML table, got "
                f"{type(raw).__name__}."
            )
        item = _resolve_item(name)
        overrides.append((item, _parse_override(name, raw)))

    return RecipeBalance(overrides=tuple(overrides))
