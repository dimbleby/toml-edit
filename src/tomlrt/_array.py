"""Array views.

`Array(list)` backs an inline-array TOML value (`[1, 2, 3]`). `AoT`
(array-of-tables) is a `list[Table]` whose entries are individually
backed by `AoTEntry` records and whose elements are `Table` views.

Phase 2 surface: read-only access (length, indexing, iteration, value
decoding for plain inline-array elements). Mutation arrives in
Phase 3 (inline) and Phase 4 (AoT and multi-line array).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tomlrt._container import Table
    from tomlrt._values import ArrayValue


class Array(list[Any]):
    """An inline array (`[ ... ]`)."""

    __slots__ = ("_multiline", "_value")

    def __init__(self) -> None:
        super().__init__()
        self._value: ArrayValue | None = None
        self._multiline: bool = False


class AoT(list["Table"]):
    """A `[[name]]`-style array of tables."""

    __slots__ = ()


__all__ = ["AoT", "Array"]
