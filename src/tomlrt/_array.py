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
    from tomlrt._container import Container, Document, Table
    from tomlrt._values import ArrayValue


class Array(list[Any]):
    """An inline array (`[ ... ]`)."""

    __slots__ = ("_multiline", "_value")

    def __init__(self) -> None:
        super().__init__()
        self._value: ArrayValue | None = None
        self._multiline: bool = False

    def to_list(self) -> list[Any]:
        """Materialise a plain-Python ``list`` (recursive)."""
        from tomlrt._container import _to_python  # noqa: PLC0415

        return [_to_python(x) for x in self]

    def array(self, index: int) -> Array:
        """Return ``self[index]`` typed as a nested `Array`."""
        v = self[index]
        if not isinstance(v, Array):
            msg = f"item at {index} is {type(v).__name__}, not Array"
            raise TypeError(msg)
        return v

    def table(self, index: int) -> Table:
        """Return ``self[index]`` typed as a `Table`."""
        from tomlrt._container import Table  # noqa: PLC0415

        v = self[index]
        if not isinstance(v, Table):
            msg = f"item at {index} is {type(v).__name__}, not Table"
            raise TypeError(msg)
        return v


class AoT(list["Table"]):
    """A `[[name]]`-style array of tables.

    Carries minimal logical-attachment state (`_layout_root`, `_path`,
    `_parent`) that survives the empty-AoT case (where `self[-1]`
    cannot be consulted as a fallback parent reference).
    """

    __slots__ = ("_layout_root", "_parent", "_path")

    def __init__(self) -> None:
        super().__init__()
        self._layout_root: Document | None = None
        self._path: tuple[str, ...] = ()
        self._parent: Container | None = None

    def to_list(self) -> list[dict[str, Any]]:
        """Materialise a list of plain-Python ``dict``s (recursive)."""
        return [t.to_dict() for t in self]


__all__ = ["AoT", "Array"]
