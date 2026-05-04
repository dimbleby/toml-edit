"""Logical container layer.

`Container(dict)` is the dict-typed base for both `Document` (the
root) and `Table` (sections + inline tables). Phase 2 only needs the
read surface: dict-storage populated in doc-stream-first-occurrence
order, typed accessors, conversion helpers, and the `render()` entry
point. The mutation-time scaffolding (`_index`, `_refs`,
`_header_ref`, `_body_tail`, `_subtree_tail`) is deferred to Phase 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tomlrt._render import render
from tomlrt._trivia import Trivia

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomlrt._slots import AoTEntry, Slot


class Container(dict[str, Any]):
    """Dict-typed base for `Document` and `Table` views.

    Phase 2 surface: dict reads + typed accessors + `to_dict`. The
    insertion-anchor / index / ref machinery used by mutation lives
    behind further attributes added in Phase 3.
    """

    __slots__ = (
        "_inline",
        "_layout_root",
        "_owner_aot_entry",
        "_parent",
        "_path",
    )

    def __init__(self) -> None:
        super().__init__()
        self._layout_root: Document | None = None
        self._path: tuple[str, ...] = ()
        self._inline: bool = False
        self._parent: Container | None = None
        self._owner_aot_entry: AoTEntry | None = None

    # ------------------------------------------------------------------
    # Typed accessors
    # ------------------------------------------------------------------

    def table(self, key: str) -> Table:
        """Return ``self[key]`` typed as a `Table`.

        Raises ``KeyError`` if the key is missing and ``TypeError`` if
        the value is not a table.
        """
        v = self[key]
        if not isinstance(v, Table):
            msg = f"value at {key!r} is {type(v).__name__}, not Table"
            raise TypeError(msg)
        return v

    def array(self, key: str) -> Array:
        """Return ``self[key]`` typed as an inline `Array`."""
        v = self[key]
        if not isinstance(v, Array):
            msg = f"value at {key!r} is {type(v).__name__}, not Array"
            raise TypeError(msg)
        return v

    def aot(self, key: str) -> AoT:
        """Return ``self[key]`` typed as an array-of-tables (`AoT`)."""
        v = self[key]
        if not isinstance(v, AoT):
            msg = f"value at {key!r} is {type(v).__name__}, not AoT"
            raise TypeError(msg)
        return v

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Materialise a plain-Python ``dict`` (recursive)."""
        out: dict[str, Any] = {}
        for k, v in self.items():
            out[k] = _to_python(v)
        return out


class Table(Container):
    """A section table, implicit table, or inline table view."""

    __slots__ = ()


class Document(Container):
    """A parsed TOML document.

    Owns the physical slot stream (head/tail of the doubly-linked list,
    plus trailing trivia and detected newline). The dict-typed body is
    inherited from `Container`.
    """

    __slots__ = ("_head", "_newline", "_tail", "_trailing")

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        if data is not None:
            msg = "Document(data=...) is not supported in Phase 2"
            raise NotImplementedError(msg)
        self._head: Slot | None = None
        self._tail: Slot | None = None
        self._trailing: Trivia = Trivia()
        self._newline: str = "\n"
        self._layout_root = self

    def render(self) -> str:
        return render(self)


def _to_python(v: Any) -> Any:
    """Recursively materialise a tomlrt view into plain Python values."""
    if isinstance(v, Container):
        return v.to_dict()
    if isinstance(v, AoT):
        return [t.to_dict() for t in v]
    if isinstance(v, Array):
        return [_to_python(x) for x in v]
    return v


# `_array` depends on `Container` for `Table`, so the import is at the
# bottom to avoid a circular import. The `Array` / `AoT` symbols are
# re-exported for convenience.
from tomlrt._array import AoT, Array  # noqa: E402

TomlInput = "Mapping[str, Any] | Document"


__all__ = ["AoT", "Array", "Container", "Document", "Table", "TomlInput"]
