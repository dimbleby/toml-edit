"""Container layer (Phase 1 stub).

Phase 1 only needs `Document` as a thin owner of the physical
slot stream (head/tail of the doubly-linked list, plus trailing
trivia and detected newline). The full `Container` machinery
(``_index``, ``_refs``, ``_header_ref``, anchors, etc.) is built
out in later phases.

`Table`, `Array`, `AoT`, `TomlInput` are placeholder symbols so
the public ``__init__.py`` re-exports work; their full APIs are
filled in by Phase 2+.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tomlrt._render import render
from tomlrt._trivia import Trivia

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomlrt._slots import Slot


class Document:
    """A parsed TOML document.

    Phase 1 surface: ``render()`` -> str produces the original
    source byte-for-byte. The full dict-shaped API is Phase 2+.
    """

    __slots__ = ("_head", "_newline", "_tail", "_trailing")

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        if data is not None:
            msg = "Document(data=...) is not supported in Phase 1"
            raise NotImplementedError(msg)
        self._head: Slot | None = None
        self._tail: Slot | None = None
        self._trailing: Trivia = Trivia()
        self._newline: str = "\n"

    @classmethod
    def _from_parse(cls, slots: list[Slot], trailing: Trivia, newline: str) -> Document:
        doc = cls.__new__(cls)
        doc._head = slots[0] if slots else None  # noqa: SLF001
        doc._tail = slots[-1] if slots else None  # noqa: SLF001
        doc._trailing = trailing  # noqa: SLF001
        doc._newline = newline  # noqa: SLF001
        return doc

    def render(self) -> str:
        return render(self)


# Phase 1 placeholder — `Container` will be the dict-like base of
# `Document`/`Table` from Phase 2 onwards. Aliased so other modules
# (e.g. `_slots.SlotRef`) can import it now.
Container = Document


# Phase 1 placeholder symbols. Tests that need these will start
# passing in later phases.


class Table:  # pragma: no cover - Phase 2+
    pass


class Array:  # pragma: no cover - Phase 2+
    pass


class AoT:  # pragma: no cover - Phase 2+
    pass


TomlInput = "Mapping[str, Any] | Document"


__all__ = ["AoT", "Array", "Document", "Table", "TomlInput"]
