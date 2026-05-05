"""Physical slot layer.

A document is an ordered intrusive doubly-linked list of physical
slots. Three slot kinds:

- ``KVSlot`` — one ``key = value`` line (possibly dotted).
- ``StructuralHeaderSlot`` — one ``[a.b]`` or ``[[a.b]]`` line.

`SlotRef` is the per-container occurrence of a slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from tomlrt._container import Container
    from tomlrt._trivia import Trivia
    from tomlrt._values import KeyPart, Value

import sys

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tomlrt._trivia import EolTrivia

# ---------------------------------------------------------------------------
# AoT entry token (physical ownership marker)
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class AoTEntry:
    """Identifies one entry of an array-of-tables.

    Carried by every physical slot that belongs to that entry
    (``owner_aot_entry``). The ``entry_slots`` list is populated by
    the slot-builder so render can iterate without scanning.
    """

    path: tuple[str, ...]
    ordinal: int
    entry_slots: list[Slot] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Slot base + kinds
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Slot:
    """Base for physical slots.

    Subclassed by `KVSlot` and `StructuralHeaderSlot`.
    """

    leading: Trivia
    _prev: Slot | None = field(default=None, repr=False, compare=False)
    _next: Slot | None = field(default=None, repr=False, compare=False)

    def render(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass(eq=False)
class KVSlot(Slot):
    """A single ``key = value`` line."""

    host_path: tuple[str, ...] = ()
    """Full path of the host table — the table whose body this KV
    physically belongs to. For top-level KVs ``()``; for KVs under
    ``[a.b]`` ``("a", "b")``.
    """

    key_parts: list[KeyPart] = field(default_factory=list)
    """The dotted-key parts as written. ``len >= 1``."""

    key_seps: list[str] = field(default_factory=list)
    """Whitespace + ``.`` between parts. Length ``len(key_parts) - 1``."""

    pre_eq: str = ""
    post_eq: str = ""
    value: Value | None = None
    eol: EolTrivia = field(default_factory=lambda: EolTrivia(None, None, None))
    owner_aot_entry: AoTEntry | None = None

    def render_key(self) -> str:
        parts = self.key_parts
        if len(parts) == 1:
            return parts[0].render()
        out: list[str] = []
        seps = self.key_seps
        for i, p in enumerate(parts):
            if i:
                out.append(seps[i - 1])
            out.append(p.render())
        return "".join(out)

    @property
    def key(self) -> tuple[str, ...]:
        """Decoded dotted key path."""
        return tuple([p.value for p in self.key_parts])

    @override
    def render(self) -> str:
        assert self.value is not None
        return (
            f"{self.leading.render()}{self.render_key()}"
            f"{self.pre_eq}={self.post_eq}"
            f"{self.value.render()}{self.eol.render()}"
        )


@dataclass(eq=False)
class StructuralHeaderSlot(Slot):
    """One ``[a.b]`` or ``[[a.b]]`` header line."""

    kind: Literal["table", "aot-entry"] = "table"
    path: tuple[str, ...] = ()
    """Full decoded path of the section / AoT entry header."""

    key_parts: list[KeyPart] = field(default_factory=list)
    key_seps: list[str] = field(default_factory=list)
    inner_pre: str = ""
    inner_post: str = ""
    eol: EolTrivia = field(default_factory=lambda: EolTrivia(None, None, None))
    owner_aot_entry: AoTEntry | None = None
    """The enclosing AoT entry, if any (independent of whether this
    header itself opens an AoT entry).
    """

    entry: AoTEntry | None = None
    """The AoT entry this header opens, when ``kind == 'aot-entry'``."""

    synthetic: bool = False
    """True iff this header was introduced by mutation."""

    def render_key(self) -> str:
        parts = self.key_parts
        if len(parts) == 1:
            return parts[0].render()
        out: list[str] = []
        seps = self.key_seps
        for i, p in enumerate(parts):
            if i:
                out.append(seps[i - 1])
            out.append(p.render())
        return "".join(out)

    @override
    def render(self) -> str:
        if self.kind == "aot-entry":
            open_br, close_br = "[[", "]]"
        else:
            open_br, close_br = "[", "]"
        return (
            f"{self.leading.render()}{open_br}{self.inner_pre}"
            f"{self.render_key()}{self.inner_post}{close_br}{self.eol.render()}"
        )


# Aliases for readability; not separate types.
HeaderSlot = StructuralHeaderSlot
AoTHeaderSlot = StructuralHeaderSlot


# ---------------------------------------------------------------------------
# SlotRef (per-container occurrence)
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class SlotRef:
    """A per-container occurrence of a slot."""

    slot: Slot
    container: Container
    local_key: str | None
    """The key under which this ref is filed in
    ``container._index`` — exactly one path component for body and
    child-binding refs. ``None`` for the container's own header
    ref (which lives in ``_refs`` + ``_header_ref``, not in
    ``_index``).
    """


__all__ = [
    "AoTEntry",
    "AoTHeaderSlot",
    "HeaderSlot",
    "KVSlot",
    "Slot",
    "SlotRef",
    "StructuralHeaderSlot",
]
