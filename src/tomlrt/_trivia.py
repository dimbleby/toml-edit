"""Trivia primitives.

A `Trivia` is an ordered run of `TriviaPiece`s — whitespace,
newlines, and comments. Trivia hangs off slots and value nodes;
together with the `lexeme`/`raw` of value-bearing nodes the trivia
captures every byte of the source so the document round-trips
exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, eq=False)
class WhitespaceNode:
    """Run of spaces and/or tabs (no newlines)."""

    text: str

    def render(self) -> str:
        return self.text


@dataclass(slots=True, eq=False)
class NewlineNode:
    r"""A single line terminator (``\n`` or ``\r\n``)."""

    text: str

    def render(self) -> str:
        return self.text


@dataclass(slots=True, eq=False)
class CommentNode:
    """A ``# ...`` comment, *not* including the trailing newline."""

    text: str  # includes the leading '#'

    def render(self) -> str:
        return self.text


TriviaPiece = WhitespaceNode | NewlineNode | CommentNode


@dataclass(slots=True, eq=False)
class Trivia:
    """An ordered run of trivia pieces."""

    pieces: list[TriviaPiece] = field(default_factory=list)

    def render(self) -> str:
        pieces = self.pieces
        if not pieces:
            return ""
        if len(pieces) == 1:
            return pieces[0].render()
        return "".join([p.render() for p in pieces])


def trivia_has_comment(t: Trivia) -> bool:
    """True iff ``t`` contains any ``CommentNode`` piece."""
    return any(isinstance(p, CommentNode) for p in t.pieces)


def trivia_has_newline(t: Trivia) -> bool:
    """True iff ``t`` contains any ``NewlineNode`` piece."""
    return any(isinstance(p, NewlineNode) for p in t.pieces)


def clone_trivia(t: Trivia) -> Trivia:
    """Return a shallow copy of ``t`` (same pieces, fresh list)."""
    return Trivia(list(t.pieces))


@dataclass(slots=True, eq=False)
class EolTrivia:
    """End-of-line tail of a single physical line.

    Used by `KVSlot` and `StructuralHeaderSlot` to capture the
    optional inline comment plus the line terminator. ``newline``
    may be ``None`` only for the last line of a file with no
    final newline.
    """

    trailing_ws: WhitespaceNode | None  # whitespace before any comment / newline
    comment: CommentNode | None
    newline: NewlineNode | None

    def render(self) -> str:
        out: list[str] = []
        if self.trailing_ws is not None:
            out.append(self.trailing_ws.text)
        if self.comment is not None:
            out.append(self.comment.text)
        if self.newline is not None:
            out.append(self.newline.text)
        return "".join(out)


__all__ = [
    "CommentNode",
    "EolTrivia",
    "NewlineNode",
    "Trivia",
    "TriviaPiece",
    "WhitespaceNode",
]
