"""Physical CST node types for toml-edit.

Design notes
------------

Every byte of the source TOML document maps to exactly one node, so that
emitting the tree concatenated together exactly reproduces the original
input ("round-trip"). To make that work we keep two kinds of children
inside container nodes:

* "structural" nodes that carry semantic content (`KeyValueNode`,
  `TableHeaderNode`, `ArrayHeaderNode`),
* "trivia" nodes (`WhitespaceNode`, `NewlineNode`, `CommentNode`)
  that carry the surrounding whitespace and comments.

Trivia ownership rule (locked in plan.md):
* Leading whitespace and comment lines (with their trailing newlines)
  belong to the **following** structural node, attached as `leading`.
* The end-of-line comment after a key/value pair (and its trailing
  newline) belong to the same line via `trailing_comment` and `newline`.
* Anything left over at end-of-file is attached to the document as
  `trailing_trivia`.

These nodes are an internal implementation detail and never leak into
the public API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence  # used in docstrings only
    from datetime import date, datetime, time


# ---------------------------------------------------------------------------
# Trivia
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WhitespaceNode:
    """Run of spaces and/or tabs (no newlines)."""

    text: str

    def render(self) -> str:
        return self.text


@dataclass(slots=True)
class NewlineNode:
    """A single line terminator (``\\n`` or ``\\r\\n``)."""

    text: str

    def render(self) -> str:
        return self.text


@dataclass(slots=True)
class CommentNode:
    """A ``# ...`` comment, *not* including the trailing newline."""

    text: str  # includes the leading '#'

    def render(self) -> str:
        return self.text


TriviaPiece = WhitespaceNode | NewlineNode | CommentNode
"""A single trivia atom."""


@dataclass(slots=True)
class Trivia:
    """An ordered run of trivia pieces."""

    pieces: list[TriviaPiece] = field(default_factory=list)

    def render(self) -> str:
        return "".join(p.render() for p in self.pieces)

    def is_empty(self) -> bool:
        return not self.pieces


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


KeyKind = Literal["bare", "basic", "literal"]


@dataclass(slots=True)
class KeyPart:
    """A single dotted-key component (the part between dots)."""

    raw: str
    """Source representation including any surrounding quotes."""

    value: str
    """The decoded key string."""

    kind: KeyKind

    def render(self) -> str:
        return self.raw


@dataclass(slots=True)
class Key:
    """A dotted key: one or more :class:`KeyPart` separated by ``.``.

    ``separators`` carries the whitespace + ``.`` between parts. It always
    has length ``len(parts) - 1``.
    """

    parts: list[KeyPart]
    separators: list[str] = field(default_factory=list)

    def render(self) -> str:
        out: list[str] = []
        for i, part in enumerate(self.parts):
            if i:
                out.append(self.separators[i - 1])
            out.append(part.render())
        return "".join(out)

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(p.value for p in self.parts)


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


StringStyle = Literal["basic", "literal", "ml-basic", "ml-literal"]
IntStyle = Literal["dec", "hex", "oct", "bin"]


@dataclass(slots=True)
class StringNode:
    raw: str  # including quotes
    value: str
    style: StringStyle

    def render(self) -> str:
        return self.raw


@dataclass(slots=True)
class IntegerNode:
    raw: str
    value: int
    style: IntStyle

    def render(self) -> str:
        return self.raw


@dataclass(slots=True)
class FloatNode:
    raw: str
    value: float

    def render(self) -> str:
        return self.raw


@dataclass(slots=True)
class BoolNode:
    raw: str  # "true" or "false"
    value: bool

    def render(self) -> str:
        return self.raw


DateLikeKind = Literal["offset-datetime", "local-datetime", "local-date", "local-time"]


@dataclass(slots=True)
class DateTimeNode:
    raw: str
    value: datetime | date | time
    kind: DateLikeKind

    def render(self) -> str:
        return self.raw


@dataclass(slots=True)
class ArrayItem:
    """One slot inside an inline array.

    Layout: ``leading value trailing [comma] [post_comma_trivia]``.
    The final item has ``has_comma=False`` unless the source had a
    trailing comma.
    """

    leading: Trivia
    value: ValueNode
    trailing: Trivia
    has_comma: bool
    post_comma_trivia: Trivia

    def render(self) -> str:
        out = self.leading.render() + render_value(self.value) + self.trailing.render()
        if self.has_comma:
            out += "," + self.post_comma_trivia.render()
        return out


@dataclass(slots=True)
class ArrayNode:
    """Inline array literal (``[ ... ]``)."""

    items: list[ArrayItem] = field(default_factory=list)
    final_trivia: Trivia = field(default_factory=Trivia)
    """Trivia after the last item (or comma) and before the closing ``]``."""

    def render(self) -> str:
        body = "".join(item.render() for item in self.items)
        return "[" + body + self.final_trivia.render() + "]"


@dataclass(slots=True)
class InlineTableEntry:
    """One ``key = value`` slot inside an inline table."""

    leading: Trivia
    key: Key
    pre_eq: Trivia
    post_eq: Trivia
    value: ValueNode
    trailing: Trivia
    has_comma: bool
    post_comma_trivia: Trivia

    def render(self) -> str:
        out = (
            self.leading.render()
            + self.key.render()
            + self.pre_eq.render()
            + "="
            + self.post_eq.render()
            + render_value(self.value)
            + self.trailing.render()
        )
        if self.has_comma:
            out += "," + self.post_comma_trivia.render()
        return out


@dataclass(slots=True)
class InlineTableNode:
    """Inline table literal (``{ a = 1, b = 2 }``)."""

    entries: list[InlineTableEntry] = field(default_factory=list)
    final_trivia: Trivia = field(default_factory=Trivia)

    def render(self) -> str:
        body = "".join(e.render() for e in self.entries)
        return "{" + body + self.final_trivia.render() + "}"


ValueNode = (
    StringNode
    | IntegerNode
    | FloatNode
    | BoolNode
    | DateTimeNode
    | ArrayNode
    | InlineTableNode
)


def render_value(node: ValueNode) -> str:
    return node.render()


# ---------------------------------------------------------------------------
# Top-level structural nodes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class KeyValueNode:
    """A ``key = value`` line in the document or a standard table.

    Source layout::

        leading  KEY  pre_eq  '='  post_eq  VALUE  trailing  [# comment]  \\n

    ``leading`` may include comment lines and blank lines that belong to
    this entry. ``trailing`` is whitespace between the value and the
    optional inline comment / newline.
    """

    leading: Trivia
    key: Key
    pre_eq: Trivia
    post_eq: Trivia
    value: ValueNode
    trailing: Trivia
    trailing_comment: CommentNode | None
    newline: NewlineNode | None

    def render(self) -> str:
        out = (
            self.leading.render()
            + self.key.render()
            + self.pre_eq.render()
            + "="
            + self.post_eq.render()
            + render_value(self.value)
            + self.trailing.render()
        )
        if self.trailing_comment is not None:
            out += self.trailing_comment.render()
        if self.newline is not None:
            out += self.newline.render()
        return out


HeaderKind = Literal["table", "array"]


@dataclass(slots=True)
class TableHeaderNode:
    """A ``[name]`` or ``[[name]]`` header line.

    Layout::

        leading  '[' or '[[' inner_pre KEY inner_post ']' or ']]' trailing [# comment] \\n
    """

    leading: Trivia
    kind: HeaderKind
    inner_pre: Trivia
    key: Key
    inner_post: Trivia
    trailing: Trivia
    trailing_comment: CommentNode | None
    newline: NewlineNode | None

    def render(self) -> str:
        open_tok = "[[" if self.kind == "array" else "["
        close_tok = "]]" if self.kind == "array" else "]"
        out = (
            self.leading.render()
            + open_tok
            + self.inner_pre.render()
            + self.key.render()
            + self.inner_post.render()
            + close_tok
            + self.trailing.render()
        )
        if self.trailing_comment is not None:
            out += self.trailing_comment.render()
        if self.newline is not None:
            out += self.newline.render()
        return out


# A "section" is a header followed by zero or more KeyValueNodes that
# belong to it. The implicit pre-header section uses ``header=None``.
@dataclass(slots=True)
class SectionNode:
    header: TableHeaderNode | None
    entries: list[KeyValueNode] = field(default_factory=list)

    def render(self) -> str:
        out: list[str] = []
        if self.header is not None:
            out.append(self.header.render())
        out.extend(entry.render() for entry in self.entries)
        return "".join(out)


@dataclass(slots=True)
class DocumentNode:
    """Root of the physical CST."""

    sections: list[SectionNode] = field(default_factory=list)
    trailing_trivia: Trivia = field(default_factory=Trivia)
    """Trivia after the final structural node up to EOF."""

    def render(self) -> str:
        return "".join(s.render() for s in self.sections) + self.trailing_trivia.render()


__all__ = [
    "ArrayItem",
    "ArrayNode",
    "BoolNode",
    "CommentNode",
    "DateLikeKind",
    "DateTimeNode",
    "DocumentNode",
    "FloatNode",
    "HeaderKind",
    "InlineTableEntry",
    "InlineTableNode",
    "IntStyle",
    "IntegerNode",
    "Key",
    "KeyKind",
    "KeyPart",
    "KeyValueNode",
    "NewlineNode",
    "SectionNode",
    "StringNode",
    "StringStyle",
    "TableHeaderNode",
    "Trivia",
    "TriviaPiece",
    "ValueNode",
    "WhitespaceNode",
    "render_value",
]


# Avoid an unused-import warning for Sequence in TYPE_CHECKING block.
if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence  # noqa: F401
