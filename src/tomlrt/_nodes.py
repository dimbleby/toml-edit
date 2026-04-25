"""Physical CST node types for tomlrt.

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

Trivia ownership rule:
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
    from collections.abc import Container
    from datetime import date, datetime, time

# CST nodes use ``eq=False`` so that ``==`` / ``in`` / ``list.index`` /
# ``list.remove`` / set membership all fall back to identity. Trivia is
# preserved per occurrence and many invariants depend on locating *the*
# node we hold a reference to, not any structurally-equal twin.


# ---------------------------------------------------------------------------
# Trivia
# ---------------------------------------------------------------------------


@dataclass(slots=True, eq=False)
class WhitespaceNode:
    """Run of spaces and/or tabs (no newlines)."""

    text: str

    def render(self) -> str:
        return self.text


@dataclass(slots=True, eq=False)
class NewlineNode:
    """A single line terminator (``\\n`` or ``\\r\\n``)."""

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
"""A single trivia atom."""


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


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


KeyKind = Literal["bare", "basic", "literal"]


@dataclass(slots=True, eq=False)
class KeyPart:
    """A single dotted-key component (the part between dots)."""

    raw: str
    """Source representation including any surrounding quotes."""

    value: str
    """The decoded key string."""

    kind: KeyKind

    def render(self) -> str:
        return self.raw


@dataclass(slots=True, eq=False)
class Key:
    """A dotted key: one or more `KeyPart` separated by ``.``.

    ``separators`` carries the whitespace + ``.`` between parts. It always
    has length ``len(parts) - 1``.
    """

    parts: list[KeyPart]
    separators: list[str] = field(default_factory=list)
    path: tuple[str, ...] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        # List-comp into tuple is measurably faster than a generator
        # expression, and `Key` is built for every key in the
        # document during parsing.
        self.path = tuple([p.value for p in self.parts])

    def render(self) -> str:
        parts = self.parts
        if len(parts) == 1:
            return parts[0].render()
        out: list[str] = []
        seps = self.separators
        for i, part in enumerate(parts):
            if i:
                out.append(seps[i - 1])
            out.append(part.render())
        return "".join(out)


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


StringStyle = Literal["basic", "literal", "ml-basic", "ml-literal"]
IntStyle = Literal["dec", "hex", "oct", "bin"]


@dataclass(slots=True, eq=False)
class StringNode:
    raw: str  # including quotes
    value: str
    style: StringStyle

    def render(self) -> str:
        return self.raw


@dataclass(slots=True, eq=False)
class IntegerNode:
    raw: str
    value: int
    style: IntStyle

    def render(self) -> str:
        return self.raw


@dataclass(slots=True, eq=False)
class FloatNode:
    raw: str
    value: float

    def render(self) -> str:
        return self.raw


@dataclass(slots=True, eq=False)
class BoolNode:
    raw: str  # "true" or "false"
    value: bool

    def render(self) -> str:
        return self.raw


DateLikeKind = Literal["offset-datetime", "local-datetime", "local-date", "local-time"]


@dataclass(slots=True, eq=False)
class DateTimeNode:
    raw: str
    value: datetime | date | time
    kind: DateLikeKind

    def render(self) -> str:
        return self.raw


@dataclass(slots=True, eq=False)
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
        out = f"{self.leading.render()}{self.value.render()}{self.trailing.render()}"
        if self.has_comma:
            out += f",{self.post_comma_trivia.render()}"
        return out


@dataclass(slots=True, eq=False)
class ArrayNode:
    """Inline array literal (``[ ... ]``)."""

    items: list[ArrayItem] = field(default_factory=list)
    final_trivia: Trivia = field(default_factory=Trivia)
    """Trivia after the last item (or comma) and before the closing ``]``."""

    def render(self) -> str:
        body = "".join([item.render() for item in self.items])
        return f"[{body}{self.final_trivia.render()}]"


@dataclass(slots=True, eq=False)
class InlineTableEntry:
    """One ``key = value`` slot inside an inline table."""

    leading: Trivia
    key: Key
    pre_eq: WhitespaceNode | None
    post_eq: WhitespaceNode | None
    value: ValueNode
    trailing: Trivia
    has_comma: bool
    post_comma_trivia: Trivia

    def render(self) -> str:
        pre_eq = self.pre_eq.text if self.pre_eq is not None else ""
        post_eq = self.post_eq.text if self.post_eq is not None else ""
        out = (
            f"{self.leading.render()}{self.key.render()}{pre_eq}={post_eq}"
            f"{self.value.render()}{self.trailing.render()}"
        )
        if self.has_comma:
            out += f",{self.post_comma_trivia.render()}"
        return out


@dataclass(slots=True, eq=False)
class InlineTableNode:
    """Inline table literal (``{ a = 1, b = 2 }``)."""

    entries: list[InlineTableEntry] = field(default_factory=list)
    final_trivia: Trivia = field(default_factory=Trivia)

    def render(self) -> str:
        body = "".join([e.render() for e in self.entries])
        return f"{{{body}{self.final_trivia.render()}}}"


ValueNode = (
    StringNode
    | IntegerNode
    | FloatNode
    | BoolNode
    | DateTimeNode
    | ArrayNode
    | InlineTableNode
)


# ---------------------------------------------------------------------------
# Top-level structural nodes
# ---------------------------------------------------------------------------


@dataclass(slots=True, eq=False)
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
    pre_eq: WhitespaceNode | None
    post_eq: WhitespaceNode | None
    value: ValueNode
    trailing: WhitespaceNode | None
    trailing_comment: CommentNode | None
    newline: NewlineNode | None

    def render(self) -> str:
        pre_eq = self.pre_eq.text if self.pre_eq is not None else ""
        post_eq = self.post_eq.text if self.post_eq is not None else ""
        trailing = self.trailing.text if self.trailing is not None else ""
        out = (
            f"{self.leading.render()}{self.key.render()}{pre_eq}={post_eq}"
            f"{self.value.render()}{trailing}"
        )
        if self.trailing_comment is not None:
            out += self.trailing_comment.render()
        if self.newline is not None:
            out += self.newline.render()
        return out


HeaderKind = Literal["table", "array"]


@dataclass(slots=True, eq=False)
class TableHeaderNode:
    """A ``[name]`` or ``[[name]]`` header line.

    Layout::

        leading '[' or '[[' inner_pre KEY inner_post ']' or ']]' trailing [#cmt] \\n
    """

    leading: Trivia
    kind: HeaderKind
    inner_pre: WhitespaceNode | None
    key: Key
    inner_post: WhitespaceNode | None
    trailing: WhitespaceNode | None
    trailing_comment: CommentNode | None
    newline: NewlineNode | None

    def render(self) -> str:
        open_tok = "[[" if self.kind == "array" else "["
        close_tok = "]]" if self.kind == "array" else "]"
        inner_pre = self.inner_pre.text if self.inner_pre is not None else ""
        inner_post = self.inner_post.text if self.inner_post is not None else ""
        trailing = self.trailing.text if self.trailing is not None else ""
        out = (
            f"{self.leading.render()}{open_tok}{inner_pre}{self.key.render()}"
            f"{inner_post}{close_tok}{trailing}"
        )
        if self.trailing_comment is not None:
            out += self.trailing_comment.render()
        if self.newline is not None:
            out += self.newline.render()
        return out


# A "section" is a header followed by zero or more KeyValueNodes that
# belong to it. The implicit pre-header section uses ``header=None``.
@dataclass(slots=True, eq=False)
class SectionNode:
    header: TableHeaderNode | None
    entries: list[KeyValueNode] = field(default_factory=list)
    synthesised_placeholder: bool = False
    """True for headers spawned by an explicit empty ``Table.section({})``
    install; such headers are dropped if a child section makes them
    redundant. User-authored empty headers are preserved.
    """

    def render(self) -> str:
        head = self.header.render() if self.header is not None else ""
        return head + "".join([entry.render() for entry in self.entries])


@dataclass(slots=True, eq=False)
class DocumentNode:
    """Root of the physical CST."""

    sections: list[SectionNode] = field(default_factory=list)
    trailing_trivia: Trivia = field(default_factory=Trivia)
    """Trivia after the final structural node up to EOF."""

    def render(self) -> str:
        return (
            "".join([s.render() for s in self.sections]) + self.trailing_trivia.render()
        )

    def has_content(self) -> bool:
        """True if the document has any header or KV entry."""
        return any(s.header is not None or s.entries for s in self.sections)

    def adopt_preamble_into(self, target: Trivia) -> None:
        """Migrate parked preamble trivia onto ``target`` if the doc is empty.

        Appends a blank-line separator if the parked content includes
        comments, so the migrated text still reads as preamble.
        """
        if self.has_content():
            return
        pieces = self.trailing_trivia.pieces
        if not pieces:
            return
        moved = list(pieces)
        has_comment = any(isinstance(p, CommentNode) for p in moved)
        if has_comment:
            ends_with_blank = (
                len(moved) >= 2
                and isinstance(moved[-1], NewlineNode)
                and isinstance(moved[-2], NewlineNode)
            )
            if not ends_with_blank:
                if not (moved and isinstance(moved[-1], NewlineNode)):
                    moved.append(NewlineNode("\n"))
                moved.append(NewlineNode("\n"))
        target.pieces[:0] = moved
        self.trailing_trivia.pieces = []

    def preamble_target(self) -> Trivia:
        """Trivia block that holds the document's preamble comments.

        That's the leading trivia of the first structural node, or the
        trailing trivia of the document if there is no structural node.
        """
        for sec in self.sections:
            if sec.header is not None:
                return sec.header.leading
            if sec.entries:
                return sec.entries[0].leading
        return self.trailing_trivia

    def purge_path(self, full_path: tuple[str, ...]) -> None:
        """Remove every node addressable as ``full_path``.

        Drops sections whose header is at or under ``full_path``
        (purging children too), and drops KV entries in ancestor
        sections whose head key would steer descent into ``full_path``.

        Pure structural removal; caller is responsible for invoking
        `normalise_top_blank` once the larger operation is done.
        """
        plen = len(full_path)
        sections = self.sections
        sections[:] = [
            sec
            for sec in sections
            if not (
                sec.header is not None
                and len(sec.header.key.path) >= plen
                and sec.header.key.path[:plen] == full_path
            )
        ]
        for sec in sections:
            sec_path: tuple[str, ...] = (
                () if sec.header is None else sec.header.key.path
            )
            if len(sec_path) >= plen or full_path[: len(sec_path)] != sec_path:
                continue
            conflict_key = full_path[len(sec_path)]
            sec.entries[:] = [
                kv for kv in sec.entries if kv.key.path[0] != conflict_key
            ]

    def normalise_top_blank(self) -> None:
        """Strip leading blank-line ``NewlineNode``\\ s from the first content.

        A leading ``NewlineNode`` on the first structural node means
        "blank line above this content"; once the preceding content is
        gone it would render as a stray top-of-file blank.
        """
        for sec in self.sections:
            if sec.header is None:
                if not sec.entries:
                    continue
                pieces = sec.entries[0].leading.pieces
            else:
                pieces = sec.header.leading.pieces
            while pieces and isinstance(pieces[0], NewlineNode):
                pieces.pop(0)
            return

    def remove_entry(self, sec: SectionNode, victim: KeyValueNode) -> None:
        """Drop ``victim`` from ``sec.entries``, then renormalise.

        Bundles the structural change with `normalise_top_blank`
        so a stray top-of-file blank can't be left behind when the
        removed entry was the document's first piece of content.
        """
        sec.entries.remove(victim)
        self.normalise_top_blank()

    def remove_sections(self, victims: Container[SectionNode]) -> None:
        """Drop every section in ``victims`` from this document, then renormalise.

        Bundles the structural change with `normalise_top_blank`
        so a stray top-of-file blank can't be left behind when the
        document's first physical section is among the removed.
        """
        kept = [s for s in self.sections if s not in victims]
        if len(kept) == len(self.sections):
            return
        self.sections = kept
        self.normalise_top_blank()

    def aot_owned_range(self, aot_sec: SectionNode) -> list[SectionNode]:
        """Sections owned by this AoT entry.

        Owned = sections that come *after* ``aot_sec`` in document
        order and whose header path strictly extends this AoT's path.
        The range ends at the next [[same-path]] header or any other
        section that doesn't extend ``aot_sec``'s path.
        """
        if aot_sec.header is None:
            return []
        sections = self.sections
        # Hot path: just-appended entry, nothing follows.
        if sections and sections[-1] is aot_sec:
            return []
        aot_path = aot_sec.header.key.path
        # Identity scan from the back: ``list.index`` would use deep
        # dataclass ``==``, which is far slower than ``is`` here.
        i = -1
        for idx in range(len(sections) - 1, -1, -1):
            if sections[idx] is aot_sec:
                i = idx
                break
        if i < 0:
            return []
        owned: list[SectionNode] = []
        for j in range(i + 1, len(sections)):
            sec = sections[j]
            hdr = sec.header
            if hdr is None:
                # The synthetic root section appears only at index 0;
                # safe to stop.
                break
            hpath = hdr.key.path
            if hdr.kind == "array" and hpath == aot_path:
                break  # next AoT entry of same path — terminate
            if len(hpath) > len(aot_path) and hpath[: len(aot_path)] == aot_path:
                owned.append(sec)
            else:
                # sibling or outer section — terminate ownership
                break
        return owned

    def aot_entry_block(self, aot_sec: SectionNode) -> list[SectionNode]:
        """The full block of sections constituting one AoT entry.

        That is the ``[[..]]`` anchor followed by its owned sub-section
        run (see `aot_owned_range`). This is the unit callers
        almost always want when copying, deleting, detaching, or
        relocating an AoT entry.
        """
        return [aot_sec, *self.aot_owned_range(aot_sec)]


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
]
