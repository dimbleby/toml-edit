"""Logical Document/Table/Array/AoT view over a CST.

This module exposes the public mapping/sequence types that users
interact with. It implements both the read path and the structural
mutation API on top of the physical CST defined in :mod:`toml_edit._nodes`.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from datetime import date, datetime, time
from typing import TYPE_CHECKING, Protocol, SupportsIndex, TypeAlias, overload

from typing_extensions import override

from toml_edit._errors import TOMLEditError
from toml_edit._nodes import (
    ArrayNode,
    BoolNode,
    CommentNode,
    DateTimeNode,
    FloatNode,
    InlineTableEntry,
    InlineTableNode,
    IntegerNode,
    Key,
    KeyValueNode,
    NewlineNode,
    SectionNode,
    StringNode,
    TableHeaderNode,
    Trivia,
    WhitespaceNode,
)
from toml_edit._synthesise import make_key_part, make_keyvalue_node, make_simple_key, value_to_node

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from typing import Self

    from toml_edit._nodes import (
        ArrayItem,
        DocumentNode,
        TriviaPiece,
        ValueNode,
    )


Scalar: TypeAlias = str | int | float | bool | datetime | date | time
TomlValue: TypeAlias = "Scalar | Array | AoT | Table"


# ---------------------------------------------------------------------------
# Read-side helpers
# ---------------------------------------------------------------------------


def _scalar_value(node: ValueNode) -> Scalar | None:
    """Return the Python scalar for a value node, or None if it's a container."""
    if isinstance(node, StringNode):
        return node.value
    if isinstance(node, BoolNode):
        return node.value
    if isinstance(node, IntegerNode):
        return node.value
    if isinstance(node, FloatNode):
        return node.value
    if isinstance(node, DateTimeNode):
        return node.value
    return None


def _value_for(node: ValueNode) -> TomlValue:
    if isinstance(node, ArrayNode):
        return Array(node)
    if isinstance(node, InlineTableNode):
        return _InlineTable(node)
    scalar = _scalar_value(node)
    assert scalar is not None  # exhaustive by construction
    return scalar


def _materialise_array(node: ArrayNode) -> list[TomlValue]:
    return [_value_for(item.value) for item in node.items]


def _detect_indent(section: SectionNode) -> str:
    """Return the leading-whitespace indent used by the section's last entry."""
    if not section.entries:
        return ""
    last = section.entries[-1]
    text = last.leading.render()
    # Take everything after the final newline (the line's indent).
    nl = text.rfind("\n")
    candidate = text[nl + 1 :] if nl >= 0 else text
    if all(c in " \t" for c in candidate):
        return candidate
    return ""


def _ensure_trailing_newline(section: SectionNode) -> None:
    """Make sure the section's last entry ends with a newline.

    A parsed file's final entry may lack a newline at EOF. Before we
    append a sibling we have to terminate the previous line so the
    output is still well-formed.
    """
    if not section.entries:
        return
    last = section.entries[-1]
    if last.newline is None:
        last.newline = NewlineNode("\n")


def _starts_with_blank_line(trivia: Trivia) -> bool:
    """``True`` iff ``trivia`` begins with a bare newline.

    A bare ``NewlineNode`` at the start of an entry's leading trivia
    is how the parser records "blank line above this line": the
    previous entry ended with its own newline, then this newline
    forms the empty line, then the entry's indent / content begins.
    """
    return bool(trivia.pieces) and isinstance(trivia.pieces[0], NewlineNode)


def _gaps_uniformly_blank(leadings: Sequence[Trivia]) -> bool:
    """Decide whether existing siblings are uniformly blank-line separated.

    ``leadings`` is the leading trivia of every sibling *except the
    first* (the first has no preceding sibling, so its leading
    describes the gap to the document preamble, not an inter-sibling
    gap). Returns ``True`` only when every such gap starts with a
    blank line; mixed layouts fall back to ``False`` so we don't
    impose spacing the user may have deliberately omitted.
    """
    if not leadings:
        return False
    return all(_starts_with_blank_line(t) for t in leadings)


def _strip_comment_marker(text: str) -> str:
    """``"# foo "`` → ``"foo"``.

    Removes a leading ``#``, an optional single space, and trailing
    horizontal whitespace.
    """
    text = text.removeprefix("#")
    text = text.removeprefix(" ")
    return text.rstrip(" \t")


def _format_comment(text: str) -> str:
    """Format user text as the payload for a :class:`CommentNode`.

    Adds a leading ``#`` plus a space (unless the user already supplied
    one). Raises :class:`TOMLEditError` if ``text`` contains any line
    terminator, since comments are single-line by definition.
    """
    if "\n" in text or "\r" in text:
        msg = "comment text must not contain a line terminator"
        raise TOMLEditError(msg)
    if text.startswith("#"):
        return text
    if text == "":
        return "#"
    return "# " + text


def _indent_after_last_newline(trivia: Trivia, *, require_newline: bool = False) -> str:
    """Indent (run of spaces/tabs) after the trivia's last newline, if any.

    Returns the empty string if the run after the last newline contains
    anything other than spaces/tabs (e.g. a comment). When
    ``require_newline`` is True, also returns "" if ``trivia`` has no
    newline at all (callers that want to seed a brand-new indented line
    should fall back to a default in that case).
    """
    text = trivia.render()
    nl = text.rfind("\n")
    if nl < 0 and require_newline:
        return ""
    candidate = text[nl + 1 :] if nl >= 0 else text
    if all(c in " \t" for c in candidate):
        return candidate
    return ""


class _HasEolComment(Protocol):
    trailing: WhitespaceNode | None
    trailing_comment: CommentNode | None
    newline: NewlineNode | None


def _set_eol_comment(node: _HasEolComment, value: str | None) -> None:
    """Set or clear the trailing ``# comment`` of a KV / header node.

    Clearing also strips trailing whitespace from ``node.trailing`` so we
    don't render ``foo = 12 \\n`` after the comment goes away.
    """
    if value is None or value == "":
        node.trailing_comment = None
        node.trailing = None
        return
    if node.trailing is None:
        node.trailing = WhitespaceNode(" ")
    node.trailing_comment = CommentNode(text=_format_comment(value))
    if node.newline is None:
        node.newline = NewlineNode("\n")


def _extract_trailing_comment_block(trivia: Trivia) -> tuple[str, ...]:
    """Return the contiguous run of comment lines at the *end* of ``trivia``.

    A "comment line" is ``[WS] CommentNode NewlineNode``. The trailing
    block ends immediately before the trivia's anchoring whitespace
    (the indent of the line that follows). Earlier comment lines that
    are separated from the run by a blank line are *not* included.
    """
    pieces = trivia.pieces
    end = len(pieces)
    while end > 0 and isinstance(pieces[end - 1], WhitespaceNode):
        end -= 1
    comments: list[str] = []
    i = end
    while i >= 2:
        nl = pieces[i - 1]
        cm = pieces[i - 2]
        if not (isinstance(nl, NewlineNode) and isinstance(cm, CommentNode)):
            break
        comments.append(cm.text)
        i -= 2
        if i > 0 and isinstance(pieces[i - 1], WhitespaceNode):
            i -= 1
    comments.reverse()
    return tuple(_strip_comment_marker(c) for c in comments)


def _replace_trailing_comment_block(
    trivia: Trivia,
    lines: Sequence[str],
    indent: str,
) -> None:
    """Replace the trailing comment block in ``trivia`` with ``lines``.

    Earlier trivia (older blank-separated comments, leading whitespace)
    and the trailing whitespace anchor are preserved.
    """
    pieces = trivia.pieces
    end = len(pieces)
    tail_ws: list[TriviaPiece] = []
    while end > 0 and isinstance(pieces[end - 1], WhitespaceNode):
        tail_ws.insert(0, pieces[end - 1])
        end -= 1
    start = end
    i = end
    while i >= 2:
        nl = pieces[i - 1]
        cm = pieces[i - 2]
        if not (isinstance(nl, NewlineNode) and isinstance(cm, CommentNode)):
            break
        i -= 2
        if i > 0 and isinstance(pieces[i - 1], WhitespaceNode):
            i -= 1
        start = i
    new_pieces: list[TriviaPiece] = []
    for line in lines:
        if indent:
            new_pieces.append(WhitespaceNode(indent))
        new_pieces.append(CommentNode(text=_format_comment(line)))
        new_pieces.append(NewlineNode("\n"))
    trivia.pieces = list(pieces[:start]) + new_pieces + tail_ws


def _extract_eol_comment(trivia: Trivia) -> str | None:
    """Return the EOL comment at the *start* of ``trivia``.

    The EOL comment is the (optional WS-then-)CommentNode that appears
    before the first NewlineNode. Returns ``None`` if none.
    """
    for piece in trivia.pieces:
        if isinstance(piece, WhitespaceNode):
            continue
        if isinstance(piece, CommentNode):
            return _strip_comment_marker(piece.text)
        return None
    return None


def _replace_eol_comment(
    trivia: Trivia,
    value: str | None,
    *,
    force_newline: bool,
) -> None:
    """Set or clear the EOL comment at the *start* of ``trivia``.

    Existing EOL prefix (``[WS]? CommentNode``) is removed if present.
    If ``value`` is non-empty, ``" # value"`` is prepended. When
    ``force_newline`` is True and the trivia would otherwise lack a
    NewlineNode after the new comment, one is inserted (so following
    content doesn't end up on the same line as the comment).
    """
    pieces = trivia.pieces
    end_eol = 0
    while end_eol < len(pieces) and isinstance(pieces[end_eol], WhitespaceNode):
        end_eol += 1
    if end_eol < len(pieces) and isinstance(pieces[end_eol], CommentNode):
        end_eol += 1
    else:
        end_eol = 0
    rest = list(pieces[end_eol:])
    new_prefix: list[TriviaPiece] = []
    if value is not None and value != "":
        new_prefix.append(WhitespaceNode(" "))
        new_prefix.append(CommentNode(text=_format_comment(value)))
        if force_newline and not any(isinstance(p, NewlineNode) for p in rest):
            new_prefix.append(NewlineNode("\n"))
    trivia.pieces = new_prefix + rest


def _is_pure_whitespace(t: Trivia) -> bool:
    """True iff trivia contains only whitespace/newline pieces (no comments)."""
    return all(isinstance(p, (WhitespaceNode, NewlineNode)) for p in t.pieces)


def _trivia_render_eq(a: Trivia, b: Trivia) -> bool:
    return a.render() == b.render()


def _clone_trivia(t: Trivia) -> Trivia:
    return Trivia([deepcopy(p) for p in t.pieces])


class _SeparatorStyle:
    """Snapshot of comma-separated container spacing (arrays & inline tables).

    Layout invariants applied to a non-empty container:

    * ``items[0].leading`` carries ``open_pad``; other items' ``leading``
      is empty whitespace.
    * Each non-last item has ``has_comma=True``; ``post_comma_trivia``
      holds the inter-item separator (or a user-supplied comment).
    * The last item's comma + trailing trivia render the close-pad
      (with or without trailing comma per ``trailing_comma``).
    * The container's ``final_trivia`` is empty (close-pad lives on the
      last item to keep parser/synthesiser representations aligned).
    """

    __slots__ = ("close_pad", "inter_separator", "open_pad", "trailing_comma")

    def __init__(
        self,
        *,
        open_pad: Trivia,
        inter_separator: Trivia,
        trailing_comma: bool,
        close_pad: Trivia,
    ) -> None:
        self.open_pad = open_pad
        self.inter_separator = inter_separator
        self.trailing_comma = trailing_comma
        self.close_pad = close_pad


class _Separated(Protocol):
    """Structural protocol satisfied by ArrayItem and InlineTableEntry."""

    leading: Trivia
    trailing: Trivia
    has_comma: bool
    post_comma_trivia: Trivia


def _derive_close_pad(inter: Trivia) -> Trivia:
    """Best-guess close-pad when the source had a comment in the close slot.

    Multi-line containers (separator contains a newline) close with a
    bare newline; single-line containers close flush.
    """
    if "\n" in inter.render():
        return Trivia([NewlineNode("\n")])
    return Trivia()


def _sample_separator_style(
    items: Sequence[_Separated],
    final_trivia: Trivia,
) -> _SeparatorStyle:
    """Snapshot the spacing style of a comma-separated container.

    Works for both inline arrays and inline tables because the parser
    and the synthesiser both put inter-item whitespace in
    ``prev.post_comma_trivia`` (never ``next.leading``).
    """
    if not items:
        # Mirror a single-space internal pad (``[ ]`` / ``{ }``) into
        # both edges so first insertion preserves it as ``[ x ]``.
        pad_text = final_trivia.render()
        single_pad = (
            Trivia([WhitespaceNode(" ")]) if pad_text == " " else _clone_trivia(final_trivia)
        )
        return _SeparatorStyle(
            open_pad=_clone_trivia(single_pad),
            inter_separator=Trivia([WhitespaceNode(" ")]),
            trailing_comma=False,
            close_pad=_clone_trivia(single_pad),
        )
    open_pad = _clone_trivia(items[0].leading)
    sep: Trivia | None = None
    for it in items[:-1]:
        if it.has_comma and _is_pure_whitespace(it.post_comma_trivia):
            sep = _clone_trivia(it.post_comma_trivia)
            break
    if sep is None:
        sep = Trivia([WhitespaceNode(" ")])
    last = items[-1]
    if last.has_comma:
        trailing_comma = True
        close_pad = _clone_trivia(last.post_comma_trivia)
    else:
        trailing_comma = False
        # Close-pad combines last.trailing + final_trivia (parser and
        # synthesiser disagree on which slot holds it). If a comment is
        # in either slot it belongs to the item, not the pad — derive a
        # sensible pad from the inter-separator instead.
        combined = Trivia(
            [deepcopy(p) for p in last.trailing.pieces]
            + [deepcopy(p) for p in final_trivia.pieces],
        )
        close_pad = combined if _is_pure_whitespace(combined) else _derive_close_pad(sep)
    return _SeparatorStyle(
        open_pad=open_pad,
        inter_separator=sep,
        trailing_comma=trailing_comma,
        close_pad=close_pad,
    )


def _apply_separator_style(
    items: Sequence[_Separated],
    style: _SeparatorStyle,
    set_final_trivia: Callable[[Trivia], None],
) -> None:
    """Re-apply a sampled :class:`_SeparatorStyle` to the items.

    Items whose separator slot contains a non-whitespace token (e.g. an
    inline ``# comment``) are left alone so authoring intent is preserved.
    """
    n = len(items)
    if n == 0:
        set_final_trivia(_clone_trivia(style.close_pad))
        return
    items[0].leading = _clone_trivia(style.open_pad)
    for it in items[1:]:
        if _is_pure_whitespace(it.leading):
            it.leading = Trivia()
    set_final_trivia(Trivia())
    for i, item in enumerate(items):
        if i < n - 1:
            if not item.has_comma:
                eol = _extract_eol_comment(item.trailing)
                item.trailing = Trivia()
                item.has_comma = True
                item.post_comma_trivia = _clone_trivia(style.inter_separator)
                if eol is not None:
                    _replace_eol_comment(
                        item.post_comma_trivia,
                        eol,
                        force_newline=True,
                    )
            elif _is_pure_whitespace(item.post_comma_trivia) and not _trivia_render_eq(
                item.post_comma_trivia,
                style.inter_separator,
            ):
                item.post_comma_trivia = _clone_trivia(style.inter_separator)
            if _is_pure_whitespace(item.trailing) and item.trailing.pieces:
                item.trailing = Trivia()
        elif style.trailing_comma:
            item.has_comma = True
            if _is_pure_whitespace(item.post_comma_trivia):
                item.post_comma_trivia = _clone_trivia(style.close_pad)
            if _is_pure_whitespace(item.trailing):
                item.trailing = Trivia()
        else:
            if item.has_comma and _is_pure_whitespace(item.post_comma_trivia):
                item.has_comma = False
                item.post_comma_trivia = Trivia()
            if _is_pure_whitespace(item.trailing) and not _trivia_render_eq(
                item.trailing,
                style.close_pad,
            ):
                item.trailing = _clone_trivia(style.close_pad)


def _array_indent(arr: ArrayNode) -> str:
    """Best-guess per-item indent for inserting comment lines."""
    for item in arr.items:
        cand = _indent_after_last_newline(item.leading, require_newline=True)
        if cand:
            return cand
    for item in arr.items[:-1]:
        cand = _indent_after_last_newline(
            item.post_comma_trivia,
            require_newline=True,
        )
        if cand:
            return cand
    return " "


def _build_promoted_section(
    path: tuple[str, ...],
    inline: InlineTableNode,
    source_kv: KeyValueNode,
) -> SectionNode:
    """Build a ``[path]`` section containing ``inline``'s entries.

    Comments that lived above the original inline-table KV (its
    ``leading`` trivia) and any inline EOL comment are carried over to
    the new header so authoring intent is preserved.
    """
    parts = [make_key_part(p) for p in path]
    seps = ["."] * (len(parts) - 1)
    header = TableHeaderNode(
        leading=Trivia(list(source_kv.leading.pieces)),
        kind="table",
        inner_pre=None,
        key=Key(parts=parts, separators=seps),
        inner_post=None,
        trailing=source_kv.trailing,
        trailing_comment=source_kv.trailing_comment,
        newline=NewlineNode("\n"),
    )
    section = SectionNode(header=header, entries=[])
    for entry in inline.entries:
        section.entries.append(
            KeyValueNode(
                leading=Trivia(),
                key=entry.key,
                pre_eq=WhitespaceNode(" "),
                post_eq=WhitespaceNode(" "),
                value=entry.value,
                trailing=None,
                trailing_comment=None,
                newline=NewlineNode("\n"),
            ),
        )
    return section


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class Table(MutableMapping[str, TomlValue]):
    """A logical TOML table.

    All mapping flavours in toml-edit (top-level document, standard
    table, inline table, and the synthetic mappings spawned by dotted
    keys) inherit from :class:`Table`, so values typed as ``Table``
    cover every nested mapping you can encounter while walking a
    document.

    Subclasses provide ``_items()`` which yields ``(key, value)`` pairs
    in document order. The read path is wired up; mutation raises
    :class:`toml_edit.TOMLEditError` until the next implementation phase.
    """

    __slots__ = ()

    def _items(self) -> Iterator[tuple[str, TomlValue]]:  # pragma: no cover
        raise NotImplementedError

    @override
    def __iter__(self) -> Iterator[str]:
        for k, _ in self._items():
            yield k

    @override
    def __len__(self) -> int:
        return sum(1 for _ in self._items())

    @override
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return any(k == key for k, _ in self._items())

    @override
    def __getitem__(self, key: str) -> TomlValue:
        for k, v in self._items():
            if k == key:
                return v
        raise KeyError(key)

    @override
    def __setitem__(self, key: str, value: object) -> None:
        self._set_value(key, value)

    @override
    def __delitem__(self, key: str) -> None:
        self._delete_value(key)

    # Subclasses override these.
    def _set_value(self, key: str, value: object) -> None:  # noqa: ARG002, pragma: no cover
        msg = "this table flavour does not support mutation"
        raise TOMLEditError(msg)

    def _delete_value(self, key: str) -> None:  # noqa: ARG002, pragma: no cover
        msg = "this table flavour does not support mutation"
        raise TOMLEditError(msg)

    @override
    def __repr__(self) -> str:
        body = ", ".join(f"{k!r}: {v!r}" for k, v in self._items())
        return f"{type(self).__name__}({{{body}}})"

    # ------------------------------------------------------------------
    # Typed accessors. These are convenience views over ``__getitem__``
    # that narrow the return type so callers can chain into the
    # comment/header API without writing ``cast`` or ``isinstance``.
    # ------------------------------------------------------------------

    def table(self, key: str) -> Table:
        """Return ``self[key]`` typed as a :class:`Table`.

        Raises :class:`TypeError` if the value is not a table (inline or
        standard).
        """
        value = self[key]
        if not isinstance(value, Table):
            msg = f"{key!r} is a {type(value).__name__}, not a Table"
            raise TypeError(msg)
        return value

    def array(self, key: str) -> Array:
        """Return ``self[key]`` typed as an :class:`Array`.

        Raises :class:`TypeError` if the value is not an inline array.
        """
        value = self[key]
        if not isinstance(value, Array):
            msg = f"{key!r} is a {type(value).__name__}, not an Array"
            raise TypeError(msg)
        return value

    def aot(self, key: str) -> AoT:
        """Return ``self[key]`` typed as an :class:`AoT` (array of tables).

        Raises :class:`TypeError` if the value is not an array of tables.
        """
        value = self[key]
        if not isinstance(value, AoT):
            msg = f"{key!r} is a {type(value).__name__}, not an AoT"
            raise TypeError(msg)
        return value

    # ------------------------------------------------------------------
    # Metadata side-channels (default raises; concrete subclasses override).
    # ------------------------------------------------------------------

    @property
    def comments(self) -> MutableMapping[str, str]:
        """Live mapping of ``key -> end-of-line comment text``.

        Only keys that currently carry a comment are present; assigning
        ``""`` or deleting a key removes its comment. Reads return the
        comment text without the leading ``#`` or surrounding whitespace.
        """
        msg = "this table flavour does not support the comment API"
        raise TOMLEditError(msg)

    @property
    def leading_comments(self) -> MutableMapping[str, tuple[str, ...]]:
        """Live mapping of ``key -> tuple of comment lines above it``.

        Only keys with a non-empty leading comment block are present.
        Assigning an empty tuple or deleting a key removes the block.
        """
        msg = "this table flavour does not support the comment API"
        raise TOMLEditError(msg)

    @property
    def header_comment(self) -> str | None:
        """End-of-line comment on this table's ``[name]`` / ``[[name]]`` line.

        ``None`` means the header has no trailing comment. Setting
        ``None`` or ``""`` removes any existing comment. Raises
        :class:`TOMLEditError` for the top-level :class:`Document`,
        for inline tables, and for any logical table that exists only
        through implicit parents (no physical header in source).

        For tables declared via multiple discontiguous ``[name]``
        sections, this refers to the *first* such header.
        """
        msg = "this table flavour does not support the header comment API"
        raise TOMLEditError(msg)

    @header_comment.setter
    def header_comment(self, value: str | None) -> None:  # noqa: ARG002
        msg = "this table flavour does not support the header comment API"
        raise TOMLEditError(msg)

    @header_comment.deleter
    def header_comment(self) -> None:
        msg = "this table flavour does not support the header comment API"
        raise TOMLEditError(msg)

    @property
    def header_leading_comments(self) -> tuple[str, ...]:
        """Comment lines immediately above this table's header.

        Returns the contiguous block of ``# ...`` lines ending right
        above the ``[name]`` / ``[[name]]`` line. Earlier blank-line
        separated comments are *not* included. Assigning an empty
        tuple removes the block. Raises like :attr:`header_comment`.
        """
        msg = "this table flavour does not support the header comment API"
        raise TOMLEditError(msg)

    @header_leading_comments.setter
    def header_leading_comments(self, value: Sequence[str]) -> None:  # noqa: ARG002
        msg = "this table flavour does not support the header comment API"
        raise TOMLEditError(msg)

    @header_leading_comments.deleter
    def header_leading_comments(self) -> None:
        msg = "this table flavour does not support the header comment API"
        raise TOMLEditError(msg)

    def promote_inline(self, key: str) -> Table:  # noqa: ARG002
        """Promote an inline-table-valued ``key`` to a standard table.

        After promotion the entry is rendered as a separate
        ``[parent.key]`` section, allowing comments and dotted-key
        expansions on its members.
        """
        msg = "this table flavour does not support inline-table promotion"
        raise TOMLEditError(msg)


class _InlineTable(Table):
    """Mapping view over an :class:`InlineTableNode`."""

    __slots__ = ("_node", "_post_eq_text", "_pre_eq_text", "_style")

    def __init__(self, node: InlineTableNode) -> None:
        self._node = node
        self._style = _sample_separator_style(node.entries, node.final_trivia)
        # ``=``-padding is per-entry, not a separator concern. Store as raw
        # text so each new entry gets a fresh WhitespaceNode without aliasing.
        if node.entries:
            self._pre_eq_text = node.entries[0].pre_eq.text if node.entries[0].pre_eq else ""
            self._post_eq_text = node.entries[0].post_eq.text if node.entries[0].post_eq else ""
        else:
            self._pre_eq_text = " "
            self._post_eq_text = " "

    @override
    def _items(self) -> Iterator[tuple[str, TomlValue]]:
        groups: dict[str, list[tuple[tuple[str, ...], ValueNode]]] = {}
        order: list[str] = []
        for entry in self._node.entries:
            head = entry.key.path[0]
            if head not in groups:
                groups[head] = []
                order.append(head)
            groups[head].append((entry.key.path, entry.value))
        for head in order:
            entries = groups[head]
            if len(entries) == 1 and len(entries[0][0]) == 1:
                yield head, _value_for(entries[0][1])
            else:
                yield head, _DottedSubTable(entries, depth=1)

    def _find_entry(self, key: str) -> InlineTableEntry | None:
        for entry in self._node.entries:
            if len(entry.key.path) == 1 and entry.key.path[0] == key:
                return entry
        return None

    def _set_final_trivia(self, t: Trivia) -> None:
        self._node.final_trivia = t

    @override
    def _set_value(self, key: str, value: object) -> None:
        # Reject conflict with dotted entries.
        for entry in self._node.entries:
            if entry.key.path[0] == key and len(entry.key.path) > 1:
                msg = (
                    f"cannot assign to {key!r} inside an inline table: "
                    "conflicts with an existing dotted-key entry."
                )
                raise TOMLEditError(msg)
        existing = self._find_entry(key)
        if existing is not None:
            existing.value = value_to_node(value)
            return
        new_entry = InlineTableEntry(
            leading=Trivia(),
            key=make_simple_key(key),
            pre_eq=WhitespaceNode(self._pre_eq_text) if self._pre_eq_text else None,
            post_eq=WhitespaceNode(self._post_eq_text) if self._post_eq_text else None,
            value=value_to_node(value),
            trailing=Trivia(),
            has_comma=False,
            post_comma_trivia=Trivia(),
        )
        self._node.entries.append(new_entry)
        _apply_separator_style(self._node.entries, self._style, self._set_final_trivia)

    @override
    def _delete_value(self, key: str) -> None:
        existing = self._find_entry(key)
        if existing is None:
            for entry in self._node.entries:
                if entry.key.path[0] == key:
                    msg = f"cannot delete dotted-key entry {key!r} from inline table"
                    raise TOMLEditError(msg)
            raise KeyError(key)
        idx = self._node.entries.index(existing)
        self._node.entries.pop(idx)
        _apply_separator_style(self._node.entries, self._style, self._set_final_trivia)


class _DottedSubTable(Table):
    """Synthetic table aggregating dotted-key entries.

    Used by both inline tables (whose entries carry full ``(path, value)``
    pairs) and standard tables (which feed in ``(kv.key.path, kv.value)``).
    The parser guarantees that, for valid documents, every path here has
    length ``>= depth + 1`` — paths of exactly ``depth + 1`` are leaves
    at this level; longer paths recurse one level deeper.
    """

    __slots__ = ("_depth", "_entries")

    def __init__(
        self,
        entries: list[tuple[tuple[str, ...], ValueNode]],
        *,
        depth: int,
    ) -> None:
        self._entries = entries
        self._depth = depth

    @override
    def _items(self) -> Iterator[tuple[str, TomlValue]]:
        groups: dict[str, list[tuple[tuple[str, ...], ValueNode]]] = {}
        order: list[str] = []
        terminals: dict[str, ValueNode] = {}
        for path, value in self._entries:
            head = path[self._depth]
            if len(path) == self._depth + 1:
                terminals[head] = value
                if head not in order:
                    order.append(head)
                continue
            if head not in groups:
                groups[head] = []
                if head not in order:
                    order.append(head)
            groups[head].append((path, value))
        for head in order:
            if head in terminals:
                yield head, _value_for(terminals[head])
            else:
                yield head, _DottedSubTable(groups[head], depth=self._depth + 1)


class _StdTable(Table):
    """Standard TOML table view: aggregates physical sections by path."""

    __slots__ = ("_doc_view", "_extra_kvs", "_owned_scope", "_path", "_pinned_sections")

    def __init__(
        self,
        doc_view: _DocumentView,
        path: tuple[str, ...],
        *,
        sections: list[SectionNode] | None = None,
        owned_scope: list[SectionNode] | None = None,
        extra_kvs: list[tuple[tuple[str, ...], KeyValueNode]] | None = None,
    ) -> None:
        self._doc_view = doc_view
        self._path = path
        self._pinned_sections = sections
        self._owned_scope = owned_scope
        self._extra_kvs = extra_kvs

    @override
    def _items(self) -> Iterator[tuple[str, TomlValue]]:
        return self._doc_view.iter_table(
            self._path,
            pinned_sections=self._pinned_sections,
            owned_scope=self._owned_scope,
            extra_kvs=self._extra_kvs,
        )

    def _direct_sections(self) -> list[SectionNode]:
        if self._pinned_sections is not None:
            return self._pinned_sections
        if self._owned_scope is not None:
            path = self._path
            return [
                s
                for s in self._owned_scope
                if s.header is not None and s.header.kind == "table" and s.header.key.path == path
            ]
        return self._doc_view._direct_sections(self._path)  # noqa: SLF001

    def _classify(self, key: str) -> tuple[str, object]:
        """Classify a key for mutation purposes.

        Returns one of:
            ("direct", KeyValueNode)         - a single-part scalar/value entry
            ("dotted", None)                 - dotted-key prefix (e.g. b.c=...)
            ("table", None)                  - child standard table [self.path.key]
            ("aot", None)                    - child AoT [[self.path.key]]
            ("absent", None)
        """
        for sec in self._direct_sections():
            for kv in sec.entries:
                if kv.key.path[0] == key:
                    if len(kv.key.path) == 1:
                        return ("direct", kv)
                    return ("dotted", None)
        child = (*self._path, key)
        if self._doc_view._aot_sections(child):  # noqa: SLF001
            return ("aot", None)
        for sec in self._doc_view._node.sections:  # noqa: SLF001
            hdr = sec.header
            if hdr is not None and hdr.kind == "table":
                hpath = hdr.key.path
                if len(hpath) >= len(child) and hpath[: len(child)] == child:
                    return ("table", None)
        return ("absent", None)

    @override
    def _set_value(self, key: str, value: object) -> None:
        # Special case: assigning an AoT (or list of dicts targeted as AoT)
        # is a *structural* edit, not a value assignment.
        if isinstance(value, AoT):
            self._set_aot_value(key, value)
            return

        kind, payload = self._classify(key)
        if kind == "direct":
            assert isinstance(payload, KeyValueNode)
            payload.value = value_to_node(value)
            return
        if kind in ("dotted", "table", "aot"):
            msg = (
                f"cannot assign to {key!r}: existing structure conflicts "
                f"({kind}). Mutate the nested table or remove it first."
            )
            raise TOMLEditError(msg)
        sections = self._direct_sections()
        if not sections:
            sections = [self._ensure_section()]
        target = sections[-1]
        indent = _detect_indent(target)
        new_kv = make_keyvalue_node(key, value, indent=indent)
        if _gaps_uniformly_blank([kv.leading for kv in target.entries[1:]]):
            new_kv.leading.pieces.insert(0, NewlineNode("\n"))
        _ensure_trailing_newline(target)
        target.entries.append(new_kv)
        # Top-level only: if this assignment is into the implicit
        # pre-header section and a ``[table]`` follows, ensure a blank
        # line separates the new key from that header.
        if self._path == () and target.header is None:
            doc_node = self._doc_view._node  # noqa: SLF001
            try:
                idx = doc_node.sections.index(target)
            except ValueError:  # pragma: no cover - defensive
                return
            if idx + 1 < len(doc_node.sections):
                next_header = doc_node.sections[idx + 1].header
                if next_header is not None and not next_header.leading.render().startswith(
                    "\n",
                ):
                    next_header.leading.pieces.insert(0, NewlineNode("\n"))

    def _set_aot_value(self, key: str, value: AoT) -> None:
        """Assign a (possibly cross-document) AoT to ``key``.

        Each source entry's ``[[..]]`` section is deep-cloned and its
        header path rewritten to ``(*self._path, key)``. New sections
        are appended to the document.
        """
        kind, _ = self._classify(key)
        if kind != "absent":
            msg = (
                f"cannot assign array-of-tables to {key!r}: name is "
                f"already in use ({kind}). Remove it first."
            )
            raise TOMLEditError(msg)
        new_path = (*self._path, key)
        new_parts = [make_key_part(p) for p in new_path]
        new_seps = ["."] * (len(new_parts) - 1)
        # Source sections to clone, in source-document order:
        src_own = value._own_sections()  # noqa: SLF001
        doc_node = self._doc_view._node  # noqa: SLF001
        for src_sec in src_own:
            cloned = deepcopy(src_sec)
            assert cloned.header is not None
            cloned.header.key = Key(parts=list(new_parts), separators=list(new_seps))
            doc_node.sections.append(cloned)

    @override
    def _delete_value(self, key: str) -> None:
        kind, payload = self._classify(key)
        if kind == "direct":
            assert isinstance(payload, KeyValueNode)
            for sec in self._direct_sections():
                if payload in sec.entries:
                    sec.entries.remove(payload)
                    return
            raise KeyError(key)
        if kind == "absent":
            raise KeyError(key)
        msg = (
            f"cannot delete {key!r}: deleting child tables and "
            "dotted-key subtrees is not supported."
        )
        raise TOMLEditError(msg)

    def _ensure_section(self) -> SectionNode:
        """Create an implicit pre-header section for the document root.

        Only valid for the top-level Document; sub-tables created via
        ``__setitem__`` raise above.
        """
        if self._path != ():  # pragma: no cover - defensive
            msg = (
                f"no [{'.'.join(self._path)}] section exists; creating "
                "new sub-tables via assignment is not supported."
            )
            raise TOMLEditError(msg)
        doc_node = self._doc_view._node  # noqa: SLF001
        new_sec = SectionNode(header=None, entries=[])
        # Ensure a blank line precedes the next section's header so the
        # newly-inserted top-level keys aren't visually glued to it.
        if doc_node.sections and doc_node.sections[0].header is not None:
            next_header = doc_node.sections[0].header
            leading_text = next_header.leading.render()
            if not leading_text.startswith("\n"):
                next_header.leading.pieces.insert(0, NewlineNode("\n"))
        doc_node.sections.insert(0, new_sec)
        return new_sec

    # ------------------------------------------------------------------
    # Comment API (live mapping side-channels)
    # ------------------------------------------------------------------

    def _find_direct_kv(self, key: str) -> tuple[SectionNode, KeyValueNode]:
        """Return the section + KV node binding ``key`` as a single segment.

        Raises :class:`KeyError` when ``key`` is absent or the binding is
        a child table / dotted-key prefix rather than a simple
        ``key = value`` line.
        """
        for sec in self._direct_sections():
            for kv in sec.entries:
                if len(kv.key.path) == 1 and kv.key.path[0] == key:
                    return sec, kv
        raise KeyError(key)

    @property
    @override
    def comments(self) -> MutableMapping[str, str]:
        return _TableCommentsView(self)

    @property
    @override
    def leading_comments(self) -> MutableMapping[str, tuple[str, ...]]:
        return _TableLeadingCommentsView(self)

    def _first_header(self) -> TableHeaderNode:
        for sec in self._direct_sections():
            if sec.header is not None:
                return sec.header
        msg = (
            f"table {'.'.join(self._path) or '<root>'!r} has no physical "
            "header (it exists only through implicit parents); the "
            "header comment API is unavailable"
        )
        raise TOMLEditError(msg)

    @property  # type: ignore[explicit-override]
    @override
    def header_comment(self) -> str | None:
        header = self._first_header()
        if header.trailing_comment is None:
            return None
        return _strip_comment_marker(header.trailing_comment.text)

    @header_comment.setter
    def header_comment(self, value: str | None) -> None:
        _set_eol_comment(self._first_header(), value)

    @header_comment.deleter
    def header_comment(self) -> None:
        _set_eol_comment(self._first_header(), None)

    @property  # type: ignore[explicit-override]
    @override
    def header_leading_comments(self) -> tuple[str, ...]:
        header = self._first_header()
        return _extract_trailing_comment_block(header.leading)

    @header_leading_comments.setter
    def header_leading_comments(self, value: Sequence[str]) -> None:
        header = self._first_header()
        _replace_trailing_comment_block(
            header.leading,
            value,
            _indent_after_last_newline(header.leading),
        )

    @header_leading_comments.deleter
    def header_leading_comments(self) -> None:
        header = self._first_header()
        _replace_trailing_comment_block(
            header.leading,
            (),
            _indent_after_last_newline(header.leading),
        )

    @override
    def promote_inline(self, key: str) -> Table:
        sec, kv = self._find_direct_kv(key)
        if not isinstance(kv.value, InlineTableNode):
            msg = f"{key!r} is not an inline table; nothing to promote"
            raise TOMLEditError(msg)
        inline = kv.value
        child_path = (*self._path, key)
        # Refuse if a [child_path] section already exists in the document
        # (defensive: the parser blocks any source where this would arise,
        # and the mutation API also refuses to create the conflicting
        # state, so this branch only fires under direct CST manipulation).
        for existing in self._doc_view._node.sections:  # noqa: SLF001
            hdr = existing.header
            if hdr is not None and hdr.key.path == child_path:  # pragma: no cover
                msg = f"cannot promote {key!r}: a [{'.'.join(child_path)}] section already exists"
                raise TOMLEditError(msg)
        new_sec = _build_promoted_section(child_path, inline, kv)
        # Remove the inline KV from its host section.
        sec.entries.remove(kv)
        # Insert the promoted section after the parent's last direct
        # section (or at end of document if the parent has none).
        sections = self._doc_view._node.sections  # noqa: SLF001
        parent_secs = self._direct_sections()
        if parent_secs:
            anchor = parent_secs[-1]
            anchor_idx = next(
                (i for i, s in enumerate(sections) if s is anchor),
                len(sections) - 1,
            )
            sections.insert(anchor_idx + 1, new_sec)
        else:
            sections.append(new_sec)
        return _StdTable(self._doc_view, child_path)


class Document(_StdTable):
    """Top-level TOML document. Subclass of :class:`Table`."""

    __slots__ = ("_node",)

    def __init__(self, node: DocumentNode) -> None:
        view = _DocumentView(node)
        self._node = node
        super().__init__(view, ())

    @property
    def cst(self) -> DocumentNode:
        """The underlying physical CST (intended for tooling/debugging)."""
        return self._node

    def render(self) -> str:
        return self._node.render()


# ---------------------------------------------------------------------------
# Comment views
# ---------------------------------------------------------------------------


class _TableCommentsView(MutableMapping[str, str]):
    """Live mapping from key name to end-of-line comment text.

    Backed by a :class:`_StdTable`; a key is "present" iff its
    ``KeyValueNode`` currently carries a ``trailing_comment``. Setting
    an empty string removes the comment, mirroring ``del``.
    """

    __slots__ = ("_table",)

    def __init__(self, table: _StdTable) -> None:
        self._table = table

    def _commented_kvs(self) -> Iterator[tuple[str, KeyValueNode]]:
        for sec in self._table._direct_sections():  # noqa: SLF001
            for kv in sec.entries:
                if kv.trailing_comment is not None and len(kv.key.path) == 1:
                    yield kv.key.path[0], kv

    @override
    def __getitem__(self, key: str) -> str:
        _, kv = self._table._find_direct_kv(key)  # noqa: SLF001
        if kv.trailing_comment is None:
            raise KeyError(key)
        return _strip_comment_marker(kv.trailing_comment.text)

    @override
    def __setitem__(self, key: str, value: str) -> None:
        _, kv = self._table._find_direct_kv(key)  # noqa: SLF001
        _set_eol_comment(kv, value if value != "" else None)

    @override
    def __delitem__(self, key: str) -> None:
        _, kv = self._table._find_direct_kv(key)  # noqa: SLF001
        if kv.trailing_comment is None:
            raise KeyError(key)
        _set_eol_comment(kv, None)

    @override
    def __iter__(self) -> Iterator[str]:
        for k, _ in self._commented_kvs():
            yield k

    @override
    def __len__(self) -> int:
        return sum(1 for _ in self._commented_kvs())

    @override
    def __repr__(self) -> str:
        body = ", ".join(f"{k!r}: {v!r}" for k, v in self.items())
        return f"{type(self).__name__}({{{body}}})"


class _TableLeadingCommentsView(MutableMapping[str, "tuple[str, ...]"]):
    """Live mapping from key name to its leading comment block.

    The block is the contiguous run of ``# ...`` lines immediately
    above the entry's source line (with their newlines), as stored in
    the entry's ``leading`` trivia. A key is "present" iff that run is
    non-empty.
    """

    __slots__ = ("_table",)

    def __init__(self, table: _StdTable) -> None:
        self._table = table

    @override
    def __getitem__(self, key: str) -> tuple[str, ...]:
        _, kv = self._table._find_direct_kv(key)  # noqa: SLF001
        block = _extract_trailing_comment_block(kv.leading)
        if not block:
            raise KeyError(key)
        return block

    @override
    def __setitem__(self, key: str, value: Sequence[str]) -> None:
        _, kv = self._table._find_direct_kv(key)  # noqa: SLF001
        _replace_trailing_comment_block(
            kv.leading,
            value,
            _indent_after_last_newline(kv.leading),
        )

    @override
    def __delitem__(self, key: str) -> None:
        _, kv = self._table._find_direct_kv(key)  # noqa: SLF001
        if not _extract_trailing_comment_block(kv.leading):
            raise KeyError(key)
        self[key] = ()

    @override
    def __iter__(self) -> Iterator[str]:
        for sec in self._table._direct_sections():  # noqa: SLF001
            for kv in sec.entries:
                if len(kv.key.path) == 1 and _extract_trailing_comment_block(kv.leading):
                    yield kv.key.path[0]

    @override
    def __len__(self) -> int:
        return sum(1 for _ in self)

    @override
    def __repr__(self) -> str:
        body = ", ".join(f"{k!r}: {list(v)!r}" for k, v in self.items())
        return f"{type(self).__name__}({{{body}}})"


class _ArrayCommentsView(MutableMapping[int, str]):
    """Live mapping from array index to that item's end-of-line comment.

    Backed by an :class:`Array`. An index is "present" iff the
    corresponding item currently carries an EOL comment in its
    ``post_comma_trivia`` (when the item has a trailing comma) or
    its ``trailing`` trivia (last item, no trailing comma).
    """

    __slots__ = ("_array",)

    def __init__(self, array: Array) -> None:
        self._array = array

    def _check_index(self, key: object) -> int:
        if not isinstance(key, int):
            msg = f"Array.comments index must be int, got {type(key).__name__}"
            raise TypeError(msg)
        n = len(self._array._node.items)  # noqa: SLF001
        if not 0 <= key < n:
            raise KeyError(key)
        return key

    def _read_eol(self, i: int) -> str | None:
        item = self._array._node.items[i]  # noqa: SLF001
        if item.has_comma:
            c = _extract_eol_comment(item.post_comma_trivia)
            if c is not None:
                return c
        return _extract_eol_comment(item.trailing)

    @override
    def __getitem__(self, key: int) -> str:
        i = self._check_index(key)
        c = self._read_eol(i)
        if c is None:
            raise KeyError(key)
        return c

    @override
    def __setitem__(self, key: int, value: str) -> None:
        i = self._check_index(key)
        items = self._array._node.items  # noqa: SLF001
        item = items[i]
        is_last = i == len(items) - 1
        if value == "":
            del self[key]
            return
        # Pick the slot that will carry the comment. Mid-array items
        # (and last items with a synthesized trailing comma) write into
        # post_comma_trivia; the only case left -- last item with no
        # comma -- writes into the value's trailing.
        if not is_last or item.has_comma:
            if not item.post_comma_trivia.pieces:
                item.post_comma_trivia = Trivia([WhitespaceNode(" ")])
            slot = item.post_comma_trivia
        else:
            slot = item.trailing
        _replace_eol_comment(slot, value, force_newline=True)
        if is_last:
            # Comment runs to EOL: `]` must drop to the next line.
            self._ensure_array_break_before_close()
        else:
            # The next item now starts on a fresh line: give it an
            # indent that matches its siblings.
            indent = _array_indent(self._array._node)  # noqa: SLF001
            next_item = items[i + 1]
            if indent and "\n" not in next_item.leading.render():
                next_item.leading = Trivia([WhitespaceNode(indent)])

    def _ensure_array_break_before_close(self) -> None:
        """Force ``]`` onto a new line when the last item carries an EOL
        comment (the comment otherwise swallows the closing bracket)."""
        node = self._array._node  # noqa: SLF001
        ft = node.final_trivia
        if "\n" in ft.render():
            return
        last = node.items[-1] if node.items else None
        if last is not None:
            preceding = last.post_comma_trivia if last.has_comma else last.trailing
            if preceding.pieces and isinstance(
                preceding.pieces[-1],
                NewlineNode,
            ):
                return
        # Strip only the leading WS so we don't render `\n   ]`.
        pieces = list(ft.pieces)
        while pieces and isinstance(pieces[0], WhitespaceNode):
            pieces.pop(0)
        ft.pieces = [NewlineNode("\n"), *pieces]

    @override
    def __delitem__(self, key: int) -> None:
        i = self._check_index(key)
        item = self._array._node.items[i]  # noqa: SLF001
        had = False
        if _extract_eol_comment(item.post_comma_trivia) is not None:
            _replace_eol_comment(item.post_comma_trivia, None, force_newline=False)
            had = True
        if _extract_eol_comment(item.trailing) is not None:
            _replace_eol_comment(item.trailing, None, force_newline=False)
            had = True
        if not had:
            raise KeyError(key)

    @override
    def __iter__(self) -> Iterator[int]:
        for i in range(len(self._array._node.items)):  # noqa: SLF001
            if self._read_eol(i) is not None:
                yield i

    @override
    def __len__(self) -> int:
        return sum(1 for _ in self)

    @override
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, int):
            return False
        n = len(self._array._node.items)  # noqa: SLF001
        if not 0 <= key < n:
            return False
        return self._read_eol(key) is not None

    @override
    def __repr__(self) -> str:
        body = ", ".join(f"{k}: {v!r}" for k, v in self.items())
        return f"{type(self).__name__}({{{body}}})"


class _ArrayLeadingCommentsView(MutableMapping[int, "tuple[str, ...]"]):
    """Live mapping from array index to its leading comment block.

    For item 0, the block is extracted from ``items[0].leading`` (the
    trivia between ``[`` and the first value). For item i > 0, it is
    extracted from ``items[i-1].post_comma_trivia`` (specifically, the
    contiguous trailing run of comment lines, ignoring any EOL portion
    that belongs to item i-1).
    """

    __slots__ = ("_array",)

    def __init__(self, array: Array) -> None:
        self._array = array

    def _check_index(self, key: object) -> int:
        if not isinstance(key, int):
            msg = f"Array.leading_comments index must be int, got {type(key).__name__}"
            raise TypeError(msg)
        n = len(self._array._node.items)  # noqa: SLF001
        if not 0 <= key < n:
            raise KeyError(key)
        return key

    def _trivia_for(self, i: int) -> Trivia:
        items = self._array._node.items  # noqa: SLF001
        if i == 0:
            return items[0].leading
        return items[i - 1].post_comma_trivia

    @override
    def __getitem__(self, key: int) -> tuple[str, ...]:
        i = self._check_index(key)
        block = _extract_trailing_comment_block(self._trivia_for(i))
        if not block:
            raise KeyError(key)
        return block

    @override
    def __setitem__(self, key: int, value: Sequence[str]) -> None:
        i = self._check_index(key)
        trivia = self._trivia_for(i)
        indent = _array_indent(self._array._node)  # noqa: SLF001
        # Ensure the trivia ends with a newline+indent anchor so the
        # comment block lands on its own line(s) before the next value.
        if value and not any(isinstance(p, NewlineNode) for p in trivia.pieces):
            while trivia.pieces and isinstance(
                trivia.pieces[-1],
                WhitespaceNode,
            ):
                trivia.pieces.pop()
            trivia.pieces.append(NewlineNode("\n"))
            if indent:
                trivia.pieces.append(WhitespaceNode(indent))
        _replace_trailing_comment_block(trivia, value, indent)

    @override
    def __delitem__(self, key: int) -> None:
        i = self._check_index(key)
        trivia = self._trivia_for(i)
        if not _extract_trailing_comment_block(trivia):
            raise KeyError(key)
        _replace_trailing_comment_block(
            trivia,
            (),
            _array_indent(self._array._node),  # noqa: SLF001
        )

    @override
    def __iter__(self) -> Iterator[int]:
        for i in range(len(self._array._node.items)):  # noqa: SLF001
            if _extract_trailing_comment_block(self._trivia_for(i)):
                yield i

    @override
    def __len__(self) -> int:
        return sum(1 for _ in self)

    @override
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, int):
            return False
        n = len(self._array._node.items)  # noqa: SLF001
        if not 0 <= key < n:
            return False
        return bool(_extract_trailing_comment_block(self._trivia_for(key)))

    @override
    def __repr__(self) -> str:
        body = ", ".join(f"{k}: {list(v)!r}" for k, v in self.items())
        return f"{type(self).__name__}({{{body}}})"


# ---------------------------------------------------------------------------
# Array (inline) and AoT (array of tables)
# ---------------------------------------------------------------------------


class Array(list[TomlValue]):
    """Inline TOML array exposed as a real :class:`list`.

    Every standard list mutator is overridden so the underlying CST
    stays in sync. Existing handles to nested ``Array``/``Table`` values
    that were *not* removed remain valid; handles to removed/replaced
    elements become detached.
    """

    __slots__ = ("_node", "_style")

    def __init__(self, node: ArrayNode) -> None:
        self._node = node
        self._style = _sample_separator_style(node.items, node.final_trivia)
        super().__init__(_materialise_array(node))

    # ------------------------------------------------------------------
    # CST <-> list synchronisation helpers
    # ------------------------------------------------------------------

    def _resync(self) -> None:
        """Rebuild the public list from the CST after a structural change."""
        list.clear(self)
        list.extend(self, _materialise_array(self._node))

    def _set_final_trivia(self, t: Trivia) -> None:
        self._node.final_trivia = t

    def _rebuild_separators(self) -> None:
        _apply_separator_style(self._node.items, self._style, self._set_final_trivia)

    @staticmethod
    def _make_item(value: TomlValue, *, with_comma: bool) -> ArrayItem:
        from toml_edit._nodes import ArrayItem  # noqa: PLC0415

        return ArrayItem(
            leading=Trivia(),
            value=value_to_node(value),
            trailing=Trivia(),
            has_comma=with_comma,
            post_comma_trivia=Trivia(),
        )

    @property
    def comments(self) -> MutableMapping[int, str]:
        """Live mapping of ``index -> end-of-line comment text``.

        Only items that currently carry an EOL comment are present.
        Setting ``""`` removes the comment, mirroring ``del``.
        Reads return the comment text without the leading ``#`` or
        surrounding whitespace.
        """
        return _ArrayCommentsView(self)

    @property
    def leading_comments(self) -> MutableMapping[int, tuple[str, ...]]:
        """Live mapping of ``index -> tuple of comment lines above item``.

        For item 0 the lines come from inside the array opening, before
        the first value. For item i > 0 they come from the trivia
        between item i-1's separator and item i's value.
        """
        return _ArrayLeadingCommentsView(self)

    # ------------------------------------------------------------------
    # Mutators (override every one)
    # ------------------------------------------------------------------

    @override
    def append(self, value: TomlValue) -> None:
        self._node.items.append(self._make_item(value, with_comma=False))
        self._rebuild_separators()
        self._resync()

    @override
    def extend(self, values: Iterable[TomlValue]) -> None:
        for v in list(values):
            self._node.items.append(self._make_item(v, with_comma=False))
        self._rebuild_separators()
        self._resync()

    @override
    def insert(self, index: SupportsIndex, value: TomlValue) -> None:
        self._node.items.insert(operator.index(index), self._make_item(value, with_comma=False))
        self._rebuild_separators()
        self._resync()

    @overload
    def __setitem__(self, index: SupportsIndex, value: TomlValue) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[TomlValue]) -> None: ...
    @override
    def __setitem__(
        self,
        index: SupportsIndex | slice,
        value: TomlValue | Iterable[TomlValue],
    ) -> None:
        if isinstance(index, slice):
            assert not isinstance(value, (str, bytes))
            new_items = [
                self._make_item(v, with_comma=False)
                for v in list(value)  # type: ignore[arg-type]
            ]
            self._node.items[index] = new_items
        else:
            i = operator.index(index)
            self._node.items[i].value = value_to_node(value)
        self._rebuild_separators()
        self._resync()

    @override
    def __delitem__(self, index: SupportsIndex | slice) -> None:
        if isinstance(index, slice):
            del self._node.items[index]
        else:
            del self._node.items[operator.index(index)]
        self._rebuild_separators()
        self._resync()

    @override
    def pop(self, index: SupportsIndex = -1) -> TomlValue:
        item = self._node.items.pop(operator.index(index))
        self._rebuild_separators()
        self._resync()
        return _value_for(item.value)

    @override
    def remove(self, value: TomlValue) -> None:
        idx = list.index(self, value)
        del self[idx]

    @override
    def clear(self) -> None:
        self._node.items.clear()
        self._rebuild_separators()
        self._resync()

    @override
    def reverse(self) -> None:
        self._node.items.reverse()
        self._rebuild_separators()
        self._resync()

    @override
    def sort(
        self,
        *,
        key: Callable[[TomlValue], object] | None = None,
        reverse: bool = False,
    ) -> None:
        pairs = list(zip(_materialise_array(self._node), self._node.items, strict=True))
        if key is None:
            pairs.sort(key=lambda p: p[0], reverse=reverse)  # type: ignore[arg-type, return-value]
        else:
            pairs.sort(key=lambda p: key(p[0]), reverse=reverse)  # type: ignore[arg-type, return-value]
        self._node.items[:] = [item for _, item in pairs]
        self._rebuild_separators()
        self._resync()

    @override
    def __iadd__(self, values: Iterable[TomlValue]) -> Self:  # type: ignore[override]
        self.extend(values)
        return self

    @override
    def __imul__(self, count: SupportsIndex) -> Self:
        n = operator.index(count)
        if n <= 0:
            self.clear()
        else:
            base = list(self._node.items)
            for _ in range(n - 1):
                self._node.items.extend(deepcopy(item) for item in base)
            self._rebuild_separators()
            self._resync()
        return self

    # ------------------------------------------------------------------
    # Typed accessors for nested values. Mirror Table.array/.table.
    # ------------------------------------------------------------------

    def array(self, index: SupportsIndex) -> Array:
        """Return ``self[index]`` typed as a nested :class:`Array`."""
        value = self[index]
        if not isinstance(value, Array):
            msg = f"item {operator.index(index)} is a {type(value).__name__}, not an Array"
            raise TypeError(msg)
        return value

    def table(self, index: SupportsIndex) -> Table:
        """Return ``self[index]`` typed as a nested :class:`Table`."""
        value = self[index]
        if not isinstance(value, Table):
            msg = f"item {operator.index(index)} is a {type(value).__name__}, not a Table"
            raise TypeError(msg)
        return value


class AoT(list[Table]):
    """Array-of-tables, e.g. ``[[products]]`` repeated.

    Subclass of :class:`list`; supports basic mutation (append/insert
    of dict-shaped or :class:`Table` entries) by synthesizing fresh
    ``[[path]]`` sections in the underlying CST.
    """

    __slots__ = ("_doc_view", "_path")

    def __init__(
        self,
        doc_view: _DocumentView,
        path: tuple[str, ...],
        tables: list[Table],
    ) -> None:
        super().__init__(tables)
        self._doc_view = doc_view
        self._path = path

    # ------------------------------------------------------------------
    # CST <-> list synchronisation
    # ------------------------------------------------------------------

    def _own_sections(self) -> list[SectionNode]:
        """Sections that act as the [[path]] entry headers (in doc order)."""
        return [
            s
            for s in self._doc_view._node.sections  # noqa: SLF001
            if s.header is not None and s.header.kind == "array" and s.header.key.path == self._path
        ]

    def _resync(self) -> None:
        own = self._own_sections()
        list.clear(self)
        for s in own:
            owned = self._doc_view._aot_owned_range(s)  # noqa: SLF001
            list.append(
                self,
                _StdTable(
                    self._doc_view,
                    self._path,
                    sections=[s],
                    owned_scope=[s, *owned],
                ),
            )

    def _make_header_section(self) -> SectionNode:
        # Build a [[path]] header.
        parts = [make_key_part(p) for p in self._path]
        seps = ["."] * (len(parts) - 1)
        key = Key(parts=parts, separators=seps)
        header = TableHeaderNode(
            leading=Trivia(),
            kind="array",
            inner_pre=None,
            key=key,
            inner_post=None,
            trailing=None,
            trailing_comment=None,
            newline=NewlineNode("\n"),
        )
        return SectionNode(header=header, entries=[])

    def _populate_section(self, sec: SectionNode, value: object) -> None:
        """Fill ``sec`` with KV entries derived from ``value``.

        Accepts a plain dict, a :class:`Table`, or any
        :class:`collections.abc.Mapping`. Cross-document Tables are
        deep-cloned to satisfy the "no shared mutable state" rule.
        """
        if isinstance(value, Table):
            # Walk the source table's items; values are deep-cloned by
            # value_to_node so inline tables / arrays aren't aliased.
            for k, v in value.items():
                sec.entries.append(make_keyvalue_node(k, v))
            return
        if isinstance(value, Mapping):
            for k, v in value.items():
                if not isinstance(k, str):
                    msg = f"AoT entry keys must be strings, got {type(k).__name__}"
                    raise TOMLEditError(msg)
                sec.entries.append(make_keyvalue_node(k, v))
            return
        msg = (
            f"cannot append a value of type {type(value).__name__} to an "
            "array-of-tables; expected a dict or Table"
        )
        raise TOMLEditError(msg)

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    @override
    def append(self, value: Table | Mapping[str, object]) -> None:
        self._insert_at(len(self), value)

    @override
    def insert(
        self,
        index: SupportsIndex,
        value: Table | Mapping[str, object],
    ) -> None:
        self._insert_at(operator.index(index), value)

    @override
    def extend(
        self,
        values: Iterable[Table | Mapping[str, object]],
    ) -> None:
        for v in list(values):
            self._insert_at(len(self), v)

    def _insert_at(
        self,
        py_index: int,
        value: Table | Mapping[str, object],
    ) -> None:
        own = self._own_sections()
        n = len(own)
        if py_index < 0:
            py_index += n
        py_index = max(0, min(py_index, n))
        new_sec = self._make_header_section()
        self._populate_section(new_sec, value)
        own_header_leadings = [sec.header.leading for sec in own[1:] if sec.header is not None]
        if _gaps_uniformly_blank(own_header_leadings):
            assert new_sec.header is not None
            new_sec.header.leading.pieces.insert(0, NewlineNode("\n"))
        sections = self._doc_view._node.sections  # noqa: SLF001
        if py_index == n:
            # Append: insert after the last [[path]] entry's owned range,
            # or at end of doc if no entries exist yet.
            if own:
                last = own[-1]
                owned = self._doc_view._aot_owned_range(last)  # noqa: SLF001
                tail = owned[-1] if owned else last
                tail_idx = _index_of(sections, tail)
                sections.insert(tail_idx + 1, new_sec)
            else:
                sections.append(new_sec)
        else:
            # Insert before the entry currently at py_index.
            target = own[py_index]
            target_idx = _index_of(sections, target)
            sections.insert(target_idx, new_sec)
        self._resync()

    @override
    def pop(self, index: SupportsIndex = -1) -> Table:
        own = self._own_sections()
        n = len(own)
        i = operator.index(index)
        if i < 0:
            i += n
        if i < 0 or i >= n:
            msg = "pop index out of range"
            raise IndexError(msg)
        target = own[i]
        owned = self._doc_view._aot_owned_range(target)  # noqa: SLF001
        sections = self._doc_view._node.sections  # noqa: SLF001
        # Remove the entry header AND every section it owned.
        to_remove = {id(target), *(id(s) for s in owned)}
        # Capture the popped Table view *before* mutating sections.
        popped = _StdTable(
            self._doc_view,
            self._path,
            sections=[target],
            owned_scope=[target, *owned],
        )
        self._doc_view._node.sections = [  # noqa: SLF001
            s for s in sections if id(s) not in to_remove
        ]
        self._resync()
        return popped

    @override
    def clear(self) -> None:
        own = self._own_sections()
        to_remove: set[int] = set()
        for s in own:
            to_remove.add(id(s))
            for sub in self._doc_view._aot_owned_range(s):  # noqa: SLF001
                to_remove.add(id(sub))
        sections = self._doc_view._node.sections  # noqa: SLF001
        self._doc_view._node.sections = [  # noqa: SLF001
            s for s in sections if id(s) not in to_remove
        ]
        self._resync()

    @override
    def __delitem__(self, index: SupportsIndex | slice) -> None:
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            for i in sorted(indices, reverse=True):
                self.pop(i)
        else:
            self.pop(index)


def _index_of(sections: list[SectionNode], target: SectionNode) -> int:
    for i, s in enumerate(sections):
        if s is target:
            return i
    msg = "section not found in document (internal error)"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# View / aggregator
# ---------------------------------------------------------------------------


class _DocumentView:
    """Computes logical structure on demand from the CST."""

    __slots__ = ("_node",)

    def __init__(self, node: DocumentNode) -> None:
        self._node = node

    def _direct_sections(self, path: tuple[str, ...]) -> list[SectionNode]:
        out: list[SectionNode] = []
        for sec in self._node.sections:
            if path == ():
                if sec.header is None:
                    out.append(sec)
            else:
                hdr = sec.header
                if hdr is not None and hdr.kind == "table" and hdr.key.path == path:
                    out.append(sec)
        return out

    def _aot_sections(self, path: tuple[str, ...]) -> list[SectionNode]:
        return [
            sec
            for sec in self._node.sections
            if sec.header is not None and sec.header.kind == "array" and sec.header.key.path == path
        ]

    def _child_table_paths(self, path: tuple[str, ...]) -> list[tuple[str, ...]]:
        seen: dict[str, None] = {}
        for sec in self._node.sections:
            hdr = sec.header
            if hdr is None:
                continue
            hpath = hdr.key.path
            if len(hpath) > len(path) and hpath[: len(path)] == path:
                seen.setdefault(hpath[len(path)], None)
        return [(*path, k) for k in seen]

    def _aot_owned_range(self, aot_sec: SectionNode) -> list[SectionNode]:
        """Sections owned by this AoT entry.

        Owned = sections that come *after* ``aot_sec`` in document order
        and whose header path strictly extends this AoT's path. The range
        ends at the next [[same-path]] header or any other section that
        doesn't extend ``aot_sec``'s path.
        """
        if aot_sec.header is None:
            return []
        aot_path = aot_sec.header.key.path
        sections = self._node.sections
        i = -1
        for idx, candidate in enumerate(sections):
            if candidate is aot_sec:
                i = idx
                break
        if i < 0:
            return []
        owned: list[SectionNode] = []
        for j in range(i + 1, len(sections)):
            sec = sections[j]
            hdr = sec.header
            if hdr is None:
                # The synthetic root section appears only at index 0; safe to stop.
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

    def iter_table(
        self,
        path: tuple[str, ...],
        *,
        pinned_sections: list[SectionNode] | None = None,
        owned_scope: list[SectionNode] | None = None,
        extra_kvs: list[tuple[tuple[str, ...], KeyValueNode]] | None = None,
    ) -> Iterator[tuple[str, TomlValue]]:
        # The pool of sections we walk in physical order.
        section_pool: list[SectionNode] = (
            owned_scope if owned_scope is not None else list(self._node.sections)
        )

        # Sections whose entries are "direct" key/values at this exact path.
        direct_secs: list[SectionNode]
        if pinned_sections is not None:
            direct_secs = pinned_sections
        elif path == ():
            direct_secs = [s for s in section_pool if s.header is None]
        else:
            direct_secs = [
                s
                for s in section_pool
                if s.header is not None and s.header.kind == "table" and s.header.key.path == path
            ]
        direct_ids = {id(s) for s in direct_secs}

        name_order: list[str] = []
        seen: set[str] = set()

        def _add(name: str) -> None:
            if name not in seen:
                seen.add(name)
                name_order.append(name)

        direct_kvs_by_head: dict[str, list[KeyValueNode]] = {}
        extras_by_head: dict[str, list[tuple[tuple[str, ...], KeyValueNode]]] = {}
        aot_by_head: dict[str, list[SectionNode]] = {}
        sub_by_head: dict[str, list[SectionNode]] = {}

        # Single physical-order walk: each section either contributes direct
        # entries (header == path) or registers a sub-table / sub-AoT head.
        # First-appearance order matches what tomllib produces.
        plen = len(path)
        for sec in section_pool:
            if id(sec) in direct_ids:
                for entry in sec.entries:
                    head = entry.key.path[0]
                    direct_kvs_by_head.setdefault(head, []).append(entry)
                    _add(head)
                continue
            hdr = sec.header
            if hdr is None:
                continue
            hpath = hdr.key.path
            if len(hpath) <= plen or hpath[:plen] != path:
                continue
            head = hpath[plen]
            if hdr.kind == "array" and hpath == (*path, head):
                aot_by_head.setdefault(head, []).append(sec)
            else:
                sub_by_head.setdefault(head, []).append(sec)
            _add(head)

        # Extras (dotted-key prefixes inherited from an ancestor) have no
        # physical position; tack their heads on at the end.
        if extra_kvs:
            for rel_path, entry in extra_kvs:
                head = rel_path[0]
                extras_by_head.setdefault(head, []).append((rel_path, entry))
                _add(head)

        for head in name_order:
            direct_kvs = direct_kvs_by_head.get(head, [])
            extras = extras_by_head.get(head, [])
            aot_secs = aot_by_head.get(head, [])
            sub_secs = sub_by_head.get(head, [])

            if aot_secs:
                tables: list[Table] = []
                for s in aot_secs:
                    owned = self._aot_owned_range(s)
                    tables.append(
                        _StdTable(
                            self,
                            (*path, head),
                            sections=[s],
                            owned_scope=[s, *owned],
                        ),
                    )
                yield head, AoT(self, (*path, head), tables)
                continue

            # Split into "terminal" (binds a value at this name) and
            # "nested" (contributes to a sub-table at this name).
            terminal: KeyValueNode | None = None
            nested_kvs: list[KeyValueNode] = []
            nested_extras: list[tuple[tuple[str, ...], KeyValueNode]] = []
            for kv in direct_kvs:
                if len(kv.key.path) == 1:
                    if terminal is None:
                        terminal = kv
                else:
                    nested_kvs.append(kv)
            for rel_path, kv in extras:
                if len(rel_path) == 1:
                    if terminal is None:
                        terminal = kv
                else:
                    nested_extras.append((rel_path, kv))

            if terminal is not None and not nested_kvs and not nested_extras and not sub_secs:
                yield head, _value_for(terminal.value)
                continue

            if not sub_secs and not nested_extras:
                # Pure dotted from this section level.
                yield head, _DottedSubTable([(kv.key.path, kv.value) for kv in nested_kvs], depth=1)
                continue

            # Merged view at path + (head,): combines sub-section content with
            # any dotted KVs from this section and any ancestor-dotted extras.
            child_extras: list[tuple[tuple[str, ...], KeyValueNode]] = [
                (kv.key.path[1:], kv) for kv in nested_kvs
            ]
            child_extras.extend((rel_path[1:], entry) for rel_path, entry in nested_extras)

            if owned_scope is not None:
                child_path = (*path, head)
                cplen = len(child_path)
                child_owned = [
                    s
                    for s in owned_scope
                    if s.header is not None
                    and len(s.header.key.path) >= cplen
                    and s.header.key.path[:cplen] == child_path
                ]
                yield (
                    head,
                    _StdTable(
                        self,
                        child_path,
                        owned_scope=child_owned,
                        extra_kvs=child_extras or None,
                    ),
                )
            else:
                yield (
                    head,
                    _StdTable(
                        self,
                        (*path, head),
                        extra_kvs=child_extras or None,
                    ),
                )


__all__ = [
    "AoT",
    "Array",
    "Document",
    "Scalar",
    "Table",
    "TomlValue",
]
