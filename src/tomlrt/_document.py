"""Logical Document/Table/Array/AoT view over a CST.

This module exposes the public mapping/sequence types that users
interact with. It implements both the read path and the structural
mutation API on top of the physical CST defined in :mod:`tomlrt._nodes`.
"""

from __future__ import annotations

import operator
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from copy import deepcopy
from datetime import date, datetime, time
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Protocol,
    SupportsIndex,
    TypeAlias,
    TypeVar,
    overload,
)

if sys.version_info >= (3, 11):
    from typing import assert_never
else:
    from typing_extensions import assert_never

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tomlrt._errors import TOMLError
from tomlrt._nodes import (
    ArrayNode,
    BoolNode,
    CommentNode,
    DateTimeNode,
    DocumentNode,
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
from tomlrt._synthesise import (
    make_key_part,
    make_keyvalue_node,
    make_simple_key,
    value_to_node,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    from tomlrt._nodes import (
        ArrayItem,
        HeaderKind,
        TriviaPiece,
        ValueNode,
    )


Scalar: TypeAlias = str | int | float | bool | datetime | date | time
TomlValue: TypeAlias = "Scalar | Array | AoT | Table"

_MISSING: Any = object()
_T = TypeVar("_T")


def _to_plain(value: object) -> Any:
    """Recursively convert tomlrt views to plain Python data.

    Tables become ``dict``s, AoTs and Arrays become ``list``s, scalars
    are returned as-is. The result shares no mutable state with the
    underlying document and is safe to hand to consumers that expect
    real ``dict``/``list`` objects.
    """
    if isinstance(value, Table):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, AoT):
        return [_to_plain(t) for t in value]
    if isinstance(value, Array):
        return [_to_plain(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Read-side helpers
# ---------------------------------------------------------------------------


_SCALAR_NODE_TYPES = (StringNode, IntegerNode, FloatNode, BoolNode, DateTimeNode)


def _value_for(node: ValueNode) -> TomlValue:
    if isinstance(node, ArrayNode):
        return Array(node)
    if isinstance(node, InlineTableNode):
        return _InlineTable(node)
    if isinstance(node, _SCALAR_NODE_TYPES):
        return node.value
    assert_never(node)


def _materialise_array(node: ArrayNode) -> list[TomlValue]:
    return [_value_for(item.value) for item in node.items]


def _walk_newline_nodes(node: object) -> Iterator[NewlineNode]:
    """Yield every :class:`NewlineNode` reachable from ``node``.

    Used to detect and normalise the document-wide line ending. Walks
    through dataclass fields and lists; ignores other primitives.
    """
    stack: list[object] = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, NewlineNode):
            yield current
        elif isinstance(current, list):
            stack.extend(current)
        elif hasattr(current, "__dataclass_fields__"):
            stack.extend(
                getattr(current, name) for name in current.__dataclass_fields__
            )


def _detect_newline(node: DocumentNode) -> str:
    """Return the document's line ending if uniform, else ``"\\n"``.

    Returns ``"\\r\\n"`` only when every :class:`NewlineNode` in the
    CST already uses CRLF; mixed or pure-LF documents return
    ``"\\n"``. The CRLF case enables ``Document.render`` to convert
    newly-synthesised ``"\\n"`` newlines to match the source. We
    deliberately leave mixed-newline documents alone — normalising
    them either way would break the no-mutation round-trip
    invariant.
    """
    saw_any = False
    for nl in _walk_newline_nodes(node):
        saw_any = True
        if nl.text != "\r\n":
            return "\n"
    return "\r\n" if saw_any else "\n"


def _normalise_newlines(node: DocumentNode, target: str) -> None:
    """Set every :class:`NewlineNode` in ``node`` to ``target``."""
    for nl in _walk_newline_nodes(node):
        if nl.text != target:
            nl.text = target


def _first_indent_after_newline(pieces: Sequence[TriviaPiece]) -> str:
    """Indent (run of spaces/tabs) immediately following the *first*
    newline in ``pieces``, or empty string if no such pattern exists.
    """
    for i in range(len(pieces) - 1):
        if isinstance(pieces[i], NewlineNode) and isinstance(
            pieces[i + 1], WhitespaceNode
        ):
            return pieces[i + 1].text
    return ""


def _detect_indent(section: SectionNode) -> str:
    """Return the leading-whitespace indent used by the section's last entry."""
    if not section.entries:
        return ""
    return _indent_after_last_newline(section.entries[-1].leading)


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


def _iter_value_trivia(value: ValueNode) -> Iterator[Trivia]:
    """Yield every trivia slot reachable inside ``value``.

    Walks into :class:`ArrayNode` and :class:`InlineTableNode`
    children (the only value kinds with internal trivia); leaf
    scalars yield nothing. Used by translators that need to assert
    "this value carries no comments anywhere".
    """
    if isinstance(value, ArrayNode):
        yield value.final_trivia
        for item in value.items:
            yield item.leading
            yield item.trailing
            yield item.post_comma_trivia
            yield from _iter_value_trivia(item.value)
    elif isinstance(value, InlineTableNode):
        yield value.final_trivia
        for entry in value.entries:
            yield entry.leading
            yield entry.trailing
            yield entry.post_comma_trivia
            yield from _iter_value_trivia(entry.value)


def _value_has_inner_comment(value: ValueNode) -> bool:
    """``True`` iff any trivia inside ``value`` carries a comment."""
    return any(_trivia_has_comment(t) for t in _iter_value_trivia(value))


def _trivia_has_comment(trivia: Trivia) -> bool:
    """``True`` iff ``trivia`` contains any :class:`CommentNode`."""
    return any(isinstance(p, CommentNode) for p in trivia.pieces)


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


def _prepend_blank_line(trivia: Trivia) -> None:
    """Prepend a blank-line ``NewlineNode`` to ``trivia`` if missing.

    Idempotent: a no-op when ``trivia`` already starts with a newline.
    """
    if not _starts_with_blank_line(trivia):
        trivia.pieces.insert(0, NewlineNode("\n"))


def _strip_comment_marker(text: str) -> str:
    """``"# foo"`` → ``"foo"``.

    Removes a leading ``#`` and one optional space. Symmetric with
    :func:`_format_comment`, which prepends exactly ``# `` (or just
    ``#`` for empty input). We deliberately do *not* strip trailing
    whitespace -- the comment view contract is that reads and writes
    are exact inverses, so trailing spaces are content.
    """
    text = text.removeprefix("#")
    return text.removeprefix(" ")


def _format_comment(text: str) -> str:
    """Format user text as the payload for a :class:`CommentNode`.

    Always emits ``"# " + text`` (or ``"#"`` for empty text). The ``#``
    marker is the renderer's responsibility, not the caller's: this
    keeps round-trip symmetric with the reader, which strips a single
    leading ``"# "`` (or ``"#"``). Comment text that itself starts
    with ``#`` is treated as literal content -- e.g. ``"#hashtag"``
    renders as ``"# #hashtag"`` and reads back as ``"#hashtag"``.
    Raises :class:`TOMLError` if ``text`` contains any character the
    TOML parser would reject in a comment: comments are single-line by
    definition, and the only control character permitted (besides line
    terminators which end them) is TAB.
    """
    for ch in text:
        cp = ord(ch)
        if ch in ("\n", "\r"):
            msg = "comment text must not contain a line terminator"
            raise TOMLError(msg)
        if (cp <= 0x1F and ch != "\t") or cp == 0x7F:
            msg = f"comment text must not contain control character U+{cp:04X}"
            raise TOMLError(msg)
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

    ``value=None`` clears the comment and strips trailing whitespace
    from ``node.trailing`` so we don't render ``foo = 12 \\n`` after
    the comment goes away. ``value=""`` is *not* a clear: it sets an
    empty comment (rendered as a bare ``#``), so the API is symmetric
    with the reader, which returns ``""`` for a parsed bare ``#``.
    """
    if value is None:
        node.trailing_comment = None
        node.trailing = None
        return
    if node.trailing is None:
        node.trailing = WhitespaceNode(" ")
    node.trailing_comment = CommentNode(text=_format_comment(value))
    if node.newline is None:
        node.newline = NewlineNode("\n")


def _trailing_comment_block_span(
    pieces: Sequence[TriviaPiece],
    *,
    min_start: int = 0,
) -> tuple[int, int]:
    """Locate the trailing-comment block within ``pieces``.

    Returns ``(start, end)`` such that ``pieces[start:end]`` is the
    contiguous run of ``[WS] CommentNode NewlineNode`` triples
    immediately preceding the trivia's anchoring whitespace, and
    ``pieces[end:]`` is that anchor. Earlier comment lines that are
    separated from the run by a blank line are excluded.

    ``min_start`` clips the backward walk: triples lying entirely below
    that index are not consumed. Used by callers that want the trailing
    block of ``post_comma_trivia`` *after* an EOL-comment prefix
    (``[WS] Comment NL``) that belongs to the previous item.
    """
    end = len(pieces)
    while end > min_start and isinstance(pieces[end - 1], WhitespaceNode):
        end -= 1
    start = end
    i = end
    while i >= min_start + 2:
        nl = pieces[i - 1]
        cm = pieces[i - 2]
        if not (isinstance(nl, NewlineNode) and isinstance(cm, CommentNode)):
            break
        i -= 2
        if i > min_start and isinstance(pieces[i - 1], WhitespaceNode):
            i -= 1
        start = i
    return start, end


def _extract_trailing_comment_block(
    trivia: Trivia,
    *,
    min_start: int = 0,
) -> tuple[str, ...]:
    """Return the contiguous run of comment lines at the *end* of ``trivia``.

    A "comment line" is ``[WS] CommentNode NewlineNode``. The trailing
    block ends immediately before the trivia's anchoring whitespace
    (the indent of the line that follows). Earlier comment lines that
    are separated from the run by a blank line are *not* included.

    ``min_start`` clips the scan; see :func:`_trailing_comment_block_span`.
    """
    pieces = trivia.pieces
    start, end = _trailing_comment_block_span(pieces, min_start=min_start)
    return tuple(
        _strip_comment_marker(p.text)
        for p in pieces[start:end]
        if isinstance(p, CommentNode)
    )


def _validate_comment_lines(lines: Sequence[str]) -> None:
    """Reject a bare ``str`` masquerading as a sequence of comment lines.

    ``str`` is technically a ``Sequence[str]`` of one-character strings,
    so accidentally passing a single comment as a string would silently
    iterate it character-by-character and produce a stack of
    one-character ``# x`` lines. Refuse instead.
    """
    if isinstance(lines, str):
        msg = (
            "expected an iterable of comment strings, got a str; "
            "wrap a single comment in a tuple, e.g. ('# my comment',)"
        )
        raise TypeError(msg)


def _replace_trailing_comment_block(
    trivia: Trivia,
    lines: Sequence[str],
    indent: str,
    *,
    min_start: int = 0,
) -> None:
    """Replace the trailing comment block in ``trivia`` with ``lines``.

    Earlier trivia (older blank-separated comments, leading whitespace)
    and the trailing whitespace anchor are preserved.
    ``min_start`` clips the scan; see :func:`_trailing_comment_block_span`.
    """
    _validate_comment_lines(lines)
    pieces = trivia.pieces
    start, end = _trailing_comment_block_span(pieces, min_start=min_start)
    new_pieces: list[TriviaPiece] = []
    for line in lines:
        if indent:
            new_pieces.append(WhitespaceNode(indent))
        new_pieces.append(CommentNode(text=_format_comment(line)))
        new_pieces.append(NewlineNode("\n"))
    trivia.pieces = list(pieces[:start]) + new_pieces + list(pieces[end:])


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
    ``value=None`` clears it; ``value=""`` writes a bare ``#`` (so the
    API is symmetric with the reader, which returns ``""`` for a parsed
    bare ``#``). When ``force_newline`` is True and the trivia would
    otherwise lack a NewlineNode after the new comment, one is inserted
    (so following content doesn't end up on the same line as the
    comment).
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
    if value is not None:
        new_prefix.append(WhitespaceNode(" "))
        new_prefix.append(CommentNode(text=_format_comment(value)))
        if force_newline and not any(isinstance(p, NewlineNode) for p in rest):
            new_prefix.append(NewlineNode("\n"))
    trivia.pieces = new_prefix + rest


def _split_pct_eol(pieces: Sequence[TriviaPiece]) -> int:
    """Return the index after the EOL-comment prefix of a ``post_comma_trivia``.

    A pct value's EOL portion is ``[WS]? Comment [NL]?`` at index 0 — that
    is, the previous item's end-of-line comment, on the same source line
    as the comma. Returns ``0`` if there is no EOL comment.

    Pass the result as ``min_start`` to :func:`_trailing_comment_block_span`
    (or its callers) so the backward walk doesn't confuse the EOL line
    with a ``leading_comments[i]`` entry.
    """
    ws_end = 1 if pieces and isinstance(pieces[0], WhitespaceNode) else 0
    if ws_end < len(pieces) and isinstance(pieces[ws_end], CommentNode):
        end = ws_end + 1
        if end < len(pieces) and isinstance(pieces[end], NewlineNode):
            end += 1
        return end
    return 0


def _is_pure_whitespace(t: Trivia) -> bool:
    """True iff trivia contains only whitespace/newline pieces (no comments)."""
    pieces = t.pieces
    if not pieces:
        return True
    # ``CommentNode`` is the only non-whitespace TriviaPiece; checking
    # for its absence dodges per-piece tuple-isinstance.
    return not _trivia_has_comment(t)


def _scan_leading_comment_run(pieces: list[TriviaPiece]) -> tuple[int, list[str]]:
    """Walk a leading run of ``[WS] # … \\n`` triples from offset 0.

    Returns ``(end_index, raw_comment_texts)``. ``end_index`` is the
    index of the first piece that is not part of the run.
    """
    n = len(pieces)
    comments: list[str] = []
    i = 0
    while i < n:
        j = i
        if j < n and isinstance(pieces[j], WhitespaceNode):
            j += 1
        if (
            j + 1 < n
            and isinstance(pieces[j], CommentNode)
            and isinstance(pieces[j + 1], NewlineNode)
        ):
            comments.append(pieces[j].text)
            i = j + 2
        else:
            break
    return i, comments


def _trivia_render_eq(a: Trivia, b: Trivia) -> bool:
    return a.render() == b.render()


def _clone_trivia(t: Trivia) -> Trivia:
    # Trivia pieces (WhitespaceNode/NewlineNode/CommentNode) are
    # never mutated in place — only replaced wholesale — so we can
    # share piece refs. Only the list container needs to be fresh
    # so that subsequent splicing doesn't disturb the original.
    return Trivia(list(t.pieces))


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


def _logical_leading_slot(
    items: Sequence[_Separated],
    i: int,
) -> tuple[Trivia, int]:
    """Return ``(trivia, min_start)`` locating ``leadings[i]`` in the CST.

    The block lives at the trailing-comment span of the returned trivia,
    starting no earlier than ``min_start``. The two on-disk encodings
    are unified here:

    * ``i == 0``: ``items[0].leading``, scan from the start
    * ``i > 0``: ``items[i-1].post_comma_trivia``, scan after the EOL
      prefix that belongs to item *i-1*
    """
    if i == 0:
        return items[0].leading, 0
    slot = items[i - 1].post_comma_trivia
    return slot, _split_pct_eol(slot.pieces)


def _leading0_is_header(leading: Trivia) -> bool:
    """True iff ``items[0].leading`` is a "header" comment stuck to ``[``.

    A header comment sits on the same source line as the container's
    opening bracket (no newline before it). It logically belongs to the
    container header / ``open_pad``, not to ``leading_comments[0]``.
    Snapshot/restore must leave it alone so that reorder operations
    don't accidentally re-attach it to a moved item.
    """
    if not _trivia_has_comment(leading):
        return False
    return not isinstance(leading.pieces[0], NewlineNode)


def _snapshot_item_leadings(items: Sequence[_Separated]) -> list[tuple[str, ...]]:
    """Capture per-item leading-comment blocks in their *logical* positions.

    The on-disk encoding splits leading-comment blocks across two slots:
    item 0's block lives in ``items[0].leading``, while item *i*'s block
    (for ``i > 0``) lives at the tail of ``items[i-1].post_comma_trivia``,
    AFTER any EOL-comment prefix that belongs to item *i-1* itself.
    Reordering ``items`` (sort, reverse, insert, pop) detaches each
    block from its logical owner. Snapshot before reorder, transform the
    parallel list alongside ``items``, then re-encode with
    :func:`_write_item_leadings`.

    A comment "stuck" to the container's opening bracket (no newline
    before it) is treated as part of the container header / ``open_pad``
    rather than as ``leadings[0]``: snapshot returns ``()`` for it and
    write leaves it alone, so it stays anchored to ``[`` across reorder.
    """
    if not items:
        return []
    out: list[tuple[str, ...]] = []
    for i in range(len(items)):
        if i == 0 and _leading0_is_header(items[0].leading):
            out.append(())
            continue
        trivia, min_start = _logical_leading_slot(items, i)
        out.append(_extract_trailing_comment_block(trivia, min_start=min_start))
    return out


def _write_item_leadings(
    items: Sequence[_Separated],
    leadings: Sequence[Sequence[str]],
) -> None:
    """Re-encode per-item leading-comment blocks into their canonical slots.

    Inverse of :func:`_snapshot_item_leadings`. Also clears any stale
    leading-of-next residue from the *last* item's pct (no logical
    owner once the item sits at the end), and leaves a header comment
    stuck to ``[`` alone (``open_pad`` already carries it).
    """
    if not items:
        return
    for i in range(len(items)):
        if i == 0 and _leading0_is_header(items[0].leading):
            continue
        trivia, min_start = _logical_leading_slot(items, i)
        ind = _indent_after_last_newline(trivia)
        _replace_trailing_comment_block(trivia, leadings[i], ind, min_start=min_start)
    last_slot = items[-1].post_comma_trivia
    _replace_trailing_comment_block(
        last_slot,
        (),
        _indent_after_last_newline(last_slot),
        min_start=_split_pct_eol(last_slot.pieces),
    )


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
        pad_text = final_trivia.render()
        if "\n" in pad_text:
            # Multiline-intent empty container: infer the per-item
            # indent from the first whitespace run that follows a
            # newline in the close pad. Scanning pieces (rather than
            # the rendered text) catches indents that sit before a
            # comment, which the simpler ``rsplit('\n', 1)`` misses.
            indent_text = _first_indent_after_newline(final_trivia.pieces) or "    "
            inter = Trivia([NewlineNode("\n"), WhitespaceNode(indent_text)])
            open_pad: Trivia
            if _is_pure_whitespace(final_trivia):
                open_pad = _clone_trivia(inter)
            else:
                # Preserve authoring content (e.g. comments) inside an
                # otherwise-empty container by parking it in open_pad,
                # where it becomes the leading trivia of the first
                # inserted item. Ensure the pad ends with the item
                # indent so the new value lines up.
                open_pad = _clone_trivia(final_trivia)
                if _indent_after_last_newline(open_pad) != indent_text:
                    open_pad.pieces.append(WhitespaceNode(indent_text))
            return _SeparatorStyle(
                open_pad=open_pad,
                inter_separator=_clone_trivia(inter),
                trailing_comma=True,
                close_pad=Trivia([NewlineNode("\n")]),
            )
        # Mirror a single-space internal pad (``[ ]`` / ``{ }``) into
        # both edges so first insertion preserves it as ``[ x ]``.
        single_pad = (
            Trivia([WhitespaceNode(" ")])
            if pad_text == " "
            else _clone_trivia(final_trivia)
        )
        return _SeparatorStyle(
            open_pad=_clone_trivia(single_pad),
            inter_separator=Trivia([WhitespaceNode(" ")]),
            trailing_comma=False,
            close_pad=_clone_trivia(single_pad),
        )
    open_pad = _clone_trivia(items[0].leading)
    last = items[-1]
    sep: Trivia | None = None
    for it in items[:-1]:
        if it.has_comma and _is_pure_whitespace(it.post_comma_trivia):
            sep = _clone_trivia(it.post_comma_trivia)
            break
    if sep is None:
        # Fall back to the structural pattern (newline + indent) of any
        # comment-bearing separator, so authoring intent is preserved
        # when every existing item carries an inline comment.
        for it in items[:-1]:
            if it.has_comma and "\n" in it.post_comma_trivia.render():
                indent = _indent_after_last_newline(it.post_comma_trivia)
                pieces: list[Any] = [NewlineNode("\n")]
                if indent:
                    pieces.append(WhitespaceNode(indent))
                sep = Trivia(pieces)
                break
    if sep is None:
        # Last-resort multiline detection: when ``items[:-1]`` yields
        # nothing usable (e.g. single-item containers), inspect the
        # surrounding trivia for any newline pattern. If present the
        # author intended multi-line layout; seed ``sep`` from the
        # first item's leading indent so appended items line up.
        hints = (items[0].leading, last.post_comma_trivia, last.trailing, final_trivia)
        if any("\n" in h.render() for h in hints):
            indent = _indent_after_last_newline(items[0].leading) or "    "
            sep = Trivia([NewlineNode("\n"), WhitespaceNode(indent)])
    if sep is None:
        sep = Trivia([WhitespaceNode(" ")])
    if last.has_comma:
        trailing_comma = True
        if _is_pure_whitespace(last.post_comma_trivia):
            close_pad = _clone_trivia(last.post_comma_trivia)
        else:
            # Comment lives in the close slot; derive a clean pad from
            # the inter-separator instead of dragging it onto new items.
            close_pad = _derive_close_pad(sep)
    else:
        trailing_comma = False
        # Close-pad combines last.trailing + final_trivia (parser and
        # synthesiser disagree on which slot holds it). If a comment is
        # in either slot it belongs to the item, not the pad — derive a
        # sensible pad from the inter-separator instead.
        combined = Trivia(
            list(last.trailing.pieces) + list(final_trivia.pieces),
        )
        close_pad = (
            combined if _is_pure_whitespace(combined) else _derive_close_pad(sep)
        )
    return _SeparatorStyle(
        open_pad=open_pad,
        inter_separator=sep,
        trailing_comma=trailing_comma,
        close_pad=close_pad,
    )


def _strip_trailing_indent(trivia: Trivia) -> None:
    """Drop any pure-whitespace pieces sitting after ``trivia``'s last newline.

    Used on the *last* item's post-comma / trailing trivia after a
    reorder: a piece like ``"\\n  "`` was the indent for whichever
    value used to follow, but if the item is now last there is no
    next value, only the closing bracket / brace. Leaving the indent
    in place renders ``"  ]"`` instead of ``"]"``.
    """
    pieces = trivia.pieces
    while pieces and isinstance(pieces[-1], WhitespaceNode):
        pieces.pop()


def _ensure_trailing_indent(trivia: Trivia, indent: str) -> None:
    """Ensure ``trivia`` ending in a newline carries ``indent`` after it.

    Only appends pure whitespace; preserves preceding comments and
    structural newlines. No-op if ``trivia`` already has any
    whitespace after its last newline (we don't second-guess the
    user's choice) or if it has no newline at all.
    """
    if not indent:
        return
    text = trivia.render()
    nl = text.rfind("\n")
    if nl < 0:
        return
    tail = text[nl + 1 :]
    if tail or not all(c in " \t" for c in tail):
        return
    trivia.pieces.append(WhitespaceNode(indent))


def _apply_separator_style(
    container: ArrayNode | InlineTableNode,
    style: _SeparatorStyle,
) -> None:
    """Re-apply a sampled :class:`_SeparatorStyle` to the items.

    Items whose separator slot contains a non-whitespace token (e.g. an
    inline ``# comment``) are left alone so authoring intent is preserved.
    """
    items: Sequence[_Separated] = (
        container.items if isinstance(container, ArrayNode) else container.entries
    )
    n = len(items)
    if n == 0:
        container.final_trivia = _clone_trivia(style.close_pad)
        return
    items[0].leading = _clone_trivia(style.open_pad)
    for it in items[1:]:
        # items[i>0].leading must be empty: inter-item content (comments
        # included) is encoded at the tail of items[i-1].post_comma_trivia.
        # Anything found here is stale residue from a previous position
        # under reorder/insert/pop. Reorder operations snapshot the
        # logical leadings via _snapshot_item_leadings beforehand and
        # restore them via _write_item_leadings afterwards.
        it.leading = Trivia()
    container.final_trivia = Trivia()
    inter_render = style.inter_separator.render()
    inter_indent = _indent_after_last_newline(style.inter_separator)
    close_render = style.close_pad.render()
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
            elif _is_pure_whitespace(item.post_comma_trivia):
                if item.post_comma_trivia.render() != inter_render:
                    item.post_comma_trivia = _clone_trivia(style.inter_separator)
            else:
                _ensure_trailing_indent(item.post_comma_trivia, inter_indent)
            if _is_pure_whitespace(item.trailing) and item.trailing.pieces:
                item.trailing = Trivia()
        elif style.trailing_comma:
            item.has_comma = True
            if _is_pure_whitespace(item.post_comma_trivia):
                item.post_comma_trivia = _clone_trivia(style.close_pad)
            else:
                # Last item carries a comment: strip any trailing indent
                # that was meant for "the next value" so the closing
                # bracket lands at column 0 (or wherever close_pad lands).
                _strip_trailing_indent(item.post_comma_trivia)
            if _is_pure_whitespace(item.trailing):
                item.trailing = Trivia()
        else:
            if item.has_comma and _is_pure_whitespace(item.post_comma_trivia):
                item.has_comma = False
                item.post_comma_trivia = Trivia()
            if _is_pure_whitespace(item.trailing):
                if item.trailing.render() != close_render:
                    item.trailing = _clone_trivia(style.close_pad)
            else:
                # Non-empty content (e.g. comment) in trailing: trim any
                # leftover "next-item indent" so the closing bracket
                # doesn't end up indented after a reorder.
                _strip_trailing_indent(item.trailing)


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


def _new_section(
    path: tuple[str, ...],
    *,
    kind: HeaderKind = "table",
    leading: Trivia | None = None,
    trailing: WhitespaceNode | None = None,
    trailing_comment: CommentNode | None = None,
) -> SectionNode:
    """Build an empty ``[path]`` (or ``[[path]]``) section.

    Trivia defaults to empty; pass ``leading`` / ``trailing`` /
    ``trailing_comment`` to carry over comment material from a node
    that the new section is replacing (used by the promotion paths).
    """
    parts = [make_key_part(p) for p in path]
    seps = ["."] * (len(parts) - 1)
    header = TableHeaderNode(
        leading=leading if leading is not None else Trivia(),
        kind=kind,
        inner_pre=None,
        key=Key(parts=parts, separators=seps),
        inner_post=None,
        trailing=trailing,
        trailing_comment=trailing_comment,
        newline=NewlineNode("\n"),
    )
    return SectionNode(header=header, entries=[])


def _kv_from_inline_entry(entry: InlineTableEntry, *, deep: bool) -> KeyValueNode:
    """Wrap an inline-table entry as a standalone ``key = value`` KV.

    Used by the promotion paths to convert inline-table contents into
    section entries. ``deep=True`` deep-clones the key and value (used
    when the source might still be reachable from elsewhere).
    """
    return KeyValueNode(
        leading=Trivia(),
        key=deepcopy(entry.key) if deep else entry.key,
        pre_eq=WhitespaceNode(" "),
        post_eq=WhitespaceNode(" "),
        value=deepcopy(entry.value) if deep else entry.value,
        trailing=None,
        trailing_comment=None,
        newline=NewlineNode("\n"),
    )


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
    section = _new_section(
        path,
        leading=Trivia(list(source_kv.leading.pieces)),
        trailing=source_kv.trailing,
        trailing_comment=source_kv.trailing_comment,
    )
    section.entries = [_kv_from_inline_entry(e, deep=False) for e in inline.entries]
    return section


def _build_promoted_aot_section(
    path: tuple[str, ...],
    inline: InlineTableNode,
) -> SectionNode:
    """Build a ``[[path]]`` section containing ``inline``'s entries.

    Used by :meth:`Table.promote_array` to convert each element of an
    inline array of inline tables into its own AoT entry.
    """
    section = _new_section(path, kind="array")
    section.entries = [_kv_from_inline_entry(e, deep=True) for e in inline.entries]
    return section


def _clone_sections_rebased(
    sections: Iterable[SectionNode],
    src_path: tuple[str, ...],
    full_path: tuple[str, ...],
) -> list[SectionNode]:
    """Deep-clone every header section under ``src_path``, rebasing the prefix.

    Implicit pre-header sections (``header is None``) and headers
    whose path doesn't extend ``src_path`` are skipped. Other sections
    are deep-cloned with their key rebased: relative depth below
    ``src_path`` is preserved, so ``[a.sub.deep]`` cloned at
    ``("t",)`` becomes ``[t.sub.deep]``. Shared rebase primitive
    behind :func:`_clone_aot_sections` and :func:`_clone_table_sections`.
    """
    splen = len(src_path)
    out: list[SectionNode] = []
    for sec in sections:
        hdr = sec.header
        if hdr is None or len(hdr.key.path) < splen or hdr.key.path[:splen] != src_path:
            continue
        cloned = deepcopy(sec)
        assert cloned.header is not None
        new_path = (*full_path, *hdr.key.path[splen:])
        cloned.header.key = _make_dotted_key(new_path)
        out.append(cloned)
    return out


def _clone_aot_sections(
    value: AoT,
    full_path: tuple[str, ...],
) -> list[SectionNode]:
    """Deep-clone every CST section that contributes to ``value``, rebased.

    Each AoT entry contributes its ``[[path]]`` header *and* any
    sub-sections in its owned range (e.g. ``[path.sub]`` /
    ``[[path.nested]]``). All of them are deep-cloned and have their
    header paths rewritten so the ``len(value._path)``-prefix is
    replaced by ``full_path``. Surrounding placement (insert index,
    blank-line policy, dict-storage sync) is the caller's job.
    """
    doc_node = value._doc_node  # noqa: SLF001
    blocks = (doc_node.aot_entry_block(header) for header in value._own_sections())  # noqa: SLF001
    sections = [sec for block in blocks for sec in block]
    return _clone_sections_rebased(sections, value._path, full_path)  # noqa: SLF001


def _new_host_section(path: tuple[str, ...]) -> SectionNode:
    """Synthesize an empty ``[path]`` section, ready to host dotted KVs."""
    return SectionNode(
        header=TableHeaderNode(
            leading=Trivia(),
            kind="table",
            inner_pre=None,
            key=_make_dotted_key(path),
            inner_post=None,
            trailing=None,
            trailing_comment=None,
            newline=NewlineNode("\n"),
        ),
        entries=[],
    )


def _clone_table_sections(
    value: _StdTable,
    full_path: tuple[str, ...],
    *,
    head_kind: HeaderKind = "table",
) -> list[SectionNode]:
    """Deep-clone every CST section that contributes to ``value``, rebased.

    The returned sections are independent of ``value._doc_node`` and have
    their headers rewritten so the ``len(value._path)``-prefix is replaced
    by ``full_path``. Surrounding placement is the caller's responsibility.

    Two sources of contributing CST data are handled: sections at or below
    ``value._path`` (cloned and re-keyed) and dotted KVs in ancestor
    sections that extend into ``value._path`` (cloned into a host section
    at ``full_path``, synthesised on demand). For a ``Document`` source
    (``value._path == ()``) the implicit pre-header section's entries are
    treated as ancestor extras, since :meth:`_compute_extras` returns
    ``None`` in that case.

    The leading section's header kind is forced to ``head_kind``: pass
    ``"table"`` (the default) to install as a ``[full_path]`` standard
    section, or ``"array"`` to install as a ``[[full_path]]`` AoT entry.
    Without this normalisation an AoT-entry source cloned into a
    standard-table slot would keep its ``[[..]]`` header (and vice
    versa).
    """
    src_path = value._path  # noqa: SLF001
    fplen = len(full_path)
    doc = value._doc_node  # noqa: SLF001

    src_scope = value._scope()  # noqa: SLF001
    src_sections = src_scope if src_scope is not None else doc.sections
    new_secs = _clone_sections_rebased(src_sections, src_path, full_path)
    head = next(
        (
            s
            for s in new_secs
            if s.header is not None and len(s.header.key.path) == fplen
        ),
        None,
    )

    if len(src_path) == 0:
        extras = [
            (kv.key.path, kv)
            for sec in doc.sections
            if sec.header is None
            for kv in sec.entries
        ]
    else:
        extras = value._compute_extras() or []  # noqa: SLF001

    if extras:
        if head is None:
            head = _new_host_section(full_path)
            new_secs.insert(0, head)
        for rel_path, kv in extras:
            cloned_kv = deepcopy(kv)
            cloned_kv.key = _make_dotted_key(rel_path)
            head.entries.append(cloned_kv)

    if head is not None and head.header is not None:
        head.header.kind = head_kind

    return new_secs


def _apply_prior_leading(
    new_secs: Sequence[SectionNode],
    prior_leading: Trivia | None,
) -> None:
    """Transplant a prior section's header trivia onto a replacement block.

    ``_prepare_section_slot`` snapshots the leading trivia of the
    section being replaced (the comments / blank lines that sat above
    its ``[name]`` / ``[[name]]`` line) before purging. When a new
    block of sections is about to be spliced into the same slot, we
    move that trivia onto the first new section's header so the
    replacement preserves the visual context of the original.

    A no-op when there was no prior section, when the new block is
    empty, or when the new block starts with an entries-only section
    (no header to attach the trivia to).
    """
    if prior_leading is None or not new_secs:
        return
    first = new_secs[0]
    if first.header is None:
        return
    first.header.leading = prior_leading


def _insert_section_block(
    doc_node: DocumentNode,
    insert_at: int,
    new_secs: Sequence[SectionNode],
    *,
    separate_within: bool = True,
) -> None:
    """Splice a freshly-built block of ``[ ... ]`` / ``[[ ... ]]`` sections.

    A blank line is inserted before the block whenever
    ``sections[:insert_at]`` already holds rendered content. With
    ``separate_within=True`` (the default) consecutive entries within
    ``new_secs`` are also blank-separated; pass ``False`` when each
    section already carries its own inter-header trivia (e.g. cloned
    sections from another document).
    """
    sections = doc_node.sections
    preceding_has_content = any(
        s.header is not None or s.entries for s in sections[:insert_at]
    )
    for i, ns in enumerate(new_secs):
        if (i == 0 and preceding_has_content) or (i > 0 and separate_within):
            assert ns.header is not None
            pieces = ns.header.leading.pieces
            # Already starts with a blank line — don't double it up.
            if not (pieces and isinstance(pieces[0], NewlineNode)):
                pieces.insert(0, NewlineNode("\n"))
    if new_secs and new_secs[0].header is not None:
        doc_node.adopt_preamble_into(new_secs[0].header.leading)
    sections[insert_at:insert_at] = new_secs


def _parse_key_path(path: str | tuple[str, ...]) -> tuple[str, ...]:
    """Split a dotted key path string into its bare segments.

    A string is split on ``.`` (``"a.b.c"`` → ``("a", "b", "c")``).
    A tuple is taken verbatim: use this form to express a single
    segment containing a literal dot (e.g. ``("foo.bar",)`` to name a
    table whose only key is the quoted name ``"foo.bar"``).
    """
    if isinstance(path, tuple):
        if not path:
            msg = "key path must not be empty"
            raise TOMLError(msg)
        for p in path:
            if not isinstance(p, str):
                msg = (  # type: ignore[unreachable]
                    f"key path {path!r} segment must be str, not {type(p).__name__}"
                )
                raise TypeError(msg)
        if any(p == "" for p in path):
            msg = f"key path {path!r} contains an empty segment"
            raise TOMLError(msg)
        return path
    if not isinstance(path, str):
        msg = (  # type: ignore[unreachable]
            f"key path must be str or tuple of str, not {type(path).__name__}"
        )
        raise TypeError(msg)
    if not path:
        msg = "key path must not be empty"
        raise TOMLError(msg)
    parts = path.split(".")
    if any(not p for p in parts):
        msg = f"key path {path!r} contains an empty segment"
        raise TOMLError(msg)
    return tuple(parts)


def _section_insert_index(
    sections: list[SectionNode],
    full_path: tuple[str, ...],
) -> int:
    """Choose where to splice a new section with ``full_path``.

    Prefers placement immediately after the last section that shares
    ``full_path``'s parent prefix (so siblings group together); falls
    back to the end of the document.
    """
    parent = full_path[:-1]
    last_sibling = -1
    for i, sec in enumerate(sections):
        hdr = sec.header
        if hdr is None:
            continue
        hpath = hdr.key.path
        if len(hpath) >= len(parent) and hpath[: len(parent)] == parent:
            last_sibling = i
    if last_sibling < 0:
        return len(sections)
    return last_sibling + 1


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class SectionSpec(dict[str, Any]):
    """Tag telling ``__setitem__`` to install a ``[k]`` standard section.

    Produced by :meth:`Table.section`; used only at assignment sites::

        doc["tool"] = Table.section({"version": 1})  # [tool] section

    A plain ``dict`` assignment would instead produce an inline table
    (``tool = { version = 1 }``).
    """

    __slots__ = ()


class Table(dict[str, Any]):
    """A logical TOML table.

    All mapping flavours in tomlrt (top-level document, standard
    table, inline table, and the synthetic mappings spawned by dotted
    keys) inherit from :class:`Table`, which is itself a subclass of
    :class:`dict`. So values typed as ``Table`` cover every nested
    mapping you can encounter while walking a document, *and*
    ``isinstance(t, dict)`` is ``True`` and ``**t`` works.

    .. rubric:: Storage model

    A :class:`Table` is a *view* over the parsed concrete syntax
    tree (CST) — the physical tree of nodes that records every
    byte of the original document, including whitespace, comments,
    quote style and key order. Every mutation writes to the CST
    first and the dict storage is then refreshed from there. The
    CST is the single source of truth — :meth:`render` and every
    iteration ultimately read from it; the dict storage is a cache
    that mirrors the CST data and exists for two reasons:

    * fast ``dict``-style lookup, ``len``, ``in``, iteration, and
      ``**`` unpacking; and
    * stable object identity for nested containers, so that
      ``doc["foo"] is doc["foo"]``.

    Once a :class:`Table` is *detached* (see below) the CST link is
    severed and the dict storage takes over as the only source of
    truth for that orphan subtree.

    .. rubric:: Held references

    Held references behave like ordinary Python dict references:

    * If the binding goes away (``del doc['foo']``), the held
      ``Table`` is *orphaned*: its dict storage is intact and reads
      still work, but it is no longer connected to the document and
      mutations through it do not appear in :meth:`Document.render`.
    * Re-binding the path (``doc['foo'] = {...}`` or
      ``doc.set_table('foo', {...})``) installs a *fresh* ``Table``;
      held references to the old table are unaffected.
    """

    __slots__ = ("_attached",)

    def __init__(self) -> None:
        super().__init__()
        self._attached = True

    # --- factories ---------------------------------------------------------

    @classmethod
    def section(
        cls,
        mapping: Mapping[str, object] | None = None,
    ) -> SectionSpec:
        """Return a spec that installs as a ``[k]`` standard section.

        Use from an assignment site: ``doc[k] = Table.section({...})``.
        The spec is a dict subclass; you can build it up further before
        assignment (``spec["sub"] = ...``). Nested dicts in the mapping
        remain inline unless they are themselves :meth:`section` specs.
        """
        spec = SectionSpec()
        if mapping is not None:
            spec.update(mapping)
        return spec

    # --- subclass hooks ----------------------------------------------------

    def _items(self) -> Iterator[tuple[str, TomlValue]]:  # pragma: no cover
        raise NotImplementedError

    def _set_value(  # pragma: no cover
        self,
        key: str,
        value: object,
    ) -> TomlValue | None:
        """Mutate the CST so ``key`` binds to ``value``.

        Returns the new dict-storage value for ``key`` if the
        implementation can compute it cheaply; otherwise returns
        ``None`` to ask :meth:`__setitem__` to fall back to the full
        :meth:`_refresh_key` walk. Returning the value short-circuits
        an O(N) re-scan of the document for a single-key update.
        """
        raise NotImplementedError

    def _delete_value(self, key: str) -> None:  # pragma: no cover
        raise NotImplementedError

    # --- dict-storage sync helpers -----------------------------------------

    def _populate(self) -> None:
        """Refill dict storage from the CST. Called from subclass __init__."""
        super().clear()
        for k, v in self._items():
            super().__setitem__(k, v)

    def _refresh_key(self, key: str) -> None:
        """Re-read ``key`` from the CST after a mutation.

        If ``key`` is no longer present in the CST, it is removed from
        dict storage. Identity for *other* keys is preserved.
        """
        for k, v in self._items():
            if k == key:
                # dict.__setitem__ on an existing key preserves position;
                # on a new key it appends. Both match CST behaviour.
                super().__setitem__(key, v)
                return
        if super().__contains__(key):
            super().__delitem__(key)

    # --- attachment / detachment ------------------------------------------

    def _detach(self, doc_node: DocumentNode | None = None) -> None:
        """Mark this table (and any nested containers) as orphaned.

        After detachment, mutations through this object only affect its
        own (now-isolated) state; CST writeback no longer reaches the
        original document. ``doc_node`` is supplied when the parent has
        already created an orphan :class:`DocumentNode` covering the
        whole detached subtree; subclasses with section-based storage
        (:class:`_StdTable`, :class:`AoT`) use it to keep nested
        structural mutations confined to that orphan view.
        """
        self._attached = False
        for v in self.values():
            if isinstance(v, (Table, AoT)):
                v._detach(doc_node)  # noqa: SLF001
            elif isinstance(v, Array):
                v._detach()  # noqa: SLF001

    # --- mutators ----------------------------------------------------------

    def _commit_value(self, key: str, value: object) -> None:
        """Plain-value write at ``key`` + dict-storage reconcile.

        Used by ``_install_flavoured`` to commit values that didn't
        match any structural-install flavour. Lifted out of
        ``__setitem__`` so single-segment installs can finish without
        recursing back through it (which would loop, since
        ``__setitem__`` now unconditionally delegates to
        ``_install_flavoured``).
        """
        new_v = self._set_value(key, value)
        if new_v is None:
            self._refresh_key(key)
        else:
            dict.__setitem__(self, key, new_v)

    @override
    def __setitem__(self, key: str, value: object) -> None:
        if not self._attached:
            dict.__setitem__(self, key, value)
            return
        # Detach any container currently at ``key`` before we overwrite
        # it, so user-held references stop reflecting later document
        # edits. ``old is value`` is the augmented-assignment / self-
        # assignment case (``d[k] |= ...`` rebinds with the same
        # object): there's nothing to detach and nothing to re-install.
        if super().__contains__(key):
            old = super().__getitem__(key)
            if old is value:
                return
            if isinstance(old, (Table, AoT, Array)):
                old._detach()  # noqa: SLF001
        # All assignment flows -- structural and plain -- funnel through
        # ``_install_flavoured`` so each Table flavour can apply one
        # consistent dispatch policy. Plain values land in the
        # subclass's ``_commit_value`` fallback.
        self._install_flavoured((key,), value)

    def install(
        self,
        path: str | tuple[str, ...],
        value: object,
    ) -> Any:
        """Install ``value`` at ``path``, descending dotted segments.

        ``path`` accepts a dotted string (split on ``.``) or a tuple
        of literal segments (use the tuple form to express a segment
        containing a literal dot, e.g. ``("foo.bar",)``).

        ``value`` may be any of:

        * a spec from :meth:`Table.section` — installs a
          ``[...]`` standard section;
        * an :class:`AoT` built standalone (``AoT([{...}])``) — installs
          ``[[...]]`` array-of-tables entries;
        * an :class:`Array` built standalone (``Array([...],
          multiline=...)``) — installs an inline array with the
          requested layout;
        * any plain Python value (scalar, ``dict``, ``list``) —
          assigned at the leaf with ordinary ``__setitem__`` semantics
          (so a leaf ``dict`` becomes an inline table, a leaf ``list``
          becomes an inline array).

        Existing values at ``path`` (including sub-sections) are
        replaced. Implicit intermediate tables are left implicit, so
        ``install(("tool", "poetry"), Table.section({}))`` produces a
        single ``[tool.poetry]`` header, not a ``[tool]`` + nested.

        Returns the freshly-installed live view (:class:`Table`,
        :class:`AoT`, :class:`Array`) or the leaf value.
        """
        parts = _parse_key_path(path)
        # Reject install paths that would have to thread through an
        # array-of-tables before any CST mutation runs, so a rejected
        # call leaves the document untouched. AoT entries don't have
        # a single addressable child container; the user must address
        # a specific entry (``aot[i].install(...)``) instead. Other
        # non-table intermediates (scalars, inline tables) are caught
        # by their own dedicated error paths downstream.
        cur: Any = self
        walked = True
        for i, part in enumerate(parts[:-1]):
            if part not in cur:
                walked = False
                break
            nxt = cur[part]
            if isinstance(nxt, AoT):
                shown = ".".join(parts[: i + 1])
                msg = (
                    f"cannot install at {'.'.join(parts)!r}: {shown!r} is "
                    "an array-of-tables; address a specific entry instead "
                    f"(e.g. aot[i].install({parts[i + 1 :]!r}, ...))"
                )
                raise TOMLError(msg)
            if not isinstance(nxt, Table):
                walked = False
                break
            cur = nxt
        # Detach any container view currently at the leaf so user-held
        # references stop tracking the document after replacement.
        # ``__setitem__`` does this for single-key overwrites; do the
        # same here once we've located the leaf's parent.
        if walked and isinstance(cur, Table) and dict.__contains__(cur, parts[-1]):
            existing = dict.__getitem__(cur, parts[-1])
            if isinstance(existing, (Table, AoT, Array)) and existing is not value:
                existing._detach()  # noqa: SLF001
        self._install_flavoured(parts, value)
        leaf: Any = self
        for part in parts:
            leaf = leaf[part]
        return leaf

    def _install_flavoured(
        self,
        parts: tuple[str, ...],
        value: object,
    ) -> None:
        """Route a flavour-bearing value through the structural installers.

        Subclasses that support structural assignment (``_StdTable``,
        ``Document``) override this. The base implementation rejects
        ``SectionSpec`` / ``AoT`` because inline-style tables cannot
        hold ``[k]`` sections or ``[[k]]`` array-of-tables, and
        rejects multi-segment paths for the same reason. Standalone
        :class:`Array` is accepted at a single-segment path and
        installed as a plain list value (the ``multiline`` layout
        request is dropped; inline tables do not admit multi-line
        array values).
        """
        if isinstance(value, SectionSpec):
            msg = "cannot install a [section] inside an inline-style table"
            raise TOMLError(msg)
        if isinstance(value, AoT):
            msg = "cannot install an array-of-tables inside an inline-style table"
            raise TOMLError(msg)
        if isinstance(value, _StdTable):
            # An attached section-backed Table is the same kind of
            # "give me a [section] here" request as a SectionSpec; the
            # only difference is whether the user spelled the spec
            # themselves or copied an existing block. Refuse it for
            # the same reason — silently flattening it into the
            # inline host loses the [section] semantics.
            msg = "cannot install a [section]-style table inside an inline-style table"
            raise TOMLError(msg)
        if len(parts) > 1:
            path = ".".join(parts)
            msg = (
                f"cannot install at multi-segment path {path!r} inside "
                "an inline-style table"
            )
            raise TOMLError(msg)
        if isinstance(value, Array) and not value._attached:  # noqa: SLF001
            self._commit_value(parts[0], list(value))
            return
        self._commit_value(parts[0], value)

    @override
    def __delitem__(self, key: str) -> None:
        if not super().__contains__(key):
            raise KeyError(key)
        old = super().__getitem__(key)
        if isinstance(old, (Table, AoT, Array)):
            old._detach()  # noqa: SLF001
        if not self._attached:
            super().__delitem__(key)
            return
        self._delete_value(key)
        if super().__contains__(key):
            super().__delitem__(key)

    @override
    def clear(self) -> None:
        for k in list(self):
            del self[k]

    @override
    def update(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if len(args) > 1:
            msg = f"update expected at most 1 positional argument, got {len(args)}"
            raise TypeError(msg)
        if args:
            other = args[0]
            if hasattr(other, "keys"):
                for k in other.keys():  # noqa: SIM118
                    self[k] = other[k]
            else:
                for k, v in other:
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    @override
    def setdefault(self, key: str, default: Any = None) -> Any:
        if super().__contains__(key):
            return super().__getitem__(key)
        self[key] = default
        return super().__getitem__(key)

    @override
    def __ior__(self, other: object) -> Self:  # type: ignore[override]
        if isinstance(other, Mapping):
            self.update(other)
        else:
            self.update(dict(other))  # type: ignore[call-overload]
        return self

    @override
    def copy(self) -> dict[str, Any]:
        """Return a shallow plain ``dict`` copy of this table."""
        return dict(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a deep, plain-Python copy of this table.

        Walks the table recursively and converts every nested
        :class:`Table` / :class:`AoT` / :class:`Array` view into an
        ordinary :class:`dict` / :class:`list`. The result shares no
        mutable state with the document and is safe to hand to
        consumers that expect real ``dict``/``list`` objects -- JSON
        encoders, ``fastjsonschema``, ``pydantic``, anything that
        does ``isinstance(x, dict)``, etc.

        Scalar values (strings, ints, floats, bools, datetimes) are
        returned as-is; they are immutable so aliasing is harmless.
        """
        return {k: _to_plain(v) for k, v in self.items()}

    @override
    def pop(self, key: str, default: object = _MISSING) -> Any:
        """Remove ``key`` and return its value, like :meth:`dict.pop`.

        For :class:`Table` / :class:`AoT` / :class:`Array` values, the
        returned object is *orphaned*: it keeps its own data but is no
        longer attached to the document. Use :meth:`to_dict` /
        :meth:`Array.to_list` first if you need a plain-Python deep
        copy.
        """
        try:
            value = super().__getitem__(key)
        except KeyError:
            if default is _MISSING:
                raise
            return default
        del self[key]
        return value

    @override
    def popitem(self) -> tuple[str, Any]:
        if not self:
            msg = "table is empty"
            raise KeyError(msg)
        key = next(reversed(self))
        return key, self.pop(key)

    @override
    def __repr__(self) -> str:
        body = ", ".join(f"{k!r}: {v!r}" for k, v in self.items())
        return f"{type(self).__name__}({{{body}}})"

    # ------------------------------------------------------------------
    # Typed accessors. These are convenience views over ``__getitem__``
    # that narrow the return type so callers can chain into the
    # comment/header API without writing ``cast`` or ``isinstance``.
    # ------------------------------------------------------------------

    def table(self, key: str) -> Table:
        """Return the table at ``key``, typed as :class:`Table`.

        ``key`` accepts a dotted path (e.g. ``"tool.poetry"``). Raises
        :class:`KeyError` if any segment is missing, or :class:`TypeError`
        if the destination is not a table.
        """
        return self._typed_lookup(key, Table)

    @overload
    def get_table(self, key: str) -> Table | None: ...
    @overload
    def get_table(self, key: str, default: _T) -> Table | _T: ...
    def get_table(self, key: str, default: object = None) -> object:
        """Like :meth:`table`, but returns ``default`` if ``key`` is missing.

        Wrong-type entries still raise :class:`TypeError`: a missing key
        is "no answer", but an entry that exists with the wrong shape is
        a real bug worth surfacing.
        """
        return self._typed_lookup(key, Table, default=default)

    def array(self, key: str) -> Array:
        """Return the array at ``key``, typed as :class:`Array`.

        ``key`` accepts a dotted path. Raises :class:`KeyError` if any
        segment is missing, or :class:`TypeError` if the destination is
        not an inline array.
        """
        return self._typed_lookup(key, Array)

    @overload
    def get_array(self, key: str) -> Array | None: ...
    @overload
    def get_array(self, key: str, default: _T) -> Array | _T: ...
    def get_array(self, key: str, default: object = None) -> object:
        """Like :meth:`array`, but returns ``default`` if ``key`` is missing.

        Wrong-type entries still raise :class:`TypeError`.
        """
        return self._typed_lookup(key, Array, default=default)

    def aot(self, key: str) -> AoT:
        """Return the array-of-tables at ``key``, typed as :class:`AoT`.

        ``key`` accepts a dotted path. Raises :class:`KeyError` if any
        segment is missing, or :class:`TypeError` if the destination is
        not an array of tables.
        """
        return self._typed_lookup(key, AoT)

    @overload
    def get_aot(self, key: str) -> AoT | None: ...
    @overload
    def get_aot(self, key: str, default: _T) -> AoT | _T: ...
    def get_aot(self, key: str, default: object = None) -> object:
        """Like :meth:`aot`, but returns ``default`` if ``key`` is missing.

        Wrong-type entries still raise :class:`TypeError`.
        """
        return self._typed_lookup(key, AoT, default=default)

    @overload
    def _typed_lookup(self, key: str, expected: type[_T]) -> _T: ...
    @overload
    def _typed_lookup(
        self,
        key: str,
        expected: type[_T],
        *,
        default: object,
    ) -> object: ...
    def _typed_lookup(
        self,
        key: str,
        expected: type[_T],
        *,
        default: object = _MISSING,
    ) -> object:
        """Shared implementation for ``table`` / ``array`` / ``aot`` and
        their ``get_*`` variants. Without ``default``, missing keys
        re-raise :class:`KeyError`; otherwise ``default`` is returned.
        Wrong-type entries always raise :class:`TypeError`.
        """
        try:
            value = self._lookup_path(key)
        except KeyError:
            if default is _MISSING:
                raise
            return default
        if not isinstance(value, expected):
            article = "an" if expected.__name__[:1] in "AaEeIiOoUu" else "a"
            msg = (
                f"{key!r} is a {type(value).__name__}, "
                f"not {article} {expected.__name__}"
            )
            raise TypeError(msg)
        return value

    def _lookup_path(self, key: str) -> TomlValue:
        parts = _parse_key_path(key)
        cur: TomlValue = self
        for i, part in enumerate(parts):
            if not isinstance(cur, Table):
                head = ".".join(parts[:i])
                msg = (
                    f"cannot descend into {head!r}: it is a "
                    f"{type(cur).__name__}, not a Table"
                )
                raise TypeError(msg)
            cur = cur[part]
        return cur

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
        raise TOMLError(msg)

    @property
    def leading_comments(self) -> MutableMapping[str, tuple[str, ...]]:
        """Live mapping of ``key -> tuple of comment lines above it``.

        Only keys with a non-empty leading comment block are present.
        Assigning an empty tuple or deleting a key removes the block.
        """
        msg = "this table flavour does not support the comment API"
        raise TOMLError(msg)

    @property
    def header_comment(self) -> str | None:
        """End-of-line comment on this table's ``[name]`` / ``[[name]]`` line.

        ``None`` means the header has no trailing comment. Setting
        ``None`` or ``""`` removes any existing comment. Raises
        :class:`TOMLError` for the top-level :class:`Document`,
        for inline tables, and for any logical table that exists only
        through implicit parents (no physical header in source).

        For tables declared via multiple discontiguous ``[name]``
        sections, this refers to the *first* such header.
        """
        msg = "this table flavour does not support the header comment API"
        raise TOMLError(msg)

    @header_comment.setter
    def header_comment(self, value: str | None) -> None:  # noqa: ARG002
        msg = "this table flavour does not support the header comment API"
        raise TOMLError(msg)

    @header_comment.deleter
    def header_comment(self) -> None:
        msg = "this table flavour does not support the header comment API"
        raise TOMLError(msg)

    @property
    def header_leading_comments(self) -> tuple[str, ...]:
        """Comment lines immediately above this table's header.

        Returns the contiguous block of ``# ...`` lines ending right
        above the ``[name]`` / ``[[name]]`` line. Earlier blank-line
        separated comments are *not* included. Assigning an empty
        tuple removes the block. Raises like :attr:`header_comment`.
        """
        msg = "this table flavour does not support the header comment API"
        raise TOMLError(msg)

    @header_leading_comments.setter
    def header_leading_comments(self, value: Sequence[str]) -> None:  # noqa: ARG002
        msg = "this table flavour does not support the header comment API"
        raise TOMLError(msg)

    @header_leading_comments.deleter
    def header_leading_comments(self) -> None:
        msg = "this table flavour does not support the header comment API"
        raise TOMLError(msg)

    def promote_inline(self, key: str) -> Table:  # noqa: ARG002
        """Promote an inline-table-valued ``key`` to a standard table.

        After promotion the entry is rendered as a separate
        ``[parent.key]`` section, allowing comments and dotted-key
        expansions on its members.
        """
        msg = "this table flavour does not support inline-table promotion"
        raise TOMLError(msg)

    def promote_array(self, key: str) -> AoT:  # noqa: ARG002
        """Promote an array-of-inline-tables-valued ``key`` to an AoT.

        After promotion the entries are rendered as repeated
        ``[[parent.key]]`` sections, allowing comments and dotted-key
        expansions on each entry's members.
        """
        msg = "this table flavour does not support array-of-tables promotion"
        raise TOMLError(msg)

    def _install_section(
        self,
        parts: tuple[str, ...],  # noqa: ARG002
        value: Mapping[str, object] = MappingProxyType({}),  # noqa: ARG002
    ) -> Table:
        msg = (
            "cannot install a standard table here: this table flavour "
            "is not section-backed"
        )
        raise TOMLError(msg)

    def ensure_table(self, key: str | tuple[str, ...]) -> Table:
        """Return the table at ``key``, creating an empty one if absent.

        ``key`` accepts a dotted path as a string, or a tuple of
        literal segments (use the tuple form to express a segment
        containing a literal dot, e.g. ``("foo.bar",)``). If the
        destination already exists and is table-shaped (an explicit
        section, an implicit super-table, or an inline table), the
        existing live view is returned and no mutation occurs. Raises
        :class:`TOMLError` when the path names a non-table value.
        """
        parts = _parse_key_path(key)
        cur: Table = self
        for i, part in enumerate(parts):
            if part in cur:
                child = cur[part]
                if not isinstance(child, Table):
                    full = ".".join(parts[: i + 1])
                    msg = (
                        f"cannot ensure table at {full!r}: existing value is "
                        f"a {type(child).__name__}"
                    )
                    raise TOMLError(msg)
                cur = child
            else:
                return cur._install_section(parts[i:], {})  # noqa: SLF001
        return cur

    def _install_array(
        self,
        parts: tuple[str, ...],
        items: Iterable[object],
        *,
        multiline: bool,
        indent: str,
    ) -> Array:
        if len(parts) == 1:
            target: Table = self
        else:
            target = self.ensure_table(parts[:-1])
        leaf = parts[-1]
        target[leaf] = list(items)
        value = dict.__getitem__(target, leaf)
        if not isinstance(value, Array):  # pragma: no cover - defensive
            msg = f"expected Array after install, got {type(value).__name__}"
            raise TOMLError(msg)
        if multiline:
            value.set_multiline(multiline=True, indent=indent)
        return value


class _InlineTable(Table):
    """Mapping view over an :class:`InlineTableNode`.

    Also acts as the :class:`_DottedHost` for any dotted-key views
    derived from its entries — the inline table itself owns all the
    state (node, separator style, ``=`` padding) those views need.
    """

    __slots__ = ("_eq_padding", "_node", "_style")

    def __init__(self, node: InlineTableNode) -> None:
        super().__init__()
        self._node = node
        self._style = _sample_separator_style(node.entries, node.final_trivia)
        # ``=``-padding is per-entry, not a separator concern. Sample
        # from the first existing entry; default to a single space.
        if node.entries:
            self._eq_padding: tuple[WhitespaceNode | None, WhitespaceNode | None] = (
                node.entries[0].pre_eq,
                node.entries[0].post_eq,
            )
        else:
            self._eq_padding = (WhitespaceNode(" "), WhitespaceNode(" "))
        self._populate()

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
                yield head, _DottedSubTable(depth=1, host=self, prefix=(head,))

    def _find_entry(self, key: str) -> InlineTableEntry | None:
        for entry in self._node.entries:
            if len(entry.key.path) == 1 and entry.key.path[0] == key:
                return entry
        return None

    def _make_entry(self, path: tuple[str, ...], value: object) -> InlineTableEntry:
        pre, post = self._eq_padding
        # ``pre``/``post`` are WhitespaceNode|None value objects; sharing
        # the ref across entries is safe because their ``text`` field is
        # never mutated in place.
        return InlineTableEntry(
            leading=Trivia(),
            key=_make_dotted_key(path) if len(path) > 1 else make_simple_key(path[0]),
            pre_eq=pre,
            post_eq=post,
            value=value_to_node(value),
            trailing=Trivia(),
            has_comma=False,
            post_comma_trivia=Trivia(),
        )

    # --- _DottedHost protocol ------------------------------------------------

    def set_at(self, path: tuple[str, ...], value: object) -> None:
        # Preserve in-place update for an exact simple match so the
        # entry's surrounding trivia and position survive round-tripping.
        if len(path) == 1:
            existing = self._find_entry(path[0])
            if existing is not None:
                existing.value = value_to_node(value)
                return
        self._node.entries[:] = [
            e for e in self._node.entries if not _path_has_prefix(e.key.path, path)
        ]
        self._node.entries.append(self._make_entry(path, value))
        _apply_separator_style(self._node, self._style)

    def del_prefix(self, prefix: tuple[str, ...]) -> bool:
        kept = [
            e for e in self._node.entries if not _path_has_prefix(e.key.path, prefix)
        ]
        if len(kept) == len(self._node.entries):
            return False
        self._node.entries[:] = kept
        _apply_separator_style(self._node, self._style)
        return True

    def entries_under(
        self, prefix: tuple[str, ...]
    ) -> list[tuple[tuple[str, ...], ValueNode]]:
        plen = len(prefix)
        return [
            (e.key.path, e.value)
            for e in self._node.entries
            if len(e.key.path) > plen and e.key.path[:plen] == prefix
        ]

    # --- mapping mutation ----------------------------------------------------

    @override
    def _set_value(self, key: str, value: object) -> TomlValue | None:
        self.set_at((key,), value)
        return None

    @override
    def _delete_value(self, key: str) -> None:
        # Best-effort CST cleanup; presence is enforced by the caller
        # (``Table.__delitem__``) at the cache level.
        self.del_prefix((key,))


def _path_has_prefix(path: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return len(path) >= len(prefix) and path[: len(prefix)] == prefix


class _DottedHost(Protocol):
    """Mutation back-channel for a synthetic dotted-key sub-table.

    A host knows how to add/replace, remove, and *enumerate* dotted-key
    entries in the underlying physical container so that views built on
    top of it re-read live state instead of a stale snapshot.
    """

    def set_at(self, path: tuple[str, ...], value: object) -> None: ...

    def del_prefix(self, prefix: tuple[str, ...]) -> bool: ...

    def entries_under(
        self, prefix: tuple[str, ...]
    ) -> list[tuple[tuple[str, ...], ValueNode]]: ...


def _make_dotted_key(path: tuple[str, ...]) -> Key:
    parts = [make_key_part(p) for p in path]
    seps = ["."] * (len(parts) - 1)
    return Key(parts=parts, separators=seps)


class _SectionDottedHost:
    """Mutates dotted-key entries inside one or more :class:`SectionNode`."""

    __slots__ = ("_sections",)

    def __init__(self, sections: list[SectionNode]) -> None:
        self._sections = sections

    def set_at(self, path: tuple[str, ...], value: object) -> None:
        # Remove any existing entry at or under this path; remember which
        # section last hosted such an entry so the new dotted KV lands
        # near its predecessors when sections are split.
        host_sec: SectionNode | None = None
        for sec in self._sections:
            kept: list[KeyValueNode] = []
            for kv in sec.entries:
                if _path_has_prefix(kv.key.path, path):
                    host_sec = sec
                else:
                    kept.append(kv)
            if len(kept) != len(sec.entries):
                sec.entries[:] = kept
        if host_sec is None:
            # No existing entry at this path: pick the section that
            # already owns dotted entries with the same head, else last.
            head = path[0]
            host_sec = next(
                (
                    sec
                    for sec in self._sections
                    if any(kv.key.path and kv.key.path[0] == head for kv in sec.entries)
                ),
                self._sections[-1],
            )
        host_sec.entries.append(
            KeyValueNode(
                leading=Trivia(),
                key=_make_dotted_key(path),
                pre_eq=WhitespaceNode(" "),
                post_eq=WhitespaceNode(" "),
                value=value_to_node(value),
                trailing=None,
                trailing_comment=None,
                newline=NewlineNode("\n"),
            ),
        )

    def del_prefix(self, prefix: tuple[str, ...]) -> bool:
        any_removed = False
        for sec in self._sections:
            kept = [
                kv for kv in sec.entries if not _path_has_prefix(kv.key.path, prefix)
            ]
            if len(kept) != len(sec.entries):
                sec.entries[:] = kept
                any_removed = True
        return any_removed

    def entries_under(
        self, prefix: tuple[str, ...]
    ) -> list[tuple[tuple[str, ...], ValueNode]]:
        plen = len(prefix)
        out: list[tuple[tuple[str, ...], ValueNode]] = []
        for sec in self._sections:
            out.extend(
                (kv.key.path, kv.value)
                for kv in sec.entries
                if len(kv.key.path) > plen and kv.key.path[:plen] == prefix
            )
        return out


class _DottedSubTable(Table):
    """Synthetic table aggregating dotted-key entries.

    The view is *live*: entries are re-read from the host on each
    access, so mutations through this view (or a sibling view onto the
    same underlying container) are immediately visible.
    """

    __slots__ = ("_depth", "_host", "_prefix")

    def __init__(
        self,
        *,
        depth: int,
        host: _DottedHost,
        prefix: tuple[str, ...],
    ) -> None:
        super().__init__()
        self._depth = depth
        self._host = host
        self._prefix = prefix
        self._populate()

    @override
    def _items(self) -> Iterator[tuple[str, TomlValue]]:
        groups: dict[str, list[tuple[tuple[str, ...], ValueNode]]] = {}
        order: list[str] = []
        terminals: dict[str, ValueNode] = {}
        for path, value in self._host.entries_under(self._prefix):
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
                yield (
                    head,
                    _DottedSubTable(
                        depth=self._depth + 1,
                        host=self._host,
                        prefix=(*self._prefix, head),
                    ),
                )

    @override
    def _set_value(self, key: str, value: object) -> TomlValue | None:
        self._host.set_at((*self._prefix, key), value)
        return None

    @override
    def _delete_value(self, key: str) -> None:
        if key not in self:
            raise KeyError(key)
        self._host.del_prefix((*self._prefix, key))


class _StdTable(Table):
    """Standard TOML table view: aggregates physical sections by path."""

    __slots__ = (
        "_anchor",
        "_doc_node",
        "_init_extras",
        "_init_pool",
        "_owner_anchor",
        "_path",
    )

    def __init__(
        self,
        doc_node: DocumentNode,
        path: tuple[str, ...],
        *,
        anchor: SectionNode | None = None,
        owner_anchor: SectionNode | None = None,
        _pool: list[SectionNode] | None = None,
        _extras: list[tuple[tuple[str, ...], KeyValueNode]] | None = None,
    ) -> None:
        super().__init__()
        self._doc_node = doc_node
        self._path = path
        # ``anchor`` is set only for AoT entries: it is the [[path]]
        # section that owns this entry. With no anchor, all sections
        # whose header matches ``path`` are direct sections of this
        # table.
        self._anchor = anchor
        # ``owner_anchor`` is the AoT [[..]] section whose owned range
        # bounds this table's universe of sections. For an AoT entry
        # itself it is the entry's own anchor; for a sub-table inside
        # an AoT entry it is the enclosing entry's anchor; for tables
        # outside any AoT entry it is ``None`` (the whole document).
        # The section pool and the inherited dotted-key "extras" are
        # both *re-derived* from ``_doc_node`` on every read so that
        # purges and inserts can never desync the dict view from what
        # ``dumps`` would render.
        self._owner_anchor = owner_anchor if owner_anchor is not None else anchor
        # Transient hints, used only during the initial ``_populate``
        # call that runs from this constructor. The parent ``_iter_table``
        # already partitioned its section pool by next-head and computed
        # the dotted-key "extras" that extend into us, so we can skip
        # the full rescan and ancestor walk that post-mutation reads do.
        self._init_pool = _pool
        self._init_extras = _extras
        try:
            self._populate()
        finally:
            self._init_pool = None
            self._init_extras = None

    def _scope(self) -> list[SectionNode] | None:
        owner = self._owner_anchor
        if owner is None:
            return None
        return self._doc_node.aot_entry_block(owner)

    def _compute_extras(
        self,
    ) -> list[tuple[tuple[str, ...], KeyValueNode]] | None:
        """Inherited dotted KVs whose path passes through ``self._path``.

        For the root table this is always ``None``: the root has no
        ancestors. For a nested table at path ``P`` of length ``n``,
        scans every section whose header is a strict prefix of ``P``
        (or ``None`` for the implicit pre-header section) and returns,
        for each dotted KV in such a section that extends into ``P``,
        the relative path inside ``P`` plus the KV node itself.

        Used for post-mutation reads. Construction-time reads receive
        their extras pre-computed top-down by the parent ``_iter_table``.
        """
        plen = len(self._path)
        if plen == 0:
            return None
        scope = self._scope()
        sections = scope if scope is not None else self._doc_node.sections
        out: list[tuple[tuple[str, ...], KeyValueNode]] = []
        for sec in sections:
            hdr = sec.header
            host_path: tuple[str, ...] = hdr.key.path if hdr is not None else ()
            hlen = len(host_path)
            if hlen >= plen or host_path != self._path[:hlen]:
                continue
            for kv in sec.entries:
                full = (*host_path, *kv.key.path)
                if len(full) <= plen or full[:plen] != self._path:
                    continue
                out.append((full[plen:], kv))
        return out or None

    @override
    def _items(self) -> Iterator[tuple[str, TomlValue]]:
        if self._init_pool is not None:
            pool = self._init_pool
        else:
            scope = self._scope()
            pool = scope if scope is not None else self._doc_node.sections
        extras = (
            self._init_extras
            if self._init_extras is not None
            else self._compute_extras()
        )
        return _iter_table(
            self._doc_node,
            self._path,
            pool=pool,
            anchor=self._anchor,
            owner_anchor=self._owner_anchor,
            extras=extras,
        )

    def _direct_sections(self) -> list[SectionNode]:
        if self._anchor is not None:
            return [self._anchor]
        path = self._path
        scope = self._scope()
        sections = scope if scope is not None else self._doc_node.sections
        if path == ():
            return [s for s in sections if s.header is None]
        return [
            s
            for s in sections
            if s.header is not None
            and s.header.kind == "table"
            and s.header.key.path == path
        ]

    @override
    def _detach(self, doc_node: DocumentNode | None = None) -> None:
        if not self._attached:
            return
        if doc_node is None:
            # Top of detachment subtree: capture every section under our
            # path and move them into a private DocumentNode so later
            # structural mutations through this table cannot reach the
            # original document. AoT entries capture exactly the anchor
            # plus its owned sub-section run; everything else captures
            # all sections rooted at our path.
            captured = (
                self._doc_node.aot_entry_block(self._anchor)
                if self._anchor is not None
                else self._sections_under_path()
            )
            doc_node = DocumentNode(sections=list(captured))
        self._doc_node = doc_node
        # The captured sections form a self-contained little document;
        # there is no longer an enclosing AoT entry to bound our world.
        self._owner_anchor = self._anchor
        super()._detach(doc_node)

    def _sections_under_path(self) -> list[SectionNode]:
        plen = len(self._path)
        out: list[SectionNode] = []
        for sec in self._doc_node.sections:
            hdr = sec.header
            if hdr is None:
                continue
            hpath = hdr.key.path
            if (
                len(hpath) >= plen
                and hpath[:plen] == self._path
                and (len(hpath) > plen or hdr.kind == "table")
            ):
                out.append(sec)
        return out

    def _classify(self, key: str) -> tuple[str, object]:
        """Classify a key for mutation purposes.

        Returns one of:
            ("direct", KeyValueNode)         - a single-part scalar/value entry
            ("dotted", None)                 - dotted-key prefix (e.g. b.c=...)
            ("table", None)                  - child standard table [self.path.key]
            ("aot", None)                    - child AoT [[self.path.key]]
            ("extras", KeyValueNode)         - terminal entry living as a dotted
                                               KV in an ancestor section
            ("extras-prefix", None)          - extras entries with this head are
                                               longer than one segment
            ("absent", None)
        """
        if self._anchor is not None:
            # AoT-anchored slow path: direct sections is just the anchor;
            # child sections may live anywhere within the entry's owned
            # sub-section run, so search just that scope (not the whole
            # document) — otherwise siblings' same-path sub-sections
            # would be misattributed to this entry.
            for kv in self._anchor.entries:
                if kv.key.path[0] == key:
                    if len(kv.key.path) == 1:
                        return ("direct", kv)
                    return ("dotted", None)
            child = (*self._path, key)
            scope = self._scope()
            assert scope is not None  # _anchor is not None ⇒ owner_anchor is set
            for sec in scope:
                hdr = sec.header
                if hdr is None:
                    continue
                hpath = hdr.key.path
                if hpath == child:
                    return ("aot" if hdr.kind == "array" else "table", None)
                if len(hpath) > len(child) and hpath[: len(child)] == child:
                    # Deeper [a.b.c] or [[a.b.c]] makes ``key`` an
                    # implicit super-table — i.e. a table.
                    return ("table", None)
            return self._classify_extras(key)

        # Common path: not AoT-anchored. Fuse the direct-entries scan and the
        # child-section scan into a single pass over the document.
        path = self._path
        plen = len(path)
        child_len = plen + 1
        scope = self._scope()
        sections = scope if scope is not None else self._doc_node.sections
        child_kind: str | None = None
        for sec in sections:
            hdr = sec.header
            if hdr is None:
                if plen == 0:
                    for kv in sec.entries:
                        if kv.key.path[0] == key:
                            if len(kv.key.path) == 1:
                                return ("direct", kv)
                            return ("dotted", None)
                continue
            hpath = hdr.key.path
            if hdr.kind == "table" and hpath == path:
                for kv in sec.entries:
                    if kv.key.path[0] == key:
                        if len(kv.key.path) == 1:
                            return ("direct", kv)
                        return ("dotted", None)
                continue
            hlen = len(hpath)
            if hlen >= child_len and hpath[:plen] == path and hpath[plen] == key:
                if hdr.kind == "array" and hlen == child_len:
                    return ("aot", None)
                # A deeper section ([a.b.c] *or* [[a.b.c]]) below ``key``
                # makes ``key`` an implicit super-table — i.e. a table.
                child_kind = "table"
        if child_kind is not None:
            return (child_kind, None)
        return self._classify_extras(key)

    def _classify_extras(self, key: str) -> tuple[str, object]:
        """Tail of :meth:`_classify`: look for ancestor-section dotted KVs."""
        extras = self._compute_extras()
        if extras:
            terminal = None
            has_dotted = False
            for rel, kv in extras:
                if rel[0] != key:
                    continue
                if len(rel) == 1:
                    terminal = kv
                else:
                    has_dotted = True
            if terminal is not None:
                return ("extras", terminal)
            if has_dotted:
                return ("extras-prefix", None)
        return ("absent", None)

    def _purge_conflicting(self, key: str) -> None:
        """Remove any existing dotted, sub-table or AoT structure under ``key``.

        Used to give Python-dict-style overwrite semantics: assigning to
        a name that already names a sub-table silently destroys that
        sub-table (and any nested children) rather than raising. Also
        removes any ancestor-section dotted entries that contribute
        to ``self._path + (key, ...)`` so they don't survive as ghosts.
        Constrained to ``self._scope()`` so an AoT entry can't reach
        across its boundary and delete a sibling entry's same-path
        sub-section.

        Pure structural removal — does not touch top-blank trivia.
        Callers run :meth:`DocumentNode.normalise_top_blank` themselves
        once the larger operation is done, so a purge-then-splice
        sequence doesn't strip a soon-to-be-meaningful blank in the
        intermediate state.
        """
        for sec in self._direct_sections():
            sec.entries[:] = [kv for kv in sec.entries if kv.key.path[0] != key]
        prefix = (*self._path, key)
        plen = len(prefix)
        scope = self._scope()
        scope_ids = None if scope is None else {id(s) for s in scope}
        doc_sections = self._doc_node.sections
        doc_sections[:] = [
            sec
            for sec in doc_sections
            if not (
                (scope_ids is None or id(sec) in scope_ids)
                and sec.header is not None
                and len(sec.header.key.path) >= plen
                and sec.header.key.path[:plen] == prefix
            )
        ]
        # Drop any ancestor-section dotted KV that contributes to our
        # path + key (e.g. ``[tool] poetry.name = "x"`` when purging
        # ``name`` from the ``tool.poetry`` view). The ``hlen >= plen``
        # check below skips every section we just removed, so iterating
        # the pre-splice scope is safe.
        ppath_len = len(self._path)
        for sec in scope if scope is not None else doc_sections:
            hdr = sec.header
            host_path: tuple[str, ...] = hdr.key.path if hdr is not None else ()
            hlen = len(host_path)
            if hlen >= plen or host_path != prefix[:hlen]:
                continue
            sec.entries[:] = [
                kv
                for kv in sec.entries
                if not (
                    len(kv.key.path) > ppath_len - hlen
                    and (*host_path, *kv.key.path)[:plen] == prefix
                )
            ]

    @override
    def _set_value(self, key: str, value: object) -> TomlValue | None:
        kind, payload = self._classify(key)
        if kind in ("direct", "extras"):
            # In-place value swap: reuse the existing KV node.
            assert isinstance(payload, KeyValueNode)
            payload.value = value_to_node(value)
            return _value_for(payload.value)
        if kind in ("dotted", "table", "aot", "extras-prefix"):
            self._purge_conflicting(key)
            self._doc_node.normalise_top_blank()
        sections = self._direct_sections()
        if not sections:
            sections = [self._ensure_section()]
        target = sections[-1]
        indent = _detect_indent(target)
        new_kv = make_keyvalue_node(key, value, indent=indent)
        if _gaps_uniformly_blank([kv.leading for kv in target.entries[1:]]):
            new_kv.leading.pieces.insert(0, NewlineNode("\n"))
        _ensure_trailing_newline(target)
        # Migrate any parked preamble (only present when this is the
        # first content being added to a previously-empty doc) ahead of
        # the new KV. No-op once the doc has structural content.
        self._doc_node.adopt_preamble_into(new_kv.leading)
        target.entries.append(new_kv)
        # Top-level only: if this assignment is into the implicit
        # pre-header section and a ``[table]`` follows, ensure a blank
        # line separates the new key from that header.
        if self._path == () and target.header is None:
            doc_node = self._doc_node
            idx = doc_node.sections.index(target)
            if idx + 1 < len(doc_node.sections):
                next_header = doc_node.sections[idx + 1].header
                if next_header is not None:
                    _prepend_blank_line(next_header.leading)
        # The new dict-storage value is exactly what we just wrote;
        # caller can skip the full _refresh_key walk. Safe for every
        # kind because _purge_conflicting only removes things keyed by
        # ``key`` in this scope, so no other dict slot is invalidated.
        return _value_for(new_kv.value)

    @override
    def _delete_value(self, key: str) -> None:
        kind, payload = self._classify(key)
        if kind == "absent":
            # Nothing to remove from the CST. Either the caller already
            # validated presence at the cache level (``Table.__delitem__``
            # does), or the key is genuinely absent and the cache will
            # raise on its own. Either way, the CST has no work to do.
            return
        if kind in ("direct", "extras") and isinstance(payload, KeyValueNode):
            # Targeted removal: drop just the matching KV from its section.
            # Avoids the O(N) section walks and full-list rebuilds in
            # ``_purge_conflicting`` when there's nothing else to remove.
            target = payload
            scope = self._scope()
            sections = scope if scope is not None else self._doc_node.sections
            for sec in sections:
                if target in sec.entries:
                    self._doc_node.remove_entry(sec, target)
                    return
            return  # pragma: no cover - defensive: kv must be reachable
        self._purge_conflicting(key)
        self._doc_node.normalise_top_blank()

    def _ensure_section(self) -> SectionNode:
        """Materialise a section that holds direct entries for ``self._path``."""
        if self._path == ():
            return self._ensure_root_section()
        return self._ensure_nested_section()

    def _ensure_root_section(self) -> SectionNode:
        """Insert an implicit pre-header section at the top of the document."""
        doc_node = self._doc_node
        new_sec = SectionNode(header=None, entries=[])
        # Ensure a blank line precedes the next section's header so the
        # newly-inserted top-level keys aren't visually glued to it.
        if doc_node.sections and doc_node.sections[0].header is not None:
            _prepend_blank_line(doc_node.sections[0].header.leading)
        doc_node.sections.insert(0, new_sec)
        return new_sec

    def _ensure_nested_section(self) -> SectionNode:
        """Insert a fresh ``[a.b.c]`` header for a nested path.

        Placement is immediately before the first descendant section so
        the new keys logically belong to the same place in the document.
        Falls back to appending when there is no descendant.
        """
        doc_node = self._doc_node
        new_sec = _new_section(self._path)
        assert new_sec.header is not None
        header = new_sec.header
        plen = len(self._path)
        for i, sec in enumerate(doc_node.sections):
            h = sec.header
            if (
                h is not None
                and len(h.key.path) > plen
                and h.key.path[:plen] == self._path
            ):
                # Insert a leading newline so the new header doesn't
                # glue against the previous section's last entry.
                header.leading.pieces.append(NewlineNode("\n"))
                doc_node.adopt_preamble_into(header.leading)
                doc_node.sections.insert(i, new_sec)
                return new_sec
        if any(s.header is not None or s.entries for s in doc_node.sections):
            header.leading.pieces.append(NewlineNode("\n"))
        doc_node.adopt_preamble_into(header.leading)
        doc_node.sections.append(new_sec)
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
        raise TOMLError(msg)

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
        sec, kv, inline = self._find_promotable(key, InlineTableNode, "an inline table")
        if _value_has_inner_comment(inline):
            msg = (
                f"cannot promote {key!r} to [..]: inline table has inner "
                "comments that would be lost; remove the inner comments first"
            )
            raise TOMLError(msg)
        child_path = (*self._path, key)
        self._refuse_existing_promoted_section(key, child_path, kind="table")
        new_sec = _build_promoted_section(child_path, inline, kv)
        self._doc_node.remove_entry(sec, kv)
        self._splice_promoted_sections([new_sec])
        view = _StdTable(self._doc_node, child_path)
        dict.__setitem__(self, key, view)
        return view

    @override
    def promote_array(self, key: str) -> AoT:
        sec, kv, array = self._find_promotable(key, ArrayNode, "an array")
        items = array.items
        if not items:
            msg = f"{key!r} is an empty array; cannot promote to array-of-tables"
            raise TOMLError(msg)
        for item in items:
            if not isinstance(item.value, InlineTableNode):
                msg = (
                    f"{key!r} contains a non-inline-table element; cannot "
                    "promote to array-of-tables"
                )
                raise TOMLError(msg)
        if _value_has_inner_comment(array):
            msg = (
                f"cannot promote {key!r} to [[..]]: array has inner comments "
                "that would be lost; remove the inner comments first"
            )
            raise TOMLError(msg)
        child_path = (*self._path, key)
        self._refuse_existing_promoted_section(key, child_path, kind="aot")
        new_secs = [
            _build_promoted_aot_section(child_path, item.value)
            for item in items
            if isinstance(item.value, InlineTableNode)  # for type narrowing
        ]
        # Carry the source KV's authoring trivia onto the new AoT
        # entries: leading comments / blank lines that sat above the
        # inline assignment go on the first ``[[..]]`` header, and any
        # trailing whitespace / EOL comment goes after the last entry's
        # final value. Without this, ``promote_array`` silently drops
        # the user's comments.
        if new_secs:
            first_hdr = new_secs[0].header
            assert first_hdr is not None
            first_hdr.leading.pieces[:0] = list(kv.leading.pieces)
            last_entries = new_secs[-1].entries
            if last_entries:
                last_entries[-1].trailing = kv.trailing
                last_entries[-1].trailing_comment = kv.trailing_comment
        self._doc_node.remove_entry(sec, kv)
        self._splice_promoted_sections(new_secs)
        aot = AoT._attached_to(self._doc_node, child_path, [])  # noqa: SLF001
        aot._resync()  # noqa: SLF001
        dict.__setitem__(self, key, aot)
        return aot

    def _find_promotable(
        self,
        key: str,
        expected: type[_T],
        label: str,
    ) -> tuple[SectionNode, KeyValueNode, _T]:
        """Find a direct KV at ``key`` whose value is an ``expected`` node.

        Raises a friendly :class:`TOMLError` if the key exists under a
        different shape (sub-section, dotted-key subtable, AoT) or if
        the value is the wrong node type. ``label`` is the
        "an inline table" / "an array" phrasing used in the message.
        Returns the host section, the KV, and the typed value node so
        callers don't need to re-narrow.
        """
        try:
            sec, kv = self._find_direct_kv(key)
        except KeyError:
            # If the key exists under a different shape (sub-section,
            # dotted-key subtable, or AoT), raise a clearer error rather
            # than a bare KeyError that contradicts ``key in self``.
            if key in self:
                msg = f"{key!r} is not {label}; nothing to promote"
                raise TOMLError(msg) from None
            raise
        if not isinstance(kv.value, expected):
            msg = f"{key!r} is not {label}; nothing to promote"
            raise TOMLError(msg)
        return sec, kv, kv.value

    def _refuse_existing_promoted_section(
        self,
        key: str,
        child_path: tuple[str, ...],
        *,
        kind: str,
    ) -> None:
        """Defensive: refuse if ``[child_path]`` (or ``[[child_path]]``)
        already exists. The parser blocks any source where this would
        arise and assignment auto-purges conflicts, so this only fires
        under direct CST manipulation.
        """
        for existing in self._doc_node.sections:
            hdr = existing.header
            if hdr is not None and hdr.key.path == child_path:  # pragma: no cover
                joined = ".".join(child_path)
                if kind == "aot":
                    msg = (
                        f"cannot promote {key!r}: a [[{joined}]] (or "
                        f"[{joined}]) section already exists"
                    )
                else:
                    msg = f"cannot promote {key!r}: a [{joined}] section already exists"
                raise TOMLError(msg)

    def _splice_promoted_sections(self, new_secs: Sequence[SectionNode]) -> None:
        """Splice freshly-promoted sections after the parent's last
        direct section (or at end of document if the parent has none).
        Uses ``_insert_section_block`` so consecutive AoT entries stay
        blank-separated.
        """
        sections = self._doc_node.sections
        parent_secs = self._direct_sections()
        if parent_secs:
            anchor = parent_secs[-1]
            insert_at = (
                next(
                    (i for i, s in enumerate(sections) if s is anchor),
                    len(sections) - 1,
                )
                + 1
            )
        else:
            insert_at = len(sections)
        _insert_section_block(self._doc_node, insert_at, new_secs)

    @override
    def _install_flavoured(self, parts: tuple[str, ...], value: object) -> None:
        if self._dispatch_structural(parts, value):
            return
        # Plain value (or same-slot identity case skipped by
        # ``_dispatch_structural``). For a single-segment install
        # commit at ``self`` directly via ``_commit_value`` -- going
        # through ``self[parts[0]] = value`` would re-enter
        # ``__setitem__``, which loops back through us. For a
        # multi-segment install, descend (creating implicit parents
        # as needed) and assign at the leaf with normal
        # ``__setitem__`` semantics.
        if len(parts) == 1:
            self._commit_value(parts[0], value)
            return
        target = self.ensure_table(parts[:-1])
        target[parts[-1]] = value

    def _dispatch_structural(
        self,
        parts: tuple[str, ...],
        value: object,
    ) -> bool:
        """Try to install ``value`` at ``parts`` as a structural edit.

        Returns ``True`` when the value carried enough flavour to drive
        a section / AoT / array install (so the caller should stop);
        ``False`` for plain values that need the ordinary value-write
        path. Centralising the dispatch keeps ``__setitem__`` /
        :meth:`_set_value` (single-segment writes) and
        :meth:`Document.install` (multi-segment writes) on the same
        decision tree.
        """
        if isinstance(value, SectionSpec):
            self._install_section(parts, value)
            return True
        if isinstance(value, AoT):
            # Both attached and detached AoTs have a backing CST: deep-clone
            # the source sections so comments, formatting, and per-value
            # layout (e.g. multiline arrays set on entries before the AoT
            # was installed) survive the move. Routing detached AoTs
            # through ``to_dict()`` here would silently strip all of that.
            self._install_attached_aot(parts, value)
            return True
        if isinstance(value, _StdTable) and not (
            value._doc_node is self._doc_node  # noqa: SLF001
            and value._path == (*self._path, *parts)  # noqa: SLF001
        ):
            # Section-backed Table: deep-clone the source CST so
            # comments and formatting survive, and so any nested AoT
            # lands as ``[[..]]`` rather than crashing the inline-table
            # synthesiser. Skip when the value is already installed at
            # the target path (e.g. during ``deepcopy`` reconstruction).
            self._install_attached_table(parts, value)
            return True
        if isinstance(value, Array) and not value._attached:  # noqa: SLF001
            # Standalone Arrays are specs: re-synthesise at the target
            # with the requested layout. Attached Arrays take the
            # deepcopy path below so comments/formatting survive.
            self._install_array(
                parts,
                list(value),
                multiline=value.multiline,
                indent=value._indent,  # noqa: SLF001
            )
            return True
        return False

    def _prepare_section_slot(
        self,
        parts: tuple[str, ...],
    ) -> tuple[tuple[str, ...], int, Trivia | None]:
        """Purge any conflicting value at ``parts`` and pick an insert index.

        Returns ``(full_path, insert_at, prior_leading)`` where:

        * ``full_path`` is the absolute CST path (``self._path + parts``).
        * ``insert_at`` is the position in ``self._doc_node.sections`` where
          a new block for ``full_path`` should be spliced in.
        * ``prior_leading`` is the leading trivia of the first matching
          section's header captured *before* purging, or ``None`` if no
          such section existed. Callers should transplant it onto the
          first new section's header so an in-place replacement preserves
          the comments / blank lines that sat above the original.

        Drops any redundant empty placeholder header at ``self._path``
        first so it doesn't survive as visual noise once the new child
        section is in place.

        If a section (or descendant of one) already sits at
        ``full_path``, remember its position before purging and reuse
        that slot, so replacing a section in place preserves its
        position among siblings instead of being appended at the end.
        """
        self._drop_redundant_anchor()
        full_path = (*self._path, *parts)
        plen = len(full_path)
        scope = self._scope()
        scope_ids = None if scope is None else {id(s) for s in scope}
        prior_index: int | None = None
        prior_leading: Trivia | None = None
        for i, sec in enumerate(self._doc_node.sections):
            hdr = sec.header
            if (
                hdr is not None
                and (scope_ids is None or id(sec) in scope_ids)
                and len(hdr.key.path) >= plen
                and hdr.key.path[:plen] == full_path
            ):
                prior_index = i
                # Capture only when the matched section's header is exactly
                # at full_path: a deeper match (e.g. ``[a.b.c]`` while we
                # replace ``[a.b]``) is a sub-section, not the slot itself,
                # and its leading belongs to it rather than to the parent.
                if len(hdr.key.path) == plen:
                    prior_leading = _clone_trivia(hdr.leading)
                break
        if len(parts) == 1:
            kind, _ = self._classify(parts[0])
            if kind != "absent":
                # Skip the top-blank normalisation here: we are about
                # to splice a replacement section into the slot we just
                # vacated, so a leading blank on whatever section sat
                # *behind* the purged one will still be a meaningful
                # inter-section separator once the new block is in
                # place. Normalising now strips it prematurely.
                self._purge_conflicting(parts[0])
        else:
            self._doc_node.purge_path(full_path)
        sections = self._doc_node.sections
        if prior_index is not None:
            # Only matching sections (within scope) get purged, so the
            # first match's index is preserved as a valid splice point.
            assert prior_index <= len(sections)
            return full_path, prior_index, prior_leading
        owner = self._owner_anchor
        if owner is None:
            return full_path, _section_insert_index(sections, full_path), None
        # AoT-entry sub-table with no prior section at this path:
        # pin the new section to the end of this entry's owned range
        # so it doesn't get re-attributed to a later entry on round-trip.
        return (
            full_path,
            sections.index(owner) + len(self._scope() or ()),
            None,
        )

    def _install_attached_aot(
        self,
        parts: tuple[str, ...],
        value: AoT,
    ) -> AoT:
        """Deep-clone an attached source AoT into ``parts``."""
        full_path = self._splice_attached(parts, value, _clone_aot_sections)
        aot = AoT._attached_to(self._doc_node, full_path, [])  # noqa: SLF001
        aot._resync()  # noqa: SLF001
        self._install_at_path(parts, aot)
        return aot

    @override
    def _install_section(
        self,
        parts: tuple[str, ...],
        value: Mapping[str, object] = MappingProxyType({}),
    ) -> Table:
        full_path, insert_at, prior_leading = self._prepare_section_slot(parts)
        new_sec = _new_section(full_path)
        new_sec.synthesised_placeholder = True
        _apply_prior_leading([new_sec], prior_leading)
        _insert_section_block(self._doc_node, insert_at, [new_sec])
        # Inherit ``owner_anchor`` from the parent so a sub-section
        # installed inside an AoT entry stays scoped to that entry —
        # otherwise reads/writes through ``view`` see same-named
        # sections in sibling entries and silently merge their values.
        view = _StdTable(self._doc_node, full_path, owner_anchor=self._owner_anchor)
        self._install_at_path(parts, view)
        for k, v in value.items():
            view[k] = v
        return view

    def _drop_redundant_anchor(self) -> None:
        """Drop an empty placeholder ``[X]`` header at ``self._path``.

        Called before installing a child section under this view. An
        empty ``[X]`` header that holds no entries and no comments
        serves no purpose once a child ``[X.Y]`` header follows it: the
        parent table is implied. AoT entry anchors (``[[X]]``) and
        anything carrying user comments are preserved verbatim.
        """
        if self._path == ():
            return
        for sec in self._doc_node.sections:
            hdr = sec.header
            if (
                hdr is None
                or hdr.kind != "table"
                or not sec.synthesised_placeholder
                or hdr.key.path != self._path
                or sec.entries
                or hdr.trailing_comment is not None
                or _trivia_has_comment(hdr.leading)
            ):
                continue
            self._doc_node.remove_sections({sec})
            if self._anchor is sec:
                self._anchor = None
            if self._owner_anchor is sec:
                self._owner_anchor = None
            return

    def _install_attached_table(
        self,
        parts: tuple[str, ...],
        value: _StdTable,
    ) -> _StdTable:
        """Deep-clone an attached source ``_StdTable`` into ``parts``.

        Implicit super-tables in the source remain implicit in the target —
        no empty intermediate ``[a]`` / ``[a.b]`` headers are emitted.
        """
        full_path = self._splice_attached(parts, value, _clone_table_sections)
        view = _StdTable(self._doc_node, full_path, owner_anchor=self._owner_anchor)
        self._install_at_path(parts, view)
        return view

    def _splice_attached(
        self,
        parts: tuple[str, ...],
        value: _StdTable | AoT,
        cloner: Callable[[Any, tuple[str, ...]], list[SectionNode]],
    ) -> tuple[str, ...]:
        """Common purge-and-splice for both attached-section installers.

        Snapshots the source CST *before* purging the destination slot so
        same-document calls where ``parts`` overlaps ``value._path``
        (e.g. ``doc["a"] = doc["a"]["b"]``,
        ``doc.install("a", doc.aot("a.inner"))``) still see their source.
        Returns the absolute target path, ready for the caller to wrap
        in a view.
        """
        full_path = (*self._path, *parts)
        new_secs = cloner(value, full_path)
        _full_path, insert_at, prior_leading = self._prepare_section_slot(parts)
        if new_secs:
            _apply_prior_leading(new_secs, prior_leading)
            _insert_section_block(
                self._doc_node,
                insert_at,
                new_secs,
                separate_within=False,
            )
        return full_path

    def _install_at_path(self, parts: tuple[str, ...], obj: object) -> None:
        """Install ``obj`` at the leaf of ``parts``, materialising any
        intermediate implicit super-tables in dict storage as we go.

        CST mutations are assumed to have already been performed; this
        method only reconciles the dict-storage view.
        """
        cur: Table = self
        for part in parts[:-1]:
            existing = super(Table, cur).get(part)
            if not isinstance(existing, Table):
                # Either absent (implicit super-table just materialised
                # in the CST) or replaced by a non-table en route: refresh
                # from the CST so dict storage matches.
                cur._refresh_key(part)  # noqa: SLF001
            nxt = super(Table, cur).__getitem__(part)
            assert isinstance(nxt, Table)
            cur = nxt
        super(Table, cur).__setitem__(parts[-1], obj)


class Document(_StdTable):
    """Top-level TOML document. Subclass of :class:`Table`."""

    __slots__ = ("_newline",)

    def __init__(self, node: DocumentNode) -> None:
        # Hand the construction walk the full section list and an
        # empty extras tuple. ``_iter_table`` then partitions sections
        # by head as it descends, so each nested ``_StdTable`` only
        # sees its own slice of the document — no per-level rescans.
        super().__init__(node, (), _pool=node.sections, _extras=[])
        self._newline = _detect_newline(node)

    @property
    def cst(self) -> DocumentNode:
        """The underlying concrete syntax tree (CST).

        Returns the root :class:`~tomlrt._nodes.DocumentNode` that
        records the document's exact byte layout. Intended for
        tooling and debugging — most users will never need this.
        """
        return self._doc_node

    def render(self) -> str:
        if self._newline != "\n":
            _normalise_newlines(self._doc_node, self._newline)
        return self._doc_node.render()

    def __copy__(self) -> Document:
        # The CST is the source of truth; sharing it across "copies" would
        # mean mutations on one bled into the other. Always clone.
        return Document(deepcopy(self._doc_node))

    def __deepcopy__(self, memo: dict[int, Any]) -> Document:
        return Document(deepcopy(self._doc_node, memo))

    @property
    def preamble(self) -> tuple[str, ...]:
        """Comment block at the top of the document.

        A "preamble" is the run of ``# …`` lines that opens the file
        and is blank-line-separated from anything below. Comments that
        sit directly above the first key (no blank line) are *not*
        preamble — they are the leading comments of that key, accessed
        via :attr:`leading_comments`. In a document with no structural
        content, the entire opening comment block is treated as
        preamble.

        Setter accepts a sequence of bare comment texts (without the
        leading ``#``) and replaces the current preamble; assign ``()``
        to remove. Newlines inside any line are rejected.
        """
        target = self._doc_node.preamble_target()
        pieces = target.pieces
        end, comments = _scan_leading_comment_run(pieces)
        if not comments:
            return ()
        has_separator = end < len(pieces) and isinstance(pieces[end], NewlineNode)
        if has_separator or not self._doc_node.has_content():
            return tuple(_strip_comment_marker(c) for c in comments)
        return ()

    @preamble.setter
    def preamble(self, value: Sequence[str]) -> None:
        _validate_comment_lines(value)
        target = self._doc_node.preamble_target()
        pieces = target.pieces
        has_content = self._doc_node.has_content()
        run_end, _ = _scan_leading_comment_run(pieces)
        has_separator = run_end < len(pieces) and isinstance(
            pieces[run_end], NewlineNode
        )
        # Drop the existing preamble run plus exactly one separator NL,
        # but only if the run is genuinely preamble (separated, or doc empty).
        is_preamble = has_separator or not has_content
        consume = (run_end + (1 if has_separator else 0)) if is_preamble else 0
        new: list[TriviaPiece] = []
        for line in value:
            new += [CommentNode(text=_format_comment(line)), NewlineNode("\n")]
        if value and has_content:
            new.append(NewlineNode("\n"))
        target.pieces = new + list(pieces[consume:])

    @property
    def epilogue(self) -> tuple[str, ...]:
        """Comment block at the very end of the document.

        Returns the trailing run of ``# …`` lines that follows all
        structural content. Empty when the document has no structural
        content (in that case everything is :attr:`preamble`).

        Setter accepts a sequence of bare comment texts and replaces
        the current epilogue. Assign ``()`` to remove. Raises
        :class:`TOMLError` if called with a non-empty value on a
        document with no structural content.
        """
        if not self._doc_node.has_content():
            return ()
        return _extract_trailing_comment_block(self._doc_node.trailing_trivia)

    @epilogue.setter
    def epilogue(self, value: Sequence[str]) -> None:
        if not self._doc_node.has_content():
            if value:
                msg = (
                    "cannot set epilogue on a document with no structural "
                    "content; use preamble instead"
                )
                raise TOMLError(msg)
            return
        _replace_trailing_comment_block(self._doc_node.trailing_trivia, value, "")


# ---------------------------------------------------------------------------
# Comment views
# ---------------------------------------------------------------------------


_VK = TypeVar("_VK")
_VV = TypeVar("_VV")


class _PresenceFilteredView(MutableMapping[_VK, _VV]):
    """Common scaffolding for the comment-views.

    A "presence-filtered" view exposes only those keys whose payload is
    currently *present* (a non-empty comment block, an EOL comment that
    actually exists, etc.). Subclasses provide:

    * :meth:`_check_key` — coerce / range-check a raw key, raising
      :class:`TypeError` for the wrong kind and :class:`KeyError` for
      an out-of-range value. Returns the canonicalised key.
    * :meth:`_keys` — yields every valid key (regardless of whether
      its payload is present).
    * :meth:`_read` — returns the payload, or ``None`` when absent.
    * :meth:`_write_absent` — the deletion primitive used by
      ``__delitem__``.
    * :meth:`_format_value` — used by ``__repr__``; defaults to
      ``repr``.

    ``__setitem__`` is left to subclasses because each view has its own
    slot-selection / anchor logic.
    """

    def _check_key(self, key: object) -> _VK:
        raise NotImplementedError

    def _keys(self) -> Iterator[_VK]:
        raise NotImplementedError

    def _read(self, key: _VK) -> _VV | None:
        raise NotImplementedError

    def _write_absent(self, key: _VK) -> None:
        raise NotImplementedError

    def _format_value(self, value: _VV) -> str:
        return repr(value)

    @override
    def __getitem__(self, key: _VK) -> _VV:
        k = self._check_key(key)
        v = self._read(k)
        if v is None:
            raise KeyError(key)
        return v

    @override
    def __delitem__(self, key: _VK) -> None:
        k = self._check_key(key)
        if self._read(k) is None:
            raise KeyError(key)
        self._write_absent(k)

    @override
    def __iter__(self) -> Iterator[_VK]:
        return (k for k in self._keys() if self._read(k) is not None)

    @override
    def __len__(self) -> int:
        return sum(1 for _ in self)

    @override
    def __contains__(self, key: object) -> bool:
        try:
            k = self._check_key(key)
        except (TypeError, KeyError):
            return False
        return self._read(k) is not None

    @override
    def __repr__(self) -> str:
        body = ", ".join(f"{k!r}: {self._format_value(v)}" for k, v in self.items())
        return f"{type(self).__name__}({{{body}}})"


class _TableKVViewBase(_PresenceFilteredView[str, _VV]):
    """Common scaffolding for :class:`_StdTable`-backed presence-filtered views.

    The table's single-segment ``KeyValueNode`` entries form the key
    universe. Subclasses provide the value-shaped methods (``_read``,
    ``_write_absent``, ``__setitem__``, optional ``_format_value``) and
    set ``_view_name`` for error messages.
    """

    __slots__ = ("_table",)

    _view_name: ClassVar[str]

    def __init__(self, table: _StdTable) -> None:
        self._table = table

    def _find_kv(self, key: str) -> KeyValueNode | None:
        try:
            _, kv = self._table._find_direct_kv(key)  # noqa: SLF001
        except KeyError:
            return None
        return kv

    @override
    def _check_key(self, key: object) -> str:
        if not isinstance(key, str):
            msg = f"Table.{self._view_name} key must be str, got {type(key).__name__}"
            raise TypeError(msg)
        return key

    @override
    def _keys(self) -> Iterator[str]:
        for sec in self._table._direct_sections():  # noqa: SLF001
            for kv in sec.entries:
                if len(kv.key.path) == 1:
                    yield kv.key.path[0]


class _TableCommentsView(_TableKVViewBase[str]):
    """Live mapping from key name to end-of-line comment text.

    Backed by a :class:`_StdTable`; a key is "present" iff its
    ``KeyValueNode`` currently carries a ``trailing_comment``. Setting
    an empty string removes the comment, mirroring ``del``.
    """

    _view_name = "comments"

    @override
    def _read(self, key: str) -> str | None:
        kv = self._find_kv(key)
        if kv is None or kv.trailing_comment is None:
            return None
        return _strip_comment_marker(kv.trailing_comment.text)

    @override
    def _write_absent(self, key: str) -> None:
        kv = self._find_kv(key)
        assert kv is not None  # presence checked by caller
        _set_eol_comment(kv, None)

    @override
    def __setitem__(self, key: str, value: str) -> None:
        _, kv = self._table._find_direct_kv(key)  # noqa: SLF001
        _set_eol_comment(kv, value)


class _TableLeadingCommentsView(_TableKVViewBase["tuple[str, ...]"]):
    """Live mapping from key name to its leading comment block.

    The block is the contiguous run of ``# ...`` lines immediately
    above the entry's source line (with their newlines), as stored in
    the entry's ``leading`` trivia. A key is "present" iff that run is
    non-empty.
    """

    _view_name = "leading_comments"

    @override
    def _read(self, key: str) -> tuple[str, ...] | None:
        kv = self._find_kv(key)
        if kv is None:
            return None
        return _extract_trailing_comment_block(kv.leading) or None

    @override
    def _write_absent(self, key: str) -> None:
        self[key] = ()

    @override
    def __setitem__(self, key: str, value: Sequence[str]) -> None:
        _, kv = self._table._find_direct_kv(key)  # noqa: SLF001
        _replace_trailing_comment_block(
            kv.leading,
            value,
            _indent_after_last_newline(kv.leading),
        )

    @override
    def _format_value(self, value: tuple[str, ...]) -> str:
        return repr(list(value))


class _ArrayItemViewBase(_PresenceFilteredView[int, _VV]):
    """Common scaffolding for :class:`Array`-item-backed views."""

    __slots__ = ("_array",)

    _view_name: ClassVar[str]

    def __init__(self, array: Array) -> None:
        self._array = array

    @override
    def _check_key(self, key: object) -> int:
        if not isinstance(key, int):
            msg = f"Array.{self._view_name} index must be int, got {type(key).__name__}"
            raise TypeError(msg)
        n = len(self._array._node.items)  # noqa: SLF001
        if -n <= key < 0:
            return key + n
        if 0 <= key < n:
            return key
        raise KeyError(key)

    @override
    def _keys(self) -> Iterator[int]:
        return iter(range(len(self._array._node.items)))  # noqa: SLF001


class _ArrayCommentsView(_ArrayItemViewBase[str]):
    """Live mapping from array index to that item's end-of-line comment.

    Backed by an :class:`Array`. An index is "present" iff the
    corresponding item currently carries an EOL comment in its
    ``post_comma_trivia`` (when the item has a trailing comma) or
    its ``trailing`` trivia (last item, no trailing comma).
    """

    _view_name = "comments"

    @override
    def _read(self, key: int) -> str | None:
        item = self._array._node.items[key]  # noqa: SLF001
        if item.has_comma:
            c = _extract_eol_comment(item.post_comma_trivia)
            if c is not None:
                return c
        return _extract_eol_comment(item.trailing)

    @override
    def _write_absent(self, key: int) -> None:
        item = self._array._node.items[key]  # noqa: SLF001
        if _extract_eol_comment(item.post_comma_trivia) is not None:
            _replace_eol_comment(item.post_comma_trivia, None, force_newline=False)
        if _extract_eol_comment(item.trailing) is not None:
            _replace_eol_comment(item.trailing, None, force_newline=False)

    @override
    def __setitem__(self, key: int, value: str) -> None:
        i = self._check_key(key)
        items = self._array._node.items  # noqa: SLF001
        item = items[i]
        is_last = i == len(items) - 1
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
            # indent that matches its siblings -- but only if neither
            # the slot we just wrote nor the next item's leading
            # already supplies one (otherwise we'd double-indent).
            indent = _array_indent(self._array._node)  # noqa: SLF001
            next_item = items[i + 1]
            slot_text = slot.render()
            slot_nl = slot_text.rfind("\n")
            slot_has_indent = slot_nl >= 0 and slot_text[slot_nl + 1 :] != ""
            next_has_break = "\n" in next_item.leading.render()
            if indent and not slot_has_indent and not next_has_break:
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


class _ArrayLeadingCommentsView(_ArrayItemViewBase["tuple[str, ...]"]):
    """Live mapping from array index to its leading comment block.

    For item 0, the block is extracted from ``items[0].leading`` (the
    trivia between ``[`` and the first value). For item i > 0, it is
    extracted from ``items[i-1].post_comma_trivia`` (specifically, the
    contiguous trailing run of comment lines, ignoring any EOL portion
    that belongs to item i-1).
    """

    _view_name = "leading_comments"

    @override
    def _read(self, key: int) -> tuple[str, ...] | None:
        items = self._array._node.items  # noqa: SLF001
        trivia, min_start = _logical_leading_slot(items, key)
        return _extract_trailing_comment_block(trivia, min_start=min_start) or None

    @override
    def _write_absent(self, key: int) -> None:
        self[key] = ()

    @override
    def __setitem__(self, key: int, value: Sequence[str]) -> None:
        i = self._check_key(key)
        items = self._array._node.items  # noqa: SLF001
        indent = _array_indent(self._array._node)  # noqa: SLF001
        trivia, min_start = _logical_leading_slot(items, i)
        # Ensure the slot ends with a newline+indent anchor so the
        # block lands on its own line(s) before the next value.
        if value and not any(isinstance(p, NewlineNode) for p in trivia.pieces):
            while trivia.pieces and isinstance(trivia.pieces[-1], WhitespaceNode):
                trivia.pieces.pop()
            trivia.pieces.append(NewlineNode("\n"))
            if indent:
                trivia.pieces.append(WhitespaceNode(indent))
        _replace_trailing_comment_block(trivia, value, indent, min_start=min_start)

    @override
    def _format_value(self, value: tuple[str, ...]) -> str:
        return repr(list(value))


# ---------------------------------------------------------------------------
# Array (inline) and AoT (array of tables)
# ---------------------------------------------------------------------------


class Array(list[Any]):
    """Inline TOML array exposed as a real :class:`list`.

    Every standard list mutator is overridden so the underlying CST
    stays in sync. Existing handles to nested ``Array``/``Table`` values
    that were *not* removed remain valid; handles to removed/replaced
    elements become detached.
    """

    __slots__ = ("_attached", "_indent", "_node", "_style")

    def __init__(
        self,
        items: Iterable[object] | ArrayNode = (),
        *,
        multiline: bool = False,
        indent: str = "    ",
    ) -> None:
        """Construct a standalone array or wrap an existing CST node.

        Public use: ``Array([1, 2, 3])`` builds an inline array;
        ``Array([1, 2, 3], multiline=True)`` lays it out one item per
        line with ``indent`` indentation. Such an array is *detached*
        until assigned into a document (``doc[k] = arr``).

        Passing an :class:`ArrayNode` directly is the internal
        attached-construction path used by the parser and CST walkers.
        """
        if isinstance(items, ArrayNode):
            self._node = items
            self._attached = True
        else:
            from tomlrt._synthesise import _list_to_array_node  # noqa: PLC0415

            self._node = _list_to_array_node(list(items))  # type: ignore[arg-type]
            self._attached = False
        self._style = _sample_separator_style(
            self._node.items,
            self._node.final_trivia,
        )
        self._indent = indent
        super().__init__(_materialise_array(self._node))
        if not self._attached and multiline:
            self.set_multiline(multiline=True, indent=indent)

    def _detach(self) -> None:
        self._attached = False
        for v in self:
            if isinstance(v, (Table, AoT, Array)):
                v._detach()  # noqa: SLF001

    # ------------------------------------------------------------------
    # CST <-> list synchronisation helpers
    # ------------------------------------------------------------------

    def _resync(self) -> None:
        """Rebuild the public list from the CST after a structural change."""
        list.clear(self)
        list.extend(self, _materialise_array(self._node))

    def __copy__(self) -> Array:
        return self.__deepcopy__({})

    def __deepcopy__(self, memo: dict[int, object]) -> Array:
        new = Array.__new__(Array)
        memo[id(self)] = new
        new._node = deepcopy(self._node, memo)  # noqa: SLF001
        new._style = deepcopy(self._style, memo)  # noqa: SLF001
        new._indent = self._indent  # noqa: SLF001
        new._attached = self._attached  # noqa: SLF001
        list.__init__(new, _materialise_array(new._node))  # noqa: SLF001
        return new

    def _rebuild_separators(self) -> None:
        _apply_separator_style(self._node, self._style)

    def _rebuild_with_leadings(
        self,
        leadings: Sequence[Sequence[str]],
    ) -> None:
        """Apply separator style and restore an explicitly-given leadings list.

        Used by reorder operations that mutate ``items`` and need to keep
        the per-item leading-comment blocks aligned with their (possibly
        moved) items rather than with the on-disk storage slots. The
        ``leadings`` list must be snapshotted **before** the items list
        is reordered, then transformed in parallel.
        """
        _apply_separator_style(self._node, self._style)
        _write_item_leadings(self._node.items, leadings)

    @staticmethod
    def _make_item(value: object, *, with_comma: bool) -> ArrayItem:
        from tomlrt._nodes import ArrayItem  # noqa: PLC0415

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

    @property
    def multiline(self) -> bool:
        """Whether the array currently renders across multiple lines."""
        return "\n" in self._style.inter_separator.render()

    @multiline.setter
    def multiline(self, multiline: bool) -> None:
        self.set_multiline(multiline=multiline)

    def set_multiline(self, *, multiline: bool, indent: str = "    ") -> Array:
        """Switch this array between single-line and multi-line layout.

        ``indent`` controls the per-item indentation when ``multiline``
        is true and is ignored otherwise. Returns ``self`` so calls may
        be chained.

        Switching to single-line layout when any item carries an EOL
        or leading comment is rejected with :class:`TOMLError`: a ``#``
        comment runs to end of line, so collapsing such an array would
        produce invalid TOML. Clear the offending comments first via
        :attr:`comments` / :attr:`leading_comments` if you really want
        a single-line layout.
        """
        if not multiline and _value_has_inner_comment(self._node):
            msg = (
                "cannot collapse a multi-line array to single-line: "
                "items carry EOL or leading comments which would "
                "produce invalid TOML; clear them first via .comments "
                "and .leading_comments"
            )
            raise TOMLError(msg)
        if multiline:
            inter = Trivia([NewlineNode("\n"), WhitespaceNode(indent)])
            self._style = _SeparatorStyle(
                open_pad=_clone_trivia(inter),
                inter_separator=_clone_trivia(inter),
                trailing_comma=True,
                close_pad=Trivia([NewlineNode("\n")]),
            )
            self._indent = indent
        else:
            self._style = _SeparatorStyle(
                open_pad=Trivia(),
                inter_separator=Trivia([WhitespaceNode(" ")]),
                trailing_comma=False,
                close_pad=Trivia(),
            )
        self._rebuild_separators()
        return self

    # ------------------------------------------------------------------
    # Mutators (override every one)
    # ------------------------------------------------------------------

    @override
    def append(self, value: object) -> None:
        new_item = self._make_item(value, with_comma=False)
        self._node.items.append(new_item)
        self._rebuild_separators()
        list.append(self, _value_for(new_item.value))

    @override
    def extend(self, values: Iterable[object]) -> None:
        new_items = [self._make_item(v, with_comma=False) for v in list(values)]
        self._node.items.extend(new_items)
        self._rebuild_separators()
        list.extend(self, [_value_for(it.value) for it in new_items])

    @override
    def insert(self, index: SupportsIndex, value: object) -> None:
        idx = operator.index(index)
        leadings = _snapshot_item_leadings(self._node.items)
        new_item = self._make_item(value, with_comma=False)
        self._node.items.insert(idx, new_item)
        leadings.insert(idx, ())
        self._rebuild_with_leadings(leadings)
        list.insert(self, idx, _value_for(new_item.value))

    @overload
    def __setitem__(self, index: SupportsIndex, value: object) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[object]) -> None: ...
    @override
    def __setitem__(
        self,
        index: SupportsIndex | slice,
        value: object,
    ) -> None:
        if isinstance(index, slice):
            if not isinstance(value, Iterable):
                msg = "must assign iterable to extended slice"
                raise TypeError(msg)
            leadings = _snapshot_item_leadings(self._node.items)
            new_items = [self._make_item(v, with_comma=False) for v in list(value)]
            self._node.items[index] = new_items
            leadings[index] = [() for _ in new_items]
            self._rebuild_with_leadings(leadings)
        else:
            i = operator.index(index)
            self._node.items[i].value = value_to_node(value)
            self._rebuild_separators()
        self._resync()

    @override
    def __delitem__(self, index: SupportsIndex | slice) -> None:
        leadings = _snapshot_item_leadings(self._node.items)
        if isinstance(index, slice):
            del self._node.items[index]
            del leadings[index]
        else:
            i = operator.index(index)
            del self._node.items[i]
            del leadings[i]
        self._rebuild_with_leadings(leadings)
        self._resync()

    @override
    def pop(self, index: SupportsIndex = -1) -> Any:
        leadings = _snapshot_item_leadings(self._node.items)
        i = operator.index(index)
        item = self._node.items.pop(i)
        del leadings[i]
        self._rebuild_with_leadings(leadings)
        self._resync()
        popped = _value_for(item.value)
        # The wrapper was constructed from an item that is no longer
        # in the document; reflect that on the returned object so that
        # later reassignment doesn't trigger the cross-doc clone path
        # (and so that `_attached` honestly describes reality).
        if isinstance(popped, (Table, Array)):
            popped._detach()  # noqa: SLF001
        return popped

    @override
    def remove(self, value: object) -> None:
        idx = list.index(self, value)
        del self[idx]

    @override
    def clear(self) -> None:
        self._node.items.clear()
        self._rebuild_separators()
        self._resync()

    @override
    def reverse(self) -> None:
        n = len(self._node.items)
        self._reorder_items(range(n - 1, -1, -1))

    @override
    def sort(
        self,
        *,
        key: Callable[[Any], object] | None = None,
        reverse: bool = False,
    ) -> None:
        values = _materialise_array(self._node)
        sort_key: Callable[[int], Any] = (
            (lambda i: values[i]) if key is None else (lambda i: key(values[i]))
        )
        self._reorder_items(sorted(range(len(values)), key=sort_key, reverse=reverse))

    def _reorder_items(self, perm: Iterable[int]) -> None:
        """Apply an index permutation to ``items`` and their leadings.

        Each item's leading carries that item's preceding comment/blank
        layout, so it must travel with the item. Inter-item separators
        get rebuilt afterwards for the new order.
        """
        order = list(perm)
        items = self._node.items
        leadings = _snapshot_item_leadings(items)
        items[:] = [items[i] for i in order]
        self._rebuild_with_leadings([leadings[i] for i in order])
        self._resync()

    @override
    def __iadd__(self, values: Iterable[object]) -> Self:
        self.extend(values)
        return self

    @override
    def __imul__(self, count: SupportsIndex) -> Self:
        n = operator.index(count)
        if n <= 0:
            self.clear()
        else:
            base = list(self._node.items)
            base_leadings = _snapshot_item_leadings(base)
            for _ in range(n - 1):
                self._node.items.extend(deepcopy(item) for item in base)
            leadings = list(base_leadings) * n
            self._rebuild_with_leadings(leadings)
            self._resync()
        return self

    def to_list(self) -> list[Any]:
        """Return a deep, plain-Python copy of this array.

        Walks recursively, converting nested :class:`Table` /
        :class:`AoT` / :class:`Array` views into ordinary
        :class:`dict` / :class:`list` containers. Scalars are
        returned as-is.
        """
        return [_to_plain(v) for v in self]

    # ------------------------------------------------------------------
    # Typed accessors for nested values. Mirror Table.array/.table.
    # ------------------------------------------------------------------

    def array(self, index: SupportsIndex) -> Array:
        """Return ``self[index]`` typed as a nested :class:`Array`."""
        value = self[index]
        if not isinstance(value, Array):
            type_name = type(value).__name__
            msg = f"item {operator.index(index)} is a {type_name}, not an Array"
            raise TypeError(msg)
        return value

    @overload
    def get_array(self, index: SupportsIndex) -> Array | None: ...
    @overload
    def get_array(self, index: SupportsIndex, default: _T) -> Array | _T: ...
    def get_array(self, index: SupportsIndex, default: object = None) -> object:
        """Like :meth:`array`, but returns ``default`` if ``index`` is out of range.

        Wrong-type entries still raise :class:`TypeError`.
        """
        try:
            value = self[index]
        except IndexError:
            return default
        if not isinstance(value, Array):
            type_name = type(value).__name__
            msg = f"item {operator.index(index)} is a {type_name}, not an Array"
            raise TypeError(msg)
        return value

    def table(self, index: SupportsIndex) -> Table:
        """Return ``self[index]`` typed as a nested :class:`Table`."""
        value = self[index]
        if not isinstance(value, Table):
            msg = (
                f"item {operator.index(index)} is a {type(value).__name__}, not a Table"
            )
            raise TypeError(msg)
        return value

    @overload
    def get_table(self, index: SupportsIndex) -> Table | None: ...
    @overload
    def get_table(self, index: SupportsIndex, default: _T) -> Table | _T: ...
    def get_table(self, index: SupportsIndex, default: object = None) -> object:
        """Like :meth:`table`, but returns ``default`` if ``index`` is out of range.

        Wrong-type entries still raise :class:`TypeError`.
        """
        try:
            value = self[index]
        except IndexError:
            return default
        if not isinstance(value, Table):
            msg = (
                f"item {operator.index(index)} is a {type(value).__name__}, not a Table"
            )
            raise TypeError(msg)
        return value


class AoT(list[Table]):
    """Array-of-tables, e.g. ``[[products]]`` repeated.

    Subclass of :class:`list`; supports basic mutation (append/insert
    of dict-shaped or :class:`Table` entries) by synthesizing fresh
    ``[[path]]`` sections in the underlying CST.
    """

    __slots__ = ("_attached", "_doc_node", "_path")

    def __init__(
        self,
        entries: Iterable[Mapping[str, object]] = (),
    ) -> None:
        """Construct a standalone array-of-tables.

        Each of ``entries`` (a dict-shaped mapping or :class:`Table`)
        is materialised into a ``[[_]]`` section in an internal orphan
        document, so all the usual list mutators (``append``, ``insert``,
        ``extend``, ``pop``, ``__setitem__`` of slots) keep working
        pre-assignment. On ``doc[k] = aot``, the pending sections are
        rewritten to ``[[k]]`` and merged into the target document.

        The 3-argument internal form (``doc_node``, ``path``,
        ``tables``) used by the parser/CST walkers remains available
        via :meth:`_attached_to`.
        """
        path: tuple[str, ...] = ("_",)
        doc_node = DocumentNode(sections=[])
        super().__init__()
        self._doc_node: DocumentNode = doc_node
        self._path = path
        self._attached = False
        for entry in entries:
            self._insert_at(len(self), entry)

    @classmethod
    def _attached_to(
        cls,
        doc_node: DocumentNode,
        path: tuple[str, ...],
        tables: list[Table],
    ) -> AoT:
        obj = cls.__new__(cls)
        list.__init__(obj, tables)
        obj._doc_node = doc_node  # noqa: SLF001
        obj._path = path  # noqa: SLF001
        obj._attached = True  # noqa: SLF001
        return obj

    def _detach(self, doc_node: DocumentNode | None = None) -> None:
        if not self._attached:
            return
        if doc_node is None:
            captured: list[SectionNode] = []
            seen: set[int] = set()
            for s in self._own_sections():
                for sec in self._doc_node.aot_entry_block(s):
                    if id(sec) not in seen:
                        captured.append(sec)
                        seen.add(id(sec))
            doc_node = DocumentNode(sections=captured)
        self._attached = False
        self._doc_node = doc_node
        for v in self:
            v._detach(doc_node)  # noqa: SLF001

    # ------------------------------------------------------------------
    # CST <-> list synchronisation
    # ------------------------------------------------------------------

    def _own_sections(self) -> list[SectionNode]:
        """Sections that act as the [[path]] entry headers (in doc order)."""
        return [
            s
            for s in self._doc_node.sections
            if s.header is not None
            and s.header.kind == "array"
            and s.header.key.path == self._path
        ]

    def _own_blocks(self) -> tuple[int, list[list[SectionNode]]]:
        """Each entry's [header, *owned-subsections] block, plus splice index.

        Reordering operations (reverse / sort) require entries to occupy a
        contiguous run of ``_doc_node.sections``. If unrelated sections sit
        between two entries this raises, since permuting blocks would change
        the meaning of those interleaved sections.
        """
        own = self._own_sections()
        if not own:
            return 0, []
        sections = self._doc_node.sections
        blocks: list[list[SectionNode]] = [
            self._doc_node.aot_entry_block(header) for header in own
        ]
        start = sections.index(blocks[0][0])
        cursor = start
        for block in blocks:
            if sections[cursor] is not block[0]:
                msg = (
                    "cannot reorder AoT entries: unrelated sections are "
                    "interleaved between entries"
                )
                raise RuntimeError(msg)
            cursor += len(block)
        return start, blocks

    def _resync(self) -> None:
        # Preserve identity for entries whose anchor section is unchanged.
        existing: dict[int, Table] = {}
        for entry in self:
            if isinstance(entry, _StdTable) and entry._anchor is not None:  # noqa: SLF001
                existing[id(entry._anchor)] = entry  # noqa: SLF001
        own = self._own_sections()
        new_entries: list[Table] = []
        kept: set[int] = set()
        for s in own:
            cached = existing.get(id(s))
            if cached is not None:
                kept.add(id(cached))
                new_entries.append(cached)
            else:
                new_entries.append(
                    _StdTable(
                        self._doc_node,
                        self._path,
                        anchor=s,
                    ),
                )
        # Detach any previous entries that are no longer in the AoT.
        for entry in self:
            if id(entry) not in kept:
                entry._detach()  # noqa: SLF001
        list.clear(self)
        list.extend(self, new_entries)

    def __copy__(self) -> AoT:
        return self.__deepcopy__({})

    def __deepcopy__(self, memo: dict[int, object]) -> AoT:
        new = AoT.__new__(AoT)
        memo[id(self)] = new
        new._doc_node = deepcopy(self._doc_node, memo)  # noqa: SLF001
        new._path = self._path  # noqa: SLF001
        new._attached = self._attached  # noqa: SLF001
        list.__init__(new)
        new._resync()  # noqa: SLF001
        return new

    def _make_header_section(self) -> SectionNode:
        return _new_section(self._path, kind="array")

    def _populate_section(self, sec: SectionNode, value: object) -> None:
        """Fill ``sec`` with KV entries derived from ``value``.

        Accepts a plain dict, a :class:`Table`, or any
        :class:`collections.abc.Mapping`. Cross-document Tables are
        deep-cloned by ``make_keyvalue_node`` (via ``value_to_node``)
        so inline tables / arrays aren't aliased.
        """
        if not isinstance(value, Mapping):
            msg = (
                f"cannot append a value of type {type(value).__name__} to an "
                "array-of-tables; expected a dict or Table"
            )
            raise TOMLError(msg)
        for k, v in value.items():
            if not isinstance(k, str):
                msg = f"AoT entry keys must be strings, got {type(k).__name__}"
                raise TOMLError(msg)
            sec.entries.append(make_keyvalue_node(k, v))

    def _build_entry_block(
        self,
        value: Table | Mapping[str, object],
    ) -> list[SectionNode]:
        """Build the section block for a new AoT entry.

        For a :class:`_StdTable` source, deep-clone its contributing
        sections via :func:`_clone_table_sections` so per-KV trivia,
        sub-section headers, and any nested AoTs are preserved
        verbatim under the new entry's path. For plain mappings, fall
        back to data-only synthesis through :meth:`_populate_section`.
        """
        if isinstance(value, _StdTable):
            cloned = _clone_table_sections(value, self._path, head_kind="array")
            if cloned:
                return cloned
        new_sec = self._make_header_section()
        self._populate_section(new_sec, value)
        return [new_sec]

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    @override
    def append(self, value: Table | Mapping[str, object]) -> None:
        self._insert_at(len(self), value)

    def add(self, entry: Mapping[str, object] = MappingProxyType({})) -> Table:
        """Append ``entry`` and return the new :class:`Table` view.

        Convenience over :meth:`append` for the common build-and-mutate
        idiom: ``pkg = aot.add({"name": "foo"}); pkg.set_table(...)``.
        ``entry`` defaults to an empty mapping, so ``aot.add()`` adds a
        blank entry and returns it for further population.
        """
        self._insert_at(len(self), entry)
        return self[-1]

    def to_list(self) -> list[dict[str, Any]]:
        """Return a deep, plain-Python copy of this array-of-tables.

        Each entry is converted to an ordinary :class:`dict` (with
        nested views recursively flattened to plain containers). The
        result shares no mutable state with the document.
        """
        return [t.to_dict() for t in self]

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
        new_block = self._build_entry_block(value)
        new_sec = new_block[0]
        sections = self._doc_node.sections
        # Pick an insertion point first; blank-line decision depends on it.
        if py_index == n:
            # Append: land after the last [[path]] entry's owned range,
            # or at end of doc if no entries exist yet.
            if own:
                tail = self._doc_node.aot_entry_block(own[-1])[-1]
                insert_idx = sections.index(tail) + 1
            else:
                insert_idx = len(sections)
        else:
            insert_idx = sections.index(own[py_index])
        # Insert a blank-line separator before the new header iff there
        # is already rendered content preceding it. When existing
        # siblings already share a uniform spacing style, copy that;
        # otherwise default to blank-separated (canonical TOML style).
        sibling_leadings = [
            sec.header.leading for sec in own[1:] if sec.header is not None
        ]
        add_blank = (
            _gaps_uniformly_blank(sibling_leadings) if sibling_leadings else True
        )
        preceding_has_content = any(
            s.header is not None or s.entries for s in sections[:insert_idx]
        )
        assert new_sec.header is not None
        if preceding_has_content and add_blank:
            _prepend_blank_line(new_sec.header.leading)
        # Symmetric: when inserting before existing content, ensure the
        # next section's header carries a blank-line separator from the
        # new one so two ``[[..]]`` headers don't render glued together.
        if py_index < n and add_blank:
            next_hdr = sections[insert_idx].header
            if next_hdr is not None:
                _prepend_blank_line(next_hdr.leading)
        self._doc_node.adopt_preamble_into(new_sec.header.leading)
        sections[insert_idx:insert_idx] = new_block
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
        block = self._doc_node.aot_entry_block(target)
        # Use the live entry as the popped object to preserve identity.
        popped = self[i]
        # Orphan the popped view onto its own ``DocumentNode`` *before*
        # removing the block from the live doc: ``_resync``'s default
        # detach path runs after ``remove_sections``, by which point
        # ``aot_owned_range`` searches an empty list and silently
        # drops the popped entry's ``[a.sub]``-style sub-sections.
        popped._detach(DocumentNode(sections=list(block)))  # noqa: SLF001
        self._doc_node.remove_sections(set(block))
        self._resync()
        return popped

    @override
    def clear(self) -> None:
        own = self._own_sections()
        to_remove: set[SectionNode] = set()
        # Pre-detach every entry into its own orphan doc, for the same
        # reason as ``pop``: once ``remove_sections`` runs, ``_resync``'s
        # detach pass cannot rebuild owned-range capture and would lose
        # any nested sub-sections from the cached entry views.
        for entry, sec in zip(list(self), own, strict=True):
            block = self._doc_node.aot_entry_block(sec)
            entry._detach(DocumentNode(sections=list(block)))  # noqa: SLF001
            to_remove.update(block)
        self._doc_node.remove_sections(to_remove)
        self._resync()

    @override
    def __delitem__(self, index: SupportsIndex | slice) -> None:
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            for i in sorted(indices, reverse=True):
                self.pop(i)
        else:
            self.pop(index)

    @staticmethod
    def _validate_entry(value: object) -> None:
        if not isinstance(value, Mapping):
            msg = "AoT entry must be a mapping"
            raise TypeError(msg)

    @overload
    def __setitem__(
        self, index: SupportsIndex, value: Mapping[str, object]
    ) -> None: ...
    @overload
    def __setitem__(
        self,
        index: slice,
        value: Iterable[Mapping[str, object]],
    ) -> None: ...
    @override
    def __setitem__(
        self,
        index: SupportsIndex | slice,
        value: Mapping[str, object] | Iterable[Mapping[str, object]],
    ) -> None:
        if isinstance(index, slice):
            new_values: list[Any] = list(value)
            for v in new_values:
                self._validate_entry(v)
            indices = range(*index.indices(len(self)))
            if index.step not in (None, 1):
                if len(new_values) != len(indices):
                    msg = (
                        f"attempt to assign sequence of size {len(new_values)} "
                        f"to extended slice of size {len(indices)}"
                    )
                    raise ValueError(msg)
                for i, v in zip(indices, new_values, strict=True):
                    self[i] = v
                return
            del self[index]
            for offset, v in enumerate(new_values):
                self.insert(indices.start + offset, v)
            return
        self._validate_entry(value)
        assert isinstance(value, Mapping)
        # Replacement is always del + insert: the splice path used by
        # ``insert`` already handles cloned-Table sources (preserving
        # per-KV trivia, sub-sections, and nested AoTs) as well as
        # plain mappings. Snapshot the existing slot's header leading
        # so the visual context (comments / blank lines above the
        # ``[[path]]`` header) survives the swap regardless of source.
        i = operator.index(index)
        n = len(self)
        if i < 0:
            i += n
        if i < 0 or i >= n:
            msg = f"AoT assignment index out of range: {index}"
            raise IndexError(msg)
        own = self._own_sections()
        hdr = own[i].header
        prior_leading = hdr.leading if hdr is not None else None
        del self[i]
        self.insert(i, value)
        _apply_prior_leading([self._own_sections()[i]], prior_leading)

    @override
    def __iadd__(self, values: Iterable[Mapping[str, object]]) -> Self:  # type: ignore[override]
        self.extend(values)
        return self

    @override
    def remove(self, value: Mapping[str, object]) -> None:
        for i, entry in enumerate(self):
            if entry == value:
                del self[i]
                return
        msg = f"{value!r} not in list"
        raise ValueError(msg)

    @override
    def __imul__(self, count: SupportsIndex) -> Self:
        n = operator.index(count)
        if n <= 0:
            self.clear()
            return self
        if n == 1:
            return self
        start, blocks = self._own_blocks()
        base: list[SectionNode] = [s for block in blocks for s in block]
        # The duplicated first block needs (a) the inter-repetition
        # separator -- so doubling a blank-line-separated doc stays
        # visually consistent -- and (b) its own deep-copied leading
        # comment (the comment logically describes the entry, not the
        # slot, so it must travel with the duplicated block).
        # Sample (a) from blocks[1].leading with the comment stripped;
        # fall back to a blank line when there's no second block to
        # sample, to avoid gluing copies header-to-header.
        if len(blocks) >= 2:
            inter_separator = Trivia(
                pieces=list(self._block_leading(blocks[1]).pieces),
            )
            _replace_trailing_comment_block(inter_separator, (), "")
        else:
            inter_separator = Trivia(pieces=[NewlineNode("\n")])
        repeated = list(base)
        for _ in range(n - 1):
            copy_blocks: list[list[SectionNode]] = [
                [deepcopy(s) for s in block] for block in blocks
            ]
            first_leading = self._block_leading(copy_blocks[0])
            first_leading.pieces = [
                *deepcopy(inter_separator).pieces,
                *first_leading.pieces,
            ]
            repeated.extend(s for block in copy_blocks for s in block)
        self._doc_node.sections[start : start + len(base)] = repeated
        self._resync()
        return self

    @override
    def reverse(self) -> None:
        start, blocks = self._own_blocks()
        if not blocks:
            return
        self._reorder_blocks(start, blocks, range(len(blocks) - 1, -1, -1))

    @override
    def sort(
        self,
        *,
        key: Callable[[Table], object] | None = None,
        reverse: bool = False,
    ) -> None:
        start, blocks = self._own_blocks()
        if not blocks:
            return
        entries = list(self)
        sort_key: Callable[[int], Any] = (
            (lambda i: entries[i]) if key is None else (lambda i: key(entries[i]))
        )
        self._reorder_blocks(
            start,
            blocks,
            sorted(range(len(blocks)), key=sort_key, reverse=reverse),
        )

    def _reorder_blocks(
        self,
        start: int,
        blocks: list[list[SectionNode]],
        perm: Iterable[int],
    ) -> None:
        """Apply an index permutation to this AoT's section blocks.

        A block's trailing-comment chunk belongs to its entry and
        travels with the block; the inter-entry separator pattern
        belongs to the slot and stays in place.
        """
        order = list(perm)
        end = start + sum(len(b) for b in blocks)
        leadings = [self._block_leading(b) for b in blocks]
        entry_comments = [_extract_trailing_comment_block(L) for L in leadings]
        new_blocks = [blocks[i] for i in order]
        new_comments = [entry_comments[i] for i in order]
        for block, leading in zip(new_blocks, leadings, strict=True):
            self._set_block_leading(block, leading)
        for block, comment in zip(new_blocks, new_comments, strict=True):
            _replace_trailing_comment_block(self._block_leading(block), comment, "")
        self._doc_node.sections[start:end] = [s for block in new_blocks for s in block]
        self._resync()

    @staticmethod
    def _block_leading(block: list[SectionNode]) -> Trivia:
        header = block[0].header
        assert header is not None
        return header.leading

    @staticmethod
    def _set_block_leading(block: list[SectionNode], leading: Trivia) -> None:
        header = block[0].header
        assert header is not None
        header.leading = leading


# ---------------------------------------------------------------------------
# View / aggregator
# ---------------------------------------------------------------------------


def _iter_table(
    doc_node: DocumentNode,
    path: tuple[str, ...],
    *,
    pool: list[SectionNode],
    anchor: SectionNode | None = None,
    owner_anchor: SectionNode | None = None,
    extras: list[tuple[tuple[str, ...], KeyValueNode]] | None = None,
) -> Iterator[tuple[str, TomlValue]]:
    # ``pool`` is the section list this table draws from: the whole
    # document at the root, the AoT-owned range for an AoT-narrowed
    # table, or — during construction — the per-head bucket the parent
    # already partitioned for us.

    # Sections whose entries are "direct" key/values at this exact path.
    direct_secs: list[SectionNode]
    if anchor is not None:
        direct_secs = [anchor]
    elif path == ():
        direct_secs = [s for s in pool if s.header is None]
    else:
        direct_secs = [
            s
            for s in pool
            if s.header is not None
            and s.header.kind == "table"
            and s.header.key.path == path
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
    for sec in pool:
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
    if extras:
        for rel_path, entry in extras:
            head = rel_path[0]
            extras_by_head.setdefault(head, []).append((rel_path, entry))
            _add(head)

    for head in name_order:
        direct_kvs = direct_kvs_by_head.get(head, [])
        head_extras = extras_by_head.get(head, [])
        aot_secs = aot_by_head.get(head, [])
        sub_secs = sub_by_head.get(head, [])

        if aot_secs:
            tables: list[Table] = [
                _StdTable(doc_node, (*path, head), anchor=s) for s in aot_secs
            ]
            yield head, AoT._attached_to(doc_node, (*path, head), tables)  # noqa: SLF001
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
        for rel_path, kv in head_extras:
            if len(rel_path) == 1:
                if terminal is None:
                    terminal = kv
            else:
                nested_extras.append((rel_path, kv))

        if (
            terminal is not None
            and not nested_kvs
            and not nested_extras
            and not sub_secs
        ):
            yield head, _value_for(terminal.value)
            continue

        if not sub_secs and not nested_extras:
            # Pure dotted from this section level. Prefix is relative
            # to the host section (where dotted KVs live), not the
            # absolute logical path.
            yield (
                head,
                _DottedSubTable(
                    depth=1,
                    host=_SectionDottedHost(direct_secs),
                    prefix=(head,),
                ),
            )
            continue

        # Merged view at path + (head,). For non-AoT children we hand
        # over the per-head bucket as their pool and the dotted-key
        # extras we already collected (one head segment stripped) so
        # they skip a full rescan and ancestor walk. AoT-narrowed
        # children fall through to the slow path: their pool depends
        # on the AoT entry's owned range, computed lazily via
        # ``_scope``.
        child_owner = anchor or owner_anchor
        if child_owner is None:
            child_extras = [(kv.key.path[1:], kv) for kv in nested_kvs]
            child_extras.extend((rp[1:], kv) for rp, kv in nested_extras)
            yield (
                head,
                _StdTable(
                    doc_node,
                    (*path, head),
                    _pool=sub_secs,
                    _extras=child_extras,
                ),
            )
        else:
            yield (
                head,
                _StdTable(doc_node, (*path, head), owner_anchor=child_owner),
            )


__all__ = [
    "AoT",
    "Array",
    "Document",
    "Scalar",
    "Table",
    "TomlValue",
]
