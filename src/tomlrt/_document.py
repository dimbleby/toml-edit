"""Logical Document/Table/Array/AoT view over a CST.

This module exposes the public mapping/sequence types that users
interact with. It implements both the read path and the structural
mutation API on top of the physical CST defined in :mod:`tomlrt._nodes`.
"""

from __future__ import annotations

import operator
import sys
from collections.abc import Iterable, Mapping, MutableMapping
from copy import deepcopy
from datetime import date, datetime, time
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
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


def _prepend_blank_line(trivia: Trivia) -> None:
    """Prepend a blank-line ``NewlineNode`` to ``trivia`` if missing.

    Idempotent: a no-op when ``trivia`` already starts with a newline.
    """
    if not _starts_with_blank_line(trivia):
        trivia.pieces.insert(0, NewlineNode("\n"))


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
    one). Raises :class:`TOMLError` if ``text`` contains any line
    terminator, since comments are single-line by definition.
    """
    if "\n" in text or "\r" in text:
        msg = "comment text must not contain a line terminator"
        raise TOMLError(msg)
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
    pieces = t.pieces
    if not pieces:
        return True
    # ``CommentNode`` is the only non-whitespace TriviaPiece; checking
    # for its absence dodges per-piece tuple-isinstance.
    return not any(isinstance(p, CommentNode) for p in pieces)


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
            # Multiline-intent empty container: infer indent from the
            # whitespace following the (last) newline in the close pad.
            indent_text = pad_text.rsplit("\n", 1)[1] or "    "
            inter = Trivia([NewlineNode("\n"), WhitespaceNode(indent_text)])
            return _SeparatorStyle(
                open_pad=_clone_trivia(inter),
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
        sep = Trivia([WhitespaceNode(" ")])
    last = items[-1]
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
        if _is_pure_whitespace(it.leading):
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
            if _is_pure_whitespace(item.trailing):
                item.trailing = Trivia()
        else:
            if item.has_comma and _is_pure_whitespace(item.post_comma_trivia):
                item.has_comma = False
                item.post_comma_trivia = Trivia()
            if (
                _is_pure_whitespace(item.trailing)
                and item.trailing.render() != close_render
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


def _insert_section_block(
    doc_node: DocumentNode,
    insert_at: int,
    new_secs: Sequence[SectionNode],
) -> None:
    """Splice a freshly-built block of ``[ ... ]`` / ``[[ ... ]]`` sections.

    Entries are blank-separated from each other, and a blank line is
    inserted before the block whenever ``sections[:insert_at]`` already
    holds rendered content.
    """
    sections = doc_node.sections
    preceding_has_content = any(
        s.header is not None or s.entries for s in sections[:insert_at]
    )
    for i, ns in enumerate(new_secs):
        if i > 0 or preceding_has_content:
            assert ns.header is not None
            ns.header.leading.pieces.insert(0, NewlineNode("\n"))
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
        if any(not p for p in path):
            msg = f"key path {path!r} contains an empty segment"
            raise TOMLError(msg)
        return path
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

    @override
    def __setitem__(self, key: str, value: object) -> None:
        if not self._attached:
            dict.__setitem__(self, key, value)
            return
        # Flavour-bearing values drive structural installation rather
        # than a raw value write. Attached Arrays/AoTs from another
        # document fall through to the deepcopy path so their full CST
        # (comments, formatting) survives the copy; only SectionSpec,
        # standalone AoTs, and standalone Arrays need structural work.
        if isinstance(value, SectionSpec) or (
            isinstance(value, (AoT, Array)) and not value._attached  # noqa: SLF001
        ):
            self._install_flavoured((key,), value)
            return
        # If we're replacing an existing container, detach the old one
        # so any held references stop reflecting later edits.
        if super().__contains__(key):
            old = super().__getitem__(key)
            if isinstance(old, (Table, AoT, Array)):
                old._detach()  # noqa: SLF001
        new_v = self._set_value(key, value)
        if new_v is None:
            self._refresh_key(key)
        else:
            dict.__setitem__(self, key, new_v)

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
        self._install_flavoured(parts, value)
        cur: Any = self
        for part in parts:
            cur = cur[part]
        return cur

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
        if len(parts) > 1:
            path = ".".join(parts)
            msg = (
                f"cannot install at multi-segment path {path!r} inside "
                "an inline-style table"
            )
            raise TOMLError(msg)
        if isinstance(value, Array) and not value._attached:  # noqa: SLF001
            self[parts[0]] = list(value)
            return
        self[parts[0]] = value

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
        value = self._lookup_path(key)
        if not isinstance(value, Table):
            msg = f"{key!r} is a {type(value).__name__}, not a Table"
            raise TypeError(msg)
        return value

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
        try:
            value = self._lookup_path(key)
        except KeyError:
            return default
        if not isinstance(value, Table):
            msg = f"{key!r} is a {type(value).__name__}, not a Table"
            raise TypeError(msg)
        return value

    def array(self, key: str) -> Array:
        """Return the array at ``key``, typed as :class:`Array`.

        ``key`` accepts a dotted path. Raises :class:`KeyError` if any
        segment is missing, or :class:`TypeError` if the destination is
        not an inline array.
        """
        value = self._lookup_path(key)
        if not isinstance(value, Array):
            msg = f"{key!r} is a {type(value).__name__}, not an Array"
            raise TypeError(msg)
        return value

    @overload
    def get_array(self, key: str) -> Array | None: ...
    @overload
    def get_array(self, key: str, default: _T) -> Array | _T: ...
    def get_array(self, key: str, default: object = None) -> object:
        """Like :meth:`array`, but returns ``default`` if ``key`` is missing.

        Wrong-type entries still raise :class:`TypeError`.
        """
        try:
            value = self._lookup_path(key)
        except KeyError:
            return default
        if not isinstance(value, Array):
            msg = f"{key!r} is a {type(value).__name__}, not an Array"
            raise TypeError(msg)
        return value

    def aot(self, key: str) -> AoT:
        """Return the array-of-tables at ``key``, typed as :class:`AoT`.

        ``key`` accepts a dotted path. Raises :class:`KeyError` if any
        segment is missing, or :class:`TypeError` if the destination is
        not an array of tables.
        """
        value = self._lookup_path(key)
        if not isinstance(value, AoT):
            msg = f"{key!r} is a {type(value).__name__}, not an AoT"
            raise TypeError(msg)
        return value

    @overload
    def get_aot(self, key: str) -> AoT | None: ...
    @overload
    def get_aot(self, key: str, default: _T) -> AoT | _T: ...
    def get_aot(self, key: str, default: object = None) -> object:
        """Like :meth:`aot`, but returns ``default`` if ``key`` is missing.

        Wrong-type entries still raise :class:`TypeError`.
        """
        try:
            value = self._lookup_path(key)
        except KeyError:
            return default
        if not isinstance(value, AoT):
            msg = f"{key!r} is a {type(value).__name__}, not an AoT"
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
        if not self.del_prefix((key,)):
            raise KeyError(key)


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
        return [owner, *self._doc_node.aot_owned_range(owner)]

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
                [self._anchor, *self._doc_node.aot_owned_range(self._anchor)]
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
            # child sections may live anywhere in the doc, so we keep the
            # original two-pass shape.
            for kv in self._anchor.entries:
                if kv.key.path[0] == key:
                    if len(kv.key.path) == 1:
                        return ("direct", kv)
                    return ("dotted", None)
            child = (*self._path, key)
            for sec in self._doc_node.sections:
                hdr = sec.header
                if hdr is None:
                    continue
                hpath = hdr.key.path
                if hdr.kind == "array" and hpath == child:
                    return ("aot", None)
                if (
                    hdr.kind == "table"
                    and len(hpath) >= len(child)
                    and hpath[: len(child)] == child
                ):
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
                if hdr.kind == "table":
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
        """
        for sec in self._direct_sections():
            sec.entries[:] = [kv for kv in sec.entries if kv.key.path[0] != key]
        prefix = (*self._path, key)
        plen = len(prefix)
        doc_sections = self._doc_node.sections
        doc_sections[:] = [
            sec
            for sec in doc_sections
            if not (
                sec.header is not None
                and len(sec.header.key.path) >= plen
                and sec.header.key.path[:plen] == prefix
            )
        ]
        # Drop any ancestor-section dotted KV that contributes to our
        # path + key (e.g. ``[tool] poetry.name = "x"`` when purging
        # ``name`` from the ``tool.poetry`` view).
        ppath = self._path
        ppath_len = len(ppath)
        for sec in doc_sections:
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
        # Special case: assigning an AoT (or list of dicts targeted as AoT)
        # is a *structural* edit, not a value assignment.
        if isinstance(value, AoT):
            self._set_aot_value(key, value)
            return None

        kind, payload = self._classify(key)
        if kind in ("direct", "extras"):
            # In-place value swap: reuse the existing KV node.
            assert isinstance(payload, KeyValueNode)
            payload.value = value_to_node(value)
            return _value_for(payload.value)
        if kind in ("dotted", "table", "aot", "extras-prefix"):
            self._purge_conflicting(key)
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
            try:
                idx = doc_node.sections.index(target)
            except ValueError:  # pragma: no cover - defensive
                idx = -1
            if idx >= 0 and idx + 1 < len(doc_node.sections):
                next_header = doc_node.sections[idx + 1].header
                if (
                    next_header is not None
                    and not next_header.leading.render().startswith("\n")
                ):
                    next_header.leading.pieces.insert(0, NewlineNode("\n"))
        # The new dict-storage value is exactly what we just wrote;
        # caller can skip the full _refresh_key walk. Safe for every
        # kind because _purge_conflicting only removes things keyed by
        # ``key`` in this scope, so no other dict slot is invalidated.
        return _value_for(new_kv.value)

    def _set_aot_value(self, key: str, value: AoT) -> None:
        """Assign a (possibly cross-document) AoT to ``key``.

        Each source entry's ``[[..]]`` section is deep-cloned and its
        header path rewritten to ``(*self._path, key)``. New sections
        are appended to the document.
        """
        kind, _ = self._classify(key)
        if kind != "absent":
            self._purge_conflicting(key)
        new_path = (*self._path, key)
        new_parts = [make_key_part(p) for p in new_path]
        new_seps = ["."] * (len(new_parts) - 1)
        # Source sections to clone, in source-document order:
        src_own = value._own_sections()  # noqa: SLF001
        doc_node = self._doc_node
        for src_sec in src_own:
            cloned = deepcopy(src_sec)
            assert cloned.header is not None
            cloned.header.key = Key(parts=list(new_parts), separators=list(new_seps))
            doc_node.adopt_preamble_into(cloned.header.leading)
            doc_node.sections.append(cloned)

    @override
    def _delete_value(self, key: str) -> None:
        kind, payload = self._classify(key)
        if kind == "absent":
            raise KeyError(key)
        if kind in ("direct", "extras") and isinstance(payload, KeyValueNode):
            # Targeted removal: drop just the matching KV from its section.
            # Avoids the O(N) section walks and full-list rebuilds in
            # ``_purge_conflicting`` when there's nothing else to remove.
            target = payload
            scope = self._scope()
            sections = scope if scope is not None else self._doc_node.sections
            for sec in sections:
                entries = sec.entries
                for idx, kv in enumerate(entries):
                    if kv is target:
                        del entries[idx]
                        return
            return  # pragma: no cover - defensive: kv must be reachable
        self._purge_conflicting(key)

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
            next_header = doc_node.sections[0].header
            if not next_header.leading.render().startswith("\n"):
                next_header.leading.pieces.insert(0, NewlineNode("\n"))
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
        if doc_node.sections:
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
        sec, kv = self._find_direct_kv(key)
        if not isinstance(kv.value, InlineTableNode):
            msg = f"{key!r} is not an inline table; nothing to promote"
            raise TOMLError(msg)
        inline = kv.value
        child_path = (*self._path, key)
        # Refuse if a [child_path] section already exists in the document
        # (defensive: the parser blocks any source where this would arise,
        # and assignment auto-purges any conflicting sections, so this
        # branch only fires under direct CST manipulation).
        for existing in self._doc_node.sections:
            hdr = existing.header
            if hdr is not None and hdr.key.path == child_path:  # pragma: no cover
                joined = ".".join(child_path)
                msg = f"cannot promote {key!r}: a [{joined}] section already exists"
                raise TOMLError(msg)
        new_sec = _build_promoted_section(child_path, inline, kv)
        # Remove the inline KV from its host section.
        sec.entries.remove(kv)
        # Insert the promoted section after the parent's last direct
        # section (or at end of document if the parent has none).
        sections = self._doc_node.sections
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
        view = _StdTable(self._doc_node, child_path)
        dict.__setitem__(self, key, view)
        return view

    @override
    def promote_array(self, key: str) -> AoT:
        sec, kv = self._find_direct_kv(key)
        if not isinstance(kv.value, ArrayNode):
            msg = f"{key!r} is not an array; nothing to promote"
            raise TOMLError(msg)
        items = kv.value.items
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
        child_path = (*self._path, key)
        # Defensive: parser/assignment paths should never let this fire.
        for existing in self._doc_node.sections:
            hdr = existing.header
            if hdr is not None and hdr.key.path == child_path:  # pragma: no cover
                joined = ".".join(child_path)
                msg = (
                    f"cannot promote {key!r}: a [[{joined}]] (or [{joined}]) "
                    "section already exists"
                )
                raise TOMLError(msg)
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
        sec.entries.remove(kv)
        sections = self._doc_node.sections
        parent_secs = self._direct_sections()
        if parent_secs:
            anchor = parent_secs[-1]
            insert_at = next(i for i, s in enumerate(sections) if s is anchor) + 1
        else:
            insert_at = len(sections)
        _insert_section_block(self._doc_node, insert_at, new_secs)
        aot = AoT._attached_to(self._doc_node, child_path, [])  # noqa: SLF001
        aot._resync()  # noqa: SLF001
        dict.__setitem__(self, key, aot)
        return aot

    @override
    def _install_flavoured(self, parts: tuple[str, ...], value: object) -> None:
        if isinstance(value, SectionSpec):
            self._install_section(parts, value)
            return
        if isinstance(value, AoT):
            # Snapshot entries as plain dicts so the target is fully
            # independent of the source (which might be attached to a
            # different document).
            entries = [t.to_dict() for t in value]
            self._install_aot(parts, entries)
            return
        if isinstance(value, Array) and not value._attached:  # noqa: SLF001
            # Standalone Arrays are specs: re-synthesise at the target
            # with the requested layout. Attached Arrays take the
            # deepcopy path below so comments/formatting survive.
            multiline = value.multiline
            indent = value._indent  # noqa: SLF001
            items = list(value)
            self._install_array(
                parts,
                items,
                multiline=multiline,
                indent=indent,
            )
            return
        # Non-flavoured value: descend (creating implicit parents as
        # needed) and assign at the leaf with normal __setitem__
        # semantics.
        target: Table = self if len(parts) == 1 else self.ensure_table(parts[:-1])
        target[parts[-1]] = value

    def _prepare_section_slot(
        self,
        parts: tuple[str, ...],
    ) -> tuple[tuple[str, ...], int]:
        """Purge any conflicting value at ``parts`` and pick an insert index.

        Returns ``(full_path, insert_at)`` where ``full_path`` is the
        absolute CST path (``self._path + parts``) and ``insert_at`` is
        the position in ``self._doc_node.sections`` where a new block
        for ``full_path`` should be spliced in.
        """
        full_path = (*self._path, *parts)
        if len(parts) == 1:
            kind, _ = self._classify(parts[0])
            if kind != "absent":
                self._purge_conflicting(parts[0])
        else:
            self._doc_node.purge_path(full_path)
        return full_path, _section_insert_index(self._doc_node.sections, full_path)

    def _install_aot(
        self,
        parts: tuple[str, ...],
        entries: Iterable[Mapping[str, object]],
    ) -> AoT:
        full_path, insert_at = self._prepare_section_slot(parts)
        aot = AoT._attached_to(self._doc_node, full_path, [])  # noqa: SLF001
        new_secs: list[SectionNode] = []
        for entry in entries:
            sec = aot._make_header_section()  # noqa: SLF001
            aot._populate_section(sec, entry)  # noqa: SLF001
            new_secs.append(sec)
        _insert_section_block(self._doc_node, insert_at, new_secs)
        aot._resync()  # noqa: SLF001
        # Make sure the AoT is reachable through dict storage even when it is
        # empty (no [[..]] sections in the CST yet) and to give it stable
        # identity across re-reads.
        self._install_at_path(parts, aot)
        return aot

    @override
    def _install_section(
        self,
        parts: tuple[str, ...],
        value: Mapping[str, object] = MappingProxyType({}),
    ) -> Table:
        full_path, insert_at = self._prepare_section_slot(parts)
        new_sec = _new_section(full_path)
        _insert_section_block(self._doc_node, insert_at, [new_sec])
        view = _StdTable(self._doc_node, full_path)
        self._install_at_path(parts, view)
        for k, v in value.items():
            view[k] = v
        return view

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
                if len(kv.key.path) == 1 and _extract_trailing_comment_block(
                    kv.leading
                ):
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

    def _rebuild_separators(self) -> None:
        _apply_separator_style(self._node, self._style)

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
        """
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
        new_item = self._make_item(value, with_comma=False)
        self._node.items.insert(idx, new_item)
        self._rebuild_separators()
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
            new_items = [self._make_item(v, with_comma=False) for v in list(value)]
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
    def pop(self, index: SupportsIndex = -1) -> Any:
        item = self._node.items.pop(operator.index(index))
        self._rebuild_separators()
        self._resync()
        return _value_for(item.value)

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
        self._node.items.reverse()
        self._rebuild_separators()
        self._resync()

    @override
    def sort(
        self,
        *,
        key: Callable[[Any], object] | None = None,
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
            for _ in range(n - 1):
                self._node.items.extend(deepcopy(item) for item in base)
            self._rebuild_separators()
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
                if id(s) not in seen:
                    captured.append(s)
                    seen.add(id(s))
                for sub in self._doc_node.aot_owned_range(s):
                    if id(sub) not in seen:
                        captured.append(sub)
                        seen.add(id(sub))
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
            [header, *self._doc_node.aot_owned_range(header)] for header in own
        ]
        start = _index_of(sections, blocks[0][0])
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

    def _make_header_section(self) -> SectionNode:
        return _new_section(self._path, kind="array")

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
                    raise TOMLError(msg)
                sec.entries.append(make_keyvalue_node(k, v))
            return
        msg = (
            f"cannot append a value of type {type(value).__name__} to an "
            "array-of-tables; expected a dict or Table"
        )
        raise TOMLError(msg)

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
        new_sec = self._make_header_section()
        self._populate_section(new_sec, value)
        sections = self._doc_node.sections
        # Pick an insertion point first; blank-line decision depends on it.
        if py_index == n:
            # Append: land after the last [[path]] entry's owned range,
            # or at end of doc if no entries exist yet.
            if own:
                last = own[-1]
                owned = self._doc_node.aot_owned_range(last)
                tail = owned[-1] if owned else last
                insert_idx = _index_of(sections, tail) + 1
            else:
                insert_idx = len(sections)
        else:
            insert_idx = _index_of(sections, own[py_index])
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
        sections.insert(insert_idx, new_sec)
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
        owned = self._doc_node.aot_owned_range(target)
        sections = self._doc_node.sections
        to_remove = {id(target), *(id(s) for s in owned)}
        # Use the live entry as the popped object to preserve identity.
        popped = self[i]
        self._doc_node.sections = [s for s in sections if id(s) not in to_remove]
        self._resync()
        popped._detach()  # noqa: SLF001
        return popped

    @override
    def clear(self) -> None:
        own = self._own_sections()
        to_remove: set[int] = set()
        for s in own:
            to_remove.add(id(s))
            for sub in self._doc_node.aot_owned_range(s):
                to_remove.add(id(sub))
        sections = self._doc_node.sections
        self._doc_node.sections = [s for s in sections if id(s) not in to_remove]
        self._resync()

    @override
    def __delitem__(self, index: SupportsIndex | slice) -> None:
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            for i in sorted(indices, reverse=True):
                self.pop(i)
        else:
            self.pop(index)

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
                if not isinstance(v, Mapping):
                    msg = "AoT entry must be a mapping"
                    raise TypeError(msg)
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
        if not isinstance(value, Mapping):
            msg = "AoT entry must be a mapping"
            raise TypeError(msg)
        target = self[index]
        target.clear()
        target.update(value)

    @override
    def __iadd__(self, values: Iterable[Mapping[str, object]]) -> Self:  # type: ignore[override]
        self.extend(values)
        return self

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
        # Use the second entry's leading (the natural inter-entry separator)
        # at the boundary between repetitions, so doubling a doc with blank-line
        # separators stays visually consistent.
        inter_leading = self._block_leading(blocks[1]) if len(blocks) >= 2 else Trivia()
        repeated = list(base)
        for _ in range(n - 1):
            copy_blocks: list[list[SectionNode]] = [
                [deepcopy(s) for s in block] for block in blocks
            ]
            self._set_block_leading(copy_blocks[0], inter_leading)
            repeated.extend(s for block in copy_blocks for s in block)
        self._doc_node.sections[start : start + len(base)] = repeated
        self._resync()
        return self

    @override
    def reverse(self) -> None:
        start, blocks = self._own_blocks()
        if not blocks:
            return
        end = start + sum(len(b) for b in blocks)
        leadings = [self._block_leading(b) for b in blocks]
        blocks.reverse()
        for block, leading in zip(blocks, leadings, strict=True):
            self._set_block_leading(block, leading)
        self._doc_node.sections[start:end] = [s for block in blocks for s in block]
        self._resync()

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
        end = start + sum(len(b) for b in blocks)
        leadings = [self._block_leading(b) for b in blocks]
        pairs = list(zip(list(self), blocks, strict=True))
        if key is None:
            pairs.sort(key=lambda p: p[0], reverse=reverse)  # type: ignore[arg-type,return-value]
        else:
            pairs.sort(key=lambda p: key(p[0]), reverse=reverse)  # type: ignore[arg-type,return-value]
        new_blocks = [block for _, block in pairs]
        for block, leading in zip(new_blocks, leadings, strict=True):
            self._set_block_leading(block, leading)
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


def _index_of(sections: list[SectionNode], target: SectionNode) -> int:
    for i, s in enumerate(sections):
        if s is target:
            return i
    msg = "section not found in document (internal error)"
    raise RuntimeError(msg)  # pragma: no cover


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
