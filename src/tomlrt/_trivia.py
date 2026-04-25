"""Trivia / comment manipulation helpers.

Pure functions over the CST primitives in `tomlrt._nodes`. None
of the helpers in this module know anything about the logical
``Document``/``Table``/``Array`` view layer; they exist so that the
view layer (and the separator-style and section-construction helpers)
can compose simple trivia operations without restating bracket-by-
bracket walking logic.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Protocol

from tomlrt._errors import TOMLError
from tomlrt._nodes import (
    ArrayNode,
    CommentNode,
    InlineTableNode,
    NewlineNode,
    Trivia,
    WhitespaceNode,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from tomlrt._nodes import (
        DocumentNode,
        SectionNode,
        TriviaPiece,
        ValueNode,
    )


def _walk_newline_nodes(node: object) -> Iterator[NewlineNode]:
    """Yield every `NewlineNode` reachable from ``node``.

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
        elif dataclasses.is_dataclass(current) and not isinstance(current, type):
            stack.extend(getattr(current, f.name) for f in dataclasses.fields(current))


def _detect_newline(node: DocumentNode) -> str:
    """Return the document's line ending if uniform, else ``"\\n"``.

    Mixed-newline documents return ``"\\n"`` and are left alone — picking
    either ending would break the no-mutation round-trip invariant.
    """
    saw_any = False
    for nl in _walk_newline_nodes(node):
        saw_any = True
        if nl.text != "\r\n":
            return "\n"
    return "\r\n" if saw_any else "\n"


def _normalise_newlines(node: DocumentNode, target: str) -> None:
    """Set every `NewlineNode` in ``node`` to ``target``."""
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

    Walks into `ArrayNode` and `InlineTableNode`
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
    """``True`` iff ``trivia`` contains any `CommentNode`."""
    return any(isinstance(p, CommentNode) for p in trivia.pieces)


def _starts_with_blank_line(trivia: Trivia) -> bool:
    """``True`` iff ``trivia`` begins with a bare newline.

    A bare ``NewlineNode`` at the start of an entry's leading trivia
    is how the parser records "blank line above this line": the
    previous entry ended with its own newline, then this newline
    forms the empty line, then the entry's indent / content begins.
    """
    return bool(trivia.pieces) and isinstance(trivia.pieces[0], NewlineNode)


def _first_gap_is_blank(
    leadings: Iterable[Trivia],
    *,
    default: bool = False,
) -> bool:
    """Should new siblings adopt a blank-line gap?

    Reads the first sibling gap as the user's chosen style; later
    divergent gaps are treated as accidental. ``leadings`` excludes the
    very first sibling (which has no preceding sibling). Returns
    ``default`` when no gap is available — KV-in-table prefers
    ``False``, AoT siblings prefer ``True``.
    """
    it = iter(leadings)
    first = next(it, None)
    return default if first is None else _starts_with_blank_line(first)


def _prepend_blank_line(trivia: Trivia) -> None:
    """Prepend a blank-line ``NewlineNode`` to ``trivia`` if missing.

    Idempotent: a no-op when ``trivia`` already starts with a newline.
    """
    if not _starts_with_blank_line(trivia):
        trivia.pieces.insert(0, NewlineNode("\n"))


def _strip_comment_marker(text: str) -> str:
    """``"# foo"`` → ``"foo"``: strip leading ``#`` and one optional space.

    Trailing whitespace is preserved (it's part of the comment payload).
    """
    text = text.removeprefix("#")
    return text.removeprefix(" ")


def _format_comment(text: str) -> str:
    """Format user text as a `CommentNode` payload.

    Emits ``"# " + text`` (or ``"#"`` for empty text). Text starting
    with ``#`` is treated as literal content (``"#hashtag"`` →
    ``"# #hashtag"``). Raises [`TOMLError`][tomlrt.TOMLError] on line terminators or
    non-TAB control characters.
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

    ``min_start`` clips the scan; see `_trailing_comment_block_span`.
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
    ``min_start`` clips the scan; see `_trailing_comment_block_span`.
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

    Pass the result as ``min_start`` to `_trailing_comment_block_span`
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
    if not t.pieces:
        return True
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


def _clone_trivia(t: Trivia) -> Trivia:
    # Trivia pieces are never mutated in place — only replaced wholesale —
    # so piece refs can be shared. Only the list container needs to be fresh.
    return Trivia(list(t.pieces))
