"""Comma-separator style sampling and re-application.

Used by inline arrays and inline tables to preserve (and reapply, after
mutation) the spacing convention chosen by the source — single-line vs
multi-line, trailing comma or not, indent width, etc. The helpers here
operate on the CST level (``ArrayNode`` / ``InlineTableNode`` items
satisfying `_Separated`) and know nothing about the logical
view layer in `tomlrt._document`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from tomlrt._nodes import ArrayNode, NewlineNode, Trivia, WhitespaceNode
from tomlrt._trivia import (
    _clone_trivia,
    _extract_eol_comment,
    _extract_trailing_comment_block,
    _first_indent_after_newline,
    _indent_after_last_newline,
    _is_pure_whitespace,
    _replace_eol_comment,
    _replace_trailing_comment_block,
    _split_pct_eol,
    _trivia_has_comment,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomlrt._nodes import InlineTableNode, TriviaPiece


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
    `_write_item_leadings`.

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

    Inverse of `_snapshot_item_leadings`. Also clears any stale
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
                pieces: list[TriviaPiece] = [NewlineNode("\n")]
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
    """Re-apply a sampled `_SeparatorStyle` to the items.

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
        # items[i>0].leading must be empty: inter-item content lives at the
        # tail of items[i-1].post_comma_trivia. Anything here is stale
        # residue from a previous position; snapshot/restore handles reorder.
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


def _apply_separator_after_append(
    container: ArrayNode | InlineTableNode,
    style: _SeparatorStyle,
    n_added: int = 1,
) -> None:
    """Incremental separator update after appending ``n_added`` items.

    Cheaper than `_apply_separator_style` for the common bulk-
    append case: only the previously-last item (now interior) and the
    newly appended items need touching, leaving the rest untouched.
    """
    items: Sequence[_Separated] = (
        container.items if isinstance(container, ArrayNode) else container.entries
    )
    n = len(items)
    if n == n_added:
        # Container was empty before: defer to the bulk path which
        # owns the open_pad / trailing-comma corner cases.
        _apply_separator_style(container, style)
        return

    inter_render = style.inter_separator.render()
    inter_indent = _indent_after_last_newline(style.inter_separator)

    # Previously-last item now looks like an interior item; mirror the
    # per-interior branch in _apply_separator_style so user comments
    # in trailing / post_comma_trivia survive.
    prev_tail = items[n - n_added - 1]
    if not prev_tail.has_comma:
        eol = _extract_eol_comment(prev_tail.trailing)
        prev_tail.trailing = Trivia()
        prev_tail.has_comma = True
        prev_tail.post_comma_trivia = _clone_trivia(style.inter_separator)
        if eol is not None:
            _replace_eol_comment(
                prev_tail.post_comma_trivia,
                eol,
                force_newline=True,
            )
    elif _is_pure_whitespace(prev_tail.post_comma_trivia):
        if prev_tail.post_comma_trivia.render() != inter_render:
            prev_tail.post_comma_trivia = _clone_trivia(style.inter_separator)
    else:
        _ensure_trailing_indent(prev_tail.post_comma_trivia, inter_indent)
    if _is_pure_whitespace(prev_tail.trailing) and prev_tail.trailing.pieces:
        prev_tail.trailing = Trivia()

    # Newly appended items at indices [n - n_added .. n - 2] are interior.
    for i in range(n - n_added, n - 1):
        item = items[i]
        item.leading = Trivia()
        item.has_comma = True
        item.post_comma_trivia = _clone_trivia(style.inter_separator)
        item.trailing = Trivia()

    # The genuinely new tail item gets the close-pad / trailing-comma
    # treatment.
    new_tail = items[-1]
    new_tail.leading = Trivia()
    if style.trailing_comma:
        new_tail.has_comma = True
        new_tail.post_comma_trivia = _clone_trivia(style.close_pad)
        new_tail.trailing = Trivia()
    else:
        new_tail.has_comma = False
        new_tail.post_comma_trivia = Trivia()
        new_tail.trailing = _clone_trivia(style.close_pad)


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
