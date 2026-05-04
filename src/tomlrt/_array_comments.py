"""Comment side-channel views for ``Array``.

``Array.comments`` (EOL comments per item, indexed by item position)
and ``Array.leading_comments`` (tuple of comment lines above each
item, indexed by item position) are implemented here. They share
the same encode/decode helpers and validation rules as the
section-level views in ``_comments``.

The physical layout of an item's per-item trivia (set by
``_parser.py``) is:

    leading | value | trailing | "," | post_comma_trivia

For a multi-line array the parser puts the EOL of an item with a
trailing comma into ``post_comma_trivia`` (the comment appears
after the comma on the same line). The EOL of a *last* item with
no trailing comma instead lands in ``trailing``. Conversely, a
"leading comment" written above item *i* (i > 0) ends up in the
``post_comma_trivia`` of item *i-1* (after that item's EOL section,
if any). Item 0's leading lives in ``items[0].leading``.

The view code therefore reads from a *combined* trivia region per
item and writes back into a single canonical place:

    EOL of item i  : trailing if not has_comma else post_comma_trivia
    leading of i   : items[0].leading      (if i == 0)
                     items[i-1].post_comma_trivia (if i > 0),
                     placed *after* item (i-1)'s EOL section.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, cast

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tomlrt._comments import (
    _decode_comment,
    _encode_comment,
    _validate_comment_seq,
    _validate_comment_text,
)
from tomlrt._errors import TOMLError
from tomlrt._trivia import (
    CommentNode,
    NewlineNode,
    WhitespaceNode,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tomlrt._array import Array
    from tomlrt._trivia import (
        Trivia,
        TriviaPiece,
    )
    from tomlrt._values import ArrayItem


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _newline_text(arr: Array) -> str:
    from tomlrt._container import Document  # noqa: PLC0415

    lr = arr._layout_root()  # noqa: SLF001
    return lr._newline if isinstance(lr, Document) else "\n"  # noqa: SLF001


def _items_or_raise(arr: Array) -> list[ArrayItem]:
    if arr._value is None:  # noqa: SLF001
        msg = "comment API is not available on detached Arrays"
        raise TOMLError(msg)
    return arr._value.items  # noqa: SLF001


def _check_index(arr: Array, key: object) -> int:
    if not isinstance(key, int) or isinstance(key, bool):
        msg = f"Array comment indices must be int, not {type(key).__name__}"
        raise TypeError(msg)
    items = _items_or_raise(arr)
    n = len(items)
    idx = key if key >= 0 else key + n
    if idx < 0 or idx >= n:
        raise KeyError(key)
    return idx


def _split_eol_section(
    pieces: list[TriviaPiece],
) -> tuple[list[TriviaPiece], list[TriviaPiece]]:
    """Split pieces into (eol_section, rest).

    The EOL section is the leading run of non-newline pieces up to
    and including the first ``NewlineNode``, *iff* it contains a
    ``CommentNode``. Otherwise the EOL section is empty and `pieces`
    is returned unchanged as `rest`.
    """
    saw_comment = False
    for idx, p in enumerate(pieces):
        if isinstance(p, CommentNode):
            saw_comment = True
        elif isinstance(p, NewlineNode):
            if saw_comment:
                return pieces[: idx + 1], pieces[idx + 1 :]
            return [], list(pieces)
    if saw_comment:
        return list(pieces), []
    return [], list(pieces)


def _eol_text_from(pieces: list[TriviaPiece]) -> str | None:
    for p in pieces:
        if isinstance(p, NewlineNode):
            return None
        if isinstance(p, CommentNode):
            return _decode_comment(p.text)
    return None


def _item_eol(item: ArrayItem) -> str | None:
    # Try the canonical "owner" first (trailing if no comma, post_comma if
    # comma) but also fall back to the other location for layouts the
    # parser may produce out-of-band.
    if item.has_comma:
        eol = _eol_text_from(list(item.post_comma_trivia.pieces))
        if eol is not None:
            return eol
        return _eol_text_from(list(item.trailing.pieces))
    eol = _eol_text_from(list(item.trailing.pieces))
    if eol is not None:
        return eol
    return _eol_text_from(list(item.post_comma_trivia.pieces))


def _eol_target(item: ArrayItem) -> Trivia:
    return item.post_comma_trivia if item.has_comma else item.trailing


def _ensure_multiline(arr: Array) -> None:
    if not arr.multiline:
        arr.set_multiline(multiline=True)


def _leading_pieces(items: list[ArrayItem], i: int) -> list[TriviaPiece]:
    """The pieces that constitute item *i*'s leading region."""
    if i == 0:
        return list(items[0].leading.pieces)
    prev = items[i - 1]
    _eol, rest = _split_eol_section(list(prev.post_comma_trivia.pieces))
    return rest + list(items[i].leading.pieces)


def _comments_from_lines(pieces: list[TriviaPiece]) -> tuple[str, ...]:
    return tuple(_decode_comment(p.text) for p in pieces if isinstance(p, CommentNode))


def _slot_indent(arr: Array) -> str:
    """Best-effort indent string for this array's items."""
    if arr._value is None:  # noqa: SLF001
        return "  "
    items = arr._value.items  # noqa: SLF001
    if items:
        # Prefer item 0's leading whitespace immediately before the value.
        for p in reversed(items[0].leading.pieces):
            if isinstance(p, WhitespaceNode):
                return p.text
            if isinstance(p, NewlineNode):
                break
        for prev in items:
            for p in reversed(prev.post_comma_trivia.pieces):
                if isinstance(p, WhitespaceNode):
                    return p.text
                if isinstance(p, NewlineNode):
                    break
    return "  "


# ---------------------------------------------------------------------------
# EOL view
# ---------------------------------------------------------------------------


class ArrayEolView(MutableMapping[int, str]):
    __slots__ = ("_arr",)

    def __init__(self, arr: Array) -> None:
        self._arr = arr

    @override
    def __getitem__(self, key: int) -> str:
        idx = _check_index(self._arr, key)
        items = _items_or_raise(self._arr)
        eol = _item_eol(items[idx])
        if eol is None:
            raise KeyError(key)
        return eol

    @override
    def __setitem__(self, key: int, value: str) -> None:
        v: object = value
        if not isinstance(v, str):
            msg = "comment text must be a string"
            raise TypeError(msg)
        _validate_comment_text(value)
        _ensure_multiline(self._arr)
        idx = _check_index(self._arr, key)
        items = _items_or_raise(self._arr)
        item = items[idx]
        nl = _newline_text(self._arr)
        target = _eol_target(item)
        existing_eol, rest = _split_eol_section(list(target.pieces))
        if not existing_eol and rest and isinstance(rest[0], NewlineNode):
            # Original layout had no EOL but did have a line break between
            # this item and the next; our new EOL synthesises its own
            # newline, so drop the leading NL from `rest` to avoid leaving
            # behind a blank line.
            rest = rest[1:]
        new_eol: list[TriviaPiece] = [
            WhitespaceNode(" "),
            CommentNode(_encode_comment(value)),
            NewlineNode(nl),
        ]
        # If there was no rest (last item, no following structure) and we
        # don't have a comma, the closing `]` would otherwise be on the same
        # line. The newline we synthesised in `new_eol` already prevents
        # that.
        target.pieces = [*new_eol, *rest]

    @override
    def __delitem__(self, key: int) -> None:
        idx = _check_index(self._arr, key)
        items = _items_or_raise(self._arr)
        item = items[idx]
        # Try the canonical target first, then the other.
        candidates: list[Trivia] = (
            [item.post_comma_trivia, item.trailing]
            if item.has_comma
            else [item.trailing, item.post_comma_trivia]
        )
        for target in candidates:
            eol, rest = _split_eol_section(list(target.pieces))
            if not eol:
                continue
            if target is item.trailing:
                # No-comma case: drop the EOL entirely (no need for a
                # synthesised newline because the next item's leading or
                # final_trivia carries the line break).
                target.pieces = list(rest)
            else:
                # post_comma_trivia: the EOL section we removed included a
                # NewlineNode that separated this line from the next item.
                # Replace it with a plain newline so the line break stays.
                nl = _newline_text(self._arr)
                target.pieces = [NewlineNode(nl), *rest]
            return
        raise KeyError(key)

    @override
    def __iter__(self) -> Iterator[int]:
        items = _items_or_raise(self._arr)
        for i, it in enumerate(items):
            if _item_eol(it) is not None:
                yield i

    @override
    def __len__(self) -> int:
        return sum(1 for _ in iter(self))

    @override
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, int) or isinstance(key, bool):
            return False
        try:
            idx = _check_index(self._arr, key)
        except KeyError:
            return False
        items = _items_or_raise(self._arr)
        return _item_eol(items[idx]) is not None

    @override
    def __repr__(self) -> str:
        return repr(dict(self))


class ArrayLeadingView(MutableMapping[int, tuple[str, ...]]):
    __slots__ = ("_arr",)

    def __init__(self, arr: Array) -> None:
        self._arr = arr

    def _read(self, idx: int) -> tuple[str, ...]:
        items = _items_or_raise(self._arr)
        return _comments_from_lines(_leading_pieces(items, idx))

    @override
    def __getitem__(self, key: int) -> tuple[str, ...]:
        idx = _check_index(self._arr, key)
        out = self._read(idx)
        if not out:
            raise KeyError(key)
        return out

    def _write_item0(self, lines: tuple[str, ...]) -> None:
        items = _items_or_raise(self._arr)
        item0 = items[0]
        nl = _newline_text(self._arr)
        ind = _slot_indent(self._arr)
        # Drop existing comments from leading; preserve at most the
        # initial newline that separates the value from the opening `[`.
        kept_initial: list[TriviaPiece] = []
        seen_nl = False
        for p in item0.leading.pieces:
            if isinstance(p, NewlineNode) and not seen_nl:
                kept_initial.append(p)
                seen_nl = True
                break
            if isinstance(p, NewlineNode):
                kept_initial.append(p)
                break
        if not seen_nl:
            kept_initial = [NewlineNode(nl)]
        new_pieces: list[TriviaPiece] = list(kept_initial)
        for line in lines:
            new_pieces.append(WhitespaceNode(ind))
            new_pieces.append(CommentNode(_encode_comment(line)))
            new_pieces.append(NewlineNode(nl))
        new_pieces.append(WhitespaceNode(ind))
        item0.leading.pieces = new_pieces

    def _write_item_after_prev(self, idx: int, lines: tuple[str, ...]) -> None:
        items = _items_or_raise(self._arr)
        prev = items[idx - 1]
        item = items[idx]
        nl = _newline_text(self._arr)
        ind = _slot_indent(self._arr)
        # Split prev.post_comma_trivia: keep EOL section, replace the rest
        # with our new leading lines + indent.
        eol_sec, _rest = _split_eol_section(list(prev.post_comma_trivia.pieces))
        new_after_eol: list[TriviaPiece] = []
        if not eol_sec:
            # No prior EOL on prev — we need a newline first to start a new
            # line for the leading comments.
            new_after_eol.append(NewlineNode(nl))
        for line in lines:
            new_after_eol.append(WhitespaceNode(ind))
            new_after_eol.append(CommentNode(_encode_comment(line)))
            new_after_eol.append(NewlineNode(nl))
        new_after_eol.append(WhitespaceNode(ind))
        prev.post_comma_trivia.pieces = [*eol_sec, *new_after_eol]
        # item.leading should be empty so the comments aren't duplicated.
        item.leading.pieces = []

    @override
    def __setitem__(self, key: int, value: tuple[str, ...] | list[str]) -> None:
        seq = _validate_comment_seq(cast("Any", value), "leading_comments")
        _ensure_multiline(self._arr)
        idx = _check_index(self._arr, key)
        if not seq:
            self._delete(idx, allow_missing=True)
            return
        if idx == 0:
            self._write_item0(seq)
        else:
            self._write_item_after_prev(idx, seq)

    def _delete(self, idx: int, *, allow_missing: bool) -> None:
        items = _items_or_raise(self._arr)
        nl = _newline_text(self._arr)
        ind = _slot_indent(self._arr)
        if idx == 0:
            item0 = items[0]
            # Strip everything except a leading newline + indent.
            if not _comments_from_lines(list(item0.leading.pieces)):
                if not allow_missing:
                    raise KeyError(idx)
                return
            item0.leading.pieces = [NewlineNode(nl), WhitespaceNode(ind)]
            return
        prev = items[idx - 1]
        item = items[idx]
        eol_sec, rest = _split_eol_section(list(prev.post_comma_trivia.pieces))
        leading_existing = _comments_from_lines(rest) + _comments_from_lines(
            list(item.leading.pieces),
        )
        if not leading_existing:
            if not allow_missing:
                raise KeyError(idx)
            return
        # Replace the leading run with a clean indent before the next item.
        new_after_eol: list[TriviaPiece] = []
        if not eol_sec:
            new_after_eol.append(NewlineNode(nl))
        new_after_eol.append(WhitespaceNode(ind))
        prev.post_comma_trivia.pieces = [*eol_sec, *new_after_eol]
        item.leading.pieces = []

    @override
    def __delitem__(self, key: int) -> None:
        idx = _check_index(self._arr, key)
        self._delete(idx, allow_missing=False)

    @override
    def __iter__(self) -> Iterator[int]:
        items = _items_or_raise(self._arr)
        for i in range(len(items)):
            if _comments_from_lines(_leading_pieces(items, i)):
                yield i

    @override
    def __len__(self) -> int:
        return sum(1 for _ in iter(self))

    @override
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, int) or isinstance(key, bool):
            return False
        try:
            idx = _check_index(self._arr, key)
        except KeyError:
            return False
        items = _items_or_raise(self._arr)
        return bool(_comments_from_lines(_leading_pieces(items, idx)))

    @override
    def __repr__(self) -> str:
        # Test contract spells the values as Python lists, e.g. "{0: ['first']}".
        return repr({k: list(v) for k, v in self.items()})


# ---------------------------------------------------------------------------
# Reorder helpers — snapshot/clear/apply per-item comments raw, so structural
# reordering ops (`reverse`, `sort`, `insert`, `pop`) can move comments
# along with the items they belong to without losing exact byte spelling.
# ---------------------------------------------------------------------------


def _raw_comment_lines(pieces: list[TriviaPiece]) -> tuple[str, ...]:
    """Raw `#...` text for each CommentNode in *pieces* (preserves spelling)."""
    return tuple(p.text for p in pieces if isinstance(p, CommentNode))


def _raw_eol_text(item: ArrayItem) -> str | None:
    """Raw `# ...` text of *item*'s EOL comment, or None."""
    if item.has_comma:
        for p in item.post_comma_trivia.pieces:
            if isinstance(p, NewlineNode):
                break
            if isinstance(p, CommentNode):
                return p.text
    for p in item.trailing.pieces:
        if isinstance(p, NewlineNode):
            break
        if isinstance(p, CommentNode):
            return p.text
    if item.has_comma:
        for p in item.trailing.pieces:
            if isinstance(p, NewlineNode):
                break
            if isinstance(p, CommentNode):
                return p.text
    else:
        for p in item.post_comma_trivia.pieces:
            if isinstance(p, NewlineNode):
                break
            if isinstance(p, CommentNode):
                return p.text
    return None


def snapshot_comments(
    arr: Array,
) -> tuple[list[tuple[str, ...]], list[str | None]]:
    """Return per-item (raw-leading-lines, raw-eol-text) snapshots."""
    if arr._value is None:  # noqa: SLF001
        return [], []
    items = arr._value.items  # noqa: SLF001
    leadings = [
        _raw_comment_lines(_leading_pieces(items, i)) for i in range(len(items))
    ]
    eols = [_raw_eol_text(items[i]) for i in range(len(items))]
    return leadings, eols


def clear_all_comments(arr: Array) -> None:
    """Strip per-item comments from all items via the canonical deleters."""
    if arr._value is None:  # noqa: SLF001
        return
    items = arr._value.items  # noqa: SLF001
    n = len(items)
    leading = ArrayLeadingView(arr)
    eol = ArrayEolView(arr)
    for i in range(n):
        leading._delete(i, allow_missing=True)  # noqa: SLF001
        with contextlib.suppress(KeyError):
            del eol[i]


def _write_eol_raw(arr: Array, idx: int, raw_text: str) -> None:
    items = _items_or_raise(arr)
    item = items[idx]
    nl = _newline_text(arr)
    target = _eol_target(item)
    existing_eol, rest = _split_eol_section(list(target.pieces))
    if not existing_eol and rest and isinstance(rest[0], NewlineNode):
        rest = rest[1:]
    new_eol: list[TriviaPiece] = [
        WhitespaceNode(" "),
        CommentNode(raw_text),
        NewlineNode(nl),
    ]
    target.pieces = [*new_eol, *rest]


def _write_leading_raw(arr: Array, idx: int, raw_lines: tuple[str, ...]) -> None:
    items = _items_or_raise(arr)
    nl = _newline_text(arr)
    ind = _slot_indent(arr)
    if idx == 0:
        item0 = items[0]
        kept_initial: list[TriviaPiece] = []
        seen_nl = False
        for p in item0.leading.pieces:
            if isinstance(p, NewlineNode):
                kept_initial.append(p)
                seen_nl = True
                break
        if not seen_nl:
            kept_initial = [NewlineNode(nl)]
        new_pieces: list[TriviaPiece] = list(kept_initial)
        for raw in raw_lines:
            new_pieces.append(WhitespaceNode(ind))
            new_pieces.append(CommentNode(raw))
            new_pieces.append(NewlineNode(nl))
        new_pieces.append(WhitespaceNode(ind))
        item0.leading.pieces = new_pieces
        return
    prev = items[idx - 1]
    item = items[idx]
    eol_sec, _rest = _split_eol_section(list(prev.post_comma_trivia.pieces))
    new_after_eol: list[TriviaPiece] = []
    if not eol_sec:
        new_after_eol.append(NewlineNode(nl))
    for raw in raw_lines:
        new_after_eol.append(WhitespaceNode(ind))
        new_after_eol.append(CommentNode(raw))
        new_after_eol.append(NewlineNode(nl))
    new_after_eol.append(WhitespaceNode(ind))
    prev.post_comma_trivia.pieces = [*eol_sec, *new_after_eol]
    item.leading.pieces = []


def apply_comments(
    arr: Array,
    leadings: list[tuple[str, ...]],
    eols: list[str | None],
) -> None:
    """Re-apply per-item comments after a structural reorder.

    Writes EOL comments first (so leading writes can split a known
    canonical EOL section out of the prev item's post_comma_trivia),
    then leading blocks. Skips empty entries.
    """
    if arr._value is None:  # noqa: SLF001
        return
    items = arr._value.items  # noqa: SLF001
    n = len(items)
    any_comment = any(eols[i] is not None for i in range(n)) or any(
        leadings[i] for i in range(n)
    )
    if any_comment:
        _ensure_multiline(arr)
    for i in range(n):
        e = eols[i]
        if e is not None:
            _write_eol_raw(arr, i, e)
    for i in range(n):
        if leadings[i]:
            _write_leading_raw(arr, i, leadings[i])
