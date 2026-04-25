"""Live, presence-filtered comment views for tables and arrays.

These ``MutableMapping`` views are returned from
[`Table.comments`][tomlrt.Table.comments],
[`Table.leading_comments`][tomlrt.Table.leading_comments],
[`Array.comments`][tomlrt.Array.comments], and
[`Array.leading_comments`][tomlrt.Array.leading_comments].
Each view is a thin facade over the CST: reads and writes go directly
to the underlying `KeyValueNode` / `ArrayItem` trivia, so the view
never holds stale state.

The four view classes share a base, `_PresenceFilteredView`,
which implements every ``MutableMapping`` method except the
view-specific ``__setitem__``. Subclasses provide the four small
hooks (``_check_key``, ``_keys``, ``_read``, ``_write_absent``) and an
optional ``_format_value`` for ``__repr__``.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, ClassVar, TypeVar

from tomlrt._nodes import (
    NewlineNode,
    Trivia,
    WhitespaceNode,
)
from tomlrt._separator import _array_indent, _logical_leading_slot
from tomlrt._trivia import (
    _extract_eol_comment,
    _extract_trailing_comment_block,
    _indent_after_last_newline,
    _replace_eol_comment,
    _replace_trailing_comment_block,
    _set_eol_comment,
    _strip_comment_marker,
)

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from tomlrt._document import Array, _StdTable
    from tomlrt._nodes import (
        KeyValueNode,
    )


_VK = TypeVar("_VK")
_VV = TypeVar("_VV")


class _PresenceFilteredView(MutableMapping[_VK, _VV], ABC):
    """Common scaffolding for the comment-views.

    A "presence-filtered" view exposes only those keys whose payload is
    currently *present* (a non-empty comment block, an EOL comment that
    actually exists, etc.). Subclasses provide:

    * `_check_key` — coerce / range-check a raw key, raising
      `TypeError` for the wrong kind and `KeyError` for
      an out-of-range value. Returns the canonicalised key.
    * `_keys` — yields every valid key (regardless of whether
      its payload is present).
    * `_read` — returns the payload, or ``None`` when absent.
    * `_write_absent` — the deletion primitive used by
      ``__delitem__``.
    * `_format_value` — used by ``__repr__``; defaults to
      ``repr``.

    ``__setitem__`` is left to subclasses because each view has its own
    slot-selection / anchor logic.
    """

    @abstractmethod
    def _check_key(self, key: object) -> _VK: ...

    @abstractmethod
    def _keys(self) -> Iterator[_VK]: ...

    @abstractmethod
    def _read(self, key: _VK) -> _VV | None: ...

    @abstractmethod
    def _write_absent(self, key: _VK) -> None: ...

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
    """Common scaffolding for `_StdTable`-backed presence-filtered views.

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

    Backed by a `_StdTable`; a key is "present" iff its
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
    """Common scaffolding for [`Array`][tomlrt.Array]-item-backed views."""

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

    Backed by an [`Array`][tomlrt.Array]. An index is "present" iff the
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
        """Force ``]`` onto a new line when the last item carries an EOL comment.

        Without this, the trailing comment swallows the closing bracket.
        """
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
