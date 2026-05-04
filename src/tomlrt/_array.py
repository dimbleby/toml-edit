"""Array views.

`Array(list)` backs an inline-array TOML value (`[1, 2, 3]`). `AoT`
(array-of-tables) is a `list[Table]` whose entries are individually
backed by `AoTEntry` records and whose elements are `Table` views.

Phase 2 surface: read-only access (length, indexing, iteration, value
decoding for plain inline-array elements). Mutation arrives in
Phase 3 (inline) and Phase 4 (AoT and multi-line array).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, SupportsIndex

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    from tomlrt._container import Container, Document, Table
    from tomlrt._values import ArrayValue


class Array(list[Any]):
    """An inline array (`[ ... ]`)."""

    __slots__ = ("_multiline", "_value")

    def __init__(
        self,
        items: Any = None,
        *,
        multiline: bool = False,
    ) -> None:
        super().__init__()
        from tomlrt._values import ArrayValue  # noqa: PLC0415

        self._value: ArrayValue | None = ArrayValue()
        self._multiline: bool = multiline
        if items is None:
            return
        from tomlrt._container import _synth_inline_array  # noqa: PLC0415

        val, arr = _synth_inline_array(list(items), layout_root=None, owner=None)
        self._value = val
        for v in arr:
            list.append(self, v)
        if multiline and val.items:
            from tomlrt._trivia import (  # noqa: PLC0415
                NewlineNode,
                Trivia,
                WhitespaceNode,
            )

            style = _ArrayStyle(
                is_multiline=True,
                inter_separator=Trivia(
                    [NewlineNode(text="\n"), WhitespaceNode(text="    ")]
                ),
                trailing_comma=True,
                trailing_post=Trivia([NewlineNode(text="\n")]),
            )
            for it in val.items:
                it.leading = Trivia()
            val.items[0].leading = Trivia(
                [NewlineNode(text="\n"), WhitespaceNode(text="    ")]
            )
            _renormalise_commas(val.items, style)

    def to_list(self) -> list[Any]:
        """Materialise a plain-Python ``list`` (recursive)."""
        from tomlrt._container import _to_python  # noqa: PLC0415

        return [_to_python(x) for x in self]

    def array(self, index: int) -> Array:
        """Return ``self[index]`` typed as a nested `Array`."""
        v = self[index]
        if not isinstance(v, Array):
            msg = f"item at {index} is {type(v).__name__}, not an Array"
            raise TypeError(msg)
        return v

    def table(self, index: int) -> Table:
        """Return ``self[index]`` typed as a `Table`."""
        from tomlrt._container import Table  # noqa: PLC0415

        v = self[index]
        if not isinstance(v, Table):
            msg = f"item at {index} is {type(v).__name__}, not a Table"
            raise TypeError(msg)
        return v

    def get_array(self, index: int, default: Any = None) -> Any:
        """Like `array(index)` but returns ``default`` for out-of-range."""
        if index < -len(self) or index >= len(self):
            return default
        return self.array(index)

    def get_table(self, index: int, default: Any = None) -> Any:
        """Like `table(index)` but returns ``default`` for out-of-range."""
        if index < -len(self) or index >= len(self):
            return default
        return self.table(index)

    # ---- mutation -----------------------------------------------------

    def _layout_root(self) -> Document | None:
        """The owning document, by walking via `_value` ownership.

        We don't directly track this — Arrays attached to a document
        get it transitively via the KV slot. For synthesis we need it
        so nested values can resolve dotted positions; passing ``None``
        for orphan arrays is fine since we only synthesise scalars/
        inline values that don't need a layout root.
        """
        return None

    def _owner_aot(self) -> Any:
        return None

    def _style(self) -> _ArrayStyle:
        return _detect_style(self._value, multiline_flag=self._multiline)

    def _synth_cst(self, value: Any) -> Any:
        from tomlrt._container import _synth_value  # noqa: PLC0415

        cst, decoded = _synth_value(
            value,
            layout_root=self._layout_root(),
            parent=None,
            path=(),
            owner=self._owner_aot(),
        )
        return cst, decoded

    @override
    def append(self, value: Any) -> None:
        from tomlrt._errors import TOMLError  # noqa: PLC0415

        if isinstance(value, AoT):
            msg = "Cannot store an array-of-tables inside an inline array"
            raise TOMLError(msg)
        if self._value is None:
            msg = "Array.append on detached Array (no backing CST)"
            raise NotImplementedError(msg)
        cst, decoded = self._synth_cst(value)
        items = self._value.items
        style = self._style()
        new_item = _new_item(cst, leading_first=not items, style=style)
        if items:
            _flip_to_internal(items[-1], style)
        items.append(new_item)
        list.append(self, decoded)

    @override
    def extend(self, values: Any) -> None:
        for v in values:
            self.append(v)

    @override
    def clear(self) -> None:
        if self._value is not None:
            self._value.items.clear()
            # Drop any inter-item trivia clutter; preserve the bracket
            # leading captured in final_trivia.
        list.clear(self)

    @override
    def pop(self, index: SupportsIndex = -1) -> Any:
        if self._value is None:
            msg = "Array.pop on detached Array"
            raise NotImplementedError(msg)
        decoded = list.pop(self, index)
        i = int(index)
        if i < 0:
            i += len(self._value.items)
        items = self._value.items
        items.pop(i)
        if items:
            _flip_to_terminal(items[-1], self._style())
        return decoded

    @override
    def remove(self, value: Any) -> None:
        for i, v in enumerate(self):
            if v == value:
                del self[i]
                return
        msg = "Array.remove(x): x not in array"
        raise ValueError(msg)

    @override
    def insert(self, index: SupportsIndex, value: Any) -> None:
        if self._value is None:
            msg = "Array.insert on detached Array"
            raise NotImplementedError(msg)
        cst, decoded = self._synth_cst(value)
        i = int(index)
        n = len(self)
        if i < 0:
            i = max(0, n + i)
        i = min(i, n)
        items = self._value.items
        style = self._style()
        if i == 0 and items:
            # Inheriting position 0's leading: the new item adopts
            # the prior items[0].leading (which carries any post-`[`
            # comment), and the displaced item gets a fresh internal
            # leading.
            adopted_leading = items[0].leading
            displaced = items[0]
            displaced.leading = _internal_leading(style)
            new_item = _new_item(
                cst, leading_first=True, style=style, leading=adopted_leading
            )
        else:
            new_item = _new_item(cst, leading_first=False, style=style)
        # Insert into items at position i.
        items.insert(i, new_item)
        # Make sure the new item has has_comma=True if not last; if it's
        # the new last (i==len(items)-1), it should follow trailing
        # policy.
        is_last_after = i == len(items) - 1
        if is_last_after:
            # Old prior-last needs internal flip if it was a singleton.
            # Restore terminal status on this one via _flip_to_terminal.
            if len(items) >= 2:
                _flip_to_internal(items[-2], style)
            _flip_to_terminal(new_item, style)
        else:
            # Internal — ensure trailing comma + standard separator.
            _flip_to_internal(new_item, style)
        list.insert(self, i, decoded)

    @override
    def reverse(self) -> None:
        if self._value is None:
            list.reverse(self)
            return
        items = self._value.items
        items.reverse()
        list.reverse(self)
        _renormalise_commas(items, self._style())

    @override
    def sort(self, *, key: Any = None, reverse: bool = False) -> None:
        if self._value is None:
            list.sort(self, key=key, reverse=reverse)
            return
        n = len(self)
        if key is None:
            order = sorted(range(n), key=lambda i: self[i], reverse=reverse)
        else:
            order = sorted(range(n), key=lambda i: key(self[i]), reverse=reverse)
        items = self._value.items
        new_items = [items[j] for j in order]
        new_decoded = [self[j] for j in order]
        items[:] = new_items
        list.clear(self)
        for v in new_decoded:
            list.append(self, v)
        _renormalise_commas(items, self._style())

    @override
    def __setitem__(
        self,
        key: SupportsIndex | slice,
        value: Any,
    ) -> None:
        if self._value is None:
            msg = "Array.__setitem__ on detached Array"
            raise NotImplementedError(msg)
        if isinstance(key, slice):
            try:
                values = list(value)
            except TypeError as exc:
                msg = "can only assign an iterable"
                raise TypeError(msg) from exc
            # Compute target indices and replace items in place.
            indices = list(range(*key.indices(len(self))))
            if key.step is not None and key.step != 1 and len(values) != len(indices):
                msg = (
                    f"attempt to assign sequence of size {len(values)} "
                    f"to extended slice of size {len(indices)}"
                )
                raise ValueError(msg)
            new_csts = []
            new_decoded = []
            for v in values:
                cst, dec = self._synth_cst(v)
                new_csts.append(cst)
                new_decoded.append(dec)
            items = self._value.items
            style = self._style()
            # Build the new ArrayItem list segment.
            new_segment: list[Any] = []
            slice_start = key.start or 0
            for i, cst in enumerate(new_csts):
                first_in_arr = slice_start == 0 and i == 0 and not items[:slice_start]
                new_segment.append(
                    _new_item(cst, leading_first=first_in_arr, style=style)
                )
            items[key] = new_segment
            list.__setitem__(self, key, new_decoded)
            _renormalise_commas(items, style)
            return
        # int index: just replace the value CST in place.
        i = int(key)
        cst, dec = self._synth_cst(value)
        items = self._value.items
        if i < 0:
            i += len(items)
        items[i].value = cst
        list.__setitem__(self, key, dec)

    @override
    def __delitem__(self, key: SupportsIndex | slice) -> None:
        if self._value is None:
            list.__delitem__(self, key)
            return
        items = self._value.items
        del items[key]
        list.__delitem__(self, key)
        if items:
            _flip_to_terminal(items[-1], self._style())

    @override
    def __iadd__(self, other: Any) -> Self:
        self.extend(other)
        return self

    @override
    def __imul__(self, n: SupportsIndex) -> Self:
        count = int(n)
        if count <= 0:
            self.clear()
            return self
        if count == 1 or self._value is None:
            return self
        # Snapshot original items + values, then append count-1 copies.
        from copy import deepcopy  # noqa: PLC0415

        original_items = list(self._value.items)
        original_values = list(self)
        for _ in range(count - 1):
            for src_item, src_val in zip(original_items, original_values, strict=True):
                cloned = _clone_item(src_item)
                style = self._style()
                items = self._value.items
                if items:
                    _flip_to_internal(items[-1], style)
                # First-of-original gets internal leading too (since we're
                # now mid-array).
                cloned.leading = _internal_leading(style)
                items.append(cloned)
                list.append(self, deepcopy(src_val))
        if self._value.items:
            _flip_to_terminal(self._value.items[-1], self._style())
        return self


# ---------------------------------------------------------------------------
# Style detection + ArrayItem builders
# ---------------------------------------------------------------------------


class _ArrayStyle:
    """Inferred separator + trailing-comma policy for an Array."""

    __slots__ = ("inter_separator", "is_multiline", "trailing_comma", "trailing_post")

    def __init__(
        self,
        *,
        is_multiline: bool,
        inter_separator: Any,  # Trivia
        trailing_comma: bool,
        trailing_post: Any,  # Trivia
    ) -> None:
        self.is_multiline = is_multiline
        self.inter_separator = inter_separator
        self.trailing_comma = trailing_comma
        self.trailing_post = trailing_post


def _detect_style(value: ArrayValue | None, *, multiline_flag: bool) -> _ArrayStyle:
    from tomlrt._trivia import NewlineNode, Trivia, WhitespaceNode  # noqa: PLC0415

    if value is None:
        return _ArrayStyle(
            is_multiline=multiline_flag,
            inter_separator=Trivia([WhitespaceNode(text=" ")]),
            trailing_comma=multiline_flag,
            trailing_post=Trivia(),
        )
    items = value.items
    has_newline = any(_trivia_has_newline(it.post_comma_trivia) for it in items) or (
        _trivia_has_newline(value.final_trivia)
    )
    is_multiline = has_newline or multiline_flag
    # Inter-item separator: first internal item's post_comma_trivia.
    inter_sep = None
    for it in items:
        if it.has_comma and (it is not items[-1] or len(items) == 1):
            inter_sep = _clone_trivia(it.post_comma_trivia)
            break
    if inter_sep is None:
        if is_multiline:
            inter_sep = Trivia([NewlineNode(text="\n"), WhitespaceNode(text="    ")])
        else:
            inter_sep = Trivia([WhitespaceNode(text=" ")])
    # Trailing-comma policy: the last item's has_comma if any.
    if items:
        trailing_comma = items[-1].has_comma
        trailing_post = (
            _clone_trivia(items[-1].post_comma_trivia)
            if items[-1].has_comma
            else Trivia()
        )
    else:
        trailing_comma = is_multiline
        trailing_post = Trivia([NewlineNode(text="\n")]) if is_multiline else Trivia()
    return _ArrayStyle(
        is_multiline=is_multiline,
        inter_separator=inter_sep,
        trailing_comma=trailing_comma,
        trailing_post=trailing_post,
    )


def _trivia_has_newline(trivia: Any) -> bool:
    from tomlrt._trivia import NewlineNode  # noqa: PLC0415

    return any(isinstance(p, NewlineNode) for p in trivia.pieces)


def _clone_trivia(trivia: Any) -> Any:
    from tomlrt._trivia import Trivia  # noqa: PLC0415

    return Trivia(list(trivia.pieces))


def _internal_leading(_style: _ArrayStyle) -> Any:
    from tomlrt._trivia import Trivia  # noqa: PLC0415

    return Trivia()  # internal items get blank leading; separator is in
    # the prior item's post_comma_trivia.


def _new_item(
    cst: Any,
    *,
    leading_first: bool,
    style: _ArrayStyle,
    leading: Any | None = None,
) -> Any:
    from tomlrt._trivia import Trivia  # noqa: PLC0415
    from tomlrt._values import ArrayItem  # noqa: PLC0415

    if leading is None:
        leading = Trivia() if leading_first else _internal_leading(style)
    return ArrayItem(
        leading=leading,
        value=cst,
        trailing=Trivia(),
        has_comma=False,  # caller will normalise via _flip_*.
        post_comma_trivia=Trivia(),
    )


def _flip_to_internal(item: Any, style: _ArrayStyle) -> None:
    """Make ``item`` look like an internal (non-last) item."""
    item.has_comma = True
    item.post_comma_trivia = _clone_trivia(style.inter_separator)


def _flip_to_terminal(item: Any, style: _ArrayStyle) -> None:
    """Make ``item`` look like the terminal (last) item per style."""
    item.has_comma = style.trailing_comma
    item.post_comma_trivia = (
        _clone_trivia(style.trailing_post) if style.trailing_comma else _trivia_empty()
    )


def _trivia_empty() -> Any:
    from tomlrt._trivia import Trivia  # noqa: PLC0415

    return Trivia()


def _renormalise_commas(items: list[Any], style: _ArrayStyle) -> None:
    """Reset has_comma + post_comma_trivia across ``items`` per style."""
    if not items:
        return
    for it in items[:-1]:
        _flip_to_internal(it, style)
    _flip_to_terminal(items[-1], style)


def _clone_item(item: Any) -> Any:
    from copy import deepcopy  # noqa: PLC0415

    return deepcopy(item)


class AoT(list["Table"]):
    """A `[[name]]`-style array of tables.

    Carries minimal logical-attachment state (`_layout_root`, `_path`,
    `_parent`) that survives the empty-AoT case (where `self[-1]`
    cannot be consulted as a fallback parent reference).
    """

    __slots__ = ("_layout_root", "_parent", "_path")

    def __init__(self, entries: Any = None) -> None:
        super().__init__()
        self._layout_root: Document | None = None
        self._path: tuple[str, ...] = ()
        self._parent: Container | None = None
        if entries is not None:
            for e in entries:
                list.append(self, _make_unattached_entry(e))

    def to_list(self) -> list[dict[str, Any]]:
        """Materialise a list of plain-Python ``dict``s (recursive)."""
        return [t.to_dict() for t in self]

    def add(self, body: object | None = None) -> Table:
        """Append a fresh ``[[path]]`` entry and return its `Table` view.

        ``body`` may be a Mapping (initial body content) or ``None``
        (empty entry). The AoT must be attached to a document.
        """
        from tomlrt import _layout_ops  # noqa: PLC0415
        from tomlrt._container import Table  # noqa: PLC0415

        if self._layout_root is None:
            list.append(self, _make_unattached_entry(body))
            return self[-1]
        result = _layout_ops.add_aot_entry(self, body)
        assert isinstance(result, Table)
        return result

    # ------------------------------------------------------------------
    # Supported list-mutator surface (Phase 4-minimal).
    # Anything not implemented here is overridden below to fail closed
    # rather than corrupt the doc-stream via inherited `list` behaviour.
    # ------------------------------------------------------------------

    @override
    def pop(self, index: SupportsIndex = -1) -> Table:
        from tomlrt import _layout_ops  # noqa: PLC0415

        idx = int(index)
        n = len(self)
        if not -n <= idx < n:
            msg = "pop index out of range"
            raise IndexError(msg)
        if self._layout_root is None:
            return list.pop(self, idx)
        result = _layout_ops.remove_aot_entry(self, idx)
        from tomlrt._container import Table  # noqa: PLC0415

        assert isinstance(result, Table)
        return result

    @override
    def __delitem__(self, key: SupportsIndex | slice) -> None:
        from tomlrt import _layout_ops  # noqa: PLC0415

        if isinstance(key, slice):
            if self._layout_root is None:
                list.__delitem__(self, key)
                return
            indices = sorted(range(*key.indices(len(self))), reverse=True)
            for i in indices:
                _layout_ops.remove_aot_entry(self, i)
            return
        if self._layout_root is None:
            list.__delitem__(self, key)
            return
        _layout_ops.remove_aot_entry(self, int(key))

    @override
    def clear(self) -> None:
        from tomlrt import _layout_ops  # noqa: PLC0415

        if self._layout_root is None:
            list.clear(self)
            return
        while len(self) > 0:
            _layout_ops.remove_aot_entry(self, -1)

    @override
    def __setitem__(  # type: ignore[override]
        self, index: int | slice, value: object
    ) -> None:
        from tomlrt import _layout_ops  # noqa: PLC0415

        if isinstance(index, slice):
            try:
                values = list(value)  # type: ignore[call-overload]
            except TypeError as exc:
                msg = "can only assign an iterable"
                raise TypeError(msg) from exc
            indices = list(range(*index.indices(len(self))))
            if (
                index.step is not None
                and index.step != 1
                and len(values) != len(indices)
            ):
                msg = (
                    f"attempt to assign sequence of size {len(values)} "
                    f"to extended slice of size {len(indices)}"
                )
                raise ValueError(msg)
            # Validate every assigned value is a Mapping/Table BEFORE
            # mutating the AoT (atomicity preflight).
            from collections.abc import Mapping  # noqa: PLC0415

            for v in values:
                if not isinstance(v, Mapping):
                    msg = f"AoT entries must be Mapping/Table; got {type(v).__name__}"
                    raise TypeError(msg)
            if self._layout_root is None:
                list.__setitem__(
                    self, index, [_make_unattached_entry(v) for v in values]
                )
                return
            # For contiguous step == 1: replace by delete-range then
            # insert at the start index. Order matters: build new
            # entries via add (appended to end), then renormalise.
            if index.step is None or index.step == 1:
                start = index.indices(len(self))[0]
                # Delete the range first.
                for i in sorted(indices, reverse=True):
                    _layout_ops.remove_aot_entry(self, i)
                # Add new entries at the end then renormalise into
                # position.
                new_entries = []
                for v in values:
                    e = _layout_ops.add_aot_entry(self, v)
                    new_entries.append(e)
                # Now reorder so the new block lives at `start`.
                cur: list[Any] = list(self)
                cur = cur[: -len(new_entries)] if new_entries else cur
                for off, e in enumerate(new_entries):
                    cur.insert(start + off, e)
                if cur != list(self):
                    _layout_ops.renormalise_aot_order(self, cur)
                return
            # Extended slice with step != 1 and matching length:
            # replace each entry in place by remove + add at end + reorder.
            # Simpler: replace via the existing single-entry path.
            for i, v in zip(indices, values, strict=True):
                _layout_ops.replace_aot_entry(self, i, v)
            return
        if self._layout_root is None:
            assert isinstance(value, dict)
            list.__setitem__(self, index, _make_unattached_entry(value))
            return
        _layout_ops.replace_aot_entry(self, index, value)

    @override
    def append(self, value: Table | dict[str, Any]) -> None:
        # Same semantics as `add(body)` but with no return value (list API).
        from tomlrt import _layout_ops  # noqa: PLC0415

        if self._layout_root is None:
            assert isinstance(value, dict)
            list.append(self, _make_unattached_entry(value))
            return
        _layout_ops.add_aot_entry(self, value)

    @override
    def extend(self, values: Any) -> None:
        for v in values:
            self.append(v)

    @override
    def insert(self, index: SupportsIndex, value: Any) -> None:
        from tomlrt import _layout_ops  # noqa: PLC0415

        if self._layout_root is None:
            assert isinstance(value, dict)
            list.insert(self, index, _make_unattached_entry(value))
            return
        # Materialise as add() then renormalise into position.
        new_entry = _layout_ops.add_aot_entry(self, value)
        from tomlrt._container import Table  # noqa: PLC0415

        assert isinstance(new_entry, Table)
        # `add` appended; move into position via renormalise.
        idx = int(index)
        n = len(self)
        if idx < 0:
            idx = max(0, n + idx)
        idx = min(idx, n - 1)
        new_order: list[Table] = list(self)
        new_order.pop()  # remove from tail
        new_order.insert(idx, new_entry)
        if new_order != list(self):
            _layout_ops.renormalise_aot_order(self, new_order)

    @override
    def remove(self, value: Any) -> None:
        for i, t in enumerate(self):
            if t is value or t == value:
                del self[i]
                return
        msg = "list.remove(x): x not in list"
        raise ValueError(msg)

    @override
    def reverse(self) -> None:
        from tomlrt import _layout_ops  # noqa: PLC0415

        if self._layout_root is None:
            list.reverse(self)
            return
        new_order = list(reversed(self))
        _layout_ops.renormalise_aot_order(self, new_order)

    @override
    def sort(self, *args: Any, **kwargs: Any) -> None:
        from tomlrt import _layout_ops  # noqa: PLC0415

        new_order = sorted(self, *args, **kwargs)
        if self._layout_root is None:
            list.clear(self)
            for t in new_order:
                list.append(self, t)
            return
        _layout_ops.renormalise_aot_order(self, new_order)

    @override
    def __iadd__(self, other: Any) -> Self:  # type: ignore[override]
        self.extend(other)
        return self

    @override
    def __imul__(self, n: SupportsIndex) -> Self:
        from tomlrt import _layout_ops  # noqa: PLC0415

        count = int(n)
        if count <= 0:
            self.clear()
            return self
        if count == 1 or self._layout_root is None:
            return self
        # Snapshot original entries then add count-1 deepcopies.
        from copy import deepcopy  # noqa: PLC0415

        originals = list(self)
        for _ in range(count - 1):
            for e in originals:
                _layout_ops.add_aot_entry(self, deepcopy(dict(e)))
        return self


def _make_unattached_entry(body: object | None) -> Table:
    """Build a fresh unattached `Table` view as an AoT-entry placeholder."""
    from tomlrt._container import Table  # noqa: PLC0415

    t = Table()
    if body is not None:
        assert isinstance(body, dict)
        for k, v in body.items():
            dict.__setitem__(t, k, v)
    return t


__all__ = ["AoT", "Array"]
