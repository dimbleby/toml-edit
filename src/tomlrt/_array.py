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

    __slots__ = ("_attached", "_multiline", "_value")

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
        # Factory mode: True once the user-facing constructor finishes.
        # Decoder paths construct an Array() then immediately overwrite
        # `_value` and clear `_attached` to indicate "this view is owned
        # by the doc, not a free orphan".
        self._attached: bool = False
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
                it.post_comma_trivia = Trivia()
            val.items[0].leading = Trivia(
                [NewlineNode(text="\n"), WhitespaceNode(text="    ")]
            )
            _renormalise_commas(val.items, style, val)

    def to_list(self) -> list[Any]:
        """Materialise a plain-Python ``list`` (recursive)."""
        from tomlrt._container import _to_python  # noqa: PLC0415

        return [_to_python(x) for x in self]

    def __copy__(self) -> Array:
        return Array(self.to_list(), multiline=self._multiline)

    def __deepcopy__(self, memo: dict[int, Any]) -> Array:
        return Array(self.to_list(), multiline=self._multiline)

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

    @property
    def multiline(self) -> bool:
        """True iff this array is rendered in multi-line form."""
        if self._value is None:
            return self._multiline
        return self._style().is_multiline

    def set_multiline(self, *, multiline: bool) -> None:
        """Switch this array between flush single-line and multi-line form.

        Raises ``TOMLError`` when collapsing a multi-line array that
        carries comments (per-item leading or post_comma trivia
        containing a comment), since those would have nowhere to live
        on a single line.
        """
        from tomlrt._errors import TOMLError  # noqa: PLC0415
        from tomlrt._trivia import (  # noqa: PLC0415
            CommentNode,
            NewlineNode,
            Trivia,
            WhitespaceNode,
        )

        if self._value is None:
            self._multiline = multiline
            return
        items = self._value.items
        if not multiline:
            for it in items:
                for piece in (*it.leading.pieces, *it.post_comma_trivia.pieces):
                    if isinstance(piece, CommentNode):
                        msg = "cannot collapse multi-line array with comments"
                        raise TOMLError(msg)
            for it in items:
                it.leading = Trivia()
                it.post_comma_trivia = Trivia()
            self._value.final_trivia = Trivia()
            self._multiline = False
            flush_style = _ArrayStyle(
                is_multiline=False,
                inter_separator=Trivia([WhitespaceNode(text=" ")]),
                trailing_comma=False,
                trailing_post=Trivia(),
            )
            _renormalise_commas(items, flush_style, self._value)
            return
        # multiline=True
        self._multiline = True
        if not items:
            self._value.final_trivia = Trivia(
                [NewlineNode(text="\n"), WhitespaceNode(text="    ")]
            )
            return
        style = _ArrayStyle(
            is_multiline=True,
            inter_separator=Trivia(
                [NewlineNode(text="\n"), WhitespaceNode(text="    ")]
            ),
            trailing_comma=True,
            trailing_post=Trivia([NewlineNode(text="\n")]),
        )
        for it in items:
            it.leading = Trivia()
        items[0].leading = Trivia([NewlineNode(text="\n"), WhitespaceNode(text="    ")])
        self._value.final_trivia = Trivia()
        _renormalise_commas(items, style, self._value)

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
        # If appending into an empty array, any inner-bracket padding
        # held in final_trivia is the prospective leading for the new
        # item.
        # - For pure whitespace (e.g. `[ ]`), adopt without clearing
        #   so both `[ ` and ` ]` survive: → `[ 1 ]`.
        # - For multiline structural content (e.g. `[\n\t# hi\n]`),
        #   adopt + clear, and stitch on a synthesised indent so the
        #   new item sits below the comment at the indent level the
        #   user established for it.
        adopted_leading: Any | None = None
        if not items and self._value.final_trivia.pieces:
            ft = self._value.final_trivia
            if _trivia_has_newline(ft):
                from tomlrt._trivia import (  # noqa: PLC0415
                    NewlineNode,
                    Trivia,
                    WhitespaceNode,
                )

                indent = _indent_from_final_trivia(ft)
                pieces = list(ft.pieces)
                # If the final_trivia is just a bare newline (no
                # interior content), use the array style's inferred
                # indent (e.g. `    `) so an appended item lands at
                # the conventional column.
                if (
                    not indent
                    and len(pieces) == 1
                    and isinstance(pieces[0], NewlineNode)
                ):
                    indent = _first_indent_after_newline(style.inter_separator)
                if indent and pieces and isinstance(pieces[-1], NewlineNode):
                    pieces.append(WhitespaceNode(text=indent))
                adopted_leading = Trivia(pieces)
                self._value.final_trivia = Trivia()
            else:
                adopted_leading = _clone_trivia(ft)
        new_item = _new_item(
            cst,
            leading_first=not items,
            style=style,
            leading=adopted_leading,
        )
        if items:
            _flip_to_internal(items[-1], style, self._value)
        items.append(new_item)
        # The new tail inherits the array's terminal style (trailing
        # comma + post-trivia), e.g. `\n    4,\n` for the multiline
        # trailing-comma layout.
        _flip_to_terminal(new_item, style)
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
        # Snapshot style before mutation so the trailing-comma policy
        # reflects the original last item, not the newly-exposed one.
        style = self._style()
        # If popping the terminal item and it carried bracket-padding
        # in its `trailing`, migrate that into final_trivia so the new
        # tail still renders as `... ]`.
        if (
            i == len(items) - 1
            and items[i].trailing.pieces
            and not self._value.final_trivia.pieces
        ):
            self._value.final_trivia = items[i].trailing
        items.pop(i)
        if items:
            _flip_to_terminal(items[-1], style)
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
            # comment block + indent). The new item's post_comma_trivia
            # (set by _flip_to_internal below) carries the inter-item
            # separator including its indent, so the displaced item
            # must NOT keep its own leading indent — that would double
            # the column position.
            adopted_leading = items[0].leading
            displaced = items[0]
            displaced.leading = _trivia_empty()
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
                _flip_to_internal(items[-2], style, self._value)
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
        # Snapshot bracket padding + style before reorder; the leading
        # of items[0] and the trailing of items[-1] are bracket-padding
        # that belong to *the array*, not to those particular items.
        style = self._style()
        bracket_leading = _clone_trivia(items[0].leading) if items else _trivia_empty()
        items.reverse()
        list.reverse(self)
        _normalise_for_renormalise(items, bracket_leading)
        _renormalise_commas(items, style, self._value)

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
        # See `reverse` — capture style + bracket padding before reorder.
        style = self._style()
        bracket_leading = _clone_trivia(items[0].leading) if items else _trivia_empty()
        new_items = [items[j] for j in order]
        new_decoded = [self[j] for j in order]
        items[:] = new_items
        list.clear(self)
        for v in new_decoded:
            list.append(self, v)
        _normalise_for_renormalise(items, bracket_leading)
        _renormalise_commas(items, style, self._value)

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
            _renormalise_commas(items, style, self._value)
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
        # Snapshot bracket padding + style before mutation so a delete
        # at index 0 doesn't strip the leading-bracket padding (which
        # was owned by the original items[0]) and so trailing-comma
        # policy reflects the original last item.
        style = self._style()
        had_leading = bool(items) and bool(items[0].leading.pieces)
        leading_first = (
            _clone_trivia(items[0].leading) if had_leading else _trivia_empty()
        )
        # Snapshot terminal item's trailing for tail-pad migration: if
        # the delete includes the original last item and final_trivia
        # is empty, migrate that trailing into final_trivia so the new
        # tail still renders with bracket padding.
        last_idx = len(items) - 1
        had_tail_pad = (
            bool(items)
            and bool(items[last_idx].trailing.pieces)
            and not self._value.final_trivia.pieces
        )
        tail_pad = (
            _clone_trivia(items[last_idx].trailing) if had_tail_pad else _trivia_empty()
        )
        if isinstance(key, slice):
            removed_indices = range(*key.indices(len(items)))
            tail_was_removed = last_idx in removed_indices
        else:
            i = int(key)
            if i < 0:
                i += len(items)
            tail_was_removed = i == last_idx
        del items[key]
        list.__delitem__(self, key)
        if items:
            if had_leading and not items[0].leading.pieces:
                items[0].leading = leading_first
            if had_tail_pad and tail_was_removed:
                self._value.final_trivia = tail_pad
            _flip_to_terminal(items[-1], style)
        elif had_tail_pad and tail_was_removed:
            self._value.final_trivia = tail_pad

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
                    _flip_to_internal(items[-1], style, self._value)
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
            # Sample newline style from any newline already in the
            # array body so CRLF stays CRLF.
            nl_text = "\n"
            if value is not None:
                for p in value.final_trivia.pieces:
                    if isinstance(p, NewlineNode):
                        nl_text = p.text
                        break
                else:
                    for it in items:
                        for p in it.post_comma_trivia.pieces:
                            if isinstance(p, NewlineNode):
                                nl_text = p.text
                                break
                        if nl_text != "\n":
                            break
            inter_sep = Trivia([NewlineNode(text=nl_text), WhitespaceNode(text="    ")])
        else:
            inter_sep = Trivia([WhitespaceNode(text=" ")])
    elif is_multiline and len(items) == 1 and not _has_ws_after_last_newline(inter_sep):
        # Single-item multiline arrays: post_comma_trivia of the
        # only item may carry a trailing comment block but lack a
        # trailing indent (the next byte is `]`). Stitch on the
        # indent from items[0].leading after the LAST newline so an
        # appended item lands at the right column without disturbing
        # any interior comment.
        indent = _first_indent_after_newline(items[0].leading)
        if indent:
            pieces_list = list(inter_sep.pieces)
            last_nl = -1
            for j, p in enumerate(pieces_list):
                if isinstance(p, NewlineNode):
                    last_nl = j
            if last_nl >= 0:
                pieces_list.insert(last_nl + 1, WhitespaceNode(text=indent))
                inter_sep = Trivia(pieces_list)
    # Trailing-comma policy: the last item's has_comma if any.
    if items:
        trailing_comma = items[-1].has_comma
        if items[-1].has_comma:
            tp = items[-1].post_comma_trivia
            if is_multiline:
                # In a multiline layout, post_comma_trivia of the last
                # item may carry a trailing comment block (e.g. a
                # `# tail` line that "belonged" to the previous tail
                # position). The structural trailing_post is just the
                # newline (and any bracket-pad indent) immediately
                # before `]`. Take everything from the LAST newline
                # forward.
                pieces_list = list(tp.pieces)
                last_nl = -1
                for j, p in enumerate(pieces_list):
                    if isinstance(p, NewlineNode):
                        last_nl = j
                if last_nl >= 0:
                    trailing_post = Trivia(list(pieces_list[last_nl:]))
                else:
                    trailing_post = _clone_trivia(tp)
            else:
                trailing_post = _clone_trivia(tp)
        else:
            trailing_post = Trivia()
    else:
        trailing_comma = is_multiline
        # Sample newline style from the empty array's final_trivia
        # so CRLF documents stay CRLF after append. Falls back to LF.
        nl_text = "\n"
        if value is not None:
            for p in value.final_trivia.pieces:
                if isinstance(p, NewlineNode):
                    nl_text = p.text
                    break
        trailing_post = (
            Trivia([NewlineNode(text=nl_text)]) if is_multiline else Trivia()
        )
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


def _has_ws_after_last_newline(trivia: Any) -> bool:
    from tomlrt._trivia import NewlineNode, WhitespaceNode  # noqa: PLC0415

    pieces = trivia.pieces
    last_nl = -1
    for i, p in enumerate(pieces):
        if isinstance(p, NewlineNode):
            last_nl = i
    if last_nl < 0:
        return False
    return last_nl + 1 < len(pieces) and isinstance(pieces[last_nl + 1], WhitespaceNode)


def _has_ws_after_newline(trivia: Any) -> bool:
    from tomlrt._trivia import NewlineNode, WhitespaceNode  # noqa: PLC0415

    pieces = trivia.pieces
    for i, p in enumerate(pieces):
        if (
            isinstance(p, NewlineNode)
            and i + 1 < len(pieces)
            and isinstance(pieces[i + 1], WhitespaceNode)
        ):
            return True
    return False


def _first_indent_after_newline(trivia: Any) -> str:
    from tomlrt._trivia import NewlineNode, WhitespaceNode  # noqa: PLC0415

    pieces = trivia.pieces
    for i, p in enumerate(pieces):
        if (
            isinstance(p, NewlineNode)
            and i + 1 < len(pieces)
            and isinstance(pieces[i + 1], WhitespaceNode)
        ):
            return str(pieces[i + 1].text)
    return ""


def _indent_from_final_trivia(ft: Any) -> str:
    """Extract a logical indent from `final_trivia` pieces.

    Prefers the indent of the last comment line (so a varied-indent
    or blank-line-prefixed comment block aligns the new item with
    the *most recent* commented line). Falls back to the indent of
    the last whitespace-after-newline block, then to "".
    """
    from tomlrt._trivia import (  # noqa: PLC0415
        CommentNode,
        NewlineNode,
        WhitespaceNode,
    )

    pieces = ft.pieces
    last_comment_indent: str | None = None
    last_ws_after_nl: str | None = None
    j = 0
    while j < len(pieces):
        if isinstance(pieces[j], NewlineNode) and j + 1 < len(pieces):
            ws = ""
            if isinstance(pieces[j + 1], WhitespaceNode):
                ws = str(pieces[j + 1].text)
                last_ws_after_nl = ws
                k = j + 2
            else:
                k = j + 1
            if k < len(pieces) and isinstance(pieces[k], CommentNode):
                last_comment_indent = ws
        j += 1
    if last_comment_indent is not None:
        return last_comment_indent
    return last_ws_after_nl or ""


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


def _flip_to_internal(
    item: Any, style: _ArrayStyle, value: ArrayValue | None = None
) -> None:
    """Make ``item`` look like an internal (non-last) item.

    When ``value`` is supplied, any whitespace in ``item.trailing`` is
    treated as bracket-padding and migrated into ``value.final_trivia``
    (if that slot is empty) before being cleared from the item — so
    that the original ``[ ..., x ]`` style survives an item being
    demoted from terminal to internal.

    If the item already has a comma + non-empty ``post_comma_trivia``
    (e.g. an EOL comment block), preserve it verbatim — it belongs to
    the item — and only ensure it ends with a structural newline +
    indent so the next item lands at the right column. Otherwise fall
    back to the style's full inter-separator template.
    """
    from tomlrt._trivia import NewlineNode, Trivia, WhitespaceNode  # noqa: PLC0415

    if value is not None and item.trailing.pieces and not value.final_trivia.pieces:
        value.final_trivia = item.trailing
    item.trailing = _trivia_empty()
    if item.has_comma and item.post_comma_trivia.pieces:
        if not _has_ws_after_last_newline(item.post_comma_trivia):
            indent = _first_indent_after_newline(style.inter_separator)
            pieces = list(item.post_comma_trivia.pieces)
            last_nl = -1
            for j, p in enumerate(pieces):
                if isinstance(p, NewlineNode):
                    last_nl = j
            if last_nl >= 0 and indent:
                pieces.insert(last_nl + 1, WhitespaceNode(text=indent))
                item.post_comma_trivia = Trivia(pieces)
            elif last_nl < 0 and indent:
                nl_text = "\n"
                for p in style.inter_separator.pieces:
                    if isinstance(p, NewlineNode):
                        nl_text = p.text
                        break
                pieces.append(NewlineNode(text=nl_text))
                pieces.append(WhitespaceNode(text=indent))
                item.post_comma_trivia = Trivia(pieces)
        return
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


def _renormalise_commas(
    items: list[Any], style: _ArrayStyle, value: ArrayValue | None = None
) -> None:
    """Reset has_comma + post_comma_trivia across ``items`` per style."""
    if not items:
        return
    for it in items[:-1]:
        _flip_to_internal(it, style, value)
    _flip_to_terminal(items[-1], style)


def _normalise_for_renormalise(items: list[Any], bracket_leading: Any) -> None:
    """Prepare ``items`` for `_renormalise_commas` after a reorder.

    Strips per-item ``leading`` (so the bracket-leading isn't double-
    counted on whichever item now sits at index 0) and re-applies the
    captured ``bracket_leading`` to the new ``items[0]``.
    """
    from tomlrt._trivia import Trivia  # noqa: PLC0415

    if not items:
        return
    for it in items:
        it.leading = Trivia()
    items[0].leading = bracket_leading


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

    def __copy__(self) -> AoT:
        return AoT(self.to_list())

    def __deepcopy__(self, memo: dict[int, Any]) -> AoT:
        return AoT(self.to_list())

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
        # Preflight: probe every entry's clone-eligibility BEFORE we
        # start mutating the document, so a failure on entry N does
        # not leave entries 0..N-1 already cloned.
        originals = list(self)
        for e in originals:
            _layout_ops.check_clone_aot_entry(self, e)
        for _ in range(count - 1):
            for e in originals:
                _layout_ops.clone_aot_entry(self, e)
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
