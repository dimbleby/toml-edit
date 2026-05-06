"""Array views.

`Array(list)` backs an inline-array TOML value (`[1, 2, 3]`). `AoT`
(array-of-tables) is a `list[Table]` whose entries are individually
backed by `AoTEntry` records and whose elements are `Table` views.
"""

from __future__ import annotations

import operator
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, SupportsIndex, TypeVar, overload

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from copy import deepcopy

from tomlrt import _layout_ops
from tomlrt._array_comments import (
    ArrayEolView,
    ArrayLeadingView,
    apply_comments,
    clear_all_comments,
    snapshot_comments,
)
from tomlrt._errors import TOMLError
from tomlrt._trivia import (
    CommentNode,
    NewlineNode,
    Trivia,
    WhitespaceNode,
    clone_trivia,
    trivia_has_comment,
    trivia_has_newline,
)
from tomlrt._values import (
    ArrayItem,
    ArrayValue,
    InlineTableValue,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from _typeshed import SupportsRichComparison

    from tomlrt._values import (
        InlineTableEntry,
        Value,
    )

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    from tomlrt._container import Container, Document, Table, TomlInput


_T = TypeVar("_T")


class Array(list[Any]):
    """An inline TOML array."""

    __slots__ = ("_attached", "_multiline", "_value")

    def __init__(
        self,
        items: Iterable[TomlInput] = (),
        *,
        multiline: bool = False,
        indent: str = "    ",
    ) -> None:
        """Construct a standalone inline array.

        ``Array([1, 2, 3])`` builds an inline array;
        ``Array([1, 2, 3], multiline=True)`` lays it out one item per
        line with ``indent`` indentation. Such an array is *detached*
        until assigned into a document (``doc[k] = arr``).
        """
        super().__init__()

        self._value: ArrayValue = ArrayValue()
        self._multiline: bool = multiline
        self._attached: bool = False
        items_list = list(items)
        if not items_list:
            if multiline:
                self._value.final_trivia = Trivia(
                    [NewlineNode(text="\n"), WhitespaceNode(text=indent)]
                )
            return
        from tomlrt._container import _synth_inline_array  # noqa: PLC0415

        val, arr = _synth_inline_array(items_list, layout_root=None, owner=None)
        self._value = val
        for v in arr:
            list.append(self, v)
        if multiline and val.items:
            style = _ArrayStyle(
                is_multiline=True,
                inter_separator=Trivia(
                    [NewlineNode(text="\n"), WhitespaceNode(text=indent)]
                ),
                trailing_comma=True,
                trailing_post=Trivia([NewlineNode(text="\n")]),
            )
            for it in val.items:
                it.leading = Trivia()
                it.post_comma_trivia = Trivia()
            val.items[0].leading = Trivia(
                [NewlineNode(text="\n"), WhitespaceNode(text=indent)]
            )
            _renormalise_commas(val.items, style, val)
        elif multiline and not val.items:
            # Empty multiline factory: park the prospective leading
            # for the first append in final_trivia, matching how an
            # empty multiline array parses (`[\n    ]`).

            val.final_trivia = Trivia(
                [NewlineNode(text="\n"), WhitespaceNode(text=indent)]
            )

    def to_list(self) -> list[Any]:
        """Materialise a plain-Python ``list`` (recursive)."""
        from tomlrt._container import _to_python  # noqa: PLC0415

        return [_to_python(x) for x in self]

    def __copy__(self) -> Array:
        return Array(self.to_list(), multiline=self._multiline)

    def __deepcopy__(self, memo: dict[int, object]) -> Array:
        return Array(self.to_list(), multiline=self._multiline)

    def array(self, index: SupportsIndex) -> Array:
        """Return ``self[index]`` typed as a nested `Array`."""
        return self._typed_item(index, Array, "an Array")

    def table(self, index: SupportsIndex) -> Table:
        """Return ``self[index]`` typed as a `Table`."""
        from tomlrt._container import Table  # noqa: PLC0415

        return self._typed_item(index, Table, "a Table")

    @overload
    def get_array(self, index: SupportsIndex) -> Array | None: ...
    @overload
    def get_array(self, index: SupportsIndex, default: _T) -> Array | _T: ...
    def get_array(self, index: SupportsIndex, default: object = None) -> object:
        """Like `array(index)` but returns ``default`` for out-of-range."""
        i = operator.index(index)
        if i < -len(self) or i >= len(self):
            return default
        return self.array(index)

    @overload
    def get_table(self, index: SupportsIndex) -> Table | None: ...
    @overload
    def get_table(self, index: SupportsIndex, default: _T) -> Table | _T: ...
    def get_table(self, index: SupportsIndex, default: object = None) -> object:
        """Like `table(index)` but returns ``default`` for out-of-range."""
        i = operator.index(index)
        if i < -len(self) or i >= len(self):
            return default
        return self.table(index)

    def _typed_item(self, index: SupportsIndex, cls: type[_T], label: str) -> _T:
        v = self[index]
        if not isinstance(v, cls):
            msg = f"item at {index} is {type(v).__name__}, not {label}"
            raise TypeError(msg)
        return v

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

    def _style(self) -> _ArrayStyle:
        return _detect_style(self._value, multiline_flag=self._multiline)

    @property
    def multiline(self) -> bool:
        """True iff this array is rendered in multi-line form."""
        return self._style().is_multiline

    @multiline.setter
    def multiline(self, value: bool) -> None:
        self.set_multiline(multiline=value)

    @property
    def comments(self) -> ArrayEolView:
        """EOL comment view, indexed by item position."""
        return ArrayEolView(self)

    @property
    def leading_comments(self) -> ArrayLeadingView:
        """Leading-comment view, indexed by item position."""
        return ArrayLeadingView(self)

    def set_multiline(self, *, multiline: bool, indent: str = "    ") -> Array:
        """Switch this array between flush single-line and multi-line form.

        Raises ``TOMLError`` when collapsing a multi-line array that
        carries comments (per-item leading or post_comma trivia
        containing a comment), since those would have nowhere to live
        on a single line.

        Returns ``self`` for chaining.
        """
        ind = indent
        items = self._value.items
        if not multiline:
            # Recursively probe for CommentNodes anywhere inside an item
            # (its own trivia *and* nested inline values), since collapse
            # would have nowhere to put them.
            for it in items:
                if _item_has_any_comment(it):
                    msg = (
                        "cannot collapse multi-line array: "
                        "items contain EOL or leading comments"
                    )
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
            return self
        # multiline=True
        self._multiline = True
        if not items:
            self._value.final_trivia = Trivia(
                [NewlineNode(text="\n"), WhitespaceNode(text=ind)]
            )
            return self
        style = _ArrayStyle(
            is_multiline=True,
            inter_separator=Trivia([NewlineNode(text="\n"), WhitespaceNode(text=ind)]),
            trailing_comma=True,
            trailing_post=Trivia([NewlineNode(text="\n")]),
        )
        for it in items:
            it.leading = Trivia()
            it.post_comma_trivia = Trivia()
        items[0].leading = Trivia([NewlineNode(text="\n"), WhitespaceNode(text=ind)])
        self._value.final_trivia = Trivia()
        _renormalise_commas(items, style, self._value)
        return self

    def _synth_cst(self, value: object) -> tuple[Value, object]:
        from tomlrt._container import _synth_value  # noqa: PLC0415

        return _synth_value(
            value,
            layout_root=self._layout_root(),
            parent=None,
            path=(),
            owner=None,
        )

    @override
    def append(self, value: Any) -> None:
        if isinstance(value, AoT):
            msg = "Cannot store an array-of-tables inside an inline array"
            raise TOMLError(msg)
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
        adopted_leading: Trivia | None = None
        if not items and self._value.final_trivia.pieces:
            ft = self._value.final_trivia
            if trivia_has_newline(ft):
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
                adopted_leading = clone_trivia(ft)
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
    def extend(self, values: Iterable[Any]) -> None:
        for v in values:
            self.append(v)

    @override
    def clear(self) -> None:
        self._value.items.clear()
        # Drop any inter-item trivia clutter; preserve the bracket
        # leading captured in final_trivia.
        list.clear(self)

    @override
    def pop(self, index: SupportsIndex = -1) -> Any:
        n = len(self)
        i = int(index)
        if i < 0:
            i += n
        if i < 0 or i >= n:
            msg = "pop index out of range"
            raise IndexError(msg)
        items = self._value.items
        item = items[i]
        # Fast path: when the popped item carries no own-trivia
        # comments, no neighbouring item's comments depend on it
        # (a comment "above" the next item would be parked in
        # items[i].post_comma_trivia, which we just checked).
        # __delitem__ already handles bracket-pad and tail-pad
        # migration and skips the O(N) snapshot/clear/apply.
        if not (
            trivia_has_comment(item.leading)
            or trivia_has_comment(item.trailing)
            or trivia_has_comment(item.post_comma_trivia)
        ):
            decoded = self[i]
            del self[i]
            return decoded
        leadings, eols = snapshot_comments(self)
        decoded = list.pop(self, i)
        style = self._style()

        trailing_has_comment = any(
            isinstance(p, CommentNode) for p in items[i].trailing.pieces
        )
        if (
            i == len(items) - 1
            and items[i].trailing.pieces
            and not self._value.final_trivia.pieces
            and not trailing_has_comment
        ):
            self._value.final_trivia = items[i].trailing
        items.pop(i)
        leadings.pop(i)
        eols.pop(i)
        if items:
            _flip_to_terminal(items[-1], style)
            clear_all_comments(self)
            apply_comments(self, leadings, eols)
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
        cst, decoded = self._synth_cst(value)
        i = int(index)
        n = len(self)
        if i < 0:
            i = max(0, n + i)
        i = min(i, n)
        items = self._value.items
        style = self._style()
        leadings, eols = snapshot_comments(self)
        if i == 0 and items:
            # Inheriting position 0's leading: split it into the
            # "indent prefix" (initial NL+WS that puts the item on
            # its own line) and the "above-item block" (comments
            # plus their surrounding trivia). When the leading
            # opens with a comment (no initial newline) it's an
            # EOL-of-`[` style — the whole thing stays at position
            # 0 and apply_comments must not re-emit it onto the
            # displaced item.

            old_leading_pieces = list(items[0].leading.pieces)
            displaced = items[0]
            first_nonws = next(
                (
                    j
                    for j, p in enumerate(old_leading_pieces)
                    if not isinstance(p, (NewlineNode, WhitespaceNode))
                ),
                None,
            )
            opens_with_comment = (
                first_nonws is not None
                and isinstance(old_leading_pieces[first_nonws], CommentNode)
                and not any(
                    isinstance(p, NewlineNode) for p in old_leading_pieces[:first_nonws]
                )
            )
            if opens_with_comment:
                adopted = Trivia(pieces=old_leading_pieces)
                displaced.leading = Trivia()
                new_item = _new_item(
                    cst, leading_first=True, style=style, leading=adopted
                )
                if leadings:
                    leadings[0] = ()
            elif first_nonws is None:
                new_item = _new_item(
                    cst,
                    leading_first=True,
                    style=style,
                    leading=Trivia(pieces=old_leading_pieces),
                )
                displaced.leading = Trivia()
            else:
                # "Above-item" leading begins at the first comment;
                # the indent prefix goes to the new item.
                new_item = _new_item(
                    cst,
                    leading_first=True,
                    style=style,
                    leading=Trivia(pieces=old_leading_pieces[:first_nonws]),
                )
                displaced.leading = Trivia(pieces=old_leading_pieces[first_nonws:])
                if leadings:
                    leadings[0] = ()
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
        # Logical leadings/eols follow each value to its new index.
        leadings.insert(i, ())
        eols.insert(i, None)
        if any(eols) or any(leadings):
            clear_all_comments(self)
            apply_comments(self, leadings, eols)

    @override
    def reverse(self) -> None:
        self._reorder(list(reversed(range(len(self)))))

    @override
    @override
    def sort(
        self,
        *,
        key: Callable[[Any], object] | None = None,
        reverse: bool = False,
    ) -> None:
        n = len(self)
        if key is None:
            sort_key: Callable[[int], Any] = lambda i: self[i]  # noqa: E731
        else:
            key_fn = key
            sort_key = lambda i: key_fn(self[i])  # noqa: E731
        order = sorted(range(n), key=sort_key, reverse=reverse)
        self._reorder(order)

    def _reorder(self, order: list[int]) -> None:
        """Apply index permutation to items, decoded list, and per-item comments."""
        items = self._value.items
        # Bracket padding (leading of items[0], trailing of items[-1]) belongs
        # to *the array* — snapshot before reorder so it stays put.
        style = self._style()
        bracket_leading = clone_trivia(items[0].leading) if items else Trivia()
        leadings, eols = snapshot_comments(self)
        new_items = [items[j] for j in order]
        new_decoded = [self[j] for j in order]
        new_leadings = [leadings[j] for j in order]
        new_eols = [eols[j] for j in order]
        items[:] = new_items
        list.clear(self)
        for v in new_decoded:
            list.append(self, v)
        _normalise_for_renormalise(items, bracket_leading)
        _renormalise_commas(items, style, self._value)
        clear_all_comments(self)
        apply_comments(self, new_leadings, new_eols)

    @overload
    def __setitem__(self, index: SupportsIndex, value: Any) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[Any]) -> None: ...
    @override
    def __setitem__(
        self,
        index: SupportsIndex | slice,
        value: Any,
    ) -> None:
        if isinstance(index, slice):
            try:
                values = list(value)
            except TypeError as exc:
                msg = "can only assign an iterable"
                raise TypeError(msg) from exc
            # Compute target indices and replace items in place.
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
            new_csts = []
            new_decoded = []
            for v in values:
                cst, dec = self._synth_cst(v)
                new_csts.append(cst)
                new_decoded.append(dec)
            items = self._value.items
            style = self._style()
            # Build the new ArrayItem list segment.
            new_segment: list[ArrayItem] = []
            slice_start = index.start or 0
            for i, cst in enumerate(new_csts):
                first_in_arr = slice_start == 0 and i == 0 and not items[:slice_start]
                new_segment.append(
                    _new_item(cst, leading_first=first_in_arr, style=style)
                )
            items[index] = new_segment
            list.__setitem__(self, index, new_decoded)
            _renormalise_commas(items, style, self._value)
            return
        # int index: just replace the value CST in place.
        i = int(index)
        cst, dec = self._synth_cst(value)
        items = self._value.items
        if i < 0:
            i += len(items)
        items[i].value = cst
        list.__setitem__(self, index, dec)

    @override
    def __delitem__(self, index: SupportsIndex | slice) -> None:
        items = self._value.items
        # Snapshot bracket padding before mutation so a delete at
        # index 0 doesn't strip the leading-bracket padding (which
        # was owned by the original items[0]).
        had_leading = bool(items) and bool(items[0].leading.pieces)
        leading_first = clone_trivia(items[0].leading) if had_leading else Trivia()
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
        tail_pad = clone_trivia(items[last_idx].trailing) if had_tail_pad else Trivia()
        if isinstance(index, slice):
            removed_indices = range(*index.indices(len(items)))
            tail_was_removed = last_idx in removed_indices
        else:
            i = int(index)
            if i < 0:
                i += len(items)
            tail_was_removed = i == last_idx
        # When we'll re-stamp a new terminal after removing the old
        # one, snapshot the original terminal's has_comma and
        # post_comma_trivia directly. _flip_to_terminal only needs
        # `style.trailing_comma` and `style.trailing_post` — both
        # derive solely from items[-1] — so we can skip the O(N)
        # `_style()` scan that would otherwise dominate pop(-1).
        new_terminal_has_comma = False
        new_terminal_post: Trivia = Trivia()
        if tail_was_removed and len(items) >= 2:
            orig_terminal = items[last_idx]
            new_terminal_has_comma = orig_terminal.has_comma
            if orig_terminal.has_comma:
                is_multiline = (
                    self._multiline
                    or trivia_has_newline(orig_terminal.post_comma_trivia)
                    or trivia_has_newline(self._value.final_trivia)
                )
                new_terminal_post = _structural_trailing_post(
                    orig_terminal.post_comma_trivia, multiline=is_multiline
                )
        del items[index]
        list.__delitem__(self, index)
        if items:
            if had_leading and not items[0].leading.pieces:
                items[0].leading = leading_first
            if had_tail_pad and tail_was_removed:
                self._value.final_trivia = tail_pad
            if tail_was_removed:
                items[-1].has_comma = new_terminal_has_comma
                items[-1].post_comma_trivia = (
                    clone_trivia(new_terminal_post)
                    if new_terminal_has_comma
                    else Trivia()
                )
        elif had_tail_pad and tail_was_removed:
            self._value.final_trivia = tail_pad

    @override
    def __iadd__(self, values: Iterable[Any]) -> Self:
        self.extend(values)
        return self

    @override
    def __imul__(self, count: SupportsIndex) -> Self:
        n = int(count)
        if n <= 0:
            self.clear()
            return self
        if n == 1:
            return self
        # Snapshot original items + values, then append n-1 copies.
        original_items = list(self._value.items)
        original_values = list(self)
        for _ in range(n - 1):
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
        inter_separator: Trivia,
        trailing_comma: bool,
        trailing_post: Trivia,
    ) -> None:
        self.is_multiline = is_multiline
        self.inter_separator = inter_separator
        self.trailing_comma = trailing_comma
        self.trailing_post = trailing_post


def _detect_style(value: ArrayValue | None, *, multiline_flag: bool) -> _ArrayStyle:

    if value is None:
        return _ArrayStyle(
            is_multiline=multiline_flag,
            inter_separator=Trivia([WhitespaceNode(text=" ")]),
            trailing_comma=multiline_flag,
            trailing_post=Trivia(),
        )
    items = value.items
    has_newline = any(trivia_has_newline(it.post_comma_trivia) for it in items) or (
        trivia_has_newline(value.final_trivia)
    )
    is_multiline = has_newline or multiline_flag
    # Inter-item separator: first internal item's post_comma_trivia.
    inter_sep = None
    for it in items:
        if it.has_comma and (it is not items[-1] or len(items) == 1):
            inter_sep = clone_trivia(it.post_comma_trivia)
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
                    trailing_post = clone_trivia(tp)
            else:
                trailing_post = clone_trivia(tp)
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


def _has_ws_after_last_newline(trivia: Trivia) -> bool:
    pieces = trivia.pieces
    last_nl = -1
    for i, p in enumerate(pieces):
        if isinstance(p, NewlineNode):
            last_nl = i
    if last_nl < 0:
        return False
    return last_nl + 1 < len(pieces) and isinstance(pieces[last_nl + 1], WhitespaceNode)


def _first_indent_after_newline(trivia: Trivia) -> str:
    pieces = trivia.pieces
    for i, p in enumerate(pieces):
        if (
            isinstance(p, NewlineNode)
            and i + 1 < len(pieces)
            and isinstance(pieces[i + 1], WhitespaceNode)
        ):
            return str(pieces[i + 1].text)
    return ""


def _indent_from_final_trivia(ft: Trivia) -> str:
    """Extract a logical indent from `final_trivia` pieces.

    Prefers the indent of the last comment line (so a varied-indent
    or blank-line-prefixed comment block aligns the new item with
    the *most recent* commented line). Falls back to the indent of
    the last whitespace-after-newline block, then to "".
    """
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


def _internal_leading(_style: _ArrayStyle) -> Trivia:
    # Internal items get blank leading; separator lives in the prior
    # item's post_comma_trivia.
    return Trivia()


def _new_item(
    cst: Value,
    *,
    leading_first: bool,
    style: _ArrayStyle,
    leading: Trivia | None = None,
) -> ArrayItem:
    if leading is None:
        leading = Trivia() if leading_first else _internal_leading(style)
    return ArrayItem(
        leading=leading,
        value=cst,
        trailing=Trivia(),
        has_comma=False,
        post_comma_trivia=Trivia(),
    )


def _flip_to_internal(
    item: ArrayItem, style: _ArrayStyle, value: ArrayValue | None = None
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
    has_comment = any(isinstance(p, CommentNode) for p in item.trailing.pieces)
    if (
        value is not None
        and item.trailing.pieces
        and not value.final_trivia.pieces
        and not has_comment
    ):
        value.final_trivia = item.trailing
    if not has_comment:
        item.trailing = Trivia()
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
    item.post_comma_trivia = clone_trivia(style.inter_separator)


def _flip_to_terminal(item: ArrayItem, style: _ArrayStyle) -> None:
    """Make ``item`` look like the terminal (last) item per style."""
    item.has_comma = style.trailing_comma
    item.post_comma_trivia = (
        clone_trivia(style.trailing_post) if style.trailing_comma else Trivia()
    )


def _structural_trailing_post(tp: Trivia, *, multiline: bool) -> Trivia:
    """Extract the structural trailing-post from a comma-terminated item.

    Mirrors `_detect_style`'s trailing_post computation but operates on
    a single ``post_comma_trivia`` so callers don't have to scan the
    whole array. In a multiline layout, drops any pre-newline comment
    block (which conceptually belonged to the popped item's row) and
    keeps the structural newline + bracket-pad indent.
    """
    if not multiline:
        return clone_trivia(tp)
    pieces_list = list(tp.pieces)
    last_nl = -1
    for j, p in enumerate(pieces_list):
        if isinstance(p, NewlineNode):
            last_nl = j
    if last_nl >= 0:
        return Trivia(list(pieces_list[last_nl:]))
    return clone_trivia(tp)


def _value_has_any_comment(val: Value) -> bool:
    if isinstance(val, ArrayValue):
        if trivia_has_comment(val.final_trivia):
            return True
        return any(_item_has_any_comment(it) for it in val.items)
    if isinstance(val, InlineTableValue):
        if trivia_has_comment(val.final_trivia):
            return True
        return any(_entry_has_any_comment(e) for e in val.entries)
    return False


def _item_has_any_comment(item: ArrayItem) -> bool:
    if (
        trivia_has_comment(item.leading)
        or trivia_has_comment(item.trailing)
        or trivia_has_comment(item.post_comma_trivia)
    ):
        return True
    return _value_has_any_comment(item.value)


def _entry_has_any_comment(entry: InlineTableEntry) -> bool:
    if trivia_has_comment(entry.leading) or trivia_has_comment(entry.trailing):
        return True
    if trivia_has_comment(entry.post_comma_trivia):
        return True
    return _value_has_any_comment(entry.value)


def _renormalise_commas(
    items: list[ArrayItem], style: _ArrayStyle, value: ArrayValue | None = None
) -> None:
    """Reset has_comma + post_comma_trivia across ``items`` per style."""
    if not items:
        return
    for it in items[:-1]:
        _flip_to_internal(it, style, value)
    _flip_to_terminal(items[-1], style)


def _normalise_for_renormalise(items: list[ArrayItem], bracket_leading: Trivia) -> None:
    """Prepare ``items`` for `_renormalise_commas` after a reorder.

    Strips per-item ``leading`` (so the bracket-leading isn't double-
    counted on whichever item now sits at index 0) and re-applies the
    captured ``bracket_leading`` to the new ``items[0]``.
    """
    if not items:
        return
    for it in items:
        it.leading = Trivia()
    items[0].leading = bracket_leading


def _clone_item(item: ArrayItem) -> ArrayItem:
    return deepcopy(item)


class AoT(list["Table"]):
    """An Array-of-tables, e.g. ``[[products]]`` repeated."""

    __slots__ = ("_layout_root", "_parent", "_path")

    def __init__(self, entries: Iterable[Mapping[str, TomlInput]] = ()) -> None:
        """Construct a standalone array-of-tables."""
        super().__init__()
        self._layout_root: Document | None = None
        self._path: tuple[str, ...] = ()
        self._parent: Container | None = None
        for e in entries:
            list.append(self, _make_unattached_entry(e))

    def to_list(self) -> list[dict[str, Any]]:
        """Materialise a list of plain-Python ``dict``s (recursive)."""
        return [t.to_dict() for t in self]

    def __copy__(self) -> AoT:
        return AoT(self.to_list())

    def __deepcopy__(self, memo: dict[int, object]) -> AoT:
        return AoT(self.to_list())

    def add(self, entry: Mapping[str, TomlInput] | None = None) -> Table:
        """Append a fresh ``[[path]]`` entry and return its `Table` view.

        ``entry`` may be a Mapping (initial body content) or ``None``
        (empty entry). The AoT must be attached to a document.
        """
        if self._layout_root is None:
            list.append(self, _make_unattached_entry(entry))
            return self[-1]
        return _layout_ops.add_aot_entry(self, entry)

    def _add_entry_attached(self, value: Mapping[str, Any]) -> Table:
        """Dispatch a new attached AoT entry from ``value``.

        Pre: ``self._layout_root is not None``. Selects the trivia-
        preserving clone path when ``value`` is itself an attached
        AoT entry or attached standard section, otherwise falls
        through to ``add_aot_entry``.
        """
        from tomlrt._container import Table as TableType  # noqa: PLC0415

        if isinstance(value, TableType) and value._layout_root is not None:  # noqa: SLF001
            if value._owner_aot_entry is not None:  # noqa: SLF001
                return _layout_ops.clone_aot_entry(self, value)
            if value._header_ref is not None and not value._inline:  # noqa: SLF001
                return _layout_ops.clone_table_as_aot_entry(self, value)
        return _layout_ops.add_aot_entry(self, value)

    def _replace_entry_attached(
        self, index: int, value: Mapping[str, Any] | None
    ) -> None:
        """Dispatch in-place replacement of an attached AoT entry."""
        from tomlrt._container import Table as TableType  # noqa: PLC0415

        if (
            isinstance(value, TableType)
            and value._layout_root is not None  # noqa: SLF001
            and value._owner_aot_entry is not None  # noqa: SLF001
        ):
            _layout_ops.replace_aot_entry_with_clone(self, index, value)
            return
        _layout_ops.replace_aot_entry(self, index, value)

    # ------------------------------------------------------------------
    # Supported list-mutator surface.
    # Anything not implemented here is overridden below to fail closed
    # rather than corrupt the doc-stream via inherited `list` behaviour.
    # ------------------------------------------------------------------

    @override
    def pop(self, index: SupportsIndex = -1) -> Table:
        idx = int(index)
        n = len(self)
        if not -n <= idx < n:
            msg = "pop index out of range"
            raise IndexError(msg)
        if self._layout_root is None:
            return list.pop(self, idx)
        return _layout_ops.remove_aot_entry(self, idx)

    @override
    def __delitem__(self, index: SupportsIndex | slice) -> None:
        if isinstance(index, slice):
            if self._layout_root is None:
                list.__delitem__(self, index)
                return
            indices = sorted(range(*index.indices(len(self))), reverse=True)
            for i in indices:
                _layout_ops.remove_aot_entry(self, i)
            return
        if self._layout_root is None:
            list.__delitem__(self, index)
            return
        _layout_ops.remove_aot_entry(self, int(index))

    @override
    def clear(self) -> None:
        if self._layout_root is None:
            list.clear(self)
            return
        while len(self) > 0:
            _layout_ops.remove_aot_entry(self, -1)

    @overload
    def __setitem__(
        self, index: SupportsIndex, value: Mapping[str, TomlInput]
    ) -> None: ...
    @overload
    def __setitem__(
        self, index: slice, value: Iterable[Mapping[str, TomlInput]]
    ) -> None: ...
    @override
    def __setitem__(
        self,
        index: SupportsIndex | slice,
        value: Mapping[str, TomlInput] | Iterable[Mapping[str, TomlInput]],
    ) -> None:
        if isinstance(index, slice):
            try:
                values = list(value)
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
            typed_values: list[Mapping[str, Any]] = []
            for v in values:
                if not isinstance(v, Mapping):
                    msg = f"AoT entries must be Mapping/Table; got {type(v).__name__}"
                    raise TypeError(msg)
                typed_values.append(v)  # ty: ignore[invalid-argument-type]
            if self._layout_root is None:
                list.__setitem__(
                    self, index, [_make_unattached_entry(v) for v in typed_values]
                )
                return
            # For contiguous step == 1: replace by delete-range then
            # insert at the start index. Order matters: build new
            # entries via the dispatcher (appended to end), then
            # renormalise.
            if index.step is None or index.step == 1:
                start = index.indices(len(self))[0]
                for i in sorted(indices, reverse=True):
                    _layout_ops.remove_aot_entry(self, i)
                new_entries = [self._add_entry_attached(v) for v in typed_values]
                cur: list[Table] = list(self)
                cur = cur[: -len(new_entries)] if new_entries else cur
                for off, e in enumerate(new_entries):
                    cur.insert(start + off, e)
                if cur != list(self):
                    _layout_ops.renormalise_aot_order(self, cur)
                return
            # Extended slice with step != 1 and matching length:
            # replace each entry in place.
            for i, v in zip(indices, typed_values, strict=True):
                self._replace_entry_attached(i, v)
            return
        assert isinstance(value, Mapping)
        if self._layout_root is None:
            assert isinstance(value, dict)
            list.__setitem__(self, index, _make_unattached_entry(value))  # ty: ignore[invalid-argument-type]
            return
        self._replace_entry_attached(operator.index(index), value)  # ty: ignore[invalid-argument-type]

    @override
    def append(self, value: Table | Mapping[str, TomlInput]) -> None:
        # Same semantics as `add(body)` but with no return value (list API).
        if self._layout_root is None:
            assert isinstance(value, dict)
            list.append(self, _make_unattached_entry(value))
            return
        self._add_entry_attached(value)

    @override
    def extend(self, values: Iterable[Table | Mapping[str, TomlInput]]) -> None:
        for v in values:
            self.append(v)

    @override
    def insert(
        self, index: SupportsIndex, value: Table | Mapping[str, TomlInput]
    ) -> None:
        if self._layout_root is None:
            assert isinstance(value, dict)
            list.insert(self, index, _make_unattached_entry(value))
            return
        new_entry = self._add_entry_attached(value)
        idx = int(index)
        n = len(self)
        if idx < 0:
            idx = max(0, n + idx)
        idx = min(idx, n - 1)
        new_order: list[Table] = list(self)
        new_order.pop()
        new_order.insert(idx, new_entry)
        if new_order != list(self):
            _layout_ops.renormalise_aot_order(self, new_order)

    @override
    def remove(self, value: Mapping[str, TomlInput]) -> None:
        for i, t in enumerate(self):
            if t is value or t == value:
                del self[i]
                return
        msg = "list.remove(x): x not in list"
        raise ValueError(msg)

    @override
    def reverse(self) -> None:
        if self._layout_root is None:
            list.reverse(self)
            return
        new_order = list(reversed(self))
        _layout_ops.renormalise_aot_order(self, new_order)

    @override
    def sort(  # type: ignore[override]
        self,
        *,
        key: Callable[[Table], SupportsRichComparison],
        reverse: bool = False,
    ) -> None:
        new_order = sorted(self, key=key, reverse=reverse)
        if self._layout_root is None:
            list.clear(self)
            for t in new_order:
                list.append(self, t)
            return
        _layout_ops.renormalise_aot_order(self, new_order)

    @override
    def __iadd__(self, values: Iterable[Mapping[str, TomlInput]]) -> Self:  # type: ignore[override]
        self.extend(values)
        return self

    @override
    def __imul__(self, count: SupportsIndex) -> Self:
        n = int(count)
        if n <= 0:
            self.clear()
            return self
        if n == 1 or self._layout_root is None:
            return self
        # Preflight: probe every entry's clone-eligibility BEFORE we
        # start mutating the document, so a failure on entry N does
        # not leave entries 0..N-1 already cloned.
        originals = list(self)
        for e in originals:
            _layout_ops.check_clone_aot_entry(self, e)
        for _ in range(n - 1):
            for e in originals:
                _layout_ops.clone_aot_entry(self, e)
        return self


def _make_unattached_entry(body: Mapping[str, TomlInput] | None) -> Table:
    """Build a fresh unattached `Table` view as an AoT-entry placeholder."""
    from tomlrt._container import Table  # noqa: PLC0415

    t = Table()
    if body is not None:
        for k, v in body.items():
            dict.__setitem__(t, k, v)
    return t


__all__ = ["AoT", "Array"]
