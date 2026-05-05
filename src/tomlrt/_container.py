"""Logical container layer.

`Container(dict)` is the dict-typed base for both `Document` (the
root) and `Table` (sections + inline tables). Reads come straight
from dict storage populated in doc-stream first-occurrence order;
mutations write to the slot stream via the per-container caches
(`_index`, `_refs`, `_header_ref`, `_body_tail`) and refresh the
dict from there.
"""

from __future__ import annotations

import math
import re
import sys
from collections.abc import Mapping
from datetime import date, datetime, time
from typing import TYPE_CHECKING, Any, TypeGuard, TypeVar

if sys.version_info >= (3, 12):
    from typing import Self, override
else:
    from typing_extensions import override

from tomlrt import _inline_ops, _layout_ops
from tomlrt._comments import (
    EolCommentView,
    LeadingCommentView,
    _direct_kv_slot,
    _doc_epilogue_get,
    _doc_epilogue_set,
    _doc_preamble_get,
    _doc_preamble_set,
    _header_comment_get,
    _header_comment_set,
    _header_leading_get,
    _header_leading_set,
)
from tomlrt._errors import TOMLError
from tomlrt._render import render
from tomlrt._slots import KVSlot, StructuralHeaderSlot
from tomlrt._trivia import (
    NewlineNode,
    Trivia,
    WhitespaceNode,
    trivia_has_comment,
)
from tomlrt._values import (
    ArrayItem,
    ArrayValue,
    BoolValue,
    DateTimeValue,
    FloatValue,
    InlineTableEntry,
    InlineTableValue,
    IntegerValue,
    KeyPart,
    StringValue,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from typing_extensions import Self

    from tomlrt._slots import AoTEntry, Slot, SlotRef
    from tomlrt._trivia import TriviaPiece
    from tomlrt._values import (
        DateLikeKind,
        Value,
    )


_T = TypeVar("_T")


class Container(dict[str, Any]):
    """Dict-typed base for `Document` and `Table` views.

    Reads are pure dict operations. Mutation paths use the per-container
    cache (`_index` / `_refs` / `_header_ref` / `_body_tail`)
    maintained alongside the dict storage. ``_subtree_tail`` is exposed
    as a derived `@property` over `_refs`. For inline tables
    (`_inline=True`) the slot-stream caches stay empty and `_value`
    points at the backing `InlineTableValue` instead — inline mutation
    lives in a separate code path.
    """

    __slots__ = (
        "_body_tail",
        "_header_ref",
        "_index",
        "_inline",
        "_layout_root",
        "_owner_aot_entry",
        "_parent",
        "_path",
        "_refs",
        "_value",
    )

    def __init__(self) -> None:
        super().__init__()
        self._layout_root: Document | None = None
        self._path: tuple[str, ...] = ()
        self._inline: bool = False
        self._parent: Container | None = None
        self._owner_aot_entry: AoTEntry | None = None
        self._index: dict[str, list[SlotRef]] = {}
        self._refs: list[SlotRef] = []
        self._header_ref: SlotRef | None = None
        self._body_tail: Slot | None = None
        self._value: InlineTableValue | None = None

    @property
    def comments(self) -> EolCommentView:
        """Mapping view of EOL comments on this container's direct keys."""
        if self._inline:
            msg = "comment API is not available on inline tables"
            raise TOMLError(msg)
        return EolCommentView(self)

    @property
    def leading_comments(self) -> LeadingCommentView:
        """Mapping view of leading-comment blocks on this container's direct keys."""
        if self._inline:
            msg = "comment API is not available on inline tables"
            raise TOMLError(msg)
        return LeadingCommentView(self)

    @property
    def header_comment(self) -> str | None:
        """The EOL comment on this container's section header, or None."""
        return _header_comment_get(self)

    @header_comment.setter
    def header_comment(self, value: str | None) -> None:
        _header_comment_set(self, value)

    @header_comment.deleter
    def header_comment(self) -> None:
        _header_comment_set(self, None)

    @property
    def header_leading_comments(self) -> tuple[str, ...]:
        """The leading comment block immediately above this container's header."""
        return _header_leading_get(self)

    @header_leading_comments.setter
    def header_leading_comments(self, value: tuple[str, ...]) -> None:
        _header_leading_set(self, value)

    @header_leading_comments.deleter
    def header_leading_comments(self) -> None:
        _header_leading_set(self, ())

    @property
    def _subtree_tail(self) -> Slot | None:
        """Last slot owned anywhere in this container's subtree.

        Derived strictly from ``_refs`` ordering; not stored. Used as
        the insert-after anchor for child structural blocks (sections,
        AoT entries, dotted-descendant slots). Reading this on an
        inline-table view (where ``_refs`` stays empty) returns
        ``None``.
        """
        refs = self._refs
        return refs[-1].slot if refs else None

    # ------------------------------------------------------------------
    # Typed accessors
    # ------------------------------------------------------------------

    def table(self, key: str | Sequence[str]) -> Table:
        """Return the value at ``key`` typed as a `Table`.

        ``key`` may be a single name, a dotted-string path, or a
        sequence of names.
        """
        return self._typed_entry(key, Table, "a Table")

    def array(self, key: str | Sequence[str]) -> Array:
        """Return the value at ``key`` typed as an `Array`."""
        return self._typed_entry(key, Array, "an Array")

    def aot(self, key: str | Sequence[str]) -> AoT:
        """Return the value at ``key`` typed as an array-of-tables (`AoT`)."""
        return self._typed_entry(key, AoT, "an AoT")

    def get_table(self, key: str | Sequence[str], default: Any = None) -> Any:
        """Like `table(key)` but returns ``default`` if the key is missing."""
        return self._typed_entry_or(key, Table, "a Table", default)

    def get_array(self, key: str | Sequence[str], default: Any = None) -> Any:
        """Like `array(key)` but returns ``default`` if the key is missing."""
        return self._typed_entry_or(key, Array, "an Array", default)

    def get_aot(self, key: str | Sequence[str], default: Any = None) -> Any:
        """Like `aot(key)` but returns ``default`` if the key is missing."""
        return self._typed_entry_or(key, AoT, "an AoT", default)

    def _typed_entry(
        self, key: str | Sequence[str], cls: type[_T], label: str
    ) -> _T:
        v = self.entry(key)
        if not isinstance(v, cls):
            msg = f"value at {key!r} is {type(v).__name__}, not {label}"
            raise TypeError(msg)
        return v

    def _typed_entry_or(
        self, key: str | Sequence[str], cls: type[_T], label: str, default: Any
    ) -> _T | Any:
        try:
            return self._typed_entry(key, cls, label)
        except KeyError:
            return default

    def entry(self, path: str | Sequence[str]) -> Any:
        """Resolve a (possibly dotted) path; raises ``KeyError`` if missing.

        Raises ``TypeError`` if descent passes through a non-table.
        """
        parts = _split_path(path)
        cur: Any = self
        for i, p in enumerate(parts):
            if not isinstance(cur, Container):
                msg = f"cannot descend into {parts[i - 1]!r}: not a table"
                raise TypeError(msg)
            if p not in cur:
                raise KeyError(p)
            cur = dict.__getitem__(cur, p)
        return cur

    def get_entry(self, path: str | Sequence[str], default: Any = None) -> Any:
        """Like `entry(path)` but returns ``default`` if the path is missing."""
        try:
            return self.entry(path)
        except KeyError:
            return default

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Materialise a plain-Python ``dict`` (recursive)."""
        out: dict[str, Any] = {}
        for k, v in self.items():
            out[k] = _to_python(v)
        return out

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    @override
    def __setitem__(self, key: str, value: Any) -> None:
        if key in self and self[key] is value:
            return
        # Reject types we explicitly do not coerce (clear error rather
        # than a confusing NIE later in the dispatch).
        if isinstance(value, tuple):
            msg = f"cannot assign tuple to TOML key {key!r}; use a list"
            raise TypeError(msg)
        if isinstance(value, (bytes, bytearray)):
            msg = f"cannot assign bytes to TOML key {key!r}; use a string"
            raise TypeError(msg)
        # Unattached factory mode: dict-only storage, transplant on attach.
        if self._layout_root is None:
            dict.__setitem__(self, key, value)
            return
        if self._inline:
            self._inline_setitem(key, value)
            return
        if key in self:
            current = dict.__getitem__(self, key)
            # Fast-path: pure scalar → scalar (cheap, no synth alloc).
            if _is_scalar(current) and _is_scalar(value):
                self._scalar_replace(key, value)
                return
            # Single-direct-KV-slot current → any synth-able value
            # (scalar or inline). The slot's `value` field is swapped
            # in place; ordering, comments, key spelling are preserved.
            if (
                _is_scalar(current)
                or _is_inline_table(current)
                or isinstance(current, Array)
            ) and (_is_scalar(value) or _is_synth_inline(value)):
                self._inline_typed_replace(key, value)
                return
            # Same-flavour structural replace: mutate existing in
            # place. Preserves position, header trivia, leading
            # comments. Identity of the *destination* container is
            # preserved; the assigned `value` is *not* used as the
            # live view (matches the dict-snapshot contract for the
            # same-flavour case). Restricted to header-bearing
            # current containers — purely implicit ones go through
            # the delete+insert fallback so the new explicit header
            # lands where the implicit subtree used to be.
            if (
                _is_section(current)
                and current._header_ref is not None  # noqa: SLF001
                and isinstance(value, Mapping)
                and not isinstance(value, AoT)
            ) or (isinstance(current, AoT) and isinstance(value, (AoT, list))):
                # Same-flavour structural replace falls through to the
                # mixed-flavour del+set+move path below: the old view
                # must be detached (user references must stop reaching
                # the live doc) and the replacement re-installed at the
                # captured position.
                pass
            # Mixed-flavour (and same-flavour) structural overwrite:
            # capture position + leading of the existing primary,
            # delete the binding (which detaches the old view into a
            # PrivateRoot), then re-enter __setitem__ at the new-key
            # path. After the new value is installed (at end-of-doc by
            # default), move its slot block back to the captured
            # position with the saved leading.
            #
            # Preflight: only accept value shapes the new-key path
            # actually supports today, so a deletion doesn't go
            # through followed by an unsupported-insert raise that
            # leaves the doc partially mutated.
            if (
                _is_scalar(value)
                or _is_synth_inline(value)
                or isinstance(value, AoT)
                or _is_section(value)
                or isinstance(value, Mapping)
            ):
                primary_refs = self._index.get(key, [])
                saved_anchor_prev = None
                saved_leading_pieces: list[TriviaPiece] = []
                successor_slot = None
                successor_leading: list[TriviaPiece] | None = None
                if primary_refs:
                    old_primary = primary_refs[0].slot
                    saved_anchor_prev = old_primary._prev  # noqa: SLF001
                    saved_leading_pieces = list(old_primary.leading.pieces)
                    owned = _layout_ops._gather_value_owned_slots(current)  # noqa: SLF001
                    if owned:
                        successor_slot = owned[-1]._next  # noqa: SLF001
                        if successor_slot is not None:
                            successor_leading = list(successor_slot.leading.pieces)
                del self[key]
                self[key] = value
                if primary_refs and (
                    isinstance(value, AoT)
                    or _is_section(value)
                    or isinstance(value, Mapping)
                ):
                    _layout_ops.move_slots_to_anchor(
                        self, key, saved_anchor_prev, saved_leading_pieces
                    )
                    # Restore the successor's leading only if it's
                    # still the live slot immediately following the
                    # moved block — otherwise we'd risk overwriting
                    # a detached/orphaned slot's trivia or the wrong
                    # boundary.
                    if successor_slot is not None and successor_leading is not None:
                        new_owned = _layout_ops._gather_value_owned_slots(  # noqa: SLF001
                            dict.__getitem__(self, key)
                        )
                        if (
                            new_owned and new_owned[-1]._next is successor_slot  # noqa: SLF001
                        ):
                            successor_slot.leading.pieces = list(successor_leading)
                return
            # Unsupported value type — TypeError, not NIE.
            msg = (
                f"Cannot convert value of type {type(value).__name__!r} "
                f"for TOML key {key!r}"
            )
            raise TypeError(msg)
        # New direct-KV insert.
        if _is_scalar(value):
            _layout_ops.append_direct_kv(self, key, _coerce_scalar(value))
            dict.__setitem__(self, key, value)
            return
        if _is_synth_inline(value):
            cst, decoded = _synth_value(
                value,
                layout_root=self._layout_root,
                parent=self,
                path=(*self._path, key),
                owner=self._owner_aot_entry,
            )
            _layout_ops.append_direct_kv(self, key, cst)
            dict.__setitem__(self, key, decoded)
            return
        # Fall-through: typed section Container / AoT live-attach.
        if isinstance(value, AoT):
            # If `value` is attached to a live doc — or to a private
            # orphan with intact entry_slots (i.e. the user just
            # deleted its old binding via the structural-overwrite
            # del+set path) — route through clone_aot to preserve
            # per-entry trivia + nested sub-sections. The to_list()
            # snapshot path drops both.
            src_root = value._layout_root  # noqa: SLF001
            if (
                src_root is not None and not src_root._is_private  # noqa: SLF001
            ):
                if key in self:
                    del self[key]
                _layout_ops.clone_aot(self, key, value)
                return
            # Snapshot existing entry tables. If their `_owner_aot_entry`
            # records still hold intact `entry_slots` (e.g. the user
            # just deleted the old binding via the structural-overwrite
            # `del+set` path, which preserves slots into a private
            # orphan), capture them so we can deep-clone the CST into
            # the rehomed AoT — preserving per-KV trivia, nested
            # sub-section slots, and inter-entry separator style. The
            # generic `add_aot_entry(rehome=)` path is lossy: it
            # rebuilds slots from dict storage and drops all of that.
            existing_entries: list[Table] = list(value)
            preserved_entries: list[AoTEntry | None] = [
                et._owner_aot_entry  # noqa: SLF001
                if et._owner_aot_entry is not None  # noqa: SLF001
                and et._owner_aot_entry.entry_slots  # noqa: SLF001
                else None
                for et in existing_entries
            ]
            can_clone = preserved_entries and all(
                e is not None for e in preserved_entries
            )
            for et in existing_entries:
                _reset_table_for_rehome(et)
            list.clear(value)
            value._layout_root = None  # noqa: SLF001
            value._parent = None  # noqa: SLF001
            value._path = ()  # noqa: SLF001
            attached = _layout_ops.attach_empty_aot(self, key, value)
            dict.__setitem__(self, key, attached)
            if can_clone:
                # Deep-clone CST from intact orphan slots. Sacrifices
                # per-entry-table Python identity (the rehomed AoT
                # will hold fresh entry tables) in exchange for
                # trivia preservation. AoT object identity is still
                # preserved.
                for src_entry in preserved_entries:
                    assert src_entry is not None
                    _layout_ops.clone_aot_entry_from(value, src_entry)
            else:
                for entry_table in existing_entries:
                    _layout_ops.add_aot_entry(value, None, rehome=entry_table)
            return
        if _is_section(value):
            # Section-flavoured Table — synthesise [path] header.
            # Already-attached Table (live doc): clone via snapshot.
            # Detached/private: rehome in place.
            src_root = value._layout_root  # noqa: SLF001
            # AoT-entry source assigned as a standard table: route via
            # entry-cloner with head_kind="table" so trivia survives
            # and the head normalises from [[..]] to [..].
            if (
                src_root is not None
                and not src_root._is_private  # noqa: SLF001
                and value._owner_aot_entry is not None  # noqa: SLF001
                and self._layout_root is not None
            ):
                if key in self:
                    del self[key]
                _layout_ops.clone_aot_entry_as_table(self, key, value)
                return
            if src_root is not None and not src_root._is_private:  # noqa: SLF001
                # Cross-doc / same-doc attached section source: deep-
                # clone slots so trivia + nested sub-sections survive.
                if value._header_ref is not None:  # noqa: SLF001
                    if key in self:
                        del self[key]
                    _layout_ops.clone_section_as_section(self, key, value)
                    return
                # Implicit source / whole-Document: walk recursively
                # and re-install each structural child via tuple-path
                # `install`, preserving sections / AoTs as such (no
                # flatten-to-inline) and keeping implicit chains
                # implicit when there are no direct KVs to host.
                if key in self:
                    del self[key]
                _install_attached_subtree(self, (key,), value)
                return
            if src_root is not None and src_root._is_private:  # noqa: SLF001
                _reset_table_for_rehome(value)
            _layout_ops.attach_section(self, key, value)
            return
        # Unknown type → TypeError via _synth_value.
        _synth_value(
            value,
            layout_root=self._layout_root,
            parent=self,
            path=(*self._path, key),
            owner=self._owner_aot_entry,
        )
        msg = "internal: unexpected fall-through in __setitem__"
        raise AssertionError(msg)

    def _scalar_replace(self, key: str, value: Any) -> None:
        refs = self._index.get(key)
        if not refs:
            msg = f"internal: key {key!r} present in dict but missing from _index"
            raise AssertionError(msg)
        primary = refs[0]
        slot = primary.slot
        if not isinstance(slot, KVSlot):
            msg = "internal: scalar replace expects KVSlot"
            raise AssertionError(msg)  # noqa: TRY004
        slot.value = _coerce_scalar(value)
        dict.__setitem__(self, key, value)

    def _inline_typed_replace(self, key: str, value: Any) -> None:
        """Swap an existing direct-KV slot's value to a synthesised inline value.

        Works for any existing scalar / inline-table / inline-array
        binding bound by a single direct-KV slot. Dotted KV slots are
        also fine: the new value is just an inline value at the same
        leaf position.

        If the displaced value is itself a typed view (inline Table,
        Array), its attachment state is cleared so a subsequent
        assignment of that view elsewhere re-attaches live with
        identity preserved (rather than going through the
        cross-doc clone path).
        """
        refs = self._index.get(key)
        if not refs or len(refs) != 1:
            msg = "structural overwrite (multiple contributing refs) is not supported"
            raise NotImplementedError(msg)
        primary = refs[0]
        slot = primary.slot
        if not isinstance(slot, KVSlot):
            msg = "structural overwrite of header-bound binding is not supported"
            raise NotImplementedError(msg)
        old = dict.__getitem__(self, key)
        cst, decoded = _synth_value(
            value,
            layout_root=self._layout_root,
            parent=self,
            path=(*self._path, key),
            owner=self._owner_aot_entry,
        )
        slot.value = cst
        dict.__setitem__(self, key, decoded)
        # Detach the displaced view so it can be reattached live.
        if _is_inline_table(old) and old is not decoded:
            _reset_inline_for_rehome(old)
        elif isinstance(old, Array) and old is not decoded:
            _reset_array_for_rehome(old)

    @override
    def __delitem__(self, key: str) -> None:
        if self._inline:
            self._inline_delitem(key)
            return
        if key not in self:
            raise KeyError(key)
        _layout_ops.delete_key(self, key)

    # ------------------------------------------------------------------
    # Dict-method overrides — route through ``__setitem__`` /
    # ``__delitem__`` so inline / section / headerless dispatch is uniform.
    # ------------------------------------------------------------------

    @override
    def clear(self) -> None:
        for k in list(dict.keys(self)):
            del self[k]

    @override
    def pop(self, key: str, /, *args: Any) -> Any:
        if len(args) > 1:
            msg = f"pop expected at most 2 arguments, got {1 + len(args)}"
            raise TypeError(msg)
        if key in self:
            value = dict.__getitem__(self, key)
            del self[key]
            return value
        if args:
            return args[0]
        raise KeyError(key)

    @override
    def popitem(self) -> tuple[str, Any]:
        try:
            key = next(reversed(self))
        except StopIteration:
            msg = "dictionary is empty"
            raise KeyError(msg) from None
        value = dict.__getitem__(self, key)
        del self[key]
        return key, value

    @override
    def update(self, *args: Any, **kwargs: Any) -> None:
        if len(args) > 1:
            msg = f"update expected at most 1 argument, got {len(args)}"
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
        if key in self:
            return dict.__getitem__(self, key)
        self[key] = default
        return dict.__getitem__(self, key)

    @override
    def __ior__(self, other: Any) -> Self:  # type: ignore[override]
        self.update(other)
        return self

    def __copy__(self) -> Container:
        # Equivalent to deepcopy: returns an independent detached
        # container preserving nested typed views, so .table() etc.
        # continue to work on the copy.
        return _deep_section_clone(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> Container:
        return _deep_section_clone(self)

    # ------------------------------------------------------------------
    # Inline-table dispatch
    # ------------------------------------------------------------------

    def _inline_setitem(self, key: str, value: Any) -> None:
        if isinstance(value, AoT):
            msg = "Cannot store an array-of-tables inside an inline table"
            raise TOMLError(msg)
        if _is_section(value):
            msg = "Cannot store a section-style table inside an inline-style table"
            raise TOMLError(msg)
        if not _is_scalar(value) and not _is_synth_inline(value):
            msg = (
                "live-attach of typed Container/Array/AoT into an inline table "
                "is not supported"
            )
            raise NotImplementedError(msg)
        if key in self and isinstance(dict.__getitem__(self, key), Container):
            # Overwriting a dotted-prefix sub-table (e.g. `server`
            # in `{server.host = "x", server.port = 80}`) with a
            # scalar / inline value: delete every `server.*` entry
            # via the canonical delete path, then re-enter to add
            # `server = value` as a fresh single-keypart entry.
            del self[key]
            self[key] = value
            return
        if _is_scalar(value):
            cst: Value = _coerce_scalar(value)
            decoded: object = value
        else:
            cst, decoded = _synth_value(
                value,
                layout_root=self._layout_root,
                parent=self,
                path=(*self._path, key),
                owner=self._owner_aot_entry,
            )
        if key in self:
            ok = _inline_ops.replace_entry_value(self, key, cst)
            if not ok:
                msg = (
                    f"internal: key {key!r} present on inline view but no "
                    "matching entry in the backing InlineTableValue"
                )
                raise AssertionError(msg)
        else:
            _inline_ops.append_entry(self, key, cst)
        dict.__setitem__(self, key, decoded)

    def _inline_delitem(self, key: str) -> None:
        if key not in self:
            raise KeyError(key)
        ok = _inline_ops.delete_entry(self, key)
        if not ok:
            msg = (
                f"internal: key {key!r} present on inline view but no "
                "matching entry in the backing InlineTableValue"
            )
            raise AssertionError(msg)
        dict.__delitem__(self, key)
        # Clean up: a synthetic dotted-prefix sub-table that is now
        # empty has no representation in the backing
        # `InlineTableValue` either, so drop it from the parent's
        # dict view as well — and propagate up the chain.
        cur: Container | None = self
        while (
            cur is not None
            and cur._value is None  # noqa: SLF001
            and len(cur) == 0
            and cur._parent is not None  # noqa: SLF001
            and cur._parent._inline  # noqa: SLF001
            and cur._path  # noqa: SLF001
        ):
            parent = cur._parent  # noqa: SLF001
            my_key = cur._path[-1]  # noqa: SLF001
            if my_key in parent:
                dict.__delitem__(parent, my_key)
            cur = parent

    def install(self, path: str | Sequence[str], value: Any) -> Any:
        """Set ``value`` at the (possibly dotted) ``path``.

        Intermediate sections are created as needed via `ensure_table`.
        Returns the live view stored at the leaf.
        """
        parts = _validate_path(path)
        if self._inline and len(parts) > 1:
            msg = "cannot install dotted path into an inline-style table"
            raise TOMLError(msg)
        # If the value is a section-flavoured Table or an AoT, route
        # straight to the multi-component attach path so intermediate
        # path components stay implicit (no [tool] header for
        # `install("tool.poetry", Table.section())`).
        is_section = (
            isinstance(value, Table)
            and value._layout_root is None  # noqa: SLF001
            and not value._inline  # noqa: SLF001
        )
        is_aot = isinstance(value, AoT)
        if (is_section or is_aot) and len(parts) > 1 and self._layout_root is not None:
            # Walk existing prefix; whatever's left is created with
            # implicit intermediates plus the final explicit binding.
            cur: Container = self
            i = 0
            while i < len(parts) - 1:
                p = parts[i]
                if p not in cur:
                    break
                nxt = dict.__getitem__(cur, p)
                if isinstance(nxt, AoT):
                    msg = (
                        f"cannot install through array-of-tables at {p!r}: "
                        "no addressable target inside an AoT"
                    )
                    raise TOMLError(msg)
                if not isinstance(nxt, Container) or nxt._inline:  # noqa: SLF001
                    break
                cur = nxt
                i += 1
            # If we stopped at an inline-table prefix containing the
            # remaining tail, drop the conflicting tail key from the
            # inline so attach_section_at can install the section
            # without leaving a stale `name = "x"` shadow.
            if (
                i < len(parts) - 1
                and parts[i] in cur
                and isinstance(dict.__getitem__(cur, parts[i]), Container)
                and dict.__getitem__(cur, parts[i])._inline  # noqa: SLF001
            ):
                inline_holder: Container = dict.__getitem__(cur, parts[i])
                # Walk inline entries looking for the next path
                # component(s); delete each in order.
                tail = parts[i + 1 :]
                if tail and tail[0] in inline_holder:
                    del inline_holder[tail[0]]
                    if len(inline_holder) == 0:
                        del cur[parts[i]]
            # Overwrite-existing path: leaf already present, fall through
            # to direct __setitem__ on the deepest existing container.
            if i == len(parts) - 1:
                cur[parts[-1]] = value
                return cur[parts[-1]]
            if is_aot:
                # Multi-component AoT install: build implicit chain via
                # ensure_table on the prefix-of-leaf, then bind the AoT.
                # We do this by going through attach_section_at-style
                # implicit chain construction, but for AoT we still need
                # a per-entry header; easiest path is to bind the AoT at
                # the deepest existing container under the missing tail.
                # Walk implicit intermediates and let __setitem__ handle
                # the final binding (which routes through attach_empty_aot
                # / add_aot_entry).
                for p in parts[i : len(parts) - 1]:
                    implicit = Table()
                    implicit._layout_root = cur._layout_root  # noqa: SLF001
                    implicit._path = (*cur._path, p)  # noqa: SLF001
                    implicit._parent = cur  # noqa: SLF001
                    implicit._owner_aot_entry = cur._owner_aot_entry  # noqa: SLF001
                    dict.__setitem__(cur, p, implicit)
                    cur = implicit
                cur[parts[-1]] = value
                return cur[parts[-1]]
            sub = parts[i:]
            return _layout_ops.attach_section_at(cur, sub, value)
        host = self if len(parts) == 1 else self.ensure_table(parts[:-1])
        host[parts[-1]] = value
        return host[parts[-1]]

    def ensure_table(self, path: str | Sequence[str]) -> Table:
        """Return the section at ``path``, creating it if missing.

        If any prefix already exists as a section, descent continues
        from there. Intermediate components missing entirely are left
        implicit; only the deepest component gets an explicit
        ``[a.b.c]`` header. An existing non-table at any component
        raises ``TypeError``.
        """
        parts = _validate_path(path)
        if self._inline:
            msg = "cannot create section table inside an inline-style table"
            raise TOMLError(msg)
        cur: Container = self
        i = 0
        while i < len(parts):
            p = parts[i]
            if p not in cur:
                break
            nxt = dict.__getitem__(cur, p)
            if isinstance(nxt, AoT):
                msg = (
                    f"cannot ensure_table through array-of-tables at {p!r}: "
                    "no addressable target inside an AoT"
                )
                raise TOMLError(msg)
            if not isinstance(nxt, Container) or nxt._inline:  # noqa: SLF001
                msg = (
                    f"existing value at {p!r} is not section-backed "
                    "(is an inline table or non-table value)"
                )
                raise TOMLError(msg)
            cur = nxt
            i += 1
        if i == len(parts):
            assert isinstance(cur, Table)
            return cur
        if cur._layout_root is None:  # noqa: SLF001
            # Detached: build nested Table.section()s purely in dict
            # storage. No layout ops.
            for p in parts[i:]:
                child = Table.section()
                dict.__setitem__(cur, p, child)
                cur = child
            assert isinstance(cur, Table)
            return cur
        new_section = Table.section()
        attached = _layout_ops.attach_section_at(cur, parts[i:], new_section)
        assert isinstance(attached, Table)
        return attached

    def promote_inline(self, key: str) -> Table:
        """Convert an inline-table KV at ``key`` into a section header.

        Returns the live view at ``key`` after promotion. Raises
        ``TOMLError`` if the key is missing or doesn't refer to an
        inline-style table. If the value is already a section table,
        returns it unchanged.
        """
        if self._inline:
            msg = "inline-table promotion is not supported on inline tables"
            raise TOMLError(msg)
        if key not in self:
            msg = f"key {key!r} not in table"
            raise KeyError(msg)
        cur = dict.__getitem__(self, key)
        if not (_is_inline_table(cur)):
            msg = f"{key!r} is not an inline table"
            raise TOMLError(msg)
        if _inline_value_has_inner_comments(cur._value):  # noqa: SLF001
            msg = (
                f"cannot promote {key!r}: inline table has inner "
                f"comments that would be lost"
            )
            raise TOMLError(msg)
        # Capture leading + eol from the existing KV slot so we can
        # transfer them onto the new section header.
        old_slot = _direct_kv_slot(self, key)
        saved_leading = old_slot.leading if old_slot is not None else None
        saved_eol = old_slot.eol if old_slot is not None else None
        snapshot = cur.to_dict()
        del self[key]
        self[key] = Table.section(snapshot)
        result = dict.__getitem__(self, key)
        assert isinstance(result, Table)
        new_header = result._header_ref.slot if result._header_ref else None  # noqa: SLF001
        if isinstance(new_header, StructuralHeaderSlot):
            if saved_leading is not None:
                new_header.leading = saved_leading
            if saved_eol is not None:
                new_header.eol = saved_eol
            # Seam: ensure a blank line separates the parent's direct
            # entries from the promoted child header. promote_inline
            # turns a KV (originally inline, no separator) into a
            # section header (deserves visual separation).
            if (
                self._body_tail is not None
                and new_header._prev is self._body_tail  # noqa: SLF001
                and not _layout_ops._leading_has_blank_line(new_header.leading)  # noqa: SLF001
            ):
                layout_root = self._layout_root
                nl = layout_root._newline if layout_root else "\n"  # noqa: SLF001
                new_header.leading.pieces.insert(0, NewlineNode(text=nl))
        return result

    def promote_array(self, key: str) -> AoT:
        """Convert an array-of-inline-tables at ``key`` into an AoT.

        Returns the live AoT view at ``key``. If the value is already
        an AoT, returns it unchanged. Raises ``TOMLError`` if the key
        is missing, refers to a non-array, an empty array, or an array
        with non-inline-table elements.
        """
        if self._inline:
            msg = "array-of-tables promotion is not supported on inline tables"
            raise TOMLError(msg)
        if key not in self:
            msg = f"key {key!r} not in table"
            raise KeyError(msg)
        cur = dict.__getitem__(self, key)
        if not isinstance(cur, Array):
            msg = f"{key!r} is not an array"
            raise TOMLError(msg)
        if len(cur) == 0:
            msg = f"cannot promote empty array {key!r}"
            raise TOMLError(msg)
        for el in cur:
            if not (_is_inline_table(el)):
                msg = f"{key!r} contains a non-inline-table element"
                raise TOMLError(msg)
        if cur._value is not None:  # noqa: SLF001
            if _array_value_has_outer_comments(cur._value):  # noqa: SLF001
                msg = f"cannot promote {key!r}: array has comments that would be lost"
                raise TOMLError(msg)
            for entry_view in cur:
                ev = entry_view._value  # noqa: SLF001
                if ev is not None and _inline_value_has_inner_comments(ev):
                    msg = (
                        f"cannot promote {key!r}: array entry has inner "
                        f"comments that would be lost"
                    )
                    raise TOMLError(msg)
        snapshot = cur.to_list()
        # Capture the original KV slot's leading + eol so we can carry
        # them onto the first new ``[[..]]`` header and the last
        # entry's tail.
        old_slot = _direct_kv_slot(self, key)
        saved_leading = old_slot.leading if old_slot is not None else None
        saved_eol = old_slot.eol if old_slot is not None else None
        del self[key]
        self[key] = AoT(snapshot)
        result = dict.__getitem__(self, key)
        assert isinstance(result, AoT)
        # Apply saved leading to the first entry's header; saved eol
        # to the last entry's last slot.
        if saved_leading is not None and len(result) > 0:
            first_entry = result[0]
            entry_record = first_entry._owner_aot_entry  # noqa: SLF001
            if entry_record is not None and entry_record.entry_slots:
                first_slot = entry_record.entry_slots[0]
                if isinstance(first_slot, StructuralHeaderSlot):
                    # Prepend saved leading pieces in front of any
                    # leading already on the header (e.g. blank-line
                    # separator from `_build_section_leading`).
                    first_slot.leading.pieces = [
                        *saved_leading.pieces,
                        *first_slot.leading.pieces,
                    ]
        if saved_eol is not None and len(result) > 0:
            last_entry = result[-1]
            entry_record = last_entry._owner_aot_entry  # noqa: SLF001
            if entry_record is not None and entry_record.entry_slots:
                last_slot = entry_record.entry_slots[-1]
                if isinstance(last_slot, (KVSlot, StructuralHeaderSlot)) and (
                    saved_eol.comment is not None and last_slot.eol.comment is None
                ):
                    last_slot.eol.comment = saved_eol.comment
                    if saved_eol.trailing_ws is not None:
                        last_slot.eol.trailing_ws = saved_eol.trailing_ws
        return result


class Table(Container):
    """A logical TOML table.

    All mapping flavours in tomlrt (top-level document, standard
    table, inline table, and the synthetic mappings spawned by dotted
    keys) inherit from [`Table`][tomlrt.Table], which is itself a subclass of
    `dict`. So values typed as ``Table`` cover every nested
    mapping you can encounter while walking a document, *and*
    ``isinstance(t, dict)`` is ``True`` and ``**t`` works.

    **Storage model**

    A [`Table`][tomlrt.Table] is a *view* over the parsed concrete syntax
    tree (CST) — the physical tree of nodes that records every
    byte of the original document, including whitespace, comments,
    quote style and key order. Every mutation writes to the CST
    first and the dict storage is then refreshed from there. The
    CST is the single source of truth — `render` and every
    iteration ultimately read from it; the dict storage is a cache
    that mirrors the CST data and exists for two reasons:

    * fast ``dict``-style lookup, ``len``, ``in``, iteration, and
      ``**`` unpacking; and
    * stable object identity for nested containers, so that
      ``doc["foo"] is doc["foo"]``.

    Once a [`Table`][tomlrt.Table] is *detached* (see below) the CST link is
    severed and the dict storage takes over as the only source of
    truth for that orphan subtree.

    **Held references**

    Held references behave like ordinary Python dict references:

    * If the binding goes away (``del doc['foo']``), the held
      ``Table`` is *orphaned*: its dict storage is intact and reads
      still work, but it is no longer connected to the document and
      mutations through it do not appear in [`Document.render`][tomlrt.Document.render].
    * Re-binding the path (``doc['foo'] = {...}`` or
      ``doc.install('foo', Table.section())``) installs a *fresh* ``Table``;
      held references to the old table are unaffected.

    **Live vs snapshot containers**

    Assignment of a *container* value follows one rule: a container
    is attached to at most one CST location.

    * Assigning a fresh, unattached [`Array`][tomlrt.Array],
      [`Table.section`][tomlrt.Table.section] result,
      [`Table.inline`][tomlrt.Table.inline] result, or
      [`AoT`][tomlrt.AoT] *attaches in place*: the user's reference
      becomes the live view at the destination, so later mutations
      through that reference show up in the document.
      ``doc[k] is myvalue`` after the assign.
    * Assigning a container that is already attached somewhere
      (any document, including ``self``) deep-clones the source.
      The two slots are independent — mutations to one don't bleed
      into the other.
    * Plain `dict` and `list` values are *snapshot* on assignment.
      Mutations to the original mapping / list after assignment are
      *not* reflected in the document.
      Use [`Table.section`][tomlrt.Table.section],
      [`Table.inline`][tomlrt.Table.inline], or
      [`Array`][tomlrt.Array] to opt in to live
      semantics. Typed containers nested inside a plain dict / list
      still attach live recursively, even though the surrounding
      plain container is a snapshot.
    """

    __slots__ = ()

    @classmethod
    def section(cls, body: Mapping[str, Any] | None = None) -> Table:
        """Return a detached ``[k]`` standard-section table.

        Use from an assignment site:

            doc[k] = Table.section({"x": 1})

        The returned [`Table`][tomlrt.Table] is *live*: it is not yet
        connected to any document, but mutations -- ``t[k] = v``,
        ``t.update(...)``, nested
        [`Table.section`][tomlrt.Table.section] /
        [`AoT`][tomlrt.AoT] / [`Array`][tomlrt.Array] assignments --
        are recorded against its own private CST and survive into the
        document on assignment. Assigning the table installs it in
        place: ``doc[k] is t`` afterwards, and further mutations
        through ``t`` are visible in [`dumps`][tomlrt.dumps].

        Assigning a section table that is already attached somewhere
        deep-clones it; a single CST section lives at one location at
        a time.
        """
        t = cls()
        if body is not None:
            for k, v in body.items():
                dict.__setitem__(t, k, v)
        return t

    @classmethod
    def inline(cls, body: Mapping[str, Any] | None = None) -> Table:
        """Return a fresh inline table that *attaches live* on assignment.

        Use from an assignment site: ``doc[k] = Table.inline({...})``.
        Unlike a plain ``dict`` (which is snapshotted on assignment),
        the returned object becomes the *live* view at the assignment
        site: subsequent mutations through the original reference are
        reflected in the document, and ``doc[k] is the_inline`` after
        assignment.

        The inline table can be populated incrementally before
        assignment (``t = Table.inline(); t["a"] = 1; doc[k] = t``);
        all such mutations end up in the document.

        If the same inline-table object is assigned a second time
        (or after it has already been installed elsewhere), it is
        cloned: a single inline table is attached to at most one
        location in at most one document.
        """
        t = cls()
        t._inline = True
        if body is not None:
            for k, v in body.items():
                _k: object = k
                if not isinstance(_k, str):
                    msg = f"inline-table key must be str, got {type(_k).__name__}"
                    raise TypeError(msg)
                dict.__setitem__(t, _k, v)
        return t


class Document(Container):
    """Top-level TOML document. Subclass of [`Table`][tomlrt.Table]."""

    __slots__ = ("_head", "_is_private", "_newline", "_tail", "_trailing")

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        """Return a fresh empty document, optionally populated from ``data``.

        With a mapping, recursively populates the document so that:

        * nested mappings become standard ``[section]`` blocks (not
          inline tables);
        * lists of mappings become ``[[array.of.tables]]`` blocks;
        * everything else is set with ordinary key-value assignment.

        Existing [`Table`][tomlrt.Table] / [`AoT`][tomlrt.AoT] /
        [`Array`][tomlrt.Array] views are deep-cloned, so the returned
        document shares no mutable state with ``data``.
        """
        super().__init__()
        self._head: Slot | None = None
        self._tail: Slot | None = None
        self._trailing: Trivia = Trivia()
        self._newline: str = "\n"
        self._is_private: bool = False
        self._layout_root = self
        if data is not None:
            for k, v in data.items():
                self[k] = _coerce_for_document_init(v)

    def render(self) -> str:
        """Serialize the document back to a TOML string.

        Equivalent to `tomlrt.dumps(self)`.
        """
        return render(self)

    @property
    def preamble(self) -> tuple[str, ...]:
        """Comment block at the top of the document.

        A "preamble" is the run of ``# …`` lines that opens the file
        and is blank-line-separated from anything below. Comments that
        sit directly above the first key (no blank line) are *not*
        preamble — they are the leading comments of that key, accessed
        via `leading_comments`. In a document with no structural
        content, the entire opening comment block is treated as
        preamble.

        Setter accepts a sequence of bare comment texts (without the
        leading ``#``) and replaces the current preamble; assign ``()``
        to remove. Newlines inside any line are rejected.
        """
        return _doc_preamble_get(self)

    @preamble.setter
    def preamble(self, value: tuple[str, ...]) -> None:
        _doc_preamble_set(self, value)

    @preamble.deleter
    def preamble(self) -> None:
        _doc_preamble_set(self, ())

    @property
    def epilogue(self) -> tuple[str, ...]:
        """Comment block at the very end of the document.

        Returns the trailing run of ``# …`` lines that follows all
        structural content. Empty when the document has no structural
        content (in that case everything is `preamble`).

        Setter accepts a sequence of bare comment texts and replaces
        the current epilogue. Assign ``()`` to remove.

        Raises [`TOMLError`][tomlrt.TOMLError] if called with a
        non-empty value on a document with no structural content.
        """
        return _doc_epilogue_get(self)

    @epilogue.setter
    def epilogue(self, value: tuple[str, ...]) -> None:
        _doc_epilogue_set(self, value)

    @epilogue.deleter
    def epilogue(self) -> None:
        _doc_epilogue_set(self, ())

    @override
    def __copy__(self) -> Document:
        # Round-trip via dumps/loads: preserves bytes exactly.
        from tomlrt._public import loads  # noqa: PLC0415

        return loads(self.render())

    @override
    def __deepcopy__(self, memo: dict[int, Any]) -> Document:
        from tomlrt._public import loads  # noqa: PLC0415

        return loads(self.render())




def _inline_value_has_inner_comments(v: object) -> bool:
    """Return True iff the inline-table value carries inner comments.

    Used to refuse ``promote_inline`` on inline tables whose comments
    would have nowhere to live in the promoted form.
    """
    if not isinstance(v, InlineTableValue):
        return False
    return _comma_value_has_outer_comments(v.final_trivia, v.entries)


def _array_value_has_outer_comments(v: object) -> bool:
    """Return True iff the array carries item-level or final comments.

    "Outer" here means comments at the array layer itself; nested
    inline-value comments are tested separately (and produce a
    different error message).
    """
    if not isinstance(v, ArrayValue):
        return False
    return _comma_value_has_outer_comments(v.final_trivia, v.items)


def _comma_value_has_outer_comments(
    final_trivia: Trivia,
    parts: Iterable[ArrayItem | InlineTableEntry],
) -> bool:
    if trivia_has_comment(final_trivia):
        return True
    return any(
        trivia_has_comment(p.leading)
        or trivia_has_comment(p.trailing)
        or trivia_has_comment(p.post_comma_trivia)
        for p in parts
    )


def _deep_section_clone(c: Container) -> Container:
    """Build a detached deep clone of ``c`` as a section-flavoured Table.

    Nested ``Container`` and ``AoT`` values are recursively cloned as
    section / AoT typed views (preserving the user's ability to use
    ``.table()`` / ``.aot()`` on the copy). Inline values and scalars
    are passed through ``to_dict()``-equivalent normalisation.
    """
    out = Table.section()
    for k, v in c.items():
        if _is_section(v):
            dict.__setitem__(out, k, _deep_section_clone(v))
        elif isinstance(v, AoT):
            dict.__setitem__(out, k, AoT([_deep_section_clone(e) for e in v]))
        else:
            dict.__setitem__(out, k, _to_python(v))
    return out


def _reset_table_for_rehome(t: Container, *, recurse: bool = False) -> None:
    """Clear a Table's slot infrastructure so it can be reattached.

    Preserves dict storage (so post-detach mutations survive) but
    drops `_layout_root` / `_path` / `_parent` / `_owner_aot_entry`
    / `_refs` / `_index` / `_header_ref` / `_body_tail` so the
    standard attach path treats `t` as if freshly constructed.

    With ``recurse=True``, walks dict values and resets nested
    non-inline ``Container`` / ``AoT`` children whose
    ``_layout_root`` matches the root's previous layout root (i.e.
    they belong to the same detached subtree). Children pointing
    at a different doc are left alone — they're an "alien" live
    view that the standard cross-doc clone path will handle.
    Recursion is opt-in because most rehome callers operate on a
    single freshly-detached Table and don't pay the subtree walk.

    Used when re-installing a held view that was detached into a
    private orphan ``Document``.
    """
    old_root = t._layout_root  # noqa: SLF001
    t._layout_root = None  # noqa: SLF001
    t._path = ()  # noqa: SLF001
    t._parent = None  # noqa: SLF001
    t._owner_aot_entry = None  # noqa: SLF001
    t._refs = []  # noqa: SLF001
    t._index = {}  # noqa: SLF001
    t._header_ref = None  # noqa: SLF001
    t._body_tail = None  # noqa: SLF001

    if not recurse:
        return
    for child in dict.values(t):
        if _is_section(child):
            if child._layout_root is old_root:  # noqa: SLF001
                _reset_table_for_rehome(child, recurse=True)
        elif isinstance(child, AoT) and child._layout_root is old_root:  # noqa: SLF001
            for entry in list.__iter__(child):
                _reset_table_for_rehome(entry, recurse=True)
            child._layout_root = None  # noqa: SLF001
            child._parent = None  # noqa: SLF001
            child._path = ()  # noqa: SLF001


def _reset_inline_for_rehome(t: Container) -> None:
    """Clear an inline Table's slot infrastructure so it can be reattached.

    Inline tables are slot-less from the doc-stream perspective but
    keep an ``InlineTableValue`` in their ``_value`` field once
    attached. Drop that and the layout-root pointer so the standard
    inline-attach path treats ``t`` as if freshly constructed.
    """
    t._layout_root = None  # noqa: SLF001
    t._parent = None  # noqa: SLF001
    t._owner_aot_entry = None  # noqa: SLF001
    if hasattr(t, "_value"):
        t._value = None  # noqa: SLF001


def _reset_array_for_rehome(a: Array) -> None:
    """Clear an Array view's attachment so it can be reattached live.

    Leaves ``_value`` (the displaced ``ArrayValue``) intact so the
    next attach can reuse it.
    """
    a._attached = False  # noqa: SLF001


def _install_attached_subtree(
    dst_parent: Container, dst_path: tuple[str, ...], src_table: Container
) -> None:
    """Recursively install an attached implicit / Document source.

    Walks ``src_table.items()`` and re-installs each entry under
    ``dst_parent`` using tuple-path :meth:`Container.install` so that
    attached sections clone via ``clone_section_as_section`` and
    AoTs clone via ``clone_aot``. Implicit chains stay implicit
    (no ``[k]`` header is synthesised unless there are direct KVs to
    host); when there *are* direct scalar / inline KVs at this level,
    a ``Table.section`` snapshot is installed at ``dst_path`` to
    carry them.
    """
    direct_kvs: list[tuple[str, Any]] = []
    structural: list[tuple[str, Any]] = []
    for k, v in src_table.items():
        if isinstance(v, AoT) or (
            _is_section(v)
        ):
            structural.append((k, v))
        else:
            direct_kvs.append((k, v))

    if direct_kvs:
        snapshot = Table.section()
        for k, v in direct_kvs:
            snapshot[k] = _to_python(v)
        dst_parent.install(dst_path, snapshot)

    for k, v in structural:
        sub_path = (*dst_path, k)
        if isinstance(v, AoT) or (
            isinstance(v, Container) and v._header_ref is not None  # noqa: SLF001
        ):
            dst_parent.install(sub_path, v)
        elif isinstance(v, Container):
            _install_attached_subtree(dst_parent, sub_path, v)


def _split_path(path: str | Sequence[str]) -> list[str]:
    """Split a path argument into a list of component names.

    A ``str`` is interpreted as a dotted path (no quoting support; for
    keys containing dots, pass a sequence). A non-string ``Sequence``
    is taken verbatim.
    """
    if isinstance(path, str):
        return path.split(".") if path else []
    return list(path)


def _validate_path(path: object) -> list[str]:
    """Validate a key-path argument and return its components.

    Raises ``TypeError`` for the wrong outer type, and ``TOMLError``
    for empty paths or paths with empty segments.
    """
    if isinstance(path, str):
        if path == "":
            msg = "key path must not be empty"
            raise TOMLError(msg)
        parts = path.split(".")
        for p in parts:
            if p == "":
                msg = f"key path {path!r} contains an empty segment"
                raise TOMLError(msg)
        return parts
    if isinstance(path, (list, tuple)):
        if len(path) == 0:
            msg = "key path must not be empty"
            raise TOMLError(msg)
        out: list[str] = []
        for seg in path:
            if not isinstance(seg, str):
                msg = f"key path segment must be str, got {type(seg).__name__}"
                raise TypeError(msg)
            if seg == "":
                msg = "key path contains an empty segment"
                raise TOMLError(msg)
            out.append(seg)
        return out
    msg = f"key path must be str or sequence of str, got {type(path).__name__}"
    raise TypeError(msg)


def _to_python(v: Any) -> Any:
    """Recursively materialise a tomlrt view into plain Python values."""
    if isinstance(v, Container):
        return v.to_dict()
    if isinstance(v, AoT):
        return [t.to_dict() for t in v]
    if isinstance(v, Array):
        return [_to_python(x) for x in v]
    return v


# ---------------------------------------------------------------------------
# Scalar coercion
# ---------------------------------------------------------------------------


def _is_scalar(v: object) -> bool:
    """True iff ``v`` is a TOML scalar (and not an array / table)."""
    # `bool` is an `int` subclass — explicit allow keeps the semantics
    # in this gate clear.
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float, str)):
        return True
    return isinstance(v, (datetime, date, time))


def _is_section(v: object) -> TypeGuard[Container]:
    """True iff ``v`` is a non-inline (section-style) Container."""
    return isinstance(v, Container) and not v._inline  # noqa: SLF001


def _is_inline_table(v: object) -> TypeGuard[Container]:
    """True iff ``v`` is an inline Container."""
    return isinstance(v, Container) and v._inline  # noqa: SLF001


def _coerce_for_document_init(v: Any) -> Any:
    """Pick a sensible structural shape for ``Document(data=...)`` values.

    * Mapping → section ``Table.section`` (recursively coerced).
    * List of mappings (non-empty) → ``AoT`` of section tables.
    * Anything else passes through unchanged.
    """
    if isinstance(v, AoT):
        return v
    if isinstance(v, Container):
        return v
    if isinstance(v, Mapping):
        return Table.section(
            {k: _coerce_for_document_init(sub) for k, sub in v.items()}
        )
    if isinstance(v, list) and v and all(isinstance(x, Mapping) for x in v):
        return AoT(
            [{k: _coerce_for_document_init(sub) for k, sub in m.items()} for m in v]
        )
    return v


def _coerce_scalar(
    v: object,
) -> StringValue | IntegerValue | FloatValue | BoolValue | DateTimeValue:
    """Coerce a Python scalar to a fresh `Value` with a default lexeme."""
    if isinstance(v, bool):
        return BoolValue(lexeme="true" if v else "false", value=v)
    if isinstance(v, int):
        return IntegerValue(lexeme=str(v), value=v, style="dec")
    if isinstance(v, float):
        return FloatValue(lexeme=_float_lexeme(v), value=v)
    if isinstance(v, str):
        return StringValue(lexeme=_basic_string_lexeme(v), value=v, style="basic")
    if isinstance(v, datetime):
        return DateTimeValue(lexeme=v.isoformat(), value=v, kind=_dt_kind(v))
    if isinstance(v, date):
        return DateTimeValue(lexeme=v.isoformat(), value=v, kind="local-date")
    if isinstance(v, time):
        return DateTimeValue(lexeme=v.isoformat(), value=v, kind="local-time")
    msg = f"cannot coerce {type(v).__name__} to a TOML scalar"
    raise TypeError(msg)


def _float_lexeme(v: float) -> str:
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "-inf" if v < 0 else "inf"
    s = repr(v)
    # Python may emit "1e10" — TOML requires a fractional component or an
    # exponent; keep the repr() output as is (TOML accepts both).
    if "." not in s and "e" not in s and "E" not in s and "n" not in s:
        s += ".0"
    return s


def _basic_string_lexeme(v: str) -> str:
    out = ['"']
    for ch in v:
        c = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\f":
            out.append("\\f")
        elif ch == "\r":
            out.append("\\r")
        elif c < 0x20 or c == 0x7F:
            out.append(f"\\u{c:04X}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _dt_kind(v: object) -> DateLikeKind:
    assert isinstance(v, datetime)
    return "offset-datetime" if v.tzinfo is not None else "local-datetime"


# `_array` depends on `Container` for `Table`, so the import is at the
# bottom to avoid a circular import. The `Array` / `AoT` symbols are
# re-exported for convenience.
from tomlrt._array import AoT, Array  # noqa: E402

TomlInput = "Mapping[str, Any] | Document"


# ---------------------------------------------------------------------------
# Plain-Python value synthesis.
# ---------------------------------------------------------------------------


def _is_synth_inline(v: object) -> bool:
    """True iff ``v`` is a value we can synthesise to an inline TOML value.

    Accepts:
    - any ``Mapping`` (dict, MappingProxyType, …) — including our own
      inline ``Container`` views (deep-copy semantics)
    - ``list`` — including our own ``Array`` views (deep-copy semantics)
    - inline ``Container`` and ``Array`` views from another document

    Rejects everything else (tuple, bytes, sets, AoT, section
    Container, …) so the caller can route to a stronger error.
    """
    if isinstance(v, AoT):
        return False
    if isinstance(v, Container):
        # Section containers need real live-attach; only inline ones
        # round-trip through value-synthesis safely.
        return v._inline  # noqa: SLF001
    if isinstance(v, Array):
        return True
    if isinstance(v, Mapping):
        return True
    # `list` only — `tuple` is intentionally not accepted (TOML has no
    # tuple, and accepting it would mask user typos).
    return type(v) is list or (isinstance(v, list) and not isinstance(v, Array))


def _make_keypart(name: str) -> KeyPart:
    """Build a `KeyPart` for a synthesised key, choosing bare vs basic."""
    if re.match(r"\A[A-Za-z0-9_\-]+\Z", name):
        return KeyPart(raw=name, value=name, kind="bare")
    return KeyPart(raw=_basic_string_lexeme(name), value=name, kind="basic")


def _synth_value(
    v: object,
    *,
    layout_root: Document | None,
    parent: Container | None,
    path: tuple[str, ...],
    owner: AoTEntry | None,
) -> tuple[Value, object]:
    """Synthesise a (CST value, decoded view) pair from ``v``.

    The CST value goes into the host slot's ``value`` field; the
    decoded view is what gets stored in the parent dict (and is the
    object the user retrieves via ``[]``).

    Plain ``dict`` / ``Mapping`` → ``InlineTableValue`` + inline ``Table``.
    ``list`` / ``Array`` view → ``ArrayValue`` + ``Array``.
    Section ``Container`` / ``AoT`` raise NIE.
    Anything else raises ``TypeError`` (mentioning the type name and
    the prefix ``"Cannot convert"``).
    """
    if _is_scalar(v):
        return _coerce_scalar(v), v
    if isinstance(v, AoT):
        msg = "live-attach of AoT through value synthesis is not supported"
        raise NotImplementedError(msg)
    if _is_section(v):
        msg = (
            "live-attach of section Container through value synthesis is not supported"
        )
        raise NotImplementedError(msg)
    # Unattached inline Container (Table.inline()) — live attach: rehome
    # the existing object instead of creating a fresh Table view, so the
    # user's reference stays the document's view. Detached inline tables
    # (sitting in a private root after a prior overwrite) are also
    # treated as unattached.
    if (
        _is_inline_table(v)
        and (
            v._layout_root is None  # noqa: SLF001
            or v._layout_root._is_private  # noqa: SLF001
        )
    ):
        if v._layout_root is not None:  # noqa: SLF001
            _reset_inline_for_rehome(v)
        return _live_attach_inline_table(
            v, layout_root=layout_root, parent=parent, path=path, owner=owner
        )
    # Unattached Array — live attach. Arrays don't carry a layout-root
    # field; their displaced state is signalled by ``_attached = False``
    # (which ``_inline_typed_replace`` resets when an Array view is
    # overwritten in-place).
    if isinstance(v, Array) and not v._attached:  # noqa: SLF001
        return _live_attach_array(v)
    # Mappings (incl. inline Container) → inline table.
    if isinstance(v, Mapping) or (_is_inline_table(v)):
        return _synth_inline_table(
            v, layout_root=layout_root, parent=parent, path=path, owner=owner
        )
    # Lists (incl. Array views) → inline array.
    if isinstance(v, list):
        return _synth_inline_array(v, layout_root=layout_root, owner=owner)
    msg = f"Cannot convert {type(v).__name__} to a TOML value"
    raise TypeError(msg)


def _live_attach_inline_table(
    t: Container,
    *,
    layout_root: Document | None,
    parent: Container | None,
    path: tuple[str, ...],
    owner: AoTEntry | None,
) -> tuple[InlineTableValue, Container]:
    """Rehome an unattached inline `Table` into ``layout_root``.

    Builds an ``InlineTableValue`` from ``t``'s current dict contents,
    points ``t._value`` at it, and records the position. Returns
    ``(value, t)`` so the caller stores ``t`` itself (preserving
    user-visible identity) in the parent dict.
    """
    val = InlineTableValue()
    t._layout_root = layout_root  # noqa: SLF001
    t._path = path  # noqa: SLF001
    t._parent = parent  # noqa: SLF001
    t._inline = True  # noqa: SLF001
    t._owner_aot_entry = owner  # noqa: SLF001
    t._value = val  # noqa: SLF001

    items = list(t.items())
    for i, (k, sub) in enumerate(items):
        sub_cst, sub_dec = _synth_value(
            sub,
            layout_root=layout_root,
            parent=t,
            path=(*path, k),
            owner=owner,
        )
        is_last = i == len(items) - 1
        entry = InlineTableEntry(
            leading=Trivia([WhitespaceNode(text=" ")]),
            key_parts=[_make_keypart(k)],
            key_seps=[],
            pre_eq=" ",
            post_eq=" ",
            value=sub_cst,
            trailing=Trivia(),
            has_comma=not is_last,
            post_comma_trivia=Trivia(),
        )
        val.entries.append(entry)
        dict.__setitem__(t, k, sub_dec)
    if items:
        val.final_trivia = Trivia([WhitespaceNode(text=" ")])
    return val, t


def _live_attach_array(a: Array) -> tuple[ArrayValue, Array]:
    """Rehome an unattached `Array` into ``layout_root``.

    The Array always has a backing ``ArrayValue`` (the constructor
    initialises one); just mark it attached and hand back the value.
    """
    a._attached = True  # noqa: SLF001
    assert a._value is not None  # noqa: SLF001
    return a._value, a  # noqa: SLF001


def _synth_inline_table(
    d: Mapping[Any, Any],
    *,
    layout_root: Document | None,
    parent: Container | None,
    path: tuple[str, ...],
    owner: AoTEntry | None,
) -> tuple[InlineTableValue, Table]:
    val = InlineTableValue()
    table = Table()
    table._layout_root = layout_root  # noqa: SLF001
    table._path = path  # noqa: SLF001
    table._parent = parent  # noqa: SLF001
    table._inline = True  # noqa: SLF001
    table._owner_aot_entry = owner  # noqa: SLF001
    table._value = val  # noqa: SLF001

    items = list(d.items())
    for i, (k, sub) in enumerate(items):
        if not isinstance(k, str):
            msg = f"inline-table key must be str, got {type(k).__name__}"
            raise TypeError(msg)
        sub_cst, sub_dec = _synth_value(
            sub,
            layout_root=layout_root,
            parent=table,
            path=(*path, k),
            owner=owner,
        )
        is_last = i == len(items) - 1
        entry = InlineTableEntry(
            leading=Trivia([WhitespaceNode(text=" ")]),
            key_parts=[_make_keypart(k)],
            key_seps=[],
            pre_eq=" ",
            post_eq=" ",
            value=sub_cst,
            trailing=Trivia(),
            has_comma=not is_last,
            post_comma_trivia=Trivia(),
        )
        val.entries.append(entry)
        dict.__setitem__(table, k, sub_dec)
    if items:
        val.final_trivia = Trivia([WhitespaceNode(text=" ")])
    return val, table


def _synth_inline_array(
    items: Sequence[object],
    *,
    layout_root: Document | None,
    owner: AoTEntry | None,
) -> tuple[ArrayValue, Array]:
    val = ArrayValue()
    arr = Array()
    arr._value = val  # noqa: SLF001

    for i, sub in enumerate(items):
        sub_cst, sub_dec = _synth_value(
            sub,
            layout_root=layout_root,
            parent=None,
            path=(),
            owner=owner,
        )
        is_last = i == len(items) - 1
        # Place the inter-item space in post_comma_trivia (matching
        # the parser's `,(space)2` shape) so _detect_style sees it
        # and subsequent appends use the right separator.
        item = ArrayItem(
            leading=Trivia(),
            value=sub_cst,
            trailing=Trivia(),
            has_comma=not is_last,
            post_comma_trivia=(
                Trivia([WhitespaceNode(text=" ")]) if not is_last else Trivia()
            ),
        )
        val.items.append(item)
        list.append(arr, sub_dec)
    return val, arr


__all__ = ["AoT", "Array", "Container", "Document", "Table", "TomlInput"]
