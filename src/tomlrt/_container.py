"""Logical container layer.

`Container(dict)` is the dict-typed base for both `Document` (the
root) and `Table` (sections + inline tables). Phase 2 only needs the
read surface: dict-storage populated in doc-stream-first-occurrence
order, typed accessors, conversion helpers, and the `render()` entry
point. The mutation-time scaffolding (`_index`, `_refs`,
`_header_ref`, `_body_tail`) is deferred to Phase 3.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if sys.version_info >= (3, 12):
    from typing import Self, override
else:
    from typing_extensions import override

from tomlrt._render import render
from tomlrt._slots import KVSlot
from tomlrt._trivia import Trivia, WhitespaceNode
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
    from collections.abc import Mapping, Sequence

    from typing_extensions import Self

    from tomlrt._slots import AoTEntry, Slot, SlotRef
    from tomlrt._values import (
        Value,
    )


class Container(dict[str, Any]):
    """Dict-typed base for `Document` and `Table` views.

    Reads are pure dict operations. Mutation paths use the per-container
    cache (`_index` / `_refs` / `_header_ref` / `_body_tail`)
    maintained alongside the dict storage. ``_subtree_tail`` is exposed
    as a derived `@property` over `_refs`. For inline tables
    (`_inline=True`) the slot-stream caches stay empty and `_value`
    points at the backing `InlineTableValue` instead — inline mutation
    lives in a separate code path (Phase 3b).
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
        v = self.entry(key)
        if not isinstance(v, Table):
            msg = f"value at {key!r} is {type(v).__name__}, not a Table"
            raise TypeError(msg)
        return v

    def array(self, key: str | Sequence[str]) -> Array:
        """Return the value at ``key`` typed as an `Array`."""
        v = self.entry(key)
        if not isinstance(v, Array):
            msg = f"value at {key!r} is {type(v).__name__}, not an Array"
            raise TypeError(msg)
        return v

    def aot(self, key: str | Sequence[str]) -> AoT:
        """Return the value at ``key`` typed as an array-of-tables (`AoT`)."""
        v = self.entry(key)
        if not isinstance(v, AoT):
            msg = f"value at {key!r} is {type(v).__name__}, not an AoT"
            raise TypeError(msg)
        return v

    def get_table(self, key: str | Sequence[str], default: Any = None) -> Any:
        """Like `table(key)` but returns ``default`` if the key is missing."""
        try:
            v = self.entry(key)
        except KeyError:
            return default
        if not isinstance(v, Table):
            msg = f"value at {key!r} is {type(v).__name__}, not a Table"
            raise TypeError(msg)
        return v

    def get_array(self, key: str | Sequence[str], default: Any = None) -> Any:
        """Like `array(key)` but returns ``default`` if the key is missing."""
        try:
            v = self.entry(key)
        except KeyError:
            return default
        if not isinstance(v, Array):
            msg = f"value at {key!r} is {type(v).__name__}, not an Array"
            raise TypeError(msg)
        return v

    def get_aot(self, key: str | Sequence[str], default: Any = None) -> Any:
        """Like `aot(key)` but returns ``default`` if the key is missing."""
        try:
            v = self.entry(key)
        except KeyError:
            return default
        if not isinstance(v, AoT):
            msg = f"value at {key!r} is {type(v).__name__}, not an AoT"
            raise TypeError(msg)
        return v

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
    # Mutation (Phase 3a: scalar-replaces-scalar only)
    # ------------------------------------------------------------------

    @override
    def __setitem__(self, key: str, value: Any) -> None:
        if key in self and self[key] is value:
            return
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
                or (isinstance(current, Container) and current._inline)  # noqa: SLF001
                or isinstance(current, Array)
            ) and (_is_scalar(value) or _is_synth_inline(value)):
                self._inline_typed_replace(key, value)
                return
            msg = (
                "non-scalar replacement (typed Container/AoT source, or "
                "section / AoT current) is Phase 4"
            )
            raise NotImplementedError(msg)
        # New direct-KV insert (Phase 3c / 3d / 4-partial).
        if _is_scalar(value):
            from tomlrt import _layout_ops  # noqa: PLC0415

            _layout_ops.append_direct_kv(self, key, _coerce_scalar(value))
            dict.__setitem__(self, key, value)
            return
        if _is_synth_inline(value):
            from tomlrt import _layout_ops  # noqa: PLC0415

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
            from tomlrt import _layout_ops  # noqa: PLC0415

            # Snapshot any pre-existing entries' bodies; rehome the
            # AoT object as empty; then re-add each body.
            pending = [dict(t) for t in value]
            list.clear(value)
            attached = _layout_ops.attach_empty_aot(self, key, value)
            dict.__setitem__(self, key, attached)
            for body in pending:
                _layout_ops.add_aot_entry(value, body)
            return
        if isinstance(value, Container) and not value._inline:  # noqa: SLF001
            # Section-flavoured Table — synthesise [path] header.
            from tomlrt import _layout_ops  # noqa: PLC0415

            attached = _layout_ops.attach_section(self, key, value)
            dict.__setitem__(self, key, attached)
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
        """
        refs = self._index.get(key)
        if not refs or len(refs) != 1:
            msg = (
                "structural overwrite (multiple contributing refs) is "
                "deferred to Phase 4"
            )
            raise NotImplementedError(msg)
        primary = refs[0]
        slot = primary.slot
        if not isinstance(slot, KVSlot):
            msg = "structural overwrite of header-bound binding is deferred to Phase 4"
            raise NotImplementedError(msg)
        cst, decoded = _synth_value(
            value,
            layout_root=self._layout_root,
            parent=self,
            path=(*self._path, key),
            owner=self._owner_aot_entry,
        )
        slot.value = cst
        dict.__setitem__(self, key, decoded)

    @override
    def __delitem__(self, key: str) -> None:
        if self._inline:
            self._inline_delitem(key)
            return
        if key not in self:
            raise KeyError(key)
        from tomlrt import _layout_ops  # noqa: PLC0415

        _layout_ops.delete_key(self, key)

    # ------------------------------------------------------------------
    # Dict-method overrides (Phase 3d-2)
    #
    # All of these route through ``self[k] = v`` / ``del self[k]`` so
    # the inline-vs-section-vs-headerless dispatch in ``__setitem__`` /
    # ``__delitem__`` handles both flavours uniformly.
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
        msg = "copy.copy on a tomlrt container is deferred to Phase 3e"
        raise NotImplementedError(msg)

    def __deepcopy__(self, memo: dict[int, Any]) -> Container:
        msg = "copy.deepcopy on a tomlrt container is deferred to Phase 3e"
        raise NotImplementedError(msg)

    # ------------------------------------------------------------------
    # Inline-table dispatch (Phase 3b)
    # ------------------------------------------------------------------

    def _inline_setitem(self, key: str, value: Any) -> None:
        from tomlrt import _inline_ops  # noqa: PLC0415

        if not _is_scalar(value) and not _is_synth_inline(value):
            msg = (
                "live-attach of typed Container/Array/AoT into an inline "
                "table is Phase 4"
            )
            raise NotImplementedError(msg)
        if key in self and isinstance(dict.__getitem__(self, key), Container):
            # Replacing a dotted-prefix sub-table (e.g. `a` in
            # `{a.b = 1}`) would have to delete every `a.*` entry and
            # add an `a = ...` entry. Structural overwrite, deferred.
            msg = (
                "overwrite of a dotted-inline sub-table is not yet "
                "supported (structural overwrite, Phase 4)"
            )
            raise NotImplementedError(msg)
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
        from tomlrt import _inline_ops  # noqa: PLC0415

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


class Table(Container):
    """A section table, implicit table, or inline table view."""

    __slots__ = ()

    @classmethod
    def section(cls, body: Mapping[str, Any] | None = None) -> Table:
        """Build an unattached section-flavoured `Table` view.

        Assigning the result into a document (``doc["k"] = t``)
        synthesises a ``[k]`` section header and migrates ``body``
        into the live document.
        """
        t = cls()
        if body is not None:
            for k, v in body.items():
                dict.__setitem__(t, k, v)
        return t

    @classmethod
    def inline(cls, body: Mapping[str, Any] | None = None) -> Table:
        """Build an unattached inline-flavoured `Table` view.

        Assigning the result into a document or another container
        creates an inline table (``k = {...}``).
        """
        t = cls()
        t._inline = True
        if body is not None:
            for k, v in body.items():
                dict.__setitem__(t, k, v)
        return t


class Document(Container):
    """A parsed TOML document.

    Owns the physical slot stream (head/tail of the doubly-linked list,
    plus trailing trivia and detected newline). The dict-typed body is
    inherited from `Container`.
    """

    __slots__ = ("_head", "_newline", "_tail", "_trailing")

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self._head: Slot | None = None
        self._tail: Slot | None = None
        self._trailing: Trivia = Trivia()
        self._newline: str = "\n"
        self._layout_root = self
        if data is not None:
            for k, v in data.items():
                self[k] = v

    def render(self) -> str:
        return render(self)

    def install(self, path: str | Sequence[str], value: Any) -> Any:
        """Set ``value`` at the (possibly dotted) ``path``.

        Intermediate sections are created as needed via `ensure_table`.
        Returns the live view stored at the leaf.
        """
        parts = _validate_path(path)
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
        # Walk as far down as existing structure goes.
        cur: Container = self
        i = 0
        while i < len(parts):
            p = parts[i]
            if p not in cur:
                break
            nxt = dict.__getitem__(cur, p)
            if not isinstance(nxt, Container) or nxt._inline:  # noqa: SLF001
                msg = f"cannot descend into {p!r}: not a section table"
                raise TypeError(msg)
            cur = nxt
            i += 1
        if i == len(parts):
            assert isinstance(cur, Table)
            return cur
        # Synthesise a single section spanning the remaining components.
        from tomlrt import _layout_ops  # noqa: PLC0415

        new_section = Table.section()
        # Compose the deepest section path relative to `cur`.
        # We need a multi-component child key under `cur`. Easiest: use
        # attach_section but adjusted. For now we only support single
        # missing tail (test case has all-missing). Build by rebinding
        # the section's path manually.
        attached = _layout_ops.attach_section_at(cur, parts[i:], new_section)
        assert isinstance(attached, Table)
        return attached


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
    from tomlrt._errors import TOMLError  # noqa: PLC0415

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
# Scalar coercion (Phase 3a — minimal; full coverage in Phase 3c via
# `_coerce.py`).
# ---------------------------------------------------------------------------


def _is_scalar(v: object) -> bool:
    """True iff ``v`` is a TOML scalar (and not an array / table)."""
    from datetime import date, datetime, time  # noqa: PLC0415

    # `bool` is an `int` subclass — explicit allow keeps the semantics
    # in this gate clear.
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float, str)):
        return True
    return isinstance(v, (datetime, date, time))


def _coerce_scalar(
    v: object,
) -> StringValue | IntegerValue | FloatValue | BoolValue | DateTimeValue:
    """Coerce a Python scalar to a fresh `Value` with a default lexeme."""
    from datetime import date, datetime, time  # noqa: PLC0415

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
    import math  # noqa: PLC0415

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


def _dt_kind(v: object) -> Any:
    from datetime import datetime  # noqa: PLC0415

    assert isinstance(v, datetime)
    return "offset-datetime" if v.tzinfo is not None else "local-datetime"


# `_array` depends on `Container` for `Table`, so the import is at the
# bottom to avoid a circular import. The `Array` / `AoT` symbols are
# re-exported for convenience.
from tomlrt._array import AoT, Array  # noqa: E402

TomlInput = "Mapping[str, Any] | Document"


# ---------------------------------------------------------------------------
# Plain-Python value synthesis (Phase 4-partial — plain dict/list only).
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
    from collections.abc import Mapping  # noqa: PLC0415

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
    import re  # noqa: PLC0415

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
    Section ``Container`` / ``AoT`` raise NIE (Phase 4 live-attach).
    Anything else raises ``TypeError`` (mentioning the type name and
    the prefix ``"Cannot convert"``).
    """
    from collections.abc import Mapping  # noqa: PLC0415

    if _is_scalar(v):
        return _coerce_scalar(v), v
    if isinstance(v, AoT):
        msg = "live-attach of AoT is Phase 4"
        raise NotImplementedError(msg)
    if isinstance(v, Container) and not v._inline:  # noqa: SLF001
        msg = "live-attach of section Container is Phase 4"
        raise NotImplementedError(msg)
    # Mappings (incl. inline Container) → inline table.
    if isinstance(v, Mapping) or (isinstance(v, Container) and v._inline):  # noqa: SLF001
        return _synth_inline_table(
            v, layout_root=layout_root, parent=parent, path=path, owner=owner
        )
    # Lists (incl. Array views) → inline array.
    if isinstance(v, list):
        return _synth_inline_array(v, layout_root=layout_root, owner=owner)
    msg = f"Cannot convert {type(v).__name__} to a TOML value"
    raise TypeError(msg)


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
            leading=Trivia([WhitespaceNode(text=" ")]) if i > 0 else Trivia(),
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
    return val, table


def _synth_inline_array(
    items: list[Any],
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
        item = ArrayItem(
            leading=Trivia([WhitespaceNode(text=" ")]) if i > 0 else Trivia(),
            value=sub_cst,
            trailing=Trivia(),
            has_comma=not is_last,
            post_comma_trivia=Trivia(),
        )
        val.items.append(item)
        list.append(arr, sub_dec)
    return val, arr


__all__ = ["AoT", "Array", "Container", "Document", "Table", "TomlInput"]
