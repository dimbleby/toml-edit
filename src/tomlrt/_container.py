"""Logical container layer.

`Container(dict)` is the dict-typed base for both `Document` (the
root) and `Table` (sections + inline tables). Phase 2 only needs the
read surface: dict-storage populated in doc-stream-first-occurrence
order, typed accessors, conversion helpers, and the `render()` entry
point. The mutation-time scaffolding (`_index`, `_refs`,
`_header_ref`, `_body_tail`, `_subtree_tail`) is deferred to Phase 3.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tomlrt._render import render
from tomlrt._slots import KVSlot
from tomlrt._trivia import Trivia
from tomlrt._values import (
    BoolValue,
    DateTimeValue,
    FloatValue,
    IntegerValue,
    StringValue,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomlrt._slots import AoTEntry, Slot, SlotRef
    from tomlrt._values import InlineTableValue


class Container(dict[str, Any]):
    """Dict-typed base for `Document` and `Table` views.

    Reads are pure dict operations. Mutation paths use the per-container
    cache (`_index` / `_refs` / `_header_ref` / `_body_tail` /
    `_subtree_tail`) maintained alongside the dict storage. For inline
    tables (`_inline=True`) the slot-stream caches stay `None` and
    `_value` points at the backing `InlineTableValue` instead — inline
    mutation lives in a separate code path (Phase 3b).
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
        "_subtree_tail",
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
        self._subtree_tail: Slot | None = None
        self._value: InlineTableValue | None = None

    # ------------------------------------------------------------------
    # Typed accessors
    # ------------------------------------------------------------------

    def table(self, key: str) -> Table:
        """Return ``self[key]`` typed as a `Table`.

        Raises ``KeyError`` if the key is missing and ``TypeError`` if
        the value is not a table.
        """
        v = self[key]
        if not isinstance(v, Table):
            msg = f"value at {key!r} is {type(v).__name__}, not Table"
            raise TypeError(msg)
        return v

    def array(self, key: str) -> Array:
        """Return ``self[key]`` typed as an inline `Array`."""
        v = self[key]
        if not isinstance(v, Array):
            msg = f"value at {key!r} is {type(v).__name__}, not Array"
            raise TypeError(msg)
        return v

    def aot(self, key: str) -> AoT:
        """Return ``self[key]`` typed as an array-of-tables (`AoT`)."""
        v = self[key]
        if not isinstance(v, AoT):
            msg = f"value at {key!r} is {type(v).__name__}, not AoT"
            raise TypeError(msg)
        return v

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
        if self._inline:
            self._inline_setitem(key, value)
            return
        if key in self:
            current = dict.__getitem__(self, key)
            if _is_scalar(current) and _is_scalar(value):
                refs = self._index.get(key)
                if not refs:
                    msg = (
                        f"internal: key {key!r} present in dict but missing from _index"
                    )
                    raise AssertionError(msg)
                primary = refs[0]
                slot = primary.slot
                if not isinstance(slot, KVSlot) or len(slot.key) != 1:
                    msg = (
                        "Phase 3a only supports scalar replacement of "
                        "a direct, non-dotted KV"
                    )
                    raise NotImplementedError(msg)
                slot.value = _coerce_scalar(value)
                dict.__setitem__(self, key, value)
                return
        msg = "non-scalar mutation arrives in Phase 3b/3c/3d"
        raise NotImplementedError(msg)

    @override
    def __delitem__(self, key: str) -> None:
        if self._inline:
            self._inline_delitem(key)
            return
        msg = "section delete arrives in Phase 3d"
        raise NotImplementedError(msg)

    # ------------------------------------------------------------------
    # Inline-table dispatch (Phase 3b)
    # ------------------------------------------------------------------

    def _inline_setitem(self, key: str, value: Any) -> None:
        from tomlrt import _inline_ops  # noqa: PLC0415

        if not _is_scalar(value):
            msg = (
                "Phase 3b inline-table mutation only supports scalar values; "
                "inline-typed and structural sources arrive in Phase 3c/3d"
            )
            raise NotImplementedError(msg)
        if key in self and isinstance(dict.__getitem__(self, key), Container):
            # Replacing a dotted-prefix sub-table (e.g. `a` in
            # `{a.b = 1}`) with a scalar would have to delete every
            # `a.*` entry and add an `a = scalar` entry. That's a
            # structural overwrite, deferred to a later phase.
            msg = (
                "scalar overwrite of a dotted-inline sub-table is not yet "
                "supported (Phase 3d structural overwrite)"
            )
            raise NotImplementedError(msg)
        coerced = _coerce_scalar(value)
        if key in self:
            ok = _inline_ops.replace_entry_value(self, key, coerced)
            if not ok:
                msg = (
                    f"internal: key {key!r} present on inline view but no "
                    "matching entry in the backing InlineTableValue"
                )
                raise AssertionError(msg)
        else:
            _inline_ops.append_entry(self, key, coerced)
        dict.__setitem__(self, key, value)

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


class Document(Container):
    """A parsed TOML document.

    Owns the physical slot stream (head/tail of the doubly-linked list,
    plus trailing trivia and detected newline). The dict-typed body is
    inherited from `Container`.
    """

    __slots__ = ("_head", "_newline", "_tail", "_trailing")

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        if data is not None:
            msg = "Document(data=...) is not supported in Phase 2"
            raise NotImplementedError(msg)
        self._head: Slot | None = None
        self._tail: Slot | None = None
        self._trailing: Trivia = Trivia()
        self._newline: str = "\n"
        self._layout_root = self

    def render(self) -> str:
        return render(self)


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


__all__ = ["AoT", "Array", "Container", "Document", "Table", "TomlInput"]
