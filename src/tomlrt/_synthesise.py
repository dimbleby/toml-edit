"""Synthesise CST nodes from plain Python values.

Used by the mutation API to convert user-provided values into fresh
CST nodes when an entry is created or replaced. Containers backed by
an existing CST are deep-cloned so that assigning a value never aliases
mutable state across keys or documents; plain mappings are emitted as
inline tables.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from copy import deepcopy
from datetime import date, datetime, time
from typing import TYPE_CHECKING

from tomlrt._errors import TOMLError
from tomlrt._nodes import (
    ArrayItem,
    ArrayNode,
    BoolNode,
    DateTimeNode,
    FloatNode,
    InlineTableEntry,
    InlineTableNode,
    IntegerNode,
    Key,
    KeyPart,
    KeyValueNode,
    NewlineNode,
    StringNode,
    Trivia,
    WhitespaceNode,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tomlrt._document import TomlValue
    from tomlrt._nodes import DateLikeKind, ValueNode


_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _escape_basic_string(s: str) -> str:
    out: list[str] = []
    for ch in s:
        code = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\f":
            out.append("\\f")
        elif code < 0x20 or code == 0x7F:
            out.append(f"\\u{code:04X}")
        else:
            out.append(ch)
    return "".join(out)


def make_key_part(name: str) -> KeyPart:
    """Build a single :class:`KeyPart`, quoting if the name is not bare-safe."""
    if not isinstance(name, str):
        msg = (  # type: ignore[unreachable]
            f"TOML keys must be str, not {type(name).__name__}"
        )
        raise TypeError(msg)
    if name == "":
        # empty bare keys are forbidden; emit "" basic string
        return KeyPart(raw='""', value="", kind="basic")
    if _BARE_KEY_RE.match(name):
        return KeyPart(raw=name, value=name, kind="bare")
    raw = '"' + _escape_basic_string(name) + '"'
    return KeyPart(raw=raw, value=name, kind="basic")


def make_simple_key(name: str) -> Key:
    """Build a single-segment :class:`Key`."""
    return Key(parts=[make_key_part(name)], separators=[])


def _string_to_node(value: str) -> StringNode:
    return StringNode(
        raw='"' + _escape_basic_string(value) + '"',
        value=value,
        style="basic",
    )


def _bool_to_node(*, value: bool) -> BoolNode:
    return BoolNode(raw="true" if value else "false", value=value)


def _int_to_node(value: int) -> IntegerNode:
    return IntegerNode(raw=str(value), value=value, style="dec")


def _float_to_node(value: float) -> FloatNode:
    if math.isnan(value):
        raw = "nan"
    elif math.isinf(value):
        raw = "inf" if value > 0 else "-inf"
    else:
        raw = repr(value)
        if "." not in raw and "e" not in raw and "E" not in raw:
            raw += ".0"
    return FloatNode(raw=raw, value=value)


def _datetime_to_node(value: datetime | date | time) -> DateTimeNode:

    raw = value.isoformat()
    kind: str
    if isinstance(value, datetime):
        kind = "offset-datetime" if value.tzinfo is not None else "local-datetime"
    elif isinstance(value, date):
        kind = "local-date"
    else:
        kind = "local-time"
    from typing import cast  # noqa: PLC0415

    return DateTimeNode(raw=raw, value=value, kind=cast("DateLikeKind", kind))


def _list_to_array_node(items: Iterable[TomlValue]) -> ArrayNode:
    items_list = list(items)
    array_items: list[ArrayItem] = []
    n = len(items_list)
    for i, v in enumerate(items_list):
        is_last = i == n - 1
        array_items.append(
            ArrayItem(
                leading=Trivia(),
                value=value_to_node(v),
                trailing=Trivia(),
                has_comma=not is_last,
                post_comma_trivia=(
                    Trivia([WhitespaceNode(" ")]) if not is_last else Trivia()
                ),
            ),
        )
    return ArrayNode(items=array_items, final_trivia=Trivia())


def _mapping_to_inline_table_node(mapping: Mapping[str, TomlValue]) -> InlineTableNode:
    entries: list[InlineTableEntry] = []
    items_list = list(mapping.items())
    n = len(items_list)
    for i, (k, v) in enumerate(items_list):
        is_last = i == n - 1
        entries.append(
            InlineTableEntry(
                leading=Trivia([WhitespaceNode(" ")]) if i == 0 else Trivia(),
                key=make_simple_key(k),
                pre_eq=WhitespaceNode(" "),
                post_eq=WhitespaceNode(" "),
                value=value_to_node(v),
                trailing=Trivia(),
                has_comma=not is_last,
                post_comma_trivia=(
                    Trivia([WhitespaceNode(" ")]) if not is_last else Trivia()
                ),
            ),
        )
    final_trivia = Trivia([WhitespaceNode(" ")]) if entries else Trivia()
    return InlineTableNode(entries=entries, final_trivia=final_trivia)


def value_to_node(value: object) -> ValueNode:
    """Convert a logical value to a fresh :class:`ValueNode`.

    Containers already backed by a CST are deep-cloned so the new node
    is independent of the source. Any :class:`Mapping` (including
    :class:`dict` and ``MappingProxyType``) becomes an inline table;
    a plain :class:`list` becomes an inline array. Tuples and other
    non-list sequences are deliberately not accepted: ``Sequence``
    would also match ``str`` / ``bytes`` / ``range`` and so requires
    case-by-case carve-outs, while ``Mapping`` has no such landmines.
    Callers who want to assign a tuple should call ``list(...)`` first.
    """
    # Local import avoids a circular dependency with _document.
    from tomlrt._document import AoT, Array, Table  # noqa: PLC0415

    if isinstance(value, Array):
        if not value._attached:  # noqa: SLF001
            # Live attach: splice the user's node into the destination
            # so subsequent mutations through their reference flow into
            # the document.
            value._attached = True  # noqa: SLF001
            return value._node  # noqa: SLF001
        return deepcopy(value._node)  # noqa: SLF001
    if isinstance(value, AoT):
        msg = (
            "Cannot store an array-of-tables as an inline value; "
            "assign it at the table-key level so it can be emitted as "
            "[[ ... ]] sections."
        )
        raise TOMLError(msg)
    if isinstance(value, Table):
        node = getattr(value, "_node", None)
        if isinstance(node, InlineTableNode):
            if not value._attached:  # noqa: SLF001
                # Live attach: hand the user's node straight to the
                # destination so subsequent mutations through their
                # reference flow into the document. The receiving
                # container becomes its sole parent.
                value._attached = True  # noqa: SLF001
                return node
            return deepcopy(node)
        return _mapping_to_inline_table_node(dict(value))
    if isinstance(value, bool):
        return _bool_to_node(value=value)
    if isinstance(value, int):
        return _int_to_node(value)
    if isinstance(value, float):
        return _float_to_node(value)
    if isinstance(value, str):
        return _string_to_node(value)
    if isinstance(value, datetime):
        return _datetime_to_node(value)
    if isinstance(value, (date, time)):
        return _datetime_to_node(value)
    if isinstance(value, list):
        return _list_to_array_node(value)
    if isinstance(value, Mapping):
        return _mapping_to_inline_table_node(value)
    msg = f"Cannot convert value of type {type(value).__name__} to TOML"
    raise TypeError(msg)


def make_keyvalue_node(
    key_name: str,
    value: object,
    *,
    indent: str = "",
) -> KeyValueNode:
    """Build a fresh ``key = value\\n`` line."""
    leading = Trivia([WhitespaceNode(indent)]) if indent else Trivia()
    return KeyValueNode(
        leading=leading,
        key=make_simple_key(key_name),
        pre_eq=WhitespaceNode(" "),
        post_eq=WhitespaceNode(" "),
        value=value_to_node(value),
        trailing=None,
        trailing_comment=None,
        newline=NewlineNode("\n"),
    )


__all__ = [
    "make_key_part",
    "make_keyvalue_node",
    "make_simple_key",
    "value_to_node",
]
