"""Direct tests for `_Validator`, exercised on hand-built CST fragments.

These complement the parser-level tests by pinning the validator's
behaviour as a standalone unit: state transitions, message text, and
error messaging are checked without going through the parser. The
helpers build just enough of a CST to drive each rule; full
round-trip coverage continues to live in `test_basic.py` and
friends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tomlrt._errors import TOMLParseError
from tomlrt._nodes import (
    BoolNode,
    InlineTableEntry,
    InlineTableNode,
    IntegerNode,
    Key,
    KeyPart,
    KeyValueNode,
    TableHeaderNode,
    Trivia,
)
from tomlrt._validator import _Validator

if TYPE_CHECKING:
    from tomlrt._nodes import HeaderKind, ValueNode


def _key(*parts: str) -> Key:
    return Key([KeyPart(raw=p, value=p, kind="bare") for p in parts])


def _header(*parts: str, kind: HeaderKind = "table") -> TableHeaderNode:
    return TableHeaderNode(
        leading=Trivia(),
        kind=kind,
        inner_pre=None,
        key=_key(*parts),
        inner_post=None,
        trailing=None,
        trailing_comment=None,
        newline=None,
    )


def _kv(*parts: str, value: ValueNode | None = None) -> KeyValueNode:
    return KeyValueNode(
        leading=Trivia(),
        key=_key(*parts),
        pre_eq=None,
        post_eq=None,
        value=value if value is not None else IntegerNode("1", 1, "dec"),
        trailing=None,
        trailing_comment=None,
        newline=None,
    )


def _validator() -> _Validator:
    # The real parser hands ``_Scanner.error`` to the validator so
    # diagnostics carry source positions; for these unit tests we just
    # need the right call signature and a real ``TOMLParseError``.
    def _err(message: str, *, at: int) -> TOMLParseError:
        return TOMLParseError(message, line=1, col=1, offset=at)

    return _Validator(_err)


def _enter(v: _Validator, *parts: str, kind: HeaderKind = "table") -> None:
    v.enter_header(_header(*parts, kind=kind), at=0)


def _record(v: _Validator, *parts: str, value: ValueNode | None = None) -> None:
    v.record_keyvalue(_kv(*parts, value=value), at=0)


# --- header rules ----------------------------------------------------


def test_redefining_table_is_rejected() -> None:
    v = _validator()
    _enter(v, "a")
    with pytest.raises(TOMLParseError, match="redefinition of table 'a'"):
        _enter(v, "a")


def test_table_then_aot_is_rejected() -> None:
    v = _validator()
    _enter(v, "a")
    with pytest.raises(
        TOMLParseError, match="cannot redefine table 'a' as an array-of-tables"
    ):
        _enter(v, "a", kind="array")


def test_aot_then_table_is_rejected() -> None:
    v = _validator()
    _enter(v, "a", kind="array")
    with pytest.raises(
        TOMLParseError, match="cannot redefine array-of-tables 'a' as a normal table"
    ):
        _enter(v, "a")


def test_implicit_table_then_aot_is_rejected() -> None:
    # `[a.b]` makes `a` an implicit table; then `[[a]]` is invalid.
    v = _validator()
    _enter(v, "a", "b")
    with pytest.raises(TOMLParseError, match="already used as an implicit table"):
        _enter(v, "a", kind="array")


def test_value_path_blocks_later_table_header() -> None:
    v = _validator()
    _record(v, "a")
    with pytest.raises(TOMLParseError, match="already defined as a value"):
        _enter(v, "a")


def test_value_prefix_blocks_later_table_header() -> None:
    # `a = 1` then `[a.b]` — the prefix `a` is a value, can't extend it.
    v = _validator()
    _record(v, "a")
    with pytest.raises(TOMLParseError, match="cannot use 'a' as a table"):
        _enter(v, "a", "b")


# --- key/value rules -------------------------------------------------


def test_duplicate_keyvalue_is_rejected() -> None:
    v = _validator()
    _record(v, "x")
    with pytest.raises(TOMLParseError, match="duplicate key 'x'"):
        _record(v, "x")


def test_keyvalue_collides_with_explicit_subtable() -> None:
    # `[a.b]` then `[a]` then `b = 1` — the path `a.b` is already an
    # explicit table.
    v = _validator()
    _enter(v, "a", "b")
    _enter(v, "a")
    with pytest.raises(TOMLParseError, match=r"key 'a\.b' already defined as a table"):
        _record(v, "b")


def test_dotted_key_cannot_extend_explicit_table() -> None:
    # `[a.b]` then `[a]` then `b.c = 1` — the dotted intermediate
    # prefix `a.b` lands on the explicit table.
    v = _validator()
    _enter(v, "a", "b")
    _enter(v, "a")
    with pytest.raises(
        TOMLParseError, match="cannot extend explicitly-defined table"
    ):
        _record(v, "b", "c")


def test_dotted_key_cannot_extend_aot() -> None:
    v = _validator()
    _enter(v, "a", "b", kind="array")
    _enter(v, "a")
    with pytest.raises(TOMLParseError, match=r"cannot extend array-of-tables"):
        _record(v, "b", "c")


# --- AoT scope reset -------------------------------------------------


def test_aot_scope_reset_lets_subkeys_repeat_across_entries() -> None:
    # `[[H]] x = 1` then `[[H]] x = 2`: the second entry's `x` must
    # not collide with the first's.
    v = _validator()
    _enter(v, "h", kind="array")
    _record(v, "x")
    _enter(v, "h", kind="array")
    _record(v, "x")  # must succeed


def test_aot_scope_reset_lets_sub_headers_repeat_across_entries() -> None:
    # `[[H]] [H.sub]` then `[[H]] [H.sub]`: each entry has its own
    # sub-table.
    v = _validator()
    _enter(v, "h", kind="array")
    _enter(v, "h", "sub")
    _enter(v, "h", kind="array")
    _enter(v, "h", "sub")  # must succeed


def test_nested_aot_resets_too() -> None:
    v = _validator()
    _enter(v, "h", kind="array")
    _enter(v, "h", "inner", kind="array")
    _enter(v, "h", "inner", "leaf")
    _enter(v, "h", kind="array")
    # The nested AoT and its leaf are forgotten with the parent
    # entry; both must be re-creatable.
    _enter(v, "h", "inner", kind="array")
    _enter(v, "h", "inner", "leaf")


# --- inline tables ---------------------------------------------------


def test_check_inline_key_conflict_duplicate() -> None:
    v = _validator()
    seen: set[tuple[str, ...]] = set()
    seen_p: set[tuple[str, ...]] = set()
    v.check_inline_key_conflict(("x",), seen, seen_p, at=0)
    seen.add(("x",))
    with pytest.raises(TOMLParseError, match="duplicate key 'x' in inline table"):
        v.check_inline_key_conflict(("x",), seen, seen_p, at=0)


def test_check_inline_key_conflict_dotted_prefix() -> None:
    # `{ a.b = 1, a = 2 }` — second key's path `("a",)` lands on the
    # first key's recorded prefix `("a",)`.
    v = _validator()
    seen: set[tuple[str, ...]] = set()
    seen_p: set[tuple[str, ...]] = set()
    v.check_inline_key_conflict(("a", "b"), seen, seen_p, at=0)
    seen.add(("a", "b"))
    with pytest.raises(
        TOMLParseError, match=r"key 'a' in inline table conflicts with"
    ):
        v.check_inline_key_conflict(("a",), seen, seen_p, at=0)


def test_inline_table_value_blocks_later_header() -> None:
    # `t = { a = 1 }` exposes `t.a` to document-level tracking, so
    # `[t.a]` later must fail.
    inline = InlineTableNode(
        entries=[
            InlineTableEntry(
                leading=Trivia(),
                key=_key("a"),
                pre_eq=None,
                post_eq=None,
                value=BoolNode("true", value=True),
                trailing=Trivia(),
                has_comma=False,
                post_comma_trivia=Trivia(),
            ),
        ],
    )
    v = _validator()
    _record(v, "t", value=inline)
    with pytest.raises(TOMLParseError, match="cannot use 't' as a table"):
        _enter(v, "t", "a")
