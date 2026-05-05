"""Phase 3b invariant + inline-table mutation tests.

These tests pin the inline-table mutation primitives (replace,
append, delete) and check that the document invariants
(``check(doc)``) hold after each operation. The three tests in
``test_mutation.py`` (``test_inline_table_replace``,
``test_inline_table_append``,
``test_inline_table_delete_last_clears_trailing_comma``) cover the
golden-bytes expectations; this file covers the cache invariants
and a few additional shapes (dotted-inline mutation, empty inline
table, multiple appends).
"""

from __future__ import annotations

from tomlrt import loads
from tomlrt._invariants import check


def test_inline_replace_keeps_invariants() -> None:
    doc = loads("obj = { a = 1, b = 2 }\n")
    obj = doc.table("obj")
    obj["a"] = 99
    check(doc)
    assert obj["a"] == 99
    assert obj["b"] == 2


def test_inline_append_keeps_invariants() -> None:
    doc = loads("obj = { a = 1 }\n")
    obj = doc.table("obj")
    obj["b"] = 2
    obj["c"] = 3
    check(doc)
    assert obj["a"] == 1
    assert obj["b"] == 2
    assert obj["c"] == 3


def test_inline_append_into_empty_keeps_invariants() -> None:
    doc = loads("obj = {}\n")
    obj = doc.table("obj")
    obj["a"] = 1
    check(doc)
    assert obj["a"] == 1


def test_inline_delete_keeps_invariants() -> None:
    doc = loads("obj = { a = 1, b = 2, c = 3 }\n")
    obj = doc.table("obj")
    del obj["c"]
    check(doc)
    assert "c" not in obj
    assert dict(obj) == {"a": 1, "b": 2}


def test_inline_delete_then_append_round_trip() -> None:
    from tomlrt import dumps  # noqa: PLC0415

    doc = loads("obj = { a = 1, b = 2 }\n")
    obj = doc.table("obj")
    del obj["b"]
    obj["c"] = 3
    check(doc)
    out = dumps(doc)
    # Parses back to the right shape; exact spacing is not pinned.
    assert loads(out).table("obj").to_dict() == {"a": 1, "c": 3}


def test_inline_dotted_replace_keeps_invariants() -> None:
    doc = loads("obj = { a.b = 1, a.c = 2 }\n")
    obj = doc.table("obj")
    obj.table("a")["b"] = 99
    check(doc)
    assert obj.table("a")["b"] == 99
    assert obj.table("a")["c"] == 2


def test_inline_dotted_append_keeps_invariants() -> None:
    from tomlrt import dumps  # noqa: PLC0415

    doc = loads("obj = { a.b = 1 }\n")
    obj = doc.table("obj")
    obj.table("a")["c"] = 2
    check(doc)
    assert obj.table("a")["b"] == 1
    assert obj.table("a")["c"] == 2
    # And the raw entry was filed under the outermost inline table
    # with a dotted ``a.c`` key_parts list (validated by the
    # invariant checker's recursive inline-table shape walk).
    out = dumps(doc)
    assert loads(out).table("obj").table("a").to_dict() == {"b": 1, "c": 2}


def test_inline_replace_with_string() -> None:
    doc = loads('obj = { a = "old" }\n')
    obj = doc.table("obj")
    obj["a"] = "new"
    check(doc)
    assert obj["a"] == "new"


def test_inline_self_assign_is_noop() -> None:
    from tomlrt import dumps  # noqa: PLC0415

    src = "obj = { a = 1 }\n"
    doc = loads(src)
    obj = doc.table("obj")
    obj["a"] = obj["a"]
    assert dumps(doc) == src


def test_inline_delete_missing_raises_keyerror() -> None:
    import pytest  # noqa: PLC0415

    doc = loads("obj = { a = 1 }\n")
    obj = doc.table("obj")
    with pytest.raises(KeyError):
        del obj["missing"]


def test_inline_delete_dotted_prefix_removes_all_subentries() -> None:
    from tomlrt import dumps  # noqa: PLC0415

    doc = loads("obj = { a.b = 1, a.c = 2, d = 3 }\n")
    obj = doc.table("obj")
    del obj["a"]
    check(doc)
    assert "a" not in obj
    assert obj["d"] == 3
    assert loads(dumps(doc)).table("obj").to_dict() == {"d": 3}


def test_inline_delete_dotted_leaf_cleans_empty_prefix_container() -> None:
    from tomlrt import dumps  # noqa: PLC0415

    doc = loads("obj = { a.b = 1 }\n")
    obj = doc.table("obj")
    inner_a = obj.table("a")
    del inner_a["b"]
    check(doc)
    # Synthetic prefix container `a` is now empty and has no entry in
    # the backing InlineTableValue; outer dict view should drop it
    # to stay consistent.
    assert "a" not in obj
    assert loads(dumps(doc)).table("obj").to_dict() == {}


def test_inline_setitem_scalar_over_dotted_prefix_replaces() -> None:
    from tomlrt import dumps  # noqa: PLC0415

    doc = loads("obj = { a.b = 1 }\n")
    obj = doc.table("obj")
    obj["a"] = 2
    out = dumps(doc)
    assert out.startswith("obj = {")
    re_parsed = loads(out)
    assert re_parsed.table("obj")["a"] == 2


def test_inline_append_does_not_steal_eol_comment_in_multiline() -> None:
    from tomlrt import dumps  # noqa: PLC0415

    # TOML 1.1 multiline inline table with an inline comment on the
    # last entry's line. The comment must stay attached to `a`, not
    # migrate to the appended entry.
    src = "obj = { a = 1 # eol-on-a\n  }\n"
    doc = loads(src)
    obj = doc.table("obj")
    obj["b"] = 2
    check(doc)
    out = dumps(doc)
    # `# eol-on-a` must come before `b`, not after it.
    assert out.index("# eol-on-a") < out.index("b = 2")
    assert loads(out).table("obj").to_dict() == {"a": 1, "b": 2}
