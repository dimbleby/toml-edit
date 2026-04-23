"""Golden tests for editing operations.

Each test exercises a mutation pathway and asserts the *exact* rendered
output, so any regression in trivia handling is immediately visible.

Coverage targets:
- discontiguous tables (multiple physical sections for one logical table)
- dotted-key tables (logical table created via dotted keys)
- AoT middle/append/insert ops; AoT entries with nested sub-sections
- inline-table edits round-tripping
- cross-document assignment with deep-clone semantics
- mutation interaction with logical-view scoping for AoT entries
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import pytest

import tomlrt
from tomlrt import AoT, Array, Table


def _reparses(src: str) -> dict[str, Any]:
    return tomllib.loads(src)


# ---------------------------------------------------------------------------
# Discontiguous tables: [a] / [a.sub] / [b] / [a] is forbidden, but a
# logical table can still aggregate keys from the [a] header *and* from
# any [a.x] sub-section headers.
# ---------------------------------------------------------------------------


def test_table_with_sub_section_iter_includes_subtable() -> None:
    src = "[a]\nx = 1\n[a.sub]\ny = 2\n"
    doc = tomlrt.parse(src)
    a = doc["a"]
    assert isinstance(a, tomlrt.Table)
    assert a["x"] == 1
    sub = a["sub"]
    assert isinstance(sub, tomlrt.Table)
    assert sub["y"] == 2


def test_table_with_sub_section_modify_subtable_value() -> None:
    src = "[a]\nx = 1\n[a.sub]\ny = 2\n"
    doc = tomlrt.parse(src)
    a = doc["a"]
    assert isinstance(a, tomlrt.Table)
    sub = a["sub"]
    assert isinstance(sub, tomlrt.Table)
    sub["y"] = 99
    out = tomlrt.dumps(doc)
    assert out == "[a]\nx = 1\n[a.sub]\ny = 99\n"


def test_table_with_sub_section_add_to_parent_appends_in_parent_block() -> None:
    src = "[a]\nx = 1\n[a.sub]\ny = 2\n"
    doc = tomlrt.parse(src)
    a = doc["a"]
    assert isinstance(a, tomlrt.Table)
    a["z"] = 3
    out = tomlrt.dumps(doc)
    # New parent-level key must land in the [a] block, BEFORE [a.sub] —
    # putting it after would make it semantically belong to [a.sub] under
    # TOML's "headers terminate a section" rule.
    assert out == "[a]\nx = 1\nz = 3\n[a.sub]\ny = 2\n"
    assert _reparses(out) == {"a": {"x": 1, "z": 3, "sub": {"y": 2}}}


# ---------------------------------------------------------------------------
# Dotted-key tables (logical table only ever lives as dotted keys)
# ---------------------------------------------------------------------------


def test_dotted_key_table_read() -> None:
    src = "a.b = 1\na.c = 2\n"
    doc = tomlrt.parse(src)
    a = doc["a"]
    assert isinstance(a, tomlrt.Table)
    assert dict(a) == {"b": 1, "c": 2}


def test_dotted_key_table_set_via_subtable_adds_dotted_entry() -> None:
    """Setting a new key on a dotted-only table appends a new dotted KV."""
    src = "a.b = 1\na.c = 2\n"
    doc = tomlrt.parse(src)
    a = doc["a"]
    assert isinstance(a, tomlrt.Table)
    a["d"] = 3
    assert dict(a) == {"b": 1, "c": 2, "d": 3}
    assert tomlrt.dumps(doc) == "a.b = 1\na.c = 2\na.d = 3\n"


def test_dotted_key_table_overwrite_via_subtable() -> None:
    src = "a.b = 1\na.c = 2\n"
    doc = tomlrt.parse(src)
    a = doc["a"]
    assert isinstance(a, tomlrt.Table)
    a["b"] = 99
    assert dict(a) == {"c": 2, "b": 99}


def test_dotted_key_table_delete_via_subtable() -> None:
    src = "a.b = 1\na.c = 2\na.d = 3\n"
    doc = tomlrt.parse(src)
    a = doc["a"]
    assert isinstance(a, tomlrt.Table)
    del a["c"]
    assert dict(a) == {"b": 1, "d": 3}
    assert "c" not in tomlrt.dumps(doc)


def test_dotted_key_table_delete_missing_raises() -> None:
    src = "a.b = 1\n"
    doc = tomlrt.parse(src)
    a = doc["a"]
    assert isinstance(a, tomlrt.Table)
    with pytest.raises(KeyError):
        del a["nope"]


def test_dotted_key_table_set_overwrites_subtree() -> None:
    src = "a.b.x = 1\na.b.y = 2\na.c = 3\n"
    doc = tomlrt.parse(src)
    a = doc["a"]
    assert isinstance(a, tomlrt.Table)
    a["b"] = 99
    assert dict(a) == {"c": 3, "b": 99}


def test_dotted_key_nested_subtable_set() -> None:
    """Setting a key on a deeply-nested dotted view works too."""
    src = "a.b.x = 1\na.b.y = 2\n"
    doc = tomlrt.parse(src)
    a = doc["a"]
    assert isinstance(a, tomlrt.Table)
    b = a["b"]
    assert isinstance(b, tomlrt.Table)
    b["z"] = 3
    assert dict(b) == {"x": 1, "y": 2, "z": 3}
    assert tomlrt.dumps(doc) == "a.b.x = 1\na.b.y = 2\na.b.z = 3\n"


def test_inline_dotted_subtable_set() -> None:
    """Same thing inside an inline table."""
    src = "t = { a.b = 1, a.c = 2 }\n"
    doc = tomlrt.parse(src)
    t = doc["t"]
    assert isinstance(t, tomlrt.Table)
    a = t["a"]
    assert isinstance(a, tomlrt.Table)
    a["d"] = 3
    assert dict(a) == {"b": 1, "c": 2, "d": 3}
    assert tomlrt.dumps(doc) == "t = { a.b = 1, a.c = 2, a.d = 3 }\n"


def test_inline_dotted_subtable_delete() -> None:
    src = "t = { a.b = 1, a.c = 2 }\n"
    doc = tomlrt.parse(src)
    t = doc["t"]
    assert isinstance(t, tomlrt.Table)
    a = t["a"]
    assert isinstance(a, tomlrt.Table)
    del a["b"]
    assert dict(a) == {"c": 2}


# ---------------------------------------------------------------------------
# Arrays-of-tables (AoT) — middle ops and entries with sub-sections
# ---------------------------------------------------------------------------


def test_aot_basic_iteration() -> None:
    src = '[[users]]\nname = "alice"\n[[users]]\nname = "bob"\n'
    doc = tomlrt.parse(src)
    users = doc["users"]
    assert isinstance(users, tomlrt.AoT)
    assert [u["name"] for u in users] == ["alice", "bob"]


def test_aot_append_entry_via_dict() -> None:
    src = '[[users]]\nname = "alice"\n'
    doc = tomlrt.parse(src)
    users = doc["users"]
    assert isinstance(users, tomlrt.AoT)
    users.append({"name": "bob"})
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"users": [{"name": "alice"}, {"name": "bob"}]}


def test_aot_modify_field_in_first_entry() -> None:
    src = '[[users]]\nname = "alice"\n[[users]]\nname = "bob"\n'
    doc = tomlrt.parse(src)
    users = doc["users"]
    assert isinstance(users, tomlrt.AoT)
    users[0]["name"] = "ALICE"
    out = tomlrt.dumps(doc)
    assert out == '[[users]]\nname = "ALICE"\n[[users]]\nname = "bob"\n'


def test_aot_modify_field_in_middle_entry() -> None:
    src = '[[users]]\nname = "a"\n[[users]]\nname = "b"\n[[users]]\nname = "c"\n'
    doc = tomlrt.parse(src)
    users = doc["users"]
    assert isinstance(users, tomlrt.AoT)
    users[1]["name"] = "B"
    out = tomlrt.dumps(doc)
    assert (
        out == '[[users]]\nname = "a"\n[[users]]\nname = "B"\n[[users]]\nname = "c"\n'
    )


def test_aot_entry_sub_section_read() -> None:
    """[[arr]] / [arr.sub] — sub belongs to the AoT entry."""
    src = "[[arr]]\nx = 1\n[arr.sub]\ny = 2\n[[arr]]\nx = 10\n[arr.sub]\ny = 20\n"
    doc = tomlrt.parse(src)
    arr = doc["arr"]
    assert isinstance(arr, tomlrt.AoT)
    assert len(arr) == 2
    assert arr[0]["x"] == 1
    sub0 = arr[0]["sub"]
    assert isinstance(sub0, tomlrt.Table)
    assert sub0["y"] == 2
    assert arr[1]["x"] == 10
    sub1 = arr[1]["sub"]
    assert isinstance(sub1, tomlrt.Table)
    assert sub1["y"] == 20


def test_aot_entry_sub_section_modify_value() -> None:
    src = "[[arr]]\nx = 1\n[arr.sub]\ny = 2\n[[arr]]\nx = 10\n[arr.sub]\ny = 20\n"
    doc = tomlrt.parse(src)
    arr = doc["arr"]
    assert isinstance(arr, tomlrt.AoT)
    sub = arr[1]["sub"]
    assert isinstance(sub, tomlrt.Table)
    sub["y"] = 999
    out = tomlrt.dumps(doc)
    assert out == (
        "[[arr]]\nx = 1\n[arr.sub]\ny = 2\n[[arr]]\nx = 10\n[arr.sub]\ny = 999\n"
    )
    assert _reparses(out) == {
        "arr": [
            {"x": 1, "sub": {"y": 2}},
            {"x": 10, "sub": {"y": 999}},
        ]
    }


# ---------------------------------------------------------------------------
# Inline tables and arrays — round-trip edits
# ---------------------------------------------------------------------------


def test_inline_table_modify_preserves_spacing() -> None:
    src = "owner = { name = 'tom', dob = 1979 }\n"
    doc = tomlrt.parse(src)
    owner = doc["owner"]
    assert isinstance(owner, tomlrt.Table)
    owner["name"] = "tim"
    out = tomlrt.dumps(doc)
    # Style of the replaced scalar regenerates as basic-quoted (default),
    # but surrounding spacing/comma trivia is preserved.
    assert out == 'owner = { name = "tim", dob = 1979 }\n'


def test_inline_array_modify_preserves_brackets() -> None:
    src = "ports = [ 80, 443, 8080 ]\n"
    doc = tomlrt.parse(src)
    ports = doc["ports"]
    assert isinstance(ports, tomlrt.Array)
    ports[1] = 444
    out = tomlrt.dumps(doc)
    assert out == "ports = [ 80, 444, 8080 ]\n"


def test_array_insert_then_pop_round_trips() -> None:
    src = "ports = [80, 443]\n"
    doc = tomlrt.parse(src)
    ports = doc["ports"]
    assert isinstance(ports, tomlrt.Array)
    ports.insert(1, 8080)
    assert list(ports) == [80, 8080, 443]
    ports.pop(1)
    out = tomlrt.dumps(doc)
    assert out == "ports = [80, 443]\n"


# ---------------------------------------------------------------------------
# Cross-document assignment — must deep-clone, never share state
# ---------------------------------------------------------------------------


def test_cross_doc_table_assign_deep_clones() -> None:
    src1 = '[srv]\nhost = "a.example"\nport = 80\n'
    src2 = ""
    a = tomlrt.parse(src1)
    b = tomlrt.parse(src2)
    b["srv"] = a["srv"]
    # Mutating `a` must not affect `b`.
    a_srv = a["srv"]
    assert isinstance(a_srv, tomlrt.Table)
    a_srv["port"] = 9999
    out_a = tomlrt.dumps(a)
    out_b = tomlrt.dumps(b)
    assert _reparses(out_a) == {"srv": {"host": "a.example", "port": 9999}}
    assert _reparses(out_b) == {"srv": {"host": "a.example", "port": 80}}


def test_cross_doc_aot_assign_deep_clones() -> None:
    src1 = '[[users]]\nname = "alice"\n[[users]]\nname = "bob"\n'
    src2 = ""
    a = tomlrt.parse(src1)
    b = tomlrt.parse(src2)
    b["users"] = a["users"]
    a_users = a["users"]
    assert isinstance(a_users, tomlrt.AoT)
    a_users[0]["name"] = "MUT"
    assert _reparses(tomlrt.dumps(a))["users"][0]["name"] == "MUT"
    assert _reparses(tomlrt.dumps(b))["users"][0]["name"] == "alice"


def test_cross_doc_array_assign_deep_clones() -> None:
    src1 = "ports = [80, 443]\n"
    src2 = ""
    a = tomlrt.parse(src1)
    b = tomlrt.parse(src2)
    b["ports"] = a["ports"]
    a_ports = a["ports"]
    assert isinstance(a_ports, tomlrt.Array)
    a_ports.append(8080)
    assert _reparses(tomlrt.dumps(a))["ports"] == [80, 443, 8080]
    assert _reparses(tomlrt.dumps(b))["ports"] == [80, 443]


def test_install_attached_aot_preserves_comments() -> None:
    # `install` and `__setitem__` should both deep-clone the source CST
    # when given an attached AoT from another document. The previous
    # `install` implementation always routed through `to_dict()`, which
    # silently stripped comments and formatting, diverging from the
    # subscript path.
    src = "[[t]]\n# leading\na = 1  # eol\n[[t]]\nb = 2\n"
    a = tomlrt.parse(src)
    b = tomlrt.parse("")
    b.install("y", a["t"])
    assert tomlrt.dumps(b) == ("[[y]]\n# leading\na = 1  # eol\n[[y]]\nb = 2\n")


def test_install_attached_aot_at_dotted_path_preserves_comments() -> None:
    src = "[[t]]\n# leading\na = 1\n"
    a = tomlrt.parse(src)
    b = tomlrt.parse("")
    b.install("p.q", a["t"])
    assert tomlrt.dumps(b) == "[[p.q]]\n# leading\na = 1\n"


def test_install_attached_aot_is_independent_of_source() -> None:
    src = '[[t]]\nname = "alice"\n'
    a = tomlrt.parse(src)
    b = tomlrt.parse("")
    b.install("y", a["t"])
    a["t"][0]["name"] = "MUT"
    assert _reparses(tomlrt.dumps(b))["y"][0]["name"] == "alice"


# ---------------------------------------------------------------------------
# Cross-section conflict on mutation
# ---------------------------------------------------------------------------


def test_set_value_overwriting_existing_subsection() -> None:
    """Assigning a scalar to a name that's currently a sub-table.

    Matches plain-dict semantics: the [a.b] section (and anything nested
    under it) is silently removed and replaced with ``b = 99`` inside
    ``[a]``.
    """
    src = "[a]\nx = 1\n[a.b]\ny = 2\n[a.b.c]\nz = 3\n"
    doc = tomlrt.parse(src)
    a = doc["a"]
    assert isinstance(a, tomlrt.Table)
    a["b"] = 99
    out = tomlrt.dumps(doc)
    assert out == "[a]\nx = 1\nb = 99\n"
    assert _reparses(out) == {"a": {"x": 1, "b": 99}}


def test_set_value_overwriting_existing_aot() -> None:
    src = '[a]\nx = 1\n[[a.items]]\nname = "first"\n[[a.items]]\nname = "second"\n'
    doc = tomlrt.parse(src)
    a = doc["a"]
    assert isinstance(a, tomlrt.Table)
    a["items"] = 5
    out = tomlrt.dumps(doc)
    assert out == "[a]\nx = 1\nitems = 5\n"
    assert _reparses(out) == {"a": {"x": 1, "items": 5}}


def test_set_value_overwriting_dotted_subtree() -> None:
    src = "[a]\nb.c = 1\nb.d = 2\nx = 9\n"
    doc = tomlrt.parse(src)
    a = doc["a"]
    assert isinstance(a, tomlrt.Table)
    a["b"] = 99
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"a": {"x": 9, "b": 99}}


def test_set_value_overwriting_top_level_table() -> None:
    src = "[a]\nx = 1\n[b]\ny = 2\n"
    doc = tomlrt.parse(src)
    doc["a"] = 99
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"a": 99, "b": {"y": 2}}


def test_del_subtable() -> None:
    src = "[a]\nx = 1\n[a.b]\ny = 2\n[a.b.c]\nz = 3\n[other]\nq = 1\n"
    doc = tomlrt.parse(src)
    a = doc["a"]
    assert isinstance(a, tomlrt.Table)
    del a["b"]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"a": {"x": 1}, "other": {"q": 1}}


def test_del_aot() -> None:
    src = '[a]\nx = 1\n[[a.items]]\nname = "first"\n[[a.items]]\nname = "second"\n'
    doc = tomlrt.parse(src)
    a = doc["a"]
    assert isinstance(a, tomlrt.Table)
    del a["items"]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"a": {"x": 1}}


def test_del_dotted_subtree() -> None:
    src = "[a]\nb.c = 1\nb.d = 2\nx = 9\n"
    doc = tomlrt.parse(src)
    a = doc["a"]
    assert isinstance(a, tomlrt.Table)
    del a["b"]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"a": {"x": 9}}


def test_del_missing_raises_keyerror() -> None:
    doc = tomlrt.parse("[a]\nx = 1\n")
    a = doc["a"]
    assert isinstance(a, tomlrt.Table)
    with pytest.raises(KeyError):
        del a["missing"]


def test_pop_returns_subtable_snapshot() -> None:
    src = "[a]\nx = 1\n[a.b]\ny = 2\n[a.b.c]\nz = 3\n"
    doc = tomlrt.parse(src)
    a = doc["a"]
    assert isinstance(a, tomlrt.Table)
    popped = a.pop("b")
    assert popped == {"y": 2, "c": {"z": 3}}
    assert _reparses(tomlrt.dumps(doc)) == {"a": {"x": 1}}


def test_pop_returns_aot_snapshot() -> None:
    doc = tomlrt.parse('[[items]]\nname = "a"\n[[items]]\nname = "b"\n')
    popped = doc.pop("items")
    assert popped == [{"name": "a"}, {"name": "b"}]
    assert tomlrt.dumps(doc) == ""


def test_pop_with_default() -> None:
    doc = tomlrt.parse("")
    assert doc.pop("missing", "fallback") == "fallback"
    with pytest.raises(KeyError):
        doc.pop("missing")


def test_popitem_is_lifo() -> None:
    doc = tomlrt.parse("a = 1\nb = 2\nc = 3\n")
    assert doc.popitem() == ("c", 3)
    assert doc.popitem() == ("b", 2)
    assert _reparses(tomlrt.dumps(doc)) == {"a": 1}


def test_popitem_empty_raises() -> None:
    doc = tomlrt.parse("")
    with pytest.raises(KeyError):
        doc.popitem()


def test_setitem_into_implicit_parent() -> None:
    """Adding a new key to an implicit-only parent materialises [a]."""
    src = "[a.b]\ny = 2\n"
    doc = tomlrt.parse(src)
    a = doc["a"]
    assert isinstance(a, tomlrt.Table)
    a["new"] = 1
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"a": {"new": 1, "b": {"y": 2}}}


def test_setitem_into_implicit_grandparent() -> None:
    src = "[a.b.c]\nz = 3\n"
    doc = tomlrt.parse(src)
    ab = doc.table("a").table("b")
    ab["new"] = 1
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"a": {"b": {"new": 1, "c": {"z": 3}}}}


def test_inline_table_setitem_overwrites_dotted_group() -> None:
    src = 'config = { server.host = "x", server.port = 80, name = "y" }\n'
    doc = tomlrt.parse(src)
    config = doc.table("config")
    config["server"] = "newval"
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"config": {"name": "y", "server": "newval"}}


def test_inline_table_delitem_removes_dotted_group() -> None:
    src = 'config = { server.host = "x", server.port = 80, name = "y" }\n'
    doc = tomlrt.parse(src)
    config = doc.table("config")
    del config["server"]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"config": {"name": "y"}}


def test_inline_table_delitem_missing_raises_keyerror() -> None:
    doc = tomlrt.parse("config = { a = 1 }\n")
    config = doc.table("config")
    with pytest.raises(KeyError):
        del config["missing"]


# ---------------------------------------------------------------------------
# Sub-table access through AoT entry (uses the new owned_scope path)
# ---------------------------------------------------------------------------


def test_aot_entry_owned_scope_isolates_sibling_sub_sections() -> None:
    """[[arr]] / [arr.sub] / x=1 / [[arr]] / [arr.sub] / x=2

    Each entry's sub.x must be independent; mutating arr[0].sub.x must
    not affect arr[1].sub.x.
    """
    src = "[[arr]]\n[arr.sub]\nx = 1\n[[arr]]\n[arr.sub]\nx = 2\n"
    doc = tomlrt.parse(src)
    arr = doc["arr"]
    assert isinstance(arr, tomlrt.AoT)
    s0 = arr[0]["sub"]
    s1 = arr[1]["sub"]
    assert isinstance(s0, tomlrt.Table)
    assert isinstance(s1, tomlrt.Table)
    assert s0["x"] == 1
    assert s1["x"] == 2
    s0["x"] = 100
    assert s1["x"] == 2  # unchanged
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"arr": [{"sub": {"x": 100}}, {"sub": {"x": 2}}]}


# ---------------------------------------------------------------------------
# Array.set_multiline / multiline property
# ---------------------------------------------------------------------------


def test_array_set_multiline_true_wraps_with_default_indent() -> None:
    doc = tomlrt.loads("a = [1, 2, 3]\n")
    arr = doc.array("a")
    assert not arr.multiline
    arr.set_multiline(multiline=True)
    assert tomlrt.dumps(doc) == "a = [\n    1,\n    2,\n    3,\n]\n"
    assert arr.multiline


def test_array_set_multiline_false_collapses() -> None:
    doc = tomlrt.loads("a = [\n    1,\n    2,\n    3,\n]\n")
    arr = doc.array("a")
    assert arr.multiline
    arr.set_multiline(multiline=False)
    assert tomlrt.dumps(doc) == "a = [1, 2, 3]\n"
    assert not arr.multiline


def test_array_set_multiline_custom_indent() -> None:
    doc = tomlrt.loads("a = [1, 2]\n")
    doc.array("a").set_multiline(multiline=True, indent="  ")
    assert tomlrt.dumps(doc) == "a = [\n  1,\n  2,\n]\n"


def test_array_multiline_property_setter() -> None:
    doc = tomlrt.loads("a = [1, 2]\n")
    arr = doc.array("a")
    arr.multiline = True
    assert tomlrt.dumps(doc) == "a = [\n    1,\n    2,\n]\n"
    arr.multiline = False
    assert tomlrt.dumps(doc) == "a = [1, 2]\n"


def test_array_set_multiline_then_append() -> None:
    doc = tomlrt.loads("a = [1]\n")
    arr = doc.array("a")
    arr.set_multiline(multiline=True)
    arr.append(2)
    assert tomlrt.dumps(doc) == "a = [\n    1,\n    2,\n]\n"


def test_array_set_multiline_returns_self() -> None:
    doc = tomlrt.loads("a = [1]\n")
    arr = doc.array("a")
    assert arr.set_multiline(multiline=True) is arr


def test_array_set_multiline_survives_view_refetch() -> None:
    doc = tomlrt.loads("a = []\n")
    doc.array("a").set_multiline(multiline=True)
    refetched = doc.array("a")
    assert refetched.multiline
    refetched.append(1)
    assert tomlrt.dumps(doc) == "a = [\n    1,\n]\n"


def test_array_set_multiline_indent_preserved_on_install() -> None:
    # Calling set_multiline(indent=...) on a standalone Array and then
    # installing it should honour the requested indent, not silently
    # revert to the indent passed to the Array constructor.
    arr = tomlrt.Array([1, 2, 3])
    arr.set_multiline(multiline=True, indent="  ")
    doc = tomlrt.document()
    doc["x"] = arr
    assert tomlrt.dumps(doc) == "x = [\n  1,\n  2,\n  3,\n]\n"


def test_append_to_multiline_array_with_eol_comments() -> None:
    # When every existing item carries an inline comment, the
    # separator-style sampler used to give up and fall back to
    # ", " for the inter-item separator, and to drag the last
    # item's trailing comment into the close-pad. Newly appended
    # items must instead inherit the structural indent and leave
    # the existing comments alone.
    src = "a = [\n    1,  # one\n    2,  # two\n]\n"
    doc = tomlrt.loads(src)
    doc.array("a").append(3)
    assert tomlrt.dumps(doc) == ("a = [\n    1,  # one\n    2,  # two\n    3,\n]\n")


def test_array_parsed_empty_with_newline_is_multiline() -> None:
    doc = tomlrt.loads("a = [\n]\n")
    arr = doc.array("a")
    assert arr.multiline
    arr.append(1)
    assert tomlrt.dumps(doc) == "a = [\n    1,\n]\n"


def test_array_parsed_empty_with_newline_indent_is_inferred() -> None:
    doc = tomlrt.loads("a = [\n  ]\n")
    arr = doc.array("a")
    arr.append(1)
    assert tomlrt.dumps(doc) == "a = [\n  1,\n]\n"


def test_append_preserves_empty_array_inner_comment() -> None:
    # An empty multiline array with only a comment inside used to lose
    # the comment entirely on first append. The comment should survive
    # as leading trivia of the newly inserted first item.
    src = "a = [\n    # placeholder\n]\n"
    doc = tomlrt.loads(src)
    doc.array("a").append(1)
    assert tomlrt.dumps(doc) == "a = [\n    # placeholder\n    1,\n]\n"


def test_append_preserves_trailing_comment_in_single_item_array() -> None:
    # A single-item multiline array whose last-item post-comma slot
    # carries a comment used to have that comment collapse onto the
    # same line as the new item, producing valid-but-ugly output.
    src = "a = [\n    1,\n    # tail\n]\n"
    doc = tomlrt.loads(src)
    doc.array("a").append(2)
    assert tomlrt.dumps(doc) == "a = [\n    1,\n    # tail\n    2,\n]\n"


def test_append_preserves_leading_comment_in_single_item_array() -> None:
    # A single-item multiline array with a leading comment used to
    # collapse to single-line layout on append because the inter-item
    # separator could not be sampled from items[:-1] (which is empty).
    src = "a = [\n    # head\n    1,\n]\n"
    doc = tomlrt.loads(src)
    doc.array("a").append(2)
    assert tomlrt.dumps(doc) == "a = [\n    # head\n    1,\n    2,\n]\n"


# ---------------------------------------------------------------------------
# Table.set_aot / Table.promote_array
# ---------------------------------------------------------------------------


def test_set_aot_creates_repeated_headers() -> None:
    doc = tomlrt.loads("")
    doc["packages"] = AoT([{"name": "a", "version": "1.0"}, {"name": "b"}])
    aot = doc["packages"]
    assert isinstance(aot, tomlrt.AoT)
    assert len(aot) == 2
    assert "[[packages]]" in tomlrt.dumps(doc)


def test_set_aot_with_no_entries_returns_appendable_view() -> None:
    doc = tomlrt.loads("")
    doc["servers"] = AoT()
    aot = doc["servers"]
    assert tomlrt.dumps(doc) == ""
    aot.append({"host": "localhost"})
    rendered = tomlrt.dumps(doc)
    assert "[[servers]]" in rendered
    assert 'host = "localhost"' in rendered


def test_set_aot_overwrites_existing_key() -> None:
    doc = tomlrt.loads("foo = 1\n")
    doc["foo"] = AoT([{"x": 1}])
    rendered = tomlrt.dumps(doc)
    assert "foo = 1" not in rendered
    assert "[[foo]]" in rendered


def test_set_aot_nested_path() -> None:
    doc = tomlrt.loads("[product]\n")
    doc.table("product")["variant"] = AoT([{"sku": "X"}])
    rendered = tomlrt.dumps(doc)
    assert "[[product.variant]]" in rendered
    assert 'sku = "X"' in rendered
    assert isinstance(doc.table("product").aot("variant"), tomlrt.AoT)


def test_set_aot_blank_separated_entries() -> None:
    doc = tomlrt.loads("")
    doc["p"] = AoT([{"x": 1}, {"x": 2}, {"x": 3}])
    assert tomlrt.dumps(doc) == ("[[p]]\nx = 1\n\n[[p]]\nx = 2\n\n[[p]]\nx = 3\n")


def test_set_aot_blank_before_first_when_preceded_by_content() -> None:
    doc = tomlrt.loads("top = 1\n")
    doc["p"] = AoT([{"x": 1}])
    assert tomlrt.dumps(doc) == "top = 1\n\n[[p]]\nx = 1\n"


def test_set_aot_blank_after_section_header() -> None:
    doc = tomlrt.loads("[product]\n")
    doc.table("product")["variant"] = AoT([{"sku": "X"}])
    assert tomlrt.dumps(doc) == ('[product]\n\n[[product.variant]]\nsku = "X"\n')


def test_promote_array_converts_inline_to_aot() -> None:
    doc = tomlrt.loads('packages = [{name = "a"}, {name = "b"}]\n')
    aot = doc.promote_array("packages")
    assert isinstance(aot, tomlrt.AoT)
    assert len(aot) == 2
    rendered = tomlrt.dumps(doc)
    assert "packages = [" not in rendered
    assert rendered.count("[[packages]]") == 2
    assert isinstance(doc.aot("packages"), tomlrt.AoT)


def test_promote_array_rejects_empty_array() -> None:
    doc = tomlrt.loads("a = []\n")
    with pytest.raises(tomlrt.TOMLError, match="empty array"):
        doc.promote_array("a")


def test_promote_array_rejects_non_table_elements() -> None:
    doc = tomlrt.loads("a = [1, 2]\n")
    with pytest.raises(tomlrt.TOMLError, match="non-inline-table"):
        doc.promote_array("a")


def test_promote_array_rejects_non_array() -> None:
    doc = tomlrt.loads("a = 1\n")
    with pytest.raises(tomlrt.TOMLError, match="not an array"):
        doc.promote_array("a")


def test_promote_inline_rejects_non_inline_for_present_keys() -> None:
    """When ``key`` is present but isn't a simple inline-table KV, the
    user should see a clear "nothing to promote" message, not a bare
    ``KeyError`` that contradicts ``key in self``.
    """
    for src, target in [
        ("[a]\nb.c = 1\n", "b"),  # dotted-key subtable
        ("[a.b]\nc = 1\n", "b"),  # subsection
        ("[[a.b]]\nc = 1\n", "b"),  # array-of-tables
    ]:
        doc = tomlrt.loads(src)
        assert target in doc["a"]
        with pytest.raises(tomlrt.TOMLError, match="not an inline table"):
            doc["a"].promote_inline(target)


def test_promote_array_rejects_non_array_for_present_keys() -> None:
    for src, target in [
        ("[a]\nb.c = 1\n", "b"),
        ("[a.b]\nc = 1\n", "b"),
        ("[[a.b]]\nc = 1\n", "b"),
    ]:
        doc = tomlrt.loads(src)
        assert target in doc["a"]
        with pytest.raises(tomlrt.TOMLError, match="not an array"):
            doc["a"].promote_array(target)


# ---------------------------------------------------------------------------
# Table.set_table / Table.ensure_table / dotted-path navigation
# ---------------------------------------------------------------------------


def test_set_table_creates_section_directly() -> None:
    doc = tomlrt.loads("")
    doc["tool"] = Table.section({"name": "x"})
    t = doc["tool"]
    assert isinstance(t, tomlrt.Table)
    assert tomlrt.dumps(doc) == '[tool]\nname = "x"\n'


def test_set_table_dotted_omits_super_table_headers() -> None:
    doc = tomlrt.loads("")
    doc.install("tool.poetry", Table.section({"name": "x", "version": "0.1"}))
    rendered = tomlrt.dumps(doc)
    assert "[tool]\n" not in rendered
    assert rendered == '[tool.poetry]\nname = "x"\nversion = "0.1"\n'


def test_set_table_implicit_super_table_navigable() -> None:
    doc = tomlrt.loads("")
    doc.install("tool.poetry", Table.section({"name": "x"}))
    assert doc.table("tool").table("poetry")["name"] == "x"
    tool = doc["tool"]
    assert isinstance(tool, tomlrt.Table)
    poetry = tool["poetry"]
    assert isinstance(poetry, tomlrt.Table)
    assert poetry["name"] == "x"
    assert doc.table("tool.poetry")["name"] == "x"


def test_set_table_sibling_section_does_not_disturb_existing() -> None:
    doc = tomlrt.loads("")
    doc.install("tool.poetry", Table.section({"name": "x"}))
    doc.install("tool.poetry.dependencies", Table.section({"requests": "^2.0"}))
    assert tomlrt.dumps(doc) == (
        '[tool.poetry]\nname = "x"\n\n[tool.poetry.dependencies]\nrequests = "^2.0"\n'
    )


def test_set_table_replaces_existing_section_and_purges_children() -> None:
    doc = tomlrt.loads('[tool.poetry]\nname = "x"\n[tool.poetry.foo]\nbar = 1\n')
    doc.install("tool.poetry", Table.section({"version": "2.0"}))
    rendered = tomlrt.dumps(doc)
    assert "name" not in rendered
    assert "[tool.poetry.foo]" not in rendered
    assert rendered == '[tool.poetry]\nversion = "2.0"\n'


def test_set_table_overwrites_inline_value() -> None:
    doc = tomlrt.loads('tool = {poetry = {name = "x"}}\n')
    doc.install("tool.poetry", Table.section({"version": "2.0"}))
    rendered = tomlrt.dumps(doc)
    assert "name" not in rendered
    assert "[tool.poetry]" in rendered
    assert "version" in rendered


def test_set_table_with_empty_value_creates_empty_section() -> None:
    doc = tomlrt.loads("")
    t = doc.install("tool.poetry", Table.section())
    assert tomlrt.dumps(doc) == "[tool.poetry]\n"
    t["name"] = "x"
    assert tomlrt.dumps(doc) == '[tool.poetry]\nname = "x"\n'


def test_ensure_table_creates_when_absent() -> None:
    doc = tomlrt.loads("")
    deps = doc.ensure_table("tool.poetry.dependencies")
    deps["pytest"] = "^7.0"
    assert tomlrt.dumps(doc) == ('[tool.poetry.dependencies]\npytest = "^7.0"\n')


def test_ensure_table_navigates_existing_explicit_section() -> None:
    doc = tomlrt.loads('[tool.poetry]\nname = "x"\n')
    t = doc.ensure_table("tool.poetry")
    t["version"] = "0.1"
    assert tomlrt.dumps(doc) == ('[tool.poetry]\nname = "x"\nversion = "0.1"\n')


def test_ensure_table_navigates_implicit_super_table() -> None:
    doc = tomlrt.loads('[tool.poetry]\nname = "x"\n')
    t = doc.ensure_table("tool")
    assert isinstance(t, tomlrt.Table)
    # No new [tool] header created.
    assert tomlrt.dumps(doc) == '[tool.poetry]\nname = "x"\n'


def test_ensure_table_creates_only_missing_tail() -> None:
    doc = tomlrt.loads('[tool.poetry]\nname = "x"\n')
    t = doc.ensure_table("tool.poetry.dependencies")
    t["requests"] = "^2.0"
    assert tomlrt.dumps(doc) == (
        '[tool.poetry]\nname = "x"\n\n[tool.poetry.dependencies]\nrequests = "^2.0"\n'
    )


def test_ensure_table_rejects_non_table_value() -> None:
    doc = tomlrt.loads("tool = 1\n")
    with pytest.raises(tomlrt.TOMLError, match=r"existing value"):
        doc.ensure_table("tool")


def test_set_aot_dotted_path() -> None:
    doc = tomlrt.loads("")
    doc.install(
        "tool.poetry.source",
        AoT(
            [{"name": "pypi"}, {"name": "private"}],
        ),
    )
    rendered = tomlrt.dumps(doc)
    assert "[tool]" not in rendered
    assert "[tool.poetry]" not in rendered
    assert rendered.count("[[tool.poetry.source]]") == 2


def test_set_table_rejects_empty_path() -> None:
    doc = tomlrt.loads("")
    with pytest.raises(tomlrt.TOMLError, match="must not be empty"):
        doc.install("", Table.section())


def test_set_table_rejects_empty_segment() -> None:
    doc = tomlrt.loads("")
    with pytest.raises(tomlrt.TOMLError, match="empty segment"):
        doc.install("tool..poetry", Table.section())


def test_install_scalar_at_dotted_path() -> None:
    doc = tomlrt.loads("")
    doc.install("tool.poetry.version", "0.1.0")
    rendered = tomlrt.dumps(doc)
    assert rendered == '[tool.poetry]\nversion = "0.1.0"\n'
    # Repeated install at a sibling under the same parent should reuse
    # the existing [tool.poetry] section rather than create a new one.
    doc.install(("tool", "poetry", "name"), "x")
    rendered = tomlrt.dumps(doc)
    assert rendered.count("[tool.poetry]") == 1


def test_install_scalar_at_literal_dot_segment() -> None:
    doc = tomlrt.loads("")
    doc.install(("tool", "weird.key"), 1)
    assert tomlrt.dumps(doc) == '[tool]\n"weird.key" = 1\n'


def test_install_plain_dict_at_dotted_path() -> None:
    doc = tomlrt.loads("")
    doc.install("tool.xy", {"x": 1, "y": 2})
    assert tomlrt.dumps(doc) == "[tool]\nxy = { x = 1, y = 2 }\n"


def test_install_scalar_on_inline_table() -> None:
    doc = tomlrt.loads("it = { a = 1 }\n")
    inline = doc.table("it")
    inline.install("b", 2)
    assert tomlrt.dumps(doc) == "it = { a = 1, b = 2 }\n"


def test_install_multi_segment_on_inline_table_errors() -> None:
    doc = tomlrt.loads("it = { a = 1 }\n")
    inline = doc.table("it")
    with pytest.raises(tomlrt.TOMLError, match="inline-style table"):
        inline.install("a.b", 1)


def test_table_accepts_dotted_path() -> None:
    doc = tomlrt.loads('[tool.poetry]\nname = "x"\n')
    assert doc.table("tool.poetry")["name"] == "x"


def test_aot_accepts_dotted_path() -> None:
    doc = tomlrt.loads('[[tool.poetry.source]]\nname = "pypi"\n')
    aot = doc.aot("tool.poetry.source")
    assert isinstance(aot, tomlrt.AoT)
    assert aot[0]["name"] == "pypi"


def test_set_table_round_trips() -> None:
    doc = tomlrt.loads("")
    doc.install("tool.poetry", Table.section({"name": "x"}))
    doc.install("tool.poetry.dependencies", Table.section({"requests": "^2.0"}))
    rendered = tomlrt.dumps(doc)
    # Re-parse and re-dump must produce identical bytes.
    assert tomlrt.dumps(tomlrt.loads(rendered)) == rendered


# ---------------------------------------------------------------------------
# Table.set_array
# ---------------------------------------------------------------------------


def test_set_array_creates_inline_array() -> None:
    doc = tomlrt.loads("")
    doc["authors"] = Array(["A", "B"])
    arr = doc["authors"]
    assert isinstance(arr, tomlrt.Array)
    assert tomlrt.dumps(doc) == 'authors = ["A", "B"]\n'


def test_set_array_multiline_lays_out_one_per_line() -> None:
    doc = tomlrt.loads("")
    doc["authors"] = Array(["A", "B", "C"], multiline=True)
    assert tomlrt.dumps(doc) == ('authors = [\n    "A",\n    "B",\n    "C",\n]\n')


def test_set_array_custom_indent() -> None:
    doc = tomlrt.loads("")
    doc["x"] = Array([1, 2], multiline=True, indent="  ")
    assert tomlrt.dumps(doc) == "x = [\n  1,\n  2,\n]\n"


def test_set_array_empty_multiline_appendable() -> None:
    doc = tomlrt.loads("")
    doc["x"] = Array(multiline=True)
    arr = doc["x"]
    arr.append(1)
    assert tomlrt.dumps(doc) == "x = [\n    1,\n]\n"


def test_set_array_dotted_path_creates_parent_section() -> None:
    doc = tomlrt.loads("")
    doc.install("tool.poetry.authors", Array(["A", "B"], multiline=True))
    rendered = tomlrt.dumps(doc)
    assert "[tool]\n" not in rendered
    assert rendered == ('[tool.poetry]\nauthors = [\n    "A",\n    "B",\n]\n')


def test_set_array_dotted_path_uses_existing_section() -> None:
    doc = tomlrt.loads('[tool.poetry]\nname = "x"\n')
    doc.install("tool.poetry.authors", Array(["A", "B"]))
    assert tomlrt.dumps(doc) == ('[tool.poetry]\nname = "x"\nauthors = ["A", "B"]\n')


def test_set_array_replaces_existing_value() -> None:
    doc = tomlrt.loads("a = 1\n")
    doc["a"] = Array([1, 2, 3])
    assert tomlrt.dumps(doc) == "a = [1, 2, 3]\n"


def test_set_array_round_trips() -> None:
    doc = tomlrt.loads("")
    doc.install("tool.poetry.authors", Array(["A", "B"], multiline=True))
    rendered = tomlrt.dumps(doc)
    assert tomlrt.dumps(tomlrt.loads(rendered)) == rendered


def test_assign_over_aot_keeps_dict_view_in_sync() -> None:
    """Regression: the dict view used to keep a stale AoT after an assign."""
    src = '[tool]\n\n[[tool.source]]\nname = "foo"\n'
    doc = tomlrt.loads(src)
    doc["tool"]["source"] = {}
    assert isinstance(doc["tool"]["source"], tomlrt.Table)
    assert dict(doc["tool"]["source"]) == {}
    assert tomlrt.dumps(doc) == "[tool]\nsource = {}\n"


def test_del_then_assign_keeps_dict_view_in_sync() -> None:
    """Regression: re-assigning a key after del used to revive the old AoT."""
    src = '[tool]\n\n[[tool.source]]\nname = "foo"\n'
    doc = tomlrt.loads(src)
    del doc["tool"]["source"]
    doc["tool"]["source"] = {}
    assert isinstance(doc["tool"]["source"], tomlrt.Table)
    assert dict(doc["tool"]["source"]) == {}
    assert tomlrt.dumps(doc) == "[tool]\nsource = {}\n"


def test_pop_then_assign_keeps_dict_view_in_sync() -> None:
    """Regression: dict view returned the old sub-table's keys after re-assign."""
    src = (
        '[tool.poetry]\nname = "x"\n\n[tool.poetry.extras]\na = ["one"]\nb = ["two"]\n'
    )
    doc = tomlrt.loads(src)
    poetry = doc["tool"]["poetry"]
    poetry.pop("extras")
    poetry["extras"] = {"a-norm": ["one"], "b-norm": ["two"]}
    extras = poetry["extras"]
    assert isinstance(extras, tomlrt.Table)
    assert dict(extras) == {"a-norm": ["one"], "b-norm": ["two"]}


def test_pop_inherited_dotted_key_from_ancestor_section() -> None:
    """Regression: ``poetry.name = "x"`` in [tool] reads via doc['tool']['poetry']
    but ``pop('name')`` used to KeyError because the mutation paths ignored
    inherited (extras) entries.
    """
    src = '[tool]\npoetry.name = "x"\n\n[tool.poetry.extras]\na = ["one"]\n'
    doc = tomlrt.loads(src)
    poetry = doc["tool"]["poetry"]
    assert poetry["name"] == "x"
    poetry.pop("name")
    assert "name" not in poetry
    rendered = tomlrt.dumps(doc)
    assert "poetry.name" not in rendered
    assert "[tool.poetry.extras]" in rendered


def test_set_inherited_dotted_key_mutates_in_place() -> None:
    """Assigning to an inherited dotted entry should update the existing KV,
    not create a duplicate in a new section.
    """
    src = '[tool]\npoetry.name = "x"\n'
    doc = tomlrt.loads(src)
    doc["tool"]["poetry"]["name"] = "y"
    rendered = tomlrt.dumps(doc)
    assert rendered == '[tool]\npoetry.name = "y"\n'


def test_preamble_set_on_empty_doc_renders_before_added_content() -> None:
    """Regression: setting preamble on an empty document parks the
    comment in trailing trivia. When the first content is added the
    comment must migrate to the top of the file rather than render
    after the new structural element.
    """
    doc = tomlrt.document()
    doc.preamble = ("This is a comment",)
    doc["x"] = 1
    rendered = tomlrt.dumps(doc)
    assert rendered == "# This is a comment\n\nx = 1\n"
    # The migrated comment must remain visible as preamble for round-trip.
    assert doc.preamble == ("This is a comment",)
    assert tomlrt.loads(rendered).preamble == ("This is a comment",)


def test_preamble_migration_for_set_table_and_set_aot() -> None:
    cases: list[tuple[str, Callable[[tomlrt.Document], object]]] = [
        ("set_table", lambda d: d.install("a", Table.section({"k": 1}))),
        ("set_aot", lambda d: d.install("a", AoT([{"k": 1}]))),
        ("inline_mapping", lambda d: d.__setitem__("a", {"b": 1})),
        ("nested_set_table", lambda d: d.install("a.b", Table.section({"k": 1}))),
    ]
    for op_name, build in cases:
        doc = tomlrt.document()
        doc.preamble = ("Top",)
        build(doc)
        rendered = tomlrt.dumps(doc)
        assert rendered.startswith("# Top\n\n"), (op_name, rendered)
        assert tomlrt.loads(rendered).preamble == ("Top",), op_name


def test_aot_insert_on_empty_doc_migrates_preamble() -> None:
    """``AoT.insert`` was bypassing the preamble-migration choke-point,
    so on an empty doc with a preamble the comment ended up after the
    inserted ``[[..]]`` section instead of before it.
    """
    doc = tomlrt.parse("")
    doc.preamble = ("Copyright",)
    doc["products"] = AoT()
    aot = doc["products"]
    aot.insert(0, {"name": "x"})
    rendered = tomlrt.dumps(doc)
    assert rendered.startswith("# Copyright\n\n[[products]]\n"), rendered
    assert tomlrt.loads(rendered).preamble == ("Copyright",)


def test_promote_array_preserves_source_kv_leading_and_trailing() -> None:
    """``promote_array`` previously dropped the inline KV's leading
    comments / blank lines and trailing EOL comment. Carry them over
    onto the first new ``[[..]]`` header and the last entry's tail.
    """
    src = '# header comment\n\nservers = [{ name = "a" }, { name = "b" }]  # tail\n'
    doc = tomlrt.loads(src)
    doc.promote_array("servers")
    rendered = tomlrt.dumps(doc)
    assert "# header comment" in rendered
    assert rendered.startswith("# header comment\n")
    assert "# tail" in rendered


def test_aot_insert_at_zero_separates_from_following_entry() -> None:
    """``AoT.insert(0, ...)`` previously left the new ``[[..]]`` glued
    to the following existing one because the blank-line policy only
    looked at *preceding* content. Now it also separates from the
    next entry, mirroring sibling-uniformity (default blank-separated).
    """
    doc = tomlrt.loads("[[a]]\nx = 1\n")
    doc["a"].insert(0, {"x": 0})
    assert tomlrt.dumps(doc) == "[[a]]\nx = 0\n\n[[a]]\nx = 1\n"

    # Tight existing layout: don't impose a blank.
    doc = tomlrt.loads("[[a]]\nx = 1\n[[a]]\nx = 2\n")
    doc["a"].insert(0, {"x": 0})
    assert tomlrt.dumps(doc) == "[[a]]\nx = 0\n[[a]]\nx = 1\n[[a]]\nx = 2\n"
