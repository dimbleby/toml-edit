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

import tomllib

import pytest

import tomle
from tomle import TOMLEditError


def _reparses(src: str) -> dict[str, object]:
    return tomllib.loads(src)


# ---------------------------------------------------------------------------
# Discontiguous tables: [a] / [a.sub] / [b] / [a] is forbidden, but a
# logical table can still aggregate keys from the [a] header *and* from
# any [a.x] sub-section headers.
# ---------------------------------------------------------------------------


def test_table_with_sub_section_iter_includes_subtable() -> None:
    src = "[a]\nx = 1\n[a.sub]\ny = 2\n"
    doc = tomle.parse(src)
    a = doc["a"]
    assert isinstance(a, tomle.Table)
    assert a["x"] == 1
    sub = a["sub"]
    assert isinstance(sub, tomle.Table)
    assert sub["y"] == 2


def test_table_with_sub_section_modify_subtable_value() -> None:
    src = "[a]\nx = 1\n[a.sub]\ny = 2\n"
    doc = tomle.parse(src)
    a = doc["a"]
    assert isinstance(a, tomle.Table)
    sub = a["sub"]
    assert isinstance(sub, tomle.Table)
    sub["y"] = 99
    out = tomle.dumps(doc)
    assert out == "[a]\nx = 1\n[a.sub]\ny = 99\n"


def test_table_with_sub_section_add_to_parent_appends_in_parent_block() -> None:
    src = "[a]\nx = 1\n[a.sub]\ny = 2\n"
    doc = tomle.parse(src)
    a = doc["a"]
    assert isinstance(a, tomle.Table)
    a["z"] = 3
    out = tomle.dumps(doc)
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
    doc = tomle.parse(src)
    a = doc["a"]
    assert isinstance(a, tomle.Table)
    assert dict(a) == {"b": 1, "c": 2}


def test_dotted_key_table_set_via_subtable_raises() -> None:
    """Adding a key to a logical table that exists only as dotted entries.

    The current mutation policy doesn't synthesize new dotted-key entries
    (it would need to choose where to put them). Document this with a
    raised error so we can revisit when the comments/inline-promotion
    work lands.
    """
    src = "a.b = 1\na.c = 2\n"
    doc = tomle.parse(src)
    a = doc["a"]
    assert isinstance(a, tomle.Table)
    with pytest.raises(TOMLEditError):
        a["d"] = 3


# ---------------------------------------------------------------------------
# Arrays-of-tables (AoT) — middle ops and entries with sub-sections
# ---------------------------------------------------------------------------


def test_aot_basic_iteration() -> None:
    src = '[[users]]\nname = "alice"\n[[users]]\nname = "bob"\n'
    doc = tomle.parse(src)
    users = doc["users"]
    assert isinstance(users, tomle.AoT)
    assert [u["name"] for u in users] == ["alice", "bob"]


def test_aot_append_entry_via_dict() -> None:
    src = '[[users]]\nname = "alice"\n'
    doc = tomle.parse(src)
    users = doc["users"]
    assert isinstance(users, tomle.AoT)
    users.append({"name": "bob"})
    out = tomle.dumps(doc)
    assert _reparses(out) == {"users": [{"name": "alice"}, {"name": "bob"}]}


def test_aot_modify_field_in_first_entry() -> None:
    src = '[[users]]\nname = "alice"\n[[users]]\nname = "bob"\n'
    doc = tomle.parse(src)
    users = doc["users"]
    assert isinstance(users, tomle.AoT)
    users[0]["name"] = "ALICE"
    out = tomle.dumps(doc)
    assert out == '[[users]]\nname = "ALICE"\n[[users]]\nname = "bob"\n'


def test_aot_modify_field_in_middle_entry() -> None:
    src = (
        '[[users]]\nname = "a"\n'
        '[[users]]\nname = "b"\n'
        '[[users]]\nname = "c"\n'
    )
    doc = tomle.parse(src)
    users = doc["users"]
    assert isinstance(users, tomle.AoT)
    users[1]["name"] = "B"
    out = tomle.dumps(doc)
    assert (
        out
        == '[[users]]\nname = "a"\n[[users]]\nname = "B"\n[[users]]\nname = "c"\n'
    )


def test_aot_entry_sub_section_read() -> None:
    """[[arr]] / [arr.sub] — sub belongs to the AoT entry."""
    src = (
        "[[arr]]\nx = 1\n"
        "[arr.sub]\ny = 2\n"
        "[[arr]]\nx = 10\n"
        "[arr.sub]\ny = 20\n"
    )
    doc = tomle.parse(src)
    arr = doc["arr"]
    assert isinstance(arr, tomle.AoT)
    assert len(arr) == 2
    assert arr[0]["x"] == 1
    sub0 = arr[0]["sub"]
    assert isinstance(sub0, tomle.Table)
    assert sub0["y"] == 2
    assert arr[1]["x"] == 10
    sub1 = arr[1]["sub"]
    assert isinstance(sub1, tomle.Table)
    assert sub1["y"] == 20


def test_aot_entry_sub_section_modify_value() -> None:
    src = (
        "[[arr]]\nx = 1\n"
        "[arr.sub]\ny = 2\n"
        "[[arr]]\nx = 10\n"
        "[arr.sub]\ny = 20\n"
    )
    doc = tomle.parse(src)
    arr = doc["arr"]
    assert isinstance(arr, tomle.AoT)
    sub = arr[1]["sub"]
    assert isinstance(sub, tomle.Table)
    sub["y"] = 999
    out = tomle.dumps(doc)
    assert out == (
        "[[arr]]\nx = 1\n"
        "[arr.sub]\ny = 2\n"
        "[[arr]]\nx = 10\n"
        "[arr.sub]\ny = 999\n"
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
    doc = tomle.parse(src)
    owner = doc["owner"]
    assert isinstance(owner, tomle.Table)
    owner["name"] = "tim"
    out = tomle.dumps(doc)
    # Style of the replaced scalar regenerates as basic-quoted (default),
    # but surrounding spacing/comma trivia is preserved.
    assert out == 'owner = { name = "tim", dob = 1979 }\n'


def test_inline_array_modify_preserves_brackets() -> None:
    src = "ports = [ 80, 443, 8080 ]\n"
    doc = tomle.parse(src)
    ports = doc["ports"]
    assert isinstance(ports, tomle.Array)
    ports[1] = 444
    out = tomle.dumps(doc)
    assert out == "ports = [ 80, 444, 8080 ]\n"


def test_array_insert_then_pop_round_trips() -> None:
    src = "ports = [80, 443]\n"
    doc = tomle.parse(src)
    ports = doc["ports"]
    assert isinstance(ports, tomle.Array)
    ports.insert(1, 8080)
    assert list(ports) == [80, 8080, 443]
    ports.pop(1)
    out = tomle.dumps(doc)
    assert out == "ports = [80, 443]\n"


# ---------------------------------------------------------------------------
# Cross-document assignment — must deep-clone, never share state
# ---------------------------------------------------------------------------


def test_cross_doc_table_assign_deep_clones() -> None:
    src1 = '[srv]\nhost = "a.example"\nport = 80\n'
    src2 = ""
    a = tomle.parse(src1)
    b = tomle.parse(src2)
    b["srv"] = a["srv"]
    # Mutating `a` must not affect `b`.
    a_srv = a["srv"]
    assert isinstance(a_srv, tomle.Table)
    a_srv["port"] = 9999
    out_a = tomle.dumps(a)
    out_b = tomle.dumps(b)
    assert _reparses(out_a) == {"srv": {"host": "a.example", "port": 9999}}
    assert _reparses(out_b) == {"srv": {"host": "a.example", "port": 80}}


def test_cross_doc_aot_assign_deep_clones() -> None:
    src1 = '[[users]]\nname = "alice"\n[[users]]\nname = "bob"\n'
    src2 = ""
    a = tomle.parse(src1)
    b = tomle.parse(src2)
    b["users"] = a["users"]
    a_users = a["users"]
    assert isinstance(a_users, tomle.AoT)
    a_users[0]["name"] = "MUT"
    assert _reparses(tomle.dumps(a))["users"][0]["name"] == "MUT"  # type: ignore[index]
    assert _reparses(tomle.dumps(b))["users"][0]["name"] == "alice"  # type: ignore[index]


def test_cross_doc_array_assign_deep_clones() -> None:
    src1 = "ports = [80, 443]\n"
    src2 = ""
    a = tomle.parse(src1)
    b = tomle.parse(src2)
    b["ports"] = a["ports"]
    a_ports = a["ports"]
    assert isinstance(a_ports, tomle.Array)
    a_ports.append(8080)
    assert _reparses(tomle.dumps(a))["ports"] == [80, 443, 8080]
    assert _reparses(tomle.dumps(b))["ports"] == [80, 443]


# ---------------------------------------------------------------------------
# Cross-section conflict on mutation
# ---------------------------------------------------------------------------


def test_set_value_conflicting_with_existing_subsection_raises() -> None:
    """Setting a name that already exists as a sub-section header.

    [a] / x=1 / [a.b] makes 'b' a sub-table inside [a]. Trying
    a["b"] = 2 would have to either redefine that sub-table as a value
    (breaking [a.b]) or be rejected. The classifier must see "table"
    here and refuse.
    """
    src = "[a]\nx = 1\n[a.b]\ny = 2\n"
    doc = tomle.parse(src)
    a = doc["a"]
    assert isinstance(a, tomle.Table)
    with pytest.raises(TOMLEditError):
        a["b"] = 99


def test_set_value_conflicting_with_existing_aot_raises() -> None:
    src = '[a]\nx = 1\n[[a.items]]\nname = "first"\n'
    doc = tomle.parse(src)
    a = doc["a"]
    assert isinstance(a, tomle.Table)
    with pytest.raises(TOMLEditError):
        a["items"] = 5


# ---------------------------------------------------------------------------
# Sub-table access through AoT entry (uses the new owned_scope path)
# ---------------------------------------------------------------------------


def test_aot_entry_owned_scope_isolates_sibling_sub_sections() -> None:
    """[[arr]] / [arr.sub] / x=1 / [[arr]] / [arr.sub] / x=2

    Each entry's sub.x must be independent; mutating arr[0].sub.x must
    not affect arr[1].sub.x.
    """
    src = (
        "[[arr]]\n"
        "[arr.sub]\nx = 1\n"
        "[[arr]]\n"
        "[arr.sub]\nx = 2\n"
    )
    doc = tomle.parse(src)
    arr = doc["arr"]
    assert isinstance(arr, tomle.AoT)
    s0 = arr[0]["sub"]
    s1 = arr[1]["sub"]
    assert isinstance(s0, tomle.Table)
    assert isinstance(s1, tomle.Table)
    assert s0["x"] == 1
    assert s1["x"] == 2
    s0["x"] = 100
    assert s1["x"] == 2  # unchanged
    out = tomle.dumps(doc)
    assert _reparses(out) == {
        "arr": [{"sub": {"x": 100}}, {"sub": {"x": 2}}]
    }
