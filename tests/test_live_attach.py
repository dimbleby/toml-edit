"""Live-attach semantics for ``Table.inline`` (and later ``Array``, ``AoT``).

A typed-container assigned to a document attaches *live* when it is not
already attached elsewhere: the user's own reference becomes the view at
the assignment site, and subsequent mutations through that reference are
visible in the document. An already-attached typed container is cloned
on assignment instead, so a single object never lives in two CST
locations. Plain ``dict`` / ``list`` continue to be snapshot-synthesised
and are unchanged.
"""

from __future__ import annotations

import sys
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import pytest

import tomlrt
from tomlrt import Array, Table


def _reparses(src: str) -> dict[str, Any]:
    return tomllib.loads(src)


# ---------------------------------------------------------------------------
# Table.inline factory
# ---------------------------------------------------------------------------


def test_inline_factory_returns_inline_table_view() -> None:
    t = Table.inline({"a": 1, "b": 2})
    assert isinstance(t, Table)
    assert dict(t) == {"a": 1, "b": 2}


def test_inline_factory_empty() -> None:
    t = Table.inline()
    assert dict(t) == {}


def test_inline_factory_can_be_populated_before_assignment() -> None:
    t = Table.inline()
    t["x"] = 1
    t["y"] = "hello"
    assert dict(t) == {"x": 1, "y": "hello"}


# ---------------------------------------------------------------------------
# Live attach on assignment
# ---------------------------------------------------------------------------


def test_mutation_after_assignment_is_visible_in_document() -> None:
    doc = tomlrt.parse("")
    t = Table.inline({"a": 1})
    doc["foo"] = t
    t["b"] = 2
    rendered = tomlrt.dumps(doc)
    assert "b = 2" in rendered
    assert _reparses(rendered) == {"foo": {"a": 1, "b": 2}}


def test_assigned_inline_is_user_reference() -> None:
    doc = tomlrt.parse("")
    t = Table.inline({"a": 1})
    doc["foo"] = t
    assert doc["foo"] is t


def test_incremental_population_then_assign_then_more_mutations() -> None:
    doc = tomlrt.parse("")
    t = Table.inline()
    t["x"] = 1
    t["y"] = 2
    doc["bar"] = t
    t["z"] = 3
    assert _reparses(tomlrt.dumps(doc)) == {"bar": {"x": 1, "y": 2, "z": 3}}


def test_mutation_through_doc_visible_on_user_reference() -> None:
    doc = tomlrt.parse("")
    t = Table.inline({"a": 1})
    doc["foo"] = t
    doc["foo"]["c"] = 3
    assert dict(t) == {"a": 1, "c": 3}


def test_del_through_doc_visible_on_user_reference() -> None:
    doc = tomlrt.parse("")
    t = Table.inline({"a": 1, "b": 2})
    doc["foo"] = t
    del doc["foo"]["a"]
    assert dict(t) == {"b": 2}


# ---------------------------------------------------------------------------
# Already-attached source clones on assignment
# ---------------------------------------------------------------------------


def test_double_assign_clones_second_slot() -> None:
    doc = tomlrt.parse("")
    t = Table.inline({"k": "v"})
    doc["p"] = t
    doc["q"] = t
    assert doc["p"] is t
    assert doc["q"] is not t
    # First slot is live, second is independent.
    t["k"] = "changed"
    rendered = tomlrt.dumps(doc)
    parsed = _reparses(rendered)
    assert parsed == {"p": {"k": "changed"}, "q": {"k": "v"}}


def test_cross_document_assignment_clones() -> None:
    d1 = tomlrt.parse("")
    d2 = tomlrt.parse("")
    t = Table.inline({"k": 1})
    d1["a"] = t
    d2["a"] = d1["a"]
    assert d2["a"] is not d1["a"]
    d1["a"]["k"] = 99
    assert _reparses(tomlrt.dumps(d1))["a"] == {"k": 99}
    assert _reparses(tomlrt.dumps(d2))["a"] == {"k": 1}


def test_intra_document_assignment_clones() -> None:
    doc = tomlrt.parse("")
    doc["a"] = Table.inline({"k": 1})
    doc["b"] = doc["a"]
    assert doc["b"] is not doc["a"]
    doc["a"]["k"] = 99
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"a": {"k": 99}, "b": {"k": 1}}


# ---------------------------------------------------------------------------
# Plain dict assignment is still snapshot
# ---------------------------------------------------------------------------


def test_plain_dict_assignment_is_snapshot() -> None:
    doc = tomlrt.parse("")
    plain = {"a": 1}
    doc["foo"] = plain
    plain["b"] = 2  # plain dict mutation must not reach doc
    assert "b" not in _reparses(tomlrt.dumps(doc))["foo"]


def test_plain_dict_assignment_returns_a_view_not_user_reference() -> None:
    doc = tomlrt.parse("")
    plain = {"a": 1}
    doc["foo"] = plain
    assert doc["foo"] is not plain


# ---------------------------------------------------------------------------
# Round-trip preservation for unrelated documents
# ---------------------------------------------------------------------------


def test_parse_dump_byte_exact_unchanged() -> None:
    src = "# header\nfoo = { a = 1, b = 2 }\n[section]\nx = 1\n"
    doc = tomlrt.parse(src)
    assert tomlrt.dumps(doc) == src


# ---------------------------------------------------------------------------
# Detached-after-overwrite still works (mutations write to the orphan node)
# ---------------------------------------------------------------------------


def test_detached_inline_still_writable() -> None:
    doc = tomlrt.parse("")
    t = Table.inline({"a": 1})
    doc["foo"] = t
    # Overwrite ``foo`` -- ``t`` is now detached.
    doc["foo"] = Table.inline({"new": True})
    # Mutations on the orphan still work locally.
    t["b"] = 2
    assert dict(t) == {"a": 1, "b": 2}
    # But they don't leak into the document.
    assert "b" not in _reparses(tomlrt.dumps(doc))["foo"]


def test_reassign_after_detach_attaches_again() -> None:
    doc = tomlrt.parse("")
    t = Table.inline({"a": 1})
    doc["foo"] = t
    doc["foo"] = Table.inline({"placeholder": True})
    # ``t`` is detached; it should be re-installable as live.
    doc["bar"] = t
    assert doc["bar"] is t
    t["c"] = 3
    assert _reparses(tomlrt.dumps(doc))["bar"] == {"a": 1, "c": 3}


# ---------------------------------------------------------------------------
# Sanity: the type errors we don't want to silently swallow
# ---------------------------------------------------------------------------


def test_inline_factory_rejects_non_string_keys() -> None:
    with pytest.raises(TypeError):
        Table.inline({1: "no"})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# Array live attach
# ---------------------------------------------------------------------------


def test_array_factory_returns_array_view() -> None:
    arr = Array([1, 2, 3])
    assert isinstance(arr, list)
    assert list(arr) == [1, 2, 3]


def test_array_mutation_after_assignment_visible_in_document() -> None:
    doc = tomlrt.parse("")
    arr = Array([1, 2, 3])
    doc["xs"] = arr
    arr.append(4)
    arr[0] = 99
    assert _reparses(tomlrt.dumps(doc)) == {"xs": [99, 2, 3, 4]}


def test_assigned_array_is_user_reference() -> None:
    doc = tomlrt.parse("")
    arr = Array([1, 2, 3])
    doc["xs"] = arr
    assert doc["xs"] is arr


def test_incremental_array_population_then_assign_then_more() -> None:
    doc = tomlrt.parse("")
    arr = Array()
    arr.append(1)
    arr.append(2)
    doc["xs"] = arr
    arr.extend([3, 4])
    assert _reparses(tomlrt.dumps(doc)) == {"xs": [1, 2, 3, 4]}


def test_mutation_through_doc_visible_on_user_reference_array() -> None:
    doc = tomlrt.parse("")
    arr = Array([1, 2])
    doc["xs"] = arr
    doc["xs"].append(3)
    assert list(arr) == [1, 2, 3]


def test_array_double_assign_clones_second_slot() -> None:
    doc = tomlrt.parse("")
    arr = Array([1, 2])
    doc["p"] = arr
    doc["q"] = arr
    assert doc["p"] is arr
    assert doc["q"] is not arr
    arr.append(99)
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"p": [1, 2, 99], "q": [1, 2]}


def test_array_cross_document_assignment_clones() -> None:
    d1 = tomlrt.parse("")
    d2 = tomlrt.parse("")
    arr = Array([1, 2, 3])
    d1["xs"] = arr
    d2["xs"] = d1["xs"]
    assert d2["xs"] is not d1["xs"]
    d1["xs"].append(99)
    assert _reparses(tomlrt.dumps(d1)) == {"xs": [1, 2, 3, 99]}
    assert _reparses(tomlrt.dumps(d2)) == {"xs": [1, 2, 3]}


def test_array_multiline_layout_preserved_through_live_attach() -> None:
    doc = tomlrt.parse("")
    arr = Array([1, 2, 3], multiline=True)
    doc["xs"] = arr
    assert doc["xs"] is arr
    out = tomlrt.dumps(doc)
    # Multiline format: items each on their own line.
    assert "\n    1" in out
    assert "\n    2" in out


def test_plain_list_assignment_is_snapshot() -> None:
    doc = tomlrt.parse("")
    plain = [1, 2, 3]
    doc["xs"] = plain
    plain.append(99)
    assert _reparses(tomlrt.dumps(doc))["xs"] == [1, 2, 3]


def test_detached_array_still_writable() -> None:
    doc = tomlrt.parse("")
    arr = Array([1, 2])
    doc["xs"] = arr
    doc["xs"] = Array([10, 20])  # arr is now detached
    arr.append(3)
    assert list(arr) == [1, 2, 3]
    assert _reparses(tomlrt.dumps(doc))["xs"] == [10, 20]


def test_reassign_array_after_detach_attaches_again() -> None:
    doc = tomlrt.parse("")
    arr = Array([1, 2])
    doc["xs"] = arr
    doc["xs"] = Array([99])
    doc["ys"] = arr  # arr is detached, so re-attaches live here
    assert doc["ys"] is arr
    arr.append(3)
    assert _reparses(tomlrt.dumps(doc))["ys"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Mixed: Array inside an inline table, both live-attached
# ---------------------------------------------------------------------------


def test_array_inside_inline_table_both_live() -> None:
    doc = tomlrt.parse("")
    arr = Array([1, 2])
    inline = Table.inline({"xs": arr})
    doc["t"] = inline
    inline["k"] = "added"
    arr.append(3)
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"t": {"xs": [1, 2, 3], "k": "added"}}


# ---------------------------------------------------------------------------
# AoT live attach
# ---------------------------------------------------------------------------


def test_aot_factory_returns_unattached() -> None:
    aot = tomlrt.AoT([{"name": "a"}, {"name": "b"}])
    assert isinstance(aot, list)
    assert [dict(t) for t in aot] == [{"name": "a"}, {"name": "b"}]


def test_aot_assignment_is_user_reference() -> None:
    doc = tomlrt.parse("")
    aot = tomlrt.AoT([{"name": "a"}])
    doc["servers"] = aot
    assert doc["servers"] is aot


def test_aot_mutation_after_assignment_visible_in_document() -> None:
    doc = tomlrt.parse("")
    aot = tomlrt.AoT([{"name": "a"}])
    doc["servers"] = aot
    aot.append({"name": "b"})
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"servers": [{"name": "a"}, {"name": "b"}]}


def test_aot_entry_mutation_after_assignment_visible_in_document() -> None:
    doc = tomlrt.parse("")
    aot = tomlrt.AoT([{"name": "a"}])
    doc["servers"] = aot
    aot[0]["extra"] = 42
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"servers": [{"name": "a", "extra": 42}]}


def test_empty_aot_attaches_then_appends_via_user_reference() -> None:
    doc = tomlrt.parse("")
    aot = tomlrt.AoT()
    doc["servers"] = aot
    aot.append({"name": "a"})
    aot.append({"name": "b"})
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"servers": [{"name": "a"}, {"name": "b"}]}


def test_aot_double_assign_clones_second_slot() -> None:
    doc = tomlrt.parse("")
    aot = tomlrt.AoT([{"name": "a"}])
    doc["p"] = aot
    doc["q"] = aot
    assert doc["p"] is aot
    assert doc["q"] is not aot
    aot.append({"name": "b"})
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"p": [{"name": "a"}, {"name": "b"}], "q": [{"name": "a"}]}


def test_aot_cross_document_assignment_clones() -> None:
    d1 = tomlrt.parse("")
    aot = tomlrt.AoT([{"name": "a"}])
    d1["servers"] = aot
    d2 = tomlrt.parse("")
    d2["servers"] = d1["servers"]
    assert d1["servers"] is aot
    assert d2["servers"] is not d1["servers"]
    aot.append({"name": "b"})
    assert _reparses(tomlrt.dumps(d1)) == {
        "servers": [{"name": "a"}, {"name": "b"}],
    }
    assert _reparses(tomlrt.dumps(d2)) == {"servers": [{"name": "a"}]}


def test_aot_intra_document_assignment_clones() -> None:
    doc = tomlrt.parse("")
    aot = tomlrt.AoT([{"name": "a"}])
    doc["p"] = aot
    doc["q"] = doc["p"]
    assert doc["p"] is aot
    assert doc["q"] is not doc["p"]
    aot.append({"name": "b"})
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"p": [{"name": "a"}, {"name": "b"}], "q": [{"name": "a"}]}


def test_detached_aot_reattaches_live() -> None:
    doc = tomlrt.parse("")
    aot = tomlrt.AoT([{"name": "a"}])
    doc["servers"] = aot
    doc["servers"] = tomlrt.AoT([{"name": "z"}])  # aot now detached
    doc["others"] = aot  # re-attaches live here
    assert doc["others"] is aot
    aot.append({"name": "b"})
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {
        "servers": [{"name": "z"}],
        "others": [{"name": "a"}, {"name": "b"}],
    }


def test_aot_entry_view_identity_preserved_through_attach() -> None:
    aot = tomlrt.AoT([{"name": "a"}])
    entry = aot[0]
    doc = tomlrt.parse("")
    doc["servers"] = aot
    # The same Table view the user grabbed before assignment is still
    # the live entry post-attach.
    assert aot[0] is entry
    entry["extra"] = 1
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"servers": [{"name": "a", "extra": 1}]}


# ---------------------------------------------------------------------------
# Recursive attach: typed containers inside plain dict/list values
# ---------------------------------------------------------------------------


def test_inline_inside_plain_dict_attaches_live() -> None:
    doc = tomlrt.parse("")
    inner = Table.inline({"z": 1})
    doc["x"] = {"y": inner}
    inner["extra"] = 99
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"x": {"y": {"z": 1, "extra": 99}}}


def test_array_inside_plain_dict_attaches_live() -> None:
    doc = tomlrt.parse("")
    arr = Array([1, 2])
    doc["x"] = {"xs": arr}
    arr.append(3)
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"x": {"xs": [1, 2, 3]}}


def test_inline_inside_plain_list_attaches_live() -> None:
    doc = tomlrt.parse("")
    inner = Table.inline({"z": 1})
    doc["xs"] = [inner, {"q": 2}]
    inner["extra"] = 99
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"xs": [{"z": 1, "extra": 99}, {"q": 2}]}


def test_array_inside_array_attaches_live() -> None:
    doc = tomlrt.parse("")
    inner = Array([1, 2])
    doc["xs"] = Array([inner, [3, 4]])
    inner.append(99)
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"xs": [[1, 2, 99], [3, 4]]}


def test_outer_plain_dict_remains_snapshot() -> None:
    # The plain-dict outer is still a snapshot: mutating it after
    # assignment does *not* show up in the document, even though a
    # nested typed container inside it attaches live.
    doc = tomlrt.parse("")
    plain: dict[str, object] = {"y": Table.inline({"z": 1})}
    doc["x"] = plain
    plain["new"] = 42  # outer is snapshot — not visible in doc
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"x": {"y": {"z": 1}}}
