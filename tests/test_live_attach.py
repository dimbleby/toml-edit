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
from tomlrt import Table


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
