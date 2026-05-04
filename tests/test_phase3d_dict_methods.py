"""Phase 3d-2 / 3d-5 — dict-method overrides + section-only-doc seam."""

from __future__ import annotations

import copy

import pytest

from _toml_str import td
from tomlrt import dumps, loads
from tomlrt._invariants import check

# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


def test_clear_section_keeps_header() -> None:
    doc = loads("[s]\nx = 1\ny = 2\n")
    doc.table("s").clear()
    check(doc)
    assert dumps(doc) == "[s]\n"
    assert dict(doc.table("s")) == {}


def test_clear_doc_root_drops_everything() -> None:
    doc = loads("a = 1\n[s]\nx = 2\n")
    doc.clear()
    check(doc)
    assert dumps(doc) == ""
    assert dict(doc) == {}


def test_clear_inline_table() -> None:
    doc = loads("x = { a = 1, b = 2 }\n")
    doc.table("x").clear()
    check(doc)
    assert dumps(doc) == "x = {}\n"


def test_clear_empty_is_noop() -> None:
    doc = loads("")
    doc.clear()
    assert dumps(doc) == ""


# ---------------------------------------------------------------------------
# pop / popitem
# ---------------------------------------------------------------------------


def test_pop_returns_and_removes() -> None:
    doc = loads("a = 1\nb = 2\n")
    v = doc.pop("a")
    assert v == 1
    check(doc)
    assert dumps(doc) == "b = 2\n"


def test_pop_missing_raises() -> None:
    doc = loads("a = 1\n")
    with pytest.raises(KeyError):
        doc.pop("missing")


def test_pop_missing_with_default_returns_default() -> None:
    doc = loads("a = 1\n")
    sentinel = object()
    assert doc.pop("missing", sentinel) is sentinel
    assert doc.pop("missing", None) is None


def test_pop_too_many_args_raises() -> None:
    doc = loads("")
    with pytest.raises(TypeError):
        doc.pop("a", 1, 2)  # type: ignore[call-arg]


def test_popitem_returns_last_lifo() -> None:
    doc = loads("a = 1\nb = 2\nc = 3\n")
    k, v = doc.popitem()
    assert (k, v) == ("c", 3)
    check(doc)
    assert dumps(doc) == "a = 1\nb = 2\n"


def test_popitem_empty_raises() -> None:
    doc = loads("")
    with pytest.raises(KeyError):
        doc.popitem()


def test_pop_inline_entry() -> None:
    doc = loads("x = { a = 1, b = 2 }\n")
    v = doc.table("x").pop("a")
    assert v == 1
    check(doc)
    # exact spacing depends on inline-ops policy; just round-trip
    assert loads(dumps(doc)).to_dict() == {"x": {"b": 2}}


# ---------------------------------------------------------------------------
# update / |=
# ---------------------------------------------------------------------------


def test_update_with_mapping() -> None:
    doc = loads("a = 1\n")
    doc.update({"b": 2, "c": 3})
    check(doc)
    assert doc["a"] == 1
    assert doc["b"] == 2
    assert doc["c"] == 3


def test_update_with_kwargs() -> None:
    doc = loads("")
    doc.update(a=1, b=2)
    check(doc)
    assert dict(doc) == {"a": 1, "b": 2}


def test_update_kwargs_override_mapping() -> None:
    doc = loads("")
    doc.update({"a": 1}, a=2)
    assert doc["a"] == 2


def test_update_with_iterable_of_pairs() -> None:
    doc = loads("")
    doc.update([("a", 1), ("b", 2)])
    check(doc)
    assert dict(doc) == {"a": 1, "b": 2}


def test_update_overwrites_existing_scalar() -> None:
    doc = loads("a = 1\n")
    doc.update({"a": 99})
    check(doc)
    assert dumps(doc) == "a = 99\n"


def test_ior_returns_self_and_mutates() -> None:
    doc = loads("a = 1\n")
    original = doc
    doc |= {"b": 2}
    assert doc is original
    check(doc)
    assert doc["b"] == 2


def test_self_assign_via_ior_is_noop() -> None:
    # |= with self: dict.update on self iterates self's items, each
    # write is a self-assign no-op via __setitem__.
    doc = loads("a = 1\nb = 2\n")
    src = dumps(doc)
    doc |= dict(doc)
    assert dumps(doc) == src


# ---------------------------------------------------------------------------
# setdefault
# ---------------------------------------------------------------------------


def test_setdefault_existing_returns_value_no_mutation() -> None:
    doc = loads("a = 1\n")
    src = dumps(doc)
    assert doc.setdefault("a", 99) == 1
    assert dumps(doc) == src


def test_setdefault_missing_inserts() -> None:
    doc = loads("a = 1\n")
    assert doc.setdefault("b", 2) == 2
    check(doc)
    assert doc["b"] == 2


# ---------------------------------------------------------------------------
# copy / deepcopy — explicit deferral
# ---------------------------------------------------------------------------


def test_copy_deferred() -> None:
    doc = loads("a = 1\n")
    with pytest.raises(NotImplementedError):
        copy.copy(doc)


def test_deepcopy_deferred() -> None:
    doc = loads("a = 1\n")
    with pytest.raises(NotImplementedError):
        copy.deepcopy(doc)


# ---------------------------------------------------------------------------
# 3d-5 — section-only-doc top-level insert
# ---------------------------------------------------------------------------


def test_insert_top_level_kv_into_section_only_doc() -> None:
    doc = loads("[s]\nx = 1\n")
    doc["new"] = 1
    check(doc)
    assert dumps(doc) == "new = 1\n\n[s]\nx = 1\n"


def test_insert_top_level_kv_first_header_already_blank() -> None:
    doc = loads("\n[s]\nx = 1\n")
    doc["new"] = 1
    check(doc)
    # The blank line above [s] is preserved; no extra one inserted.
    assert dumps(doc) == "new = 1\n\n[s]\nx = 1\n"


def test_insert_top_level_kv_first_header_has_comment() -> None:
    src = td("""
        # heading comment
        [s]
        x = 1
    """)
    doc = loads(src)
    doc["new"] = 1
    check(doc)
    out = dumps(doc)
    # Comment block stays attached to [s]; new KV gets a blank
    # separator before the comment.
    assert out == "new = 1\n\n# heading comment\n[s]\nx = 1\n"


def test_insert_two_top_level_kvs_into_section_only_doc() -> None:
    doc = loads("[s]\nx = 1\n")
    doc["a"] = 1
    doc["b"] = 2
    check(doc)
    # Second insert uses the standard insert_after(_body_tail) path.
    assert dumps(doc) == "a = 1\nb = 2\n\n[s]\nx = 1\n"


def test_insert_top_level_kv_crlf_doc() -> None:
    doc = loads("[s]\r\nx = 1\r\n")
    doc["new"] = 1
    check(doc)
    assert dumps(doc) == "new = 1\r\n\r\n[s]\r\nx = 1\r\n"


def test_delete_inserted_top_level_kv_round_trips() -> None:
    doc = loads("[s]\nx = 1\n")
    doc["new"] = 1
    del doc["new"]
    check(doc)
    # Blank-line separator left behind on [s] is acceptable residue;
    # invariant + reparse is what matters.
    assert loads(dumps(doc)).to_dict() == {"s": {"x": 1}}
