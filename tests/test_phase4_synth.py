"""Phase 4-partial: plain-dict / plain-list synthesis at assignment time."""

from __future__ import annotations

import pytest

import tomlrt
from tomlrt._invariants import check


def _rt(doc: tomlrt.Document) -> str:
    out = tomlrt.dumps(doc)
    check(doc)
    # And the result must reparse with the same logical shape.
    tomlrt.loads(out)
    return out


def test_assign_plain_dict_creates_inline_table() -> None:
    doc = tomlrt.loads("")
    doc["obj"] = {"a": 1, "b": "two"}
    assert _rt(doc) == 'obj = {a = 1, b = "two"}\n'
    assert doc["obj"]["a"] == 1
    assert doc["obj"]["b"] == "two"


def test_assign_plain_list_creates_inline_array() -> None:
    doc = tomlrt.loads("")
    doc["nums"] = [1, 2, 3]
    assert _rt(doc) == "nums = [1, 2, 3]\n"
    assert list(doc["nums"]) == [1, 2, 3]


def test_assign_empty_dict_emits_empty_inline_table() -> None:
    doc = tomlrt.loads("")
    doc["obj"] = {}
    assert _rt(doc) == "obj = {}\n"


def test_assign_empty_list_emits_empty_inline_array() -> None:
    doc = tomlrt.loads("")
    doc["xs"] = []
    assert _rt(doc) == "xs = []\n"


def test_replace_scalar_with_dict_swaps_value_in_place() -> None:
    doc = tomlrt.loads("k = 1  # trailing\n")
    doc["k"] = {"a": 1}
    out = _rt(doc)
    # Original EOL trivia is preserved (the slot is reused).
    assert out == "k = {a = 1}  # trailing\n"


def test_replace_scalar_with_list_swaps_value_in_place() -> None:
    doc = tomlrt.loads("k = 1\n")
    doc["k"] = [True, False]
    assert _rt(doc) == "k = [true, false]\n"


def test_replace_inline_table_with_dict_swaps_value_in_place() -> None:
    doc = tomlrt.loads("k = {x = 1}\n")
    doc["k"] = {"y": 2, "z": 3}
    assert _rt(doc) == "k = {y = 2, z = 3}\n"


def test_replace_inline_array_with_dict_swaps_value_in_place() -> None:
    doc = tomlrt.loads("k = [1, 2]\n")
    doc["k"] = {"a": 1}
    assert _rt(doc) == "k = {a = 1}\n"


def test_nested_dict_synthesises_to_nested_inline_tables() -> None:
    doc = tomlrt.loads("")
    doc["t"] = {"inner": {"x": 1, "y": 2}}
    assert _rt(doc) == "t = {inner = {x = 1, y = 2}}\n"
    assert doc["t"]["inner"]["x"] == 1


def test_list_of_dicts_round_trips() -> None:
    doc = tomlrt.loads("")
    doc["xs"] = [{"a": 1}, {"b": 2}]
    assert _rt(doc) == "xs = [{a = 1}, {b = 2}]\n"


def test_quoted_inline_keys_use_basic_strings() -> None:
    doc = tomlrt.loads("")
    doc["t"] = {"has space": 1, "ok-bare_K": 2}
    assert _rt(doc) == 't = {"has space" = 1, ok-bare_K = 2}\n'


def test_string_value_in_synth_uses_basic_string() -> None:
    doc = tomlrt.loads("")
    doc["t"] = {"k": 'v"with"quotes'}
    out = _rt(doc)
    # round-trip without claiming exact lexeme
    assert tomlrt.loads(out)["t"]["k"] == 'v"with"quotes'


def test_typed_container_assign_still_defers_to_phase4_proper() -> None:
    src = tomlrt.loads("[a]\nx = 1\n")
    dst = tomlrt.loads("")
    with pytest.raises(NotImplementedError):
        dst["a"] = src["a"]


def test_invariant_passes_for_array_of_dicts_at_top_level() -> None:
    """Regression: invariant checker used to collide path=() with doc root."""
    doc = tomlrt.loads("y = 0\n")
    doc["xs"] = [{"a": 1}, {"b": 2}]
    check(doc)


def test_dict_method_setitem_through_synth_for_inline_value() -> None:
    doc = tomlrt.loads("")
    doc.update({"a": 1, "b": [1, 2, 3], "c": {"k": "v"}})
    out = _rt(doc)
    assert "a = 1" in out
    assert "b = [1, 2, 3]" in out
    assert 'c = {k = "v"}' in out
