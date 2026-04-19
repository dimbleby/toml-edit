"""Spacing heuristics: new entries mimic the local blank-line convention."""

from __future__ import annotations

import toml_edit

# ---------------------------------------------------------------------------
# KV append
# ---------------------------------------------------------------------------


def test_kv_append_to_packed_section_stays_packed() -> None:
    doc = toml_edit.parse("a = 1\nb = 2\n")
    doc["c"] = 3
    assert toml_edit.dumps(doc) == "a = 1\nb = 2\nc = 3\n"


def test_kv_append_to_uniformly_spaced_section_adds_blank() -> None:
    doc = toml_edit.parse("a = 1\n\nb = 2\n")
    doc["c"] = 3
    assert toml_edit.dumps(doc) == "a = 1\n\nb = 2\n\nc = 3\n"


def test_kv_append_to_mixed_layout_does_not_add_blank() -> None:
    doc = toml_edit.parse("a = 1\nb = 2\n\nc = 3\n")
    doc["d"] = 4
    assert toml_edit.dumps(doc) == "a = 1\nb = 2\n\nc = 3\nd = 4\n"


def test_kv_append_to_single_entry_does_not_add_blank() -> None:
    doc = toml_edit.parse("a = 1\n")
    doc["b"] = 2
    assert toml_edit.dumps(doc) == "a = 1\nb = 2\n"


def test_kv_append_to_empty_table_does_not_add_blank() -> None:
    doc = toml_edit.parse("[t]\n")
    tbl = doc["t"]
    assert isinstance(tbl, toml_edit.Table)
    tbl["a"] = 1
    assert toml_edit.dumps(doc) == "[t]\na = 1\n"


def test_kv_append_preserves_indent_and_adds_blank() -> None:
    src = "[t]\n    a = 1\n\n    b = 2\n"
    doc = toml_edit.parse(src)
    tbl = doc["t"]
    assert isinstance(tbl, toml_edit.Table)
    tbl["c"] = 3
    assert toml_edit.dumps(doc) == "[t]\n    a = 1\n\n    b = 2\n\n    c = 3\n"


# ---------------------------------------------------------------------------
# AoT append / insert
# ---------------------------------------------------------------------------


def test_aot_append_to_packed_aot_stays_packed() -> None:
    src = "[[i]]\nx = 1\n[[i]]\nx = 2\n"
    doc = toml_edit.parse(src)
    aot = doc["i"]
    assert isinstance(aot, toml_edit.AoT)
    aot.append({"x": 3})
    assert toml_edit.dumps(doc) == "[[i]]\nx = 1\n[[i]]\nx = 2\n[[i]]\nx = 3\n"


def test_aot_append_to_spaced_aot_adds_blank() -> None:
    src = "[[i]]\nx = 1\n\n[[i]]\nx = 2\n"
    doc = toml_edit.parse(src)
    aot = doc["i"]
    assert isinstance(aot, toml_edit.AoT)
    aot.append({"x": 3})
    assert toml_edit.dumps(doc) == (
        "[[i]]\nx = 1\n\n[[i]]\nx = 2\n\n[[i]]\nx = 3\n"
    )


def test_aot_insert_middle_uniformly_spaced_adds_blank() -> None:
    src = "[[i]]\nx = 1\n\n[[i]]\nx = 3\n"
    doc = toml_edit.parse(src)
    aot = doc["i"]
    assert isinstance(aot, toml_edit.AoT)
    aot.insert(1, {"x": 2})
    assert toml_edit.dumps(doc) == (
        "[[i]]\nx = 1\n\n[[i]]\nx = 2\n\n[[i]]\nx = 3\n"
    )


def test_aot_insert_into_mixed_does_not_add_blank() -> None:
    src = "[[i]]\nx = 1\n[[i]]\nx = 2\n\n[[i]]\nx = 4\n"
    doc = toml_edit.parse(src)
    aot = doc["i"]
    assert isinstance(aot, toml_edit.AoT)
    aot.append({"x": 5})
    assert toml_edit.dumps(doc) == (
        "[[i]]\nx = 1\n[[i]]\nx = 2\n\n[[i]]\nx = 4\n[[i]]\nx = 5\n"
    )


def test_aot_append_to_single_entry_does_not_add_blank() -> None:
    src = "[[i]]\nx = 1\n"
    doc = toml_edit.parse(src)
    aot = doc["i"]
    assert isinstance(aot, toml_edit.AoT)
    aot.append({"x": 2})
    assert toml_edit.dumps(doc) == "[[i]]\nx = 1\n[[i]]\nx = 2\n"


def test_aot_extend_inherits_blank_after_first_added_entry() -> None:
    src = "[[i]]\nx = 1\n\n[[i]]\nx = 2\n"
    doc = toml_edit.parse(src)
    aot = doc["i"]
    assert isinstance(aot, toml_edit.AoT)
    aot.extend([{"x": 3}, {"x": 4}])
    assert toml_edit.dumps(doc) == (
        "[[i]]\nx = 1\n\n[[i]]\nx = 2\n\n[[i]]\nx = 3\n\n[[i]]\nx = 4\n"
    )
