"""Spacing heuristics: new entries mimic the local blank-line convention."""

from __future__ import annotations

import tomlrt

# ---------------------------------------------------------------------------
# KV append
# ---------------------------------------------------------------------------


def test_kv_append_to_packed_section_stays_packed() -> None:
    doc = tomlrt.parse("a = 1\nb = 2\n")
    doc["c"] = 3
    assert tomlrt.dumps(doc) == "a = 1\nb = 2\nc = 3\n"


def test_kv_append_to_uniformly_spaced_section_adds_blank() -> None:
    doc = tomlrt.parse("a = 1\n\nb = 2\n")
    doc["c"] = 3
    assert tomlrt.dumps(doc) == "a = 1\n\nb = 2\n\nc = 3\n"


def test_kv_append_to_mixed_layout_does_not_add_blank() -> None:
    doc = tomlrt.parse("a = 1\nb = 2\n\nc = 3\n")
    doc["d"] = 4
    assert tomlrt.dumps(doc) == "a = 1\nb = 2\n\nc = 3\nd = 4\n"


def test_kv_append_to_single_entry_does_not_add_blank() -> None:
    doc = tomlrt.parse("a = 1\n")
    doc["b"] = 2
    assert tomlrt.dumps(doc) == "a = 1\nb = 2\n"


def test_kv_append_to_empty_table_does_not_add_blank() -> None:
    doc = tomlrt.parse("[t]\n")
    tbl = doc["t"]
    assert isinstance(tbl, tomlrt.Table)
    tbl["a"] = 1
    assert tomlrt.dumps(doc) == "[t]\na = 1\n"


def test_kv_append_preserves_indent_and_adds_blank() -> None:
    src = "[t]\n    a = 1\n\n    b = 2\n"
    doc = tomlrt.parse(src)
    tbl = doc["t"]
    assert isinstance(tbl, tomlrt.Table)
    tbl["c"] = 3
    assert tomlrt.dumps(doc) == "[t]\n    a = 1\n\n    b = 2\n\n    c = 3\n"


# ---------------------------------------------------------------------------
# AoT append / insert
# ---------------------------------------------------------------------------


def test_aot_append_to_packed_aot_stays_packed() -> None:
    src = "[[i]]\nx = 1\n[[i]]\nx = 2\n"
    doc = tomlrt.parse(src)
    aot = doc["i"]
    assert isinstance(aot, tomlrt.AoT)
    aot.append({"x": 3})
    assert tomlrt.dumps(doc) == "[[i]]\nx = 1\n[[i]]\nx = 2\n[[i]]\nx = 3\n"


def test_aot_append_to_spaced_aot_adds_blank() -> None:
    src = "[[i]]\nx = 1\n\n[[i]]\nx = 2\n"
    doc = tomlrt.parse(src)
    aot = doc["i"]
    assert isinstance(aot, tomlrt.AoT)
    aot.append({"x": 3})
    assert tomlrt.dumps(doc) == ("[[i]]\nx = 1\n\n[[i]]\nx = 2\n\n[[i]]\nx = 3\n")


def test_aot_insert_middle_uniformly_spaced_adds_blank() -> None:
    src = "[[i]]\nx = 1\n\n[[i]]\nx = 3\n"
    doc = tomlrt.parse(src)
    aot = doc["i"]
    assert isinstance(aot, tomlrt.AoT)
    aot.insert(1, {"x": 2})
    assert tomlrt.dumps(doc) == ("[[i]]\nx = 1\n\n[[i]]\nx = 2\n\n[[i]]\nx = 3\n")


def test_aot_insert_into_mixed_does_not_add_blank() -> None:
    src = "[[i]]\nx = 1\n[[i]]\nx = 2\n\n[[i]]\nx = 4\n"
    doc = tomlrt.parse(src)
    aot = doc["i"]
    assert isinstance(aot, tomlrt.AoT)
    aot.append({"x": 5})
    assert tomlrt.dumps(doc) == (
        "[[i]]\nx = 1\n[[i]]\nx = 2\n\n[[i]]\nx = 4\n[[i]]\nx = 5\n"
    )


def test_aot_append_to_single_entry_adds_blank() -> None:
    """With only one prior entry the style is ambiguous, so default to
    canonical blank-separated TOML — matches what most users expect and
    what re-parsing the result then dumping again produces."""
    src = "[[i]]\nx = 1\n"
    doc = tomlrt.parse(src)
    aot = doc["i"]
    assert isinstance(aot, tomlrt.AoT)
    aot.append({"x": 2})
    assert tomlrt.dumps(doc) == "[[i]]\nx = 1\n\n[[i]]\nx = 2\n"


def test_aot_append_preserves_user_no_blank_style() -> None:
    """When the existing entries clearly show no-blank-line spacing
    (≥2 sibling gaps to learn from), respect the user's choice."""
    src = "[[i]]\nx = 1\n[[i]]\nx = 2\n"
    doc = tomlrt.parse(src)
    aot = doc["i"]
    assert isinstance(aot, tomlrt.AoT)
    aot.append({"x": 3})
    assert tomlrt.dumps(doc) == "[[i]]\nx = 1\n[[i]]\nx = 2\n[[i]]\nx = 3\n"


def test_aot_extend_inherits_blank_after_first_added_entry() -> None:
    src = "[[i]]\nx = 1\n\n[[i]]\nx = 2\n"
    doc = tomlrt.parse(src)
    aot = doc["i"]
    assert isinstance(aot, tomlrt.AoT)
    aot.extend([{"x": 3}, {"x": 4}])
    assert tomlrt.dumps(doc) == (
        "[[i]]\nx = 1\n\n[[i]]\nx = 2\n\n[[i]]\nx = 3\n\n[[i]]\nx = 4\n"
    )


# ---------------------------------------------------------------------------
# Inline array style preservation on append/sort/delete
# ---------------------------------------------------------------------------


def test_inline_array_append_preserves_single_space_separator() -> None:
    doc = tomlrt.parse("x = [1, 2, 3]\n")
    doc.array("x").append(4)
    assert tomlrt.dumps(doc) == "x = [1, 2, 3, 4]\n"


def test_inline_array_append_preserves_compact_separator() -> None:
    doc = tomlrt.parse("x = [1,2,3]\n")
    doc.array("x").append(4)
    assert tomlrt.dumps(doc) == "x = [1,2,3,4]\n"


def test_inline_array_append_preserves_bracket_padding() -> None:
    doc = tomlrt.parse("x = [ 1, 2, 3 ]\n")
    doc.array("x").append(4)
    assert tomlrt.dumps(doc) == "x = [ 1, 2, 3, 4 ]\n"


def test_inline_array_append_into_empty_padded() -> None:
    doc = tomlrt.parse("x = [ ]\n")
    doc.array("x").append(1)
    assert tomlrt.dumps(doc) == "x = [ 1 ]\n"


def test_inline_array_append_into_empty_flush() -> None:
    doc = tomlrt.parse("x = []\n")
    doc.array("x").append(1)
    assert tomlrt.dumps(doc) == "x = [1]\n"


def test_multiline_array_append_preserves_trailing_comma_layout() -> None:
    src = "x = [\n    1,\n    2,\n    3,\n]\n"
    doc = tomlrt.parse(src)
    doc.array("x").append(4)
    assert tomlrt.dumps(doc) == "x = [\n    1,\n    2,\n    3,\n    4,\n]\n"


def test_multiline_array_append_without_trailing_comma() -> None:
    src = "x = [\n    1,\n    2,\n    3\n]\n"
    doc = tomlrt.parse(src)
    doc.array("x").append(4)
    assert tomlrt.dumps(doc) == "x = [\n    1,\n    2,\n    3,\n    4\n]\n"


def test_array_of_inline_tables_append_preserves_separator() -> None:
    doc = tomlrt.parse("x = [{ a = 1 }, { a = 2 }]\n")
    template = tomlrt.parse("x = { a = 3 }\n").table("x")
    doc.array("x").append(template)
    assert tomlrt.dumps(doc) == "x = [{ a = 1 }, { a = 2 }, { a = 3 }]\n"


def test_array_sort_drops_dangling_trailing_comma() -> None:
    doc = tomlrt.parse("x = [3, 1, 2]\n")
    doc.array("x").sort()
    assert tomlrt.dumps(doc) == "x = [1, 2, 3]\n"


def test_array_sort_preserves_bracket_padding() -> None:
    doc = tomlrt.parse("x = [ 3, 1, 2 ]\n")
    doc.array("x").sort()
    assert tomlrt.dumps(doc) == "x = [ 1, 2, 3 ]\n"


def test_array_sort_compact_stays_compact() -> None:
    doc = tomlrt.parse("x = [3,1,2]\n")
    doc.array("x").sort()
    assert tomlrt.dumps(doc) == "x = [1,2,3]\n"


def test_array_pop_preserves_bracket_padding() -> None:
    doc = tomlrt.parse("x = [ 1, 2, 3 ]\n")
    doc.array("x").pop()
    assert tomlrt.dumps(doc) == "x = [ 1, 2 ]\n"


def test_array_del_first_preserves_bracket_padding() -> None:
    doc = tomlrt.parse("x = [ 1, 2, 3 ]\n")
    del doc.array("x")[0]
    assert tomlrt.dumps(doc) == "x = [ 2, 3 ]\n"


# ---------------------------------------------------------------------------
# Inline table style preservation on insert/delete
# ---------------------------------------------------------------------------


def test_inline_table_insert_preserves_padded_style() -> None:
    doc = tomlrt.parse("x = { a = 1, b = 2 }\n")
    doc.table("x")["c"] = 3
    assert tomlrt.dumps(doc) == "x = { a = 1, b = 2, c = 3 }\n"


def test_inline_table_insert_preserves_compact_style() -> None:
    doc = tomlrt.parse("x={a=1, b=2}\n")
    doc.table("x")["c"] = 3
    assert tomlrt.dumps(doc) == "x={a=1, b=2, c=3}\n"


def test_inline_table_insert_into_single_entry_compact() -> None:
    doc = tomlrt.parse("x={a=1}\n")
    doc.table("x")["b"] = 2
    assert tomlrt.dumps(doc) == "x={a=1, b=2}\n"


def test_inline_table_insert_into_empty_padded() -> None:
    doc = tomlrt.parse("x = { }\n")
    doc.table("x")["a"] = 1
    assert tomlrt.dumps(doc) == "x = { a = 1 }\n"


def test_inline_table_insert_into_empty_flush() -> None:
    doc = tomlrt.parse("x = {}\n")
    doc.table("x")["a"] = 1
    assert tomlrt.dumps(doc) == "x = {a = 1}\n"


def test_inline_table_delete_last_preserves_padding() -> None:
    doc = tomlrt.parse("x = { a = 1, b = 2 }\n")
    del doc.table("x")["b"]
    assert tomlrt.dumps(doc) == "x = { a = 1 }\n"


def test_inline_table_delete_first_preserves_padding() -> None:
    doc = tomlrt.parse("x = { a = 1, b = 2 }\n")
    del doc.table("x")["a"]
    assert tomlrt.dumps(doc) == "x = { b = 2 }\n"


def test_multiline_inline_table_insert_preserves_layout() -> None:
    src = "x = {\n  a = 1,\n  b = 2,\n}\n"
    doc = tomlrt.parse(src)
    doc.table("x")["c"] = 3
    assert tomlrt.dumps(doc) == "x = {\n  a = 1,\n  b = 2,\n  c = 3,\n}\n"


# ---------------------------------------------------------------------------
# A7: blank line before the first ``[table]`` when inserting into the
# implicit pre-header section.
# ---------------------------------------------------------------------------


def test_insert_top_level_kv_adds_blank_before_first_table() -> None:
    doc = tomlrt.parse("[a]\nx = 1\n")
    doc["z"] = 99
    assert tomlrt.dumps(doc) == "z = 99\n\n[a]\nx = 1\n"


def test_insert_top_level_kv_preserves_existing_blank() -> None:
    doc = tomlrt.parse("[a]\nx = 1\n\n[b]\ny = 2\n")
    doc["z"] = 99
    assert tomlrt.dumps(doc) == "z = 99\n\n[a]\nx = 1\n\n[b]\ny = 2\n"


def test_multiple_top_level_kv_inserts_share_one_blank() -> None:
    doc = tomlrt.parse("[a]\nx = 1\n")
    doc["y"] = 1
    doc["z"] = 2
    assert tomlrt.dumps(doc) == "y = 1\nz = 2\n\n[a]\nx = 1\n"


def test_top_level_insert_into_existing_pre_header_section() -> None:
    doc = tomlrt.parse("first = 0\n[a]\nx = 1\n")
    doc["z"] = 99
    # Previously gap between ``first`` and ``[a]`` was zero blank lines;
    # after insertion the new key follows ``first`` packed but a blank
    # line is added before ``[a]`` for visual clarity.
    assert tomlrt.dumps(doc) == "first = 0\nz = 99\n\n[a]\nx = 1\n"


def test_aot_first_two_appends_into_empty_doc_blank_separate() -> None:
    """Bug report: programmatically-built AoT entries rendered glued."""
    doc = tomlrt.parse("")
    aot = doc.set_aot("package")
    aot.append({"name": "foo"})
    aot.append({"name": "bar"})
    assert tomlrt.dumps(doc) == (
        '[[package]]\nname = "foo"\n\n[[package]]\nname = "bar"\n'
    )


def test_aot_append_after_sub_section_blank_separates() -> None:
    """Bug report: an AoT entry following a previous entry's sub-section
    must still be blank-separated from that sub-section."""
    doc = tomlrt.parse("")
    aot = doc.set_aot("package")
    aot.append({"name": "foo", "version": "1.0"})
    aot[-1].set_table("dependencies", {"bar": "^1"})
    aot.append({"name": "baz", "version": "2.0"})
    assert tomlrt.dumps(doc) == (
        '[[package]]\nname = "foo"\nversion = "1.0"\n'
        "\n"
        '[package.dependencies]\nbar = "^1"\n'
        "\n"
        '[[package]]\nname = "baz"\nversion = "2.0"\n'
    )
