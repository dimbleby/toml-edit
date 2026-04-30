"""Tests for value synthesis (``value_to_node``) and the public file I/O.

These cover the corners of ``_synthesise.py`` and ``_public.py`` that
the rest of the suite skirts past: every escape branch in basic
strings, every scalar flavour accepted by ``value_to_node``, and the
``loads`` / ``load`` / ``dump`` wrappers.
"""

from __future__ import annotations

import io
import math
from copy import copy, deepcopy
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING

import pytest

import tomlrt
from _toml_str import td
from tomlrt import Document, Table

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Public I/O wrappers
# ---------------------------------------------------------------------------


def test_loads_is_alias_for_parse() -> None:
    src = "x = 1\ny = 'hi'\n"
    a = tomlrt.loads(src)
    b = tomlrt.parse(src)
    assert tomlrt.dumps(a) == tomlrt.dumps(b) == src


def test_load_from_binary_stream() -> None:
    fp = io.BytesIO(b"name = 'ada'\n")
    doc = tomlrt.load(fp)
    assert doc["name"] == "ada"


def test_load_from_real_file_path(tmp_path: Path) -> None:
    p = tmp_path / "doc.toml"
    p.write_text("k = 42\n", encoding="utf-8")
    with p.open("rb") as fp:
        doc = tomlrt.load(fp)
    assert doc["k"] == 42


def test_load_rejects_text_stream() -> None:
    fp = io.StringIO("port = 8080\n")
    with pytest.raises(TypeError, match="binary"):
        tomlrt.load(fp)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_load_preserves_crlf_line_endings(tmp_path: Path) -> None:
    p = tmp_path / "win.toml"
    p.write_bytes(b"a = 1\r\nb = 2\r\n")
    with p.open("rb") as fp:
        doc = tomlrt.load(fp)
    out = io.BytesIO()
    tomlrt.dump(doc, out)
    assert out.getvalue() == b"a = 1\r\nb = 2\r\n"


def test_crlf_document_keeps_crlf_after_mutation() -> None:
    doc = tomlrt.loads("a = 1\r\nb = 2\r\n")
    doc["c"] = 3
    assert tomlrt.dumps(doc) == "a = 1\r\nb = 2\r\nc = 3\r\n"


def test_dump_writes_to_binary_stream() -> None:
    doc = tomlrt.parse("x = 1\n")
    out = io.BytesIO()
    tomlrt.dump(doc, out)
    assert out.getvalue() == b"x = 1\n"


def test_dump_emits_utf8_for_non_ascii() -> None:
    doc = tomlrt.parse("name = 'café'\n")
    out = io.BytesIO()
    tomlrt.dump(doc, out)
    assert out.getvalue() == "name = 'café'\n".encode()


# ---------------------------------------------------------------------------
# String escaping (every branch in _escape_basic_string)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("py_value", "expected_quoted"),
    [
        ("plain", '"plain"'),
        ("back\\slash", '"back\\\\slash"'),
        ('with"quote', '"with\\"quote"'),
        ("line\nbreak", '"line\\nbreak"'),
        ("carriage\rreturn", '"carriage\\rreturn"'),
        ("tab\there", '"tab\\there"'),
        ("bell\bback", '"bell\\bback"'),
        ("form\ffeed", '"form\\ffeed"'),
        ("ctrl\x01char", '"ctrl\\u0001char"'),
        ("del\x7fchar", '"del\\u007Fchar"'),
    ],
)
def test_string_escape_emits_canonical_form(
    py_value: str, expected_quoted: str
) -> None:
    doc = tomlrt.parse("x = 0\n")
    doc["x"] = py_value
    out = tomlrt.dumps(doc)
    assert out == f"x = {expected_quoted}\n"
    # And it round-trips back to the same Python value.
    assert tomlrt.parse(out)["x"] == py_value


# ---------------------------------------------------------------------------
# value_to_node: every accepted Python type
# ---------------------------------------------------------------------------


def test_assign_bool_renders_as_toml_bool() -> None:
    doc = tomlrt.parse("x = 0\n")
    doc["x"] = True
    doc["y"] = False
    out = tomlrt.dumps(doc)
    assert "x = true" in out
    assert "y = false" in out
    re = tomlrt.parse(out)
    assert re["x"] is True
    assert re["y"] is False


def test_assign_int_renders_decimal() -> None:
    doc = tomlrt.parse("x = 0\n")
    doc["x"] = -123
    assert tomlrt.dumps(doc) == "x = -123\n"


def test_assign_float_basic_gets_dot_zero_when_missing() -> None:
    doc = tomlrt.parse("x = 0\n")
    doc["x"] = 3.0
    out = tomlrt.dumps(doc)
    # repr(3.0) is "3.0" already, but values like 1e10 round-trip via repr
    # which emits no dot; the helper appends one.
    assert "x = 3.0" in out
    assert tomlrt.parse(out)["x"] == 3.0


def test_assign_float_scientific_no_dot_added() -> None:
    doc = tomlrt.parse("x = 0\n")
    doc["x"] = 1e20
    out = tomlrt.dumps(doc)
    re = tomlrt.parse(out)
    assert re["x"] == 1e20


def test_assign_float_inf_and_nan() -> None:
    doc = tomlrt.parse("x = 0\n")
    doc["x"] = math.inf
    doc["y"] = -math.inf
    doc["z"] = math.nan
    out = tomlrt.dumps(doc)
    assert "x = inf" in out
    assert "y = -inf" in out
    assert "z = nan" in out
    re = tomlrt.parse(out)
    assert re["x"] == math.inf
    assert re["y"] == -math.inf
    assert math.isnan(re["z"])


def test_assign_local_date() -> None:
    doc = tomlrt.parse("x = 0\n")
    doc["x"] = date(2024, 7, 4)
    out = tomlrt.dumps(doc)
    assert "x = 2024-07-04" in out
    assert tomlrt.parse(out)["x"] == date(2024, 7, 4)


def test_assign_local_time() -> None:
    doc = tomlrt.parse("x = 0\n")
    doc["x"] = time(13, 30, 45)
    out = tomlrt.dumps(doc)
    assert "x = 13:30:45" in out
    assert tomlrt.parse(out)["x"] == time(13, 30, 45)


def test_assign_local_datetime() -> None:
    doc = tomlrt.parse("x = 0\n")
    doc["x"] = datetime(2024, 7, 4, 12, 0, 0)  # noqa: DTZ001
    out = tomlrt.dumps(doc)
    assert "x = 2024-07-04T12:00:00" in out
    assert tomlrt.parse(out)["x"] == datetime(2024, 7, 4, 12, 0, 0)  # noqa: DTZ001


def test_assign_offset_datetime() -> None:
    doc = tomlrt.parse("x = 0\n")
    tz = timezone(timedelta(hours=2))
    doc["x"] = datetime(2024, 7, 4, 12, 0, 0, tzinfo=tz)
    out = tomlrt.dumps(doc)
    re_value = tomlrt.parse(out)["x"]
    assert isinstance(re_value, datetime)
    assert re_value == datetime(2024, 7, 4, 12, 0, 0, tzinfo=tz)


def test_assign_plain_list_becomes_inline_array() -> None:
    doc = tomlrt.parse("x = 0\n")
    doc["x"] = [1, 2, 3]
    out = tomlrt.dumps(doc)
    re = tomlrt.parse(out)
    assert list(re.array("x")) == [1, 2, 3]


def test_assign_plain_dict_becomes_inline_table() -> None:
    doc = tomlrt.parse("x = 0\n")
    doc["x"] = {"a": 1, "b": "two"}
    out = tomlrt.dumps(doc)
    re = tomlrt.parse(out)
    tbl = re.table("x")
    assert tbl["a"] == 1
    assert tbl["b"] == "two"


def test_assign_tuple_rejected() -> None:
    doc = tomlrt.parse("x = 0\n")
    with pytest.raises(TypeError, match="tuple"):
        doc["x"] = (1, 2, 3)


def test_assign_mappingproxy_becomes_inline_table() -> None:
    from types import MappingProxyType  # noqa: PLC0415

    doc = tomlrt.parse("x = 0\n")
    doc["x"] = MappingProxyType({"a": 1, "b": 2})
    out = tomlrt.dumps(doc)
    re = tomlrt.parse(out)
    tbl = re.table("x")
    assert tbl["a"] == 1
    assert tbl["b"] == 2


def test_assign_bytes_rejected() -> None:
    doc = tomlrt.parse("x = 0\n")
    with pytest.raises(TypeError, match="bytes"):
        doc["x"] = b"hi"


def test_assign_nested_dict_in_list() -> None:
    doc = tomlrt.parse("x = 0\n")
    doc["x"] = [{"a": 1}, {"a": 2}]
    out = tomlrt.dumps(doc)
    re = tomlrt.parse(out)
    arr = re.array("x")
    assert arr.table(0)["a"] == 1
    assert arr.table(1)["a"] == 2


def test_assign_existing_array_deep_copies() -> None:
    src = tomlrt.parse("source = [1, 2, 3]\n")
    dest = tomlrt.parse("dest = []\n")
    dest["dest"] = src.array("source")
    src.array("source")[0] = 99
    # The mutation on `source` must not leak into `dest`.
    assert list(dest.array("dest")) == [1, 2, 3]


def test_assign_existing_inline_table_deep_copies() -> None:
    src = tomlrt.parse("source = {a = 1}\n")
    dest = tomlrt.parse("dest = {}\n")
    dest["dest"] = src.table("source")
    src.table("source")["a"] = 99
    assert dest.table("dest")["a"] == 1


def test_assign_unsupported_type_raises() -> None:
    doc = tomlrt.parse("x = 0\n")
    with pytest.raises(TypeError, match="Cannot convert"):
        doc["x"] = object()


def test_assign_aot_over_scalar() -> None:
    src = tomlrt.parse(
        td("""
            [[products]]
            name = 'a'
            [[products]]
            name = 'b'
            """),
    )
    dest = tomlrt.parse("dest = 0\n")
    dest["dest"] = src.aot("products")
    assert tomlrt.loads(tomlrt.dumps(dest)) == {
        "dest": [{"name": "a"}, {"name": "b"}],
    }


def test_document_factory_returns_empty_document() -> None:
    doc = Document()
    assert isinstance(doc, tomlrt.Document)
    assert len(doc) == 0
    assert tomlrt.dumps(doc) == ""


def test_document_factory_is_independent_of_other_calls() -> None:
    a = Document()
    b = Document()
    a["x"] = 1
    assert "x" not in b
    assert tomlrt.dumps(b) == ""


def test_document_factory_supports_full_build_and_dump() -> None:
    doc = Document()
    doc["title"] = "demo"
    doc["server"] = Table.section({"port": 8080})
    out = tomlrt.dumps(doc)
    parsed = tomlrt.parse(out)
    assert parsed["title"] == "demo"
    server = parsed.table("server")
    assert server["port"] == 8080


def test_document_factory_with_data_uses_sections_for_nested_mappings() -> None:
    doc = Document({"server": {"port": 8080, "host": "localhost"}})
    out = tomlrt.dumps(doc)
    assert "[server]" in out
    assert "{" not in out  # no inline tables
    assert tomlrt.parse(out) == {"server": {"port": 8080, "host": "localhost"}}


def test_document_factory_with_data_uses_aot_for_list_of_mappings() -> None:
    doc = Document(
        {"package": [{"name": "foo"}, {"name": "bar"}]},
    )
    out = tomlrt.dumps(doc)
    assert out.count("[[package]]") == 2
    assert tomlrt.parse(out) == {"package": [{"name": "foo"}, {"name": "bar"}]}


def test_document_factory_with_data_keeps_leaf_arrays_inline() -> None:
    doc = Document({"xs": [1, 2, 3]})
    out = tomlrt.dumps(doc)
    assert "[[" not in out  # not promoted to AoT
    assert tomlrt.parse(out) == {"xs": [1, 2, 3]}


def test_document_factory_with_data_keeps_top_level_scalars_at_top() -> None:
    doc = Document({"title": "demo", "server": {"port": 8080}})
    out = tomlrt.dumps(doc)
    # Top-level scalar must precede the [server] section header.
    assert out.index('title = "demo"') < out.index("[server]")


def test_document_factory_with_data_recurses_deeply() -> None:
    data = {
        "tool": {
            "poetry": {
                "name": "demo",
                "dependencies": {"requests": "^2.0"},
            },
        },
    }
    doc = Document(data)
    out = tomlrt.dumps(doc)
    assert "[tool.poetry]" in out
    assert "[tool.poetry.dependencies]" in out
    assert tomlrt.parse(out) == data


def test_document_factory_with_data_aot_with_nested_table() -> None:
    data = {
        "package": [
            {"name": "foo", "version": "1.0", "dep": {"x": 1}},
            {"name": "bar", "version": "2.0"},
        ],
    }
    doc = Document(data)
    out = tomlrt.dumps(doc)
    assert tomlrt.parse(out) == data


def test_document_factory_with_empty_list_stays_inline_empty_array() -> None:
    doc = Document({"xs": []})
    out = tomlrt.dumps(doc)
    assert "[[" not in out
    assert tomlrt.parse(out) == {"xs": []}


def test_document_factory_with_data_does_not_share_mutable_state() -> None:
    data: dict[str, object] = {"server": {"port": 8080}}
    doc = Document(data)
    server_dict = data["server"]
    assert isinstance(server_dict, dict)
    server_dict["port"] = 9999  # mutate the source after construction
    server = doc.table("server")
    assert server["port"] == 8080


def test_document_factory_emits_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match="tomlrt.Document"):
        doc = tomlrt.document({"x": 1})
    assert isinstance(doc, Document)
    assert doc["x"] == 1


def test_deepcopy_preserves_document_structure() -> None:

    src = td("""
        [a]
        x = 1

        [[b]]
        y = 2
        [[b]]
        y = 3
        """)
    doc1 = tomlrt.loads(src)
    doc2 = deepcopy(doc1)
    assert tomlrt.dumps(doc2) == src


def test_deepcopy_yields_independent_document() -> None:

    src = "[a]\nx = 1\n"
    doc1 = tomlrt.loads(src)
    doc2 = deepcopy(doc1)
    doc2["a"]["x"] = 99
    assert doc1["a"]["x"] == 1
    assert doc2["a"]["x"] == 99
    # And the unmutated half stays format-preserved.
    assert tomlrt.dumps(doc1) == src


def test_copy_yields_independent_document() -> None:

    src = "[a]\nx = 1\n"
    doc1 = tomlrt.loads(src)
    doc2 = copy(doc1)
    doc2["a"]["x"] = 99
    assert doc1["a"]["x"] == 1


def test_deepcopy_table_subview_is_independent_and_round_trips() -> None:

    src = td("""
        [t]
        x = 1
        y = 2
        """)
    doc = tomlrt.loads(src)
    t = doc.table("t")
    t2 = deepcopy(t)
    assert dict(t2) == {"x": 1, "y": 2}
    t2["x"] = 99
    assert t["x"] == 1
    assert t2["x"] == 99
    # The original document's bytes are unaffected.
    assert tomlrt.dumps(doc) == src


def test_deepcopy_array_subview_does_not_double_cst() -> None:

    src = "xs = [1, 2, 3]\n"
    doc = tomlrt.loads(src)
    arr = doc.array("xs")
    arr2 = deepcopy(arr)
    assert list(arr2) == [1, 2, 3]
    # The CST must not have doubled items: appending one and rendering
    # the array node in isolation should reflect exactly four entries.
    arr2.append(4)
    assert list(arr2) == [1, 2, 3, 4]
    # Re-attach the detached copy to a fresh document and render through
    # the public API: any doubled CST items would surface here.
    fresh = tomlrt.parse("")
    fresh["ys"] = arr2
    assert tomlrt.dumps(fresh) == "ys = [1, 2, 3, 4]\n"
    # Original is untouched.
    assert tomlrt.dumps(doc) == src


def test_deepcopy_aot_subview_preserves_length() -> None:

    src = td("""
        [[t]]
        x = 1
        [[t]]
        x = 2
        """)
    doc = tomlrt.loads(src)
    aot = doc.aot("t")
    aot2 = deepcopy(aot)
    assert len(aot2) == 2
    assert [dict(e) for e in aot2] == [{"x": 1}, {"x": 2}]
    # Mutations on the copy do not leak.
    aot2[0]["x"] = 99
    assert aot[0]["x"] == 1
    assert tomlrt.dumps(doc) == src


def test_copy_array_subview_does_not_double_cst() -> None:

    src = "xs = [1, 2, 3]\n"
    doc = tomlrt.loads(src)
    arr = doc.array("xs")
    arr2 = copy(arr)
    arr2.append(4)
    fresh = tomlrt.parse("")
    fresh["ys"] = arr2
    assert tomlrt.dumps(fresh) == "ys = [1, 2, 3, 4]\n"
    assert tomlrt.dumps(doc) == src


def test_copy_aot_subview_preserves_length() -> None:

    src = td("""
        [[t]]
        x = 1
        [[t]]
        x = 2
        """)
    doc = tomlrt.loads(src)
    aot = doc.aot("t")
    aot2 = copy(aot)
    assert len(aot2) == 2


def test_copy_table_subview_is_independent() -> None:
    src = td("""
        [t]
        x = 1
        """)
    doc = tomlrt.loads(src)
    t = doc.table("t")
    t2 = copy(t)
    t2["x"] = 99
    assert t["x"] == 1
    assert t2["x"] == 99
    assert tomlrt.dumps(doc) == src


def test_deepcopy_table_subview_supports_nested_mutation() -> None:

    src = td("""
        [t]
        [t.inner]
        x = 1
        """)
    doc = tomlrt.loads(src)
    t = doc.table("t")
    t2 = deepcopy(t)
    t2.table("inner")["x"] = 42
    assert t.table("inner")["x"] == 1
    assert tomlrt.dumps(doc) == src
