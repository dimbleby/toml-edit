"""Tests for value synthesis (``value_to_node``) and the public file I/O.

These cover the corners of ``_synthesise.py`` and ``_public.py`` that
the rest of the suite skirts past: every escape branch in basic
strings, every scalar flavour accepted by ``value_to_node``, and the
``loads`` / ``load`` / ``dump`` wrappers.
"""

from __future__ import annotations

import io
import math
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING

import pytest

import tomle
from tomle._errors import TOMLEditError

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Public I/O wrappers
# ---------------------------------------------------------------------------


def test_loads_is_alias_for_parse() -> None:
    src = "x = 1\ny = 'hi'\n"
    a = tomle.loads(src)
    b = tomle.parse(src)
    assert tomle.dumps(a) == tomle.dumps(b) == src


def test_load_from_binary_stream() -> None:
    fp = io.BytesIO(b"name = 'ada'\n")
    doc = tomle.load(fp)
    assert doc["name"] == "ada"


def test_load_from_real_file_path(tmp_path: Path) -> None:
    p = tmp_path / "doc.toml"
    p.write_text("k = 42\n", encoding="utf-8")
    with p.open("rb") as fp:
        doc = tomle.load(fp)
    assert doc["k"] == 42


def test_load_rejects_text_stream() -> None:
    fp = io.StringIO("port = 8080\n")
    with pytest.raises(TypeError, match="binary"):
        tomle.load(fp)  # type: ignore[arg-type]


def test_load_preserves_crlf_line_endings(tmp_path: Path) -> None:
    p = tmp_path / "win.toml"
    p.write_bytes(b"a = 1\r\nb = 2\r\n")
    with p.open("rb") as fp:
        doc = tomle.load(fp)
    out = io.BytesIO()
    tomle.dump(doc, out)
    assert out.getvalue() == b"a = 1\r\nb = 2\r\n"


def test_dump_writes_to_binary_stream() -> None:
    doc = tomle.parse("x = 1\n")
    out = io.BytesIO()
    tomle.dump(doc, out)
    assert out.getvalue() == b"x = 1\n"


def test_dump_emits_utf8_for_non_ascii() -> None:
    doc = tomle.parse("name = 'café'\n")
    out = io.BytesIO()
    tomle.dump(doc, out)
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
def test_string_escape_emits_canonical_form(py_value: str, expected_quoted: str) -> None:
    doc = tomle.parse("x = 0\n")
    doc["x"] = py_value
    out = tomle.dumps(doc)
    assert out == f"x = {expected_quoted}\n"
    # And it round-trips back to the same Python value.
    assert tomle.parse(out)["x"] == py_value


# ---------------------------------------------------------------------------
# value_to_node: every accepted Python type
# ---------------------------------------------------------------------------


def test_assign_bool_renders_as_toml_bool() -> None:
    doc = tomle.parse("x = 0\n")
    doc["x"] = True
    doc["y"] = False
    out = tomle.dumps(doc)
    assert "x = true" in out
    assert "y = false" in out
    re = tomle.parse(out)
    assert re["x"] is True
    assert re["y"] is False


def test_assign_int_renders_decimal() -> None:
    doc = tomle.parse("x = 0\n")
    doc["x"] = -123
    assert tomle.dumps(doc) == "x = -123\n"


def test_assign_float_basic_gets_dot_zero_when_missing() -> None:
    doc = tomle.parse("x = 0\n")
    doc["x"] = 3.0
    out = tomle.dumps(doc)
    # repr(3.0) is "3.0" already, but values like 1e10 round-trip via repr
    # which emits no dot; the helper appends one.
    assert "x = 3.0" in out
    assert tomle.parse(out)["x"] == 3.0


def test_assign_float_scientific_no_dot_added() -> None:
    doc = tomle.parse("x = 0\n")
    doc["x"] = 1e20
    out = tomle.dumps(doc)
    re = tomle.parse(out)
    assert re["x"] == 1e20


def test_assign_float_inf_and_nan() -> None:
    doc = tomle.parse("x = 0\n")
    doc["x"] = math.inf
    doc["y"] = -math.inf
    doc["z"] = math.nan
    out = tomle.dumps(doc)
    assert "x = inf" in out
    assert "y = -inf" in out
    assert "z = nan" in out
    re = tomle.parse(out)
    assert re["x"] == math.inf
    assert re["y"] == -math.inf
    assert math.isnan(re["z"])  # type: ignore[arg-type]


def test_assign_local_date() -> None:
    doc = tomle.parse("x = 0\n")
    doc["x"] = date(2024, 7, 4)
    out = tomle.dumps(doc)
    assert "x = 2024-07-04" in out
    assert tomle.parse(out)["x"] == date(2024, 7, 4)


def test_assign_local_time() -> None:
    doc = tomle.parse("x = 0\n")
    doc["x"] = time(13, 30, 45)
    out = tomle.dumps(doc)
    assert "x = 13:30:45" in out
    assert tomle.parse(out)["x"] == time(13, 30, 45)


def test_assign_local_datetime() -> None:
    doc = tomle.parse("x = 0\n")
    doc["x"] = datetime(2024, 7, 4, 12, 0, 0)  # noqa: DTZ001
    out = tomle.dumps(doc)
    assert "x = 2024-07-04T12:00:00" in out
    assert tomle.parse(out)["x"] == datetime(2024, 7, 4, 12, 0, 0)  # noqa: DTZ001


def test_assign_offset_datetime() -> None:
    doc = tomle.parse("x = 0\n")
    tz = timezone(timedelta(hours=2))
    doc["x"] = datetime(2024, 7, 4, 12, 0, 0, tzinfo=tz)
    out = tomle.dumps(doc)
    re_value = tomle.parse(out)["x"]
    assert isinstance(re_value, datetime)
    assert re_value == datetime(2024, 7, 4, 12, 0, 0, tzinfo=tz)


def test_assign_plain_list_becomes_inline_array() -> None:
    doc = tomle.parse("x = 0\n")
    doc["x"] = [1, 2, 3]
    out = tomle.dumps(doc)
    re = tomle.parse(out)
    assert list(re.array("x")) == [1, 2, 3]


def test_assign_plain_dict_becomes_inline_table() -> None:
    doc = tomle.parse("x = 0\n")
    doc["x"] = {"a": 1, "b": "two"}
    out = tomle.dumps(doc)
    re = tomle.parse(out)
    tbl = re.table("x")
    assert tbl["a"] == 1
    assert tbl["b"] == "two"


def test_assign_nested_dict_in_list() -> None:
    doc = tomle.parse("x = 0\n")
    doc["x"] = [{"a": 1}, {"a": 2}]
    out = tomle.dumps(doc)
    re = tomle.parse(out)
    arr = re.array("x")
    assert arr.table(0)["a"] == 1
    assert arr.table(1)["a"] == 2


def test_assign_existing_array_deep_copies() -> None:
    src = tomle.parse("source = [1, 2, 3]\n")
    dest = tomle.parse("dest = []\n")
    dest["dest"] = src.array("source")
    src.array("source")[0] = 99
    # The mutation on `source` must not leak into `dest`.
    assert list(dest.array("dest")) == [1, 2, 3]


def test_assign_existing_inline_table_deep_copies() -> None:
    src = tomle.parse("source = {a = 1}\n")
    dest = tomle.parse("dest = {}\n")
    dest["dest"] = src.table("source")
    src.table("source")["a"] = 99
    assert dest.table("dest")["a"] == 1


def test_assign_unsupported_type_raises() -> None:
    doc = tomle.parse("x = 0\n")
    with pytest.raises(TOMLEditError, match="Cannot convert"):
        doc["x"] = object()


def test_assign_aot_inline_raises() -> None:
    src = tomle.parse(
        "[[products]]\nname = 'a'\n[[products]]\nname = 'b'\n",
    )
    dest = tomle.parse("dest = 0\n")
    with pytest.raises(TOMLEditError, match="array-of-tables"):
        dest["dest"] = src.aot("products")
