"""Smoke tests: parse, value access, exact round-trip."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from textwrap import dedent

import pytest

import tomlrt
from _toml_str import td

UTC = timezone.utc

# ---------------------------------------------------------------------------
# Round-trip corpus: dumps(parse(s)) == s, byte-for-byte.
# ---------------------------------------------------------------------------

ROUND_TRIP_CORPUS: list[str] = [
    "",
    "\n",
    "# just a comment\n",
    "key = 1\n",
    "key = 1",  # no trailing newline
    'name = "Tom"\n',
    "title = 'literal'\n",
    "pi = 3.14\n",
    "neg = -7\n",
    "hex = 0xCAFE_BABE\n",
    "oct = 0o755\n",
    "bin = 0b1010\n",
    "huge = 1_000_000\n",
    "flag = true\nother = false\n",
    "arr = [1, 2, 3]\n",
    'mixed = [1, "two", 3.0]\n',
    "nested = [[1, 2], [3, 4]]\n",
    "point = { x = 1, y = 2 }\n",
    dedent(
        """\
        # leading file comment
        title = "TOML Example"

        [owner]
        name = "Tom Preston-Werner"
        dob = 1979-05-27T07:32:00-08:00 # First class dates

        [database]
        enabled = true
        ports = [ 8000, 8001, 8002 ]
        data = [ ["delta", "phi"], [3.14] ]
        temp_targets = { cpu = 79.5, case = 72.0 }

        [servers]

          [servers.alpha]
          ip = "10.0.0.1"
          role = "frontend"

          [servers.beta]
          ip = "10.0.0.2"
          role = "backend"
        """,
    ),
    dedent(
        """\
        [[products]]
        name = "Hammer"
        sku = 738594937

        [[products]]  # empty table within the array

        [[products]]
        name = "Nail"
        sku = 284758393

        color = "gray"
        """,
    ),
    dedent(
        """\
        # dotted keys
        physical.color = "orange"
        physical.shape = "round"
        site."google.com" = true
        """,
    ),
    dedent(
        '''\
        str1 = """
        Roses are red
        Violets are blue"""
        str2 = """\\
            The quick brown \\
            fox jumps over \\
            the lazy dog.\\
            """
        path = 'C:\\Users\\nodejs\\templates'
        regex2 = \'\'\'I [dw]on\'t need \\d{2} apples\'\'\'
        '''
    ),
    dedent(
        """\
        odt1 = 1979-05-27T07:32:00Z
        odt2 = 1979-05-27T00:32:00-07:00
        ldt1 = 1979-05-27T07:32:00
        ldt2 = 1979-05-27T00:32:00.999999
        ld1 = 1979-05-27
        lt1 = 07:32:00
        lt2 = 00:32:00.999999
        """,
    ),
]


@pytest.mark.parametrize("src", ROUND_TRIP_CORPUS)
def test_round_trip(src: str) -> None:
    doc = tomlrt.parse(src)
    assert tomlrt.dumps(doc) == src


# ---------------------------------------------------------------------------
# Value access
# ---------------------------------------------------------------------------


def test_basic_value_access() -> None:
    doc = tomlrt.parse(
        dedent(
            """\
            title = "Example"
            count = 42
            ratio = 0.75
            on = true

            [owner]
            name = "Tom"
            """,
        ),
    )
    assert doc["title"] == "Example"
    assert doc["count"] == 42
    assert doc["ratio"] == 0.75
    assert doc["on"] is True
    owner = doc["owner"]
    assert isinstance(owner, tomlrt.Table)
    assert owner["name"] == "Tom"


def test_arrays_are_real_lists() -> None:
    doc = tomlrt.parse("xs = [1, 2, 3]\n")
    xs = doc["xs"]
    assert isinstance(xs, list)
    assert xs == [1, 2, 3]


def test_inline_table_access() -> None:
    doc = tomlrt.parse("p = { x = 1, y = 2 }\n")
    p = doc["p"]
    assert isinstance(p, tomlrt.Table)
    assert dict(p) == {"x": 1, "y": 2}


def test_nested_tables_and_dotted_keys() -> None:
    src = dedent(
        """\
        [a.b.c]
        v = 1

        [a]
        x = 2
        """,
    )
    doc = tomlrt.parse(src)
    a = doc["a"]
    assert isinstance(a, tomlrt.Table)
    assert a["x"] == 2
    b = a["b"]
    assert isinstance(b, tomlrt.Table)
    c = b["c"]
    assert isinstance(c, tomlrt.Table)
    assert c["v"] == 1


def test_array_of_tables() -> None:
    src = dedent(
        """\
        [[products]]
        name = "Hammer"

        [[products]]
        name = "Nail"
        """,
    )
    doc = tomlrt.parse(src)
    products = doc["products"]
    assert isinstance(products, list)
    assert len(products) == 2
    p0 = products[0]
    p1 = products[1]
    assert isinstance(p0, tomlrt.Table)
    assert isinstance(p1, tomlrt.Table)
    assert p0["name"] == "Hammer"
    assert p1["name"] == "Nail"


def test_datetime_values() -> None:
    doc = tomlrt.parse(
        dedent(
            """\
            odt = 1979-05-27T07:32:00Z
            ldt = 1979-05-27T07:32:00
            ld = 1979-05-27
            lt = 07:32:00
            """,
        ),
    )
    assert doc["odt"] == datetime(1979, 5, 27, 7, 32, 0, tzinfo=UTC)
    assert doc["ldt"] == datetime(1979, 5, 27, 7, 32, 0)  # noqa: DTZ001 - local datetime is naive by spec
    assert doc["ld"] == date(1979, 5, 27)
    assert doc["lt"] == time(7, 32, 0)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src",
    [
        "key =\n",
        "= 1\n",
        '"unterminated\n',
        "x = 01\n",
        "x = 1__0\n",
        td("""
            [a]
            x = 1
            [a]
            x = 2
            """),
        "x = 1\nx = 2\n",
        "x = 1.\n",
        "x = .1\n",
    ],
)
def test_parse_errors(src: str) -> None:
    with pytest.raises(tomlrt.TOMLParseError):
        tomlrt.parse(src)


def test_deep_array_nesting_raises_parse_error_not_recursionerror() -> None:
    payload = "x = " + "[" * 500 + "1" + "]" * 500 + "\n"
    with pytest.raises(tomlrt.TOMLParseError, match="nesting exceeds"):
        tomlrt.parse(payload)


def test_deep_inline_table_nesting_raises_parse_error() -> None:
    payload = "x = " + "{a=" * 500 + "1" + "}" * 500 + "\n"
    with pytest.raises(tomlrt.TOMLParseError, match="nesting exceeds"):
        tomlrt.parse(payload)


def test_inline_table_dotted_key_conflict_reports_inline_position() -> None:
    # The conflict between `x = 1` and `x.y = 2` lives on line 1 inside
    # the inline table; the error must point there, not at the start of
    # the next line.
    src = "a = { x = 1, x.y = 2 }\n"
    with pytest.raises(tomlrt.TOMLParseError) as exc_info:
        tomlrt.parse(src)
    assert exc_info.value.line == 1
    # The conflicting "x.y" key starts at column 14 (1-based).
    assert exc_info.value.col == 14


def test_moderate_array_nesting_still_parses() -> None:
    payload = "x = " + "[" * 50 + "1" + "]" * 50 + "\n"
    doc = tomlrt.parse(payload)
    assert tomlrt.dumps(doc) == payload


# ---------------------------------------------------------------------------
# Out-of-order tables — iteration order should match tomllib (first-appearance)
# ---------------------------------------------------------------------------


def test_iteration_order_child_section_before_parent() -> None:
    src = td("""
        [a.b]
        x = 1
        [a]
        y = 2
        """)
    doc = tomlrt.parse(src)
    a = doc.table("a")
    assert list(a) == ["b", "y"]


def test_iteration_order_parent_then_child_then_more_direct_keys() -> None:
    # With a single [a] block plus a sub-section after it, direct keys
    # come first because they appear first physically.
    src = td("""
        [a]
        x = 1
        [a.b]
        y = 2
        """)
    doc = tomlrt.parse(src)
    assert list(doc.table("a")) == ["x", "b"]


def test_iteration_order_sibling_interleaved_between_parent_and_child() -> None:
    src = td("""
        [a]
        x = 1
        [b]
        y = 2
        [a.sub]
        z = 3
        """)
    doc = tomlrt.parse(src)
    assert list(doc) == ["a", "b"]
    assert list(doc.table("a")) == ["x", "sub"]


def test_iteration_order_aot_then_sibling_then_more_aot() -> None:
    src = td("""
        [[fruits]]
        name = "apple"
        [[other]]
        n = 1
        [[fruits]]
        name = "banana"
        """)
    doc = tomlrt.parse(src)
    assert list(doc) == ["fruits", "other"]
    fruits = doc.aot("fruits")
    assert [t["name"] for t in fruits] == ["apple", "banana"]
