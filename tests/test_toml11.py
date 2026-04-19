"""TOML 1.1.0 spec additions.

Each test pins one new behaviour from the v1.1.0 spec
(<https://toml.io/en/v1.1.0>). Round-trip preservation is asserted
explicitly because that is the whole point of this library.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta

import pytest

import tomle

# ---------------------------------------------------------------------------
# String escapes: \xHH and \e
# ---------------------------------------------------------------------------


def test_basic_string_xhh_escape_decodes() -> None:
    src = r"""a = "hi \xe9 there"
"""
    doc = tomle.parse(src)
    assert doc["a"] == "hi \u00e9 there"


def test_basic_string_xhh_escape_round_trips_verbatim() -> None:
    src = r"""a = "hi \xe9 there"
"""
    assert tomle.dumps(tomle.parse(src)) == src


def test_basic_string_x00_and_x7f_decode() -> None:
    src = r"""a = "\x00\x7f"
"""
    doc = tomle.parse(src)
    assert doc["a"] == "\x00\x7f"


def test_basic_string_e_escape_decodes_to_esc() -> None:
    src = r"""a = "esc\e"
"""
    doc = tomle.parse(src)
    assert doc["a"] == "esc\x1b"


def test_basic_string_e_escape_round_trips_verbatim() -> None:
    src = r"""a = "esc\e"
"""
    assert tomle.dumps(tomle.parse(src)) == src


def test_multiline_basic_string_xhh_escape() -> None:
    src = '''a = """\\xe9 line"""
'''
    doc = tomle.parse(src)
    assert doc["a"] == "\u00e9 line"
    assert tomle.dumps(doc) == src


def test_multiline_basic_string_e_escape() -> None:
    src = '''a = """before\\eafter"""
'''
    doc = tomle.parse(src)
    assert doc["a"] == "before\x1bafter"
    assert tomle.dumps(doc) == src


def test_xhh_uppercase_hex_digits() -> None:
    src = r"""a = "\xE9"
"""
    doc = tomle.parse(src)
    assert doc["a"] == "\u00e9"
    assert tomle.dumps(doc) == src


@pytest.mark.parametrize("bad", [r'a = "\xZZ"', r'a = "\x1"', 'a = "\\x"'])
def test_invalid_xhh_escape_raises(bad: str) -> None:
    with pytest.raises(tomle.TOMLParseError):
        tomle.parse(bad + "\n")


def test_xhh_does_not_apply_to_literal_strings() -> None:
    # Literal strings have no escapes; \xe9 is two characters.
    src = "a = '\\xe9'\n"
    doc = tomle.parse(src)
    assert doc["a"] == "\\xe9"
    assert tomle.dumps(doc) == src


def test_e_does_not_apply_to_literal_strings() -> None:
    src = "a = '\\e'\n"
    doc = tomle.parse(src)
    assert doc["a"] == "\\e"
    assert tomle.dumps(doc) == src


# ---------------------------------------------------------------------------
# Optional seconds in datetime / time
# ---------------------------------------------------------------------------


def test_local_time_no_seconds_decodes() -> None:
    doc = tomle.parse("t = 07:32\n")
    assert doc["t"] == time(7, 32, 0)


def test_local_time_no_seconds_round_trips() -> None:
    src = "t = 07:32\n"
    assert tomle.dumps(tomle.parse(src)) == src


def test_local_datetime_no_seconds() -> None:
    src = "ldt = 1979-05-27T07:32\n"
    doc = tomle.parse(src)
    # Local datetime carries no tz — equality with a naive datetime is fine.
    assert doc["ldt"] == datetime(1979, 5, 27, 7, 32, 0)  # noqa: DTZ001
    assert tomle.dumps(doc) == src


def test_offset_datetime_no_seconds_z() -> None:
    src = "odt = 1979-05-27 07:32Z\n"
    doc = tomle.parse(src)
    assert doc["odt"] == datetime(1979, 5, 27, 7, 32, 0, tzinfo=UTC)
    assert tomle.dumps(doc) == src


def test_offset_datetime_no_seconds_explicit_offset() -> None:
    src = "odt = 1979-05-27 07:32-07:00\n"
    doc = tomle.parse(src)
    assert tomle.dumps(doc) == src
    val = doc["odt"]
    assert isinstance(val, datetime)
    assert val.utcoffset() == timedelta(hours=-7)


def test_legacy_full_seconds_still_parse() -> None:
    src = "t = 07:32:15\n"
    assert tomle.dumps(tomle.parse(src)) == src


def test_partial_seconds_form_rejected() -> None:
    # "07:32:" (trailing colon, missing seconds digits) is malformed.
    with pytest.raises(tomle.TOMLParseError):
        tomle.parse("t = 07:32:\n")


def test_too_short_time_rejected() -> None:
    with pytest.raises(tomle.TOMLParseError):
        tomle.parse("t = 07:3\n")
