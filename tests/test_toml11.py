"""TOML 1.1.0 spec additions.

Each test pins one new behaviour from the v1.1.0 spec
(<https://toml.io/en/v1.1.0>). Round-trip preservation is asserted
explicitly because that is the whole point of this library.
"""

from __future__ import annotations

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
