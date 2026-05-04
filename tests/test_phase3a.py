"""Phase 3a invariant + scalar-replace tests.

These tests exercise the cache-population machinery added in Phase 3a
on a representative slice of TOML constructs (per duck #12 of the
Phase 3 plan review):

- top-level dotted KV
- preamble + later sections
- implicit super-table opened by `[a.b]`
- AoT with nested dotted KV inside an entry
- inline table with dotted inline keys
- section-only document where top-level `_body_tail` is `None`
- comments-heavy document

Plus a minimal scalar-replace acceptance.
"""

from __future__ import annotations

import pytest

from _toml_str import td
from tomlrt import dumps, loads
from tomlrt._invariants import check

INVARIANT_DOCS = {
    "top_level_dotted": td(
        """
        a.b = 1
        a.c = 2
        """
    ),
    "preamble_then_section": td(
        """
        # preamble comment
        title = "x"
        version = 3

        [pkg]
        name = "y"
        """
    ),
    "implicit_supertable": td(
        """
        [a.b.c]
        x = 1
        """
    ),
    "aot_with_dotted_inside": td(
        """
        [[pkg]]
        name = "a"
        meta.version = "1"

        [[pkg]]
        name = "b"
        meta.version = "2"
        """
    ),
    "inline_table_dotted": td(
        """
        x = { a.b = 1, a.c = 2 }
        """
    ),
    "section_only": td(
        """
        [a]
        x = 1
        """
    ),
    "comments_heavy": td(
        """
        # top
        # block

        a = 1  # eol
        # mid
        b = 2

        # before section
        [s]
        # in section
        c = 3
        """
    ),
    "parent_after_child": td(
        """
        [a.b]
        y = 2

        [a]
        x = 1
        """
    ),
    "interleaved_aot": td(
        """
        [[a]]
        x = 1

        [[b]]
        z = 1

        [[a]]
        x = 2
        """
    ),
}


@pytest.mark.parametrize("name", sorted(INVARIANT_DOCS))
def test_invariants_after_parse(name: str) -> None:
    doc = loads(INVARIANT_DOCS[name])
    check(doc)


def test_invariants_round_trip_preserved() -> None:
    """All the invariant fixtures still round-trip byte-exact."""

    for src in INVARIANT_DOCS.values():
        assert dumps(loads(src)) == src


def test_replace_scalar_in_place() -> None:
    src = td(
        """
        a = 1
        b = "hello"
        """
    )
    doc = loads(src)
    doc["a"] = 99
    doc["b"] = "world"
    check(doc)
    out = "".join(line for line in [str(doc.render())])
    assert "a = 99" in out
    assert 'b = "world"' in out


def test_replace_scalar_preserves_surrounding_format() -> None:
    src = td(
        """
        # top
        a = 1   # trailing
        # bottom
        """
    )
    doc = loads(src)
    doc["a"] = 42
    expected = src.replace("a = 1", "a = 42")

    assert dumps(doc) == expected
    check(doc)


def test_replace_scalar_inside_section() -> None:
    src = td(
        """
        [pkg]
        name = "old"
        version = 1
        """
    )
    doc = loads(src)
    doc.table("pkg")["name"] = "new"
    check(doc)

    assert dumps(doc) == src.replace("old", "new")


def test_setitem_self_assign_is_noop() -> None:
    src = "a = 1\n"
    doc = loads(src)
    val = doc["a"]
    doc["a"] = val

    assert dumps(doc) == src
    check(doc)
