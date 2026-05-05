"""Direct-KV insert + leaf delete in section containers.

Each test asserts byte-exact output and runs the invariant checker.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _toml_str import td
from tomlrt import dumps, loads
from tomlrt._invariants import check

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomlrt import Document


def _roundtrip(src: str, *, expect: str, mutate: Callable[[Document], None]) -> None:
    doc = loads(src)
    mutate(doc)
    check(doc)
    assert dumps(doc) == expect


def test_append_top_level_kv() -> None:
    _roundtrip(
        td("""
            a = 1
            b = 2
        """),
        expect=td("""
            a = 1
            b = 2
            c = 3
        """),
        mutate=lambda d: d.__setitem__("c", 3),
    )


def test_append_kv_in_section() -> None:
    _roundtrip(
        td("""
            [s]
            x = 1
        """),
        expect=td("""
            [s]
            x = 1
            y = 2
        """),
        mutate=lambda d: d.table("s").__setitem__("y", 2),
    )


def test_append_kv_in_header_only_section() -> None:
    _roundtrip(
        td("""
            [s]
        """),
        expect=td("""
            [s]
            x = 1
        """),
        mutate=lambda d: d.table("s").__setitem__("x", 1),
    )


def test_append_into_empty_doc() -> None:
    _roundtrip(
        "",
        expect="a = 1\n",
        mutate=lambda d: d.__setitem__("a", 1),
    )


def test_append_pins_anchor_under_parent_after_child_layout() -> None:
    # Parent header appears AFTER its child section. The new direct-KV
    # `z` for `[a]` must land after `x = 1`, not after `y = 2`.
    _roundtrip(
        td("""
            [a.b]
            y = 2

            [a]
            x = 1
        """),
        expect=td("""
            [a.b]
            y = 2

            [a]
            x = 1
            z = 3
        """),
        mutate=lambda d: d.table("a").__setitem__("z", 3),
    )


def test_append_promotes_anchor_eol_when_no_final_newline() -> None:
    # Source without a trailing newline; appending must terminate the
    # previous line before emitting ours.
    _roundtrip(
        "a = 1",
        expect="a = 1\nb = 2\n",
        mutate=lambda d: d.__setitem__("b", 2),
    )


def test_delete_top_level_scalar() -> None:
    _roundtrip(
        td("""
            a = 1
            b = 2
            c = 3
        """),
        expect=td("""
            a = 1
            c = 3
        """),
        mutate=lambda d: d.__delitem__("b"),
    )


def test_delete_takes_leading_comment_line() -> None:
    # The comment block above `b` belongs to `b`'s leading trivia and
    # must be removed with `b`.
    _roundtrip(
        td("""
            a = 1
            # comment for b
            b = 2
        """),
        expect=td("""
            a = 1
        """),
        mutate=lambda d: d.__delitem__("b"),
    )


def test_delete_then_append_at_same_position() -> None:
    _roundtrip(
        td("""
            a = 1
            b = 2
        """),
        expect=td("""
            a = 1
            c = 3
        """),
        mutate=lambda d: (d.__delitem__("b"), d.__setitem__("c", 3)),  # type: ignore[arg-type]
    )


def test_delete_only_kv_in_section_then_reinsert() -> None:
    # Tests _body_tail fallback to header_ref when the body is emptied.
    src = td("""
        [s]
        only = 1
    """)
    doc = loads(src)
    del doc.table("s")["only"]
    check(doc)
    assert dumps(doc) == "[s]\n"
    s = doc.table("s")
    assert s._header_ref is not None  # noqa: SLF001
    assert s._body_tail is s._header_ref.slot  # noqa: SLF001
    s["fresh"] = 99
    check(doc)
    assert dumps(doc) == td("""
        [s]
        fresh = 99
    """)


def test_delete_missing_key_raises_keyerror() -> None:
    doc = loads("a = 1\n")
    with pytest.raises(KeyError):
        del doc["missing"]


def test_section_only_doc_top_level_insert() -> None:
    # Inserting a top-level KV into a section-only doc inserts a
    # blank-line seam between the new KV and the first header.
    doc = loads("[s]\nx = 1\n")
    doc["new"] = 1
    assert dumps(doc) == "new = 1\n\n[s]\nx = 1\n"


def test_insert_into_implicit_table() -> None:
    # `a` exists implicitly via a dotted top-level key; dotted-KV
    # insert under an implicit container.
    doc = loads("a.x = 1\n")
    doc.table("a")["y"] = 2
    assert dumps(doc) == "a.x = 1\na.y = 2\n"


def test_insert_into_implicit_grandparent() -> None:
    doc = loads("a.b.c = 1\n")
    doc.table("a").table("b")["d"] = 2
    assert dumps(doc) == "a.b.c = 1\na.b.d = 2\n"


def test_insert_into_comment_only_doc_migrates_preamble() -> None:
    # Slotless doc with preamble trivia: inserting migrates the
    # comment block onto the new slot's leading so it stays visually
    # at the top of the file.
    doc = loads("# preamble\n")
    doc["a"] = 1
    assert dumps(doc) == "# preamble\n\na = 1\n"
    assert loads(dumps(doc)).preamble == ("preamble",)


def test_aot_entry_body_insert_now_works() -> None:
    doc = loads("[[arr]]\nx = 1\n")
    doc.aot("arr")[0]["y"] = 2
    assert dumps(doc) == "[[arr]]\nx = 1\ny = 2\n"


def test_quoted_key_for_non_bare_name() -> None:
    _roundtrip(
        "a = 1\n",
        expect='a = 1\n"weird key" = 2\n',
        mutate=lambda d: d.__setitem__("weird key", 2),
    )


def test_self_assign_no_op() -> None:
    src = "a = 1\nb = 2\n"
    doc = loads(src)
    doc["a"] = doc["a"]
    check(doc)
    assert dumps(doc) == src


def test_mixed_insert_delete_sequence() -> None:
    # Hammer the cache through several local mutations.
    doc = loads(
        td("""
            [s]
            x = 1
            y = 2
            z = 3
        """)
    )
    s = doc.table("s")
    del s["y"]
    s["w"] = 4
    del s["x"]
    s["v"] = 5
    check(doc)
    assert dumps(doc) == td("""
        [s]
        z = 3
        w = 4
        v = 5
    """)
