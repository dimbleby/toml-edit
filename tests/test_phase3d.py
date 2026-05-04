"""Phase 3d-1 — structural delete (section / AoT / dotted-subtree)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from typing import TYPE_CHECKING

from _toml_str import td
from tomlrt import dumps, loads
from tomlrt._invariants import check

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomlrt import Document


def _rt(src: str, mutate: Callable[[Document], object], *, expect: str) -> None:
    doc = loads(src)
    mutate(doc)
    check(doc)
    assert dumps(doc) == expect


def test_delete_simple_section() -> None:
    _rt(
        td("""
            [a]
            x = 1
            [b]
            y = 2
        """),
        lambda d: d.__delitem__("a"),
        expect=td("""
            [b]
            y = 2
        """),
    )


def test_delete_section_with_subsections() -> None:
    _rt(
        td("""
            [a]
            x = 1
            [a.b]
            y = 2
            [a.b.c]
            z = 3
            [other]
            w = 4
        """),
        lambda d: d.__delitem__("a"),
        expect=td("""
            [other]
            w = 4
        """),
    )


def test_delete_aot() -> None:
    _rt(
        td("""
            [[arr]]
            x = 1
            [[arr]]
            y = 2
            [other]
            z = 3
        """),
        lambda d: d.__delitem__("arr"),
        expect=td("""
            [other]
            z = 3
        """),
    )


def test_delete_aot_with_owned_subsections() -> None:
    _rt(
        td("""
            [[arr]]
            x = 1
            [arr.sub]
            inner = 1
            [[arr]]
            y = 2
            [other]
            z = 3
        """),
        lambda d: d.__delitem__("arr"),
        expect=td("""
            [other]
            z = 3
        """),
    )


def test_delete_top_level_dotted_in_preamble() -> None:
    _rt(
        "a.b = 1\nc = 2\n",
        lambda d: d.__delitem__("a"),
        expect="c = 2\n",
    )


def test_delete_top_level_dotted_with_multiple_a_keys() -> None:
    # Both `a.b = 1` and `a.c = 2` go away when deleting "a".
    _rt(
        "a.b = 1\na.c = 2\nz = 3\n",
        lambda d: d.__delitem__("a"),
        expect="z = 3\n",
    )


def test_delete_dotted_inside_aot_entry() -> None:
    _rt(
        "[[items]]\nfoo.bar = 1\nkeep = 2\n",
        lambda d: d.aot("items")[0].__delitem__("foo"),
        expect="[[items]]\nkeep = 2\n",
    )


def test_delete_subsection_via_parent_view() -> None:
    _rt(
        td("""
            [a]
            x = 1
            [a.b]
            y = 2
        """),
        lambda d: d.table("a").__delitem__("b"),
        expect=td("""
            [a]
            x = 1
        """),
    )


def test_delete_only_subsection_drops_implicit_parent() -> None:
    # `a` is implicit (no [a] header). Deleting its only [a.b]
    # leaves nothing referencing `a`; pruning drops `a` from the
    # doc dict to keep cache invariants clean.
    src = td("""
        [a.b]
        x = 1
    """)
    doc = loads(src)
    del doc.table("a")["b"]
    check(doc)
    assert dumps(doc) == ""
    assert "a" not in doc


def test_delete_one_of_two_implicit_subsections_keeps_parent() -> None:
    # `a` is still bound by [a.c] after deleting [a.b].
    _rt(
        td("""
            [a.b]
            x = 1
            [a.c]
            y = 2
        """),
        lambda d: d.table("a").__delitem__("b"),
        expect=td("""
            [a.c]
            y = 2
        """),
    )


def test_delete_inline_table_value() -> None:
    _rt(
        "obj = { a = 1, b = 2 }\nx = 1\n",
        lambda d: d.__delitem__("obj"),
        expect="x = 1\n",
    )


def test_delete_section_with_leading_comment_block() -> None:
    # Section's leading trivia (comment block + blank line) goes
    # with the section. The trailing blank above [a] was attached
    # to [a]'s leading trivia, so it disappears too.
    _rt(
        td("""
            [keep]
            v = 0

            # comment for a
            [a]
            x = 1
        """),
        lambda d: d.__delitem__("a"),
        expect=td("""
            [keep]
            v = 0
        """),
    )


def test_delete_missing_key_raises() -> None:
    doc = loads("a = 1\n")
    with pytest.raises(KeyError):
        del doc["nope"]


def test_delete_then_reinsert_at_top_level_after_section_delete() -> None:
    # After dropping section [a], [b] survives and we can append a
    # new top-level scalar via the doc-root body_tail (previously
    # body_tail pointed inside [a]'s region — recompute must catch
    # it correctly).
    src = td("""
        x = 1
        [a]
        inner = 2
    """)
    doc = loads(src)
    del doc["a"]
    check(doc)
    # x is still the body_tail.
    doc["y"] = 3
    check(doc)
    assert dumps(doc) == "x = 1\ny = 3\n"


def test_held_view_after_delete_does_not_corrupt_doc() -> None:
    # Phase 3e will give held views a private root + live mutation.
    # In 3d-1 we just need to guarantee the doc itself stays
    # consistent and the held view's structural mutation either
    # raises NIE or no-ops — never a partial corruption.
    doc = loads("[a]\nx = 1\n[b]\ny = 2\n")
    held = doc.table("a")
    del doc["a"]
    check(doc)
    assert "a" not in doc
    assert dumps(doc) == "[b]\ny = 2\n"
    # Mutating the orphan should not affect the live document.
    with pytest.raises(NotImplementedError):
        held["new"] = 99
    check(doc)
    assert dumps(doc) == "[b]\ny = 2\n"


def test_delete_aot_entry_internal_kv_keeps_other_entries() -> None:
    _rt(
        td("""
            [[arr]]
            x = 1
            y = 2
            [[arr]]
            z = 3
        """),
        lambda d: d.aot("arr")[0].__delitem__("y"),
        expect=td("""
            [[arr]]
            x = 1
            [[arr]]
            z = 3
        """),
    )


def test_section_delete_then_dump_reparses_to_expected() -> None:
    src = td("""
        [a]
        x = 1
        [b]
        y = 2
    """)
    doc = loads(src)
    del doc["a"]
    out = dumps(doc)
    assert loads(out).to_dict() == {"b": {"y": 2}}
