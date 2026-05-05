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


def test_delete_only_subsection_keeps_implicit_parent() -> None:
    # `a` is implicit (no [a] header). Deleting its only [a.b]
    # leaves `a` reachable as an empty implicit table — Python-dict
    # semantics: `del` removes only the named key. The implicit
    # parent has no rendering presence (no header, no refs) so
    # byte-exact dumps stays empty.
    src = td("""
        [a.b]
        x = 1
    """)
    doc = loads(src)
    del doc.table("a")["b"]
    check(doc)
    assert dumps(doc) == ""
    assert "a" in doc
    assert dict(doc.table("a")) == {}


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
    # Phase 3e gives held views a private orphan root + live mutation.
    # Mutating the orphan must not affect the live document.
    doc = loads("[a]\nx = 1\n[b]\ny = 2\n")
    held = doc.table("a")
    del doc["a"]
    check(doc)
    assert "a" not in doc
    assert dumps(doc) == "[b]\ny = 2\n"
    # Held-view mutation lands in the orphan, not the live doc.
    held["new"] = 99
    check(doc)
    assert dumps(doc) == "[b]\ny = 2\n"
    assert held["new"] == 99


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


def test_delete_deep_implicit_inside_aot_keeps_implicit_chain() -> None:
    # Per Python-dict semantics: `del` removes exactly the named
    # key. Implicit ancestors (`foo`, `bar`) inside an AoT entry
    # stay as empty `Table` views — they have no rendering presence
    # (no header, no refs), so the AoT entry serialises as just its
    # own header.
    doc = loads("[[arr]]\nfoo.bar.baz = 1\n")
    del doc.aot("arr")[0].table("foo")["bar"]
    check(doc)
    assert dumps(doc) == "[[arr]]\n"
    # `foo` survives in the entry's dict storage as an empty
    # implicit table; `to_dict()` reflects that.
    assert doc.to_dict() == {"arr": [{"foo": {}}]}


def test_readd_into_emptied_aot_implicit_anchors_inside_entry() -> None:
    # Regression: after deleting the only descendant of an
    # AoT-owned implicit chain, re-adding under that chain must
    # synthesise the [arr.foo] header INSIDE the owning entry's
    # slot region — not at doc tail. Otherwise a re-parse would
    # attribute the resurrected sub-section to the wrong (later)
    # AoT entry.
    doc = loads("[[arr]]\nfoo.bar.baz = 1\n\n[[arr]]\nname = 2\n")
    foo = doc.aot("arr")[0].table("foo")
    del foo["bar"]
    foo["new"] = 1
    check(doc)
    out = dumps(doc)
    # Synthetic [arr.foo] sits between the two [[arr]] entries,
    # belonging to the first.
    assert "[arr.foo]\nnew = 1" in out
    # Round-trip through the parser preserves the logical tree.
    assert loads(out).to_dict() == doc.to_dict() == {
        "arr": [{"foo": {"new": 1}}, {"name": 2}],
    }


def test_delete_header_only_section() -> None:
    # `[s]` with no body — header-only.
    _rt(
        "[s]\n",
        lambda d: d.__delitem__("s"),
        expect="",
    )


def test_delete_inline_array_value() -> None:
    _rt(
        "arr = [1, 2, 3]\ny = 4\n",
        lambda d: d.__delitem__("arr"),
        expect="y = 4\n",
    )


def test_delete_inline_array_with_inline_table_inside() -> None:
    _rt(
        "arr = [1, {x = 2}]\ny = 3\n",
        lambda d: d.__delitem__("arr"),
        expect="y = 3\n",
    )


def test_delete_deep_non_aot_implicit_keeps_chain() -> None:
    # `[a.b.c.d]\nx=1` → a, b, c are all implicit. Deleting `d`
    # under Python-dict semantics removes only `d`; the implicit
    # chain `a.b.c` survives as nested empty `Table` views. None of
    # the implicit ancestors render (no header, no refs), so dumps
    # is byte-empty.
    doc = loads("[a.b.c.d]\nx = 1\n")
    del doc.table("a").table("b").table("c")["d"]
    check(doc)
    assert dumps(doc) == ""
    assert "a" in doc
    assert dict(doc.table("a").table("b").table("c")) == {}


def test_held_deleted_section_view_has_clean_orphan_state() -> None:
    # Defence-in-depth: held view is internally consistent after
    # delete (Phase 3e detaches it into a private orphan root, so
    # the held view's refs/header/_body_tail still resolve — but
    # against the orphan, not the live doc).
    doc = loads("[a]\nx = 1\n[b]\ny = 2\n")
    held = doc.table("a")
    del doc["a"]
    # Held view's layout root is a private orphan, not `doc`.
    assert held._layout_root is not doc  # noqa: SLF001
    assert held._layout_root._is_private  # noqa: SLF001
    # Slot infra is intact (refs point at orphan slots).
    assert held["x"] == 1
