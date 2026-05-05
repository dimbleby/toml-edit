"""Per-container cache invariant + scalar-replace tests.

Exercise the cache-population machinery on a representative slice of
TOML constructs:

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

from typing import TYPE_CHECKING

import pytest

from _toml_str import td
from tomlrt import dumps, loads
from tomlrt._invariants import check
from tomlrt._slots import KVSlot, StructuralHeaderSlot

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tomlrt._slots import Slot

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
    "nested_aot": td(
        """
        [[fruit]]
        name = "apple"

        [[fruit.variety]]
        name = "red"

        [[fruit.variety]]
        name = "green"

        [[fruit]]
        name = "banana"

        [[fruit.variety]]
        name = "plain"
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


# ---------------------------------------------------------------------------
# White-box anchor / cache shape assertions (per duck blocker #3).
#
# These pin the *shape* of the caches the slot-builder produces, so the
# Ref-propagation rule and body-region rule are not just internally
# consistent (caught by check()) but also match the plan exactly.
# ---------------------------------------------------------------------------


def _iter_slots(doc: object) -> Iterator[Slot]:
    """Walk the doc-stream linked list (test helper)."""
    s: Slot | None = doc._head  # type: ignore[attr-defined]  # noqa: SLF001
    while s is not None:
        yield s
        s = s._next  # noqa: SLF001


def _kv(doc: object, key: tuple[str, ...]) -> KVSlot:
    """Find a KVSlot by its decoded key tuple."""
    return next(s for s in _iter_slots(doc) if isinstance(s, KVSlot) and s.key == key)


def test_section_only_doc_has_no_top_level_body_tail() -> None:
    """`[a]\\nx=1` — `doc._body_tail` stays None; `a._body_tail` is `x`."""
    doc = loads(td("[a]\nx = 1\n"))
    a = doc.table("a")
    assert doc._body_tail is None  # noqa: SLF001
    assert a._body_tail is not None  # noqa: SLF001
    assert a._body_tail is doc._tail  # noqa: SLF001
    # doc._refs holds only `[a]`'s binding ref under "a"; it must NOT
    # have absorbed the inner `x = 1` KV (the over-propagation bug).
    assert len(doc._refs) == 1  # noqa: SLF001
    assert doc._refs[0].local_key == "a"  # noqa: SLF001
    assert "a" in doc._index  # noqa: SLF001
    assert "x" not in doc._index  # noqa: SLF001


def test_section_with_dotted_under_header_doesnt_touch_doc() -> None:
    """`[a]\\nb.c = 1` — `doc._refs` has only the [a] binding ref."""
    doc = loads(td("[a]\nb.c = 1\n"))
    assert len(doc._refs) == 1  # noqa: SLF001
    assert doc._refs[0].local_key == "a"  # noqa: SLF001
    assert doc._body_tail is None  # noqa: SLF001
    a = doc.table("a")
    # The dotted KV's host_path == ("a",) IS `a`'s path, so it
    # qualifies as a body-region slot for `a`.
    assert "b" in a._index  # noqa: SLF001
    assert a._body_tail is _kv(doc, ("b", "c"))  # noqa: SLF001


def test_parent_after_child_body_tail_is_x_not_y() -> None:
    """Pinned in plan: `[a.b]\\ny=2\\n[a]\\nx=1` — `a._body_tail` is `x`."""
    src = td(
        """
        [a.b]
        y = 2

        [a]
        x = 1
        """
    )
    doc = loads(src)
    a = doc.table("a")
    b = a.table("b")
    # a's header appears physically AFTER [a.b]'s; a's body region is
    # whatever follows its own header — `x = 1`, not `y = 2`.
    assert a._body_tail is _kv(doc, ("x",))  # noqa: SLF001
    assert b._body_tail is _kv(doc, ("y",))  # noqa: SLF001


def test_implicit_supertable_has_no_own_header_ref() -> None:
    """`[a.b.c]\\nx=1` — `a` and `a.b` are implicit; `a.b.c` is explicit."""
    doc = loads(td("[a.b.c]\nx = 1\n"))
    a = doc.table("a")
    b = a.table("b")
    c = b.table("c")
    assert a._header_ref is None  # noqa: SLF001
    assert b._header_ref is None  # noqa: SLF001
    assert c._header_ref is not None  # noqa: SLF001
    assert isinstance(c._header_ref.slot, StructuralHeaderSlot)  # noqa: SLF001
    assert c._header_ref.slot.path == ("a", "b", "c")  # noqa: SLF001
    assert a._body_tail is None  # noqa: SLF001
    assert b._body_tail is None  # noqa: SLF001
    assert c._body_tail is _kv(doc, ("x",))  # noqa: SLF001


def test_top_level_dotted_propagates_through_doc() -> None:
    """Top-level `a.b.c = 1` — refs at doc, a, a.b under each next step."""
    doc = loads(td("a.b.c = 1\n"))
    a = doc.table("a")
    b = a.table("b")
    kv = _kv(doc, ("a", "b", "c"))
    assert len(doc._refs) == 1  # noqa: SLF001
    assert doc._refs[0].local_key == "a"  # noqa: SLF001
    # The doc IS the host; doc._body_tail advances to this top-level KV.
    assert doc._body_tail is kv  # noqa: SLF001
    assert len(a._refs) == 1  # noqa: SLF001
    assert a._refs[0].local_key == "b"  # noqa: SLF001
    assert len(b._refs) == 1  # noqa: SLF001
    assert b._refs[0].local_key == "c"  # noqa: SLF001


def test_nested_aot_binding_refs_attach_to_active_outer_entry() -> None:
    """`[[fruit.variety]]` under `[[fruit]]` files its `("fruit",)` binding ref
    on the **active** ``fruit`` entry, not on ``doc``.
    """
    src = td(
        """
        [[fruit]]
        name = "apple"

        [[fruit.variety]]
        name = "red"

        [[fruit]]
        name = "banana"

        [[fruit.variety]]
        name = "plain"
        """
    )
    doc = loads(src)
    fruit_aot = doc.aot("fruit")
    apple, banana = fruit_aot[0], fruit_aot[1]
    # The first variety AoT is held by the apple entry; the second by
    # the banana entry. A bug that filed [[fruit.variety]] binding refs
    # only on doc (skipping the active fruit entry) would leave the
    # entry-level "variety" binding missing.
    assert "variety" in apple._index  # noqa: SLF001
    assert "variety" in banana._index  # noqa: SLF001
    # Each [[fruit.variety]] entry also files a "fruit" binding ref at
    # doc per the prefix-container rule, on top of the [[fruit]] ones.
    # Total under doc["fruit"]: 2 [[fruit]] + 2 [[fruit.variety]] = 4.
    doc_refs_for_fruit = [r for r in doc._refs if r.local_key == "fruit"]  # noqa: SLF001
    assert len(doc_refs_for_fruit) == 4
    # The apple entry's own variety AoT should bind a `red` first
    # entry; banana's should bind a `plain` first entry. A bug that
    # routed [[fruit.variety]] to the wrong entry (or to doc only)
    # would scramble these assignments.
    assert apple.aot("variety")[0]["name"] == "red"
    assert banana.aot("variety")[0]["name"] == "plain"
