"""Phase 3d-4 — dotted-KV insert under implicit-headerless container."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tomlrt import dumps, loads
from tomlrt._invariants import check
from tomlrt._slots import KVSlot

if TYPE_CHECKING:
    from collections.abc import Callable


def _rt(src: str, mut: Callable[[Any], None], expect: str) -> None:
    doc = loads(src)
    mut(doc)
    check(doc)
    assert dumps(doc) == expect


def test_simple_implicit_top_level() -> None:
    _rt(
        "a.x = 1\n",
        lambda d: d.table("a").__setitem__("y", 2),
        "a.x = 1\na.y = 2\n",
    )


def test_implicit_inside_section_with_later_body() -> None:
    _rt(
        "[s]\nfoo = 1\na.x = 2\nbar = 3\n",
        lambda d: d.table("s").table("a").__setitem__("y", 9),
        "[s]\nfoo = 1\na.x = 2\na.y = 9\nbar = 3\n",
    )


def test_implicit_inside_aot_entry() -> None:
    _rt(
        "[[arr]]\na.x = 1\n",
        lambda d: d.aot("arr")[0].table("a").__setitem__("y", 2),
        "[[arr]]\na.x = 1\na.y = 2\n",
    )


def test_aot_entry_slots_updated() -> None:
    doc = loads("[[arr]]\na.x = 1\nbar = 3\n")
    doc.aot("arr")[0].table("a")["y"] = 2
    check(doc)
    aot_entry = doc._refs[0].slot.entry  # noqa: SLF001
    kvs = [s for s in aot_entry.entry_slots if isinstance(s, KVSlot)]
    assert [".".join(p.value for p in s.key_parts) for s in kvs] == [
        "a.x",
        "a.y",
        "bar",
    ]


def test_insert_before_later_child_section() -> None:
    _rt(
        "a.x = 1\n\n[a.b]\ny = 2\n",
        lambda d: d.table("a").__setitem__("z", 3),
        "a.x = 1\na.z = 3\n\n[a.b]\ny = 2\n",
    )


def test_quoted_key_parts() -> None:
    _rt(
        '"a b".x = 1\n',
        lambda d: d.table("a b").__setitem__("y z", 2),
        '"a b".x = 1\n"a b"."y z" = 2\n',
    )


def test_no_final_newline_on_anchor() -> None:
    _rt(
        "a.x = 1",
        lambda d: d.table("a").__setitem__("y", 2),
        "a.x = 1\na.y = 2\n",
    )


def test_structural_only_implicit_now_synthesises_dotted_kv() -> None:
    # `a` exists only via the descendant header [a.b]; no body
    # contributors. Phase 4 now synthesises a top-level dotted KV
    # `a.x = 2` immediately before `[a.b]`.
    doc = loads("[a.b]\ny = 1\n")
    doc.table("a")["x"] = 2
    out = dumps(doc)
    assert "a.x = 2" in out
    # Round-trips correctly.
    re_parsed = loads(out)
    assert re_parsed.table("a")["x"] == 2
    assert re_parsed.table("a").table("b")["y"] == 1


def test_post_insert_delete_round_trips() -> None:
    doc = loads("a.x = 1\n")
    doc.table("a")["y"] = 2
    del doc.table("a")["y"]
    check(doc)
    assert dumps(doc) == "a.x = 1\n"


def test_multiple_aot_entries_independent() -> None:
    src = "[[arr]]\na.x = 1\n\n[[arr]]\na.x = 2\n"
    doc = loads(src)
    doc.aot("arr")[0].table("a")["y"] = 9
    check(doc)
    out = dumps(doc)
    # Second entry untouched.
    assert out == "[[arr]]\na.x = 1\na.y = 9\n\n[[arr]]\na.x = 2\n"


def test_deeply_nested_implicit() -> None:
    _rt(
        "a.b.c = 1\n",
        lambda d: d.table("a").table("b").__setitem__("d", 2),
        "a.b.c = 1\na.b.d = 2\n",
    )


def test_intermediate_implicit() -> None:
    _rt(
        "a.b.c = 1\n",
        lambda d: d.table("a").__setitem__("z", 9),
        "a.b.c = 1\na.z = 9\n",
    )
