"""Debug-only cache-invariant checker.

Walks an attached `Document` and verifies the cache fields populated
by `_build` (`_index`, `_refs`, `_header_ref`, `_body_tail`,
`_subtree_tail`) are consistent with the dict storage and with each
other. Used by tests via `check(doc)` and (when explicitly enabled)
by mutation paths in debug builds.

The checker is deliberately strict: any caller-visible mutation that
leaves the caches in an inconsistent state is a bug, period.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from tomlrt._array import AoT, Array
from tomlrt._container import Container
from tomlrt._slots import StructuralHeaderSlot

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tomlrt._container import Document
    from tomlrt._slots import Slot


class InvariantError(AssertionError):
    """Raised by `check` when a cache invariant is violated."""


def _fail(msg: str) -> NoReturn:
    raise InvariantError(msg)


def check(doc: Document) -> None:
    """Verify all cache invariants on an attached document."""
    _check_linked_list(doc)
    _check_container(doc)


# ---------------------------------------------------------------------------
# Linked-list integrity
# ---------------------------------------------------------------------------


def _check_linked_list(doc: Document) -> None:
    head = doc._head  # noqa: SLF001
    tail = doc._tail  # noqa: SLF001
    if head is None:
        if tail is not None:
            _fail("head is None but tail is not")
        return
    if head._prev is not None:  # noqa: SLF001
        _fail("doc head has non-None _prev")
    seen: set[int] = set()
    cur: Slot | None = head
    last: Slot | None = None
    while cur is not None:
        if id(cur) in seen:
            _fail("linked-list cycle detected")
        seen.add(id(cur))
        if cur._prev is not last:  # noqa: SLF001
            _fail(f"slot._prev mismatch at {cur!r}: expected {last!r}")
        last = cur
        cur = cur._next  # noqa: SLF001
    if last is not tail:
        _fail("doc tail does not match linked-list end")


# ---------------------------------------------------------------------------
# Container caches
# ---------------------------------------------------------------------------


def _check_container(c: Container) -> None:
    if c._inline:  # noqa: SLF001
        _check_inline(c)
        return
    _check_section_caches(c)
    # Recurse into nested containers.
    for v in c.values():
        _walk(v)


def _check_inline(c: Container) -> None:
    # Inline tables don't use slot-stream caches.
    if c._index:  # noqa: SLF001
        _fail(f"inline table {c._path} has non-empty _index")  # noqa: SLF001
    if c._refs:  # noqa: SLF001
        _fail(f"inline table {c._path} has non-empty _refs")  # noqa: SLF001
    if c._header_ref is not None:  # noqa: SLF001
        _fail(f"inline table {c._path} has _header_ref set")  # noqa: SLF001
    if c._body_tail is not None or c._subtree_tail is not None:  # noqa: SLF001
        _fail(
            f"inline table {c._path} has body_tail/subtree_tail set"  # noqa: SLF001
        )
    # Inline tables built from a top-level inline-table value carry
    # `_value`. Inline sub-tables synthesised from a dotted inline-key
    # don't (their backing entries live in the parent's
    # InlineTableValue). So we don't assert _value is set in either
    # direction; just recurse.
    for v in c.values():
        _walk(v)


def _check_section_caches(c: Container) -> None:
    # Every dict key must have an _index entry, except empty AoTs which
    # have a logical binding but zero physical slots.
    for k, v in c.items():
        if isinstance(v, AoT) and len(v) == 0:
            if k in c._index:  # noqa: SLF001
                _fail(
                    f"empty AoT {c._path}.{k} has unexpected _index entry"  # noqa: SLF001
                )
            continue
        if k not in c._index:  # noqa: SLF001
            _fail(
                f"key {k!r} in dict but missing from _index (container {c._path})"  # noqa: SLF001
            )
    # Every _index entry must correspond to a dict key.
    for k in c._index:  # noqa: SLF001
        if k not in c:
            _fail(
                f"_index has {k!r} but dict does not (container {c._path})"  # noqa: SLF001
            )
    # Every ref in _index[k] must have local_key == k and live in _refs.
    refs_seen: set[int] = set()
    for k, refs in c._index.items():  # noqa: SLF001
        if not refs:
            _fail(f"empty _index[{k!r}] in {c._path}")  # noqa: SLF001
        for r in refs:
            if r.local_key != k:
                _fail(f"index ref for {k!r} has local_key={r.local_key!r}")
            if r.container is not c:
                _fail(f"index ref for {k!r} has wrong container back-pointer")
            refs_seen.add(id(r))
    for r in c._refs:  # noqa: SLF001
        if r.local_key is None:
            if r is not c._header_ref:  # noqa: SLF001
                _fail(
                    f"unkeyed ref in _refs is not _header_ref ({c._path})"  # noqa: SLF001
                )
            continue
        if id(r) not in refs_seen:
            _fail(
                f"keyed ref local_key={r.local_key!r} is in _refs but not in "
                f"_index ({c._path})"  # noqa: SLF001
            )
    # _header_ref invariants.
    hr = c._header_ref  # noqa: SLF001
    if hr is not None:
        if hr.local_key is not None:
            _fail("_header_ref has non-None local_key")
        if hr.container is not c:
            _fail("_header_ref points at wrong container")
        if not isinstance(hr.slot, StructuralHeaderSlot):
            _fail("_header_ref does not point at a header")
        if hr.slot.path != c._path:  # noqa: SLF001
            _fail(
                f"_header_ref.slot.path={hr.slot.path} != {c._path}"  # noqa: SLF001
            )
        if hr not in c._refs:  # noqa: SLF001
            _fail("_header_ref not present in _refs")
    # _body_tail / _subtree_tail must point at slots actually in this
    # container's _refs (i.e., _refs contains a ref to that slot).
    if c._refs:  # noqa: SLF001
        ref_slots = {id(r.slot) for r in c._refs}  # noqa: SLF001
        if c._subtree_tail is not None and id(c._subtree_tail) not in ref_slots:  # noqa: SLF001
            _fail(
                f"_subtree_tail not in _refs of container {c._path}"  # noqa: SLF001
            )
        if c._body_tail is not None and id(c._body_tail) not in ref_slots:  # noqa: SLF001
            _fail(
                f"_body_tail not in _refs of container {c._path}"  # noqa: SLF001
            )


def _walk(v: object) -> None:
    if isinstance(v, Container):
        _check_container(v)
    elif isinstance(v, AoT):
        for entry in v:
            _check_container(entry)
    elif isinstance(v, Array):
        for item in v:
            _walk(item)


def iter_all_slots(doc: Document) -> Iterable[Slot]:
    """Iterate over every slot in the doc-stream."""
    cur: Slot | None = doc._head  # noqa: SLF001
    while cur is not None:
        yield cur
        cur = cur._next  # noqa: SLF001


__all__ = ["InvariantError", "check", "iter_all_slots"]
