"""Debug-only cache-invariant checker.

Walks an attached `Document` and verifies the cache fields populated
by `_build` (`_index`, `_refs`, `_header_ref`, `_body_tail`) are
consistent with the dict storage and with each other.
``_subtree_tail`` is a derived `@property` over `_refs` and so cannot
drift; we don't check it here. Used by tests via `check(doc)` and
(when explicitly enabled) by mutation paths in debug builds.

The checker is deliberately strict: any caller-visible mutation that
leaves the caches in an inconsistent state is a bug, period.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from tomlrt._array import AoT, Array
from tomlrt._container import Container
from tomlrt._slots import KVSlot, StructuralHeaderSlot

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tomlrt._container import Document
    from tomlrt._slots import AoTEntry, Slot
    from tomlrt._values import InlineTableValue


class InvariantError(AssertionError):
    """Raised by `check` when a cache invariant is violated."""


def _fail(msg: str) -> NoReturn:
    raise InvariantError(msg)


def check(doc: Document) -> None:
    """Verify all cache invariants on an attached document."""
    _check_linked_list(doc)
    _check_container(doc)
    _check_cross_stream(doc)


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
    if c._body_tail is not None:  # noqa: SLF001
        _fail(f"inline table {c._path} has body_tail set")  # noqa: SLF001
    # When this inline table owns its `InlineTableValue` (a top-level
    # inline table — i.e. one not synthesised from a dotted inline-key
    # in a parent), verify dict storage matches the entries' first
    # key parts. Dotted entries roll into sub-containers under the
    # first key part; verify those sub-containers have the rest of
    # the entry rolled in too. This catches the bug where the inline
    # value's entries diverge from the dict view (which mutation in
    # Phase 3b will have to maintain).
    iv = c._value  # noqa: SLF001
    if iv is not None:
        _check_inline_table_dict_shape(c, iv)
    # Recurse into nested containers.
    for v in c.values():
        _walk(v)


def _check_inline_table_dict_shape(c: Container, iv: InlineTableValue) -> None:
    """Assert ``dict(c)`` and its nested inline sub-tables match ``iv.entries``.

    Builds the expected nested-dict shape from the entries' dotted keys
    (preserving first-occurrence order at every level) and walks both
    side-by-side, checking key order and identity at every level.
    """
    expected = _build_inline_expected(iv)
    _compare_inline_shape(c, expected, ())


def _build_inline_expected(
    iv: InlineTableValue,
) -> dict[str, object]:
    """Return a nested dict mapping first-occurrence-ordered keys.

    Leaves are represented by the sentinel ``None``; nested inline
    tables are nested dicts (recursively built from their dotted-key
    paths in ``iv.entries``).
    """
    out: dict[str, object] = {}
    for entry in iv.entries:
        cur: dict[str, object] = out
        parts = [p.value for p in entry.key_parts]
        for part in parts[:-1]:
            existing = cur.get(part)
            if existing is None:
                new: dict[str, object] = {}
                cur[part] = new
                cur = new
            else:
                assert isinstance(existing, dict)
                cur = existing
        leaf = parts[-1]
        if leaf not in cur:
            cur[leaf] = (
                None  # leaf marker (or replaced below if value is itself inline)
            )
        # If the leaf's value is itself an InlineTableValue, recurse to
        # capture its nested key order. We don't unify here — the
        # leaf's own _value will be checked when we recurse into the
        # logical container.
    return out


def _compare_inline_shape(
    c: Container,
    expected: dict[str, object],
    where: tuple[str, ...],
) -> None:
    if list(c.keys()) != list(expected.keys()):
        _fail(
            f"inline table at {where}: dict keys {list(c.keys())!r} differ "
            f"from InlineTableValue dotted-key shape {list(expected.keys())!r}"
        )
    for k, sub_expected in expected.items():
        if isinstance(sub_expected, dict):
            sub_c = c[k]
            if not isinstance(sub_c, Container):
                _fail(
                    f"inline table at {(*where, k)}: expected nested Container "
                    f"(dotted inline key) but got {type(sub_c).__name__}"
                )
            _compare_inline_shape(sub_c, sub_expected, (*where, k))


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
    # _body_tail must point at a slot actually in this container's _refs.
    # ``_subtree_tail`` is a derived property over ``_refs`` so it cannot
    # drift; no separate membership check needed.
    if c._refs:  # noqa: SLF001
        ref_slots = {id(r.slot) for r in c._refs}  # noqa: SLF001
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


# ---------------------------------------------------------------------------
# Cross-stream check: derive expected refs from the slot stream and compare.
# ---------------------------------------------------------------------------
#
# This is the strong check that complements the structural per-container
# checks above. The structural checks confirm "_refs is internally
# consistent with _index/_header_ref/dict storage"; this check confirms
# "_refs actually matches what the Ref-propagation rule says it should
# be, given the doc-stream of slots". A bug that over-propagates KV
# refs (filing them at ancestors above the host) trips this check even
# though the structural ones pass.


def _enumerate_containers(
    doc: Document,
) -> dict[tuple[tuple[str, ...], int | None], Container]:
    """Map ``(path, id(owner_aot_entry) or None) -> container`` for the live tree."""
    out: dict[tuple[tuple[str, ...], int | None], Container] = {}

    def visit(c: Container) -> None:
        if c._inline:  # noqa: SLF001
            # Inline-value containers do not participate in slot-stream
            # caches; they are validated separately by `_check_inline`.
            # Their `_path` may collide with section-stream containers
            # (e.g. a `dict` inside a top-level array has `_path=()`),
            # so do NOT register them in the cross-stream map.
            return
        owner = c._owner_aot_entry  # noqa: SLF001
        out[(c._path, id(owner) if owner is not None else None)] = c  # noqa: SLF001
        for v in c.values():
            if isinstance(v, Container):
                visit(v)
            elif isinstance(v, AoT):
                for entry in v:
                    visit(entry)
            elif isinstance(v, Array):
                for item in v:
                    if isinstance(item, Container):
                        visit(item)

    visit(doc)
    return out


def _resolve_active_owner(
    target_path: tuple[str, ...],
    active: dict[tuple[str, ...], AoTEntry],
) -> AoTEntry | None:
    """Pick the active AoT entry whose path is the longest prefix of ``target_path``.

    ``active`` maps an AoT path to the most recently opened entry at
    that path. The container at ``target_path`` is owned by whichever
    active entry has the deepest path that is a prefix of (or equal
    to) ``target_path``. Returns ``None`` if no active entry covers
    ``target_path``.
    """
    best: AoTEntry | None = None
    best_len = -1
    for p, entry in active.items():
        plen = len(p)
        if plen > best_len and len(target_path) >= plen and target_path[:plen] == p:
            best = entry
            best_len = plen
    return best


def _expected_refs_for_kv(
    s: KVSlot,
    active: dict[tuple[str, ...], AoTEntry],
) -> list[tuple[tuple[str, ...], int | None, str]]:
    """For a KVSlot, return list of (target_path, target_owner_id, local_key).

    Owner for each target container is resolved from the current
    active AoT-entries map (deepest matching prefix), not from the
    slot's own ``owner_aot_entry`` — they typically agree for the
    deepest target but the active map is also correct for shallower
    targets that fall outside the AoT scope.
    """
    out: list[tuple[tuple[str, ...], int | None, str]] = []
    host = s.host_path
    key = s.key
    for j in range(len(key)):
        target_path = (*host, *key[:j])
        owner = _resolve_active_owner(target_path, active)
        out.append((target_path, id(owner) if owner is not None else None, key[j]))
    return out


def _expected_refs_for_header(
    s: StructuralHeaderSlot,
    active: dict[tuple[str, ...], AoTEntry],
) -> list[tuple[tuple[str, ...], int | None, str | None]]:
    """For a StructuralHeaderSlot, return list of (target_path, owner_id, local_key).

    Binding refs are resolved against the **pre-header** active map
    (caller passes the map as it stood before this header opened any
    new entry). The own-header for an ``aot-entry`` header uses
    ``s.entry`` itself as owner; for a ``table`` header it uses
    whichever active entry covers the target path.
    """
    out: list[tuple[tuple[str, ...], int | None, str | None]] = []
    path = s.path
    for j in range(len(path)):
        target_path = path[:j]
        owner = _resolve_active_owner(target_path, active)
        out.append((target_path, id(owner) if owner is not None else None, path[j]))
    if s.kind == "aot-entry":
        own_owner: AoTEntry | None = s.entry
    else:
        own_owner = _resolve_active_owner(path, active)
    out.append((path, id(own_owner) if own_owner is not None else None, None))
    return out


def _check_cross_stream(doc: Document) -> None:
    container_map = _enumerate_containers(doc)
    expected: dict[int, list[tuple[Slot, str | None]]] = {}
    # Active AoT entries by path: updated *after* an aot-entry header
    # is processed (its own bindings still see the pre-state).
    active: dict[tuple[str, ...], AoTEntry] = {}
    # Active rendered table/AoT-entry header path: the scope a KV
    # slot will appear in when rendered. Updated whenever we cross a
    # `StructuralHeaderSlot`. Used to catch slots whose stated
    # ``host_path`` disagrees with the rendered scope they sit in
    # (which would mean parsing the rendered output gives back a
    # different document).
    active_header_path: tuple[str, ...] = ()
    cur: Slot | None = doc._head  # noqa: SLF001
    while cur is not None:
        emissions: list[tuple[tuple[str, ...], int | None, str | None]]
        if isinstance(cur, KVSlot):
            if cur.host_path != active_header_path:
                _fail(
                    f"KVSlot {cur.key} has host_path={cur.host_path} but "
                    f"renders under active header {active_header_path}"
                )
            emissions = list(_expected_refs_for_kv(cur, active))
        else:
            assert isinstance(cur, StructuralHeaderSlot)
            emissions = _expected_refs_for_header(cur, active)
            active_header_path = cur.path
        for target_path, owner_id, local_key in emissions:
            target = container_map.get((target_path, owner_id))
            if target is None:
                _fail(
                    f"slot {cur!r} expects ref at "
                    f"(path={target_path}, owner_id={owner_id}) but no such "
                    "container exists in the live tree"
                )
            expected.setdefault(id(target), []).append((cur, local_key))
        # Update active map for aot-entry headers: stale entries under
        # the new path go away (a fresh [[a]] makes any old [[a.x]]
        # entries inactive).
        if isinstance(cur, StructuralHeaderSlot) and cur.kind == "aot-entry":
            assert cur.entry is not None
            stale = [
                p
                for p in active
                if len(p) > len(cur.path) and p[: len(cur.path)] == cur.path
            ]
            for p in stale:
                del active[p]
            active[cur.path] = cur.entry
        cur = cur._next  # noqa: SLF001
    for (path, _owner_id), c in container_map.items():
        actual: list[tuple[Slot, str | None]] = [
            (r.slot, r.local_key)
            for r in c._refs  # noqa: SLF001
        ]
        exp = expected.get(id(c), [])
        if actual != exp:
            _fail(
                f"container {path}: _refs mismatch\n"
                f"  expected: {[(type(s).__name__, lk) for s, lk in exp]}\n"
                f"  actual:   {[(type(s).__name__, lk) for s, lk in actual]}"
            )
        # _index[k] must exactly equal the ordered keyed subset of _refs.
        expected_index: dict[str, list[object]] = {}
        for r in c._refs:  # noqa: SLF001
            if r.local_key is not None:
                expected_index.setdefault(r.local_key, []).append(r)
        actual_index = {k: list(v) for k, v in c._index.items()}  # noqa: SLF001
        if actual_index != expected_index:
            _fail(
                f"container {path}: _index does not equal keyed subset of _refs\n"
                f"  expected keys: {sorted(expected_index)}\n"
                f"  actual keys:   {sorted(actual_index)}"
            )
        # ``_subtree_tail`` is a derived property; no separate check.
        # Body tail: last slot in _refs satisfying body-region predicate.
        expected_body = _expected_body_tail(c)
        if c._body_tail is not expected_body:  # noqa: SLF001
            _fail(
                f"container {path}: _body_tail mismatch\n"
                f"  expected: {expected_body!r}\n"
                f"  actual:   {c._body_tail!r}"  # noqa: SLF001
            )


def _expected_body_tail(c: Container) -> Slot | None:
    """Recompute ``_body_tail`` from c._refs per the body-region rule."""
    has_header = c._header_ref is not None  # noqa: SLF001
    own_path = c._path  # noqa: SLF001
    own_owner = c._owner_aot_entry  # noqa: SLF001
    last: Slot | None = None
    for r in c._refs:  # noqa: SLF001
        slot = r.slot
        if not isinstance(slot, KVSlot):
            continue
        if slot.owner_aot_entry is not own_owner:
            continue
        if has_header and slot.host_path != own_path:
            continue
        last = slot
    if last is None and has_header:
        # Header-only container with no body slots: body_tail is the header.
        assert c._header_ref is not None  # noqa: SLF001
        return c._header_ref.slot  # noqa: SLF001
    return last


def iter_all_slots(doc: Document) -> Iterable[Slot]:
    """Iterate over every slot in the doc-stream."""
    cur: Slot | None = doc._head  # noqa: SLF001
    while cur is not None:
        yield cur
        cur = cur._next  # noqa: SLF001


__all__ = ["InvariantError", "check", "iter_all_slots"]
