"""Section-side mutation primitives (Phase 3c onward).

Linked-list and per-container cache updates for direct (non-dotted)
KV insert and leaf delete. Inline-table mutation lives in
``_inline_ops.py``; this module never touches inline-tables.

Design notes:

* The doc-stream linked list is the single source of physical
  ordering. Insert primitives splice exactly one slot at a time at
  an explicitly-named anchor — no list-index search, no
  doc-stream-wide rescans.

* ``c._refs`` mirrors the doc-stream subset of slots referenced by
  ``c``. For direct (non-dotted) KV inserts, the new ref's correct
  position in ``c._refs`` is **immediately after the anchor's ref**
  (or at the front if there is no anchor ref). This preserves the
  invariant that ``c._subtree_tail`` (a property = ``c._refs[-1]``)
  matches doc-stream order, and avoids drift in layouts where ``c``
  has later child-section refs sitting after its body region.

* ``c._body_tail`` is maintained incrementally: O(1) on insert, and
  O(len(c._refs)) only on the rare delete-of-current-tail.

* No ancestor walk: a non-dotted direct KV files exactly one ref,
  on its host container. Ancestors are unaffected.
"""

from __future__ import annotations

import contextlib
import re
from typing import TYPE_CHECKING

from tomlrt._slots import KVSlot, SlotRef, StructuralHeaderSlot
from tomlrt._trivia import EolTrivia, NewlineNode, Trivia
from tomlrt._values import KeyPart

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomlrt._container import Container, Document
    from tomlrt._slots import Slot
    from tomlrt._values import Value


_RE_BARE_KEY = re.compile(r"\A[A-Za-z0-9_\-]+\Z")


# ---------------------------------------------------------------------------
# Pure linked-list ops
# ---------------------------------------------------------------------------


def insert_after(anchor: Slot, new_slot: Slot, doc: Document) -> None:
    """Splice ``new_slot`` immediately after ``anchor`` in ``doc``."""
    nxt = anchor._next  # noqa: SLF001
    new_slot._prev = anchor  # noqa: SLF001
    new_slot._next = nxt  # noqa: SLF001
    anchor._next = new_slot  # noqa: SLF001
    if nxt is not None:
        nxt._prev = new_slot  # noqa: SLF001
    else:
        doc._tail = new_slot  # noqa: SLF001


def insert_before_head(new_slot: Slot, doc: Document) -> None:
    """Splice ``new_slot`` at the start of ``doc``'s linked list."""
    head = doc._head  # noqa: SLF001
    new_slot._prev = None  # noqa: SLF001
    new_slot._next = head  # noqa: SLF001
    if head is not None:
        head._prev = new_slot  # noqa: SLF001
    else:
        doc._tail = new_slot  # noqa: SLF001
    doc._head = new_slot  # noqa: SLF001


def unlink_slot(slot: Slot, doc: Document) -> None:
    """Remove ``slot`` from ``doc``'s linked list (does not touch caches)."""
    p = slot._prev  # noqa: SLF001
    n = slot._next  # noqa: SLF001
    if p is not None:
        p._next = n  # noqa: SLF001
    else:
        doc._head = n  # noqa: SLF001
    if n is not None:
        n._prev = p  # noqa: SLF001
    else:
        doc._tail = p  # noqa: SLF001
    slot._prev = None  # noqa: SLF001
    slot._next = None  # noqa: SLF001


# ---------------------------------------------------------------------------
# Higher-level ops
# ---------------------------------------------------------------------------


def append_direct_kv(c: Container, key: str, value: Value) -> None:
    """Append a fresh direct (non-dotted) KV to ``c``.

    Updates ``c._refs`` / ``_index`` / ``_body_tail`` and the dict
    storage. Phase 3c only — defers AoT-entry body inserts, detached
    containers, and the empty-doc-with-only-sections case.
    """
    if c._owner_aot_entry is not None:  # noqa: SLF001
        msg = "insert into AoT entry body is not yet supported"
        raise NotImplementedError(msg)
    if c._path and c._header_ref is None:  # noqa: SLF001
        # Implicit / headerless non-root container: a fresh
        # `host_path = c._path` slot would render in whatever scope
        # the previous header (or the doc root) established, not in
        # `c`'s logical scope. Correct insertion needs either dotted
        # rendering under the host or synthetic-header creation —
        # both Phase 3d.
        msg = (
            "insert into an implicit / headerless table requires dotted or "
            "synthetic-header rendering and is deferred to Phase 3d"
        )
        raise NotImplementedError(msg)
    layout_root = c._layout_root  # noqa: SLF001
    if layout_root is None:
        msg = "internal: container has no layout root"
        raise AssertionError(msg)
    doc = layout_root  # PrivateRoot arrives in Phase 3e

    # Capture the anchor *before* mutating any cache.
    body_tail = c._body_tail  # noqa: SLF001
    header_ref = c._header_ref  # noqa: SLF001

    new_slot = _build_kv_slot(c, key, value, doc)

    if body_tail is not None:
        insert_after(body_tail, new_slot, doc)
    elif header_ref is not None:
        # Header-only container: anchor at the header itself.
        insert_after(header_ref.slot, new_slot, doc)
    elif doc._head is not None:  # noqa: SLF001
        # Doc root with no top-level KV but the doc has section
        # headers — Phase 3d will own the seam-with-blank-line case.
        msg = (
            "inserting a top-level KV into a section-only document is not "
            "yet supported (Phase 3d structural seam)"
        )
        raise NotImplementedError(msg)
    elif doc._trailing.pieces:  # noqa: SLF001
        # Slotless doc with preamble-only trivia (e.g. comment-only
        # source). Inserting would either silently relocate that
        # trivia to the epilogue or require a preamble-migration
        # policy — Phase 3d / Phase 4 territory.
        msg = (
            "inserting into a slotless doc that already has trivia "
            "(comment-only source) requires preamble migration; "
            "deferred to a later phase"
        )
        raise NotImplementedError(msg)
    else:
        # Genuinely empty doc.
        insert_before_head(new_slot, doc)

    new_ref = SlotRef(slot=new_slot, container=c, local_key=key)
    # The new ref's correct position in ``c._refs`` is immediately
    # after the anchor ref (the previous body_tail's ref). For an
    # implicit / header-only container with no body refs yet, the
    # new ref goes at the end (after the header ref, if any).
    if body_tail is None:
        c._refs.append(new_ref)  # noqa: SLF001
    else:
        anchor_idx = _find_ref_index_by_slot(c, body_tail)
        c._refs.insert(anchor_idx + 1, new_ref)  # noqa: SLF001
    c._index.setdefault(key, []).append(new_ref)  # noqa: SLF001
    c._body_tail = new_slot  # noqa: SLF001


def delete_key(c: Container, key: str) -> None:
    """Delete ``key`` from ``c`` — scalar, inline, section, AoT, or dotted-subtree.

    Single primitive that handles every flavour. Steps (mirroring the
    plan v17 "Slot deletion ordering" recipe):

    1. Compute the owned-slot identity set: every physical slot whose
       refs all live in ``c._index[key]`` plus every descendant
       container's ``_refs``.
    2. Compute the containers-to-scrub set: ``c``'s ancestor chain
       (up to and including the doc root) plus every non-inline
       container in the subtree rooted at ``c[key]``.
    3. Scrub: rebuild ``_refs``/``_index`` of each scrubbed container,
       dropping refs whose slot is in the owned set. Recompute
       ``_body_tail`` on each container whose old tail was unlinked.
       Drop any unlinked slot from its owning ``AoTEntry.entry_slots``.
    4. Debug-only: assert no still-live container retains a ref to any
       owned slot.
    5. Unlink each owned slot from the doc linked list.
    6. Drop the dict entry on ``c``.
    7. Recursively prune empty implicit super-tables on the ancestor
       chain (those with no header, no refs, and an empty dict body)
       so the cache invariants stay clean.

    No live-detach: held views of the deleted subtree retain stale
    ``_layout_root`` / ``_path``; mutating them through their old
    reference is Phase 3e (PrivateRoot). With Phase 3c's
    implicit-headerless guard, structural mutation through such a
    held view raises ``NotImplementedError`` rather than corrupting
    the live document, which is the safest interim behaviour.
    """
    if key not in c:
        raise KeyError(key)
    val = dict.__getitem__(c, key)
    layout_root = c._layout_root  # noqa: SLF001
    if layout_root is None:
        msg = "internal: container has no layout root"
        raise AssertionError(msg)
    doc = layout_root  # PrivateRoot arrives in Phase 3e

    # 1. Owned-slot identity set + retained slot objects (for unlink).
    owned_ids: set[int] = set()
    owned_slots: list[Slot] = []
    seen_ref_ids: set[int] = set()

    def _add_ref(r: SlotRef) -> None:
        if id(r) in seen_ref_ids:
            return
        seen_ref_ids.add(id(r))
        if id(r.slot) not in owned_ids:
            owned_ids.add(id(r.slot))
            owned_slots.append(r.slot)

    for r in c._index.get(key, []):  # noqa: SLF001
        _add_ref(r)

    # 2. Subtree containers + descendant refs.
    subtree_containers: list[Container] = []
    _collect_subtree(val, subtree_containers, _add_ref)

    # 3. Scrub ancestor chain + subtree containers.
    chain: list[Container] = []
    cur: Container | None = c
    while cur is not None:
        chain.append(cur)
        cur = cur._parent  # noqa: SLF001
    scrub: list[Container] = [*chain, *subtree_containers]

    for cc in scrub:
        old_refs = cc._refs  # noqa: SLF001
        kept: list[SlotRef] = [r for r in old_refs if id(r.slot) not in owned_ids]
        if len(kept) != len(old_refs):
            cc._refs = kept  # noqa: SLF001
            cc._index = {}  # noqa: SLF001
            for r in kept:
                if r.local_key is not None:
                    cc._index.setdefault(r.local_key, []).append(r)  # noqa: SLF001
            # Clear an owned `_header_ref` BEFORE recomputing `_body_tail`,
            # so the recompute does not fall back to a header slot that is
            # itself about to be unlinked. (Live containers never have an
            # owned header — only discarded subtree containers do — but
            # leaving the orphan internally consistent is friendlier to
            # held views and Phase 3e detach work.)
            if (
                cc._header_ref is not None  # noqa: SLF001
                and id(cc._header_ref.slot) in owned_ids  # noqa: SLF001
            ):
                cc._header_ref = None  # noqa: SLF001
            if (
                cc._body_tail is not None  # noqa: SLF001
                and id(cc._body_tail) in owned_ids  # noqa: SLF001
            ):
                cc._body_tail = _recompute_body_tail(cc)  # noqa: SLF001

    # 4. Defensive: no live container retains a ref to any owned slot.
    if __debug__:
        for cc in chain:
            for r in cc._refs:  # noqa: SLF001
                assert id(r.slot) not in owned_ids, (
                    "internal: live container still references a slot we are about "
                    "to unlink — owned-set is incomplete"
                )

    # 5. Unlink owned slots; clean up AoTEntry.entry_slots for live entries.
    surviving_aot_entries = _surviving_aot_entries(doc)
    for slot in owned_slots:
        owner = getattr(slot, "owner_aot_entry", None)
        if owner is not None and id(owner) in surviving_aot_entries:
            with contextlib.suppress(ValueError):
                owner.entry_slots.remove(slot)
        unlink_slot(slot, doc)

    # 6. Drop the dict entry.
    dict.__delitem__(c, key)

    # 7. Prune empty implicit super-tables walking up the chain.
    _prune_empty_implicit_ancestors(c)


def _collect_subtree(
    val: object,
    containers_out: list[Container],
    add_ref: Callable[[SlotRef], None],
) -> None:
    """Walk ``val``'s container subtree, collecting non-inline containers and refs."""
    from tomlrt._array import AoT  # noqa: PLC0415
    from tomlrt._container import Container  # noqa: PLC0415

    if isinstance(val, Container):
        if val._inline:  # noqa: SLF001
            return
        containers_out.append(val)
        for r in val._refs:  # noqa: SLF001
            add_ref(r)
        for child in val.values():
            _collect_subtree(child, containers_out, add_ref)
    elif isinstance(val, AoT):
        for entry in val:
            _collect_subtree(entry, containers_out, add_ref)


def _surviving_aot_entries(doc: Document) -> set[int]:
    """Set of ``id(AoTEntry)`` for entries still reachable from doc dict tree."""
    from tomlrt._array import AoT  # noqa: PLC0415
    from tomlrt._container import Container  # noqa: PLC0415

    surviving: set[int] = set()

    def visit(v: object) -> None:
        if isinstance(v, Container):
            owner = v._owner_aot_entry  # noqa: SLF001
            if owner is not None:
                surviving.add(id(owner))
            if not v._inline:  # noqa: SLF001
                for child in v.values():
                    visit(child)
        elif isinstance(v, AoT):
            for entry in v:
                visit(entry)

    visit(doc)
    return surviving


def _prune_empty_implicit_ancestors(c: Container) -> None:
    """Drop implicit-empty containers from their parent's dict storage.

    A container is implicit-empty iff: not the doc root, no
    ``_header_ref``, empty ``_refs``, empty dict, not inline.

    Such a container has no rendering presence and no slot ownership;
    leaving it in the parent's dict would violate the
    "every dict key has an `_index` entry" invariant.

    The walk does NOT need a special stop for AoT-entry root tables
    or implicit descendants beneath them: AoT-entry root tables are
    protected by their own ``_header_ref``; implicit descendants
    inside an AoT entry that become empty must be pruned just as at
    the doc root, otherwise stale ``foo = {}`` containers would
    linger inside surviving AoT entries.
    """
    cur: Container | None = c
    while cur is not None:
        parent = cur._parent  # noqa: SLF001
        if parent is None:
            return
        if (
            cur._header_ref is not None  # noqa: SLF001
            or cur._refs  # noqa: SLF001
            or len(cur) > 0
            or cur._inline  # noqa: SLF001
        ):
            return
        # Find the local key of cur in parent. If parent stores an
        # `AoT` under cur's path (cur is an AoT entry root rather
        # than a dict-keyed sub-container), the identity check below
        # protects us — entries are never stored directly in the
        # parent's dict.
        local_key = cur._path[-1] if cur._path else None  # noqa: SLF001
        if local_key is None or local_key not in parent:
            return
        if dict.__getitem__(parent, local_key) is not cur:
            return
        dict.__delitem__(parent, local_key)
        # parent._index[local_key] should already be gone (no refs remaining).
        cur = parent


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_kv_slot(c: Container, key: str, value: Value, doc: Document) -> KVSlot:
    """Synthesise a new ``KVSlot`` carrying default trivia + style."""
    # Promote a header without final newline: e.g. user parsed `a = 1`
    # (no trailing newline) and now appends `b = 2`. The anchor's eol
    # must terminate its line before our new slot starts.
    body_tail = c._body_tail  # noqa: SLF001
    header_ref = c._header_ref  # noqa: SLF001
    anchor_slot: Slot | None = body_tail or (
        header_ref.slot if header_ref is not None else None
    )
    if anchor_slot is not None:
        _ensure_terminator(anchor_slot, doc)

    kp = (
        KeyPart(raw=key, value=key, kind="bare")
        if _RE_BARE_KEY.match(key)
        else KeyPart(raw=_quote_basic(key), value=key, kind="basic")
    )
    return KVSlot(
        leading=Trivia(),
        host_path=c._path,  # noqa: SLF001
        key_parts=[kp],
        key_seps=[],
        pre_eq=" ",
        post_eq=" ",
        value=value,
        eol=EolTrivia(
            trailing_ws=None,
            comment=None,
            newline=NewlineNode(text=doc._newline),  # noqa: SLF001
        ),
        owner_aot_entry=c._owner_aot_entry,  # noqa: SLF001
    )


def _ensure_terminator(slot: Slot, doc: Document) -> None:
    """Give ``slot`` a trailing newline if it lacks one (no-final-newline doc)."""
    if isinstance(slot, (KVSlot, StructuralHeaderSlot)) and slot.eol.newline is None:
        slot.eol = EolTrivia(
            trailing_ws=slot.eol.trailing_ws,
            comment=slot.eol.comment,
            newline=NewlineNode(text=doc._newline),  # noqa: SLF001
        )


def _find_ref_index_by_slot(c: Container, slot: Slot) -> int:
    refs = c._refs  # noqa: SLF001
    for i, r in enumerate(refs):
        if r.slot is slot:
            return i
    msg = "internal: anchor slot not found in c._refs"
    raise AssertionError(msg)


def _recompute_body_tail(c: Container) -> Slot | None:
    """Last body-region ref's slot in ``c._refs`` (mirrors invariants rule)."""
    has_header = c._header_ref is not None  # noqa: SLF001
    own_aot = c._owner_aot_entry  # noqa: SLF001
    own_path = c._path  # noqa: SLF001
    for ref in reversed(c._refs):  # noqa: SLF001
        s = ref.slot
        if not isinstance(s, KVSlot):
            continue
        if s.owner_aot_entry is not own_aot:
            continue
        if has_header and s.host_path != own_path:
            continue
        return s
    if has_header:
        # Header-only container falls back to its own header.
        header_ref = c._header_ref  # noqa: SLF001
        return header_ref.slot if header_ref is not None else None
    return None


def _quote_basic(s: str) -> str:
    out = ['"']
    for ch in s:
        c = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif c < 0x20 or c == 0x7F:
            out.append(f"\\u{c:04X}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


__all__ = [
    "append_direct_kv",
    "delete_key",
    "insert_after",
    "insert_before_head",
    "unlink_slot",
]
