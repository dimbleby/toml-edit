"""Section-side mutation primitives.

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
import copy
from typing import TYPE_CHECKING, Any, Literal

from tomlrt._scalar import is_scalar
from tomlrt._slots import AoTEntry, KVSlot, SlotRef, StructuralHeaderSlot
from tomlrt._trivia import (
    CommentNode,
    EolTrivia,
    NewlineNode,
    Trivia,
    WhitespaceNode,
)
from tomlrt._values import make_keypart, make_keyparts

HeaderKind = Literal["table", "aot-entry"]

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from tomlrt._array import AoT
    from tomlrt._container import Container, Document, Table
    from tomlrt._slots import Slot
    from tomlrt._trivia import TriviaPiece
    from tomlrt._values import Value


# ---------------------------------------------------------------------------
# Pure linked-list ops
# ---------------------------------------------------------------------------


def _ancestor_chain(c: Container) -> list[Container]:
    """Ancestors from ``c._parent`` up to (and including) the document root."""
    out: list[Container] = []
    cur = c._parent  # noqa: SLF001
    while cur is not None:
        out.append(cur)
        cur = cur._parent  # noqa: SLF001
    return out


def _file_ref_at_tail(c: Container, ref: SlotRef) -> None:
    """Append ``ref`` to ``c._refs`` and (when keyed) ``c._index``."""
    c._refs.append(ref)  # noqa: SLF001
    if ref.local_key is not None:
        c._index.setdefault(ref.local_key, []).append(ref)  # noqa: SLF001


def _file_synthetic_header_and_kv(
    c: Container,
    *,
    header_slot: StructuralHeaderSlot,
    key: str,
    value: Value,
    doc: Document,
    owner: AoTEntry | None,
    header_ref_index: int,
) -> KVSlot:
    """Common tail of the two header-synthesis paths.

    Files ``c``'s own-header ref at ``header_ref_index``, builds and
    inserts the ``key = value`` KV directly after ``header_slot``,
    files the KV ref at ``header_ref_index + 1``, and updates
    ``c._header_ref`` / ``c._index[key]`` / ``c._body_tail``. Returns
    the new KV slot so callers can register it on
    ``owner.entry_slots`` and similar bookkeeping.

    Anchoring (where ``header_slot`` itself sits in the slot stream)
    and ancestor binding-ref filing remain explicit in callers; both
    are highly position-sensitive and not safe to share.
    """
    own_header_ref = SlotRef(slot=header_slot, container=c)
    c._refs.insert(header_ref_index, own_header_ref)  # noqa: SLF001
    c._header_ref = own_header_ref  # noqa: SLF001

    new_kv = _new_kv_slot(c, key, value, doc, owner, leading=Trivia())
    insert_after(header_slot, new_kv, doc)
    kv_ref = SlotRef(slot=new_kv, container=c)
    c._refs.insert(header_ref_index + 1, kv_ref)  # noqa: SLF001
    c._index.setdefault(key, []).append(kv_ref)  # noqa: SLF001
    c._body_tail = new_kv  # noqa: SLF001
    return new_kv


def _wire_section_container(
    c: Container,
    *,
    doc: Document,
    path: tuple[str, ...],
    parent: Container,
    owner: AoTEntry | None,
    header: StructuralHeaderSlot,
) -> SlotRef:
    """Initialise a freshly-built container as the owner of ``header``.

    Wires ``_layout_root`` / ``_path`` / ``_parent`` / ``_owner_aot_entry``,
    files the own-header ref onto ``c._refs``, and sets ``_header_ref``.
    Returns the filed ref so callers can keep it in scope (e.g. for
    insertion bookkeeping).
    """
    c._layout_root = doc  # noqa: SLF001
    c._path = path  # noqa: SLF001
    c._parent = parent  # noqa: SLF001
    c._owner_aot_entry = owner  # noqa: SLF001
    ref = SlotRef(slot=header, container=c)
    c._refs.append(ref)  # noqa: SLF001
    c._header_ref = ref  # noqa: SLF001
    return ref


def _init_implicit_table(
    doc: Document,
    path: tuple[str, ...],
    parent: Container,
    owner: AoTEntry | None,
) -> Table:
    """Build an implicit (header-less) Table wired into ``doc`` at ``path``."""
    from tomlrt._container import Table  # noqa: PLC0415

    child = Table()
    child._layout_root = doc  # noqa: SLF001
    child._path = path  # noqa: SLF001
    child._parent = parent  # noqa: SLF001
    child._owner_aot_entry = owner  # noqa: SLF001
    return child


def _rebuild_index_for_key(c: Container, local_key: str) -> None:
    """Restore ``c._index[local_key]`` as the doc-stream subset of ``c._refs``.

    The invariant is that ``_index[k]`` equals the in-order list of refs
    in ``_refs`` whose ``local_key == k``. Rebuild after any mid-stream
    insertion under ``k`` rather than blindly appending — appending
    would be wrong when the new ref is followed by other contributors
    sharing the same key (e.g. a later ``[a.b]`` header).
    """
    c._index[local_key] = [  # noqa: SLF001
        r
        for r in c._refs  # noqa: SLF001
        if r.local_key == local_key
    ]


def _default_eol(doc: Document) -> EolTrivia:
    """A bare-newline `EolTrivia` for a freshly synthesised slot."""
    return EolTrivia(
        trailing_ws=None,
        comment=None,
        newline=NewlineNode(text=doc._newline),  # noqa: SLF001
    )


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


def insert_before(anchor: Slot, new_slot: Slot, doc: Document) -> None:
    """Splice ``new_slot`` immediately before ``anchor`` in ``doc``."""
    p = anchor._prev  # noqa: SLF001
    new_slot._prev = p  # noqa: SLF001
    new_slot._next = anchor  # noqa: SLF001
    anchor._prev = new_slot  # noqa: SLF001
    if p is not None:
        p._next = new_slot  # noqa: SLF001
    else:
        doc._head = new_slot  # noqa: SLF001


def insert_before_head(new_slot: Slot, doc: Document) -> None:
    """Splice ``new_slot`` at the start of ``doc``'s linked list."""
    # Preamble migration: if the doc was previously slotless and any
    # preamble lives in `_trailing` (e.g. set via `Document.preamble`
    # on an empty doc, or a comment-only source), prepend that trivia
    # to the new head's leading and clear the trailing.
    head = doc._head  # noqa: SLF001
    if head is None and doc._trailing.pieces:  # noqa: SLF001
        nl = doc._newline  # noqa: SLF001
        migrated = list(doc._trailing.pieces)  # noqa: SLF001
        # Add a blank-line separator between preamble and content.
        migrated.append(NewlineNode(nl))
        new_slot.leading.pieces = [*migrated, *new_slot.leading.pieces]
        doc._trailing.pieces = []  # noqa: SLF001
    new_slot._prev = None  # noqa: SLF001
    new_slot._next = head  # noqa: SLF001
    if head is not None:
        head._prev = new_slot  # noqa: SLF001
    else:
        doc._tail = new_slot  # noqa: SLF001
    doc._head = new_slot  # noqa: SLF001


def unlink_slot(
    slot: Slot, doc: Document, *, strip_new_head_leading: bool = True
) -> None:
    """Remove ``slot`` from ``doc``'s linked list (does not touch caches).

    When ``strip_new_head_leading`` is True (default), if the unlink
    promotes a successor to be the new doc head, leading blank-line
    pieces on that successor are stripped — what was a separator from
    the removed first slot must not show up as a stray blank at the
    top of the file. Pass False for transient unlinks (e.g. AoT
    renormalise that re-splices the same slots) where the leading
    must be preserved.
    """
    p = slot._prev  # noqa: SLF001
    n = slot._next  # noqa: SLF001
    if p is not None:
        p._next = n  # noqa: SLF001
    else:
        doc._head = n  # noqa: SLF001
        if n is not None and strip_new_head_leading:
            _strip_leading_blank_lines(n)
    if n is not None:
        n._prev = p  # noqa: SLF001
    else:
        doc._tail = p  # noqa: SLF001
    slot._prev = None  # noqa: SLF001
    slot._next = None  # noqa: SLF001


def _strip_leading_blank_lines(slot: Slot) -> None:
    """Drop leading newline-only pieces from ``slot.leading``.

    Comment pieces are preserved (we don't want to silently drop user
    comments). Stops at the first non-newline piece.
    """
    pieces = slot.leading.pieces
    i = 0
    while i < len(pieces) and isinstance(pieces[i], NewlineNode):
        i += 1
    if i:
        del pieces[:i]


# ---------------------------------------------------------------------------
# Higher-level ops
# ---------------------------------------------------------------------------


def append_direct_kv(c: Container, key: str, value: Value) -> None:
    """Append a fresh direct (non-dotted) KV to ``c``.

    Updates ``c._refs`` / ``_index`` / ``_body_tail`` and the dict
    storage.

    Routing:

    * implicit-headerless non-root container with a body anchor →
      dotted-KV synthesis under the nearest header-bearing ancestor;
    * AoT-entry sub-table body → not yet supported;
    * everything else → direct single-keypart KV with anchor =
      body_tail / header_ref / head-of-doc seam.
    """
    if c._path and c._header_ref is None:  # noqa: SLF001
        # Implicit / headerless non-root container. A fresh
        # ``host_path = c._path`` slot would render in whatever scope
        # the previous header (or the doc root) established, not in
        # ``c``'s logical scope — semantic mismatch. Insert via a
        # dotted KV under the nearest header-bearing ancestor instead.
        if c._body_tail is None:  # noqa: SLF001
            # Implicit ``c`` whose only contributors are descendant
            # headers (e.g. ``[a.b]\ny = 1`` then mutating
            # ``doc.table('a')['x']``). Promote ``c`` to an explicit
            # section by synthesising a ``[c._path]`` header before
            # the first descendant slot, then insert the KV directly
            # under it.
            _synthesise_header_then_insert_kv(c, key, value)
            return
        _append_dotted_kv_under_implicit(c, key, value)
        return
    if c._owner_aot_entry is not None and c._header_ref is not None:  # noqa: SLF001
        # AoT-entry root container: header-bearing, header is `[[arr]]`.
        # Body inserts work like normal header-bearing container, but
        # we also need to maintain the entry's `entry_slots` list in
        # doc-stream order.
        _append_kv_in_aot_entry(c, key, value)
        return
    if c._owner_aot_entry is not None:  # noqa: SLF001
        msg = "insert into AoT entry sub-table body is not yet supported"
        raise NotImplementedError(msg)
    layout_root = c._layout_root  # noqa: SLF001
    if layout_root is None:  # pragma: no cover
        msg = "internal: container has no layout root"
        raise AssertionError(msg)
    doc = layout_root
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
        # Section-only doc: insert the new KV at the head of the doc
        # stream, before the first existing slot. Ensure a blank-line
        # separator on what is about to become the second slot, so the
        # new KV does not visually collide with `[s]`.
        old_head = doc._head  # noqa: SLF001
        insert_before_head(new_slot, doc)
        _ensure_leading_blank_line(old_head, doc)
    else:
        # Empty doc (slotless), possibly with preamble trivia in
        # _trailing — insert_before_head migrates that onto the new
        # slot's leading.
        insert_before_head(new_slot, doc)

    new_ref = SlotRef(slot=new_slot, container=c)
    # The new ref's correct position in ``c._refs`` is immediately
    # after the anchor ref (the previous body_tail's ref). With no
    # anchor: head-of-doc insert (3d-5) → index 0, so the ref
    # ordering matches doc-stream (existing section-header refs come
    # after); header-only / empty-doc → end (no preceding refs to
    # order against).
    if body_tail is not None:
        anchor_idx = _find_ref_index_by_slot(c, body_tail)
        c._refs.insert(anchor_idx + 1, new_ref)  # noqa: SLF001
    elif header_ref is None and doc._head is new_slot:  # noqa: SLF001
        # Head-of-doc insert: the new ref must precede any existing
        # section-header refs in c._refs to keep doc-stream order.
        c._refs.insert(0, new_ref)  # noqa: SLF001
    else:
        c._refs.append(new_ref)  # noqa: SLF001
    c._index.setdefault(key, []).append(new_ref)  # noqa: SLF001
    c._body_tail = new_slot  # noqa: SLF001


def delete_key(c: Container, key: str) -> None:
    """Delete ``key`` from ``c`` — scalar, inline, section, AoT, or dotted-subtree.

    Steps:

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

    Cascade-prune is intentionally *not* performed: ``del c[k]``
    follows Python-dict semantics, removing exactly ``k`` and leaving
    any now-emptied implicit ancestor chain reachable as nested empty
    ``Table`` views. Such slotless implicit tables have no rendering
    presence (no header_ref, no refs), so dumps stay byte-correct.

    Held views of the deleted subtree retain stale ``_layout_root`` /
    ``_path``; structural mutation through them raises
    ``NotImplementedError`` rather than corrupting the live document.
    """
    if key not in c:
        raise KeyError(key)
    val = dict.__getitem__(c, key)
    layout_root = c._layout_root  # noqa: SLF001
    if layout_root is None:  # pragma: no cover
        msg = "internal: container has no layout root"
        raise AssertionError(msg)
    doc = layout_root

    # 1. Owned-slot identity set + retained slot objects (for unlink).
    owned_ids: set[int] = set()
    owned_slots: list[Slot] = []

    def _add_slot(s: Slot) -> None:
        if id(s) in owned_ids:
            return
        owned_ids.add(id(s))
        owned_slots.append(s)

    for r in c._index.get(key, []):  # noqa: SLF001
        _add_slot(r.slot)

    # 2. Subtree containers + AoTs + descendant owned slots.
    subtree_containers: list[Container] = []
    subtree_aots: list[AoT] = []
    _collect_subtree(val, subtree_containers, subtree_aots, _add_slot)

    # 3. Slot-driven scrub via back-pointers, *skipping* subtree
    # containers — those are about to be orphaned to a fresh
    # Document and must keep their internal `_refs` / `_index`
    # intact. Chain containers (ancestors + ``c``) and any other
    # live container holding a ref to an owned slot are scrubbed.
    skip_ids = frozenset(id(sc) for sc in subtree_containers)
    _scrub_owned_slots_via_backptrs(owned_slots, skip_container_ids=skip_ids)

    # 4. Body-tail recompute on the ancestor chain. An ancestor at
    # depth ``d < min_owned_depth`` cannot have its body_tail
    # pointing into the owned set, so we don't need to walk past
    # that. For the common leaf-KV delete the chain is just ``c``.
    min_owned_depth = len(c._path)  # noqa: SLF001
    for s in owned_slots:
        d = len(s.host_path) if isinstance(s, KVSlot) else 0
        if d < min_owned_depth:
            min_owned_depth = d
    cur: Container | None = c
    while cur is not None and len(cur._path) >= min_owned_depth:  # noqa: SLF001
        if (
            cur._body_tail is not None  # noqa: SLF001
            and id(cur._body_tail) in owned_ids  # noqa: SLF001
        ):
            cur._body_tail = _recompute_body_tail(cur)  # noqa: SLF001
        cur = cur._parent  # noqa: SLF001

    # 5. Unlink owned slots from the doc; clean up AoTEntry.entry_slots
    # for live (still-attached) entries. Owned slots are then
    # transplanted to an orphan Document if there are subtree
    # containers / AoTs the user may still hold references to.
    #
    # Skip the entry_slots strip for AoTEntries that belong to AoTs
    # being moved to the orphan — those entries are leaving with
    # their slots, and we want their `entry_slots` lists intact so
    # downstream `clone_aot` / re-install paths can read the full
    # CST instead of rebuilding from dict storage.
    moving_aot_entry_ids: set[int] = set()
    for ao in subtree_aots:
        for entry_table in list.__iter__(ao):
            owner_e = entry_table._owner_aot_entry  # noqa: SLF001
            if owner_e is not None:
                moving_aot_entry_ids.add(id(owner_e))

    candidate_owners: set[int] = set()
    for slot in owned_slots:
        owner = slot.owner_aot_entry
        if owner is not None and id(owner) not in moving_aot_entry_ids:
            candidate_owners.add(id(owner))
    surviving_aot_entries = (
        _surviving_aot_entries(doc, candidate_owners) if candidate_owners else set()
    )
    for slot in owned_slots:
        owner = slot.owner_aot_entry
        if (
            owner is not None
            and id(owner) in surviving_aot_entries
            and id(owner) not in moving_aot_entry_ids
        ):
            with contextlib.suppress(ValueError):
                owner.entry_slots.remove(slot)
        unlink_slot(slot, doc)

    if subtree_containers or subtree_aots:
        from tomlrt._container import Document  # noqa: PLC0415

        orphan = Document()
        orphan._newline = doc._newline  # noqa: SLF001
        orphan._is_private = True  # noqa: SLF001
        for slot in owned_slots:
            _splice_at_end(slot, orphan)
        for sc in subtree_containers:
            sc._layout_root = orphan  # noqa: SLF001
        for ao in subtree_aots:
            ao._layout_root = orphan  # noqa: SLF001

    # 6. Drop the dict entry.
    dict.__delitem__(c, key)


def _collect_subtree(
    val: object,
    containers_out: list[Container],
    aots_out: list[AoT],
    add_slot: Callable[[Slot], None],
) -> None:
    """Walk ``val``'s container subtree, collecting containers, AoTs and owned slots."""
    from tomlrt._array import AoT  # noqa: PLC0415
    from tomlrt._container import Container  # noqa: PLC0415

    if isinstance(val, Container):
        if val._inline:  # noqa: SLF001
            return
        containers_out.append(val)
        for r in val._refs:  # noqa: SLF001
            add_slot(r.slot)
        for child in val.values():
            _collect_subtree(child, containers_out, aots_out, add_slot)
    elif isinstance(val, AoT):
        aots_out.append(val)
        for entry in val:
            _collect_subtree(entry, containers_out, aots_out, add_slot)


def _surviving_aot_entries(doc: Document, candidates: set[int]) -> set[int]:
    """Return ``id(AoTEntry)`` values from ``candidates`` still reachable in ``doc``.

    Bails out as soon as every candidate has been spotted.
    """
    from tomlrt._array import AoT  # noqa: PLC0415
    from tomlrt._container import Container  # noqa: PLC0415

    surviving: set[int] = set()
    remaining = set(candidates)

    def visit(v: object) -> None:
        if not remaining:
            return
        if isinstance(v, Container):
            owner = v._owner_aot_entry  # noqa: SLF001
            if owner is not None:
                oid = id(owner)
                if oid in remaining:
                    surviving.add(oid)
                    remaining.discard(oid)
            if not v._inline:  # noqa: SLF001
                for child in v.values():
                    if not remaining:
                        return
                    visit(child)
        elif isinstance(v, AoT):
            for entry in v:
                if not remaining:
                    return
                visit(entry)

    visit(doc)
    return surviving


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _last_kv(c: Container, predicate: Callable[[KVSlot], bool]) -> KVSlot | None:
    """Reverse-walk ``c._refs`` for the last KVSlot satisfying ``predicate``."""
    for ref in reversed(c._refs):  # noqa: SLF001
        s = ref.slot
        if isinstance(s, KVSlot) and predicate(s):
            return s
    return None


def _is_direct_kv(c: Container, s: Slot) -> bool:
    """True iff ``s`` is a direct (single-key-part, host=c) KV of ``c``."""
    return (
        isinstance(s, KVSlot)
        and s.host_path == c._path  # noqa: SLF001
        and len(s.key_parts) == 1
        and s.owner_aot_entry is c._owner_aot_entry  # noqa: SLF001
    )


def _last_direct_kv(c: Container) -> KVSlot | None:
    """Return the most-recent direct KV slot of ``c`` in doc-stream order.

    Fast path: ``c._body_tail`` is by construction the latest body-region
    slot, and on every direct-KV append it IS the new direct KV — so for
    the typical "just-appended" case this is O(1). Otherwise (body_tail
    is a header or a dotted KV) reverse-walk ``c._refs``.
    """
    body_tail = c._body_tail  # noqa: SLF001
    if body_tail is not None and _is_direct_kv(c, body_tail):
        assert isinstance(body_tail, KVSlot)
        return body_tail
    own_path = c._path  # noqa: SLF001
    own_aot = c._owner_aot_entry  # noqa: SLF001
    return _last_kv(
        c,
        lambda s: (
            s.host_path == own_path
            and len(s.key_parts) == 1
            and s.owner_aot_entry is own_aot
        ),
    )


def _extract_indent(leading: Trivia) -> str:
    """Return indent (whitespace after the last newline) of ``leading``."""
    pieces = leading.pieces
    last_nl = -1
    for i, p in enumerate(pieces):
        if isinstance(p, NewlineNode):
            last_nl = i
    text = ""
    for p in pieces[last_nl + 1 :]:
        if isinstance(p, WhitespaceNode):
            text += p.text
        else:
            break
    return text


def _aot_sibling_last_kv(c: Container) -> KVSlot | None:
    """Return the last direct KV of the most recent prior AoT sibling.

    Used to inherit indent when ``c`` is an AoT entry root with no
    direct KVs of its own yet.
    """
    from tomlrt._array import AoT  # noqa: PLC0415

    owner = c._owner_aot_entry  # noqa: SLF001
    if owner is None:
        return None
    parent = c._parent  # noqa: SLF001
    if parent is None:
        return None
    key = c._path[-1] if c._path else None  # noqa: SLF001
    if key is None or key not in parent:
        return None
    aot = dict.__getitem__(parent, key)
    if not isinstance(aot, AoT):
        return None
    found_self = False
    for entry_table in reversed(aot):
        if entry_table is c:
            found_self = True
            continue
        if not found_self:
            continue
        sib = _last_direct_kv(entry_table)
        if sib is not None:
            return sib
    return None


def _leading_has_blank_line(leading: Trivia) -> bool:
    r"""Whether ``leading`` contains at least one blank physical line.

    A blank line is a line in the leading-trivia stream that contains
    no comment piece. A comment-line newline (e.g. ``# foo\n``) does
    not count as a blank — the newline belongs to the comment.
    """
    has_comment = False
    for p in leading.pieces:
        if isinstance(p, CommentNode):
            has_comment = True
        elif isinstance(p, NewlineNode):
            if not has_comment:
                return True
            has_comment = False
    return False


def _peer_separator(prev_leading: Trivia | None, doc: Document) -> Trivia:
    """Mirror a peer's blank-gap when emitting a new structural sibling.

    Returns a single-newline ``Trivia`` (one blank line of separation)
    iff ``prev_leading`` itself contains a blank line, or when there
    is no peer to mirror (the conventional default for the first
    sibling of its kind). Otherwise returns empty ``Trivia``.

    This is the shared "match the last peer" rule used by KV append,
    section-header insertion, and AoT-entry append; each caller wraps
    it with kind-specific peer lookup and any extra decoration (e.g.
    KV indent).
    """
    if prev_leading is None or _leading_has_blank_line(prev_leading):
        return Trivia([NewlineNode(text=doc._newline)])  # noqa: SLF001
    return Trivia()


def _kv_leading_after(
    prev: KVSlot | None, doc: Document, fallback_indent: str = ""
) -> Trivia:
    """Build leading trivia for a new KV slot following ``prev``.

    Inherits indent from ``prev`` and mirrors its blank-gap so the
    new KV continues the user's most recent spacing convention. With
    no prior sibling, falls back to a bare ``fallback_indent``.
    """
    if prev is None:
        if fallback_indent:
            return Trivia([WhitespaceNode(text=fallback_indent)])
        return Trivia()
    pieces: list[TriviaPiece] = list(_peer_separator(prev.leading, doc).pieces)
    indent_text = _extract_indent(prev.leading)
    if indent_text:
        pieces.append(WhitespaceNode(text=indent_text))
    return Trivia(pieces)


def _kv_separator_leading(c: Container, doc: Document) -> Trivia:
    """Pick leading trivia for a new direct-KV slot in container ``c``.

    For an AoT entry with no own KVs yet, falls back to inheriting
    indent (only) from the previous sibling entry's last KV.
    """
    last = _last_direct_kv(c)
    if last is not None:
        return _kv_leading_after(last, doc)
    sibling = _aot_sibling_last_kv(c)
    fallback = _extract_indent(sibling.leading) if sibling is not None else ""
    return _kv_leading_after(None, doc, fallback_indent=fallback)


def _last_host_kv(host: Container) -> KVSlot | None:
    """Last KV slot whose ``host_path`` matches ``host._path`` (any keypath length)."""
    own_path = host._path  # noqa: SLF001
    own_aot = host._owner_aot_entry  # noqa: SLF001
    return _last_kv(
        host,
        lambda s: s.host_path == own_path and s.owner_aot_entry is own_aot,
    )


def _host_kv_separator_leading(host: Container, doc: Document) -> Trivia:
    """Pick leading trivia for a new dotted-KV slot whose host is ``host``."""
    return _kv_leading_after(_last_host_kv(host), doc)


def _new_kv_slot(
    c: Container,
    key: str,
    value: Value,
    doc: Document,
    owner: AoTEntry | None,
    *,
    leading: Trivia,
) -> KVSlot:
    """Synthesise a fresh single-keypart KV slot under ``c``."""
    return KVSlot(
        leading=leading,
        host_path=c._path,  # noqa: SLF001
        key_parts=[make_keypart(key)],
        key_seps=[],
        pre_eq=" ",
        post_eq=" ",
        value=value,
        eol=_default_eol(doc),
        owner_aot_entry=owner,
    )


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

    return _new_kv_slot(
        c,
        key,
        value,
        doc,
        owner=c._owner_aot_entry,  # noqa: SLF001
        leading=_kv_separator_leading(c, doc),
    )


def _append_dotted_kv_under_implicit(c: Container, key: str, value: Value) -> None:
    """Insert into an implicit-headerless container via dotted KV.

    Routes through the nearest header-bearing ancestor (or the doc
    root). Files refs on every implicit ancestor between that host
    and ``c`` per the dotted-KV ref-propagation rule.

    Pre-conditions (checked by caller):
      * ``c._path`` is non-empty (c is not the doc root)
      * ``c._header_ref is None`` (c is implicit-headerless)
      * ``c._body_tail is not None`` (c has at least one dotted-KV
        contributor — anchors the new slot)
    """
    body_tail = c._body_tail  # noqa: SLF001
    assert body_tail is not None
    layout_root = c._layout_root  # noqa: SLF001
    assert layout_root is not None
    doc = layout_root

    # Find host: nearest ancestor with a header, or the doc root.
    host: Container = c
    while host._parent is not None and host._header_ref is None:  # noqa: SLF001
        host = host._parent  # noqa: SLF001

    # Build chain [host, ..., c] via _parent walk + reverse.
    chain: list[Container] = []
    cur: Container | None = c
    while cur is not host:
        assert cur is not None
        chain.append(cur)
        cur = cur._parent  # noqa: SLF001
    chain.append(host)
    chain.reverse()

    # Per-step local keys: from host's path down to (key,).
    local_keys = [*c._path[len(host._path) :], key]  # noqa: SLF001
    assert len(local_keys) == len(chain)

    # AoT consistency: every container in the chain shares the same
    # owner. (The host is either the AoT entry root itself, or the
    # doc root; either way owners match all the way down.)
    owner = c._owner_aot_entry  # noqa: SLF001
    for anc in chain:
        assert anc._owner_aot_entry is owner  # noqa: SLF001

    _ensure_terminator(body_tail, doc)

    # Build the dotted slot: keypath = host..key.
    keypath = (*c._path[len(host._path) :], key)  # noqa: SLF001
    parts = [make_keypart(k) for k in keypath]
    seps = ["."] * (len(parts) - 1)
    new_slot = KVSlot(
        leading=_host_kv_separator_leading(host, doc),
        host_path=host._path,  # noqa: SLF001
        key_parts=parts,
        key_seps=seps,
        pre_eq=" ",
        post_eq=" ",
        value=value,
        eol=_default_eol(doc),
        owner_aot_entry=owner,
    )

    insert_after(body_tail, new_slot, doc)

    # File refs on every chain ancestor. ``_refs`` is the doc-stream
    # subset; ``_index`` preserves "primary at index 0 + all
    # contributors". Appending to ``_index`` keeps any existing
    # structural primary (e.g. a header-owning ref on the host) at
    # index 0; a new dotted contributor is always secondary.
    for i, anc in enumerate(chain):
        new_ref = SlotRef(slot=new_slot, container=anc)
        anchor_idx = _find_ref_index_by_slot(anc, body_tail)
        anc._refs.insert(anchor_idx + 1, new_ref)  # noqa: SLF001
        # ``_index[local_key]`` must equal the doc-stream-ordered
        # subset of ``_refs`` for that key (an invariant). Rebuild it
        # for the affected key rather than blindly appending —
        # appending is only correct when the new ref is also the
        # last with its key in ``_refs``, which fails when the
        # ancestor has later structural-header refs under the same
        # name (e.g. ``a.x = 1`` then ``[a.b]`` — the ``[a.b]``
        # header sits after our new ``a.z`` slot, so the new ref
        # belongs in the middle of ``doc._index['a']``, not the end).
        _rebuild_index_for_key(anc, local_keys[i])
        if anc._body_tail is body_tail:  # noqa: SLF001
            anc._body_tail = new_slot  # noqa: SLF001

    # Maintain AoTEntry.entry_slots in doc-stream order.
    if owner is not None:
        try:
            anchor_idx = owner.entry_slots.index(body_tail)
        except ValueError:
            owner.entry_slots.append(new_slot)
        else:
            owner.entry_slots.insert(anchor_idx + 1, new_slot)


def _synthesise_header_then_insert_kv(c: Container, key: str, value: Value) -> None:
    """Promote a purely-implicit container ``c`` to an explicit section.

    Pre-conditions (checked by caller):
      * ``c._path`` is non-empty
      * ``c._header_ref is None``
      * ``c._body_tail is None``
      * ``c._refs`` is non-empty (at least one descendant binding ref)

    Synthesises a ``[c._path]`` header before the first descendant
    slot in doc-stream order, adopts that descendant's old leading
    onto the synthetic header (preserving the existing seam-from-
    above), and rewrites the descendant's leading to the document's
    current inter-section separator style (compact or blank-line).
    Then inserts a fresh single-keypart KV ``key = value`` directly
    after the synthetic header.
    """
    layout_root = c._layout_root  # noqa: SLF001
    assert layout_root is not None
    doc = layout_root

    if not c._refs:  # noqa: SLF001
        # No descendants left — typically the position-preserving
        # structural-replace path: the previous binding's slots
        # were just deleted, leaving ``c`` purely implicit and
        # empty. Append at end of doc; the outer caller's
        # ``move_slots_to_anchor`` will reposition the synthesised
        # block at the captured anchor.
        _synthesise_header_then_insert_kv_at_doc_tail(c, key, value)
        return
    anchor_slot = c._refs[0].slot  # noqa: SLF001
    owner = c._owner_aot_entry  # noqa: SLF001

    # Build the synthetic header. Adopt the descendant's existing
    # leading (so any preamble / inter-section separator that used
    # to land on the descendant lands on the synthetic header
    # instead) and give the descendant a fresh inter-section leading
    # in the doc's current style (compact / blank-separated).
    adopted_leading = anchor_slot.leading
    new_descendant_leading = _build_section_leading(doc)
    header_slot = _new_section_header(
        c._path,  # noqa: SLF001
        leading=adopted_leading,
        doc=doc,
        kind="table",
        owner_aot_entry=owner,
    )
    insert_before(anchor_slot, header_slot, doc)
    anchor_slot.leading = new_descendant_leading

    # File the new header's refs:
    #   * own-header ref on c (local_key=None);
    #   * binding refs on every ancestor along c._path.
    # Walk ancestor chain (excluding c) top-down so we can name
    # local_keys correctly.
    ancestors = _ancestor_chain(c)
    # ancestors[0] = c._parent, ..., ancestors[-1] = doc root.
    # local_key on each ancestor is c._path[-(distance from c)] —
    # for ancestor at distance d from c, local_key = c._path[-d].
    for d, anc in enumerate(ancestors, start=1):
        local_key = c._path[-d]  # noqa: SLF001
        binding_ref = SlotRef(slot=header_slot, container=anc)
        anchor_idx_anc = _find_ref_index_by_slot(anc, anchor_slot)
        anc._refs.insert(anchor_idx_anc, binding_ref)  # noqa: SLF001
        # Rebuild _index[local_key] to preserve doc-stream order
        # (binding_ref now sits before the descendant's existing
        # binding ref, so it becomes the primary).
        _rebuild_index_for_key(anc, local_key)

    new_kv = _file_synthetic_header_and_kv(
        c,
        header_slot=header_slot,
        key=key,
        value=value,
        doc=doc,
        owner=owner,
        header_ref_index=0,
    )

    # Maintain the AoT entry's slot list when applicable.
    if owner is not None:
        try:
            anchor_idx = owner.entry_slots.index(anchor_slot)
        except ValueError:
            owner.entry_slots.append(header_slot)
            owner.entry_slots.append(new_kv)
        else:
            owner.entry_slots.insert(anchor_idx, header_slot)
            owner.entry_slots.insert(anchor_idx + 1, new_kv)


def _synthesise_header_then_insert_kv_at_doc_tail(
    c: Container, key: str, value: Value
) -> None:
    """Append ``[c._path]`` + ``key = value`` at the end of the doc.

    Used by the structural-replace path when ``c``'s previous
    contributors were just deleted, leaving ``c`` empty and implicit.
    The outer caller (typically ``move_slots_to_anchor``) is
    responsible for repositioning the resulting block to the captured
    anchor when one exists.
    """
    layout_root = c._layout_root  # noqa: SLF001
    assert layout_root is not None
    doc = layout_root
    owner = c._owner_aot_entry  # noqa: SLF001

    header_slot = _new_section_header(
        c._path,  # noqa: SLF001
        leading=_build_section_leading(doc),
        doc=doc,
        kind="table",
        owner_aot_entry=owner,
    )
    # When ``c`` lives inside an AoT entry, the synthesised header
    # MUST sit physically inside that entry's slot region (before the
    # next sibling [[arr]] header), otherwise a re-parse would
    # attribute it to the next entry. Anchor after the entry's last
    # slot rather than ``doc._tail``.
    if owner is not None and owner.entry_slots:
        anchor = owner.entry_slots[-1]
        insert_after(anchor, header_slot, doc)
    elif doc._tail is None:  # noqa: SLF001
        doc._head = header_slot  # noqa: SLF001
        doc._tail = header_slot  # noqa: SLF001
        # Empty doc → no preceding header → drop the leading.
        header_slot.leading = Trivia()
    else:
        insert_after(doc._tail, header_slot, doc)  # noqa: SLF001

    ancestors = _ancestor_chain(c)
    # When ``c`` lives inside an AoT entry and was anchored after
    # ``owner.entry_slots[-1]`` above, the synthesised header sits
    # in the middle of the doc-stream (between this entry's last
    # slot and the next sibling [[arr]] entry). Each ancestor's
    # ``_refs`` is doc-stream-ordered, so we must INSERT the binding
    # ref at the right position rather than appending. Use the set
    # of slots already known to belong to this entry as the marker:
    # find the position just after the last ref whose slot is in
    # that set, then insert there.
    entry_slot_set: set[Slot] | None = None
    if owner is not None and owner.entry_slots:
        entry_slot_set = set(owner.entry_slots)
    for d, anc in enumerate(ancestors, start=1):
        local_key = c._path[-d]  # noqa: SLF001
        binding_ref = SlotRef(slot=header_slot, container=anc)
        if entry_slot_set is not None:
            insert_idx = len(anc._refs)  # noqa: SLF001
            for i in range(len(anc._refs) - 1, -1, -1):  # noqa: SLF001
                if anc._refs[i].slot in entry_slot_set:  # noqa: SLF001
                    insert_idx = i + 1
                    break
            anc._refs.insert(insert_idx, binding_ref)  # noqa: SLF001
            _rebuild_index_for_key(anc, local_key)
        else:
            _file_ref_at_tail(anc, binding_ref)

    new_kv = _file_synthetic_header_and_kv(
        c,
        header_slot=header_slot,
        key=key,
        value=value,
        doc=doc,
        owner=owner,
        header_ref_index=len(c._refs),  # noqa: SLF001
    )

    if owner is not None:
        owner.entry_slots.append(header_slot)
        owner.entry_slots.append(new_kv)


def _append_kv_in_aot_entry(c: Container, key: str, value: Value) -> None:
    """Append a direct KV in an AoT-entry root container's body.

    Mirrors the header-bearing path in `append_direct_kv` but also
    keeps the entry's `entry_slots` list in doc-stream order.
    """
    layout_root = c._layout_root  # noqa: SLF001
    if layout_root is None:  # pragma: no cover
        msg = "internal: container has no layout root"
        raise AssertionError(msg)
    doc = layout_root
    owner = c._owner_aot_entry  # noqa: SLF001
    assert owner is not None
    body_tail = c._body_tail  # noqa: SLF001
    header_ref = c._header_ref  # noqa: SLF001
    assert header_ref is not None

    new_slot = _build_kv_slot(c, key, value, doc)
    anchor: Slot = body_tail if body_tail is not None else header_ref.slot
    _ensure_terminator(anchor, doc)
    insert_after(anchor, new_slot, doc)

    new_ref = SlotRef(slot=new_slot, container=c)
    if body_tail is not None:
        anchor_idx = _find_ref_index_by_slot(c, body_tail)
        c._refs.insert(anchor_idx + 1, new_ref)  # noqa: SLF001
    else:
        c._refs.append(new_ref)  # noqa: SLF001
    c._index.setdefault(key, []).append(new_ref)  # noqa: SLF001
    c._body_tail = new_slot  # noqa: SLF001

    # Maintain entry_slots in doc-stream order. Insert after the anchor
    # if it is in the list, else append.
    try:
        idx = owner.entry_slots.index(anchor)
    except ValueError:
        owner.entry_slots.append(new_slot)
    else:
        owner.entry_slots.insert(idx + 1, new_slot)


def _ensure_terminator(slot: Slot, doc: Document) -> None:
    """Give ``slot`` a trailing newline if it lacks one (no-final-newline doc)."""
    if isinstance(slot, (KVSlot, StructuralHeaderSlot)) and slot.eol.newline is None:
        slot.eol = EolTrivia(
            trailing_ws=slot.eol.trailing_ws,
            comment=slot.eol.comment,
            newline=NewlineNode(text=doc._newline),  # noqa: SLF001
        )


def _ensure_leading_blank_line(slot: Slot, doc: Document) -> None:
    """Ensure ``slot.leading`` begins with a blank line.

    Used by the section-only-doc head-insert path (3d-5) to separate
    a freshly inserted top-level KV from the section header that
    used to be the doc head.

    A run of `pieces` is considered to "start with a blank line"
    when the first non-whitespace piece is a `NewlineNode` (i.e.
    optional leading indent then a bare newline). If a comment
    appears before any newline, we prepend a fresh `NewlineNode`
    so the comment block is visually detached from the new KV.
    """
    pieces = slot.leading.pieces
    for p in pieces:
        if isinstance(p, NewlineNode):
            return
        if isinstance(p, CommentNode):
            break
        # WhitespaceNode: keep scanning.
    pieces.insert(0, NewlineNode(text=doc._newline))  # noqa: SLF001


def _find_ref_index_by_slot(c: Container, slot: Slot) -> int:
    """Locate ``slot``'s ref in ``c._refs``, scanning from both ends.

    Callers pass body-tail / anchor slots whose position in ``c._refs``
    is either near the end (typical: body sits before a few trailing
    sub-section header refs) or near the start (e.g. doc-root with a
    handful of top-level KVs preceding many section headers, as in
    bulk ``doc[k] = inline`` patterns). A two-pronged scan converges
    in O(min(P, N-P)) instead of always degrading to O(N) at one end.
    """
    refs = c._refs  # noqa: SLF001
    lo, hi = 0, len(refs) - 1
    while lo <= hi:
        if refs[hi].slot is slot:
            return hi
        if refs[lo].slot is slot:
            return lo
        lo += 1
        hi -= 1
    msg = "internal: anchor slot not found in c._refs"
    raise AssertionError(msg)


def _recompute_body_tail(c: Container) -> Slot | None:
    """Last body-region ref's slot in ``c._refs`` (mirrors invariants rule)."""
    has_header = c._header_ref is not None  # noqa: SLF001
    own_aot = c._owner_aot_entry  # noqa: SLF001
    own_path = c._path  # noqa: SLF001
    found = _last_kv(
        c,
        lambda s: (
            s.owner_aot_entry is own_aot and (not has_header or s.host_path == own_path)
        ),
    )
    if found is not None:
        return found
    if has_header:
        # Header-only container falls back to its own header.
        header_ref = c._header_ref  # noqa: SLF001
        return header_ref.slot if header_ref is not None else None
    return None


# ---------------------------------------------------------------------------
# Structural attach — section / AoT synthesis
# ---------------------------------------------------------------------------


def _new_section_header(
    path: tuple[str, ...],
    *,
    leading: Trivia,
    doc: Document,
    kind: HeaderKind = "table",
    entry: AoTEntry | None = None,
    owner_aot_entry: AoTEntry | None = None,
) -> StructuralHeaderSlot:
    return StructuralHeaderSlot(
        leading=leading,
        kind=kind,
        path=path,
        key_parts=make_keyparts(path),
        key_seps=["."] * (len(path) - 1),
        eol=_default_eol(doc),
        entry=entry,
        owner_aot_entry=owner_aot_entry,
        synthetic=True,
    )


def _doc_tail_anchor(doc: Document) -> Slot | None:
    """Return the slot to insert *after* when appending at end-of-doc."""
    return doc._tail  # noqa: SLF001


def _slot_in_subtree(slot: Slot, base_path: tuple[str, ...]) -> bool:
    """True iff ``slot``'s logical position is within ``base_path``.

    A slot is "within" if its host (KVSlot) or path (StructuralHeaderSlot)
    starts with ``base_path``.
    """
    if isinstance(slot, KVSlot):
        path = slot.host_path
    elif isinstance(slot, StructuralHeaderSlot):
        path = slot.path
    else:
        return False
    return len(path) >= len(base_path) and path[: len(base_path)] == base_path


def _parent_subtree_tail(parent: Container) -> Slot | None:
    """Return the last slot in ``parent``'s physical subtree.

    Walks forward in the doc-stream linked list from ``parent._refs[-1]``
    while subsequent slots are still descendants of ``parent``'s logical
    path (and share its AoT-entry owner).
    """
    refs = parent._refs  # noqa: SLF001
    if not refs:
        return None
    base_path = parent._path  # noqa: SLF001
    base_owner = parent._owner_aot_entry  # noqa: SLF001
    cur = refs[-1].slot
    while cur._next is not None:  # noqa: SLF001
        nxt = cur._next  # noqa: SLF001
        if nxt.owner_aot_entry is not base_owner:
            break
        if not _slot_in_subtree(nxt, base_path):
            break
        cur = nxt
    return cur


def _splice_at_end(slot: Slot, doc: Document) -> None:
    """Insert ``slot`` at the end of the doc-stream."""
    anchor = _doc_tail_anchor(doc)
    if anchor is None:
        # Empty doc.
        insert_before_head(slot, doc)
    else:
        _ensure_terminator(anchor, doc)
        insert_after(anchor, slot, doc)


def _splice_block_at_parent_anchor(
    slots: list[Slot], parent: Container, doc: Document
) -> None:
    """Splice a contiguous block immediately after parent's subtree tail.

    Used by attach / clone primitives when installing a new structural
    block under ``parent``. Falls back to splice-at-end if ``parent``
    has no contributing slots yet (so anchor would be None).
    """
    anchor = _parent_subtree_tail(parent)
    if anchor is None:
        _splice_at_end(slots[0], doc)
    else:
        _ensure_terminator(anchor, doc)
        insert_after(anchor, slots[0], doc)
    prev: Slot = slots[0]
    for s in slots[1:]:
        _ensure_terminator(prev, doc)
        insert_after(prev, s, doc)
        prev = s


def _maybe_demote_synthetic_empty_header(parent: Container) -> None:
    """Drop ``parent``'s header if it is synthetic and has no direct KV body.

    Used after attaching a child header under ``parent``: if ``parent``
    was synthesised as an empty placeholder (e.g.
    ``doc["tool"] = Table.section({})``) and the new child gives it a
    dotted-implicit anchor (``[tool.poetry]``), the placeholder header
    is redundant and is removed.
    """
    hdr_ref = parent._header_ref  # noqa: SLF001
    if hdr_ref is None:
        return
    header = hdr_ref.slot
    assert isinstance(header, StructuralHeaderSlot)
    if not header.synthetic or header.kind != "table":
        return
    # The header's physical body is the doc-stream span from the slot
    # after the header up to (but not including) the next structural
    # header or EOF.  Walk it; if any KVSlot lives there, keep the
    # header.
    s = header._next  # noqa: SLF001
    while s is not None and not isinstance(s, StructuralHeaderSlot):
        if isinstance(s, KVSlot):
            return
        s = s._next  # noqa: SLF001
    layout_root = parent._layout_root  # noqa: SLF001
    from tomlrt._container import Document  # noqa: PLC0415

    assert isinstance(layout_root, Document)
    doc = layout_root
    # Remove the header from the doc stream and from all caches.
    unlink_slot(header, doc, strip_new_head_leading=True)
    parent._header_ref = None  # noqa: SLF001
    parent._refs = [r for r in parent._refs if r is not hdr_ref]  # noqa: SLF001
    # Also clear it from any prefix container's _refs / _index.
    grand: Container | None = parent._parent  # noqa: SLF001
    while grand is not None:
        kept = [r for r in grand._refs if r.slot is not header]  # noqa: SLF001
        if len(kept) != len(grand._refs):  # noqa: SLF001
            grand._refs = kept  # noqa: SLF001
            new_index: dict[str, list[SlotRef]] = {}
            for r in kept:
                if r.local_key is not None:
                    new_index.setdefault(r.local_key, []).append(r)
            grand._index = new_index  # noqa: SLF001
        grand = grand._parent  # noqa: SLF001
    # Owner aot-entry, if any, also drops it.
    owner = header.owner_aot_entry
    if owner is not None:
        with contextlib.suppress(ValueError):
            owner.entry_slots.remove(header)


def _split_leading_structural(leading: Trivia) -> tuple[Trivia, Trivia]:
    """Split a leading-trivia stream into (structural-prefix, comment-remainder).

    The structural prefix is the run of whitespace and newline pieces
    before the first comment piece (if any). The remainder starts at
    the first comment piece and includes everything after it. If
    there is no comment piece, the whole leading is structural and
    the remainder is empty.

    Used by AoT reorder: structural separators stay positional;
    comment remainders travel with their entry.
    """
    pieces = leading.pieces
    cut = len(pieces)
    for i, p in enumerate(pieces):
        if isinstance(p, CommentNode):
            cut = i
            break
    return Trivia(list(pieces[:cut])), Trivia(list(pieces[cut:]))


def _build_section_leading(doc: Document) -> Trivia:
    """Trivia for a fresh section header.

    Empty doc → no leading; non-empty → mirror the most recent
    structural-header's blank-gap. The first header in the doc is
    treated as having an "implicit blank" peer (its own leading is
    the file preamble, not a separator), so subsequent headers get
    one blank line by default.
    """
    if doc._head is None:  # noqa: SLF001
        return Trivia()
    cur: Slot | None = doc._tail  # noqa: SLF001
    last_header: StructuralHeaderSlot | None = None
    while cur is not None:
        if isinstance(cur, StructuralHeaderSlot):
            last_header = cur
            break
        cur = cur._prev  # noqa: SLF001
    if last_header is None:
        return Trivia([NewlineNode(text=doc._newline)])  # noqa: SLF001
    p: Slot | None = last_header._prev  # noqa: SLF001
    while p is not None:
        if isinstance(p, StructuralHeaderSlot):
            return _peer_separator(last_header.leading, doc)
        p = p._prev  # noqa: SLF001
    # last_header is the first header in the doc; its leading is the
    # preamble, not a peer separator. Treat as no-peer.
    return _peer_separator(None, doc)


def attach_empty_aot(parent: Container, key: str, source_aot: AoT) -> AoT:
    """Bind an empty AoT under ``parent[key]``.

    No physical slots are created; subsequent ``aot.add(...)`` calls
    will materialise the first ``[[path]]`` header. The ``source_aot``
    is rehomed in place (identity preserved).
    """
    if len(source_aot) > 0:  # pragma: no cover
        msg = "non-empty AoT live-attach has its own routing"
        raise AssertionError(msg)
    # Rehome the orphan AoT into this parent's logical scope.
    source_aot._layout_root = parent._layout_root  # noqa: SLF001
    source_aot._path = (*parent._path, key)  # noqa: SLF001
    source_aot._parent = parent  # noqa: SLF001
    return source_aot


def _aot_separator(aot: AoT, doc: Document) -> Trivia:
    """Pick the leading-trivia for a newly-appended AoT entry header.

    Mirrors the most recent entry's blank-gap; for the first entry
    (or an empty/zero-slot last entry), defaults to one blank line.
    """
    if len(aot) <= 1:
        return _peer_separator(None, doc)
    last_entry = aot[-1]._owner_aot_entry  # noqa: SLF001
    if last_entry is None or not last_entry.entry_slots:
        return _peer_separator(None, doc)
    return _peer_separator(last_entry.entry_slots[0].leading, doc)


def add_aot_entry(
    aot: AoT, body: Mapping[str, Any] | None, *, rehome: Table | None = None
) -> Table:
    """Append a ``[[path]]`` entry to ``aot`` and return its `Table` view.

    If ``rehome`` is supplied (must be an unattached ``Table``), it is
    used as the entry view so the caller can preserve identity for a
    user reference. ``body`` is then ignored — ``rehome``'s own dict
    storage is used as the source body and is cleared/repopulated
    in place.
    """
    from tomlrt._container import (  # noqa: PLC0415
        Table,
        _is_synth_inline,
        _synth_value,
    )

    parent = aot._parent  # noqa: SLF001
    layout_root = aot._layout_root  # noqa: SLF001
    path = aot._path  # noqa: SLF001
    if layout_root is None or parent is None or not path:
        msg = "AoT.add requires the AoT to be attached to a document"
        raise RuntimeError(msg)
    doc = layout_root

    ordinal = len(aot)
    entry = AoTEntry(path=path, ordinal=ordinal)
    leading = _build_section_leading(doc) if ordinal == 0 else _aot_separator(aot, doc)
    header = _new_section_header(
        path,
        leading=leading,
        doc=doc,
        kind="aot-entry",
        entry=entry,
        owner_aot_entry=entry,
    )
    entry.entry_slots.append(header)

    # Build entry-root container (or rehome an existing one).
    body_items: list[tuple[str, object]]
    if rehome is not None:
        assert isinstance(rehome, Table)
        assert rehome._layout_root is None  # noqa: SLF001
        entry_table = rehome
        body_items = list(rehome.items())
        dict.clear(entry_table)
    else:
        entry_table = Table()
        body_items = list(_items_for_synth(body)) if body is not None else []
    _wire_section_container(
        entry_table,
        doc=doc,
        path=path,
        parent=parent,
        owner=entry,
        header=header,
    )

    # Splice header after the last existing AoT-owned slot if any,
    # else at end-of-doc.
    anchor = _last_aot_slot(aot)
    if anchor is None:
        _splice_at_end(header, doc)
    else:
        _ensure_terminator(anchor, doc)
        insert_after(anchor, header, doc)

    # File parent-chain refs for this entry's header.
    parent_ref = SlotRef(slot=header, container=parent)
    _file_ref_at_tail(parent, parent_ref)

    # Append the new entry to the AoT view list.
    list.append(aot, entry_table)

    # If this is the very first entry of the AoT and the parent was
    # an empty synthetic placeholder section (e.g.
    # `doc["tool"] = Table.section({}); doc["tool"]["list"] = AoT(...)`),
    # the parent's header is now redundant — the dotted-implicit
    # anchor lives entirely in `[[tool.list]]`.
    if ordinal == 0:
        _maybe_demote_synthetic_empty_header(parent)

    # Populate body.
    for k, v in body_items:
        if not (is_scalar(v) or _is_synth_inline(v)):
            entry_table[k] = v
            continue
        cst, dec = _synth_value(
            v,
            layout_root=doc,
            parent=entry_table,
            path=(*path, k),
            owner=entry,
        )
        _append_kv_in_aot_entry(entry_table, k, cst)
        dict.__setitem__(entry_table, k, dec)
    return entry_table


def clone_aot_entry(
    aot: AoT,
    src_entry_table: Container,
    *,
    dst_path: tuple[str, ...] | None = None,
) -> Table:
    """Append a deep CST clone of ``src_entry_table`` to ``aot``.

    Preserves the source entry's per-slot leading / EOL / lexeme bytes
    so per-entry comments and trailing-comment formatting survive.
    The cloned entry's header leading is rewritten to the
    ``_aot_separator`` policy for "next entry", but any post-structural
    comment block on the source header is retained.

    Supports source entries that include nested ``[a.sub]`` headers
    and their KVSlots. ``dst_path`` (defaults to ``aot._path``) lets
    callers rebase the entry under a different key — both the
    entry header path and any nested sub-section paths are rewritten
    so e.g. ``[a.sub]`` becomes ``[b.sub]`` when cloning into ``b``.

    Returns the new ``Table`` view.
    """
    src_entry = src_entry_table._owner_aot_entry  # noqa: SLF001
    if src_entry is None:  # pragma: no cover
        msg = "Source entry has no owning AoTEntry"
        raise RuntimeError(msg)
    src_layout_root = src_entry_table._layout_root  # noqa: SLF001
    return _clone_aot_entry_impl(
        aot,
        src_entry,
        src_layout_root=src_layout_root,
        dst_path=dst_path,
    )


def clone_aot_entry_from(aot: AoT, src_entry: AoTEntry) -> Table:
    """Like ``clone_aot_entry`` but driven by a bare ``AoTEntry``.

    Used by the AoT private-orphan rehome path where the source entry
    table has already been reset (so its ``_owner_aot_entry`` /
    ``_layout_root`` are gone) but the underlying ``AoTEntry``'s
    ``entry_slots`` are intact in a private orphan document. We
    deep-clone those slots into a fresh entry under ``aot``,
    preserving per-KV trivia and any nested sub-section formatting
    that the lossy ``add_aot_entry(rehome=)`` path would drop.
    """
    return _clone_aot_entry_impl(
        aot,
        src_entry,
        src_layout_root=None,
        dst_path=None,
    )


def _install_cloned_aot_entry(
    aot: AoT,
    src_slots: list[Slot],
    src_prefix: tuple[str, ...],
    *,
    target_path: tuple[str, ...],
    rewrite_separator: bool,
) -> Table:
    """Common installer for appending a cloned aot-entry to ``aot``.

    Deep-clones ``src_slots`` (head_kind="aot-entry"), wires a fresh
    entry container, splices the entry's slots after the AoT's last
    slot (or at doc end), files the parent binding ref, and populates
    child views.

    ``rewrite_separator``: if True, the source's structural leading
    is replaced with destination-style preamble (entry 0) or the
    AoT's existing inter-entry separator (entry > 0). If False
    (cross-doc / cross-key clone of an existing AoT entry past the
    first), keep the source's leading verbatim so the original
    inter-entry separator survives.
    """
    from tomlrt._container import Table  # noqa: PLC0415

    parent = aot._parent  # noqa: SLF001
    layout_root = aot._layout_root  # noqa: SLF001
    if layout_root is None or parent is None or not target_path:
        msg = "AoT entry install requires the AoT to be attached to a document"
        raise RuntimeError(msg)
    doc = layout_root

    ordinal = len(aot)
    new_entry = AoTEntry(path=target_path, ordinal=ordinal)

    cloned_slots = _clone_entry_slots(
        src_slots,
        new_entry=new_entry,
        body_owner=new_entry,
        src_prefix=src_prefix,
        target_prefix=target_path,
        head_kind="aot-entry",
    )

    head = cloned_slots[0]
    assert isinstance(head, StructuralHeaderSlot)
    cloned_header: StructuralHeaderSlot = head
    if ordinal == 0:
        _structural, remainder = _split_leading_structural(cloned_header.leading)
        sep = _build_section_leading(doc)
        cloned_header.leading = Trivia([*sep.pieces, *remainder.pieces])
    elif rewrite_separator:
        _structural, remainder = _split_leading_structural(cloned_header.leading)
        sep = _aot_separator(aot, doc)
        cloned_header.leading = Trivia([*sep.pieces, *remainder.pieces])
    # else: keep source leading verbatim (cross-doc / cross-key).

    entry_table = Table()
    _wire_section_container(
        entry_table,
        doc=doc,
        path=target_path,
        parent=parent,
        owner=new_entry,
        header=cloned_header,
    )

    anchor = _last_aot_slot(aot)
    if anchor is None:
        _splice_at_end(cloned_header, doc)
    else:
        _ensure_terminator(anchor, doc)
        insert_after(anchor, cloned_header, doc)
    prev: Slot = cloned_header
    for s in cloned_slots[1:]:  # pragma: no cover
        _ensure_terminator(prev, doc)
        insert_after(prev, s, doc)
        prev = s

    parent_ref = SlotRef(slot=cloned_header, container=parent)
    _file_ref_at_tail(parent, parent_ref)

    _populate_entry_views(
        entry_table=entry_table,
        cloned_slots=cloned_slots[1:],
        target_prefix=target_path,
        body_owner=new_entry,
        doc=doc,
    )

    list.append(aot, entry_table)
    return entry_table


def _clone_aot_entry_impl(
    aot: AoT,
    src_entry: AoTEntry,
    *,
    src_layout_root: Document | None,
    dst_path: tuple[str, ...] | None,
) -> Table:
    layout_root = aot._layout_root  # noqa: SLF001
    path = aot._path  # noqa: SLF001
    target_path = dst_path if dst_path is not None else path
    src_slots = _validate_clonable_aot_entry(src_entry)
    same_aot_clone = target_path == src_entry.path and src_layout_root is layout_root
    return _install_cloned_aot_entry(
        aot,
        src_slots,
        src_entry.path,
        target_path=target_path,
        rewrite_separator=same_aot_clone,
    )


def _install_cloned_section(
    parent: Container,
    key: str,
    src_slots: list[Slot],
    src_prefix: tuple[str, ...],
) -> Table:
    """Common installer for ``parent[key] = <cloned section>``.

    Deep-clones ``src_slots`` (rewriting head from ``[..]`` / ``[[..]]``
    to ``[<key>]``, rebasing paths from ``src_prefix`` to
    ``parent._path + (key,)``), wires the section container, splices
    the slots in at the parent's subtree anchor, files the parent
    binding ref, and populates child views. Used by both
    ``clone_aot_entry_as_table`` and ``clone_section_as_section``.
    """
    from tomlrt._container import Table  # noqa: PLC0415

    layout_root = parent._layout_root  # noqa: SLF001
    if layout_root is None:  # pragma: no cover
        msg = "cloned-section install requires parent attached to a document"
        raise RuntimeError(msg)
    doc = layout_root
    target_path = (*parent._path, key)  # noqa: SLF001

    cloned_slots = _clone_entry_slots(
        src_slots,
        new_entry=None,
        body_owner=parent._owner_aot_entry,  # noqa: SLF001
        src_prefix=src_prefix,
        target_prefix=target_path,
        head_kind="table",
    )

    head = cloned_slots[0]
    assert isinstance(head, StructuralHeaderSlot)
    cloned_header: StructuralHeaderSlot = head
    cloned_header.leading = _build_section_leading(doc)

    section = Table.section()
    _wire_section_container(
        section,
        doc=doc,
        path=target_path,
        parent=parent,
        owner=parent._owner_aot_entry,  # noqa: SLF001
        header=cloned_header,
    )

    _splice_block_at_parent_anchor(cloned_slots, parent, doc)

    parent_ref = SlotRef(slot=cloned_header, container=parent)
    _file_ref_at_tail(parent, parent_ref)

    _populate_entry_views(
        entry_table=section,
        cloned_slots=cloned_slots[1:],
        target_prefix=target_path,
        body_owner=parent._owner_aot_entry,  # noqa: SLF001
        doc=doc,
    )

    dict.__setitem__(parent, key, section)
    return section


def clone_aot_entry_as_table(
    parent: Container,
    key: str,
    src_entry_table: Container,
) -> Table:
    """Install an AoT entry under ``parent[key]`` as a standard ``[key]`` table.

    Used by ``parent[key] = some_aot_entry`` and ``install`` paths.
    Deep-clones the source entry's slots, rewriting the head from
    ``[[..]]`` to ``[..]``, rebasing all paths from the source's
    AoT prefix to ``parent._path + (key,)``.
    """
    src_entry = src_entry_table._owner_aot_entry  # noqa: SLF001
    if src_entry is None:  # pragma: no cover
        msg = "Source entry has no owning AoTEntry"
        raise RuntimeError(msg)
    src_slots = _validate_clonable_aot_entry(src_entry)
    return _install_cloned_section(parent, key, src_slots, src_entry.path)


def _gather_section_slots(src_table: Container) -> list[Slot]:
    """Collect a standard section's owned slots in doc-stream order.

    Includes the section's own header, every direct/dotted KV slot,
    and every nested sub-section's header + KV slots — i.e. the
    entire physical body of ``src_table``.
    """
    if src_table._header_ref is None:  # noqa: SLF001
        msg = "Source table has no structural header to clone"
        raise RuntimeError(msg)

    owned: set[int] = set()

    def _add_slot(s: Slot) -> None:
        owned.add(id(s))

    containers_out: list[Container] = []
    aots_out: list[AoT] = []
    _collect_subtree(src_table, containers_out, aots_out, _add_slot)

    head_slot: Slot = src_table._header_ref.slot  # noqa: SLF001
    out: list[Slot] = [head_slot]
    seen_slots: set[int] = {id(head_slot)}
    cur: Slot | None = head_slot._next  # noqa: SLF001
    while cur is not None and id(cur) in owned:
        if id(cur) not in seen_slots:
            out.append(cur)
            seen_slots.add(id(cur))
        cur = cur._next  # noqa: SLF001
    return out


def clone_table_as_aot_entry(
    aot: AoT,
    src_table: Container,
) -> Table:
    """Append ``src_table`` (a standard ``[k]`` section) to ``aot`` as an entry.

    Deep-clones the source section's slots, rewriting the head from
    ``[k]`` to ``[[aot._path]]``, rebasing all paths from the source's
    section path to ``aot._path``. Preserves per-slot leading / EOL
    / lexeme bytes (so per-section comments survive).
    """
    src_slots = _gather_section_slots(src_table)
    if not isinstance(src_slots[0], StructuralHeaderSlot):  # pragma: no cover
        msg = "Source section's first owned slot is not a header"
        raise AssertionError(msg)  # noqa: TRY004
    if src_slots[0].kind != "table":  # pragma: no cover
        msg = "clone_table_as_aot_entry: source must be a standard section"
        raise RuntimeError(msg)
    return _install_cloned_aot_entry(
        aot,
        src_slots,
        src_table._path,  # noqa: SLF001
        target_path=aot._path,  # noqa: SLF001
        rewrite_separator=True,
    )


def clone_section_as_section(
    parent: Container,
    key: str,
    src_table: Container,
) -> Table:
    """Install a deep clone of a standard section under ``parent[key]``.

    Used for cross-doc table assignment / same-doc clone of an
    attached standard ``[k]`` section. Preserves per-slot trivia
    (header leading, KV leading / EOL) and any nested sub-sections
    by deep-cloning every owned slot and rebasing paths from
    ``src_table._path`` to ``parent._path + (key,)``.
    """
    src_slots = _gather_section_slots(src_table)
    if not isinstance(src_slots[0], StructuralHeaderSlot):  # pragma: no cover
        msg = "Source section's first owned slot is not a header"
        raise AssertionError(msg)  # noqa: TRY004
    return _install_cloned_section(parent, key, src_slots, src_table._path)  # noqa: SLF001


def clone_aot(
    parent: Container,
    key: str,
    src_aot: AoT,
) -> AoT:
    """Install ``src_aot`` (an attached AoT) under ``parent[key]``.

    Each entry is deep-cloned with path-rebasing so any nested
    sub-sections stay logically inside the new key.
    """
    from tomlrt._array import AoT  # noqa: PLC0415

    layout_root = parent._layout_root  # noqa: SLF001
    assert layout_root is not None
    target_path = (*parent._path, key)  # noqa: SLF001

    new_aot = AoT()
    new_aot._layout_root = layout_root  # noqa: SLF001
    new_aot._path = target_path  # noqa: SLF001
    new_aot._parent = parent  # noqa: SLF001

    parent_index_present = key in parent._index  # noqa: SLF001
    if not parent_index_present:
        # No physical primary yet (empty AoT placeholder).
        pass
    dict.__setitem__(parent, key, new_aot)
    for src_entry_table in list(src_aot):
        clone_aot_entry(new_aot, src_entry_table, dst_path=target_path)
    return new_aot


def _clone_entry_slots(
    src_slots: list[Slot],
    *,
    new_entry: AoTEntry | None,
    body_owner: AoTEntry | None,
    src_prefix: tuple[str, ...],
    target_prefix: tuple[str, ...],
    head_kind: HeaderKind,
    has_header: bool = True,
) -> list[Slot]:
    """Deep-clone an entry's slot list with path/owner rebasing.

    When ``has_header`` is True (default), the first slot must be the
    entry header and its ``kind`` is set to ``head_kind``. When
    False, the slot list is treated as body-only — useful for
    in-place body replacement that keeps the destination's existing
    header.

    ``body_owner`` is written to every slot's ``owner_aot_entry`` (so
    cloning into a table that itself sits under another AoT entry
    keeps physical ownership coherent). ``new_entry`` is the
    AoTEntry the cloned slots are *logically* owned by — used only
    for ``entry`` back-pointers on aot-entry headers and for the
    ``entry_slots`` membership list.
    """
    cloned: list[Slot] = []
    for s in src_slots:
        c: Slot = copy.deepcopy(s)
        c._prev = None  # noqa: SLF001
        c._next = None  # noqa: SLF001
        if isinstance(c, KVSlot):
            c.owner_aot_entry = body_owner
            c.host_path = _rebase_path(c.host_path, src_prefix, target_prefix)
        elif isinstance(c, StructuralHeaderSlot):
            c.owner_aot_entry = body_owner
            c.path = _rebase_path(c.path, src_prefix, target_prefix)
            c.key_parts = make_keyparts(c.path)
            c.key_seps = ["."] * (len(c.key_parts) - 1)
            c.entry = new_entry if c.kind == "aot-entry" else None
        cloned.append(c)
        if new_entry is not None:
            new_entry.entry_slots.append(c)

    if not has_header or not cloned:
        return cloned
    head = cloned[0]
    assert isinstance(head, StructuralHeaderSlot)
    head.kind = head_kind
    if head_kind == "aot-entry":
        head.entry = new_entry
        head.owner_aot_entry = new_entry
    else:
        head.entry = None
    return cloned


def _rebase_path(
    p: tuple[str, ...],
    src_prefix: tuple[str, ...],
    target_prefix: tuple[str, ...],
) -> tuple[str, ...]:
    """Replace a leading ``src_prefix`` in ``p`` with ``target_prefix``."""
    if src_prefix == target_prefix:
        return p
    if p[: len(src_prefix)] == src_prefix:
        return target_prefix + p[len(src_prefix) :]
    return p


def _populate_entry_views(
    *,
    entry_table: Container,
    cloned_slots: list[Slot],
    target_prefix: tuple[str, ...],
    body_owner: AoTEntry | None,
    doc: Document,
) -> None:
    """Walk cloned non-header slots, building child Container views.

    Mirrors the parser's slot-builder for an entry: KV slots file
    refs into their host container; sub-section headers create child
    Containers under the entry root and file own-header refs +
    parent-binding refs.

    Decoded Python values are derived from each slot's (already
    deep-cloned) ``Value`` via ``_decode_value`` — never aliased
    from the source dict — so the destination view is fully
    independent of the source.
    """
    from tomlrt._build import _decode_value  # noqa: PLC0415
    from tomlrt._container import Table  # noqa: PLC0415

    # path -> Container for every container in the entry sub-tree.
    containers: dict[tuple[str, ...], Container] = {target_prefix: entry_table}

    def _ensure_container(path: tuple[str, ...]) -> Container:
        if path in containers:
            return containers[path]
        cur: Container = entry_table
        cur_path = target_prefix
        for comp in path[len(target_prefix) :]:
            cur_path = (*cur_path, comp)
            if cur_path in containers:
                cur = containers[cur_path]
                continue
            child = _init_implicit_table(doc, cur_path, cur, body_owner)
            containers[cur_path] = child
            dict.__setitem__(cur, comp, child)
            cur = child
        return cur

    for s in cloned_slots:
        if isinstance(s, StructuralHeaderSlot):
            assert s.kind == "table"
            container = _ensure_container(s.path)
            own_ref = SlotRef(slot=s, container=container)
            container._refs.append(own_ref)  # noqa: SLF001
            container._header_ref = own_ref  # noqa: SLF001
            if s.path == target_prefix:
                continue
            parent_path = s.path[:-1]
            parent_view = _ensure_container(parent_path)
            binding = SlotRef(slot=s, container=parent_view)
            _file_ref_at_tail(parent_view, binding)
            continue
        assert isinstance(s, KVSlot)
        host = _ensure_container(s.host_path)
        assert s.value is not None
        slot_value = s.value
        if len(s.key_parts) == 1:
            key = s.key_parts[0].value
            kv_ref = SlotRef(slot=s, container=host)
            _file_ref_at_tail(host, kv_ref)
            decoded = _decode_value(
                slot_value,
                layout_root=doc,
                parent=host,
                path=(*host._path, key),  # noqa: SLF001
                owner=body_owner,
            )
            dict.__setitem__(host, key, decoded)
        else:
            cur = host
            for kp in s.key_parts[:-1]:
                comp = kp.value
                ref = SlotRef(slot=s, container=cur)
                _file_ref_at_tail(cur, ref)
                if comp not in cur:
                    sub = _init_implicit_table(doc, (*cur._path, comp), cur, body_owner)  # noqa: SLF001
                    containers[sub._path] = sub  # noqa: SLF001
                    dict.__setitem__(cur, comp, sub)
                nxt = dict.__getitem__(cur, comp)
                if not isinstance(nxt, Table):  # pragma: no cover
                    msg = "internal: dotted KV traversal hit non-Table"
                    raise AssertionError(msg)  # noqa: TRY004
                cur = nxt
            leaf_key = s.key_parts[-1].value
            kv_ref = SlotRef(slot=s, container=cur)
            _file_ref_at_tail(cur, kv_ref)
            decoded = _decode_value(
                slot_value,
                layout_root=doc,
                parent=cur,
                path=(*cur._path, leaf_key),  # noqa: SLF001
                owner=body_owner,
            )
            dict.__setitem__(cur, leaf_key, decoded)


def _validate_clonable_aot_entry(src_entry: AoTEntry) -> list[Slot]:
    """Validate the source entry can be cloned; return its slot list.

    Pure validation — no mutation. Raises NotImplementedError for
    shapes the cloner does not yet support, RuntimeError for
    invariant violations. Shared between `clone_aot_entry` and
    `check_clone_aot_entry` so the two cannot drift.
    """
    src_slots = list(src_entry.entry_slots)
    if not src_slots or not isinstance(
        src_slots[0], StructuralHeaderSlot
    ):  # pragma: no cover
        msg = "Source entry has no header slot"
        raise RuntimeError(msg)
    for s in src_slots[1:]:
        if isinstance(s, StructuralHeaderSlot):
            if s.kind != "table":
                msg = "AoT clone for nested non-table headers is not yet implemented"
                raise NotImplementedError(msg)
            continue
        if not isinstance(s, KVSlot):  # pragma: no cover
            msg = "AoT clone: unexpected slot type"
            raise AssertionError(msg)  # noqa: TRY004
    return src_slots


def check_clone_aot_entry(aot: AoT, src_entry_table: Container) -> None:
    """Raise NotImplementedError/RuntimeError if `clone_aot_entry` would.

    Same preconditions as `clone_aot_entry`, but without any side
    effects. Used by `AoT.__imul__` to preflight every source entry
    so a failure on entry N does not leave entries 0..N-1 cloned.
    """
    if (
        aot._layout_root is None  # noqa: SLF001
        or aot._parent is None  # noqa: SLF001
        or not aot._path  # noqa: SLF001
    ):
        msg = "AoT.clone_entry requires the AoT to be attached to a document"
        raise RuntimeError(msg)
    src_entry = src_entry_table._owner_aot_entry  # noqa: SLF001
    if src_entry is None:  # pragma: no cover
        msg = "Source entry has no owning AoTEntry"
        raise RuntimeError(msg)
    _validate_clonable_aot_entry(src_entry)


def attach_section_at(
    parent: Container,
    sub_path: tuple[str, ...] | list[str],
    source: Mapping[str, Any] | Container | None = None,
) -> Any:
    """Synthesise ``[parent_path.sub_path]`` (multi-component) at end-of-doc.

    Intermediate components in ``sub_path[:-1]`` become implicit tables;
    the deepest component gets the explicit header. ``source`` may be
    a `Table` (rehomed) or a Mapping (snapshotted) or ``None``.
    """
    from tomlrt._container import (  # noqa: PLC0415
        Container,
        Table,
        _is_synth_inline,
        _synth_value,
    )

    sub = tuple(sub_path)
    if not sub:
        msg = "sub_path must not be empty"
        raise ValueError(msg)
    if len(sub) == 1:
        return attach_section(parent, sub[0], source)

    layout_root = parent._layout_root  # noqa: SLF001
    if layout_root is None:  # pragma: no cover
        msg = "internal: parent has no layout root"
        raise AssertionError(msg)
    doc = layout_root
    full_path = (*parent._path, *sub)  # noqa: SLF001

    leading = _build_section_leading(doc)
    owner = parent._owner_aot_entry  # noqa: SLF001
    header = _new_section_header(
        full_path,
        leading=leading,
        doc=doc,
        kind="table",
        owner_aot_entry=owner,
    )

    # Build implicit chain: each intermediate is a Table view living
    # in dict storage but with no own header ref.
    chain: list[Container] = [parent]
    for j, comp in enumerate(sub[:-1]):
        cur = chain[-1]
        if comp in cur:
            nxt = dict.__getitem__(cur, comp)
            if not isinstance(nxt, Container):
                msg = f"intermediate {comp!r} is not a table"
                raise TypeError(msg)
            chain.append(nxt)
            continue
        implicit = _init_implicit_table(
            doc,
            (*parent._path, *sub[: j + 1]),  # noqa: SLF001
            cur,
            owner,
        )
        dict.__setitem__(cur, comp, implicit)
        chain.append(implicit)

    if isinstance(source, Table) and source._layout_root is None:  # noqa: SLF001
        section = source
        pending: list[tuple[str, object]] = list(source.items())
        dict.clear(section)
    else:
        section = Table()
        pending = list(_items_for_synth(source)) if source is not None else []

    _wire_section_container(
        section,
        doc=doc,
        path=full_path,
        parent=chain[-1],
        owner=None,
        header=header,
    )

    if owner is not None and owner.entry_slots:
        # AoT-entry-aware splice: place the synthetic header inside the
        # owning entry's physical span and have the entry own it so a
        # later removal of the entry takes the header with it.
        anchor = owner.entry_slots[-1]
        _ensure_terminator(anchor, doc)
        insert_after(anchor, header, doc)
        owner.entry_slots.append(header)
        section._owner_aot_entry = owner  # noqa: SLF001
    else:
        _splice_at_end(header, doc)

    # File the binding ref under the deepest implicit parent.
    deepest_parent = chain[-1]
    parent_ref = SlotRef(slot=header, container=deepest_parent)
    _file_ref_at_tail(deepest_parent, parent_ref)
    dict.__setitem__(deepest_parent, sub[-1], section)

    # Also propagate ancestor-prefix bindings so the implicit ancestors
    # have an _index entry with this header as a contributor.
    for j in range(len(sub) - 1):
        anc = chain[j]
        comp = sub[j]
        anc_ref = SlotRef(slot=header, container=anc)
        _file_ref_at_tail(anc, anc_ref)

    _maybe_demote_synthetic_empty_header(parent)

    for k, v in pending:
        if not (is_scalar(v) or _is_synth_inline(v)):
            section[k] = v
            continue
        cst, dec = _synth_value(
            v,
            layout_root=doc,
            parent=section,
            path=(*full_path, k),
            owner=owner,
        )
        append_direct_kv(section, k, cst)
        dict.__setitem__(section, k, dec)
    return section


def attach_section(
    parent: Container, key: str, source: Mapping[str, Any] | Container | None = None
) -> Table:
    """Synthesise ``[parent_path.key]`` at end-of-doc and attach.

    ``source`` may be ``None`` (empty section) or a Mapping (initial body).
    Returns the live `Table` view.
    """
    from tomlrt._container import (  # noqa: PLC0415
        Table,
        _is_synth_inline,
        _synth_value,
    )

    layout_root = parent._layout_root  # noqa: SLF001
    if layout_root is None:  # pragma: no cover
        msg = "internal: parent has no layout root"
        raise AssertionError(msg)
    doc = layout_root
    new_path = (*parent._path, key)  # noqa: SLF001

    leading = _build_section_leading(doc)
    owner = parent._owner_aot_entry  # noqa: SLF001
    header = _new_section_header(
        new_path,
        leading=leading,
        doc=doc,
        kind="table",
        owner_aot_entry=owner,
    )

    # Rehome the source if it is an unattached Table; otherwise build new.
    if isinstance(source, Table) and source._layout_root is None:  # noqa: SLF001
        section = source
        pending: list[tuple[str, object]] = list(source.items())
        dict.clear(section)
    else:
        section = Table()
        pending = list(_items_for_synth(source)) if source is not None else []

    _wire_section_container(
        section,
        doc=doc,
        path=new_path,
        parent=parent,
        owner=None,
        header=header,
    )

    if owner is not None and owner.entry_slots:
        # AoT-entry-aware attach: splice immediately after the entry's
        # current tail slot, and own the new header so a later delete
        # of the entry takes the promoted section with it.
        anchor = owner.entry_slots[-1]
        _ensure_terminator(anchor, doc)
        insert_after(anchor, header, doc)
        owner.entry_slots.append(header)
        section._owner_aot_entry = owner  # noqa: SLF001
    else:
        _splice_block_at_parent_anchor([header], parent, doc)

    parent_ref = SlotRef(slot=header, container=parent)
    _file_ref_at_tail(parent, parent_ref)
    dict.__setitem__(parent, key, section)

    _maybe_demote_synthetic_empty_header(parent)

    # Process scalars (and synth-inlines) before nested structural
    # children. TOML semantics require all direct KVs of a section to
    # appear before any sub-section header — re-opening a section
    # after a child header is illegal. Re-ordering here also avoids
    # a subtle bug: the recursive ``section[k] = v`` path may demote
    # ``section``'s synthetic empty header on its first sub-section
    # attach, leaving subsequent scalar siblings with no header to
    # bind to and triggering ``_synthesise_header_then_insert_kv``,
    # whose ancestor-binding walk does not maintain ``parent_ref``
    # entries on the grand-ancestor chain that attach_section would
    # have skipped. Process all scalars first so the section's KV
    # body is fully populated (and the header is no longer empty)
    # before any sub-section attach can demote it.
    scalars: list[tuple[str, object]] = []
    structurals: list[tuple[str, object]] = []
    for k, v in pending:
        if is_scalar(v) or _is_synth_inline(v):
            scalars.append((k, v))
        else:
            structurals.append((k, v))
    for k, v in scalars:
        cst, dec = _synth_value(
            v,
            layout_root=doc,
            parent=section,
            path=(*new_path, k),
            owner=owner,
        )
        append_direct_kv(section, k, cst)
        dict.__setitem__(section, k, dec)
    for k, v in structurals:
        section[k] = v
    return section


def _items_for_synth(source: Mapping[str, Any] | Container) -> list[tuple[str, object]]:
    """Iterate items of a Mapping/dict/Container source as (key, value)."""
    return list(source.items())


def _last_aot_slot(aot: AoT) -> Slot | None:
    """Return the last doc-stream slot owned by any entry of ``aot``.

    AoT entries are stored in document order and each entry's
    `entry_slots` list is also in document order, so the answer is
    the last slot of the last entry that has any slots. Walks
    backwards to keep this O(1) in the common case.
    """
    for entry_table in reversed(aot):
        e = entry_table._owner_aot_entry  # noqa: SLF001
        if e is None or not e.entry_slots:
            continue
        return e.entry_slots[-1]
    return None


def _pop_or_remove(lst: list[Any], item: Any) -> None:
    """O(1) pop if ``item`` is at the tail; else C-level ``list.remove``.

    Both branches are C-implemented; the tail check avoids an
    O(N) scan when the caller is consuming a list in reverse
    (the common case for batched scrubs).
    """
    if lst[-1] is item:
        lst.pop()
    else:
        lst.remove(item)


def unfile_ref(ref: SlotRef) -> None:
    """Remove ``ref`` from its container's ``_refs``/``_index`` and from ``slot._refs``.

    Each affected list uses the tail-fast-path via
    `_pop_or_remove`. Also clears ``container._header_ref`` if the
    ref was the container's own-header ref.
    """
    c = ref.container
    if not c._inline:  # noqa: SLF001
        _pop_or_remove(c._refs, ref)  # noqa: SLF001
        local_key = ref.local_key
        if local_key is None:
            if c._header_ref is ref:  # noqa: SLF001
                c._header_ref = None  # noqa: SLF001
        else:
            bucket = c._index[local_key]  # noqa: SLF001
            _pop_or_remove(bucket, ref)
            if not bucket:
                del c._index[local_key]  # noqa: SLF001
    _pop_or_remove(ref.slot._refs, ref)  # noqa: SLF001


def _scrub_owned_slots_via_backptrs(
    owned: Iterable[Slot],
    *,
    skip_container_ids: frozenset[int] = frozenset(),
) -> None:
    """Remove every live ref to each slot in ``owned`` via slot back-pointers.

    Walks ``slot._refs`` directly (length ≤ path depth, bounded
    independent of doc size) instead of scanning ancestor containers'
    ``_index``/``_refs`` lists.

    ``skip_container_ids`` names containers whose internal refs to
    owned slots should be left in place — the typical caller is
    `delete_key`, which transplants the deleted subtree to a fresh
    orphan doc and needs the subtree containers' internal
    structure intact. The default (empty) is correct for AoT removal,
    which discards the popped entries' containers entirely.
    """
    for s in owned:
        # Snapshot — unfile_ref mutates slot._refs.
        for ref in list(s._refs):  # noqa: SLF001
            if id(ref.container) in skip_container_ids:
                continue
            unfile_ref(ref)


def remove_aot_entry(aot: AoT, index: int) -> Table:
    """Remove ``aot[index]``, unlink its slots, and return a snapshot.

    The snapshot is a fresh unattached `Table` populated from the
    removed entry's dict storage (deep-copied for plain values; nested
    live typed containers are not detached).
    """
    n = len(aot)
    if not -n <= index < n:
        msg = f"AoT index {index} out of range (len {n})"
        raise IndexError(msg)
    if index < 0:
        index += n
    return remove_aot_entries(aot, [index])[0]


def remove_aot_entries(aot: AoT, indices: Iterable[int]) -> list[Table]:
    """Remove ``aot[i]`` for each ``i`` in ``indices`` in one batch.

    The indices must already be **non-negative, in-range, distinct,
    and ascending**; callers are responsible for normalising. Returns
    snapshots in the same order as ``indices``.

    Batching matters because the per-pop ref-scrub is O(parent
    siblings); doing it once for the union of all popped entries'
    slots makes ``AoT.clear`` and slice-delete linear instead of
    quadratic.
    """
    from tomlrt._array import AoT  # noqa: PLC0415
    from tomlrt._container import (  # noqa: PLC0415
        Table,
        _is_section,
        _reset_table_for_rehome,
    )

    idx_list = list(indices)
    if not idx_list:
        return []
    layout_root = aot._layout_root  # noqa: SLF001
    parent = aot._parent  # noqa: SLF001
    assert layout_root is not None
    assert parent is not None
    doc = layout_root

    # Per-entry: collect owned slots (entry + nested AoT entry slots)
    # and a snapshot of the entry's dict storage.
    owned_per_entry: list[list[Slot]] = []
    snapshots: list[Table] = []
    union_owned: set[Slot] = set()
    union_owned_ordered: list[Slot] = []  # in doc-stream order

    def _collect_nested_aot_slots(c: Container, sink: list[Slot]) -> None:
        for v in c.values():
            if isinstance(v, AoT):
                for nested_entry_table in v:
                    ne = nested_entry_table._owner_aot_entry  # noqa: SLF001
                    if ne is not None:
                        sink.extend(ne.entry_slots)
                    _collect_nested_aot_slots(nested_entry_table, sink)
            elif _is_section(v):
                _collect_nested_aot_slots(v, sink)

    for i in idx_list:
        entry_table = aot[i]
        e = entry_table._owner_aot_entry  # noqa: SLF001
        assert e is not None
        owned_ordered: list[Slot] = list(e.entry_slots)
        _collect_nested_aot_slots(entry_table, owned_ordered)
        # Dedupe while preserving order, in case nested collection
        # produces overlap with entry_slots.
        seen: set[int] = set()
        deduped: list[Slot] = []
        for s in owned_ordered:
            if id(s) in seen:
                continue
            seen.add(id(s))
            deduped.append(s)
            if s not in union_owned:
                union_owned.add(s)
                union_owned_ordered.append(s)
        owned_per_entry.append(deduped)

        snapshot = Table()
        for k, v in entry_table.items():
            dict.__setitem__(snapshot, k, v)
        snapshots.append(snapshot)

    # Slot-driven scrub via back-pointers, in REVERSE doc-stream
    # order so each unfile_ref hits the tail-fast-path of every
    # affected `_refs` / `_index[k]` list. This is what makes the
    # batched case (clear / slice-delete) linear: a parent bucket
    # of N AoT-entry binding refs is emptied tail-pop by tail-pop
    # at C-speed O(1) each, rather than middle-of-bucket
    # O(N) C-removes.
    _scrub_owned_slots_via_backptrs(reversed(union_owned_ordered))

    # Body-tail invalidation: clear any `_body_tail` on the parent
    # chain that points into the popped slot set. Stale tails are
    # otherwise recomputed lazily, but check explicitly so that
    # callers iterating right after a pop get a correct anchor.
    cur: Container | None = parent
    while cur is not None:
        if cur._body_tail is not None and cur._body_tail in union_owned:  # noqa: SLF001
            cur._body_tail = None  # noqa: SLF001
        cur = cur._parent  # noqa: SLF001

    for owned in owned_per_entry:
        # Unlink in reverse order so the entry's leftmost slot (the
        # ``[[a]]`` header) goes last — see remove_aot_entry's
        # original comment for the trivia-promotion hazard.
        for slot in reversed(owned):
            unlink_slot(slot, doc)

    # Drop entries from the logical list in reverse so earlier
    # indices stay valid as we go.
    for i in reversed(idx_list):
        list.pop(aot, i)

    for snapshot in snapshots:
        snapshot._layout_root = doc  # noqa: SLF001
        _reset_table_for_rehome(snapshot, recurse=True)

    last_key = aot._path[-1]  # noqa: SLF001
    if len(aot) == 0 and not parent._index.get(last_key):  # noqa: SLF001
        parent._index.pop(last_key, None)  # noqa: SLF001

    return snapshots


def replace_aot_entry_with_clone(
    aot: AoT,
    index: int,
    src_entry_table: Container,
) -> None:
    """Replace ``aot[index]`` with a deep clone of ``src_entry_table``.

    Preserves the *destination* entry header's leading trivia (and
    therefore any pre-header comment block above the original
    ``[[..]]`` line) while replacing the entry's body with a clone
    of the source entry's slots — preserving the source's per-KV
    leading / EOL / lexeme trivia.

    Both entries must be attached AoT-entry tables.
    """
    n = len(aot)
    if not -n <= index < n:
        msg = f"AoT index {index} out of range (len {n})"
        raise IndexError(msg)
    if index < 0:
        index += n

    layout_root = aot._layout_root  # noqa: SLF001
    path = aot._path  # noqa: SLF001
    if layout_root is None or not path:
        msg = "replace_aot_entry_with_clone requires the AoT to be attached"
        raise RuntimeError(msg)
    doc = layout_root

    dst_entry_table = aot[index]
    if dst_entry_table is src_entry_table:
        return

    dst_entry = dst_entry_table._owner_aot_entry  # noqa: SLF001
    src_entry = src_entry_table._owner_aot_entry  # noqa: SLF001
    if dst_entry is None or src_entry is None:
        msg = "replace_aot_entry_with_clone needs AoT-entry tables on both sides"
        raise RuntimeError(msg)

    src_slots = _validate_clonable_aot_entry(src_entry)
    src_prefix = src_entry.path

    # Save the destination header (we keep it in place, only its body
    # changes). The header's leading carries any pre-header comment
    # block — that's the trivia the test pins.
    dst_header = dst_entry.entry_slots[0]
    assert isinstance(dst_header, StructuralHeaderSlot)

    # Pre-clone source body before any destructive cleanup, so a clone
    # failure can't leave the destination half-emptied. Also covers
    # the source-inside-destination case (e.g. self-nested clone).
    cloned_body = (
        _clone_entry_slots(
            src_slots[1:],
            new_entry=dst_entry,
            body_owner=dst_entry,
            src_prefix=src_prefix,
            target_prefix=path,
            head_kind="table",  # unused (no header in this list)
            has_header=False,
        )
        if len(src_slots) > 1
        else []
    )
    # _clone_entry_slots appended the cloned slots to dst_entry's
    # entry_slots prematurely; we splice them in below, so back them
    # out to keep ownership state consistent during clear().
    if cloned_body:
        dst_entry.entry_slots = dst_entry.entry_slots[: -len(cloned_body)]

    # Reuse the structural-delete path to tear down the destination's
    # body: orphans held sub-sections / AoTs into a PrivateRoot,
    # unlinks all body slots from the doc, recomputes tails, and
    # cleans up nested AoTEntry membership. The destination header
    # stays in place because it is not a dict-storage entry.
    dst_entry_table.clear()
    # After clear(), dst_entry.entry_slots should be [dst_header].
    assert dst_entry.entry_slots == [dst_header]

    # Splice cloned body slots into the doc immediately after the
    # destination header.
    prev: Slot = dst_header
    for s in cloned_body:
        _ensure_terminator(prev, doc)
        insert_after(prev, s, doc)
        prev = s
        dst_entry.entry_slots.append(s)

    # Rebuild views / dict storage from the cloned body.
    _populate_entry_views(
        entry_table=dst_entry_table,
        cloned_slots=cloned_body,
        target_prefix=path,
        body_owner=dst_entry,
        doc=doc,
    )


def replace_aot_entry(aot: AoT, index: int, body: Mapping[str, Any] | None) -> None:
    """Replace ``aot[index]`` in place.

    Keeps the entry's header slot and live `Table` view; just clears
    the body and re-populates from ``body``.

    O(m) in the size of ``body``, independent of AoT length and
    document size. Header position and `_refs` ordering are preserved
    by construction (no slot splicing involved).
    """
    n = len(aot)
    if not -n <= index < n:
        msg = f"AoT index {index} out of range (len {n})"
        raise IndexError(msg)
    if index < 0:
        index += n
    entry_table = aot[index]
    if body is entry_table:
        return
    items = list(body.items()) if body is not None else []
    entry_table.clear()
    for k, v in items:
        entry_table[k] = v


def renormalise_aot_order(aot: AoT, new_logical_order: Sequence[Table]) -> None:
    """Re-order an attached AoT's entries to ``new_logical_order``.

    Implements the locked-in "normalise on reorder" policy from the
    plan: snapshot a stable splice anchor (the slot just before the
    AoT's first owned slot in doc-stream); unlink every slot owned
    by any of this AoT's entries; reinsert the entries in the new
    order, each entry as a contiguous block, immediately after the
    anchor.

    ``new_logical_order`` must be a permutation of the AoT's current
    entries (same set of `Table` objects, possibly reordered).
    """
    if len(aot) <= 1:
        # Reverse / sort on 0 or 1 elements is a no-op.
        list.clear(aot)
        for t in new_logical_order:
            list.append(aot, t)
        return
    layout_root = aot._layout_root  # noqa: SLF001
    assert layout_root is not None
    doc = layout_root

    # Collect every entry's owned slots, in current logical order.
    per_entry_slots: list[list[Slot]] = []
    for entry_table in list(aot):
        e = entry_table._owner_aot_entry  # noqa: SLF001
        assert e is not None
        per_entry_slots.append(list(e.entry_slots))

    # Snapshot the structural-separator part of each entry header's
    # leading, by position. The structural separator (leading blank
    # lines / whitespace before any comment) belongs to the position
    # in the doc; the remainder (comment block + interior whitespace)
    # belongs to the entry and travels with it on reorder.
    structural_by_position: list[Trivia] = []
    remainder_by_entry_id: dict[int, Trivia] = {}
    for i, entry_table in enumerate(list(aot)):
        if not per_entry_slots[i]:
            structural_by_position.append(Trivia())
            remainder_by_entry_id[id(entry_table)] = Trivia()
            continue
        head_slot = per_entry_slots[i][0]
        structural, remainder = _split_leading_structural(head_slot.leading)
        structural_by_position.append(structural)
        remainder_by_entry_id[id(entry_table)] = remainder

    # Earliest owned slot in doc-stream gives us the splice anchor.
    # Walk the doc once, first hit wins. O(N_doc) but predictable
    # and faster than the pairwise back-walk for typical cases.
    owned_ids = {id(s) for slots in per_entry_slots for s in slots}
    earliest_slot: Slot | None = None
    cur = doc._head  # noqa: SLF001
    while cur is not None:
        if id(cur) in owned_ids:
            earliest_slot = cur
            break
        cur = cur._next  # noqa: SLF001
    assert earliest_slot is not None
    anchor_prev = earliest_slot._prev  # noqa: SLF001

    # Unlink every owned slot from the doc-stream linked list. We
    # don't touch refs / index / dict storage — only the linked-list
    # pointers — since the logical mapping doesn't change.
    for slots in per_entry_slots:
        for s in slots:
            unlink_slot(s, doc, strip_new_head_leading=False)

    # Build a per-entry-Table -> entry_slots map (the user-facing
    # `Table`s in new_logical_order may be re-arrangements of the
    # current ones; we need to re-attach via owner_aot_entry).
    slot_blocks: dict[int, list[Slot]] = {
        id(t): per_entry_slots[i] for i, t in enumerate(list(aot))
    }

    # Re-insert entries in new order, each as a contiguous block
    # after `anchor_prev` (or at doc head if anchor_prev is None).
    insert_after_slot = anchor_prev
    for entry_table in new_logical_order:
        block = slot_blocks[id(entry_table)]
        for slot in block:
            if insert_after_slot is None:
                insert_before_head(slot, doc)
            else:
                insert_after(insert_after_slot, slot, doc)
            insert_after_slot = slot

    # Re-apply the structural-separator portion of each new-position
    # entry's header leading from the snapshot (position-keyed),
    # stitched onto that entry's own comment-remainder (entry-keyed).
    for new_pos, entry_table in enumerate(new_logical_order):
        block = slot_blocks[id(entry_table)]
        if not block:
            continue
        head_slot = block[0]
        structural = structural_by_position[new_pos]
        remainder = remainder_by_entry_id[id(entry_table)]
        head_slot.leading = Trivia(list(structural.pieces) + list(remainder.pieces))

    # Reflect the new order in the AoT's own list view.
    list.clear(aot)
    for t in new_logical_order:
        list.append(aot, t)

    # Resort _refs lists on every container in the AoT's parent chain.
    # Each ancestor holds entry-header refs (one per entry, filed under
    # the relevant path component); after splicing, those refs are out
    # of doc-stream order. Sort by slot's new doc-stream position.
    chain: list[Container] = []
    anc: Container | None = aot._parent  # noqa: SLF001
    while anc is not None:
        chain.append(anc)
        anc = anc._parent  # noqa: SLF001
    _resort_refs_by_doc_order(chain, doc)


def _resort_refs_by_doc_order(containers: list[Container], doc: Document) -> None:
    """Resort each container's ``_refs`` and ``_index[k]`` by linked-list position."""
    position: dict[int, int] = {}
    cur = doc._head  # noqa: SLF001
    idx = 0
    while cur is not None:
        position[id(cur)] = idx
        idx += 1
        cur = cur._next  # noqa: SLF001
    for c in containers:
        c._refs.sort(key=lambda r: position.get(id(r.slot), 0))  # noqa: SLF001
        for refs in c._index.values():  # noqa: SLF001
            refs.sort(key=lambda r: position.get(id(r.slot), 0))


def _gather_value_owned_slots(val: object) -> list[Slot]:
    """Return all physical slots logically owned by ``val`` in doc-stream order.

    For a section Table: header + body + nested sub-section slots.
    For an AoT: every entry's ``entry_slots``, in entry order.
    For anything else (scalar, inline, empty AoT): the empty list.
    """
    from tomlrt._array import AoT  # noqa: PLC0415
    from tomlrt._container import Container  # noqa: PLC0415

    if isinstance(val, AoT):
        out: list[Slot] = []
        for et in val:
            e = et._owner_aot_entry  # noqa: SLF001
            if e is not None:
                out.extend(e.entry_slots)
        return out
    if (
        isinstance(val, Container)
        and not val._inline  # noqa: SLF001
        and val._header_ref is not None  # noqa: SLF001
    ):
        return list(_gather_section_slots(val))
    return []


def move_slots_to_anchor(
    parent: Container,
    key: str,
    saved_anchor_prev: Slot | None,
    saved_leading_pieces: list[TriviaPiece],
) -> None:
    """Move ``parent[key]``'s owned slots to ``saved_anchor_prev``.

    Splices the slot block immediately after ``saved_anchor_prev`` (or
    to doc head if None), applies ``saved_leading_pieces`` to the new
    head, and resorts the affected ancestor ``_refs``. Used by the
    ``Container.__setitem__`` position-preserving structural replace
    path: capture old position + leading before the replacement is
    installed at end-of-doc, then move it back.
    """
    doc = parent._layout_root  # noqa: SLF001
    if doc is None:
        return
    val = dict.__getitem__(parent, key)
    slots = _gather_value_owned_slots(val)
    if not slots:
        # Empty AoT or other slotless binding — nothing to move.
        return
    head = slots[0]
    tail = slots[-1]

    if head._prev is saved_anchor_prev:  # noqa: SLF001
        # Already at the saved position — only the leading needs fixing.
        head.leading.pieces = list(saved_leading_pieces)
        return

    # Detach [head .. tail] from its current position in the linked list.
    p = head._prev  # noqa: SLF001
    n = tail._next  # noqa: SLF001
    if p is not None:
        p._next = n  # noqa: SLF001
    else:
        doc._head = n  # noqa: SLF001
    if n is not None:
        n._prev = p  # noqa: SLF001
    else:
        doc._tail = p  # noqa: SLF001

    # Splice [head .. tail] in after saved_anchor_prev (or at doc head).
    if saved_anchor_prev is None:
        next_after = doc._head  # noqa: SLF001
        head._prev = None  # noqa: SLF001
        tail._next = next_after  # noqa: SLF001
        if next_after is not None:
            next_after._prev = tail  # noqa: SLF001
        else:
            doc._tail = tail  # noqa: SLF001
        doc._head = head  # noqa: SLF001
    else:
        next_after = saved_anchor_prev._next  # noqa: SLF001
        head._prev = saved_anchor_prev  # noqa: SLF001
        saved_anchor_prev._next = head  # noqa: SLF001
        tail._next = next_after  # noqa: SLF001
        if next_after is not None:
            next_after._prev = tail  # noqa: SLF001
        else:
            doc._tail = tail  # noqa: SLF001

    head.leading.pieces = list(saved_leading_pieces)

    # Resort ancestor refs by linked-list position; also recompute
    # _body_tail on each (the move may have invalidated the cached
    # tail when the moved slot block was the staging-tail of any
    # ancestor body).
    chain: list[Container] = []
    anc: Container | None = parent
    while anc is not None:
        chain.append(anc)
        anc = anc._parent  # noqa: SLF001
    _resort_refs_by_doc_order(chain, doc)
    for c in chain:
        if c._body_tail is not None:  # noqa: SLF001
            c._body_tail = _recompute_body_tail(c)  # noqa: SLF001


__all__ = [
    "add_aot_entry",
    "append_direct_kv",
    "attach_empty_aot",
    "attach_section",
    "attach_section_at",
    "delete_key",
    "insert_after",
    "insert_before",
    "insert_before_head",
    "move_slots_to_anchor",
    "remove_aot_entry",
    "renormalise_aot_order",
    "replace_aot_entry",
    "unlink_slot",
]
