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
from typing import TYPE_CHECKING, Any

from tomlrt._slots import AoTEntry, KVSlot, SlotRef, StructuralHeaderSlot
from tomlrt._trivia import CommentNode, EolTrivia, NewlineNode, Trivia
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
    storage. Phase 3c onward.

    Routing:

    * implicit-headerless non-root container with a body anchor →
      Phase 3d-4 dotted-KV synthesis under the nearest header-bearing
      ancestor;
    * AoT-entry body insert (header-bearing ``c`` with
      ``_owner_aot_entry is not None``) → still deferred (Phase 4);
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
            # ``doc.table('a')['x']``). Insert as a dotted KV under
            # the nearest header-bearing ancestor, positioned
            # immediately *before* ``c``'s first descendant slot
            # in doc-stream order (so the new top-level / scope-
            # local key lands inside that ancestor's region rather
            # than after the descendant header).
            _insert_dotted_kv_before_descendants(c, key, value)
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
        # Section-only doc: insert the new KV at the head of the doc
        # stream, before the first existing slot. Ensure a blank-line
        # separator on what is about to become the second slot, so the
        # new KV does not visually collide with `[s]`.
        old_head = doc._head  # noqa: SLF001
        insert_before_head(new_slot, doc)
        _ensure_leading_blank_line(old_head, doc)
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


def _make_keypart(name: str) -> KeyPart:
    """Build a `KeyPart` for an inserted key, choosing bare vs basic-quoted."""
    if _RE_BARE_KEY.match(name):
        return KeyPart(raw=name, value=name, kind="bare")
    return KeyPart(raw=_quote_basic(name), value=name, kind="basic")


def _append_dotted_kv_under_implicit(c: Container, key: str, value: Value) -> None:
    """3d-4: insert into an implicit-headerless container via dotted KV.

    Routes through the nearest header-bearing ancestor (or the doc
    root). Files refs on every implicit ancestor between that host
    and ``c`` per the dotted-KV ref-propagation rule (plan v17).

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
    parts = [_make_keypart(k) for k in keypath]
    seps = ["."] * (len(parts) - 1)
    new_slot = KVSlot(
        leading=Trivia(),
        host_path=host._path,  # noqa: SLF001
        key_parts=parts,
        key_seps=seps,
        pre_eq=" ",
        post_eq=" ",
        value=value,
        eol=EolTrivia(
            trailing_ws=None,
            comment=None,
            newline=NewlineNode(text=doc._newline),  # noqa: SLF001
        ),
        owner_aot_entry=owner,
    )

    insert_after(body_tail, new_slot, doc)

    # File refs on every chain ancestor. ``_refs`` is the doc-stream
    # subset; ``_index`` preserves "primary at index 0 + all
    # contributors". Appending to ``_index`` keeps any existing
    # structural primary (e.g. a header-owning ref on the host) at
    # index 0; a new dotted contributor is always secondary.
    for i, anc in enumerate(chain):
        new_ref = SlotRef(slot=new_slot, container=anc, local_key=local_keys[i])
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
        anc._index[local_keys[i]] = [  # noqa: SLF001
            r
            for r in anc._refs  # noqa: SLF001
            if r.local_key == local_keys[i]
        ]
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


def _insert_dotted_kv_before_descendants(c: Container, key: str, value: Value) -> None:
    """Insert a dotted KV into a structural-only implicit container.

    Targets implicit-headerless ``c`` with only descendant-header
    contributors (no body anchor). Locates ``c``'s first
    contributor in doc-stream order (the topmost descendant header
    binding ref), and splices a new dotted KV ``host_path.key``
    immediately *before* that header, where ``host_path`` is the
    nearest header-bearing ancestor (or the doc root for top-level
    implicits).

    Pre-conditions (checked by caller):

    * ``c._path`` is non-empty
    * ``c._header_ref is None``
    * ``c._body_tail is None``
    * ``c._refs`` is non-empty (there is at least one descendant
      binding ref to use as the "before" anchor)
    """
    layout_root = c._layout_root  # noqa: SLF001
    assert layout_root is not None
    doc = layout_root

    if not c._refs:  # noqa: SLF001
        msg = (
            "internal: implicit container has no descendant refs to anchor "
            "structural insert"
        )
        raise AssertionError(msg)
    anchor_slot = c._refs[0].slot  # noqa: SLF001

    # Find host: nearest ancestor with a header, or the doc root.
    host: Container = c
    while host._parent is not None and host._header_ref is None:  # noqa: SLF001
        host = host._parent  # noqa: SLF001

    chain: list[Container] = []
    cur: Container | None = c
    while cur is not host:
        assert cur is not None
        chain.append(cur)
        cur = cur._parent  # noqa: SLF001
    chain.append(host)
    chain.reverse()

    local_keys = [*c._path[len(host._path) :], key]  # noqa: SLF001
    assert len(local_keys) == len(chain)

    owner = c._owner_aot_entry  # noqa: SLF001
    for anc in chain:
        assert anc._owner_aot_entry is owner  # noqa: SLF001

    keypath = (*c._path[len(host._path) :], key)  # noqa: SLF001
    parts = [_make_keypart(k) for k in keypath]
    seps = ["."] * (len(parts) - 1)
    new_slot = KVSlot(
        leading=Trivia(),
        host_path=host._path,  # noqa: SLF001
        key_parts=parts,
        key_seps=seps,
        pre_eq=" ",
        post_eq=" ",
        value=value,
        eol=EolTrivia(
            trailing_ws=None,
            comment=None,
            newline=NewlineNode(text=doc._newline),  # noqa: SLF001
        ),
        owner_aot_entry=owner,
    )

    insert_before(anchor_slot, new_slot, doc)

    # File refs on every chain ancestor at the position just before
    # the anchor's ref.
    for i, anc in enumerate(chain):
        new_ref = SlotRef(slot=new_slot, container=anc, local_key=local_keys[i])
        anchor_idx = _find_ref_index_by_slot(anc, anchor_slot)
        anc._refs.insert(anchor_idx, new_ref)  # noqa: SLF001
        anc._index[local_keys[i]] = [  # noqa: SLF001
            r
            for r in anc._refs  # noqa: SLF001
            if r.local_key == local_keys[i]
        ]
        # Update _body_tail on the immediate target container only:
        # the new slot is now the deepest container's only body
        # contributor.
        if anc is c:
            anc._body_tail = new_slot  # noqa: SLF001

    if owner is not None:
        try:
            anchor_idx = owner.entry_slots.index(anchor_slot)
        except ValueError:
            owner.entry_slots.append(new_slot)
        else:
            owner.entry_slots.insert(anchor_idx, new_slot)


def _append_kv_in_aot_entry(c: Container, key: str, value: Value) -> None:
    """Append a direct KV in an AoT-entry root container's body.

    Mirrors the header-bearing path in `append_direct_kv` but also
    keeps the entry's `entry_slots` list in doc-stream order.
    """
    layout_root = c._layout_root  # noqa: SLF001
    if layout_root is None:
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

    new_ref = SlotRef(slot=new_slot, container=c, local_key=key)
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


# ---------------------------------------------------------------------------
# Structural attach (Phase 4 — section / AoT synthesis)
# ---------------------------------------------------------------------------


def _build_header_keyparts(path: tuple[str, ...]) -> list[KeyPart]:
    out: list[KeyPart] = []
    for p in path:
        if _RE_BARE_KEY.match(p):
            out.append(KeyPart(raw=p, value=p, kind="bare"))
        else:
            out.append(KeyPart(raw=_quote_basic(p), value=p, kind="basic"))
    return out


def _new_section_header(
    path: tuple[str, ...],
    *,
    leading: Trivia,
    doc: Document,
    kind: str = "table",
    entry: AoTEntry | None = None,
    owner_aot_entry: AoTEntry | None = None,
) -> StructuralHeaderSlot:
    return StructuralHeaderSlot(
        leading=leading,
        kind=kind,  # type: ignore[arg-type]
        path=path,
        key_parts=_build_header_keyparts(path),
        key_seps=["."] * (len(path) - 1),
        eol=EolTrivia(
            trailing_ws=None,
            comment=None,
            newline=NewlineNode(text=doc._newline),  # noqa: SLF001
        ),
        entry=entry,
        owner_aot_entry=owner_aot_entry,
        synthetic=True,
    )


def _doc_tail_anchor(doc: Document) -> Slot | None:
    """Return the slot to insert *after* when appending at end-of-doc."""
    return doc._tail  # noqa: SLF001


def _splice_at_end(slot: Slot, doc: Document) -> None:
    """Insert ``slot`` at the end of the doc-stream."""
    anchor = _doc_tail_anchor(doc)
    if anchor is None:
        # Empty doc.
        insert_before_head(slot, doc)
    else:
        _ensure_terminator(anchor, doc)
        insert_after(anchor, slot, doc)


def _build_section_leading(doc: Document) -> Trivia:
    """Trivia for a fresh section header.

    Empty doc → no leading; non-empty → one blank line of separation.
    """
    if doc._head is None:  # noqa: SLF001
        return Trivia()
    return Trivia([NewlineNode(text=doc._newline)])  # noqa: SLF001


def attach_empty_aot(parent: Container, key: str, source_aot: object) -> object:
    """Bind an empty AoT under ``parent[key]``.

    No physical slots are created; subsequent ``aot.add(...)`` calls
    will materialise the first ``[[path]]`` header. The ``source_aot``
    is rehomed in place (identity preserved).
    """
    from tomlrt._array import AoT  # noqa: PLC0415
    from tomlrt._container import Container as ContainerType  # noqa: PLC0415

    assert isinstance(source_aot, AoT)
    assert isinstance(parent, ContainerType)
    if len(source_aot) > 0:
        msg = "non-empty AoT live-attach has its own routing"
        raise AssertionError(msg)
    # Rehome the orphan AoT into this parent's logical scope.
    source_aot._layout_root = parent._layout_root  # noqa: SLF001
    source_aot._path = (*parent._path, key)  # noqa: SLF001
    source_aot._parent = parent  # noqa: SLF001
    return source_aot


def add_aot_entry(aot: object, body: object) -> object:
    """Append a ``[[path]]`` entry to ``aot`` and return its `Table` view."""
    from tomlrt._array import AoT  # noqa: PLC0415
    from tomlrt._container import (  # noqa: PLC0415
        Table,
        _is_scalar,
        _is_synth_inline,
        _synth_value,
    )

    assert isinstance(aot, AoT)
    parent = aot._parent  # noqa: SLF001
    layout_root = aot._layout_root  # noqa: SLF001
    path = aot._path  # noqa: SLF001
    if layout_root is None or parent is None or not path:
        msg = "AoT.add requires the AoT to be attached to a document"
        raise RuntimeError(msg)
    doc = layout_root

    ordinal = len(aot)
    entry = AoTEntry(path=path, ordinal=ordinal)
    leading = (
        _build_section_leading(doc)
        if ordinal == 0
        else Trivia([NewlineNode(text=doc._newline)])  # noqa: SLF001
    )
    header = _new_section_header(
        path,
        leading=leading,
        doc=doc,
        kind="aot-entry",
        entry=entry,
        owner_aot_entry=entry,
    )
    entry.entry_slots.append(header)

    # Build entry-root container.
    entry_table = Table()
    entry_table._layout_root = doc  # noqa: SLF001
    entry_table._path = path  # noqa: SLF001
    entry_table._parent = parent  # noqa: SLF001
    entry_table._owner_aot_entry = entry  # noqa: SLF001
    entry_ref = SlotRef(slot=header, container=entry_table, local_key=None)
    entry_table._refs.append(entry_ref)  # noqa: SLF001
    entry_table._header_ref = entry_ref  # noqa: SLF001

    # Splice header after the last existing AoT-owned slot if any,
    # else at end-of-doc.
    anchor = _last_aot_slot(aot)
    if anchor is None:
        _splice_at_end(header, doc)
    else:
        _ensure_terminator(anchor, doc)
        insert_after(anchor, header, doc)

    # File parent-chain refs for this entry's header.
    parent_ref = SlotRef(slot=header, container=parent, local_key=path[-1])
    parent._refs.append(parent_ref)  # noqa: SLF001
    parent._index.setdefault(path[-1], []).append(parent_ref)  # noqa: SLF001

    # Append the new entry to the AoT view list.
    list.append(aot, entry_table)

    # Populate body.
    if body is not None:
        for k, v in _items_for_synth(body):
            if not isinstance(k, str):
                msg = f"AoT entry key must be str, got {type(k).__name__}"
                raise TypeError(msg)
            if not (_is_scalar(v) or _is_synth_inline(v)):
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


def attach_section_at(
    parent: Container,
    sub_path: tuple[str, ...] | list[str],
    source: object | None = None,
) -> Any:
    """Synthesise ``[parent_path.sub_path]`` (multi-component) at end-of-doc.

    Intermediate components in ``sub_path[:-1]`` become implicit tables;
    the deepest component gets the explicit header. ``source`` may be
    a `Table` (rehomed) or a Mapping (snapshotted) or ``None``.
    """
    from tomlrt._container import (  # noqa: PLC0415
        Container as ContainerType,
    )
    from tomlrt._container import (  # noqa: PLC0415
        Table,
        _is_scalar,
        _is_synth_inline,
        _synth_value,
    )

    assert isinstance(parent, ContainerType)
    sub = tuple(sub_path)
    if not sub:
        msg = "sub_path must not be empty"
        raise ValueError(msg)
    if len(sub) == 1:
        return attach_section(parent, sub[0], source)

    layout_root = parent._layout_root  # noqa: SLF001
    if layout_root is None:
        msg = "internal: parent has no layout root"
        raise AssertionError(msg)
    doc = layout_root
    full_path = (*parent._path, *sub)  # noqa: SLF001

    leading = _build_section_leading(doc)
    header = _new_section_header(full_path, leading=leading, doc=doc, kind="table")

    # Build implicit chain: each intermediate is a Table view living
    # in dict storage but with no own header ref.
    chain: list[Container] = [parent]
    for j, comp in enumerate(sub[:-1]):
        cur = chain[-1]
        if comp in cur:
            nxt = dict.__getitem__(cur, comp)
            if not isinstance(nxt, ContainerType):
                msg = f"intermediate {comp!r} is not a table"
                raise TypeError(msg)
            chain.append(nxt)
            continue
        implicit = Table()
        implicit._layout_root = doc  # noqa: SLF001
        implicit._path = (*parent._path, *sub[: j + 1])  # noqa: SLF001
        implicit._parent = cur  # noqa: SLF001
        dict.__setitem__(cur, comp, implicit)
        chain.append(implicit)

    if isinstance(source, Table) and source._layout_root is None:  # noqa: SLF001
        section = source
        pending: list[tuple[Any, Any]] = list(source.items())
        dict.clear(section)
    else:
        section = Table()
        pending = list(_items_for_synth(source)) if source is not None else []

    section._layout_root = doc  # noqa: SLF001
    section._path = full_path  # noqa: SLF001
    section._parent = chain[-1]  # noqa: SLF001
    header_ref = SlotRef(slot=header, container=section, local_key=None)
    section._refs.append(header_ref)  # noqa: SLF001
    section._header_ref = header_ref  # noqa: SLF001

    _splice_at_end(header, doc)

    # File the binding ref under the deepest implicit parent.
    deepest_parent = chain[-1]
    parent_ref = SlotRef(slot=header, container=deepest_parent, local_key=sub[-1])
    deepest_parent._refs.append(parent_ref)  # noqa: SLF001
    deepest_parent._index.setdefault(sub[-1], []).append(parent_ref)  # noqa: SLF001
    dict.__setitem__(deepest_parent, sub[-1], section)

    # Also propagate ancestor-prefix bindings so the implicit ancestors
    # have an _index entry with this header as a contributor.
    for j in range(len(sub) - 1):
        anc = chain[j]
        comp = sub[j]
        anc_ref = SlotRef(slot=header, container=anc, local_key=comp)
        anc._refs.append(anc_ref)  # noqa: SLF001
        anc._index.setdefault(comp, []).append(anc_ref)  # noqa: SLF001

    for k, v in pending:
        if not (_is_scalar(v) or _is_synth_inline(v)):
            section[k] = v
            continue
        cst, dec = _synth_value(
            v,
            layout_root=doc,
            parent=section,
            path=(*full_path, k),
            owner=None,
        )
        append_direct_kv(section, k, cst)
        dict.__setitem__(section, k, dec)
    return section


def attach_section(parent: Container, key: str, source: object | None = None) -> object:
    """Synthesise ``[parent_path.key]`` at end-of-doc and attach.

    ``source`` may be ``None`` (empty section) or a Mapping (initial body).
    Returns the live `Table` view.
    """
    from tomlrt._container import (  # noqa: PLC0415
        Container as ContainerType,
    )
    from tomlrt._container import (  # noqa: PLC0415
        Table,
        _is_scalar,
        _is_synth_inline,
        _synth_value,
    )

    assert isinstance(parent, ContainerType)
    layout_root = parent._layout_root  # noqa: SLF001
    if layout_root is None:
        msg = "internal: parent has no layout root"
        raise AssertionError(msg)
    doc = layout_root
    new_path = (*parent._path, key)  # noqa: SLF001

    leading = _build_section_leading(doc)
    header = _new_section_header(new_path, leading=leading, doc=doc, kind="table")

    # Rehome the source if it is an unattached Table; otherwise build new.
    if isinstance(source, Table) and source._layout_root is None:  # noqa: SLF001
        section = source
        pending: list[tuple[Any, Any]] = list(source.items())
        dict.clear(section)
    else:
        section = Table()
        pending = list(_items_for_synth(source)) if source is not None else []

    section._layout_root = doc  # noqa: SLF001
    section._path = new_path  # noqa: SLF001
    section._parent = parent  # noqa: SLF001
    header_ref = SlotRef(slot=header, container=section, local_key=None)
    section._refs.append(header_ref)  # noqa: SLF001
    section._header_ref = header_ref  # noqa: SLF001

    _splice_at_end(header, doc)

    parent_ref = SlotRef(slot=header, container=parent, local_key=key)
    parent._refs.append(parent_ref)  # noqa: SLF001
    parent._index.setdefault(key, []).append(parent_ref)  # noqa: SLF001

    for k, v in pending:
        if not (_is_scalar(v) or _is_synth_inline(v)):
            # Nested structural — recurse via the live __setitem__
            # path now that `section` is fully attached.
            section[k] = v
            continue
        cst, dec = _synth_value(
            v,
            layout_root=doc,
            parent=section,
            path=(*new_path, k),
            owner=None,
        )
        append_direct_kv(section, k, cst)
        dict.__setitem__(section, k, dec)
    return section


def _items_for_synth(source: object) -> list[tuple[Any, Any]]:
    """Iterate items of a Mapping/dict/Container source as (key, value)."""
    from collections.abc import Mapping  # noqa: PLC0415

    if isinstance(source, Mapping):
        return list(source.items())
    msg = f"cannot iterate items from {type(source).__name__}"
    raise TypeError(msg)


def _last_aot_slot(aot: object) -> Slot | None:
    """Return the last doc-stream slot owned by any entry of ``aot``."""
    from tomlrt._array import AoT  # noqa: PLC0415

    assert isinstance(aot, AoT)
    last: Slot | None = None
    for entry_table in aot:
        e = entry_table._owner_aot_entry  # noqa: SLF001
        if e is None:
            continue
        for s in e.entry_slots:
            last = s
    return last


def _scrub_refs_to_owned_slots(c: Container, owned: set[Slot]) -> None:
    """Remove every SlotRef in ``c`` that points at a slot in ``owned``.

    Recurses into nested live `Container` / `Array` / `AoT` values held
    in dict/list storage. Inline containers are skipped (they can't
    reference doc-stream slots).
    """
    from tomlrt._array import AoT, Array  # noqa: PLC0415
    from tomlrt._container import Container as ContainerType  # noqa: PLC0415

    if c._inline:  # noqa: SLF001
        return
    new_refs = [r for r in c._refs if r.slot not in owned]  # noqa: SLF001
    if len(new_refs) != len(c._refs):  # noqa: SLF001
        c._refs[:] = new_refs  # noqa: SLF001
        # Rebuild _index from remaining refs that have a local_key.
        new_index: dict[str, list[SlotRef]] = {}
        for r in new_refs:
            if r.local_key is not None:
                new_index.setdefault(r.local_key, []).append(r)
        c._index.clear()  # noqa: SLF001
        c._index.update(new_index)  # noqa: SLF001
        # Clear header_ref if it pointed at an owned slot.
        if c._header_ref is not None and c._header_ref.slot in owned:  # noqa: SLF001
            c._header_ref = None  # noqa: SLF001
        # Clear body_tail if it pointed at an owned slot.
        if c._body_tail is not None and c._body_tail in owned:  # noqa: SLF001
            c._body_tail = None  # noqa: SLF001
    # Recurse.
    for v in list(dict.values(c)):
        if isinstance(v, ContainerType):
            _scrub_refs_to_owned_slots(v, owned)
        elif isinstance(v, AoT):
            for sub in v:
                _scrub_refs_to_owned_slots(sub, owned)
        elif isinstance(v, Array):
            # Inline arrays don't carry SlotRefs.
            pass


def remove_aot_entry(aot: object, index: int) -> object:
    """Remove ``aot[index]``, unlink its slots, and return a snapshot.

    The snapshot is a fresh unattached `Table` populated from the
    removed entry's dict storage (via deep-ish copy of plain values
    only — nested live typed containers in the entry are not yet
    detached; that lands with Phase 3e).
    """
    from tomlrt._array import AoT  # noqa: PLC0415
    from tomlrt._container import Table  # noqa: PLC0415

    assert isinstance(aot, AoT)
    n = len(aot)
    if not -n <= index < n:
        msg = f"AoT index {index} out of range (len {n})"
        raise IndexError(msg)
    if index < 0:
        index += n
    entry_table = aot[index]
    layout_root = aot._layout_root  # noqa: SLF001
    parent = aot._parent  # noqa: SLF001
    assert layout_root is not None
    assert parent is not None
    doc = layout_root
    e = entry_table._owner_aot_entry  # noqa: SLF001
    assert e is not None
    owned = set(e.entry_slots)

    # Snapshot the entry's dict-storage values to a fresh Table.
    snapshot = Table()
    for k, v in entry_table.items():
        dict.__setitem__(snapshot, k, v)

    # Scrub refs from every still-live container, walking from the doc.
    _scrub_refs_to_owned_slots(doc, owned)

    # Unlink owned slots from the doc-stream linked list.
    for slot in list(e.entry_slots):
        unlink_slot(slot, doc)

    # Pop entry from the AoT logical list.
    list.pop(aot, index)

    # If empty AoT now, also remove the parent _index[k] entry entirely
    # (dict storage of parent retains the AoT object).
    last_key = aot._path[-1]  # noqa: SLF001
    if len(aot) == 0 and not parent._index.get(last_key):  # noqa: SLF001
        parent._index.pop(last_key, None)  # noqa: SLF001

    return snapshot


def replace_aot_entry(aot: object, index: int, body: object) -> None:
    """Replace ``aot[index]`` in place with a fresh entry from ``body``."""
    from tomlrt._array import AoT  # noqa: PLC0415

    assert isinstance(aot, AoT)
    n = len(aot)
    if not -n <= index < n:
        msg = f"AoT index {index} out of range (len {n})"
        raise IndexError(msg)
    if index < 0:
        index += n
    # For now: pre-validate body (must be Mapping or None), then
    # remove + re-insert. True in-place splicing at the old position
    # is a follow-up; tests that pin doc-position fidelity will tell us.
    if body is not None and not isinstance(body, dict):
        from collections.abc import Mapping  # noqa: PLC0415

        if not isinstance(body, Mapping):
            msg = (
                f"AoT entry replacement body must be Mapping, got {type(body).__name__}"
            )
            raise TypeError(msg)
    remove_aot_entry(aot, index)
    add_aot_entry(aot, body)


__all__ = [
    "add_aot_entry",
    "append_direct_kv",
    "attach_empty_aot",
    "attach_section",
    "attach_section_at",
    "delete_key",
    "insert_after",
    "insert_before_head",
    "remove_aot_entry",
    "replace_aot_entry",
    "unlink_slot",
]
