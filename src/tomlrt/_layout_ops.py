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
from tomlrt._trivia import CommentNode, EolTrivia, NewlineNode, Trivia, WhitespaceNode
from tomlrt._values import KeyPart

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomlrt._array import AoT
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
    # Preamble migration: if the doc was previously slotless and any
    # preamble lives in `_trailing` (e.g. set via `Document.preamble`
    # on an empty doc, or a comment-only source), prepend that trivia
    # to the new head's leading and clear the trailing.
    head = doc._head  # noqa: SLF001
    if head is None and doc._trailing.pieces:  # noqa: SLF001
        nl = doc._newline  # noqa: SLF001
        migrated = list(doc._trailing.pieces)  # noqa: SLF001
        # Add a blank-line separator between preamble and content.
        from tomlrt._trivia import NewlineNode  # noqa: PLC0415

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
    from tomlrt._trivia import NewlineNode  # noqa: PLC0415

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
    else:
        # Empty doc (slotless), possibly with preamble trivia in
        # _trailing — insert_before_head migrates that onto the new
        # slot's leading.
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

    # 2. Subtree containers + AoTs + descendant refs.
    subtree_containers: list[Container] = []
    subtree_aots: list[Any] = []
    _collect_subtree(val, subtree_containers, subtree_aots, _add_ref)

    # 3. Scrub ancestor chain only — subtree containers will be
    # detached as a unit (their refs preserved against an orphan doc).
    chain: list[Container] = []
    cur: Container | None = c
    while cur is not None:
        chain.append(cur)
        cur = cur._parent  # noqa: SLF001

    for cc in chain:
        old_refs = cc._refs  # noqa: SLF001
        kept: list[SlotRef] = [r for r in old_refs if id(r.slot) not in owned_ids]
        if len(kept) != len(old_refs):
            cc._refs = kept  # noqa: SLF001
            cc._index = {}  # noqa: SLF001
            for r in kept:
                if r.local_key is not None:
                    cc._index.setdefault(r.local_key, []).append(r)  # noqa: SLF001
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

    # 5. Unlink owned slots from the doc; clean up AoTEntry.entry_slots
    # for live (still-attached) entries. Owned slots are then
    # transplanted to an orphan Document if there are subtree
    # containers / AoTs the user may still hold references to.
    surviving_aot_entries = _surviving_aot_entries(doc)
    for slot in owned_slots:
        owner = getattr(slot, "owner_aot_entry", None)
        if owner is not None and id(owner) in surviving_aot_entries:
            with contextlib.suppress(ValueError):
                owner.entry_slots.remove(slot)
        unlink_slot(slot, doc)

    if subtree_containers or subtree_aots:
        from tomlrt._container import Document as _Document  # noqa: PLC0415

        orphan = _Document()
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

    # 7. Prune empty implicit super-tables walking up the chain.
    _prune_empty_implicit_ancestors(c)


def _collect_subtree(
    val: object,
    containers_out: list[Container],
    aots_out: list[Any],
    add_ref: Callable[[SlotRef], None],
) -> None:
    """Walk ``val``'s container subtree, collecting containers, AoTs and refs."""
    from tomlrt._array import AoT  # noqa: PLC0415
    from tomlrt._container import Container  # noqa: PLC0415

    if isinstance(val, Container):
        if val._inline:  # noqa: SLF001
            return
        containers_out.append(val)
        for r in val._refs:  # noqa: SLF001
            add_ref(r)
        for child in val.values():
            _collect_subtree(child, containers_out, aots_out, add_ref)
    elif isinstance(val, AoT):
        aots_out.append(val)
        for entry in val:
            _collect_subtree(entry, containers_out, aots_out, add_ref)


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


def _collect_direct_kvs(c: Container) -> list[KVSlot]:
    same_owner = c._owner_aot_entry  # noqa: SLF001
    out: list[KVSlot] = []
    for ref in c._refs:  # noqa: SLF001
        s = ref.slot
        if (
            isinstance(s, KVSlot)
            and s.host_path == c._path  # noqa: SLF001
            and len(s.key_parts) == 1
            and s.owner_aot_entry is same_owner
        ):
            out.append(s)
    return out


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


def _aot_sibling_kvs(c: Container) -> list[KVSlot]:
    """Return the previous sibling entry's direct KVs.

    If ``c`` is an AoT entry root, return the direct KVs of the most
    recent prior sibling entry that has any (else empty).
    """
    from tomlrt._array import AoT  # noqa: PLC0415

    owner = c._owner_aot_entry  # noqa: SLF001
    if owner is None:
        return []
    parent = c._parent  # noqa: SLF001
    if parent is None:
        return []
    key = c._path[-1] if c._path else None  # noqa: SLF001
    if key is None or key not in parent:
        return []
    aot = dict.__getitem__(parent, key)
    if not isinstance(aot, AoT):
        return []
    found_self = False
    for entry_table in reversed(list(aot)):
        if entry_table is c:
            found_self = True
            continue
        if not found_self:
            continue
        sibs = _collect_direct_kvs(entry_table)
        if sibs:
            return sibs
    return []


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


def _kv_separator_leading(c: Container, doc: Document) -> Trivia:
    """Pick leading trivia for a new direct-KV slot in container ``c``.

    Inherits indentation from the most recent existing direct-KV slot
    in ``c``. Adds a leading blank line iff every prior gap between
    same-owner direct-KV slots already has one (i.e. user is uniformly
    blank-separating their KVs).

    For an AoT entry with no own KVs yet, falls back to inheriting
    indent (only) from the previous sibling entry's KVs.
    """
    kvs = _collect_direct_kvs(c)
    if not kvs:
        sibling_kvs = _aot_sibling_kvs(c)
        if sibling_kvs:
            indent_text = _extract_indent(sibling_kvs[-1].leading)
            if indent_text:
                return Trivia([WhitespaceNode(text=indent_text)])
        return Trivia()
    indent_text = _extract_indent(kvs[-1].leading)
    add_blank = False
    if len(kvs) >= 2:
        add_blank = all(_leading_has_blank_line(kv.leading) for kv in kvs[1:])
    pieces: list[Any] = []
    if add_blank:
        pieces.append(NewlineNode(text=doc._newline))  # noqa: SLF001
    if indent_text:
        pieces.append(WhitespaceNode(text=indent_text))
    return Trivia(pieces)


def _collect_host_kvs(host: Container) -> list[KVSlot]:
    """All KV slots whose ``host_path`` matches ``host._path`` (any keypath length)."""
    same_owner = host._owner_aot_entry  # noqa: SLF001
    out: list[KVSlot] = []
    for ref in host._refs:  # noqa: SLF001
        s = ref.slot
        if (
            isinstance(s, KVSlot)
            and s.host_path == host._path  # noqa: SLF001
            and s.owner_aot_entry is same_owner
        ):
            out.append(s)
    # Deduplicate (a KV with dotted path generates multiple refs in
    # the host's _refs — once per intermediate container — but we
    # only want each slot once for indent/blank inspection).
    seen: set[int] = set()
    deduped: list[KVSlot] = []
    for s in out:
        if id(s) in seen:
            continue
        seen.add(id(s))
        deduped.append(s)
    return deduped


def _host_kv_separator_leading(host: Container, doc: Document) -> Trivia:
    """Pick leading trivia for a new dotted-KV slot whose host is ``host``.

    Inherits indent + blank-line policy from existing KVs (any
    keypath length) under the same host.
    """
    kvs = _collect_host_kvs(host)
    if not kvs:
        return Trivia()
    indent_text = _extract_indent(kvs[-1].leading)
    add_blank = False
    if len(kvs) >= 2:
        add_blank = all(_leading_has_blank_line(kv.leading) for kv in kvs[1:])
    pieces: list[Any] = []
    if add_blank:
        pieces.append(NewlineNode(text=doc._newline))  # noqa: SLF001
    if indent_text:
        pieces.append(WhitespaceNode(text=indent_text))
    return Trivia(pieces)


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
        leading=_kv_separator_leading(c, doc),
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
        leading=_host_kv_separator_leading(host, doc),
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
        # Container has no slots and no contributors at all — most
        # likely a held view of a deleted subtree (Phase 3e
        # PrivateRoot detach territory). Surface as NIE so callers
        # can distinguish from internal-bug assertion failures.
        msg = (
            "structural-only implicit container with no contributors — "
            "likely a held view of a deleted subtree (Phase 3e detach)"
        )
        raise NotImplementedError(msg)
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


def _maybe_demote_synthetic_empty_header(parent: Container) -> None:
    """Drop ``parent``'s header if it is synthetic and has no direct KV body.

    Used after attaching a child header under ``parent``: if ``parent``
    was synthesised as an empty placeholder (e.g.
    ``doc["tool"] = Table.section({})``) and the new child gives it a
    dotted-implicit anchor (``[tool.poetry]``), the placeholder header
    is redundant and is removed.
    """
    from tomlrt._slots import KVSlot, StructuralHeaderSlot  # noqa: PLC0415

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
    from tomlrt._container import Document as _Document  # noqa: PLC0415

    assert isinstance(layout_root, _Document)
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

    Empty doc → no leading; non-empty → respect the user's existing
    structural-header separator convention (compact-style with no
    blanks → no leading; otherwise one blank line of separation).
    """
    if doc._head is None:  # noqa: SLF001
        return Trivia()
    headers: list[StructuralHeaderSlot] = []
    cur: Slot | None = doc._head  # noqa: SLF001
    while cur is not None:
        if isinstance(cur, StructuralHeaderSlot):
            headers.append(cur)
        cur = cur._next  # noqa: SLF001
    # Look at separators between consecutive headers (skip the first
    # — its leading is the file preamble, not a separator).
    if len(headers) >= 2:
        any_no_blank = any(not _leading_has_blank_line(h.leading) for h in headers[1:])
        if any_no_blank:
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


def _aot_separator(aot: AoT, doc: Document) -> Trivia:
    """Pick the leading-trivia for a newly-appended AoT entry header.

    Inspects the leading trivia on existing entry headers (ordinal ≥1)
    to learn the user's separator convention:

    - 0 prior separators (`len(aot) <= 1`): default to a single
      newline (which produces one blank line).
    - any prior separator has *no* blank: respect that — emit empty
      leading so the new header sits on the next line with no blank.
    - all prior separators have a blank: emit a single newline.
    """
    nl = doc._newline  # noqa: SLF001
    headers: list[Any] = []
    for entry_table in aot:
        e = entry_table._owner_aot_entry  # noqa: SLF001
        if e is not None and e.entry_slots:
            headers.append(e.entry_slots[0])
    # headers[0] is the file-leading entry, not a separator.
    separators = headers[1:]
    if not separators:
        return Trivia([NewlineNode(text=nl)])
    any_no_blank = any(not _leading_has_blank_line(h.leading) for h in separators)
    if any_no_blank:
        return Trivia()
    return Trivia([NewlineNode(text=nl)])


def add_aot_entry(aot: object, body: object, *, rehome: object | None = None) -> object:
    """Append a ``[[path]]`` entry to ``aot`` and return its `Table` view.

    If ``rehome`` is supplied (must be an unattached ``Table``), it is
    used as the entry view so the caller can preserve identity for a
    user reference. ``body`` is then ignored — ``rehome``'s own dict
    storage is used as the source body and is cleared/repopulated
    in place.
    """
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
    body_items: list[tuple[Any, Any]]
    if rehome is not None:
        assert isinstance(rehome, Table)
        assert rehome._layout_root is None  # noqa: SLF001
        entry_table = rehome
        body_items = list(rehome.items())
        dict.clear(entry_table)
    else:
        entry_table = Table()
        body_items = list(_items_for_synth(body)) if body is not None else []
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

    # If this is the very first entry of the AoT and the parent was
    # an empty synthetic placeholder section (e.g.
    # `doc["tool"] = Table.section({}); doc["tool"]["list"] = AoT(...)`),
    # the parent's header is now redundant — the dotted-implicit
    # anchor lives entirely in `[[tool.list]]`.
    if ordinal == 0:
        _maybe_demote_synthetic_empty_header(parent)

    # Populate body.
    for k, v in body_items:
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


def clone_aot_entry(
    aot: object,
    src_entry_table: object,
    *,
    dst_path: tuple[str, ...] | None = None,
) -> object:
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
    from tomlrt._array import AoT  # noqa: PLC0415
    from tomlrt._container import Table  # noqa: PLC0415

    assert isinstance(aot, AoT)
    assert isinstance(src_entry_table, Table)
    parent = aot._parent  # noqa: SLF001
    layout_root = aot._layout_root  # noqa: SLF001
    path = aot._path  # noqa: SLF001
    if layout_root is None or parent is None or not path:
        msg = "AoT.clone_entry requires the AoT to be attached to a document"
        raise RuntimeError(msg)
    doc = layout_root
    target_path = dst_path if dst_path is not None else path

    src_entry = src_entry_table._owner_aot_entry  # noqa: SLF001
    if src_entry is None:
        msg = "Source entry has no owning AoTEntry"
        raise RuntimeError(msg)

    src_slots = _validate_clonable_aot_entry(src_entry, src_entry_table)
    src_prefix = src_entry.path

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
    same_aot_clone = (
        target_path == src_entry.path and src_entry_table._layout_root is doc  # noqa: SLF001
    )
    if ordinal == 0:
        # Strip any structural prefix from the source's first header
        # (it was the source's file preamble or inter-entry separator,
        # neither relevant in the destination doc) and re-prepend the
        # destination's own structural lead-in.
        _structural, remainder = _split_leading_structural(cloned_header.leading)
        sep = _build_section_leading(doc)
        cloned_header.leading = Trivia([*sep.pieces, *remainder.pieces])
    elif same_aot_clone:
        # __imul__ / "append-from-self": use destination's existing
        # inter-entry separator style so the duplicate is laid out the
        # same way as the originals.
        _structural, remainder = _split_leading_structural(cloned_header.leading)
        sep = _aot_separator(aot, doc)
        cloned_header.leading = Trivia([*sep.pieces, *remainder.pieces])
    # Cross-doc / cross-key clone: keep the source's leading verbatim
    # so the original inter-entry separator (and any pre-header
    # comment block) survives unchanged.

    entry_table = Table()
    entry_table._layout_root = doc  # noqa: SLF001
    entry_table._path = target_path  # noqa: SLF001
    entry_table._parent = parent  # noqa: SLF001
    entry_table._owner_aot_entry = new_entry  # noqa: SLF001
    own_header_ref = SlotRef(slot=cloned_header, container=entry_table, local_key=None)
    entry_table._refs.append(own_header_ref)  # noqa: SLF001
    entry_table._header_ref = own_header_ref  # noqa: SLF001

    anchor = _last_aot_slot(aot)
    if anchor is None:
        _splice_at_end(cloned_header, doc)
    else:
        _ensure_terminator(anchor, doc)
        insert_after(anchor, cloned_header, doc)
    prev: Slot = cloned_header
    for s in cloned_slots[1:]:
        _ensure_terminator(prev, doc)
        insert_after(prev, s, doc)
        prev = s

    parent_ref = SlotRef(
        slot=cloned_header, container=parent, local_key=target_path[-1]
    )
    parent._refs.append(parent_ref)  # noqa: SLF001
    parent._index.setdefault(target_path[-1], []).append(parent_ref)  # noqa: SLF001

    _populate_entry_views(
        entry_table=entry_table,
        cloned_slots=cloned_slots[1:],
        target_prefix=target_path,
        body_owner=new_entry,
        doc=doc,
    )

    list.append(aot, entry_table)
    return entry_table


def clone_aot_entry_as_table(
    parent: Container,
    key: str,
    src_entry_table: object,
) -> object:
    """Install an AoT entry under ``parent[key]`` as a standard ``[key]`` table.

    Used by ``parent[key] = some_aot_entry`` and ``install`` paths.
    Deep-clones the source entry's slots, rewriting the head from
    ``[[..]]`` to ``[..]``, rebasing all paths from the source's
    AoT prefix to ``parent._path + (key,)``.
    """
    from tomlrt._container import Container as ContainerType  # noqa: PLC0415
    from tomlrt._container import Table  # noqa: PLC0415

    assert isinstance(parent, ContainerType)
    assert isinstance(src_entry_table, Table)
    layout_root = parent._layout_root  # noqa: SLF001
    if layout_root is None:
        msg = "clone_aot_entry_as_table requires parent attached to a document"
        raise RuntimeError(msg)
    doc = layout_root
    target_path = (*parent._path, key)  # noqa: SLF001

    src_entry = src_entry_table._owner_aot_entry  # noqa: SLF001
    if src_entry is None:
        msg = "Source entry has no owning AoTEntry"
        raise RuntimeError(msg)
    src_slots = _validate_clonable_aot_entry(src_entry, src_entry_table)
    src_prefix = src_entry.path

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
    section._layout_root = doc  # noqa: SLF001
    section._path = target_path  # noqa: SLF001
    section._parent = parent  # noqa: SLF001
    section._owner_aot_entry = parent._owner_aot_entry  # noqa: SLF001
    own_ref = SlotRef(slot=cloned_header, container=section, local_key=None)
    section._refs.append(own_ref)  # noqa: SLF001
    section._header_ref = own_ref  # noqa: SLF001

    _splice_at_end(cloned_header, doc)
    prev: Slot = cloned_header
    for s in cloned_slots[1:]:
        _ensure_terminator(prev, doc)
        insert_after(prev, s, doc)
        prev = s

    parent_ref = SlotRef(slot=cloned_header, container=parent, local_key=key)
    parent._refs.append(parent_ref)  # noqa: SLF001
    parent._index.setdefault(key, []).append(parent_ref)  # noqa: SLF001

    _populate_entry_views(
        entry_table=section,
        cloned_slots=cloned_slots[1:],
        target_prefix=target_path,
        body_owner=parent._owner_aot_entry,  # noqa: SLF001
        doc=doc,
    )

    dict.__setitem__(parent, key, section)
    return section


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
    seen_refs: set[int] = set()

    def _add_ref(r: SlotRef) -> None:
        if id(r) in seen_refs:
            return
        seen_refs.add(id(r))
        owned.add(id(r.slot))

    containers_out: list[Container] = []
    aots_out: list[Any] = []
    _collect_subtree(src_table, containers_out, aots_out, _add_ref)

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
    aot: object,
    src_table: object,
) -> object:
    """Append ``src_table`` (a standard ``[k]`` section) to ``aot`` as an entry.

    Deep-clones the source section's slots, rewriting the head from
    ``[k]`` to ``[[aot._path]]``, rebasing all paths from the source's
    section path to ``aot._path``. Preserves per-slot leading / EOL
    / lexeme bytes (so per-section comments survive).
    """
    from tomlrt._array import AoT  # noqa: PLC0415
    from tomlrt._container import Container as ContainerType  # noqa: PLC0415
    from tomlrt._container import Table  # noqa: PLC0415

    assert isinstance(aot, AoT)
    assert isinstance(src_table, ContainerType)
    parent = aot._parent  # noqa: SLF001
    layout_root = aot._layout_root  # noqa: SLF001
    path = aot._path  # noqa: SLF001
    if layout_root is None or parent is None or not path:
        msg = "AoT.clone_table requires the AoT to be attached to a document"
        raise RuntimeError(msg)
    doc = layout_root

    src_slots = _gather_section_slots(src_table)
    if not isinstance(src_slots[0], StructuralHeaderSlot):
        msg = "Source section's first owned slot is not a header"
        raise AssertionError(msg)  # noqa: TRY004
    if src_slots[0].kind != "table":
        msg = "clone_table_as_aot_entry: source must be a standard section"
        raise RuntimeError(msg)
    src_prefix = src_table._path  # noqa: SLF001

    ordinal = len(aot)
    new_entry = AoTEntry(path=path, ordinal=ordinal)

    cloned_slots = _clone_entry_slots(
        src_slots,
        new_entry=new_entry,
        body_owner=new_entry,
        src_prefix=src_prefix,
        target_prefix=path,
        head_kind="aot-entry",
    )

    head = cloned_slots[0]
    assert isinstance(head, StructuralHeaderSlot)
    cloned_header: StructuralHeaderSlot = head
    sep = _build_section_leading(doc) if ordinal == 0 else _aot_separator(aot, doc)
    _structural, remainder = _split_leading_structural(cloned_header.leading)
    cloned_header.leading = Trivia([*sep.pieces, *remainder.pieces])

    entry_table = Table()
    entry_table._layout_root = doc  # noqa: SLF001
    entry_table._path = path  # noqa: SLF001
    entry_table._parent = parent  # noqa: SLF001
    entry_table._owner_aot_entry = new_entry  # noqa: SLF001
    own_header_ref = SlotRef(slot=cloned_header, container=entry_table, local_key=None)
    entry_table._refs.append(own_header_ref)  # noqa: SLF001
    entry_table._header_ref = own_header_ref  # noqa: SLF001

    anchor = _last_aot_slot(aot)
    if anchor is None:
        _splice_at_end(cloned_header, doc)
    else:
        _ensure_terminator(anchor, doc)
        insert_after(anchor, cloned_header, doc)
    prev: Slot = cloned_header
    for s in cloned_slots[1:]:
        _ensure_terminator(prev, doc)
        insert_after(prev, s, doc)
        prev = s

    parent_ref = SlotRef(slot=cloned_header, container=parent, local_key=path[-1])
    parent._refs.append(parent_ref)  # noqa: SLF001
    parent._index.setdefault(path[-1], []).append(parent_ref)  # noqa: SLF001

    _populate_entry_views(
        entry_table=entry_table,
        cloned_slots=cloned_slots[1:],
        target_prefix=path,
        body_owner=new_entry,
        doc=doc,
    )

    list.append(aot, entry_table)
    return entry_table


def clone_aot(
    parent: Container,
    key: str,
    src_aot: object,
) -> object:
    """Install ``src_aot`` (an attached AoT) under ``parent[key]``.

    Each entry is deep-cloned with path-rebasing so any nested
    sub-sections stay logically inside the new key.
    """
    from tomlrt._array import AoT  # noqa: PLC0415
    from tomlrt._container import Container as ContainerType  # noqa: PLC0415

    assert isinstance(parent, ContainerType)
    assert isinstance(src_aot, AoT)
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
    head_kind: str,
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
    import copy  # noqa: PLC0415

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
            c.key_parts = _build_header_keyparts(c.path)
            c.key_seps = ["."] * (len(c.key_parts) - 1)
            c.entry = new_entry if c.kind == "aot-entry" else None
        cloned.append(c)
        if new_entry is not None:
            new_entry.entry_slots.append(c)

    if not has_header or not cloned:
        return cloned
    head = cloned[0]
    assert isinstance(head, StructuralHeaderSlot)
    head.kind = head_kind  # type: ignore[assignment]
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
            child = Table()
            child._layout_root = doc  # noqa: SLF001
            child._path = cur_path  # noqa: SLF001
            child._parent = cur  # noqa: SLF001
            child._owner_aot_entry = body_owner  # noqa: SLF001
            containers[cur_path] = child
            dict.__setitem__(cur, comp, child)
            cur = child
        return cur

    for s in cloned_slots:
        if isinstance(s, StructuralHeaderSlot):
            assert s.kind == "table"
            container = _ensure_container(s.path)
            own_ref = SlotRef(slot=s, container=container, local_key=None)
            container._refs.append(own_ref)  # noqa: SLF001
            container._header_ref = own_ref  # noqa: SLF001
            if s.path == target_prefix:
                continue
            local = s.path[-1]
            parent_path = s.path[:-1]
            parent_view = _ensure_container(parent_path)
            binding = SlotRef(slot=s, container=parent_view, local_key=local)
            parent_view._refs.append(binding)  # noqa: SLF001
            parent_view._index.setdefault(local, []).append(binding)  # noqa: SLF001
            continue
        assert isinstance(s, KVSlot)
        host = _ensure_container(s.host_path)
        assert s.value is not None
        slot_value = s.value
        if len(s.key_parts) == 1:
            key = s.key_parts[0].value
            kv_ref = SlotRef(slot=s, container=host, local_key=key)
            host._refs.append(kv_ref)  # noqa: SLF001
            host._index.setdefault(key, []).append(kv_ref)  # noqa: SLF001
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
                ref = SlotRef(slot=s, container=cur, local_key=comp)
                cur._refs.append(ref)  # noqa: SLF001
                cur._index.setdefault(comp, []).append(ref)  # noqa: SLF001
                if comp not in cur:
                    sub = Table()
                    sub._layout_root = doc  # noqa: SLF001
                    sub._path = (*cur._path, comp)  # noqa: SLF001
                    sub._parent = cur  # noqa: SLF001
                    sub._owner_aot_entry = body_owner  # noqa: SLF001
                    containers[sub._path] = sub  # noqa: SLF001
                    dict.__setitem__(cur, comp, sub)
                nxt = dict.__getitem__(cur, comp)
                if not isinstance(nxt, Table):
                    msg = "internal: dotted KV traversal hit non-Table"
                    raise AssertionError(msg)  # noqa: TRY004
                cur = nxt
            leaf_key = s.key_parts[-1].value
            kv_ref = SlotRef(slot=s, container=cur, local_key=leaf_key)
            cur._refs.append(kv_ref)  # noqa: SLF001
            cur._index.setdefault(leaf_key, []).append(kv_ref)  # noqa: SLF001
            decoded = _decode_value(
                slot_value,
                layout_root=doc,
                parent=cur,
                path=(*cur._path, leaf_key),  # noqa: SLF001
                owner=body_owner,
            )
            dict.__setitem__(cur, leaf_key, decoded)


def _walk_path(table: Container, path: tuple[str, ...]) -> Container:
    cur: Any = table
    for comp in path:
        cur = dict.__getitem__(cur, comp)
    return cur  # type: ignore[no-any-return]


def _validate_clonable_aot_entry(
    src_entry: AoTEntry,
    src_entry_table: object,  # noqa: ARG001
) -> list[Slot]:
    """Validate the source entry can be cloned; return its slot list.

    Pure validation — no mutation. Raises NotImplementedError for
    shapes the cloner does not yet support, RuntimeError for
    invariant violations. Shared between `clone_aot_entry` and
    `check_clone_aot_entry` so the two cannot drift.
    """
    src_slots = list(src_entry.entry_slots)
    if not src_slots or not isinstance(src_slots[0], StructuralHeaderSlot):
        msg = "Source entry has no header slot"
        raise RuntimeError(msg)
    for s in src_slots[1:]:
        if isinstance(s, StructuralHeaderSlot):
            if s.kind != "table":
                msg = "AoT clone for nested non-table headers is not yet implemented"
                raise NotImplementedError(msg)
            continue
        if not isinstance(s, KVSlot):
            msg = "AoT clone: unexpected slot type"
            raise AssertionError(msg)  # noqa: TRY004
    return src_slots


def check_clone_aot_entry(aot: object, src_entry_table: object) -> None:
    """Raise NotImplementedError/RuntimeError if `clone_aot_entry` would.

    Same preconditions as `clone_aot_entry`, but without any side
    effects. Used by `AoT.__imul__` to preflight every source entry
    so a failure on entry N does not leave entries 0..N-1 cloned.
    """
    from tomlrt._array import AoT  # noqa: PLC0415
    from tomlrt._container import Table  # noqa: PLC0415

    assert isinstance(aot, AoT)
    assert isinstance(src_entry_table, Table)
    if (
        aot._layout_root is None  # noqa: SLF001
        or aot._parent is None  # noqa: SLF001
        or not aot._path  # noqa: SLF001
    ):
        msg = "AoT.clone_entry requires the AoT to be attached to a document"
        raise RuntimeError(msg)
    src_entry = src_entry_table._owner_aot_entry  # noqa: SLF001
    if src_entry is None:
        msg = "Source entry has no owning AoTEntry"
        raise RuntimeError(msg)
    _validate_clonable_aot_entry(src_entry, src_entry_table)


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
            if not isinstance(nxt, ContainerType):
                msg = f"intermediate {comp!r} is not a table"
                raise TypeError(msg)
            chain.append(nxt)
            continue
        implicit = Table()
        implicit._layout_root = doc  # noqa: SLF001
        implicit._path = (*parent._path, *sub[: j + 1])  # noqa: SLF001
        implicit._parent = cur  # noqa: SLF001
        implicit._owner_aot_entry = owner  # noqa: SLF001
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

    _maybe_demote_synthetic_empty_header(parent)

    for k, v in pending:
        if not (_is_scalar(v) or _is_synth_inline(v)):
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
        _splice_at_end(header, doc)

    parent_ref = SlotRef(slot=header, container=parent, local_key=key)
    parent._refs.append(parent_ref)  # noqa: SLF001
    parent._index.setdefault(key, []).append(parent_ref)  # noqa: SLF001
    dict.__setitem__(parent, key, section)

    _maybe_demote_synthetic_empty_header(parent)

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
            owner=owner,
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
    # Also collect slots owned by any nested AoT entries reachable
    # from this entry's subtree (e.g. promoted [[arr.xs]] entries):
    # those have their own AoTEntry with its own entry_slots, not in
    # `e.entry_slots`, but they must be removed alongside the parent
    # entry so deletion is structurally complete.
    from tomlrt._array import AoT as _AoT  # noqa: PLC0415

    def _collect_nested_aot_slots(c: Container) -> None:
        for v in c.values():
            if isinstance(v, _AoT):
                for nested_entry_table in v:
                    ne = nested_entry_table._owner_aot_entry  # noqa: SLF001
                    if ne is not None:
                        owned.update(ne.entry_slots)
                    _collect_nested_aot_slots(nested_entry_table)
            elif isinstance(v, ContainerType) and not v._inline:  # noqa: SLF001
                _collect_nested_aot_slots(v)

    from tomlrt._container import Container as ContainerType  # noqa: PLC0415

    _collect_nested_aot_slots(entry_table)

    # Snapshot the entry's dict-storage values to a fresh Table.
    snapshot = Table()
    for k, v in entry_table.items():
        dict.__setitem__(snapshot, k, v)

    # Scrub refs from every still-live container, walking from the doc.
    _scrub_refs_to_owned_slots(doc, owned)

    # Unlink owned slots from the doc-stream linked list.
    for slot in list(owned):
        unlink_slot(slot, doc)

    # Pop entry from the AoT logical list.
    list.pop(aot, index)

    # If empty AoT now, also remove the parent _index[k] entry entirely
    # (dict storage of parent retains the AoT object).
    last_key = aot._path[-1]  # noqa: SLF001
    if len(aot) == 0 and not parent._index.get(last_key):  # noqa: SLF001
        parent._index.pop(last_key, None)  # noqa: SLF001

    return snapshot


def replace_aot_entry_with_clone(
    aot: object,
    index: int,
    src_entry_table: object,
) -> None:
    """Replace ``aot[index]`` with a deep clone of ``src_entry_table``.

    Preserves the *destination* entry header's leading trivia (and
    therefore any pre-header comment block above the original
    ``[[..]]`` line) while replacing the entry's body with a clone
    of the source entry's slots — preserving the source's per-KV
    leading / EOL / lexeme trivia.

    Both entries must be attached AoT-entry tables.
    """
    from tomlrt._array import AoT  # noqa: PLC0415
    from tomlrt._container import Table  # noqa: PLC0415

    assert isinstance(aot, AoT)
    assert isinstance(src_entry_table, Table)
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

    src_slots = _validate_clonable_aot_entry(src_entry, src_entry_table)
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
    # out for now to keep ownership state consistent during clear().
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


def replace_aot_entry(aot: object, index: int, body: object) -> None:
    """Replace ``aot[index]`` in place.

    Keeps the entry's header slot and live `Table` view; just clears
    the body and re-populates from ``body``.

    O(m) in the size of ``body``, independent of AoT length and
    document size. Header position and `_refs` ordering are preserved
    by construction (no slot splicing involved).
    """
    from collections.abc import Mapping  # noqa: PLC0415

    from tomlrt._array import AoT  # noqa: PLC0415

    assert isinstance(aot, AoT)
    n = len(aot)
    if not -n <= index < n:
        msg = f"AoT index {index} out of range (len {n})"
        raise IndexError(msg)
    if index < 0:
        index += n
    if body is not None and not isinstance(body, Mapping):
        msg = f"AoT entry replacement body must be Mapping, got {type(body).__name__}"
        raise TypeError(msg)
    entry_table = aot[index]
    if body is entry_table:
        return
    items = list(body.items()) if body is not None else []
    entry_table.clear()
    for k, v in items:
        entry_table[k] = v


def renormalise_aot_order(aot: object, new_logical_order: list[Any]) -> None:
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
    from tomlrt._array import AoT  # noqa: PLC0415

    assert isinstance(aot, AoT)
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
    position: dict[int, int] = {}
    cur = doc._head  # noqa: SLF001
    idx = 0
    while cur is not None:
        position[id(cur)] = idx
        idx += 1
        cur = cur._next  # noqa: SLF001
    chain: list[Container] = []
    anc: Container | None = aot._parent  # noqa: SLF001
    while anc is not None:
        chain.append(anc)
        anc = anc._parent  # noqa: SLF001
    for c in chain:
        c._refs.sort(key=lambda r: position[id(r.slot)])  # noqa: SLF001
        for k, refs in c._index.items():  # noqa: SLF001
            refs.sort(key=lambda r: position[id(r.slot)])
            del k


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
    "remove_aot_entry",
    "renormalise_aot_order",
    "replace_aot_entry",
    "unlink_slot",
]
