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

import re
from typing import TYPE_CHECKING

from tomlrt._slots import KVSlot, SlotRef, StructuralHeaderSlot
from tomlrt._trivia import EolTrivia, NewlineNode, Trivia
from tomlrt._values import KeyPart

if TYPE_CHECKING:
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


def delete_direct_kv(c: Container, key: str) -> None:
    """Delete a single direct (non-dotted) KV from ``c``."""
    if c._owner_aot_entry is not None:  # noqa: SLF001
        msg = "delete from AoT entry body is not yet supported"
        raise NotImplementedError(msg)
    refs = c._index.get(key)  # noqa: SLF001
    if not refs:
        raise KeyError(key)
    if len(refs) > 1:
        msg = "delete of dotted-shared / multi-ref KV deferred to Phase 3d"
        raise NotImplementedError(msg)
    ref = refs[0]
    slot = ref.slot
    if not isinstance(slot, KVSlot) or len(slot.key_parts) != 1:
        msg = "delete of dotted-key KV deferred to Phase 3d"
        raise NotImplementedError(msg)
    if slot.host_path != c._path:  # noqa: SLF001
        msg = "delete of inherited dotted-host KV deferred to Phase 3d"
        raise NotImplementedError(msg)

    layout_root = c._layout_root  # noqa: SLF001
    if layout_root is None:
        msg = "internal: container has no layout root"
        raise AssertionError(msg)
    doc = layout_root  # PrivateRoot arrives in Phase 3e

    c._refs.remove(ref)  # noqa: SLF001
    del c._index[key]  # noqa: SLF001

    if c._body_tail is slot:  # noqa: SLF001
        c._body_tail = _recompute_body_tail(c)  # noqa: SLF001

    unlink_slot(slot, doc)


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
    "delete_direct_kv",
    "insert_after",
    "insert_before_head",
    "unlink_slot",
]
