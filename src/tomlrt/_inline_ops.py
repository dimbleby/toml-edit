"""Inline-table mutation primitives (Phase 3b).

Inline tables are decoupled from the doc-stream linked list: a top-
level inline table is wrapped by a single `KVSlot` whose `value` is
an `InlineTableValue`. Mutation of the inline-table contents is a
local operation on the `InlineTableValue.entries` list, plus a
matching `dict.__setitem__` / `__delitem__` on the logical view.

This module owns the trivia fixups required to keep the result a
valid, nicely-spaced inline table:

* `append_entry` — splice a new entry at the end, transferring
  the prior closing space to the new entry's trailing and giving
  the previous entry a comma + a single space after it.
* `replace_entry_value` — overwrite the `value` field of the entry
  matching the given logical key (no spacing changes).
* `delete_entry` — remove the entry, then if the deleted entry was
  last, fold the prior entry's post-comma trivia into its trailing
  and clear its comma so we don't render a trailing comma (illegal
  in TOML 1.0; allowed in 1.1 but not what we want by default for a
  delete).

All entry lookups walk up the inline-table chain (via `_parent`) to
the outermost inline-table that owns the backing
`InlineTableValue` — entries for dotted keys like ``{a.b = 1}`` are
filed there, with multi-component `key_parts`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from tomlrt._trivia import Trivia, WhitespaceNode
from tomlrt._values import InlineTableEntry

if TYPE_CHECKING:
    from tomlrt._container import Container
    from tomlrt._values import InlineTableValue, KeyPart, Value
else:
    from tomlrt._values import KeyPart  # runtime: instantiated below


_RE_BARE_KEY = re.compile(r"\A[A-Za-z0-9_\-]+\Z")


def _outermost_inline(t: Container) -> Container:
    """Walk up `_parent` until reaching the inline table that owns `_value`."""
    cur = t
    while cur._value is None:  # noqa: SLF001
        parent = cur._parent  # noqa: SLF001
        if parent is None or not parent._inline:  # noqa: SLF001
            msg = "internal: inline-table chain has no _value-bearing root"
            raise AssertionError(msg)
        cur = parent
    return cur


def _entry_key_path(t: Container, leaf: str) -> tuple[str, ...]:
    """Full dotted path used as `key_parts` in the outermost inline value."""
    root = _outermost_inline(t)
    suffix = t._path[len(root._path) :]  # noqa: SLF001
    return (*suffix, leaf)


def _find_entry(
    iv: InlineTableValue, key_path: tuple[str, ...]
) -> tuple[int, InlineTableEntry] | None:
    for i, e in enumerate(iv.entries):
        if tuple(p.value for p in e.key_parts) == key_path:
            return i, e
    return None


def _find_prefix_entries(iv: InlineTableValue, key_path: tuple[str, ...]) -> list[int]:
    """Indices of entries whose `key_parts` start with `key_path`.

    Used when deleting a synthetic dotted-prefix container — e.g.
    ``del obj["a"]`` for ``{a.b = 1, a.c = 2}`` removes both entries.
    """
    n = len(key_path)
    out: list[int] = []
    for i, e in enumerate(iv.entries):
        kp = tuple(p.value for p in e.key_parts)
        if len(kp) > n and kp[:n] == key_path:
            out.append(i)
    return out


def _is_ws_only(trivia: Trivia) -> bool:
    return all(isinstance(p, WhitespaceNode) for p in trivia.pieces)


def _make_key_parts(path: tuple[str, ...]) -> list[KeyPart]:
    out: list[KeyPart] = []
    for p in path:
        if _RE_BARE_KEY.match(p):
            out.append(KeyPart(raw=p, value=p, kind="bare"))
        else:
            out.append(KeyPart(raw=_quote_basic(p), value=p, kind="basic"))
    return out


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


def _ws(text: str) -> Trivia:
    return Trivia(pieces=[WhitespaceNode(text=text)])


# ---------------------------------------------------------------------------
# Public ops
# ---------------------------------------------------------------------------


def replace_entry_value(t: Container, key: str, new_value: Value) -> bool:
    """Replace the value of an existing entry in place.

    Returns True iff an entry was found and replaced. No trivia is
    altered.
    """
    root = _outermost_inline(t)
    iv = root._value  # noqa: SLF001
    assert iv is not None
    found = _find_entry(iv, _entry_key_path(t, key))
    if found is None:
        return False
    _, entry = found
    entry.value = new_value
    return True


def append_entry(t: Container, key: str, new_value: Value) -> None:
    """Append a fresh entry for `key` to the outermost inline table."""
    root = _outermost_inline(t)
    iv = root._value  # noqa: SLF001
    assert iv is not None
    key_path = _entry_key_path(t, key)
    if _find_entry(iv, key_path) is not None:
        msg = f"internal: append_entry called for existing key {key!r}"
        raise AssertionError(msg)

    new_entry = InlineTableEntry(
        leading=Trivia(),
        key_parts=_make_key_parts(key_path),
        key_seps=["."] * (len(key_path) - 1),
        pre_eq=" ",
        post_eq=" ",
        value=new_value,
        trailing=Trivia(),
        has_comma=False,
        post_comma_trivia=Trivia(),
    )

    if not iv.entries:
        # Empty {} → { key = value }.  Mirror the conventional spacing
        # whatever the user originally wrote (final_trivia might already
        # encode it).
        new_entry.leading = _ws(" ")
        new_entry.trailing = iv.final_trivia if iv.final_trivia.pieces else _ws(" ")
        iv.final_trivia = Trivia()
        iv.entries.append(new_entry)
        return

    last = iv.entries[-1]
    if last.has_comma:
        # Existing trailing comma (TOML 1.1 style); the new entry slots in
        # before whatever post-comma trivia carried the closing space —
        # but only if that trivia is whitespace-only. If it carries a
        # comment or newline, those belong logically to the existing
        # layout, not to the inserted entry.
        if _is_ws_only(last.post_comma_trivia):
            new_entry.trailing = last.post_comma_trivia
            last.post_comma_trivia = _ws(" ")
        else:
            new_entry.leading = _ws(" ")
            new_entry.trailing = _ws(" ")
    elif _is_ws_only(last.trailing):
        # No comma yet — promote `last`: take its (whitespace-only)
        # trailing as the new closing space for the inserted entry and
        # replace it with a ", " separator.
        new_entry.trailing = last.trailing if last.trailing.pieces else _ws(" ")
        last.trailing = Trivia()
        last.has_comma = True
        last.post_comma_trivia = _ws(" ")
    else:
        # `last.trailing` carries a comment / newline (TOML 1.1
        # multiline). Don't migrate — leave it where the user put it
        # and append a comma + new entry with default spacing.
        last.has_comma = True
        last.post_comma_trivia = _ws(" ")
        new_entry.trailing = _ws(" ")
    iv.entries.append(new_entry)


def delete_entry(t: Container, key: str) -> bool:
    """Remove the entry (or all dotted-prefix entries) matching `key`.

    Returns True iff at least one entry was removed. When ``key`` names
    a synthetic dotted-prefix container (e.g. ``a`` in
    ``{a.b = 1, a.c = 2}``), every entry whose ``key_parts`` start with
    that prefix is removed.
    """
    root = _outermost_inline(t)
    iv = root._value  # noqa: SLF001
    assert iv is not None
    full_path = _entry_key_path(t, key)

    # Single exact match (the common, leaf-key case).
    found = _find_entry(iv, full_path)
    if found is not None:
        idx, removed = found
        iv.entries.pop(idx)
        _fix_tail_after_delete(iv, idx, removed)
        return True

    # Prefix delete: dotted-prefix container.
    indices = _find_prefix_entries(iv, full_path)
    if not indices:
        return False
    last_removed_idx = indices[-1]
    last_removed_entry = iv.entries[last_removed_idx]
    for i in reversed(indices):
        iv.entries.pop(i)
    # Tail fixup: only if the removed block extended through the old end.
    if last_removed_idx == len(iv.entries):  # entries shrank past `last_removed_idx`
        _fix_tail_after_delete(iv, last_removed_idx, last_removed_entry)
    return True


def _fix_tail_after_delete(
    iv: InlineTableValue, removed_idx: int, removed: InlineTableEntry
) -> None:
    """If the removed entry was the last, drop the new last's trailing comma."""
    if not iv.entries or removed_idx != len(iv.entries):
        return
    new_last = iv.entries[-1]
    if new_last.has_comma:
        new_last.trailing = Trivia(
            pieces=[
                *new_last.trailing.pieces,
                *new_last.post_comma_trivia.pieces,
                *removed.leading.pieces,
                *removed.trailing.pieces,
                *removed.post_comma_trivia.pieces,
            ]
        )
        new_last.has_comma = False
        new_last.post_comma_trivia = Trivia()
    else:
        new_last.trailing = Trivia(
            pieces=[
                *new_last.trailing.pieces,
                *removed.leading.pieces,
                *removed.trailing.pieces,
            ]
        )


__all__ = ["append_entry", "delete_entry", "replace_entry_value"]
