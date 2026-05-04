"""Initial logical-container build.

Single linear pass over a `ParseResult`'s slot stream that
constructs the `Document` body and all nested `Table` / `Array` /
`AoT` views, populating dict storage in doc-stream-first-occurrence
order.

Per the duck-reviewed Phase 1/2 boundary: this is the *one* place
that derives implicit containers from slot paths. The parser does
not build logical containers; `_container.py` does not duplicate the
derivation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tomlrt._array import AoT, Array
from tomlrt._container import Container, Document, Table
from tomlrt._slots import KVSlot, StructuralHeaderSlot
from tomlrt._values import (
    ArrayValue,
    InlineTableValue,
)

if TYPE_CHECKING:
    from tomlrt._parser import ParseResult
    from tomlrt._slots import AoTEntry, Slot
    from tomlrt._values import (
        Value,
    )


def build_initial_containers(doc: Document, slots: list[Slot]) -> None:
    """Walk the slot stream and populate ``doc`` and its descendants."""
    for slot in slots:
        if isinstance(slot, StructuralHeaderSlot):
            _apply_header(doc, slot)
        else:
            assert isinstance(slot, KVSlot)
            _apply_kv(doc, slot)


# ---------------------------------------------------------------------------
# Header handling
# ---------------------------------------------------------------------------


def _apply_header(doc: Document, slot: StructuralHeaderSlot) -> None:
    if slot.kind == "aot-entry":
        assert slot.entry is not None
        _open_aot_entry(doc, slot.path, slot.entry)
    else:
        _open_table(doc, slot.path, slot)


def _open_table(
    doc: Document, path: tuple[str, ...], header: StructuralHeaderSlot
) -> Table:
    """Open ``[a.b.c]`` — return the `Table` view for ``path``.

    Walks (and creates as needed) all implicit ancestors. Raises if an
    intermediate name is bound to a non-table value (the validator
    should have already rejected this; the assertion guards against
    drift).
    """
    parent = _resolve_parent(doc, path[:-1])
    name = path[-1]
    existing = parent.get(name)
    if existing is None:
        table = _make_table(parent, path, owner=header.owner_aot_entry)
        parent[name] = table
        return table
    # Re-opening an implicit table promoted by an earlier dotted KV
    # or child header. The validator has already enforced that this
    # is legal.
    assert isinstance(existing, Table), (
        f"header [{'.'.join(path)}] reopens a non-table at "
        f"{name!r} (got {type(existing).__name__}); validator drift"
    )
    return existing


def _open_aot_entry(
    doc: Document,
    path: tuple[str, ...],
    entry: AoTEntry,
) -> Table:
    """Open ``[[a.b]]`` — append a fresh `Table` to the AoT at ``path``."""
    parent = _resolve_parent(doc, path[:-1])
    name = path[-1]
    aot = parent.get(name)
    if aot is None:
        aot = AoT()
        parent[name] = aot
    assert isinstance(aot, AoT), (
        f"AoT header [[{'.'.join(path)}]] collides with non-AoT at "
        f"{name!r} (got {type(aot).__name__}); validator drift"
    )
    table = _make_table(parent, path, owner=entry)
    aot.append(table)
    return table


def _resolve_parent(doc: Document, prefix: tuple[str, ...]) -> Container:
    """Walk ``prefix`` from ``doc``, creating implicit tables as needed.

    For an AoT prefix, descends into the *most recent* entry.
    """
    cur: Container = doc
    for i, name in enumerate(prefix):
        sub = cur.get(name)
        if sub is None:
            child_path = prefix[: i + 1]
            child = _make_table(cur, child_path, owner=cur._owner_aot_entry)  # noqa: SLF001
            cur[name] = child
            cur = child
        elif isinstance(sub, Table):
            cur = sub
        elif isinstance(sub, AoT):
            assert sub, "validator should have rejected empty-AoT prefix"
            cur = sub[-1]
        else:
            msg = (
                f"path component {name!r} is bound to "
                f"{type(sub).__name__}, not a Table/AoT (validator drift)"
            )
            raise AssertionError(msg)
    return cur


def _make_table(
    parent: Container, path: tuple[str, ...], *, owner: AoTEntry | None
) -> Table:
    table = Table()
    table._layout_root = parent._layout_root  # noqa: SLF001
    table._path = path  # noqa: SLF001
    table._parent = parent  # noqa: SLF001
    table._inline = False  # noqa: SLF001
    table._owner_aot_entry = owner  # noqa: SLF001
    return table


# ---------------------------------------------------------------------------
# KV slot handling
# ---------------------------------------------------------------------------


def _apply_kv(doc: Document, slot: KVSlot) -> None:
    """Bind a `key = value` slot into its host container."""
    host = _resolve_host(doc, slot.host_path)
    decoded = slot.key
    target = _walk_dotted(host, decoded[:-1], slot.owner_aot_entry)
    name = decoded[-1]
    assert slot.value is not None
    target[name] = _value_to_python(slot.value, parent=target, name=name)


def _resolve_host(doc: Document, host_path: tuple[str, ...]) -> Container:
    """Find the host container of a KV (the table its header opened).

    Identical to `_resolve_parent` for the head path; an empty
    `host_path` is the document root.
    """
    if not host_path:
        return doc
    return _resolve_parent_or_self(doc, host_path)


def _resolve_parent_or_self(doc: Document, path: tuple[str, ...]) -> Container:
    cur: Container = doc
    for name in path:
        sub = cur[name]
        if isinstance(sub, AoT):
            cur = sub[-1]
        else:
            assert isinstance(sub, Table)
            cur = sub
    return cur


def _walk_dotted(
    host: Container, prefix: tuple[str, ...], owner: AoTEntry | None
) -> Container:
    """Walk a dotted-KV intermediate path inside an already-open host."""
    cur: Container = host
    for i, name in enumerate(prefix):
        sub = cur.get(name)
        if sub is None:
            child_path = cur._path + tuple(prefix[: i + 1])  # noqa: SLF001
            child = _make_table(cur, child_path, owner=owner)
            cur[name] = child
            cur = child
        else:
            assert isinstance(sub, Table), (
                f"dotted-key step {name!r} hits {type(sub).__name__} (validator drift)"
            )
            cur = sub
    return cur


# ---------------------------------------------------------------------------
# Value decoding
# ---------------------------------------------------------------------------


def _value_to_python(value: Value, *, parent: Container, name: str) -> object:
    if isinstance(value, ArrayValue):
        return _decode_array(value)
    if isinstance(value, InlineTableValue):
        return _decode_inline_table(value, parent=parent, name=name)
    # All scalar value types carry the decoded Python value as `.value`.
    return value.value


def _decode_array(value: ArrayValue) -> Array:
    arr = Array()
    arr._value = value  # noqa: SLF001
    for item in value.items:
        arr.append(_value_to_python_top(item.value))
    return arr


def _decode_inline_table(
    value: InlineTableValue, *, parent: Container, name: str
) -> Table:
    table = Table()
    table._layout_root = parent._layout_root  # noqa: SLF001
    table._path = (*parent._path, name)  # noqa: SLF001
    table._parent = parent  # noqa: SLF001
    table._inline = True  # noqa: SLF001
    table._owner_aot_entry = parent._owner_aot_entry  # noqa: SLF001
    for entry in value.entries:
        # Inline-table entries can themselves be dotted (TOML 1.1).
        decoded_key = [p.value for p in entry.key_parts]
        cur: Container = table
        for step in decoded_key[:-1]:
            sub = cur.get(step)
            if sub is None:
                inner = Table()
                inner._layout_root = table._layout_root  # noqa: SLF001
                inner._path = (*cur._path, step)  # noqa: SLF001
                inner._parent = cur  # noqa: SLF001
                inner._inline = True  # noqa: SLF001
                inner._owner_aot_entry = cur._owner_aot_entry  # noqa: SLF001
                cur[step] = inner
                cur = inner
            else:
                assert isinstance(sub, Table)
                cur = sub
        leaf = decoded_key[-1]
        cur[leaf] = _value_to_python(entry.value, parent=cur, name=leaf)
    return table


def _value_to_python_top(value: Value) -> object:
    """Decode a value that lives directly inside an inline array.

    The parent is the array, not a table, so the inline-table case has
    no `parent`/`name` — synthesise a detached one.
    """
    if isinstance(value, ArrayValue):
        return _decode_array(value)
    if isinstance(value, InlineTableValue):
        # Detached inline table — `_path` is empty and `_parent` is None.
        # That's fine for read-only Phase 2; mutation paths will care.
        sentinel = Table()
        return _decode_inline_table(value, parent=sentinel, name="")
    return value.value


def build_from_parse(result: ParseResult) -> Document:
    """One-shot: parse-result → fully constructed `Document`."""
    doc = Document.__new__(Document)
    Container.__init__(doc)
    doc._head = result.slots[0] if result.slots else None  # noqa: SLF001
    doc._tail = result.slots[-1] if result.slots else None  # noqa: SLF001
    doc._trailing = result.trailing  # noqa: SLF001
    doc._newline = result.newline  # noqa: SLF001
    doc._layout_root = doc  # noqa: SLF001
    build_initial_containers(doc, result.slots)
    return doc


__all__ = ["build_from_parse", "build_initial_containers"]
