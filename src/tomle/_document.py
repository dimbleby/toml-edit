"""Logical Document/Table/Array/AoT view over a CST.

This module exposes the public mapping/sequence types that users
interact with. The read path is implemented here; mutation will be
added in a follow-up phase (see plan.md).
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from datetime import date, datetime, time
from typing import TYPE_CHECKING, TypeAlias

from typing_extensions import override

from tomle._errors import TOMLEditError
from tomle._nodes import (
    ArrayNode,
    BoolNode,
    DateTimeNode,
    FloatNode,
    InlineTableNode,
    IntegerNode,
    StringNode,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tomle._nodes import (
        DocumentNode,
        KeyValueNode,
        SectionNode,
        ValueNode,
    )


Scalar: TypeAlias = str | int | float | bool | datetime | date | time
TomlValue: TypeAlias = "Scalar | Array | AoT | Table"


# ---------------------------------------------------------------------------
# Read-side helpers
# ---------------------------------------------------------------------------


def _scalar_value(node: ValueNode) -> Scalar | None:
    """Return the Python scalar for a value node, or None if it's a container."""
    if isinstance(node, StringNode):
        return node.value
    if isinstance(node, BoolNode):
        return node.value
    if isinstance(node, IntegerNode):
        return node.value
    if isinstance(node, FloatNode):
        return node.value
    if isinstance(node, DateTimeNode):
        return node.value
    return None


def _value_for(node: ValueNode) -> TomlValue:
    if isinstance(node, ArrayNode):
        return Array(node)
    if isinstance(node, InlineTableNode):
        return _InlineTable(node)
    scalar = _scalar_value(node)
    assert scalar is not None  # exhaustive by construction
    return scalar


def _materialise_array(node: ArrayNode) -> list[TomlValue]:
    return [_value_for(item.value) for item in node.items]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class Table(MutableMapping[str, TomlValue]):
    """A logical TOML table.

    All mapping flavours in toml-edit (top-level document, standard
    table, inline table, and the synthetic mappings spawned by dotted
    keys) inherit from :class:`Table`, so values typed as ``Table``
    cover every nested mapping you can encounter while walking a
    document.

    Subclasses provide ``_items()`` which yields ``(key, value)`` pairs
    in document order. The read path is wired up; mutation raises
    :class:`tomle.TOMLEditError` until the next implementation phase.
    """

    __slots__ = ()

    def _items(self) -> Iterator[tuple[str, TomlValue]]:  # pragma: no cover
        raise NotImplementedError

    @override
    def __iter__(self) -> Iterator[str]:
        for k, _ in self._items():
            yield k

    @override
    def __len__(self) -> int:
        return sum(1 for _ in self._items())

    @override
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return any(k == key for k, _ in self._items())

    @override
    def __getitem__(self, key: str) -> TomlValue:
        for k, v in self._items():
            if k == key:
                return v
        raise KeyError(key)

    @override
    def __setitem__(self, key: str, value: TomlValue) -> None:  # pragma: no cover
        raise TOMLEditError(
            "mutation is not yet implemented in this build of toml-edit",
        )

    @override
    def __delitem__(self, key: str) -> None:  # pragma: no cover
        raise TOMLEditError(
            "mutation is not yet implemented in this build of toml-edit",
        )

    @override
    def __repr__(self) -> str:
        body = ", ".join(f"{k!r}: {v!r}" for k, v in self._items())
        return f"{type(self).__name__}({{{body}}})"


class _InlineTable(Table):
    """Mapping view over an :class:`InlineTableNode`."""

    __slots__ = ("_node",)

    def __init__(self, node: InlineTableNode) -> None:
        self._node = node

    @override
    def _items(self) -> Iterator[tuple[str, TomlValue]]:
        groups: dict[str, list[tuple[tuple[str, ...], ValueNode]]] = {}
        order: list[str] = []
        for entry in self._node.entries:
            head = entry.key.path[0]
            if head not in groups:
                groups[head] = []
                order.append(head)
            groups[head].append((entry.key.path, entry.value))
        for head in order:
            entries = groups[head]
            if len(entries) == 1 and len(entries[0][0]) == 1:
                yield head, _value_for(entries[0][1])
            else:
                yield head, _DottedInlineSubTable(entries, depth=1)


class _DottedInlineSubTable(Table):
    """Inline view for the tail of a dotted-key chain."""

    __slots__ = ("_depth", "_entries")

    def __init__(
        self,
        entries: list[tuple[tuple[str, ...], ValueNode]],
        *,
        depth: int,
    ) -> None:
        self._entries = entries
        self._depth = depth

    @override
    def _items(self) -> Iterator[tuple[str, TomlValue]]:
        groups: dict[str, list[tuple[tuple[str, ...], ValueNode]]] = {}
        order: list[str] = []
        terminals: dict[str, ValueNode] = {}
        for path, value in self._entries:
            if len(path) <= self._depth:
                terminals[path[-1]] = value
                if path[-1] not in order:
                    order.append(path[-1])
                continue
            head = path[self._depth]
            if head not in groups:
                groups[head] = []
                if head not in order:
                    order.append(head)
            groups[head].append((path, value))
        for head in order:
            if head in terminals:
                yield head, _value_for(terminals[head])
            else:
                yield head, _DottedInlineSubTable(groups[head], depth=self._depth + 1)


class _StdTable(Table):
    """Standard TOML table view: aggregates physical sections by path."""

    __slots__ = ("_doc_view", "_path", "_pinned_sections")

    def __init__(
        self,
        doc_view: _DocumentView,
        path: tuple[str, ...],
        *,
        sections: list[SectionNode] | None = None,
    ) -> None:
        self._doc_view = doc_view
        self._path = path
        self._pinned_sections = sections

    @override
    def _items(self) -> Iterator[tuple[str, TomlValue]]:
        return self._doc_view.iter_table(self._path, self._pinned_sections)


class Document(_StdTable):
    """Top-level TOML document. Subclass of :class:`Table`."""

    __slots__ = ("_node",)

    def __init__(self, node: DocumentNode) -> None:
        view = _DocumentView(node)
        self._node = node
        super().__init__(view, ())

    @property
    def cst(self) -> DocumentNode:
        """The underlying physical CST (intended for tooling/debugging)."""
        return self._node

    def render(self) -> str:
        return self._node.render()


# ---------------------------------------------------------------------------
# Array (inline) and AoT (array of tables)
# ---------------------------------------------------------------------------


class Array(list[TomlValue]):
    """Inline TOML array exposed as a real :class:`list`.

    Mutation is not yet wired through to the underlying CST in this MVP
    cut; reads return plain Python values.
    """

    __slots__ = ("_node",)

    def __init__(self, node: ArrayNode) -> None:
        self._node = node
        super().__init__(_materialise_array(node))


class AoT(list[Table]):
    """Array-of-tables, e.g. ``[[products]]`` repeated.

    Subclass of :class:`list`; reads are wired up, mutation is not yet.
    """

    __slots__ = ()

    def __init__(self, tables: list[Table]) -> None:
        super().__init__(tables)


# ---------------------------------------------------------------------------
# View / aggregator
# ---------------------------------------------------------------------------


class _DocumentView:
    """Computes logical structure on demand from the CST."""

    __slots__ = ("_node",)

    def __init__(self, node: DocumentNode) -> None:
        self._node = node

    def _direct_sections(self, path: tuple[str, ...]) -> list[SectionNode]:
        out: list[SectionNode] = []
        for sec in self._node.sections:
            if path == ():
                if sec.header is None:
                    out.append(sec)
            else:
                hdr = sec.header
                if hdr is not None and hdr.kind == "table" and hdr.key.path == path:
                    out.append(sec)
        return out

    def _aot_sections(self, path: tuple[str, ...]) -> list[SectionNode]:
        return [
            sec
            for sec in self._node.sections
            if sec.header is not None
            and sec.header.kind == "array"
            and sec.header.key.path == path
        ]

    def _child_table_paths(self, path: tuple[str, ...]) -> list[tuple[str, ...]]:
        seen: dict[str, None] = {}
        for sec in self._node.sections:
            hdr = sec.header
            if hdr is None:
                continue
            hpath = hdr.key.path
            if len(hpath) > len(path) and hpath[: len(path)] == path:
                seen.setdefault(hpath[len(path)], None)
        return [(*path, k) for k in seen]

    def iter_table(
        self,
        path: tuple[str, ...],
        pinned: list[SectionNode] | None = None,
    ) -> Iterator[tuple[str, TomlValue]]:
        emitted: set[str] = set()
        sections = pinned if pinned is not None else self._direct_sections(path)
        for sec in sections:
            yield from self._iter_section_entries(sec, emitted)
        if pinned is not None:
            return
        for child_path in self._child_table_paths(path):
            name = child_path[-1]
            if name in emitted:
                continue
            emitted.add(name)
            aot_secs = self._aot_sections(child_path)
            if aot_secs:
                tables: list[Table] = [
                    _StdTable(self, child_path, sections=[sec]) for sec in aot_secs
                ]
                yield name, AoT(tables)
            else:
                yield name, _StdTable(self, child_path)

    def _iter_section_entries(
        self,
        section: SectionNode,
        emitted: set[str],
    ) -> Iterator[tuple[str, TomlValue]]:
        order: list[str] = []
        groups: dict[str, list[KeyValueNode]] = {}
        for entry in section.entries:
            head = entry.key.path[0]
            if head not in groups:
                groups[head] = []
                order.append(head)
            groups[head].append(entry)
        for head in order:
            if head in emitted:
                continue
            emitted.add(head)
            kvs = groups[head]
            if len(kvs) == 1 and len(kvs[0].key.path) == 1:
                yield head, _value_for(kvs[0].value)
            else:
                yield head, _DottedKvSubTable(kvs, depth=1)


class _DottedKvSubTable(Table):
    """Synthetic table aggregating dotted-key entries from a section."""

    __slots__ = ("_depth", "_entries")

    def __init__(self, entries: list[KeyValueNode], *, depth: int) -> None:
        self._entries = entries
        self._depth = depth

    @override
    def _items(self) -> Iterator[tuple[str, TomlValue]]:
        order: list[str] = []
        groups: dict[str, list[KeyValueNode]] = {}
        terminals: dict[str, KeyValueNode] = {}
        for entry in self._entries:
            path = entry.key.path
            if len(path) == self._depth:
                terminals[path[-1]] = entry
                if path[-1] not in order:
                    order.append(path[-1])
                continue
            head = path[self._depth]
            if head not in groups:
                groups[head] = []
                if head not in order:
                    order.append(head)
            groups[head].append(entry)
        for head in order:
            if head in terminals:
                yield head, _value_for(terminals[head].value)
            else:
                yield head, _DottedKvSubTable(groups[head], depth=self._depth + 1)


__all__ = [
    "AoT",
    "Array",
    "Document",
    "Scalar",
    "Table",
    "TomlValue",
]
