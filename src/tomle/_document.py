"""Logical Document/Table/Array/AoT view over a CST.

This module exposes the public mapping/sequence types that users
interact with. The read path is implemented here; mutation will be
added in a follow-up phase (see plan.md).
"""

from __future__ import annotations

import operator
from collections.abc import MutableMapping
from copy import deepcopy
from datetime import date, datetime, time
from typing import TYPE_CHECKING, SupportsIndex, TypeAlias, overload

from typing_extensions import override

from tomle._errors import TOMLEditError
from tomle._nodes import (
    ArrayNode,
    BoolNode,
    DateTimeNode,
    FloatNode,
    InlineTableEntry,
    InlineTableNode,
    IntegerNode,
    SectionNode,
    StringNode,
    Trivia,
    WhitespaceNode,
)
from tomle._synthesise import make_keyvalue_node, value_to_node

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from typing import Self

    from tomle._nodes import (
        ArrayItem,
        DocumentNode,
        Key,
        KeyValueNode,
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


def _detect_indent(section: SectionNode) -> str:
    """Return the leading-whitespace indent used by the section's last entry."""
    if not section.entries:
        return ""
    last = section.entries[-1]
    text = last.leading.render()
    # Take everything after the final newline (the line's indent).
    nl = text.rfind("\n")
    candidate = text[nl + 1 :] if nl >= 0 else text
    if all(c in " \t" for c in candidate):
        return candidate
    return ""


def _ensure_trailing_newline(section: SectionNode) -> None:
    """Make sure the section's last entry ends with a newline.

    A parsed file's final entry may lack a newline at EOF. Before we
    append a sibling we have to terminate the previous line so the
    output is still well-formed.
    """
    if not section.entries:
        return
    last = section.entries[-1]
    if last.newline is None:
        from tomle._nodes import NewlineNode  # noqa: PLC0415

        last.newline = NewlineNode("\n")


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
    def __setitem__(self, key: str, value: object) -> None:
        self._set_value(key, value)

    @override
    def __delitem__(self, key: str) -> None:
        self._delete_value(key)

    # Subclasses override these.
    def _set_value(self, key: str, value: object) -> None:  # noqa: ARG002, pragma: no cover
        raise TOMLEditError(
            "this table flavour does not support mutation in this build",
        )

    def _delete_value(self, key: str) -> None:  # noqa: ARG002, pragma: no cover
        raise TOMLEditError(
            "this table flavour does not support mutation in this build",
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

    def _find_entry(self, key: str) -> InlineTableEntry | None:
        for entry in self._node.entries:
            if len(entry.key.path) == 1 and entry.key.path[0] == key:
                return entry
        return None

    @override
    def _set_value(self, key: str, value: object) -> None:
        # Reject conflict with dotted entries.
        for entry in self._node.entries:
            if entry.key.path[0] == key and len(entry.key.path) > 1:
                msg = (
                    f"cannot assign to {key!r} inside an inline table: "
                    "conflicts with an existing dotted-key entry."
                )
                raise TOMLEditError(msg)
        existing = self._find_entry(key)
        if existing is not None:
            existing.value = value_to_node(value)
            return
        # Append a new entry, fixing up the previous last entry's comma.
        new_entry = InlineTableEntry(
            leading=Trivia([WhitespaceNode(" ")]),
            key=_make_simple_key_for_inline(key),
            pre_eq=Trivia([WhitespaceNode(" ")]),
            post_eq=Trivia([WhitespaceNode(" ")]),
            value=value_to_node(value),
            trailing=Trivia(),
            has_comma=False,
            post_comma_trivia=Trivia(),
        )
        if self._node.entries:
            prev = self._node.entries[-1]
            if not prev.has_comma:
                prev.has_comma = True
                prev.post_comma_trivia = Trivia()
        else:
            # Empty inline table: drop any whitespace-only final_trivia.
            self._node.final_trivia = Trivia([WhitespaceNode(" ")])
        self._node.entries.append(new_entry)

    @override
    def _delete_value(self, key: str) -> None:
        existing = self._find_entry(key)
        if existing is None:
            # might be dotted-only → unsupported / KeyError per semantics
            for entry in self._node.entries:
                if entry.key.path[0] == key:
                    msg = (
                        f"cannot delete dotted-key entry {key!r} from inline "
                        "table in this build."
                    )
                    raise TOMLEditError(msg)
            raise KeyError(key)
        idx = self._node.entries.index(existing)
        was_last = idx == len(self._node.entries) - 1
        self._node.entries.pop(idx)
        if was_last and self._node.entries:
            new_last = self._node.entries[-1]
            new_last.has_comma = False
            new_last.post_comma_trivia = Trivia()
        if not self._node.entries:
            self._node.final_trivia = Trivia()


def _make_simple_key_for_inline(name: str) -> Key:
    from tomle._synthesise import make_simple_key  # noqa: PLC0415

    return make_simple_key(name)


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

    def _direct_sections(self) -> list[SectionNode]:
        if self._pinned_sections is not None:
            return self._pinned_sections
        return self._doc_view._direct_sections(self._path)  # noqa: SLF001

    def _classify(self, key: str) -> tuple[str, object]:
        """Classify a key for mutation purposes.

        Returns one of:
            ("direct", KeyValueNode)         - a single-part scalar/value entry
            ("dotted", None)                 - dotted-key prefix (e.g. b.c=...)
            ("table", None)                  - child standard table [self.path.key]
            ("aot", None)                    - child AoT [[self.path.key]]
            ("absent", None)
        """
        for sec in self._direct_sections():
            for kv in sec.entries:
                if kv.key.path[0] == key:
                    if len(kv.key.path) == 1:
                        return ("direct", kv)
                    return ("dotted", None)
        child = (*self._path, key)
        if self._doc_view._aot_sections(child):  # noqa: SLF001
            return ("aot", None)
        for sec in self._doc_view._node.sections:  # noqa: SLF001
            hdr = sec.header
            if hdr is not None and hdr.kind == "table":
                hpath = hdr.key.path
                if len(hpath) >= len(child) and hpath[: len(child)] == child:
                    return ("table", None)
        return ("absent", None)

    @override
    def _set_value(self, key: str, value: object) -> None:
        from tomle._nodes import KeyValueNode  # noqa: PLC0415

        kind, payload = self._classify(key)
        if kind == "direct":
            assert isinstance(payload, KeyValueNode)
            payload.value = value_to_node(value)
            return
        if kind in ("dotted", "table", "aot"):
            msg = (
                f"cannot assign to {key!r}: existing structure conflicts "
                f"({kind}). Mutate the nested table or remove it first."
            )
            raise TOMLEditError(msg)
        sections = self._direct_sections()
        if not sections:
            sections = [self._ensure_section()]
        target = sections[-1]
        indent = _detect_indent(target)
        new_kv = make_keyvalue_node(key, value, indent=indent)
        _ensure_trailing_newline(target)
        target.entries.append(new_kv)

    @override
    def _delete_value(self, key: str) -> None:
        from tomle._nodes import KeyValueNode  # noqa: PLC0415

        kind, payload = self._classify(key)
        if kind == "direct":
            assert isinstance(payload, KeyValueNode)
            for sec in self._direct_sections():
                if payload in sec.entries:
                    sec.entries.remove(payload)
                    return
            raise KeyError(key)
        if kind == "absent":
            raise KeyError(key)
        msg = (
            f"cannot delete {key!r}: deleting child tables / dotted-key "
            "subtrees is not yet supported in this build."
        )
        raise TOMLEditError(msg)

    def _ensure_section(self) -> SectionNode:
        """Create an implicit pre-header section for the document root.

        Only valid for the top-level Document; sub-tables created via
        ``__setitem__`` are deferred (raise above).
        """
        if self._path != ():  # pragma: no cover - defensive
            msg = (
                f"no [{'.'.join(self._path)}] section exists; creating "
                "new sub-tables via assignment is not yet supported."
            )
            raise TOMLEditError(msg)
        doc_node = self._doc_view._node  # noqa: SLF001
        new_sec = SectionNode(header=None, entries=[])
        doc_node.sections.insert(0, new_sec)
        return new_sec


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

    Every standard list mutator is overridden so the underlying CST
    stays in sync. Existing handles to nested ``Array``/``Table`` values
    that were *not* removed remain valid; handles to removed/replaced
    elements become detached.
    """

    __slots__ = ("_node",)

    def __init__(self, node: ArrayNode) -> None:
        self._node = node
        super().__init__(_materialise_array(node))

    # ------------------------------------------------------------------
    # CST <-> list synchronisation helpers
    # ------------------------------------------------------------------

    def _resync(self) -> None:
        """Rebuild the public list from the CST after a structural change."""
        list.clear(self)
        list.extend(self, _materialise_array(self._node))

    @staticmethod
    def _make_item(value: TomlValue, *, with_comma: bool) -> ArrayItem:
        from tomle._nodes import ArrayItem  # noqa: PLC0415

        return ArrayItem(
            leading=Trivia([WhitespaceNode(" ")]),
            value=value_to_node(value),
            trailing=Trivia(),
            has_comma=with_comma,
            post_comma_trivia=(
                Trivia([WhitespaceNode(" ")]) if with_comma else Trivia()
            ),
        )

    def _rebuild_separators(self) -> None:
        """Normalise commas/spacing across the underlying ArrayItems."""
        items = self._node.items
        n = len(items)
        for i, item in enumerate(items):
            if i < n - 1:
                if not item.has_comma:
                    item.has_comma = True
                if not item.post_comma_trivia.pieces:
                    item.post_comma_trivia = Trivia([WhitespaceNode(" ")])
            else:
                # Last item: drop trailing comma we synthesized.
                if item.has_comma and not item.post_comma_trivia.pieces:
                    item.has_comma = False
        if not items:
            self._node.final_trivia = Trivia()
        elif not self._node.final_trivia.pieces:
            self._node.final_trivia = Trivia([WhitespaceNode(" ")])

    # ------------------------------------------------------------------
    # Mutators (override every one)
    # ------------------------------------------------------------------

    @override
    def append(self, value: TomlValue) -> None:
        self._node.items.append(self._make_item(value, with_comma=False))
        self._rebuild_separators()
        self._resync()

    @override
    def extend(self, values: Iterable[TomlValue]) -> None:
        for v in list(values):
            self._node.items.append(self._make_item(v, with_comma=False))
        self._rebuild_separators()
        self._resync()

    @override
    def insert(self, index: SupportsIndex, value: TomlValue) -> None:
        self._node.items.insert(operator.index(index), self._make_item(value, with_comma=False))
        self._rebuild_separators()
        self._resync()

    @overload
    def __setitem__(self, index: SupportsIndex, value: TomlValue) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[TomlValue]) -> None: ...
    @override
    def __setitem__(
        self,
        index: SupportsIndex | slice,
        value: TomlValue | Iterable[TomlValue],
    ) -> None:
        if isinstance(index, slice):
            assert not isinstance(value, (str, bytes))
            new_items = [
                self._make_item(v, with_comma=False)
                for v in list(value)  # type: ignore[arg-type]
            ]
            self._node.items[index] = new_items
        else:
            i = operator.index(index)
            self._node.items[i].value = value_to_node(value)
        self._rebuild_separators()
        self._resync()

    @override
    def __delitem__(self, index: SupportsIndex | slice) -> None:
        if isinstance(index, slice):
            del self._node.items[index]
        else:
            del self._node.items[operator.index(index)]
        self._rebuild_separators()
        self._resync()

    @override
    def pop(self, index: SupportsIndex = -1) -> TomlValue:
        item = self._node.items.pop(operator.index(index))
        self._rebuild_separators()
        self._resync()
        return _value_for(item.value)

    @override
    def remove(self, value: TomlValue) -> None:
        idx = list.index(self, value)
        del self[idx]

    @override
    def clear(self) -> None:
        self._node.items.clear()
        self._rebuild_separators()
        self._resync()

    @override
    def reverse(self) -> None:
        self._node.items.reverse()
        self._rebuild_separators()
        self._resync()

    @override
    def sort(
        self,
        *,
        key: Callable[[TomlValue], object] | None = None,
        reverse: bool = False,
    ) -> None:
        pairs = list(zip(_materialise_array(self._node), self._node.items, strict=True))
        if key is None:
            pairs.sort(key=lambda p: p[0], reverse=reverse)  # type: ignore[arg-type, return-value]
        else:
            pairs.sort(key=lambda p: key(p[0]), reverse=reverse)  # type: ignore[arg-type, return-value]
        self._node.items[:] = [item for _, item in pairs]
        self._rebuild_separators()
        self._resync()

    @override
    def __iadd__(self, values: Iterable[TomlValue]) -> Self:  # type: ignore[override]
        self.extend(values)
        return self

    @override
    def __imul__(self, count: SupportsIndex) -> Self:
        n = operator.index(count)
        if n <= 0:
            self.clear()
        else:
            base = list(self._node.items)
            for _ in range(n - 1):
                self._node.items.extend(deepcopy(item) for item in base)
            self._rebuild_separators()
            self._resync()
        return self


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
