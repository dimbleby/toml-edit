"""Logical Document/Table/Array/AoT view over a CST.

This module exposes the public mapping/sequence types that users
interact with. It implements both the read path and the structural
mutation API on top of the physical CST defined in :mod:`tomle._nodes`.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from datetime import date, datetime, time
from typing import TYPE_CHECKING, SupportsIndex, TypeAlias, overload

from typing_extensions import override

from tomle._errors import TOMLEditError
from tomle._nodes import (
    ArrayNode,
    BoolNode,
    CommentNode,
    DateTimeNode,
    FloatNode,
    InlineTableEntry,
    InlineTableNode,
    IntegerNode,
    Key,
    KeyValueNode,
    NewlineNode,
    SectionNode,
    StringNode,
    TableHeaderNode,
    Trivia,
    WhitespaceNode,
)
from tomle._synthesise import make_key_part, make_keyvalue_node, value_to_node

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from typing import Self

    from tomle._nodes import (
        ArrayItem,
        DocumentNode,
        TriviaPiece,
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
        last.newline = NewlineNode("\n")


def _strip_comment_marker(text: str) -> str:
    """``"# foo "`` → ``"foo"``.

    Removes a leading ``#``, an optional single space, and trailing
    horizontal whitespace.
    """
    text = text.removeprefix("#")
    text = text.removeprefix(" ")
    return text.rstrip(" \t")


def _format_comment(text: str) -> str:
    """Format user text as the payload for a :class:`CommentNode`.

    Adds a leading ``#`` plus a space (unless the user already supplied
    one). Raises :class:`TOMLEditError` if ``text`` contains any line
    terminator, since comments are single-line by definition.
    """
    if "\n" in text or "\r" in text:
        msg = "comment text must not contain a line terminator"
        raise TOMLEditError(msg)
    if text.startswith("#"):
        return text
    if text == "":
        return "#"
    return "# " + text


def _trivia_ends_with_space(trivia: Trivia) -> bool:
    if not trivia.pieces:
        return False
    last = trivia.pieces[-1]
    return isinstance(last, WhitespaceNode) and last.text.endswith((" ", "\t"))


def _kv_indent(kv: KeyValueNode) -> str:
    """Indent (run of spaces/tabs) the entry's source line started with."""
    text = kv.leading.render()
    nl = text.rfind("\n")
    candidate = text[nl + 1 :] if nl >= 0 else text
    if all(c in " \t" for c in candidate):
        return candidate
    return ""


def _build_promoted_section(
    path: tuple[str, ...],
    inline: InlineTableNode,
    source_kv: KeyValueNode,
) -> SectionNode:
    """Build a ``[path]`` section containing ``inline``'s entries.

    Comments that lived above the original inline-table KV (its
    ``leading`` trivia) and any inline EOL comment are carried over to
    the new header so authoring intent is preserved.
    """
    parts = [make_key_part(p) for p in path]
    seps = ["."] * (len(parts) - 1)
    header = TableHeaderNode(
        leading=Trivia(list(source_kv.leading.pieces)),
        kind="table",
        inner_pre=Trivia(),
        key=Key(parts=parts, separators=seps),
        inner_post=Trivia(),
        trailing=Trivia(list(source_kv.trailing.pieces)),
        trailing_comment=source_kv.trailing_comment,
        newline=NewlineNode("\n"),
    )
    section = SectionNode(header=header, entries=[])
    for entry in inline.entries:
        section.entries.append(
            KeyValueNode(
                leading=Trivia(),
                key=entry.key,
                pre_eq=Trivia([WhitespaceNode(" ")]),
                post_eq=Trivia([WhitespaceNode(" ")]),
                value=entry.value,
                trailing=Trivia(),
                trailing_comment=None,
                newline=NewlineNode("\n"),
            ),
        )
    return section


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
        msg = "this table flavour does not support mutation"
        raise TOMLEditError(msg)

    def _delete_value(self, key: str) -> None:  # noqa: ARG002, pragma: no cover
        msg = "this table flavour does not support mutation"
        raise TOMLEditError(msg)

    @override
    def __repr__(self) -> str:
        body = ", ".join(f"{k!r}: {v!r}" for k, v in self._items())
        return f"{type(self).__name__}({{{body}}})"

    # ------------------------------------------------------------------
    # Comment API (default raises; concrete subclasses override).
    # ------------------------------------------------------------------

    def comment(self, key: str) -> str | None:  # noqa: ARG002
        """Return the end-of-line comment text bound to ``key``.

        The leading ``#`` and a single separating space are stripped.
        Returns ``None`` when the key has no inline comment.
        """
        msg = "this table flavour does not support the comment API"
        raise TOMLEditError(msg)

    def set_comment(self, key: str, text: str | None) -> None:  # noqa: ARG002
        """Set or remove the end-of-line comment bound to ``key``.

        Pass ``None`` to delete the comment. The supplied ``text`` may
        omit the leading ``#``; toml-edit will add it.
        """
        msg = "this table flavour does not support the comment API"
        raise TOMLEditError(msg)

    def leading_comments(self, key: str) -> list[str]:  # noqa: ARG002
        """Return the contiguous comment block immediately above ``key``."""
        msg = "this table flavour does not support the comment API"
        raise TOMLEditError(msg)

    def set_leading_comments(self, key: str, lines: Sequence[str]) -> None:  # noqa: ARG002
        """Replace the contiguous comment block immediately above ``key``."""
        msg = "this table flavour does not support the comment API"
        raise TOMLEditError(msg)

    def promote_inline(self, key: str) -> Table:  # noqa: ARG002
        """Promote an inline-table-valued ``key`` to a standard table.

        After promotion the entry is rendered as a separate
        ``[parent.key]`` section, allowing comments and dotted-key
        expansions on its members.
        """
        msg = "this table flavour does not support inline-table promotion"
        raise TOMLEditError(msg)


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
                        f"cannot delete dotted-key entry {key!r} from "
                        "inline table"
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
            if len(path) <= self._depth + 1:
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

    __slots__ = ("_doc_view", "_extra_kvs", "_owned_scope", "_path", "_pinned_sections")

    def __init__(
        self,
        doc_view: _DocumentView,
        path: tuple[str, ...],
        *,
        sections: list[SectionNode] | None = None,
        owned_scope: list[SectionNode] | None = None,
        extra_kvs: list[tuple[tuple[str, ...], KeyValueNode]] | None = None,
    ) -> None:
        self._doc_view = doc_view
        self._path = path
        self._pinned_sections = sections
        self._owned_scope = owned_scope
        self._extra_kvs = extra_kvs

    @override
    def _items(self) -> Iterator[tuple[str, TomlValue]]:
        return self._doc_view.iter_table(
            self._path,
            pinned_sections=self._pinned_sections,
            owned_scope=self._owned_scope,
            extra_kvs=self._extra_kvs,
        )

    def _direct_sections(self) -> list[SectionNode]:
        if self._pinned_sections is not None:
            return self._pinned_sections
        if self._owned_scope is not None:
            path = self._path
            return [
                s
                for s in self._owned_scope
                if s.header is not None
                and s.header.kind == "table"
                and s.header.key.path == path
            ]
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
        # Special case: assigning an AoT (or list of dicts targeted as AoT)
        # is a *structural* edit, not a value assignment.
        if isinstance(value, AoT):
            self._set_aot_value(key, value)
            return

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

    def _set_aot_value(self, key: str, value: AoT) -> None:
        """Assign a (possibly cross-document) AoT to ``key``.

        Each source entry's ``[[..]]`` section is deep-cloned and its
        header path rewritten to ``(*self._path, key)``. New sections
        are appended to the document.
        """
        kind, _ = self._classify(key)
        if kind != "absent":
            msg = (
                f"cannot assign array-of-tables to {key!r}: name is "
                f"already in use ({kind}). Remove it first."
            )
            raise TOMLEditError(msg)
        new_path = (*self._path, key)
        new_parts = [make_key_part(p) for p in new_path]
        new_seps = ["."] * (len(new_parts) - 1)
        # Source sections to clone, in source-document order:
        src_own = value._own_sections()  # noqa: SLF001
        doc_node = self._doc_view._node  # noqa: SLF001
        for src_sec in src_own:
            cloned = deepcopy(src_sec)
            assert cloned.header is not None
            cloned.header.key = Key(parts=list(new_parts), separators=list(new_seps))
            doc_node.sections.append(cloned)

    @override
    def _delete_value(self, key: str) -> None:
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
            f"cannot delete {key!r}: deleting child tables and "
            "dotted-key subtrees is not supported."
        )
        raise TOMLEditError(msg)

    def _ensure_section(self) -> SectionNode:
        """Create an implicit pre-header section for the document root.

        Only valid for the top-level Document; sub-tables created via
        ``__setitem__`` raise above.
        """
        if self._path != ():  # pragma: no cover - defensive
            msg = (
                f"no [{'.'.join(self._path)}] section exists; creating "
                "new sub-tables via assignment is not supported."
            )
            raise TOMLEditError(msg)
        doc_node = self._doc_view._node  # noqa: SLF001
        new_sec = SectionNode(header=None, entries=[])
        doc_node.sections.insert(0, new_sec)
        return new_sec

    # ------------------------------------------------------------------
    # Comment API
    # ------------------------------------------------------------------

    def _find_direct_kv(self, key: str) -> tuple[SectionNode, KeyValueNode]:
        """Return the section + KV node binding ``key`` as a single segment.

        Raises :class:`KeyError` when ``key`` is absent or the binding is
        a child table / dotted-key prefix rather than a simple
        ``key = value`` line.
        """
        for sec in self._direct_sections():
            for kv in sec.entries:
                if (
                    kv.key.path
                    and len(kv.key.path) == 1
                    and kv.key.path[0] == key
                ):
                    return sec, kv
        raise KeyError(key)

    @override
    def comment(self, key: str) -> str | None:
        _, kv = self._find_direct_kv(key)
        if kv.trailing_comment is None:
            return None
        return _strip_comment_marker(kv.trailing_comment.text)

    @override
    def set_comment(self, key: str, text: str | None) -> None:
        _, kv = self._find_direct_kv(key)
        if text is None:
            kv.trailing_comment = None
            return
        if not _trivia_ends_with_space(kv.trailing):
            kv.trailing.pieces.append(WhitespaceNode(" "))
        kv.trailing_comment = CommentNode(text=_format_comment(text))
        if kv.newline is None:
            kv.newline = NewlineNode("\n")

    @override
    def leading_comments(self, key: str) -> list[str]:
        _, kv = self._find_direct_kv(key)
        return [
            _strip_comment_marker(p.text)
            for p in kv.leading.pieces
            if isinstance(p, CommentNode)
        ]

    @override
    def set_leading_comments(self, key: str, lines: Sequence[str]) -> None:
        _, kv = self._find_direct_kv(key)
        indent = _kv_indent(kv)
        # Preserve the trailing whitespace run that anchored the key
        # to its column (everything after the last non-whitespace
        # piece of the existing leading trivia).
        tail: list[TriviaPiece] = []
        for piece in reversed(kv.leading.pieces):
            if isinstance(piece, WhitespaceNode):
                tail.insert(0, piece)
            else:
                break
        new_pieces: list[TriviaPiece] = []
        for line in lines:
            if indent:
                new_pieces.append(WhitespaceNode(indent))
            new_pieces.append(CommentNode(text=_format_comment(line)))
            new_pieces.append(NewlineNode("\n"))
        kv.leading.pieces = new_pieces + tail

    @override
    def promote_inline(self, key: str) -> Table:
        sec, kv = self._find_direct_kv(key)
        if not isinstance(kv.value, InlineTableNode):
            msg = f"{key!r} is not an inline table; nothing to promote"
            raise TOMLEditError(msg)
        inline = kv.value
        child_path = (*self._path, key)
        # Refuse if a [child_path] section already exists in the document
        # (defensive: the parser blocks any source where this would arise,
        # and the mutation API also refuses to create the conflicting
        # state, so this branch only fires under direct CST manipulation).
        for existing in self._doc_view._node.sections:  # noqa: SLF001
            hdr = existing.header
            if hdr is not None and hdr.key.path == child_path:  # pragma: no cover
                msg = (
                    f"cannot promote {key!r}: a "
                    f"[{'.'.join(child_path)}] section already exists"
                )
                raise TOMLEditError(msg)
        new_sec = _build_promoted_section(child_path, inline, kv)
        # Remove the inline KV from its host section.
        sec.entries.remove(kv)
        # Insert the promoted section after the parent's last direct
        # section (or at end of document if the parent has none).
        sections = self._doc_view._node.sections  # noqa: SLF001
        parent_secs = self._direct_sections()
        if parent_secs:
            anchor = parent_secs[-1]
            anchor_idx = next(
                (i for i, s in enumerate(sections) if s is anchor),
                len(sections) - 1,
            )
            sections.insert(anchor_idx + 1, new_sec)
        else:
            sections.append(new_sec)
        return _StdTable(self._doc_view, child_path)


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

    Subclass of :class:`list`; supports basic mutation (append/insert
    of dict-shaped or :class:`Table` entries) by synthesizing fresh
    ``[[path]]`` sections in the underlying CST.
    """

    __slots__ = ("_doc_view", "_path")

    def __init__(
        self,
        doc_view: _DocumentView,
        path: tuple[str, ...],
        tables: list[Table],
    ) -> None:
        super().__init__(tables)
        self._doc_view = doc_view
        self._path = path

    # ------------------------------------------------------------------
    # CST <-> list synchronisation
    # ------------------------------------------------------------------

    def _own_sections(self) -> list[SectionNode]:
        """Sections that act as the [[path]] entry headers (in doc order)."""
        return [
            s
            for s in self._doc_view._node.sections  # noqa: SLF001
            if s.header is not None
            and s.header.kind == "array"
            and s.header.key.path == self._path
        ]

    def _resync(self) -> None:
        own = self._own_sections()
        list.clear(self)
        for s in own:
            owned = self._doc_view._aot_owned_range(s)  # noqa: SLF001
            list.append(
                self,
                _StdTable(
                    self._doc_view,
                    self._path,
                    sections=[s],
                    owned_scope=[s, *owned],
                ),
            )

    def _make_header_section(self) -> SectionNode:
        # Build a [[path]] header.
        parts = [make_key_part(p) for p in self._path]
        seps = ["."] * (len(parts) - 1)
        key = Key(parts=parts, separators=seps)
        header = TableHeaderNode(
            leading=Trivia(),
            kind="array",
            inner_pre=Trivia(),
            key=key,
            inner_post=Trivia(),
            trailing=Trivia(),
            trailing_comment=None,
            newline=NewlineNode("\n"),
        )
        return SectionNode(header=header, entries=[])

    def _populate_section(self, sec: SectionNode, value: object) -> None:
        """Fill ``sec`` with KV entries derived from ``value``.

        Accepts a plain dict, a :class:`Table`, or any
        :class:`collections.abc.Mapping`. Cross-document Tables are
        deep-cloned to satisfy the "no shared mutable state" rule.
        """
        if isinstance(value, Table):
            # Walk the source table's items; values are deep-cloned by
            # value_to_node so inline tables / arrays aren't aliased.
            for k, v in value.items():
                sec.entries.append(make_keyvalue_node(k, v))
            return
        if isinstance(value, Mapping):
            for k, v in value.items():
                if not isinstance(k, str):
                    msg = f"AoT entry keys must be strings, got {type(k).__name__}"
                    raise TOMLEditError(msg)
                sec.entries.append(make_keyvalue_node(k, v))
            return
        msg = (
            f"cannot append a value of type {type(value).__name__} to an "
            "array-of-tables; expected a dict or Table"
        )
        raise TOMLEditError(msg)

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    @override
    def append(self, value: Table | Mapping[str, object]) -> None:
        self._insert_at(len(self), value)

    @override
    def insert(
        self,
        index: SupportsIndex,
        value: Table | Mapping[str, object],
    ) -> None:
        self._insert_at(operator.index(index), value)

    @override
    def extend(
        self,
        values: Iterable[Table | Mapping[str, object]],
    ) -> None:
        for v in list(values):
            self._insert_at(len(self), v)

    def _insert_at(
        self,
        py_index: int,
        value: Table | Mapping[str, object],
    ) -> None:
        own = self._own_sections()
        n = len(own)
        if py_index < 0:
            py_index += n
        py_index = max(0, min(py_index, n))
        new_sec = self._make_header_section()
        self._populate_section(new_sec, value)
        sections = self._doc_view._node.sections  # noqa: SLF001
        if py_index == n:
            # Append: insert after the last [[path]] entry's owned range,
            # or at end of doc if no entries exist yet.
            if own:
                last = own[-1]
                owned = self._doc_view._aot_owned_range(last)  # noqa: SLF001
                tail = owned[-1] if owned else last
                tail_idx = _index_of(sections, tail)
                sections.insert(tail_idx + 1, new_sec)
            else:
                sections.append(new_sec)
        else:
            # Insert before the entry currently at py_index.
            target = own[py_index]
            target_idx = _index_of(sections, target)
            sections.insert(target_idx, new_sec)
        self._resync()

    @override
    def pop(self, index: SupportsIndex = -1) -> Table:
        own = self._own_sections()
        n = len(own)
        i = operator.index(index)
        if i < 0:
            i += n
        if i < 0 or i >= n:
            msg = "pop index out of range"
            raise IndexError(msg)
        target = own[i]
        owned = self._doc_view._aot_owned_range(target)  # noqa: SLF001
        sections = self._doc_view._node.sections  # noqa: SLF001
        # Remove the entry header AND every section it owned.
        to_remove = {id(target), *(id(s) for s in owned)}
        # Capture the popped Table view *before* mutating sections.
        popped = _StdTable(
            self._doc_view,
            self._path,
            sections=[target],
            owned_scope=[target, *owned],
        )
        self._doc_view._node.sections = [  # noqa: SLF001
            s for s in sections if id(s) not in to_remove
        ]
        self._resync()
        return popped

    @override
    def clear(self) -> None:
        own = self._own_sections()
        to_remove: set[int] = set()
        for s in own:
            to_remove.add(id(s))
            for sub in self._doc_view._aot_owned_range(s):  # noqa: SLF001
                to_remove.add(id(sub))
        sections = self._doc_view._node.sections  # noqa: SLF001
        self._doc_view._node.sections = [  # noqa: SLF001
            s for s in sections if id(s) not in to_remove
        ]
        self._resync()

    @override
    def __delitem__(self, index: SupportsIndex | slice) -> None:
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            for i in sorted(indices, reverse=True):
                self.pop(i)
        else:
            self.pop(index)


def _index_of(sections: list[SectionNode], target: SectionNode) -> int:
    for i, s in enumerate(sections):
        if s is target:
            return i
    msg = "section not found in document (internal error)"
    raise RuntimeError(msg)


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

    def _aot_owned_range(self, aot_sec: SectionNode) -> list[SectionNode]:
        """Sections owned by this AoT entry.

        Owned = sections that come *after* ``aot_sec`` in document order
        and whose header path strictly extends this AoT's path. The range
        ends at the next [[same-path]] header or any other section that
        doesn't extend ``aot_sec``'s path.
        """
        if aot_sec.header is None:
            return []
        aot_path = aot_sec.header.key.path
        sections = self._node.sections
        i = -1
        for idx, candidate in enumerate(sections):
            if candidate is aot_sec:
                i = idx
                break
        if i < 0:
            return []
        owned: list[SectionNode] = []
        for j in range(i + 1, len(sections)):
            sec = sections[j]
            hdr = sec.header
            if hdr is None:
                # The synthetic root section appears only at index 0; safe to stop.
                break
            hpath = hdr.key.path
            if hdr.kind == "array" and hpath == aot_path:
                break  # next AoT entry of same path — terminate
            if len(hpath) > len(aot_path) and hpath[: len(aot_path)] == aot_path:
                owned.append(sec)
            else:
                # sibling or outer section — terminate ownership
                break
        return owned

    def iter_table(
        self,
        path: tuple[str, ...],
        *,
        pinned_sections: list[SectionNode] | None = None,
        owned_scope: list[SectionNode] | None = None,
        extra_kvs: list[tuple[tuple[str, ...], KeyValueNode]] | None = None,
    ) -> Iterator[tuple[str, TomlValue]]:
        # Pool of sections to consult for sub-tables / sub-AoTs.
        sub_pool: list[SectionNode] = (
            owned_scope if owned_scope is not None else list(self._node.sections)
        )

        # Direct sections that contribute KV entries at this exact path.
        direct_secs: list[SectionNode]
        if pinned_sections is not None:
            direct_secs = pinned_sections
        elif path == ():
            direct_secs = [
                s
                for s in (
                    owned_scope
                    if owned_scope is not None
                    else self._node.sections
                )
                if s.header is None
            ]
        else:
            direct_secs = [
                s
                for s in (
                    owned_scope
                    if owned_scope is not None
                    else self._node.sections
                )
                if s.header is not None
                and s.header.kind == "table"
                and s.header.key.path == path
            ]

        name_order: list[str] = []
        seen: set[str] = set()

        def _add(name: str) -> None:
            if name not in seen:
                seen.add(name)
                name_order.append(name)

        direct_kvs_by_head: dict[str, list[KeyValueNode]] = {}
        extras_by_head: dict[
            str, list[tuple[tuple[str, ...], KeyValueNode]]
        ] = {}
        aot_by_head: dict[str, list[SectionNode]] = {}
        sub_by_head: dict[str, list[SectionNode]] = {}

        for sec in direct_secs:
            for entry in sec.entries:
                head = entry.key.path[0]
                direct_kvs_by_head.setdefault(head, []).append(entry)
                _add(head)

        if extra_kvs:
            for rel_path, entry in extra_kvs:
                head = rel_path[0]
                extras_by_head.setdefault(head, []).append((rel_path, entry))
                _add(head)

        plen = len(path)
        for sec in sub_pool:
            hdr = sec.header
            if hdr is None:
                continue
            hpath = hdr.key.path
            if len(hpath) <= plen or hpath[:plen] != path:
                continue
            head = hpath[plen]
            if hdr.kind == "array" and hpath == (*path, head):
                aot_by_head.setdefault(head, []).append(sec)
            else:
                sub_by_head.setdefault(head, []).append(sec)
            _add(head)

        for head in name_order:
            direct_kvs = direct_kvs_by_head.get(head, [])
            extras = extras_by_head.get(head, [])
            aot_secs = aot_by_head.get(head, [])
            sub_secs = sub_by_head.get(head, [])

            if aot_secs:
                tables: list[Table] = []
                for s in aot_secs:
                    owned = self._aot_owned_range(s)
                    tables.append(
                        _StdTable(
                            self,
                            (*path, head),
                            sections=[s],
                            owned_scope=[s, *owned],
                        ),
                    )
                yield head, AoT(self, (*path, head), tables)
                continue

            # Split into "terminal" (binds a value at this name) and
            # "nested" (contributes to a sub-table at this name).
            terminal: KeyValueNode | None = None
            nested_kvs: list[KeyValueNode] = []
            nested_extras: list[tuple[tuple[str, ...], KeyValueNode]] = []
            for kv in direct_kvs:
                if len(kv.key.path) == 1:
                    if terminal is None:
                        terminal = kv
                else:
                    nested_kvs.append(kv)
            for rel_path, kv in extras:
                if len(rel_path) == 1:
                    if terminal is None:
                        terminal = kv
                else:
                    nested_extras.append((rel_path, kv))

            if (
                terminal is not None
                and not nested_kvs
                and not nested_extras
                and not sub_secs
            ):
                yield head, _value_for(terminal.value)
                continue

            if not sub_secs and not nested_extras:
                # Pure dotted from this section level.
                yield head, _DottedKvSubTable(nested_kvs, depth=1)
                continue

            # Merged view at path + (head,): combines sub-section content with
            # any dotted KVs from this section and any ancestor-dotted extras.
            child_extras: list[tuple[tuple[str, ...], KeyValueNode]] = [
                (kv.key.path[1:], kv) for kv in nested_kvs
            ]
            child_extras.extend((rel_path[1:], entry) for rel_path, entry in nested_extras)

            if owned_scope is not None:
                child_path = (*path, head)
                cplen = len(child_path)
                child_owned = [
                    s
                    for s in owned_scope
                    if s.header is not None
                    and len(s.header.key.path) >= cplen
                    and s.header.key.path[:cplen] == child_path
                ]
                yield head, _StdTable(
                    self,
                    child_path,
                    owned_scope=child_owned,
                    extra_kvs=child_extras or None,
                )
            else:
                yield head, _StdTable(
                    self,
                    (*path, head),
                    extra_kvs=child_extras or None,
                )


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
            if len(path) == self._depth + 1:
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
