"""Logical Document/Table/Array/AoT view over a CST.

This module exposes the public mapping/sequence types that users
interact with. It implements both the read path and the structural
mutation API on top of the physical CST defined in :mod:`tomlrt._nodes`.
"""

from __future__ import annotations

import operator
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from copy import deepcopy
from datetime import date, datetime, time
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    SupportsIndex,
    TypeAlias,
    TypeVar,
    overload,
)

if sys.version_info >= (3, 11):
    from typing import assert_never
else:
    from typing_extensions import assert_never

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tomlrt._comment_views import (
    _ArrayCommentsView,
    _ArrayLeadingCommentsView,
    _TableCommentsView,
    _TableLeadingCommentsView,
)
from tomlrt._errors import TOMLError
from tomlrt._nodes import (
    ArrayNode,
    BoolNode,
    CommentNode,
    DateTimeNode,
    DocumentNode,
    FloatNode,
    InlineTableEntry,
    InlineTableNode,
    IntegerNode,
    KeyValueNode,
    NewlineNode,
    SectionNode,
    StringNode,
    Trivia,
    WhitespaceNode,
)
from tomlrt._section_build import (
    _aot_owned_sections,
    _apply_prior_leading,
    _build_promoted_aot_section,
    _build_promoted_section,
    _clone_aot_sections,
    _clone_table_sections,
    _insert_section_block,
    _make_dotted_key,
    _new_section,
    _parse_key_path,
    _rebase_aot_sections_inplace,
    _section_insert_index,
)
from tomlrt._separator import (
    _apply_separator_after_append,
    _apply_separator_style,
    _sample_separator_style,
    _SeparatorStyle,
    _snapshot_item_leadings,
    _write_item_leadings,
)
from tomlrt._synthesise import (
    make_keyvalue_node,
    make_simple_key,
    value_to_node,
)
from tomlrt._trivia import (
    _clone_trivia,
    _detect_indent,
    _detect_newline,
    _ensure_trailing_newline,
    _extract_trailing_comment_block,
    _first_gap_is_blank,
    _format_comment,
    _indent_after_last_newline,
    _normalise_newlines,
    _prepend_blank_line,
    _replace_trailing_comment_block,
    _scan_leading_comment_run,
    _set_eol_comment,
    _strip_comment_marker,
    _trivia_has_comment,
    _validate_comment_lines,
    _value_has_inner_comment,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, MutableMapping, Sequence

    from tomlrt._nodes import (
        TableHeaderNode,
    )

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    from tomlrt._nodes import (
        ArrayItem,
        TriviaPiece,
        ValueNode,
    )


Scalar: TypeAlias = str | int | float | bool | datetime | date | time
TomlValue: TypeAlias = "Scalar | Array | AoT | Table"

_MISSING: Any = object()
_T = TypeVar("_T")


def _to_plain(value: object) -> Any:
    """Recursively convert tomlrt views to plain Python data.

    Tables become ``dict``s, AoTs and Arrays become ``list``s, scalars
    are returned as-is. The result shares no mutable state with the
    underlying document and is safe to hand to consumers that expect
    real ``dict``/``list`` objects.
    """
    if isinstance(value, Table):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, AoT):
        return [_to_plain(t) for t in value]
    if isinstance(value, Array):
        return [_to_plain(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Read-side helpers
# ---------------------------------------------------------------------------


_SCALAR_NODE_TYPES = (StringNode, IntegerNode, FloatNode, BoolNode, DateTimeNode)


def _value_for(node: ValueNode) -> TomlValue:
    if isinstance(node, ArrayNode):
        return Array(node)
    if isinstance(node, InlineTableNode):
        return _InlineTable(node)
    if isinstance(node, _SCALAR_NODE_TYPES):
        return node.value
    assert_never(node)


def _materialise_array(node: ArrayNode) -> list[TomlValue]:
    return [_value_for(item.value) for item in node.items]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class SectionSpec(dict[str, Any]):
    """Tag telling ``__setitem__`` to install a ``[k]`` standard section.

    Produced by :meth:`Table.section`; used only at assignment sites::

        doc["tool"] = Table.section({"version": 1})  # [tool] section

    A plain ``dict`` assignment would instead produce an inline table
    (``tool = { version = 1 }``).
    """

    __slots__ = ()


class Table(dict[str, Any]):
    """A logical TOML table.

    All mapping flavours in tomlrt (top-level document, standard
    table, inline table, and the synthetic mappings spawned by dotted
    keys) inherit from :class:`Table`, which is itself a subclass of
    :class:`dict`. So values typed as ``Table`` cover every nested
    mapping you can encounter while walking a document, *and*
    ``isinstance(t, dict)`` is ``True`` and ``**t`` works.

    .. rubric:: Storage model

    A :class:`Table` is a *view* over the parsed concrete syntax
    tree (CST) — the physical tree of nodes that records every
    byte of the original document, including whitespace, comments,
    quote style and key order. Every mutation writes to the CST
    first and the dict storage is then refreshed from there. The
    CST is the single source of truth — :meth:`render` and every
    iteration ultimately read from it; the dict storage is a cache
    that mirrors the CST data and exists for two reasons:

    * fast ``dict``-style lookup, ``len``, ``in``, iteration, and
      ``**`` unpacking; and
    * stable object identity for nested containers, so that
      ``doc["foo"] is doc["foo"]``.

    Once a :class:`Table` is *detached* (see below) the CST link is
    severed and the dict storage takes over as the only source of
    truth for that orphan subtree.

    .. rubric:: Held references

    Held references behave like ordinary Python dict references:

    * If the binding goes away (``del doc['foo']``), the held
      ``Table`` is *orphaned*: its dict storage is intact and reads
      still work, but it is no longer connected to the document and
      mutations through it do not appear in :meth:`Document.render`.
    * Re-binding the path (``doc['foo'] = {...}`` or
      ``doc.set_table('foo', {...})``) installs a *fresh* ``Table``;
      held references to the old table are unaffected.

    .. rubric:: Live vs snapshot containers

    Assignment of a *container* value follows one rule: a container
    is attached to at most one CST location.

    * Assigning a fresh, unattached :class:`Array`,
      :meth:`Table.inline` result, or :class:`AoT` *attaches in
      place*: the user's reference becomes the live view at the
      destination, so later mutations through that reference show
      up in the document. ``doc[k] is myinline`` after the assign.
    * Assigning a container that is already attached somewhere
      (any document, including ``self``) deep-clones the source.
      The two slots are independent — mutations to one don't bleed
      into the other.
    * Plain :class:`dict` and :class:`list` values continue to be
      *snapshot* on assignment. Mutations to the original mapping
      / list after assignment are *not* reflected in the document.
      Use :meth:`Table.inline` or :class:`Array` to opt in to live
      semantics. Typed containers nested inside a plain dict / list
      still attach live recursively, even though the surrounding
      plain container is a snapshot.
    """

    __slots__ = ("_attached",)

    def __init__(self) -> None:
        super().__init__()
        self._attached = True

    # --- factories ---------------------------------------------------------

    @classmethod
    def section(
        cls,
        mapping: Mapping[str, object] | None = None,
    ) -> SectionSpec:
        """Return a spec that installs as a ``[k]`` standard section.

        Use from an assignment site: ``doc[k] = Table.section({...})``.
        The spec is a dict subclass; you can build it up further before
        assignment (``spec["sub"] = ...``). Nested dicts in the mapping
        remain inline unless they are themselves :meth:`section` specs.
        """
        spec = SectionSpec()
        if mapping is not None:
            spec.update(mapping)
        return spec

    @classmethod
    def inline(
        cls,
        mapping: Mapping[str, object] | None = None,
    ) -> _InlineTable:
        """Return a fresh inline table that *attaches live* on assignment.

        Use from an assignment site: ``doc[k] = Table.inline({...})``.
        Unlike a plain ``dict`` (which is snapshotted on assignment),
        the returned object becomes the *live* view at the assignment
        site: subsequent mutations through the original reference are
        reflected in the document, and ``doc[k] is the_inline`` after
        assignment.

        The inline table can be populated incrementally before
        assignment (``t = Table.inline(); t["a"] = 1; doc[k] = t``);
        all such mutations end up in the document.

        If the same inline-table object is assigned a second time
        (or after it has already been installed elsewhere), it is
        cloned: a single inline table is attached to at most one
        location in at most one document.
        """
        from tomlrt._synthesise import (  # noqa: PLC0415
            _mapping_to_inline_table_node,
        )

        inline = _InlineTable(_mapping_to_inline_table_node({}))
        inline._attached = False  # noqa: SLF001
        if mapping is not None:
            for k, v in mapping.items():
                inline[k] = v
        return inline

    # --- subclass hooks ----------------------------------------------------

    def _items(self) -> Iterator[tuple[str, TomlValue]]:  # pragma: no cover
        raise NotImplementedError

    def _set_value(  # pragma: no cover
        self,
        key: str,
        value: object,
    ) -> TomlValue | None:
        """Mutate the CST so ``key`` binds to ``value``.

        Returns the new dict-storage value for ``key`` if the
        implementation can compute it cheaply; otherwise returns
        ``None`` to ask :meth:`__setitem__` to fall back to the full
        :meth:`_refresh_key` walk. Returning the value short-circuits
        an O(N) re-scan of the document for a single-key update.
        """
        raise NotImplementedError

    def _delete_value(self, key: str) -> None:  # pragma: no cover
        raise NotImplementedError

    # --- dict-storage sync helpers -----------------------------------------

    def _populate(self) -> None:
        """Refill dict storage from the CST. Called from subclass __init__."""
        super().clear()
        for k, v in self._items():
            super().__setitem__(k, v)

    def _refresh_key(self, key: str) -> None:
        """Re-read ``key`` from the CST after a mutation.

        If ``key`` is no longer present in the CST, it is removed from
        dict storage. Identity for *other* keys is preserved.
        """
        for k, v in self._items():
            if k == key:
                # dict.__setitem__ on an existing key preserves position;
                # on a new key it appends. Both match CST behaviour.
                super().__setitem__(key, v)
                return
        if super().__contains__(key):
            super().__delitem__(key)

    # --- attachment / detachment ------------------------------------------

    def _detach(self, doc_node: DocumentNode | None = None) -> None:
        """Mark this table (and any nested containers) as orphaned.

        After detachment, mutations through this object only affect its
        own (now-isolated) state; CST writeback no longer reaches the
        original document. ``doc_node`` is supplied when the parent has
        already created an orphan :class:`DocumentNode` covering the
        whole detached subtree; subclasses with section-based storage
        (:class:`_StdTable`, :class:`AoT`) use it to keep nested
        structural mutations confined to that orphan view.
        """
        self._attached = False
        for v in self.values():
            if isinstance(v, (Table, AoT)):
                v._detach(doc_node)  # noqa: SLF001
            elif isinstance(v, Array):
                v._detach()  # noqa: SLF001

    # --- mutators ----------------------------------------------------------

    def _commit_value(self, key: str, value: object) -> None:
        """Plain-value write at ``key`` + dict-storage reconcile.

        Used by ``_install_flavoured`` to commit values that didn't
        match any structural-install flavour. Lifted out of
        ``__setitem__`` so single-segment installs can finish without
        recursing back through it (which would loop, since
        ``__setitem__`` now unconditionally delegates to
        ``_install_flavoured``).

        When ``value`` is an unattached :class:`_InlineTable` or
        :class:`Array`, the synthesise step (``value_to_node`` ->
        ``_attach_or_clone``) splices ``value._node`` into the
        destination KV verbatim and flips ``_attached``. We capture
        that pre-call and use it post-call to drop ``value`` itself
        into dict storage, so ``self[key] is value`` holds and later
        mutations through ``value`` flow through the same cached view.
        """
        will_live_attach = (
            isinstance(value, (_InlineTable, Array)) and not value._attached  # noqa: SLF001
        )
        new_v = self._set_value(key, value)
        if will_live_attach:
            dict.__setitem__(self, key, value)
        elif new_v is None:
            self._refresh_key(key)
        else:
            dict.__setitem__(self, key, new_v)

    @override
    def __setitem__(self, key: str, value: object) -> None:
        # Detach any container currently at ``key`` before we overwrite
        # it, so user-held references stop reflecting later document
        # edits. ``old is value`` is the augmented-assignment / self-
        # assignment case (``d[k] |= ...`` rebinds with the same
        # object): there's nothing to detach and nothing to re-install.
        #
        # We intentionally route writes through the install machinery
        # even when ``self`` is itself detached: a detached Table /
        # _InlineTable owns a private CST (a section list in an orphan
        # ``DocumentNode``, or just its own ``InlineTableNode``), and
        # writes need to land there so re-attach (deep-clone or
        # live-splice) sees them.
        if super().__contains__(key):
            old = super().__getitem__(key)
            if old is value:
                return
            if isinstance(old, (Table, AoT, Array)):
                old._detach()  # noqa: SLF001
        # All assignment flows -- structural and plain -- funnel through
        # ``_install_flavoured`` so each Table flavour can apply one
        # consistent dispatch policy. Plain values land in the
        # subclass's ``_commit_value`` fallback.
        self._install_flavoured((key,), value)

    def install(
        self,
        path: str | tuple[str, ...],
        value: object,
    ) -> Any:
        """Install ``value`` at ``path``, descending dotted segments.

        ``path`` accepts a dotted string (split on ``.``) or a tuple
        of literal segments (use the tuple form to express a segment
        containing a literal dot, e.g. ``("foo.bar",)``).

        ``value`` may be any of:

        * a spec from :meth:`Table.section` — installs a
          ``[...]`` standard section;
        * an :class:`AoT` built standalone (``AoT([{...}])``) — installs
          ``[[...]]`` array-of-tables entries;
        * an :class:`Array` built standalone (``Array([...],
          multiline=...)``) — installs an inline array with the
          requested layout;
        * any plain Python value (scalar, ``dict``, ``list``) —
          assigned at the leaf with ordinary ``__setitem__`` semantics
          (so a leaf ``dict`` becomes an inline table, a leaf ``list``
          becomes an inline array).

        Existing values at ``path`` (including sub-sections) are
        replaced. Implicit intermediate tables are left implicit, so
        ``install(("tool", "poetry"), Table.section({}))`` produces a
        single ``[tool.poetry]`` header, not a ``[tool]`` + nested.

        Returns the freshly-installed live view (:class:`Table`,
        :class:`AoT`, :class:`Array`) or the leaf value.
        """
        parts = _parse_key_path(path)
        # Reject install paths that would have to thread through an
        # array-of-tables before any CST mutation runs, so a rejected
        # call leaves the document untouched. AoT entries don't have
        # a single addressable child container; the user must address
        # a specific entry (``aot[i].install(...)``) instead. Other
        # non-table intermediates (scalars, inline tables) are caught
        # by their own dedicated error paths downstream.
        cur: Any = self
        walked = True
        for i, part in enumerate(parts[:-1]):
            if part not in cur:
                walked = False
                break
            nxt = cur[part]
            if isinstance(nxt, AoT):
                shown = ".".join(parts[: i + 1])
                msg = (
                    f"cannot install at {'.'.join(parts)!r}: {shown!r} is "
                    "an array-of-tables; address a specific entry instead "
                    f"(e.g. aot[i].install({parts[i + 1 :]!r}, ...))"
                )
                raise TOMLError(msg)
            if not isinstance(nxt, Table):
                walked = False
                break
            cur = nxt
        # Detach any container view currently at the leaf so user-held
        # references stop tracking the document after replacement.
        # ``__setitem__`` does this for single-key overwrites; do the
        # same here once we've located the leaf's parent.
        if walked and isinstance(cur, Table) and dict.__contains__(cur, parts[-1]):
            existing = dict.__getitem__(cur, parts[-1])
            if isinstance(existing, (Table, AoT, Array)) and existing is not value:
                existing._detach()  # noqa: SLF001
        self._install_flavoured(parts, value)
        leaf: Any = self
        for part in parts:
            leaf = leaf[part]
        return leaf

    def _install_flavoured(
        self,
        parts: tuple[str, ...],
        value: object,
    ) -> None:
        """Route a flavour-bearing value through the structural installers.

        Subclasses that support structural assignment (``_StdTable``,
        ``Document``) override this. The base implementation rejects
        ``SectionSpec`` / ``AoT`` because inline-style tables cannot
        hold ``[k]`` sections or ``[[k]]`` array-of-tables, and
        rejects multi-segment paths for the same reason. Standalone
        :class:`Array` and :class:`Table.inline` values are accepted
        and attach live (their ``_node`` is spliced into the
        destination so the user's reference remains the live view).
        """
        if isinstance(value, SectionSpec):
            msg = "cannot install a [section] inside an inline-style table"
            raise TOMLError(msg)
        if isinstance(value, AoT):
            msg = "cannot install an array-of-tables inside an inline-style table"
            raise TOMLError(msg)
        if isinstance(value, _StdTable):
            # An attached section-backed Table is the same kind of
            # "give me a [section] here" request as a SectionSpec; the
            # only difference is whether the user spelled the spec
            # themselves or copied an existing block. Refuse it for
            # the same reason — silently flattening it into the
            # inline host loses the [section] semantics.
            msg = "cannot install a [section]-style table inside an inline-style table"
            raise TOMLError(msg)
        if len(parts) > 1:
            path = ".".join(parts)
            msg = (
                f"cannot install at multi-segment path {path!r} inside "
                "an inline-style table"
            )
            raise TOMLError(msg)
        self._commit_value(parts[0], value)

    @override
    def __delitem__(self, key: str) -> None:
        if not super().__contains__(key):
            raise KeyError(key)
        old = super().__getitem__(key)
        if isinstance(old, (Table, AoT, Array)):
            old._detach()  # noqa: SLF001
        self._delete_value(key)
        if super().__contains__(key):
            super().__delitem__(key)

    @override
    def clear(self) -> None:
        for k in list(self):
            del self[k]

    @override
    def update(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if len(args) > 1:
            msg = f"update expected at most 1 positional argument, got {len(args)}"
            raise TypeError(msg)
        if args:
            other = args[0]
            if hasattr(other, "keys"):
                for k in other.keys():  # noqa: SIM118
                    self[k] = other[k]
            else:
                for k, v in other:
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    @override
    def setdefault(self, key: str, default: Any = None) -> Any:
        if super().__contains__(key):
            return super().__getitem__(key)
        self[key] = default
        return super().__getitem__(key)

    @override
    def __ior__(self, other: object) -> Self:  # type: ignore[override]
        if isinstance(other, Mapping):
            self.update(other)
        else:
            self.update(dict(other))  # type: ignore[call-overload]
        return self

    @override
    def copy(self) -> dict[str, Any]:
        """Return a shallow plain ``dict`` copy of this table."""
        return dict(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a deep, plain-Python copy of this table.

        Walks the table recursively and converts every nested
        :class:`Table` / :class:`AoT` / :class:`Array` view into an
        ordinary :class:`dict` / :class:`list`. The result shares no
        mutable state with the document and is safe to hand to
        consumers that expect real ``dict``/``list`` objects -- JSON
        encoders, ``fastjsonschema``, ``pydantic``, anything that
        does ``isinstance(x, dict)``, etc.

        Scalar values (strings, ints, floats, bools, datetimes) are
        returned as-is; they are immutable so aliasing is harmless.
        """
        return {k: _to_plain(v) for k, v in self.items()}

    @override
    def pop(self, key: str, default: object = _MISSING) -> Any:
        """Remove ``key`` and return its value, like :meth:`dict.pop`.

        For :class:`Table` / :class:`AoT` / :class:`Array` values, the
        returned object is *orphaned*: it keeps its own data but is no
        longer attached to the document. Use :meth:`to_dict` /
        :meth:`Array.to_list` first if you need a plain-Python deep
        copy.
        """
        try:
            value = super().__getitem__(key)
        except KeyError:
            if default is _MISSING:
                raise
            return default
        del self[key]
        return value

    @override
    def popitem(self) -> tuple[str, Any]:
        if not self:
            msg = "table is empty"
            raise KeyError(msg)
        key = next(reversed(self))
        return key, self.pop(key)

    @override
    def __repr__(self) -> str:
        body = ", ".join(f"{k!r}: {v!r}" for k, v in self.items())
        return f"{type(self).__name__}({{{body}}})"

    # ------------------------------------------------------------------
    # Typed accessors. These are convenience views over ``__getitem__``
    # that narrow the return type so callers can chain into the
    # comment/header API without writing ``cast`` or ``isinstance``.
    # ------------------------------------------------------------------

    def table(self, key: str) -> Table:
        """Return the table at ``key``, typed as :class:`Table`.

        ``key`` accepts a dotted path (e.g. ``"tool.poetry"``). Raises
        :class:`KeyError` if any segment is missing, or :class:`TypeError`
        if the destination is not a table.
        """
        return self._typed_lookup(key, Table)

    @overload
    def get_table(self, key: str) -> Table | None: ...
    @overload
    def get_table(self, key: str, default: _T) -> Table | _T: ...
    def get_table(self, key: str, default: object = None) -> object:
        """Like :meth:`table`, but returns ``default`` if ``key`` is missing.

        Wrong-type entries still raise :class:`TypeError`: a missing key
        is "no answer", but an entry that exists with the wrong shape is
        a real bug worth surfacing.
        """
        return self._typed_lookup(key, Table, default=default)

    def array(self, key: str) -> Array:
        """Return the array at ``key``, typed as :class:`Array`.

        ``key`` accepts a dotted path. Raises :class:`KeyError` if any
        segment is missing, or :class:`TypeError` if the destination is
        not an inline array.
        """
        return self._typed_lookup(key, Array)

    @overload
    def get_array(self, key: str) -> Array | None: ...
    @overload
    def get_array(self, key: str, default: _T) -> Array | _T: ...
    def get_array(self, key: str, default: object = None) -> object:
        """Like :meth:`array`, but returns ``default`` if ``key`` is missing.

        Wrong-type entries still raise :class:`TypeError`.
        """
        return self._typed_lookup(key, Array, default=default)

    def aot(self, key: str) -> AoT:
        """Return the array-of-tables at ``key``, typed as :class:`AoT`.

        ``key`` accepts a dotted path. Raises :class:`KeyError` if any
        segment is missing, or :class:`TypeError` if the destination is
        not an array of tables.
        """
        return self._typed_lookup(key, AoT)

    @overload
    def get_aot(self, key: str) -> AoT | None: ...
    @overload
    def get_aot(self, key: str, default: _T) -> AoT | _T: ...
    def get_aot(self, key: str, default: object = None) -> object:
        """Like :meth:`aot`, but returns ``default`` if ``key`` is missing.

        Wrong-type entries still raise :class:`TypeError`.
        """
        return self._typed_lookup(key, AoT, default=default)

    @overload
    def _typed_lookup(self, key: str, expected: type[_T]) -> _T: ...
    @overload
    def _typed_lookup(
        self,
        key: str,
        expected: type[_T],
        *,
        default: object,
    ) -> object: ...
    def _typed_lookup(
        self,
        key: str,
        expected: type[_T],
        *,
        default: object = _MISSING,
    ) -> object:
        """Shared implementation for ``table`` / ``array`` / ``aot`` and
        their ``get_*`` variants. Without ``default``, missing keys
        re-raise :class:`KeyError`; otherwise ``default`` is returned.
        Wrong-type entries always raise :class:`TypeError`.
        """
        try:
            value = self._lookup_path(key)
        except KeyError:
            if default is _MISSING:
                raise
            return default
        if not isinstance(value, expected):
            article = "an" if expected.__name__[:1] in "AaEeIiOoUu" else "a"
            msg = (
                f"{key!r} is a {type(value).__name__}, "
                f"not {article} {expected.__name__}"
            )
            raise TypeError(msg)
        return value

    def _lookup_path(self, key: str) -> TomlValue:
        parts = _parse_key_path(key)
        cur: TomlValue = self
        for i, part in enumerate(parts):
            if not isinstance(cur, Table):
                head = ".".join(parts[:i])
                msg = (
                    f"cannot descend into {head!r}: it is a "
                    f"{type(cur).__name__}, not a Table"
                )
                raise TypeError(msg)
            cur = cur[part]
        return cur

    # ------------------------------------------------------------------
    # Metadata side-channels (default raises; concrete subclasses override).
    # ------------------------------------------------------------------

    @property
    def comments(self) -> MutableMapping[str, str]:
        """Live mapping of ``key -> end-of-line comment text``.

        Only keys that currently carry a comment are present; assigning
        ``""`` or deleting a key removes its comment. Reads return the
        comment text without the leading ``#`` or surrounding whitespace.
        """
        msg = "this table flavour does not support the comment API"
        raise TOMLError(msg)

    @property
    def leading_comments(self) -> MutableMapping[str, tuple[str, ...]]:
        """Live mapping of ``key -> tuple of comment lines above it``.

        Only keys with a non-empty leading comment block are present.
        Assigning an empty tuple or deleting a key removes the block.
        """
        msg = "this table flavour does not support the comment API"
        raise TOMLError(msg)

    @property
    def header_comment(self) -> str | None:
        """End-of-line comment on this table's ``[name]`` / ``[[name]]`` line.

        ``None`` means the header has no trailing comment. Setting
        ``None`` or ``""`` removes any existing comment. Raises
        :class:`TOMLError` for the top-level :class:`Document`,
        for inline tables, and for any logical table that exists only
        through implicit parents (no physical header in source).

        For tables declared via multiple discontiguous ``[name]``
        sections, this refers to the *first* such header.
        """
        msg = "this table flavour does not support the header comment API"
        raise TOMLError(msg)

    @header_comment.setter
    def header_comment(self, value: str | None) -> None:  # noqa: ARG002
        msg = "this table flavour does not support the header comment API"
        raise TOMLError(msg)

    @header_comment.deleter
    def header_comment(self) -> None:
        msg = "this table flavour does not support the header comment API"
        raise TOMLError(msg)

    @property
    def header_leading_comments(self) -> tuple[str, ...]:
        """Comment lines immediately above this table's header.

        Returns the contiguous block of ``# ...`` lines ending right
        above the ``[name]`` / ``[[name]]`` line. Earlier blank-line
        separated comments are *not* included. Assigning an empty
        tuple removes the block. Raises like :attr:`header_comment`.
        """
        msg = "this table flavour does not support the header comment API"
        raise TOMLError(msg)

    @header_leading_comments.setter
    def header_leading_comments(self, value: Sequence[str]) -> None:  # noqa: ARG002
        msg = "this table flavour does not support the header comment API"
        raise TOMLError(msg)

    @header_leading_comments.deleter
    def header_leading_comments(self) -> None:
        msg = "this table flavour does not support the header comment API"
        raise TOMLError(msg)

    def promote_inline(self, key: str) -> Table:  # noqa: ARG002
        """Promote an inline-table-valued ``key`` to a standard table.

        After promotion the entry is rendered as a separate
        ``[parent.key]`` section, allowing comments and dotted-key
        expansions on its members.
        """
        msg = "this table flavour does not support inline-table promotion"
        raise TOMLError(msg)

    def promote_array(self, key: str) -> AoT:  # noqa: ARG002
        """Promote an array-of-inline-tables-valued ``key`` to an AoT.

        After promotion the entries are rendered as repeated
        ``[[parent.key]]`` sections, allowing comments and dotted-key
        expansions on each entry's members.
        """
        msg = "this table flavour does not support array-of-tables promotion"
        raise TOMLError(msg)

    def _install_section(
        self,
        parts: tuple[str, ...],  # noqa: ARG002
        value: Mapping[str, object] = MappingProxyType({}),  # noqa: ARG002
    ) -> Table:
        msg = (
            "cannot install a standard table here: this table flavour "
            "is not section-backed"
        )
        raise TOMLError(msg)

    def ensure_table(self, key: str | tuple[str, ...]) -> Table:
        """Return the table at ``key``, creating an empty one if absent.

        ``key`` accepts a dotted path as a string, or a tuple of
        literal segments (use the tuple form to express a segment
        containing a literal dot, e.g. ``("foo.bar",)``). If the
        destination already exists and is table-shaped (an explicit
        section, an implicit super-table, or an inline table), the
        existing live view is returned and no mutation occurs. Raises
        :class:`TOMLError` when the path names a non-table value.
        """
        parts = _parse_key_path(key)
        cur: Table = self
        for i, part in enumerate(parts):
            if part in cur:
                child = cur[part]
                if not isinstance(child, Table):
                    full = ".".join(parts[: i + 1])
                    msg = (
                        f"cannot ensure table at {full!r}: existing value is "
                        f"a {type(child).__name__}"
                    )
                    raise TOMLError(msg)
                cur = child
            else:
                return cur._install_section(parts[i:], {})  # noqa: SLF001
        return cur


class _InlineTable(Table):
    """Mapping view over an :class:`InlineTableNode`.

    Also acts as the :class:`_DottedHost` for any dotted-key views
    derived from its entries — the inline table itself owns all the
    state (node, separator style, ``=`` padding) those views need.
    """

    __slots__ = ("_eq_padding", "_node", "_style")

    def __init__(self, node: InlineTableNode) -> None:
        super().__init__()
        self._node = node
        self._style = _sample_separator_style(node.entries, node.final_trivia)
        # ``=``-padding is per-entry, not a separator concern. Sample
        # from the first existing entry; default to a single space.
        if node.entries:
            self._eq_padding: tuple[WhitespaceNode | None, WhitespaceNode | None] = (
                node.entries[0].pre_eq,
                node.entries[0].post_eq,
            )
        else:
            self._eq_padding = (WhitespaceNode(" "), WhitespaceNode(" "))
        self._populate()

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
                yield head, _DottedSubTable(depth=1, host=self, prefix=(head,))

    def _find_entry(self, key: str) -> InlineTableEntry | None:
        for entry in self._node.entries:
            if len(entry.key.path) == 1 and entry.key.path[0] == key:
                return entry
        return None

    def _make_entry(self, path: tuple[str, ...], value: object) -> InlineTableEntry:
        pre, post = self._eq_padding
        # ``pre``/``post`` are WhitespaceNode|None value objects; sharing
        # the ref across entries is safe because their ``text`` field is
        # never mutated in place.
        return InlineTableEntry(
            leading=Trivia(),
            key=_make_dotted_key(path) if len(path) > 1 else make_simple_key(path[0]),
            pre_eq=pre,
            post_eq=post,
            value=value_to_node(value),
            trailing=Trivia(),
            has_comma=False,
            post_comma_trivia=Trivia(),
        )

    # --- _DottedHost protocol ------------------------------------------------

    def set_at(self, path: tuple[str, ...], value: object) -> None:
        # Preserve in-place update for an exact simple match so the
        # entry's surrounding trivia and position survive round-tripping.
        if len(path) == 1:
            existing = self._find_entry(path[0])
            if existing is not None:
                existing.value = value_to_node(value)
                return
        self._node.entries[:] = [
            e for e in self._node.entries if not _path_has_prefix(e.key.path, path)
        ]
        self._node.entries.append(self._make_entry(path, value))
        _apply_separator_style(self._node, self._style)

    def del_prefix(self, prefix: tuple[str, ...]) -> bool:
        kept = [
            e for e in self._node.entries if not _path_has_prefix(e.key.path, prefix)
        ]
        if len(kept) == len(self._node.entries):
            return False
        self._node.entries[:] = kept
        _apply_separator_style(self._node, self._style)
        return True

    def entries_under(
        self, prefix: tuple[str, ...]
    ) -> list[tuple[tuple[str, ...], ValueNode]]:
        plen = len(prefix)
        return [
            (e.key.path, e.value)
            for e in self._node.entries
            if len(e.key.path) > plen and e.key.path[:plen] == prefix
        ]

    # --- mapping mutation ----------------------------------------------------

    @override
    def _set_value(self, key: str, value: object) -> TomlValue | None:
        self.set_at((key,), value)
        return None

    @override
    def _delete_value(self, key: str) -> None:
        # Best-effort CST cleanup; presence is enforced by the caller
        # (``Table.__delitem__``) at the cache level.
        self.del_prefix((key,))


def _path_has_prefix(path: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return len(path) >= len(prefix) and path[: len(prefix)] == prefix


class _DottedHost(Protocol):
    """Mutation back-channel for a synthetic dotted-key sub-table.

    A host knows how to add/replace, remove, and *enumerate* dotted-key
    entries in the underlying physical container so that views built on
    top of it re-read live state instead of a stale snapshot.
    """

    def set_at(self, path: tuple[str, ...], value: object) -> None: ...

    def del_prefix(self, prefix: tuple[str, ...]) -> bool: ...

    def entries_under(
        self, prefix: tuple[str, ...]
    ) -> list[tuple[tuple[str, ...], ValueNode]]: ...


class _SectionDottedHost:
    """Mutates dotted-key entries inside one or more :class:`SectionNode`."""

    __slots__ = ("_sections",)

    def __init__(self, sections: list[SectionNode]) -> None:
        self._sections = sections

    def set_at(self, path: tuple[str, ...], value: object) -> None:
        # Remove any existing entry at or under this path; remember which
        # section last hosted such an entry so the new dotted KV lands
        # near its predecessors when sections are split.
        host_sec: SectionNode | None = None
        for sec in self._sections:
            kept: list[KeyValueNode] = []
            for kv in sec.entries:
                if _path_has_prefix(kv.key.path, path):
                    host_sec = sec
                else:
                    kept.append(kv)
            if len(kept) != len(sec.entries):
                sec.entries[:] = kept
        if host_sec is None:
            # No existing entry at this path: pick the section that
            # already owns dotted entries with the same head, else last.
            head = path[0]
            host_sec = next(
                (
                    sec
                    for sec in self._sections
                    if any(kv.key.path and kv.key.path[0] == head for kv in sec.entries)
                ),
                self._sections[-1],
            )
        host_sec.entries.append(
            KeyValueNode(
                leading=Trivia(),
                key=_make_dotted_key(path),
                pre_eq=WhitespaceNode(" "),
                post_eq=WhitespaceNode(" "),
                value=value_to_node(value),
                trailing=None,
                trailing_comment=None,
                newline=NewlineNode("\n"),
            ),
        )

    def del_prefix(self, prefix: tuple[str, ...]) -> bool:
        any_removed = False
        for sec in self._sections:
            kept = [
                kv for kv in sec.entries if not _path_has_prefix(kv.key.path, prefix)
            ]
            if len(kept) != len(sec.entries):
                sec.entries[:] = kept
                any_removed = True
        return any_removed

    def entries_under(
        self, prefix: tuple[str, ...]
    ) -> list[tuple[tuple[str, ...], ValueNode]]:
        plen = len(prefix)
        out: list[tuple[tuple[str, ...], ValueNode]] = []
        for sec in self._sections:
            out.extend(
                (kv.key.path, kv.value)
                for kv in sec.entries
                if len(kv.key.path) > plen and kv.key.path[:plen] == prefix
            )
        return out


class _DottedSubTable(Table):
    """Synthetic table aggregating dotted-key entries.

    The view is *live*: entries are re-read from the host on each
    access, so mutations through this view (or a sibling view onto the
    same underlying container) are immediately visible.
    """

    __slots__ = ("_depth", "_host", "_prefix")

    def __init__(
        self,
        *,
        depth: int,
        host: _DottedHost,
        prefix: tuple[str, ...],
    ) -> None:
        super().__init__()
        self._depth = depth
        self._host = host
        self._prefix = prefix
        self._populate()

    @override
    def _items(self) -> Iterator[tuple[str, TomlValue]]:
        groups: dict[str, list[tuple[tuple[str, ...], ValueNode]]] = {}
        order: list[str] = []
        terminals: dict[str, ValueNode] = {}
        for path, value in self._host.entries_under(self._prefix):
            head = path[self._depth]
            if len(path) == self._depth + 1:
                terminals[head] = value
                if head not in order:
                    order.append(head)
                continue
            if head not in groups:
                groups[head] = []
                if head not in order:
                    order.append(head)
            groups[head].append((path, value))
        for head in order:
            if head in terminals:
                yield head, _value_for(terminals[head])
            else:
                yield (
                    head,
                    _DottedSubTable(
                        depth=self._depth + 1,
                        host=self._host,
                        prefix=(*self._prefix, head),
                    ),
                )

    @override
    def _set_value(self, key: str, value: object) -> TomlValue | None:
        self._host.set_at((*self._prefix, key), value)
        return None

    @override
    def _delete_value(self, key: str) -> None:
        if key not in self:
            raise KeyError(key)
        self._host.del_prefix((*self._prefix, key))


class _StdTable(Table):
    """Standard TOML table view: aggregates physical sections by path."""

    __slots__ = (
        "_anchor",
        "_doc_node",
        "_init_extras",
        "_init_pool",
        "_owner_anchor",
        "_path",
    )

    def __init__(
        self,
        doc_node: DocumentNode,
        path: tuple[str, ...],
        *,
        anchor: SectionNode | None = None,
        owner_anchor: SectionNode | None = None,
        _pool: list[SectionNode] | None = None,
        _extras: list[tuple[tuple[str, ...], KeyValueNode]] | None = None,
    ) -> None:
        super().__init__()
        self._doc_node = doc_node
        self._path = path
        # ``anchor`` is set only for AoT entries: it is the [[path]]
        # section that owns this entry. With no anchor, all sections
        # whose header matches ``path`` are direct sections of this
        # table.
        self._anchor = anchor
        # ``owner_anchor`` is the AoT [[..]] section whose owned range
        # bounds this table's universe of sections. For an AoT entry
        # itself it is the entry's own anchor; for a sub-table inside
        # an AoT entry it is the enclosing entry's anchor; for tables
        # outside any AoT entry it is ``None`` (the whole document).
        # The section pool and the inherited dotted-key "extras" are
        # both *re-derived* from ``_doc_node`` on every read so that
        # purges and inserts can never desync the dict view from what
        # ``dumps`` would render.
        self._owner_anchor = owner_anchor if owner_anchor is not None else anchor
        # Transient hints, used only during the initial ``_populate``
        # call that runs from this constructor. The parent ``_iter_table``
        # already partitioned its section pool by next-head and computed
        # the dotted-key "extras" that extend into us, so we can skip
        # the full rescan and ancestor walk that post-mutation reads do.
        self._init_pool = _pool
        self._init_extras = _extras
        try:
            self._populate()
        finally:
            self._init_pool = None
            self._init_extras = None

    def _scope(self) -> list[SectionNode] | None:
        owner = self._owner_anchor
        if owner is None:
            return None
        return self._doc_node.aot_entry_block(owner)

    def _compute_extras(
        self,
    ) -> list[tuple[tuple[str, ...], KeyValueNode]] | None:
        """Inherited dotted KVs whose path passes through ``self._path``.

        For the root table this is always ``None``: the root has no
        ancestors. For a nested table at path ``P`` of length ``n``,
        scans every section whose header is a strict prefix of ``P``
        (or ``None`` for the implicit pre-header section) and returns,
        for each dotted KV in such a section that extends into ``P``,
        the relative path inside ``P`` plus the KV node itself.

        Used for post-mutation reads. Construction-time reads receive
        their extras pre-computed top-down by the parent ``_iter_table``.
        """
        plen = len(self._path)
        if plen == 0:
            return None
        scope = self._scope()
        sections = scope if scope is not None else self._doc_node.sections
        out: list[tuple[tuple[str, ...], KeyValueNode]] = []
        for sec in sections:
            hdr = sec.header
            host_path: tuple[str, ...] = hdr.key.path if hdr is not None else ()
            hlen = len(host_path)
            if hlen >= plen or host_path != self._path[:hlen]:
                continue
            for kv in sec.entries:
                full = (*host_path, *kv.key.path)
                if len(full) <= plen or full[:plen] != self._path:
                    continue
                out.append((full[plen:], kv))
        return out or None

    @override
    def _items(self) -> Iterator[tuple[str, TomlValue]]:
        if self._init_pool is not None:
            pool = self._init_pool
        else:
            scope = self._scope()
            pool = scope if scope is not None else self._doc_node.sections
        extras = (
            self._init_extras
            if self._init_extras is not None
            else self._compute_extras()
        )
        return _iter_table(
            self._doc_node,
            self._path,
            pool=pool,
            anchor=self._anchor,
            owner_anchor=self._owner_anchor,
            extras=extras,
        )

    def _direct_sections(self) -> list[SectionNode]:
        if self._anchor is not None:
            return [self._anchor]
        path = self._path
        scope = self._scope()
        sections = scope if scope is not None else self._doc_node.sections
        if path == ():
            return [s for s in sections if s.header is None]
        return [
            s
            for s in sections
            if s.header is not None
            and s.header.kind == "table"
            and s.header.key.path == path
        ]

    @override
    def _detach(self, doc_node: DocumentNode | None = None) -> None:
        if not self._attached:
            return
        if doc_node is None:
            # Top of detachment subtree: capture every section under our
            # path and move them into a private DocumentNode so later
            # structural mutations through this table cannot reach the
            # original document. AoT entries capture exactly the anchor
            # plus its owned sub-section run; everything else captures
            # all sections rooted at our path.
            captured = (
                self._doc_node.aot_entry_block(self._anchor)
                if self._anchor is not None
                else self._sections_under_path()
            )
            doc_node = DocumentNode(sections=list(captured))
        self._doc_node = doc_node
        # The captured sections form a self-contained little document;
        # there is no longer an enclosing AoT entry to bound our world.
        self._owner_anchor = self._anchor
        super()._detach(doc_node)

    def __copy__(self) -> _StdTable:
        return self.__deepcopy__({})

    def __deepcopy__(self, memo: dict[int, Any]) -> _StdTable:
        # Copy via the CST + a fresh view rather than dict's default
        # ``_reconstruct``, which would re-enter ``__setitem__`` against
        # an empty dict cache + a populated CST and double the CST
        # entries. ``deepcopy(self._doc_node, memo)`` registers each
        # cloned ``SectionNode`` in ``memo`` so anchors map across.
        new_doc = deepcopy(self._doc_node, memo)
        new_anchor = memo[id(self._anchor)] if self._anchor is not None else None
        new_owner = (
            memo[id(self._owner_anchor)] if self._owner_anchor is not None else None
        )
        new = self.__class__(
            new_doc, self._path, anchor=new_anchor, owner_anchor=new_owner
        )
        new._attached = self._attached  # noqa: SLF001
        memo[id(self)] = new
        return new

    def _sections_under_path(self) -> list[SectionNode]:
        plen = len(self._path)
        out: list[SectionNode] = []
        for sec in self._doc_node.sections:
            hdr = sec.header
            if hdr is None:
                continue
            hpath = hdr.key.path
            if (
                len(hpath) >= plen
                and hpath[:plen] == self._path
                and (len(hpath) > plen or hdr.kind == "table")
            ):
                out.append(sec)
        return out

    def _classify(self, key: str) -> tuple[str, object]:
        """Classify a key for mutation purposes.

        Returns one of:
            ("direct", KeyValueNode)         - a single-part scalar/value entry
            ("dotted", None)                 - dotted-key prefix (e.g. b.c=...)
            ("table", None)                  - child standard table [self.path.key]
            ("aot", None)                    - child AoT [[self.path.key]]
            ("extras", KeyValueNode)         - terminal entry living as a dotted
                                               KV in an ancestor section
            ("extras-prefix", None)          - extras entries with this head are
                                               longer than one segment
            ("absent", None)
        """
        # Dict storage is a complete index of every key visible at this
        # path: ``_populate`` and ``_refresh_key`` keep it in sync with
        # the CST. If the key isn't cached, no scan can find it.
        if not dict.__contains__(self, key):
            return ("absent", None)
        if self._anchor is not None:
            # AoT-anchored slow path: direct sections is just the anchor;
            # child sections may live anywhere within the entry's owned
            # sub-section run, so search just that scope (not the whole
            # document) — otherwise siblings' same-path sub-sections
            # would be misattributed to this entry.
            for kv in self._anchor.entries:
                if kv.key.path[0] == key:
                    if len(kv.key.path) == 1:
                        return ("direct", kv)
                    return ("dotted", None)
            child = (*self._path, key)
            scope = self._scope()
            assert scope is not None  # _anchor is not None ⇒ owner_anchor is set
            for sec in scope:
                hdr = sec.header
                if hdr is None:
                    continue
                hpath = hdr.key.path
                if hpath == child:
                    return ("aot" if hdr.kind == "array" else "table", None)
                if len(hpath) > len(child) and hpath[: len(child)] == child:
                    # Deeper [a.b.c] or [[a.b.c]] makes ``key`` an
                    # implicit super-table — i.e. a table.
                    return ("table", None)
            return self._classify_extras(key)

        # Common path: not AoT-anchored. Fuse the direct-entries scan and the
        # child-section scan into a single pass over the document.
        path = self._path
        plen = len(path)
        child_len = plen + 1
        scope = self._scope()
        sections = scope if scope is not None else self._doc_node.sections
        child_kind: str | None = None
        for sec in sections:
            hdr = sec.header
            if hdr is None:
                if plen == 0:
                    for kv in sec.entries:
                        if kv.key.path[0] == key:
                            if len(kv.key.path) == 1:
                                return ("direct", kv)
                            return ("dotted", None)
                continue
            hpath = hdr.key.path
            if hdr.kind == "table" and hpath == path:
                for kv in sec.entries:
                    if kv.key.path[0] == key:
                        if len(kv.key.path) == 1:
                            return ("direct", kv)
                        return ("dotted", None)
                continue
            hlen = len(hpath)
            if hlen >= child_len and hpath[:plen] == path and hpath[plen] == key:
                if hdr.kind == "array" and hlen == child_len:
                    return ("aot", None)
                # A deeper section ([a.b.c] *or* [[a.b.c]]) below ``key``
                # makes ``key`` an implicit super-table — i.e. a table.
                child_kind = "table"
        if child_kind is not None:
            return (child_kind, None)
        return self._classify_extras(key)

    def _classify_extras(self, key: str) -> tuple[str, object]:
        """Tail of :meth:`_classify`: look for ancestor-section dotted KVs."""
        extras = self._compute_extras()
        if extras:
            terminal = None
            has_dotted = False
            for rel, kv in extras:
                if rel[0] != key:
                    continue
                if len(rel) == 1:
                    terminal = kv
                else:
                    has_dotted = True
            if terminal is not None:
                return ("extras", terminal)
            if has_dotted:
                return ("extras-prefix", None)
        return ("absent", None)

    def _purge_conflicting(self, key: str) -> None:
        """Remove any existing dotted, sub-table or AoT structure under ``key``.

        Used to give Python-dict-style overwrite semantics: assigning to
        a name that already names a sub-table silently destroys that
        sub-table (and any nested children) rather than raising. Also
        removes any ancestor-section dotted entries that contribute
        to ``self._path + (key, ...)`` so they don't survive as ghosts.
        Constrained to ``self._scope()`` so an AoT entry can't reach
        across its boundary and delete a sibling entry's same-path
        sub-section.

        Pure structural removal — does not touch top-blank trivia.
        Callers run :meth:`DocumentNode.normalise_top_blank` themselves
        once the larger operation is done, so a purge-then-splice
        sequence doesn't strip a soon-to-be-meaningful blank in the
        intermediate state.
        """
        for sec in self._direct_sections():
            sec.entries[:] = [kv for kv in sec.entries if kv.key.path[0] != key]
        prefix = (*self._path, key)
        plen = len(prefix)
        scope = self._scope()
        scope_ids = None if scope is None else {id(s) for s in scope}
        doc_sections = self._doc_node.sections
        doc_sections[:] = [
            sec
            for sec in doc_sections
            if not (
                (scope_ids is None or id(sec) in scope_ids)
                and sec.header is not None
                and len(sec.header.key.path) >= plen
                and sec.header.key.path[:plen] == prefix
            )
        ]
        # Drop any ancestor-section dotted KV that contributes to our
        # path + key (e.g. ``[tool] poetry.name = "x"`` when purging
        # ``name`` from the ``tool.poetry`` view). The ``hlen >= plen``
        # check below skips every section we just removed, so iterating
        # the pre-splice scope is safe.
        ppath_len = len(self._path)
        for sec in scope if scope is not None else doc_sections:
            hdr = sec.header
            host_path: tuple[str, ...] = hdr.key.path if hdr is not None else ()
            hlen = len(host_path)
            if hlen >= plen or host_path != prefix[:hlen]:
                continue
            sec.entries[:] = [
                kv
                for kv in sec.entries
                if not (
                    len(kv.key.path) > ppath_len - hlen
                    and (*host_path, *kv.key.path)[:plen] == prefix
                )
            ]

    @override
    def _set_value(self, key: str, value: object) -> TomlValue | None:
        kind, payload = self._classify(key)
        if kind in ("direct", "extras"):
            # In-place value swap: reuse the existing KV node.
            assert isinstance(payload, KeyValueNode)
            payload.value = value_to_node(value)
            return _value_for(payload.value)
        if kind in ("dotted", "table", "aot", "extras-prefix"):
            self._purge_conflicting(key)
            self._doc_node.normalise_top_blank()
        sections = self._direct_sections()
        if not sections:
            sections = [self._ensure_section()]
        target = sections[-1]
        indent = _detect_indent(target)
        new_kv = make_keyvalue_node(key, value, indent=indent)
        if _first_gap_is_blank(kv.leading for kv in target.entries[1:]):
            new_kv.leading.pieces.insert(0, NewlineNode("\n"))
        _ensure_trailing_newline(target)
        # Migrate any parked preamble (only present when this is the
        # first content being added to a previously-empty doc) ahead of
        # the new KV. No-op once the doc has structural content.
        self._doc_node.adopt_preamble_into(new_kv.leading)
        target_was_empty = not target.entries
        target.entries.append(new_kv)
        # Top-level only: if this is the *first* key being added to the
        # implicit pre-header section and a ``[table]`` follows, separate
        # the new key from that header with a blank line. When the
        # pre-header section already had user-authored entries we leave
        # the layout below alone — the user chose it deliberately.
        if self._path == () and target.header is None and target_was_empty:
            doc_node = self._doc_node
            idx = doc_node.sections.index(target)
            if idx + 1 < len(doc_node.sections):
                next_header = doc_node.sections[idx + 1].header
                if next_header is not None:
                    _prepend_blank_line(next_header.leading)
        # The new dict-storage value is exactly what we just wrote;
        # caller can skip the full _refresh_key walk. Safe for every
        # kind because _purge_conflicting only removes things keyed by
        # ``key`` in this scope, so no other dict slot is invalidated.
        return _value_for(new_kv.value)

    @override
    def _delete_value(self, key: str) -> None:
        kind, payload = self._classify(key)
        if kind == "absent":
            # Nothing to remove from the CST. Either the caller already
            # validated presence at the cache level (``Table.__delitem__``
            # does), or the key is genuinely absent and the cache will
            # raise on its own. Either way, the CST has no work to do.
            return
        if kind in ("direct", "extras") and isinstance(payload, KeyValueNode):
            # Targeted removal: drop just the matching KV from its section.
            # Avoids the O(N) section walks and full-list rebuilds in
            # ``_purge_conflicting`` when there's nothing else to remove.
            target = payload
            scope = self._scope()
            sections = scope if scope is not None else self._doc_node.sections
            for sec in sections:
                if target in sec.entries:
                    self._doc_node.remove_entry(sec, target)
                    return
            return  # pragma: no cover - defensive: kv must be reachable
        self._purge_conflicting(key)
        self._doc_node.normalise_top_blank()

    def _ensure_section(self) -> SectionNode:
        """Materialise a section that holds direct entries for ``self._path``."""
        if self._path == ():
            return self._ensure_root_section()
        return self._ensure_nested_section()

    def _ensure_root_section(self) -> SectionNode:
        """Insert an implicit pre-header section at the top of the document."""
        doc_node = self._doc_node
        new_sec = SectionNode(header=None, entries=[])
        # Ensure a blank line precedes the next section's header so the
        # newly-inserted top-level keys aren't visually glued to it.
        if doc_node.sections and doc_node.sections[0].header is not None:
            _prepend_blank_line(doc_node.sections[0].header.leading)
        doc_node.sections.insert(0, new_sec)
        return new_sec

    def _ensure_nested_section(self) -> SectionNode:
        """Insert a fresh ``[a.b.c]`` header for a nested path.

        Placement is immediately before the first descendant section so
        the new keys logically belong to the same place in the document.
        Falls back to appending when there is no descendant.
        """
        doc_node = self._doc_node
        new_sec = _new_section(self._path)
        assert new_sec.header is not None
        header = new_sec.header
        plen = len(self._path)
        for i, sec in enumerate(doc_node.sections):
            h = sec.header
            if (
                h is not None
                and len(h.key.path) > plen
                and h.key.path[:plen] == self._path
            ):
                # Insert a leading newline so the new header doesn't
                # glue against the previous section's last entry.
                header.leading.pieces.append(NewlineNode("\n"))
                doc_node.adopt_preamble_into(header.leading)
                doc_node.sections.insert(i, new_sec)
                return new_sec
        if any(s.header is not None or s.entries for s in doc_node.sections):
            header.leading.pieces.append(NewlineNode("\n"))
        doc_node.adopt_preamble_into(header.leading)
        doc_node.sections.append(new_sec)
        return new_sec

    # ------------------------------------------------------------------
    # Comment API (live mapping side-channels)
    # ------------------------------------------------------------------

    def _find_direct_kv(self, key: str) -> tuple[SectionNode, KeyValueNode]:
        """Return the section + KV node binding ``key`` as a single segment.

        Raises :class:`KeyError` when ``key`` is absent or the binding is
        a child table / dotted-key prefix rather than a simple
        ``key = value`` line.
        """
        for sec in self._direct_sections():
            for kv in sec.entries:
                if len(kv.key.path) == 1 and kv.key.path[0] == key:
                    return sec, kv
        raise KeyError(key)

    @property
    @override
    def comments(self) -> MutableMapping[str, str]:
        return _TableCommentsView(self)

    @property
    @override
    def leading_comments(self) -> MutableMapping[str, tuple[str, ...]]:
        return _TableLeadingCommentsView(self)

    def _first_header(self) -> TableHeaderNode:
        for sec in self._direct_sections():
            if sec.header is not None:
                return sec.header
        msg = (
            f"table {'.'.join(self._path) or '<root>'!r} has no physical "
            "header (it exists only through implicit parents); the "
            "header comment API is unavailable"
        )
        raise TOMLError(msg)

    @property  # type: ignore[explicit-override]
    @override
    def header_comment(self) -> str | None:
        header = self._first_header()
        if header.trailing_comment is None:
            return None
        return _strip_comment_marker(header.trailing_comment.text)

    @header_comment.setter
    def header_comment(self, value: str | None) -> None:
        _set_eol_comment(self._first_header(), value)

    @header_comment.deleter
    def header_comment(self) -> None:
        _set_eol_comment(self._first_header(), None)

    @property  # type: ignore[explicit-override]
    @override
    def header_leading_comments(self) -> tuple[str, ...]:
        header = self._first_header()
        return _extract_trailing_comment_block(header.leading)

    @header_leading_comments.setter
    def header_leading_comments(self, value: Sequence[str]) -> None:
        header = self._first_header()
        _replace_trailing_comment_block(
            header.leading,
            value,
            _indent_after_last_newline(header.leading),
        )

    @header_leading_comments.deleter
    def header_leading_comments(self) -> None:
        header = self._first_header()
        _replace_trailing_comment_block(
            header.leading,
            (),
            _indent_after_last_newline(header.leading),
        )

    @override
    def promote_inline(self, key: str) -> Table:
        sec, kv, inline = self._find_promotable(key, InlineTableNode, "an inline table")
        if _value_has_inner_comment(inline):
            msg = (
                f"cannot promote {key!r} to [..]: inline table has inner "
                "comments that would be lost; remove the inner comments first"
            )
            raise TOMLError(msg)
        child_path = (*self._path, key)
        self._refuse_existing_promoted_section(key, child_path, kind="table")
        new_sec = _build_promoted_section(child_path, inline, kv)
        self._doc_node.remove_entry(sec, kv)
        self._splice_promoted_sections([new_sec])
        view = _StdTable(self._doc_node, child_path)
        dict.__setitem__(self, key, view)
        return view

    @override
    def promote_array(self, key: str) -> AoT:
        sec, kv, array = self._find_promotable(key, ArrayNode, "an array")
        items = array.items
        if not items:
            msg = f"{key!r} is an empty array; cannot promote to array-of-tables"
            raise TOMLError(msg)
        for item in items:
            if not isinstance(item.value, InlineTableNode):
                msg = (
                    f"{key!r} contains a non-inline-table element; cannot "
                    "promote to array-of-tables"
                )
                raise TOMLError(msg)
        if _value_has_inner_comment(array):
            msg = (
                f"cannot promote {key!r} to [[..]]: array has inner comments "
                "that would be lost; remove the inner comments first"
            )
            raise TOMLError(msg)
        child_path = (*self._path, key)
        self._refuse_existing_promoted_section(key, child_path, kind="aot")
        new_secs = [
            _build_promoted_aot_section(child_path, item.value)
            for item in items
            if isinstance(item.value, InlineTableNode)  # for type narrowing
        ]
        # Carry the source KV's authoring trivia onto the new AoT
        # entries: leading comments / blank lines that sat above the
        # inline assignment go on the first ``[[..]]`` header, and any
        # trailing whitespace / EOL comment goes after the last entry's
        # final value. Without this, ``promote_array`` silently drops
        # the user's comments.
        if new_secs:
            first_hdr = new_secs[0].header
            assert first_hdr is not None
            first_hdr.leading.pieces[:0] = list(kv.leading.pieces)
            last_entries = new_secs[-1].entries
            if last_entries:
                last_entries[-1].trailing = kv.trailing
                last_entries[-1].trailing_comment = kv.trailing_comment
        self._doc_node.remove_entry(sec, kv)
        self._splice_promoted_sections(new_secs)
        aot = AoT._attached_to(self._doc_node, child_path, [])  # noqa: SLF001
        aot._resync()  # noqa: SLF001
        dict.__setitem__(self, key, aot)
        return aot

    def _find_promotable(
        self,
        key: str,
        expected: type[_T],
        label: str,
    ) -> tuple[SectionNode, KeyValueNode, _T]:
        """Find a direct KV at ``key`` whose value is an ``expected`` node.

        Raises a friendly :class:`TOMLError` if the key exists under a
        different shape (sub-section, dotted-key subtable, AoT) or if
        the value is the wrong node type. ``label`` is the
        "an inline table" / "an array" phrasing used in the message.
        Returns the host section, the KV, and the typed value node so
        callers don't need to re-narrow.
        """
        try:
            sec, kv = self._find_direct_kv(key)
        except KeyError:
            # If the key exists under a different shape (sub-section,
            # dotted-key subtable, or AoT), raise a clearer error rather
            # than a bare KeyError that contradicts ``key in self``.
            if key in self:
                msg = f"{key!r} is not {label}; nothing to promote"
                raise TOMLError(msg) from None
            raise
        if not isinstance(kv.value, expected):
            msg = f"{key!r} is not {label}; nothing to promote"
            raise TOMLError(msg)
        return sec, kv, kv.value

    def _refuse_existing_promoted_section(
        self,
        key: str,
        child_path: tuple[str, ...],
        *,
        kind: str,
    ) -> None:
        """Defensive: refuse if ``[child_path]`` (or ``[[child_path]]``)
        already exists within this view's scope. The parser blocks any
        source where this would arise and assignment auto-purges
        conflicts, so this is mostly a guard against direct CST
        manipulation -- but the scope restriction matters in normal
        use too: an AoT entry must not see a same-named promoted
        section that lives in a sibling entry's block.
        """
        scope = self._scope()
        sections = scope if scope is not None else self._doc_node.sections
        for existing in sections:
            hdr = existing.header
            if hdr is not None and hdr.key.path == child_path:
                joined = ".".join(child_path)
                if kind == "aot":
                    msg = (
                        f"cannot promote {key!r}: a [[{joined}]] (or "
                        f"[{joined}]) section already exists"
                    )
                else:
                    msg = f"cannot promote {key!r}: a [{joined}] section already exists"
                raise TOMLError(msg)

    def _splice_promoted_sections(self, new_secs: Sequence[SectionNode]) -> None:
        """Splice freshly-promoted sections after the parent's last
        direct section (or at end of document if the parent has none).
        Uses ``_insert_section_block`` so consecutive AoT entries stay
        blank-separated.
        """
        sections = self._doc_node.sections
        parent_secs = self._direct_sections()
        if parent_secs:
            anchor = parent_secs[-1]
            insert_at = (
                next(
                    (i for i, s in enumerate(sections) if s is anchor),
                    len(sections) - 1,
                )
                + 1
            )
        else:
            insert_at = len(sections)
        _insert_section_block(self._doc_node, insert_at, new_secs)

    @override
    def _install_flavoured(self, parts: tuple[str, ...], value: object) -> None:
        if self._dispatch_structural(parts, value):
            return
        # Plain value (or same-slot identity case skipped by
        # ``_dispatch_structural``). For a single-segment install
        # commit at ``self`` directly via ``_commit_value`` -- going
        # through ``self[parts[0]] = value`` would re-enter
        # ``__setitem__``, which loops back through us. For a
        # multi-segment install, descend (creating implicit parents
        # as needed) and assign at the leaf with normal
        # ``__setitem__`` semantics.
        if len(parts) == 1:
            self._commit_value(parts[0], value)
            return
        target = self.ensure_table(parts[:-1])
        target[parts[-1]] = value

    def _dispatch_structural(
        self,
        parts: tuple[str, ...],
        value: object,
    ) -> bool:
        """Try to install ``value`` at ``parts`` as a structural edit.

        Returns ``True`` when the value carried enough flavour to drive
        a section / AoT / array install (so the caller should stop);
        ``False`` for plain values that need the ordinary value-write
        path. Centralising the dispatch keeps ``__setitem__`` /
        :meth:`_set_value` (single-segment writes) and
        :meth:`Document.install` (multi-segment writes) on the same
        decision tree.
        """
        if isinstance(value, SectionSpec):
            self._install_section(parts, value)
            return True
        if isinstance(value, AoT):
            # An unattached AoT (e.g. ``AoT([{...}])`` or freshly
            # detached) live-attaches: its orphan section nodes
            # migrate into the live doc with their headers rebased
            # in place, and the user's ``value`` becomes the view at
            # this slot. An already-attached source is deep-cloned so
            # two slots never share CST sections.
            self._install_aot(parts, value)
            return True
        if isinstance(value, _StdTable) and not (
            value._doc_node is self._doc_node  # noqa: SLF001
            and value._path == (*self._path, *parts)  # noqa: SLF001
        ):
            # Section-backed Table: deep-clone the source CST so
            # comments and formatting survive, and so any nested AoT
            # lands as ``[[..]]`` rather than crashing the inline-table
            # synthesiser. Skip when the value is already installed at
            # the target path (e.g. during ``deepcopy`` reconstruction).
            self._install_attached_table(parts, value)
            return True
        # No bespoke Array branch: an unattached :class:`Array` falls
        # through to the plain-value commit in ``_install_flavoured``,
        # which routes through ``_set_value`` -> ``value_to_node``.
        # ``value_to_node`` splices the user's ``_node`` into the
        # destination KV (live attach), and ``_commit_value`` then
        # swaps the user's reference into dict storage so
        # ``self[k] is value`` holds. Attached Arrays deep-clone in
        # ``value_to_node`` for the same reason.
        return False

    def _prepare_section_slot(
        self,
        parts: tuple[str, ...],
    ) -> tuple[tuple[str, ...], int, Trivia | None]:
        """Purge any conflicting value at ``parts`` and pick an insert index.

        Returns ``(full_path, insert_at, prior_leading)`` where:

        * ``full_path`` is the absolute CST path (``self._path + parts``).
        * ``insert_at`` is the position in ``self._doc_node.sections`` where
          a new block for ``full_path`` should be spliced in.
        * ``prior_leading`` is the leading trivia of the first matching
          section's header captured *before* purging, or ``None`` if no
          such section existed. Callers should transplant it onto the
          first new section's header so an in-place replacement preserves
          the comments / blank lines that sat above the original.

        Drops any redundant empty placeholder header at ``self._path``
        first so it doesn't survive as visual noise once the new child
        section is in place.

        If a section (or descendant of one) already sits at
        ``full_path``, remember its position before purging and reuse
        that slot, so replacing a section in place preserves its
        position among siblings instead of being appended at the end.
        """
        self._drop_redundant_anchor()
        full_path = (*self._path, *parts)
        plen = len(full_path)
        scope = self._scope()
        scope_ids = None if scope is None else {id(s) for s in scope}
        prior_index: int | None = None
        prior_leading: Trivia | None = None
        for i, sec in enumerate(self._doc_node.sections):
            hdr = sec.header
            if (
                hdr is not None
                and (scope_ids is None or id(sec) in scope_ids)
                and len(hdr.key.path) >= plen
                and hdr.key.path[:plen] == full_path
            ):
                prior_index = i
                # Capture only when the matched section's header is exactly
                # at full_path: a deeper match (e.g. ``[a.b.c]`` while we
                # replace ``[a.b]``) is a sub-section, not the slot itself,
                # and its leading belongs to it rather than to the parent.
                if len(hdr.key.path) == plen:
                    prior_leading = _clone_trivia(hdr.leading)
                break
        if len(parts) == 1:
            kind, _ = self._classify(parts[0])
            if kind != "absent":
                # Skip the top-blank normalisation here: we are about
                # to splice a replacement section into the slot we just
                # vacated, so a leading blank on whatever section sat
                # *behind* the purged one will still be a meaningful
                # inter-section separator once the new block is in
                # place. Normalising now strips it prematurely.
                self._purge_conflicting(parts[0])
        else:
            self._doc_node.purge_path(full_path)
        sections = self._doc_node.sections
        if prior_index is not None:
            # Only matching sections (within scope) get purged, so the
            # first match's index is preserved as a valid splice point.
            assert prior_index <= len(sections)
            return full_path, prior_index, prior_leading
        owner = self._owner_anchor
        if owner is None:
            return full_path, _section_insert_index(sections, full_path), None
        # AoT-entry sub-table with no prior section at this path:
        # pin the new section to the end of this entry's owned range
        # so it doesn't get re-attributed to a later entry on round-trip.
        return (
            full_path,
            sections.index(owner) + len(self._scope() or ()),
            None,
        )

    def _install_aot(
        self,
        parts: tuple[str, ...],
        value: AoT,
    ) -> None:
        """Install ``value`` (attached or unattached) at ``parts``.

        Attached source: deep-clone and splice; the destination gets a
        fresh view, the source is unchanged.

        Unattached source: rebase the source's section nodes in place
        and splice them into the live document, then rehome the user's
        ``value`` so it *is* the live view at the destination.

        The two paths share the slot-prep / blank-line-policy / splice
        mechanics in :meth:`_splice_attached`; only the source-section
        provider and the post-splice "wire up the view" step diverge.
        """
        was_attached = value._attached  # noqa: SLF001
        sources = _clone_aot_sections if was_attached else _rebase_aot_sections_inplace
        full_path = self._splice_attached(parts, value, sources)
        if was_attached:
            view: AoT = AoT._attached_to(self._doc_node, full_path, [])  # noqa: SLF001
        else:
            value._doc_node = self._doc_node  # noqa: SLF001
            value._path = full_path  # noqa: SLF001
            value._attached = True  # noqa: SLF001
            view = value
        view._resync()  # rehomes cached entries  # noqa: SLF001
        self._install_at_path(parts, view)

    @override
    def _install_section(
        self,
        parts: tuple[str, ...],
        value: Mapping[str, object] = MappingProxyType({}),
    ) -> Table:
        full_path, insert_at, prior_leading = self._prepare_section_slot(parts)
        new_sec = _new_section(full_path)
        new_sec.synthesised_placeholder = True
        _apply_prior_leading([new_sec], prior_leading)
        _insert_section_block(self._doc_node, insert_at, [new_sec])
        # Inherit ``owner_anchor`` from the parent so a sub-section
        # installed inside an AoT entry stays scoped to that entry —
        # otherwise reads/writes through ``view`` see same-named
        # sections in sibling entries and silently merge their values.
        view = _StdTable(self._doc_node, full_path, owner_anchor=self._owner_anchor)
        self._install_at_path(parts, view)
        for k, v in value.items():
            view[k] = v
        return view

    def _drop_redundant_anchor(self) -> None:
        """Drop an empty placeholder ``[X]`` header at ``self._path``.

        Called before installing a child section under this view. An
        empty ``[X]`` header that holds no entries and no comments
        serves no purpose once a child ``[X.Y]`` header follows it: the
        parent table is implied. AoT entry anchors (``[[X]]``) and
        anything carrying user comments are preserved verbatim.
        """
        if self._path == ():
            return
        for sec in self._doc_node.sections:
            hdr = sec.header
            if (
                hdr is None
                or hdr.kind != "table"
                or not sec.synthesised_placeholder
                or hdr.key.path != self._path
                or sec.entries
                or hdr.trailing_comment is not None
                or _trivia_has_comment(hdr.leading)
            ):
                continue
            self._doc_node.remove_sections({sec})
            if self._anchor is sec:
                self._anchor = None
            if self._owner_anchor is sec:
                self._owner_anchor = None
            return

    def _install_attached_table(
        self,
        parts: tuple[str, ...],
        value: _StdTable,
    ) -> _StdTable:
        """Deep-clone an attached source ``_StdTable`` into ``parts``.

        Implicit super-tables in the source remain implicit in the target —
        no empty intermediate ``[a]`` / ``[a.b]`` headers are emitted.
        """
        full_path = self._splice_attached(parts, value, _clone_table_sections)
        view = _StdTable(self._doc_node, full_path, owner_anchor=self._owner_anchor)
        self._install_at_path(parts, view)
        return view

    def _splice_attached(
        self,
        parts: tuple[str, ...],
        value: _StdTable | AoT,
        cloner: Callable[[Any, tuple[str, ...]], list[SectionNode]],
    ) -> tuple[str, ...]:
        """Common purge-and-splice for both attached-section installers.

        Snapshots the source CST *before* purging the destination slot so
        same-document calls where ``parts`` overlaps ``value._path``
        (e.g. ``doc["a"] = doc["a"]["b"]``,
        ``doc.install("a", doc.aot("a.inner"))``) still see their source.
        Returns the absolute target path, ready for the caller to wrap
        in a view.
        """
        full_path = (*self._path, *parts)
        new_secs = cloner(value, full_path)
        _full_path, insert_at, prior_leading = self._prepare_section_slot(parts)
        if new_secs:
            _apply_prior_leading(new_secs, prior_leading)
            _insert_section_block(
                self._doc_node,
                insert_at,
                new_secs,
                separate_within=False,
            )
        return full_path

    def _install_at_path(self, parts: tuple[str, ...], obj: object) -> None:
        """Install ``obj`` at the leaf of ``parts``, materialising any
        intermediate implicit super-tables in dict storage as we go.

        CST mutations are assumed to have already been performed; this
        method only reconciles the dict-storage view.
        """
        cur: Table = self
        for part in parts[:-1]:
            existing = super(Table, cur).get(part)
            if not isinstance(existing, Table):
                # Either absent (implicit super-table just materialised
                # in the CST) or replaced by a non-table en route: refresh
                # from the CST so dict storage matches.
                cur._refresh_key(part)  # noqa: SLF001
            nxt = super(Table, cur).__getitem__(part)
            assert isinstance(nxt, Table)
            cur = nxt
        super(Table, cur).__setitem__(parts[-1], obj)


class Document(_StdTable):
    """Top-level TOML document. Subclass of :class:`Table`."""

    __slots__ = ("_newline",)

    def __init__(self, node: DocumentNode) -> None:
        # Hand the construction walk the full section list and an
        # empty extras tuple. ``_iter_table`` then partitions sections
        # by head as it descends, so each nested ``_StdTable`` only
        # sees its own slice of the document — no per-level rescans.
        super().__init__(node, (), _pool=node.sections, _extras=[])
        self._newline = _detect_newline(node)

    @property
    def cst(self) -> DocumentNode:
        """The underlying concrete syntax tree (CST).

        Returns the root :class:`~tomlrt._nodes.DocumentNode` that
        records the document's exact byte layout. Intended for
        tooling and debugging — most users will never need this.
        """
        return self._doc_node

    def render(self) -> str:
        if self._newline != "\n":
            _normalise_newlines(self._doc_node, self._newline)
        return self._doc_node.render()

    @override
    def __copy__(self) -> Document:
        # The CST is the source of truth; sharing it across "copies" would
        # mean mutations on one bled into the other. Always clone.
        return Document(deepcopy(self._doc_node))

    @override
    def __deepcopy__(self, memo: dict[int, Any]) -> Document:
        return Document(deepcopy(self._doc_node, memo))

    @property
    def preamble(self) -> tuple[str, ...]:
        """Comment block at the top of the document.

        A "preamble" is the run of ``# …`` lines that opens the file
        and is blank-line-separated from anything below. Comments that
        sit directly above the first key (no blank line) are *not*
        preamble — they are the leading comments of that key, accessed
        via :attr:`leading_comments`. In a document with no structural
        content, the entire opening comment block is treated as
        preamble.

        Setter accepts a sequence of bare comment texts (without the
        leading ``#``) and replaces the current preamble; assign ``()``
        to remove. Newlines inside any line are rejected.
        """
        target = self._doc_node.preamble_target()
        pieces = target.pieces
        end, comments = _scan_leading_comment_run(pieces)
        if not comments:
            return ()
        has_separator = end < len(pieces) and isinstance(pieces[end], NewlineNode)
        if has_separator or not self._doc_node.has_content():
            return tuple(_strip_comment_marker(c) for c in comments)
        return ()

    @preamble.setter
    def preamble(self, value: Sequence[str]) -> None:
        _validate_comment_lines(value)
        target = self._doc_node.preamble_target()
        pieces = target.pieces
        has_content = self._doc_node.has_content()
        run_end, _ = _scan_leading_comment_run(pieces)
        has_separator = run_end < len(pieces) and isinstance(
            pieces[run_end], NewlineNode
        )
        # Drop the existing preamble run plus exactly one separator NL,
        # but only if the run is genuinely preamble (separated, or doc empty).
        is_preamble = has_separator or not has_content
        consume = (run_end + (1 if has_separator else 0)) if is_preamble else 0
        new: list[TriviaPiece] = []
        for line in value:
            new += [CommentNode(text=_format_comment(line)), NewlineNode("\n")]
        if value and has_content:
            new.append(NewlineNode("\n"))
        target.pieces = new + list(pieces[consume:])

    @property
    def epilogue(self) -> tuple[str, ...]:
        """Comment block at the very end of the document.

        Returns the trailing run of ``# …`` lines that follows all
        structural content. Empty when the document has no structural
        content (in that case everything is :attr:`preamble`).

        Setter accepts a sequence of bare comment texts and replaces
        the current epilogue. Assign ``()`` to remove. Raises
        :class:`TOMLError` if called with a non-empty value on a
        document with no structural content.
        """
        if not self._doc_node.has_content():
            return ()
        return _extract_trailing_comment_block(self._doc_node.trailing_trivia)

    @epilogue.setter
    def epilogue(self, value: Sequence[str]) -> None:
        if not self._doc_node.has_content():
            if value:
                msg = (
                    "cannot set epilogue on a document with no structural "
                    "content; use preamble instead"
                )
                raise TOMLError(msg)
            return
        _replace_trailing_comment_block(self._doc_node.trailing_trivia, value, "")


# ---------------------------------------------------------------------------
# Array (inline) and AoT (array of tables)
# ---------------------------------------------------------------------------


class Array(list[Any]):
    """Inline TOML array exposed as a real :class:`list`.

    Every standard list mutator is overridden so the underlying CST
    stays in sync. Existing handles to nested ``Array``/``Table`` values
    that were *not* removed remain valid; handles to removed/replaced
    elements become detached.
    """

    __slots__ = ("_attached", "_indent", "_node", "_style")

    def __init__(
        self,
        items: Iterable[object] | ArrayNode = (),
        *,
        multiline: bool = False,
        indent: str = "    ",
    ) -> None:
        """Construct a standalone array or wrap an existing CST node.

        Public use: ``Array([1, 2, 3])`` builds an inline array;
        ``Array([1, 2, 3], multiline=True)`` lays it out one item per
        line with ``indent`` indentation. Such an array is *detached*
        until assigned into a document (``doc[k] = arr``).

        Passing an :class:`ArrayNode` directly is the internal
        attached-construction path used by the parser and CST walkers.
        """
        if isinstance(items, ArrayNode):
            self._node = items
            self._attached = True
        else:
            from tomlrt._synthesise import _list_to_array_node  # noqa: PLC0415

            self._node = _list_to_array_node(list(items))  # type: ignore[arg-type]
            self._attached = False
        self._style = _sample_separator_style(
            self._node.items,
            self._node.final_trivia,
        )
        self._indent = indent
        super().__init__(_materialise_array(self._node))
        if not self._attached and multiline:
            self.set_multiline(multiline=True, indent=indent)

    def _detach(self) -> None:
        self._attached = False
        for v in self:
            if isinstance(v, (Table, AoT, Array)):
                v._detach()  # noqa: SLF001

    # ------------------------------------------------------------------
    # CST <-> list synchronisation helpers
    # ------------------------------------------------------------------

    def _resync(self) -> None:
        """Rebuild the public list from the CST after a structural change."""
        list.clear(self)
        list.extend(self, _materialise_array(self._node))

    def __copy__(self) -> Array:
        return self.__deepcopy__({})

    def __deepcopy__(self, memo: dict[int, object]) -> Array:
        new = Array.__new__(Array)
        memo[id(self)] = new
        new._node = deepcopy(self._node, memo)  # noqa: SLF001
        new._style = deepcopy(self._style, memo)  # noqa: SLF001
        new._indent = self._indent  # noqa: SLF001
        new._attached = self._attached  # noqa: SLF001
        list.__init__(new, _materialise_array(new._node))  # noqa: SLF001
        return new

    def _rebuild_separators(self) -> None:
        _apply_separator_style(self._node, self._style)

    def _rebuild_with_leadings(
        self,
        leadings: Sequence[Sequence[str]],
    ) -> None:
        """Apply separator style and restore an explicitly-given leadings list.

        Used by reorder operations that mutate ``items`` and need to keep
        the per-item leading-comment blocks aligned with their (possibly
        moved) items rather than with the on-disk storage slots. The
        ``leadings`` list must be snapshotted **before** the items list
        is reordered, then transformed in parallel.
        """
        _apply_separator_style(self._node, self._style)
        _write_item_leadings(self._node.items, leadings)

    @staticmethod
    def _make_item(value: object, *, with_comma: bool) -> ArrayItem:
        from tomlrt._nodes import ArrayItem  # noqa: PLC0415

        return ArrayItem(
            leading=Trivia(),
            value=value_to_node(value),
            trailing=Trivia(),
            has_comma=with_comma,
            post_comma_trivia=Trivia(),
        )

    @property
    def comments(self) -> MutableMapping[int, str]:
        """Live mapping of ``index -> end-of-line comment text``.

        Only items that currently carry an EOL comment are present.
        Setting ``""`` removes the comment, mirroring ``del``.
        Reads return the comment text without the leading ``#`` or
        surrounding whitespace.
        """
        return _ArrayCommentsView(self)

    @property
    def leading_comments(self) -> MutableMapping[int, tuple[str, ...]]:
        """Live mapping of ``index -> tuple of comment lines above item``.

        For item 0 the lines come from inside the array opening, before
        the first value. For item i > 0 they come from the trivia
        between item i-1's separator and item i's value.
        """
        return _ArrayLeadingCommentsView(self)

    @property
    def multiline(self) -> bool:
        """Whether the array currently renders across multiple lines."""
        return "\n" in self._style.inter_separator.render()

    @multiline.setter
    def multiline(self, multiline: bool) -> None:
        self.set_multiline(multiline=multiline)

    def set_multiline(self, *, multiline: bool, indent: str = "    ") -> Array:
        """Switch this array between single-line and multi-line layout.

        ``indent`` controls the per-item indentation when ``multiline``
        is true and is ignored otherwise. Returns ``self`` so calls may
        be chained.

        Switching to single-line layout when any item carries an EOL
        or leading comment is rejected with :class:`TOMLError`: a ``#``
        comment runs to end of line, so collapsing such an array would
        produce invalid TOML. Clear the offending comments first via
        :attr:`comments` / :attr:`leading_comments` if you really want
        a single-line layout.
        """
        if not multiline and _value_has_inner_comment(self._node):
            msg = (
                "cannot collapse a multi-line array to single-line: "
                "items carry EOL or leading comments which would "
                "produce invalid TOML; clear them first via .comments "
                "and .leading_comments"
            )
            raise TOMLError(msg)
        if multiline:
            inter = Trivia([NewlineNode("\n"), WhitespaceNode(indent)])
            self._style = _SeparatorStyle(
                open_pad=_clone_trivia(inter),
                inter_separator=_clone_trivia(inter),
                trailing_comma=True,
                close_pad=Trivia([NewlineNode("\n")]),
            )
            self._indent = indent
        else:
            self._style = _SeparatorStyle(
                open_pad=Trivia(),
                inter_separator=Trivia([WhitespaceNode(" ")]),
                trailing_comma=False,
                close_pad=Trivia(),
            )
        self._rebuild_separators()
        return self

    # ------------------------------------------------------------------
    # Mutators (override every one)
    # ------------------------------------------------------------------

    @override
    def append(self, value: object) -> None:
        new_item = self._make_item(value, with_comma=False)
        self._node.items.append(new_item)
        _apply_separator_after_append(self._node, self._style)
        list.append(self, _value_for(new_item.value))

    @override
    def extend(self, values: Iterable[object]) -> None:
        new_items = [self._make_item(v, with_comma=False) for v in list(values)]
        if not new_items:
            return
        self._node.items.extend(new_items)
        _apply_separator_after_append(self._node, self._style, len(new_items))
        list.extend(self, [_value_for(it.value) for it in new_items])

    @override
    def insert(self, index: SupportsIndex, value: object) -> None:
        idx = operator.index(index)
        leadings = _snapshot_item_leadings(self._node.items)
        new_item = self._make_item(value, with_comma=False)
        self._node.items.insert(idx, new_item)
        leadings.insert(idx, ())
        self._rebuild_with_leadings(leadings)
        list.insert(self, idx, _value_for(new_item.value))

    @overload
    def __setitem__(self, index: SupportsIndex, value: object) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[object]) -> None: ...
    @override
    def __setitem__(
        self,
        index: SupportsIndex | slice,
        value: object,
    ) -> None:
        if isinstance(index, slice):
            if not isinstance(value, Iterable):
                msg = "must assign iterable to extended slice"
                raise TypeError(msg)
            leadings = _snapshot_item_leadings(self._node.items)
            new_items = [self._make_item(v, with_comma=False) for v in list(value)]
            self._node.items[index] = new_items
            leadings[index] = [() for _ in new_items]
            self._rebuild_with_leadings(leadings)
        else:
            i = operator.index(index)
            self._node.items[i].value = value_to_node(value)
            self._rebuild_separators()
        self._resync()

    @override
    def __delitem__(self, index: SupportsIndex | slice) -> None:
        leadings = _snapshot_item_leadings(self._node.items)
        if isinstance(index, slice):
            del self._node.items[index]
            del leadings[index]
        else:
            i = operator.index(index)
            del self._node.items[i]
            del leadings[i]
        self._rebuild_with_leadings(leadings)
        self._resync()

    @override
    def pop(self, index: SupportsIndex = -1) -> Any:
        leadings = _snapshot_item_leadings(self._node.items)
        i = operator.index(index)
        item = self._node.items.pop(i)
        del leadings[i]
        self._rebuild_with_leadings(leadings)
        self._resync()
        popped = _value_for(item.value)
        # The wrapper was constructed from an item that is no longer
        # in the document; reflect that on the returned object so that
        # later reassignment doesn't trigger the cross-doc clone path
        # (and so that `_attached` honestly describes reality).
        if isinstance(popped, (Table, Array)):
            popped._detach()  # noqa: SLF001
        return popped

    @override
    def remove(self, value: object) -> None:
        idx = list.index(self, value)
        del self[idx]

    @override
    def clear(self) -> None:
        self._node.items.clear()
        self._rebuild_separators()
        self._resync()

    @override
    def reverse(self) -> None:
        n = len(self._node.items)
        self._reorder_items(range(n - 1, -1, -1))

    @override
    def sort(
        self,
        *,
        key: Callable[[Any], object] | None = None,
        reverse: bool = False,
    ) -> None:
        values = _materialise_array(self._node)
        sort_key: Callable[[int], Any] = (
            (lambda i: values[i]) if key is None else (lambda i: key(values[i]))
        )
        self._reorder_items(sorted(range(len(values)), key=sort_key, reverse=reverse))

    def _reorder_items(self, perm: Iterable[int]) -> None:
        """Apply an index permutation to ``items`` and their leadings.

        Each item's leading carries that item's preceding comment/blank
        layout, so it must travel with the item. Inter-item separators
        get rebuilt afterwards for the new order.
        """
        order = list(perm)
        items = self._node.items
        leadings = _snapshot_item_leadings(items)
        items[:] = [items[i] for i in order]
        self._rebuild_with_leadings([leadings[i] for i in order])
        self._resync()

    @override
    def __iadd__(self, values: Iterable[object]) -> Self:
        self.extend(values)
        return self

    @override
    def __imul__(self, count: SupportsIndex) -> Self:
        n = operator.index(count)
        if n <= 0:
            self.clear()
        else:
            base = list(self._node.items)
            base_leadings = _snapshot_item_leadings(base)
            for _ in range(n - 1):
                self._node.items.extend(deepcopy(item) for item in base)
            leadings = list(base_leadings) * n
            self._rebuild_with_leadings(leadings)
            self._resync()
        return self

    def to_list(self) -> list[Any]:
        """Return a deep, plain-Python copy of this array.

        Walks recursively, converting nested :class:`Table` /
        :class:`AoT` / :class:`Array` views into ordinary
        :class:`dict` / :class:`list` containers. Scalars are
        returned as-is.
        """
        return [_to_plain(v) for v in self]

    # ------------------------------------------------------------------
    # Typed accessors for nested values. Mirror Table.array/.table.
    # ------------------------------------------------------------------

    def array(self, index: SupportsIndex) -> Array:
        """Return ``self[index]`` typed as a nested :class:`Array`."""
        value = self[index]
        if not isinstance(value, Array):
            type_name = type(value).__name__
            msg = f"item {operator.index(index)} is a {type_name}, not an Array"
            raise TypeError(msg)
        return value

    @overload
    def get_array(self, index: SupportsIndex) -> Array | None: ...
    @overload
    def get_array(self, index: SupportsIndex, default: _T) -> Array | _T: ...
    def get_array(self, index: SupportsIndex, default: object = None) -> object:
        """Like :meth:`array`, but returns ``default`` if ``index`` is out of range.

        Wrong-type entries still raise :class:`TypeError`.
        """
        try:
            value = self[index]
        except IndexError:
            return default
        if not isinstance(value, Array):
            type_name = type(value).__name__
            msg = f"item {operator.index(index)} is a {type_name}, not an Array"
            raise TypeError(msg)
        return value

    def table(self, index: SupportsIndex) -> Table:
        """Return ``self[index]`` typed as a nested :class:`Table`."""
        value = self[index]
        if not isinstance(value, Table):
            msg = (
                f"item {operator.index(index)} is a {type(value).__name__}, not a Table"
            )
            raise TypeError(msg)
        return value

    @overload
    def get_table(self, index: SupportsIndex) -> Table | None: ...
    @overload
    def get_table(self, index: SupportsIndex, default: _T) -> Table | _T: ...
    def get_table(self, index: SupportsIndex, default: object = None) -> object:
        """Like :meth:`table`, but returns ``default`` if ``index`` is out of range.

        Wrong-type entries still raise :class:`TypeError`.
        """
        try:
            value = self[index]
        except IndexError:
            return default
        if not isinstance(value, Table):
            msg = (
                f"item {operator.index(index)} is a {type(value).__name__}, not a Table"
            )
            raise TypeError(msg)
        return value


class AoT(list[Table]):
    """Array-of-tables, e.g. ``[[products]]`` repeated.

    Subclass of :class:`list`; supports basic mutation (append/insert
    of dict-shaped or :class:`Table` entries) by synthesizing fresh
    ``[[path]]`` sections in the underlying CST.
    """

    __slots__ = ("_attached", "_doc_node", "_path")

    def __init__(
        self,
        entries: Iterable[Mapping[str, object]] = (),
    ) -> None:
        """Construct a standalone array-of-tables.

        Each of ``entries`` (a dict-shaped mapping or :class:`Table`)
        is materialised into a ``[[_]]`` section in an internal orphan
        document, so all the usual list mutators (``append``, ``insert``,
        ``extend``, ``pop``, ``__setitem__`` of slots) keep working
        pre-assignment. On ``doc[k] = aot``, the pending sections are
        rewritten to ``[[k]]`` and merged into the target document.

        The 3-argument internal form (``doc_node``, ``path``,
        ``tables``) used by the parser/CST walkers remains available
        via :meth:`_attached_to`.
        """
        path: tuple[str, ...] = ("_",)
        doc_node = DocumentNode(sections=[])
        super().__init__()
        self._doc_node: DocumentNode = doc_node
        self._path = path
        self._attached = False
        for entry in entries:
            self._insert_at(len(self), entry)

    @classmethod
    def _attached_to(
        cls,
        doc_node: DocumentNode,
        path: tuple[str, ...],
        tables: list[Table],
    ) -> AoT:
        obj = cls.__new__(cls)
        list.__init__(obj, tables)
        obj._doc_node = doc_node  # noqa: SLF001
        obj._path = path  # noqa: SLF001
        obj._attached = True  # noqa: SLF001
        return obj

    def _detach(self, doc_node: DocumentNode | None = None) -> None:
        if not self._attached:
            return
        if doc_node is None:
            doc_node = DocumentNode(sections=_aot_owned_sections(self))
        self._attached = False
        self._doc_node = doc_node
        for v in self:
            v._detach(doc_node)  # noqa: SLF001

    # ------------------------------------------------------------------
    # CST <-> list synchronisation
    # ------------------------------------------------------------------

    def _own_sections(self) -> list[SectionNode]:
        """Sections that act as the [[path]] entry headers (in doc order)."""
        return [
            s
            for s in self._doc_node.sections
            if s.header is not None
            and s.header.kind == "array"
            and s.header.key.path == self._path
        ]

    def _own_blocks(self) -> tuple[int, list[list[SectionNode]]]:
        """Each entry's [header, *owned-subsections] block, plus splice index.

        Reordering operations (reverse / sort) require entries to occupy a
        contiguous run of ``_doc_node.sections``. If unrelated sections sit
        between two entries this raises, since permuting blocks would change
        the meaning of those interleaved sections.
        """
        own = self._own_sections()
        if not own:
            return 0, []
        sections = self._doc_node.sections
        blocks: list[list[SectionNode]] = [
            self._doc_node.aot_entry_block(header) for header in own
        ]
        start = sections.index(blocks[0][0])
        cursor = start
        for block in blocks:
            if sections[cursor] is not block[0]:
                msg = (
                    "cannot reorder AoT entries: unrelated sections are "
                    "interleaved between entries"
                )
                raise RuntimeError(msg)
            cursor += len(block)
        return start, blocks

    def _resync(self) -> None:
        # Preserve identity for entries whose anchor section is unchanged.
        existing: dict[int, _StdTable] = {}
        for entry in self:
            if isinstance(entry, _StdTable) and entry._anchor is not None:  # noqa: SLF001
                existing[id(entry._anchor)] = entry  # noqa: SLF001
        own = self._own_sections()
        new_entries: list[Table] = []
        kept: set[int] = set()
        for s in own:
            cached = existing.get(id(s))
            if cached is not None:
                # A cached entry is, by construction, an entry of *this*
                # AoT, so it shares our home. Reconcile in case the AoT
                # was just rehomed (e.g. live-attached into a document).
                cached._doc_node = self._doc_node  # noqa: SLF001
                cached._path = self._path  # noqa: SLF001
                kept.add(id(cached))
                new_entries.append(cached)
            else:
                new_entries.append(
                    _StdTable(
                        self._doc_node,
                        self._path,
                        anchor=s,
                    ),
                )
        # Detach any previous entries that are no longer in the AoT.
        for entry in self:
            if id(entry) not in kept:
                entry._detach()  # noqa: SLF001
        list.clear(self)
        list.extend(self, new_entries)

    def __copy__(self) -> AoT:
        return self.__deepcopy__({})

    def __deepcopy__(self, memo: dict[int, object]) -> AoT:
        new = AoT.__new__(AoT)
        memo[id(self)] = new
        new._doc_node = deepcopy(self._doc_node, memo)  # noqa: SLF001
        new._path = self._path  # noqa: SLF001
        new._attached = self._attached  # noqa: SLF001
        list.__init__(new)
        new._resync()  # noqa: SLF001
        return new

    def _make_header_section(self) -> SectionNode:
        return _new_section(self._path, kind="array")

    def _populate_via_view(
        self,
        header: SectionNode,
        value: Mapping[str, object],
    ) -> None:
        """Install ``value``'s items into the AoT entry rooted at ``header``.

        Routes each KV through a temporary :class:`_StdTable` view
        scoped (via ``owner_anchor``) to the new entry's block, so
        flavoured values (``SectionSpec``, ``AoT``, layout-bearing
        ``Array``) take their structural install paths -- a nested
        ``Table.section`` becomes ``[path.k]``, a nested ``AoT``
        becomes ``[[path.k]]`` -- instead of being inlined by the
        synthesiser.
        """
        view = _StdTable(
            self._doc_node,
            self._path,
            anchor=header,
            owner_anchor=header,
        )
        for k, v in value.items():
            view[k] = v

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    @override
    def append(self, value: Table | Mapping[str, object]) -> None:
        self._insert_at(len(self), value)

    def add(self, entry: Mapping[str, object] = MappingProxyType({})) -> Table:
        """Append ``entry`` and return the new :class:`Table` view.

        Convenience over :meth:`append` for the common build-and-mutate
        idiom: ``pkg = aot.add({"name": "foo"}); pkg.set_table(...)``.
        ``entry`` defaults to an empty mapping, so ``aot.add()`` adds a
        blank entry and returns it for further population.
        """
        self._insert_at(len(self), entry)
        return self[-1]

    def to_list(self) -> list[dict[str, Any]]:
        """Return a deep, plain-Python copy of this array-of-tables.

        Each entry is converted to an ordinary :class:`dict` (with
        nested views recursively flattened to plain containers). The
        result shares no mutable state with the document.
        """
        return [t.to_dict() for t in self]

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
        self._validate_entry(value)
        own = self._own_sections()
        n = len(own)
        if py_index < 0:
            py_index += n
        py_index = max(0, min(py_index, n))
        sections = self._doc_node.sections
        # Pick an insertion point first; blank-line decision depends on it.
        if py_index == n:
            # Append: land after the last [[path]] entry's owned range,
            # or at end of doc if no entries exist yet.
            if own:
                tail = self._doc_node.aot_entry_block(own[-1])[-1]
                insert_idx = sections.index(tail) + 1
            else:
                insert_idx = len(sections)
        else:
            insert_idx = sections.index(own[py_index])
        # Build the entry's block. ``_StdTable`` sources are deep-cloned
        # so per-KV trivia, sub-section headers, and nested AoTs survive
        # verbatim. Plain mappings get an empty header spliced in first
        # and then populated through a scoped view, so flavoured values
        # (``SectionSpec``, ``AoT``) install as proper sub-sections /
        # sub-AoTs under the new entry instead of being inlined.
        if isinstance(value, _StdTable):
            new_block = _clone_table_sections(value, self._path, head_kind="array")
            if not new_block:
                new_block = [self._make_header_section()]
        else:
            new_block = [self._make_header_section()]
        new_sec = new_block[0]
        # Insert a blank-line separator before the new header iff there
        # is already rendered content preceding it. When existing
        # siblings already share a uniform spacing style, copy that;
        # otherwise default to blank-separated (canonical TOML style).
        sibling_leadings = [
            sec.header.leading for sec in own[1:] if sec.header is not None
        ]
        add_blank = _first_gap_is_blank(sibling_leadings) if sibling_leadings else True
        preceding_has_content = any(
            s.header is not None or s.entries for s in sections[:insert_idx]
        )
        assert new_sec.header is not None
        if preceding_has_content and add_blank:
            _prepend_blank_line(new_sec.header.leading)
        # Symmetric: when inserting before existing content, ensure the
        # next section's header carries a blank-line separator from the
        # new one so two ``[[..]]`` headers don't render glued together.
        if py_index < n and add_blank:
            next_hdr = sections[insert_idx].header
            if next_hdr is not None:
                _prepend_blank_line(next_hdr.leading)
        self._doc_node.adopt_preamble_into(new_sec.header.leading)
        sections[insert_idx:insert_idx] = new_block
        if not isinstance(value, _StdTable):
            assert isinstance(value, Mapping)
            self._populate_via_view(new_sec, value)
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
        popped = self[i]
        self._orphan_and_remove([(popped, own[i])])
        return popped

    @override
    def clear(self) -> None:
        own = self._own_sections()
        self._orphan_and_remove(list(zip(list(self), own, strict=True)))

    def _orphan_and_remove(
        self,
        targets: list[tuple[Table, SectionNode]],
    ) -> None:
        """Orphan ``targets`` then strip their blocks from the live doc.

        Each cached entry view is rehomed onto its own
        :class:`DocumentNode` carrying the entry's full
        ``aot_entry_block`` *before* ``remove_sections`` runs --
        otherwise ``_resync``'s default detach pass would search a
        section list that's already been emptied of the block, and
        any nested ``[a.sub]``-style sub-sections of the dying entry
        would be silently dropped from the orphaned view.
        """
        to_remove: set[SectionNode] = set()
        for entry, sec in targets:
            block = self._doc_node.aot_entry_block(sec)
            entry._detach(DocumentNode(sections=list(block)))  # noqa: SLF001
            to_remove.update(block)
        self._doc_node.remove_sections(to_remove)
        self._resync()

    @override
    def __delitem__(self, index: SupportsIndex | slice) -> None:
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            for i in sorted(indices, reverse=True):
                self.pop(i)
        else:
            self.pop(index)

    @staticmethod
    def _validate_entry(value: object) -> None:
        if not isinstance(value, Mapping):
            msg = "AoT entry must be a mapping"
            raise TypeError(msg)

    @overload
    def __setitem__(
        self, index: SupportsIndex, value: Mapping[str, object]
    ) -> None: ...
    @overload
    def __setitem__(
        self,
        index: slice,
        value: Iterable[Mapping[str, object]],
    ) -> None: ...
    @override
    def __setitem__(
        self,
        index: SupportsIndex | slice,
        value: Mapping[str, object] | Iterable[Mapping[str, object]],
    ) -> None:
        if isinstance(index, slice):
            new_values: list[Any] = list(value)
            for v in new_values:
                self._validate_entry(v)
            indices = range(*index.indices(len(self)))
            if index.step not in (None, 1):
                if len(new_values) != len(indices):
                    msg = (
                        f"attempt to assign sequence of size {len(new_values)} "
                        f"to extended slice of size {len(indices)}"
                    )
                    raise ValueError(msg)
                for i, v in zip(indices, new_values, strict=True):
                    self[i] = v
                return
            del self[index]
            for offset, v in enumerate(new_values):
                self.insert(indices.start + offset, v)
            return
        self._validate_entry(value)
        assert isinstance(value, Mapping)
        # Replacement is always del + insert: the splice path used by
        # ``insert`` already handles cloned-Table sources (preserving
        # per-KV trivia, sub-sections, and nested AoTs) as well as
        # plain mappings. Snapshot the existing slot's header leading
        # so the visual context (comments / blank lines above the
        # ``[[path]]`` header) survives the swap regardless of source.
        i = operator.index(index)
        n = len(self)
        if i < 0:
            i += n
        if i < 0 or i >= n:
            msg = f"AoT assignment index out of range: {index}"
            raise IndexError(msg)
        own = self._own_sections()
        hdr = own[i].header
        prior_leading = hdr.leading if hdr is not None else None
        del self[i]
        self.insert(i, value)
        _apply_prior_leading([self._own_sections()[i]], prior_leading)

    @override
    def __iadd__(self, values: Iterable[Mapping[str, object]]) -> Self:  # type: ignore[override]
        self.extend(values)
        return self

    @override
    def remove(self, value: Mapping[str, object]) -> None:
        for i, entry in enumerate(self):
            if entry == value:
                del self[i]
                return
        msg = f"{value!r} not in list"
        raise ValueError(msg)

    @override
    def __imul__(self, count: SupportsIndex) -> Self:
        n = operator.index(count)
        if n <= 0:
            self.clear()
            return self
        if n == 1:
            return self
        start, blocks = self._own_blocks()
        base: list[SectionNode] = [s for block in blocks for s in block]
        # The duplicated first block needs (a) the inter-repetition
        # separator -- so doubling a blank-line-separated doc stays
        # visually consistent -- and (b) its own deep-copied leading
        # comment (the comment logically describes the entry, not the
        # slot, so it must travel with the duplicated block).
        # Sample (a) from blocks[1].leading with the comment stripped;
        # fall back to a blank line when there's no second block to
        # sample, to avoid gluing copies header-to-header.
        if len(blocks) >= 2:
            inter_separator = Trivia(
                pieces=list(self._block_leading(blocks[1]).pieces),
            )
            _replace_trailing_comment_block(inter_separator, (), "")
        else:
            inter_separator = Trivia(pieces=[NewlineNode("\n")])
        repeated = list(base)
        for _ in range(n - 1):
            copy_blocks: list[list[SectionNode]] = [
                [deepcopy(s) for s in block] for block in blocks
            ]
            first_leading = self._block_leading(copy_blocks[0])
            first_leading.pieces = [
                *deepcopy(inter_separator).pieces,
                *first_leading.pieces,
            ]
            repeated.extend(s for block in copy_blocks for s in block)
        self._doc_node.sections[start : start + len(base)] = repeated
        self._resync()
        return self

    @override
    def reverse(self) -> None:
        start, blocks = self._own_blocks()
        if not blocks:
            return
        self._reorder_blocks(start, blocks, range(len(blocks) - 1, -1, -1))

    @override
    def sort(
        self,
        *,
        key: Callable[[Table], object] | None = None,
        reverse: bool = False,
    ) -> None:
        start, blocks = self._own_blocks()
        if not blocks:
            return
        entries = list(self)
        sort_key: Callable[[int], Any] = (
            (lambda i: entries[i]) if key is None else (lambda i: key(entries[i]))
        )
        self._reorder_blocks(
            start,
            blocks,
            sorted(range(len(blocks)), key=sort_key, reverse=reverse),
        )

    def _reorder_blocks(
        self,
        start: int,
        blocks: list[list[SectionNode]],
        perm: Iterable[int],
    ) -> None:
        """Apply an index permutation to this AoT's section blocks.

        A block's trailing-comment chunk belongs to its entry and
        travels with the block; the inter-entry separator pattern
        belongs to the slot and stays in place.
        """
        order = list(perm)
        end = start + sum(len(b) for b in blocks)
        leadings = [self._block_leading(b) for b in blocks]
        entry_comments = [_extract_trailing_comment_block(L) for L in leadings]
        new_blocks = [blocks[i] for i in order]
        new_comments = [entry_comments[i] for i in order]
        for block, leading in zip(new_blocks, leadings, strict=True):
            self._set_block_leading(block, leading)
        for block, comment in zip(new_blocks, new_comments, strict=True):
            _replace_trailing_comment_block(self._block_leading(block), comment, "")
        self._doc_node.sections[start:end] = [s for block in new_blocks for s in block]
        self._resync()

    @staticmethod
    def _block_leading(block: list[SectionNode]) -> Trivia:
        header = block[0].header
        assert header is not None
        return header.leading

    @staticmethod
    def _set_block_leading(block: list[SectionNode], leading: Trivia) -> None:
        header = block[0].header
        assert header is not None
        header.leading = leading


# ---------------------------------------------------------------------------
# View / aggregator
# ---------------------------------------------------------------------------


def _iter_table(
    doc_node: DocumentNode,
    path: tuple[str, ...],
    *,
    pool: list[SectionNode],
    anchor: SectionNode | None = None,
    owner_anchor: SectionNode | None = None,
    extras: list[tuple[tuple[str, ...], KeyValueNode]] | None = None,
) -> Iterator[tuple[str, TomlValue]]:
    # ``pool`` is the section list this table draws from: the whole
    # document at the root, the AoT-owned range for an AoT-narrowed
    # table, or — during construction — the per-head bucket the parent
    # already partitioned for us.

    # Sections whose entries are "direct" key/values at this exact path.
    direct_secs: list[SectionNode]
    if anchor is not None:
        direct_secs = [anchor]
    elif path == ():
        direct_secs = [s for s in pool if s.header is None]
    else:
        direct_secs = [
            s
            for s in pool
            if s.header is not None
            and s.header.kind == "table"
            and s.header.key.path == path
        ]
    direct_ids = {id(s) for s in direct_secs}

    name_order: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        if name not in seen:
            seen.add(name)
            name_order.append(name)

    direct_kvs_by_head: dict[str, list[KeyValueNode]] = {}
    extras_by_head: dict[str, list[tuple[tuple[str, ...], KeyValueNode]]] = {}
    aot_by_head: dict[str, list[SectionNode]] = {}
    sub_by_head: dict[str, list[SectionNode]] = {}

    # Single physical-order walk: each section either contributes direct
    # entries (header == path) or registers a sub-table / sub-AoT head.
    # First-appearance order matches what tomllib produces.
    plen = len(path)
    for sec in pool:
        if id(sec) in direct_ids:
            for entry in sec.entries:
                head = entry.key.path[0]
                direct_kvs_by_head.setdefault(head, []).append(entry)
                _add(head)
            continue
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

    # Extras (dotted-key prefixes inherited from an ancestor) have no
    # physical position; tack their heads on at the end.
    if extras:
        for rel_path, entry in extras:
            head = rel_path[0]
            extras_by_head.setdefault(head, []).append((rel_path, entry))
            _add(head)

    for head in name_order:
        direct_kvs = direct_kvs_by_head.get(head, [])
        head_extras = extras_by_head.get(head, [])
        aot_secs = aot_by_head.get(head, [])
        sub_secs = sub_by_head.get(head, [])

        if aot_secs:
            tables: list[Table] = [
                _StdTable(doc_node, (*path, head), anchor=s) for s in aot_secs
            ]
            yield head, AoT._attached_to(doc_node, (*path, head), tables)  # noqa: SLF001
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
        for rel_path, kv in head_extras:
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
            # Pure dotted from this section level. Prefix is relative
            # to the host section (where dotted KVs live), not the
            # absolute logical path.
            yield (
                head,
                _DottedSubTable(
                    depth=1,
                    host=_SectionDottedHost(direct_secs),
                    prefix=(head,),
                ),
            )
            continue

        # Merged view at path + (head,). For non-AoT children we hand
        # over the per-head bucket as their pool and the dotted-key
        # extras we already collected (one head segment stripped) so
        # they skip a full rescan and ancestor walk. AoT-narrowed
        # children fall through to the slow path: their pool depends
        # on the AoT entry's owned range, computed lazily via
        # ``_scope``.
        child_owner = anchor or owner_anchor
        if child_owner is None:
            child_extras = [(kv.key.path[1:], kv) for kv in nested_kvs]
            child_extras.extend((rp[1:], kv) for rp, kv in nested_extras)
            yield (
                head,
                _StdTable(
                    doc_node,
                    (*path, head),
                    _pool=sub_secs,
                    _extras=child_extras,
                ),
            )
        else:
            yield (
                head,
                _StdTable(doc_node, (*path, head), owner_anchor=child_owner),
            )


__all__ = [
    "AoT",
    "Array",
    "Document",
    "Scalar",
    "Table",
    "TomlValue",
]
