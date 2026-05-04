"""Semantic validator for parsed TOML documents.

The CST produced by `tomlrt._parser` is purely syntactic: it tells you
what the source said, not whether the source said anything legal at the
table level. TOML's table semantics include rules that span multiple
syntactic constructs and many lines of source:

- a key bound as a value cannot later be opened as a table;
- a `[H]` table header cannot redefine a previously opened table;
- a `[[H]]` array-of-tables header opens a fresh entry whose
  per-entry key tracking is independent of prior entries;
- a dotted key cannot extend an explicitly defined table or AoT;
- inline tables locally enforce duplicate-key and dotted-prefix
  rules, and their nested keys must also be visible to later
  cross-section conflict checks.

`_Validator` owns the bookkeeping these rules require and is invoked
by the parser at three points: when a `[H]` / `[[H]]` header has just
been parsed, when a `key = value` line has just been built, and (with
caller-supplied per-table state) when a key inside an inline table is
about to be added.

Diagnostics are raised through a caller-supplied error builder so the
validator stays decoupled from the scanner: every `TOMLParseError`
still flows through the same line/column machinery, but the validator
itself depends on nothing more than the CST node types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tomlrt._nodes import ArrayNode, InlineTableNode

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomlrt._errors import TOMLParseError
    from tomlrt._nodes import KeyValueNode, TableHeaderNode

    ErrorBuilder = Callable[..., TOMLParseError]


class _Validator:
    __slots__ = (
        "_aot_paths",
        "_aot_subpaths",
        "_current_section",
        "_dotted_paths",
        "_error",
        "_explicit_table_paths",
        "_implicit_table_paths",
        "_value_paths",
    )

    def __init__(self, error_builder: ErrorBuilder) -> None:
        # ``error_builder(message, *, at=offset)`` returns a fully-formed
        # ``TOMLParseError``. The parser supplies ``_Scanner.error`` so
        # diagnostics carry the correct line/column; tests can pass any
        # callable with the same shape.
        self._error = error_builder
        # Persistent structural facts (not cleared when re-entering an AoT).
        self._explicit_table_paths: set[tuple[str, ...]] = set()
        self._implicit_table_paths: set[tuple[str, ...]] = set()
        self._aot_paths: set[tuple[str, ...]] = set()
        # Per-AoT-entry: cleared (for paths under H) when [[H]] opens a new entry.
        self._value_paths: set[tuple[str, ...]] = set()
        self._dotted_paths: set[tuple[str, ...]] = set()
        # Index from each active AoT path to all sub-paths registered under
        # it across the 5 sets above. Lets `_reset_scope_under` work in
        # O(descendants) instead of O(all tracked paths).
        self._aot_subpaths: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
        self._current_section: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Headers
    # ------------------------------------------------------------------

    def enter_header(self, header: TableHeaderNode, *, at: int) -> None:
        """Validate and register a `[H]` or `[[H]]` header.

        Updates the current-section tracking that subsequent
        `record_keyvalue` calls read from.
        """
        path = header.key.path
        kind = header.kind
        # Prefix overlaps with a bound value would mean overwriting a scalar
        # (or an inline-table value) with a table — always invalid.
        for i in range(1, len(path)):
            prefix = path[:i]
            if prefix in self._value_paths:
                joined = ".".join(prefix)
                msg = f"cannot use {joined!r} as a table: already defined as a value"
                raise self._error(msg, at=at)
        if path in self._value_paths:
            joined = ".".join(path)
            msg = f"cannot define {joined!r} as a table: already defined as a value"
            raise self._error(msg, at=at)
        if path in self._dotted_paths:
            joined = ".".join(path)
            msg = (
                f"cannot define {joined!r} as a table: already created via dotted keys"
            )
            raise self._error(msg, at=at)
        if kind == "table":
            if path in self._explicit_table_paths:
                msg = f"redefinition of table {'.'.join(path)!r}"
                raise self._error(msg, at=at)
            if path in self._aot_paths:
                joined = ".".join(path)
                msg = f"cannot redefine array-of-tables {joined!r} as a normal table"
                raise self._error(msg, at=at)
            self._explicit_table_paths.add(path)
            self._track(path)
        else:  # array-of-tables
            if path in self._explicit_table_paths:
                msg = f"cannot redefine table {'.'.join(path)!r} as an array-of-tables"
                raise self._error(msg, at=at)
            if path in self._implicit_table_paths and path not in self._aot_paths:
                msg = (
                    f"cannot define {'.'.join(path)!r} as an array-of-tables: "
                    "already used as an implicit table"
                )
                raise self._error(msg, at=at)
            # Opening a new AoT entry at `path` invalidates any per-entry
            # tracking that was scoped to the previous entry.
            self._reset_scope_under(path)
            self._aot_paths.add(path)
            self._track(path)
        # Intermediate prefixes become implicit tables (only mark new ones).
        for i in range(1, len(path)):
            sub = path[:i]
            if (
                sub not in self._explicit_table_paths
                and sub not in self._aot_paths
                and sub not in self._implicit_table_paths
            ):
                self._implicit_table_paths.add(sub)
                self._track(sub)
        self._current_section = path

    # ------------------------------------------------------------------
    # Key/value lines
    # ------------------------------------------------------------------

    def record_keyvalue(self, kv: KeyValueNode, *, at: int) -> None:
        """Validate and register a `key = value` line.

        ``at`` is the source offset to report errors against (the
        parser supplies its post-line cursor, matching prior
        behaviour).
        """
        section = self._current_section
        path = kv.key.path
        full = section + path if section else path
        # Final-path conflicts.
        value_paths = self._value_paths
        if full in value_paths:
            msg = f"duplicate key {'.'.join(full)!r}"
            raise self._error(msg, at=at)
        if (
            full in self._explicit_table_paths
            or full in self._aot_paths
            or full in self._implicit_table_paths
            or full in self._dotted_paths
        ):
            msg = f"key {'.'.join(full)!r} already defined as a table"
            raise self._error(msg, at=at)
        # Intermediate-prefix conflicts (paths between section and full).
        slen = len(section)
        flen = len(full)
        if flen > slen + 1:
            for i in range(slen + 1, flen):
                sub = full[:i]
                if sub in value_paths:
                    msg = f"key {'.'.join(sub)!r} already defined as a value"
                    raise self._error(msg, at=at)
                if sub in self._explicit_table_paths:
                    joined = ".".join(sub)
                    msg = (
                        f"cannot extend explicitly-defined table {joined!r} "
                        "via dotted keys"
                    )
                    raise self._error(msg, at=at)
                if sub in self._aot_paths:
                    msg = (
                        f"cannot extend array-of-tables {'.'.join(sub)!r} "
                        "via dotted keys"
                    )
                    raise self._error(msg, at=at)
                self._dotted_paths.add(sub)
                self._track(sub)
        value_paths.add(full)
        self._track(full)
        # Inline-table values: register their nested key paths so
        # cross-section headers/keys see the conflicts. (Local
        # duplicate / dotted-prefix conflicts are caught at parse time
        # via `check_inline_key_conflict`.)
        value = kv.value
        if isinstance(value, InlineTableNode):
            self._register_inline_table(value, abs_prefix=full)
        elif isinstance(value, ArrayNode):
            for item in value.items:
                if isinstance(item.value, InlineTableNode):
                    self._register_inline_table(item.value, abs_prefix=None)

    # ------------------------------------------------------------------
    # Inline tables
    # ------------------------------------------------------------------

    def check_inline_key_conflict(
        self,
        path: tuple[str, ...],
        seen_values: set[tuple[str, ...]],
        seen_prefixes: set[tuple[str, ...]],
        *,
        at: int,
    ) -> None:
        """Validate ``path`` against keys already seen in this inline table.

        Mutates ``seen_prefixes`` to add the new path's strict prefixes.
        Caller is responsible for adding ``path`` itself to
        ``seen_values`` after successful validation. The two sets are
        per-inline-table local state, owned by the parser's
        ``_parse_inline_table`` loop — they must not bleed across
        sibling inline tables, so we don't keep them on the validator.
        """
        if path in seen_values:
            msg = f"duplicate key {'.'.join(path)!r} in inline table"
            raise self._error(msg, at=at)
        if path in seen_prefixes:
            msg = (
                f"key {'.'.join(path)!r} in inline table conflicts with "
                "an existing dotted-key prefix"
            )
            raise self._error(msg, at=at)
        for i in range(1, len(path)):
            sub = path[:i]
            if sub in seen_values:
                msg = f"inline-table key {'.'.join(sub)!r} already defined as a value"
                raise self._error(msg, at=at)
            seen_prefixes.add(sub)

    def _register_inline_table(
        self,
        table: InlineTableNode,
        *,
        abs_prefix: tuple[str, ...] | None,
    ) -> None:
        """Register an inline table's keys for cross-section conflict checks.

        When ``abs_prefix`` is given, exposes the inline table's keys
        to document-wide tracking. Walks nested inline tables either
        way (since arrays inside the inline table may themselves
        contain inline tables that need walking, even though their
        keys are not exposed at the top level).
        """
        for entry in table.entries:
            path = entry.key.path
            if abs_prefix is not None:
                full = abs_prefix + path
                self._value_paths.add(full)
                self._track(full)
                for i in range(1, len(path)):
                    sub = abs_prefix + path[:i]
                    self._dotted_paths.add(sub)
                    self._track(sub)
            sub_abs: tuple[str, ...] | None
            if isinstance(entry.value, InlineTableNode):
                sub_abs = (abs_prefix + path) if abs_prefix is not None else None
                self._register_inline_table(entry.value, abs_prefix=sub_abs)
            elif isinstance(entry.value, ArrayNode):
                for item in entry.value.items:
                    if isinstance(item.value, InlineTableNode):
                        self._register_inline_table(item.value, abs_prefix=None)

    # ------------------------------------------------------------------
    # AoT scope tracking
    # ------------------------------------------------------------------

    def _track(self, p: tuple[str, ...]) -> None:
        """Index ``p`` under its longest active-AoT-path ancestor, if any.

        Only paths registered here can be reset by ``_reset_scope_under``.
        Documents with no AoTs (the common case) skip this entirely.
        """
        aot_paths = self._aot_paths
        if not aot_paths:
            return
        for i in range(len(p) - 1, 0, -1):
            prefix = p[:i]
            if prefix in aot_paths:
                self._aot_subpaths.setdefault(prefix, []).append(p)
                return

    def _reset_scope_under(self, path: tuple[str, ...]) -> None:
        """Forget per-entry tracking for paths strictly under ``path``.

        Called when a new AoT entry at ``path`` is opened: prior entries'
        keys, dotted-prefix paths and explicit sub-table headers are
        replaced by the fresh entry's own. Runs in O(k) where k is the
        number of paths actually registered under ``path``.
        """
        subs = self._aot_subpaths.pop(path, None)
        if not subs:
            return
        nested_aots: list[tuple[str, ...]] = []
        for p in subs:
            if p in self._aot_paths:
                nested_aots.append(p)
            self._value_paths.discard(p)
            self._dotted_paths.discard(p)
            self._explicit_table_paths.discard(p)
            self._implicit_table_paths.discard(p)
            self._aot_paths.discard(p)
        for nested in nested_aots:
            self._reset_scope_under(nested)


__all__ = ["_Validator"]
