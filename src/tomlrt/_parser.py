"""TOML 1.0.0 parser producing a `tomlrt._nodes` CST.

Hand-written recursive-descent over a ``str`` plus an integer cursor.
Bulk character runs (whitespace, bare keys, string bodies, comment
bodies) are scanned with module-level compiled regexes so each run is
a single C call rather than a Python-level character loop.

The parser is responsible *only* for producing the physical CST. Logical
table semantics (duplicate-key detection across discontiguous headers,
dotted-key conflicts, etc.) are enforced here; the read-side wrappers
in `tomlrt._document` rely on this validation having happened.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tomlrt._nodes import (
    ArrayItem,
    ArrayNode,
    DocumentNode,
    InlineTableEntry,
    InlineTableNode,
    Key,
    KeyValueNode,
    SectionNode,
    TableHeaderNode,
    Trivia,
)
from tomlrt._scanner import _Scanner

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomlrt._nodes import (
        HeaderKind,
        KeyPart,
        ValueNode,
    )


class _Parser:
    __slots__ = (
        "_aot_paths",
        "_aot_subpaths",
        "_current_section",
        "_dotted_paths",
        "_explicit_table_paths",
        "_implicit_table_paths",
        "_sc",
        "_value_depth",
        "_value_paths",
    )

    # Hard cap on nested array / inline-table depth. Well above any
    # realistic data (tomllib, for reference, accepts whatever Python's
    # recursion limit allows; we pick a friendlier ceiling so the user
    # gets a TOMLParseError instead of a RecursionError).
    _MAX_VALUE_DEPTH = 100

    def __init__(self, src: str) -> None:
        self._sc = _Scanner(src)
        # Persistent structural facts (not cleared when re-entering an AoT).
        self._explicit_table_paths: set[tuple[str, ...]] = set()
        self._implicit_table_paths: set[tuple[str, ...]] = set()
        self._aot_paths: set[tuple[str, ...]] = set()
        # Per-AoT-entry: cleared (for paths under H) when [[H]] opens a new entry.
        self._value_paths: set[tuple[str, ...]] = set()
        self._dotted_paths: set[tuple[str, ...]] = set()
        # Index from each active AoT path to all sub-paths registered under
        # it across the 5 sets above. Lets ``_reset_scope_under`` work in
        # O(descendants) instead of O(all tracked paths).
        self._aot_subpaths: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
        self._current_section: tuple[str, ...] = ()
        self._value_depth = 0

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def parse(self) -> DocumentNode:
        doc = DocumentNode()
        current = SectionNode(header=None)
        doc.sections.append(current)
        sc = self._sc
        src = sc.src
        end = sc.end

        while sc.pos < end:
            leading = sc.scan_doc_trivia()
            pos = sc.pos
            if pos >= end:
                # leading is purely trailing trivia for the document.
                doc.trailing_trivia.pieces.extend(leading.pieces)
                break

            if src[pos] == "[":
                header = self._parse_header(leading)
                current = SectionNode(header=header)
                doc.sections.append(current)
                self._current_section = header.key.path
            else:
                kv = self._parse_key_value(leading)
                self._record_keyvalue(kv)
                current.entries.append(kv)

        return doc

    # ------------------------------------------------------------------
    # Headers
    # ------------------------------------------------------------------

    def _parse_header(self, leading: Trivia) -> TableHeaderNode:
        kind: HeaderKind
        if self._sc.starts_with("[["):
            self._sc.advance(2)
            kind = "array"
        else:
            assert self._sc.peek() == "["
            self._sc.advance(1)
            kind = "table"

        inner_pre = self._sc.scan_inline_ws()
        key = self._parse_key()
        inner_post = self._sc.scan_inline_ws()

        if kind == "array":
            if not self._sc.starts_with("]]"):
                msg = "expected ']]' to close array-of-tables header"
                raise self._sc.error(msg)
            self._sc.advance(2)
        else:
            if self._sc.peek() != "]":
                msg = "expected ']' to close table header"
                raise self._sc.error(msg)
            self._sc.advance(1)

        trailing, comment, newline = self._sc.scan_eol()

        path = key.path
        self._validate_header(path, kind, at=self._sc.pos)
        if kind == "table":
            self._explicit_table_paths.add(path)
            self._track(path)
        else:
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

        return TableHeaderNode(
            leading,
            kind,
            inner_pre,
            key,
            inner_post,
            trailing,
            comment,
            newline,
        )

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------

    def _parse_key(self) -> Key:
        sc = self._sc
        parts: list[KeyPart] = [sc.scan_key_part()]
        separators: list[str] = []
        while True:
            sep = sc.scan_key_separator()
            if sep is None:
                break
            separators.append(sep)
            parts.append(sc.scan_key_part())
        return Key(parts, separators)

    # ------------------------------------------------------------------
    # Key/value lines
    # ------------------------------------------------------------------

    def _parse_key_value(self, leading: Trivia) -> KeyValueNode:
        key = self._parse_key()
        pre_eq = self._sc.scan_inline_ws()
        src = self._sc.src
        pos = self._sc.pos
        if pos >= self._sc.end or src[pos] != "=":
            ch = src[pos] if pos < self._sc.end else ""
            msg = f"expected '=' after key, got {ch!r}"
            raise self._sc.error(msg)
        self._sc.pos = pos + 1
        post_eq = self._sc.scan_inline_ws()
        value = self._parse_value()
        trailing, comment, newline = self._sc.scan_eol()
        return KeyValueNode(
            leading,
            key,
            pre_eq,
            post_eq,
            value,
            trailing,
            comment,
            newline,
        )

    def _validate_header(
        self,
        path: tuple[str, ...],
        kind: HeaderKind,
        *,
        at: int,
    ) -> None:
        # Prefix overlaps with a bound value would mean overwriting a scalar
        # (or an inline-table value) with a table — always invalid.
        for i in range(1, len(path)):
            prefix = path[:i]
            if prefix in self._value_paths:
                joined = ".".join(prefix)
                msg = f"cannot use {joined!r} as a table: already defined as a value"
                raise self._sc.error(msg, at=at)
        if path in self._value_paths:
            joined = ".".join(path)
            msg = f"cannot define {joined!r} as a table: already defined as a value"
            raise self._sc.error(msg, at=at)
        if path in self._dotted_paths:
            joined = ".".join(path)
            msg = (
                f"cannot define {joined!r} as a table: already created via dotted keys"
            )
            raise self._sc.error(msg, at=at)
        if kind == "table":
            if path in self._explicit_table_paths:
                msg = f"redefinition of table {'.'.join(path)!r}"
                raise self._sc.error(msg, at=at)
            if path in self._aot_paths:
                joined = ".".join(path)
                msg = f"cannot redefine array-of-tables {joined!r} as a normal table"
                raise self._sc.error(msg, at=at)
        else:  # array-of-tables
            if path in self._explicit_table_paths:
                msg = f"cannot redefine table {'.'.join(path)!r} as an array-of-tables"
                raise self._sc.error(msg, at=at)
            if path in self._implicit_table_paths and path not in self._aot_paths:
                msg = (
                    f"cannot define {'.'.join(path)!r} as an array-of-tables: "
                    "already used as an implicit table"
                )
                raise self._sc.error(msg, at=at)

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

    def _record_keyvalue(self, kv: KeyValueNode) -> None:
        section = self._current_section
        path = kv.key.path
        full = section + path if section else path
        # Final-path conflicts.
        value_paths = self._value_paths
        if full in value_paths:
            at = self._sc.pos
            msg = f"duplicate key {'.'.join(full)!r}"
            raise self._sc.error(msg, at=at)
        if (
            full in self._explicit_table_paths
            or full in self._aot_paths
            or full in self._implicit_table_paths
            or full in self._dotted_paths
        ):
            at = self._sc.pos
            msg = f"key {'.'.join(full)!r} already defined as a table"
            raise self._sc.error(msg, at=at)
        # Intermediate-prefix conflicts (paths between section and full).
        slen = len(section)
        flen = len(full)
        if flen > slen + 1:
            at = self._sc.pos
            for i in range(slen + 1, flen):
                sub = full[:i]
                if sub in value_paths:
                    msg = f"key {'.'.join(sub)!r} already defined as a value"
                    raise self._sc.error(msg, at=at)
                if sub in self._explicit_table_paths:
                    joined = ".".join(sub)
                    msg = (
                        f"cannot extend explicitly-defined table {joined!r} "
                        "via dotted keys"
                    )
                    raise self._sc.error(msg, at=at)
                if sub in self._aot_paths:
                    msg = (
                        f"cannot extend array-of-tables {'.'.join(sub)!r} "
                        "via dotted keys"
                    )
                    raise self._sc.error(msg, at=at)
                self._dotted_paths.add(sub)
                self._track(sub)
        value_paths.add(full)
        self._track(full)
        # Inline-table values: register their nested key paths so
        # cross-section headers/keys see the conflicts. (Local
        # duplicate / dotted-prefix conflicts are caught at parse time
        # in `_parse_inline_table`.)
        value = kv.value
        if isinstance(value, InlineTableNode):
            self._validate_inline_table(value, abs_prefix=full)
        elif isinstance(value, ArrayNode):
            for item in value.items:
                if isinstance(item.value, InlineTableNode):
                    self._validate_inline_table(item.value, abs_prefix=None)

    def _check_inline_key_conflict(
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
        ``seen_values`` after successful validation.
        """
        if path in seen_values:
            msg = f"duplicate key {'.'.join(path)!r} in inline table"
            raise self._sc.error(msg, at=at)
        if path in seen_prefixes:
            msg = (
                f"key {'.'.join(path)!r} in inline table conflicts with "
                "an existing dotted-key prefix"
            )
            raise self._sc.error(msg, at=at)
        for i in range(1, len(path)):
            sub = path[:i]
            if sub in seen_values:
                msg = f"inline-table key {'.'.join(sub)!r} already defined as a value"
                raise self._sc.error(msg, at=at)
            seen_prefixes.add(sub)

    def _validate_inline_table(
        self,
        table: InlineTableNode,
        *,
        abs_prefix: tuple[str, ...] | None,
    ) -> None:
        """Register an inline table's keys for cross-section conflict checks.

        When ``abs_prefix`` is given, exposes the inline table's keys to
        document-wide tracking. Walks nested inline tables either way.
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
                self._validate_inline_table(entry.value, abs_prefix=sub_abs)
            elif isinstance(entry.value, ArrayNode):
                for item in entry.value.items:
                    if isinstance(item.value, InlineTableNode):
                        self._validate_inline_table(item.value, abs_prefix=None)

    # ------------------------------------------------------------------
    # Values
    # ------------------------------------------------------------------

    def _parse_value(self) -> ValueNode:
        ch = self._sc.peek()
        if ch == '"':
            return self._sc.scan_string('"')
        if ch == "'":
            return self._sc.scan_string("'")
        if ch == "[":
            return self._parse_nested_value(self._parse_array)
        if ch == "{":
            return self._parse_nested_value(self._parse_inline_table)
        # Bools, special floats, integers, floats and date/time literals
        # all funnel through one bare-token scan-then-classify path.
        return self._sc.scan_value_atom()

    def _parse_nested_value(self, parser: Callable[[], ValueNode]) -> ValueNode:
        if self._value_depth >= self._MAX_VALUE_DEPTH:
            msg = f"value nesting exceeds maximum depth ({self._MAX_VALUE_DEPTH})"
            raise self._sc.error(msg)
        self._value_depth += 1
        try:
            return parser()
        finally:
            self._value_depth -= 1

    # --- arrays -------------------------------------------------------

    def _parse_array(self) -> ArrayNode:
        assert self._sc.peek() == "["
        self._sc.advance(1)
        node = ArrayNode()
        leading = self._sc.scan_array_trivia()
        if self._sc.peek() == "]":
            node.final_trivia = leading
            self._sc.advance(1)
            return node
        while True:
            value = self._parse_value()
            trailing = self._sc.scan_array_trivia()
            has_comma = False
            post_comma = Trivia()
            if self._sc.peek() == ",":
                self._sc.advance(1)
                has_comma = True
                post_comma = self._sc.scan_array_trivia()
            elif self._sc.peek() != "]":
                msg = f"expected ',' or ']' in array, got {self._sc.peek()!r}"
                raise self._sc.error(msg)
            node.items.append(
                ArrayItem(leading, value, trailing, has_comma, post_comma),
            )
            if not has_comma:
                # We're at ']'.
                self._sc.advance(1)
                return node
            leading = Trivia()  # next item's leading is empty; trivia
            # already attached as post_comma of previous item.
            if self._sc.peek() == "]":
                # Trailing comma followed by closer.
                self._sc.advance(1)
                return node
            # Next iteration starts a new value.

    # --- inline tables ------------------------------------------------

    def _parse_inline_table(self) -> InlineTableNode:
        assert self._sc.peek() == "{"
        self._sc.advance(1)
        node = InlineTableNode()
        # TOML 1.1: newlines and trailing comma are allowed inside an
        # inline table, with the same trivia rules as arrays.
        leading = self._sc.scan_array_trivia()
        if self._sc.peek() == "}":
            node.final_trivia = leading
            self._sc.advance(1)
            return node
        # Inline-table-local key tracking. Values and prefixes are
        # tracked separately so that ``{ x = 1, x.y = 2 }`` and
        # ``{ a.b = 1, a = 2 }`` both fail at the offending key.
        seen_values: set[tuple[str, ...]] = set()
        seen_prefixes: set[tuple[str, ...]] = set()
        while True:
            key_at = self._sc.pos
            key = self._parse_key()
            self._check_inline_key_conflict(
                key.path,
                seen_values,
                seen_prefixes,
                at=key_at,
            )
            seen_values.add(key.path)
            pre_eq = self._sc.scan_inline_ws()
            if self._sc.peek() != "=":
                msg = f"expected '=' in inline table, got {self._sc.peek()!r}"
                raise self._sc.error(msg)
            self._sc.advance(1)
            post_eq = self._sc.scan_inline_ws()
            value = self._parse_value()
            trailing = self._sc.scan_array_trivia()
            has_comma = False
            post_comma = Trivia()
            if self._sc.peek() == ",":
                self._sc.advance(1)
                has_comma = True
                post_comma = self._sc.scan_array_trivia()
            elif self._sc.peek() != "}":
                msg = f"expected ',' or '}}' in inline table, got {self._sc.peek()!r}"
                raise self._sc.error(msg)
            node.entries.append(
                InlineTableEntry(
                    leading,
                    key,
                    pre_eq,
                    post_eq,
                    value,
                    trailing,
                    has_comma,
                    post_comma,
                ),
            )
            if not has_comma:
                # We're at '}'.
                self._sc.advance(1)
                return node
            leading = Trivia()
            if self._sc.peek() == "}":
                # Trailing comma followed by the closer (TOML 1.1).
                self._sc.advance(1)
                return node


# Re-export convenience.
def parse(src: str) -> DocumentNode:
    return _Parser(src).parse()


__all__ = ["parse"]
