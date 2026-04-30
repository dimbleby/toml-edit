"""TOML 1.0 / 1.1 parser producing a `tomlrt._nodes` CST.

Hand-written recursive-descent over a `_Scanner`. The parser is purely
syntactic — it builds the CST and decides nothing about table-level
semantics (duplicate keys, dotted-prefix conflicts, AoT scope, etc.).
Those rules are owned by `tomlrt._validator._Validator`, which the
parser instantiates and calls into at three points: when a header has
just been parsed, when a key/value line has just been built, and when
each key inside an inline table is about to be added.
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
from tomlrt._validator import _Validator

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomlrt._nodes import (
        HeaderKind,
        KeyPart,
        ValueNode,
    )


class _Parser:
    __slots__ = ("_sc", "_validator", "_value_depth")

    # Hard cap on nested array / inline-table depth. Well above any
    # realistic data (tomllib, for reference, accepts whatever Python's
    # recursion limit allows; we pick a friendlier ceiling so the user
    # gets a TOMLParseError instead of a RecursionError).
    _MAX_VALUE_DEPTH = 100

    def __init__(self, src: str) -> None:
        self._sc = _Scanner(src)
        self._validator = _Validator(self._sc.error)
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
            else:
                kv = self._parse_key_value(leading)
                self._validator.record_keyvalue(kv, at=sc.pos)
                current.entries.append(kv)

        return doc

    def detected_newline(self) -> str:
        """Document-wide newline kind seen during scanning."""
        return self._sc.detected_newline()

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

        header = TableHeaderNode(
            leading,
            kind,
            inner_pre,
            key,
            inner_post,
            trailing,
            comment,
            newline,
        )
        self._validator.enter_header(header, at=self._sc.pos)
        return header

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
            self._validator.check_inline_key_conflict(
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
