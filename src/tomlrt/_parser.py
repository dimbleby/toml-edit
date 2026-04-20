"""TOML 1.0.0 parser producing a :mod:`tomlrt._nodes` CST.

Hand-written recursive-descent over a ``str`` plus an integer cursor.
Bulk character runs (whitespace, bare keys, string bodies, comment
bodies) are scanned with module-level compiled regexes so each run is
a single C call rather than a Python-level character loop.

The parser is responsible *only* for producing the physical CST. Logical
table semantics (duplicate-key detection across discontiguous headers,
dotted-key conflicts, etc.) are enforced here; the read-side wrappers
in :mod:`tomlrt._document` rely on this validation having happened.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, Final

from tomlrt._errors import TOMLParseError
from tomlrt._nodes import (
    ArrayItem,
    ArrayNode,
    BoolNode,
    CommentNode,
    DateTimeNode,
    DocumentNode,
    FloatNode,
    InlineTableEntry,
    InlineTableNode,
    IntegerNode,
    Key,
    KeyPart,
    KeyValueNode,
    NewlineNode,
    SectionNode,
    StringNode,
    TableHeaderNode,
    Trivia,
    WhitespaceNode,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomlrt._nodes import HeaderKind, IntStyle, ValueNode

_BARE_KEY_CHARS: Final[frozenset[str]] = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-",
)
_HEX_DIGITS: Final[frozenset[str]] = frozenset("0123456789abcdefABCDEF")
_DEC_DIGITS: Final[frozenset[str]] = frozenset("0123456789")
_OCT_DIGITS: Final[frozenset[str]] = frozenset("01234567")
_BIN_DIGITS: Final[frozenset[str]] = frozenset("01")

# Compiled regexes used for bulk-scanning hot paths. Each pattern is
# anchored at the cursor (we use ``match`` with an explicit ``pos``);
# the resulting ``end()`` tells us how many characters to consume in a
# single C-level call instead of a Python-level loop.
_RE_INLINE_WS = re.compile(r"[ \t]+")
_RE_BARE_KEY = re.compile(r"[A-Za-z0-9_\-]+")
# Body of a basic string: any run of chars that are NOT a quote,
# backslash, newline, or control char (control = U+0000-U+001F or
# U+007F, except tab which we *do* allow).
_RE_BASIC_STR_BODY = re.compile(r'[^"\\\n\r\x00-\x08\x0b-\x1f\x7f]+')
# Body of a literal string: anything except quote, newline, control
# char (tab and newline-not-allowed handled by the caller).
_RE_LITERAL_STR_BODY = re.compile(r"[^'\n\r\x00-\x08\x0b-\x1f\x7f]+")
# Body of a multi-line basic string fragment: stops at " or \\ or \r or
# \n or a control char. \n and \r\n are valid in ML strings, so the
# caller handles them; we stop at \r so the caller can verify it's
# followed by \n and emit a normalized pair.
_RE_ML_BASIC_BODY = re.compile(r'[^"\\\r\n\x00-\x08\x0b-\x1f\x7f]+')
_RE_ML_LITERAL_BODY = re.compile(r"[^'\r\n\x00-\x08\x0b-\x1f\x7f]+")
# Comment body: anything except newline + control chars (tab is OK).
_RE_COMMENT_BODY = re.compile(r"[^\r\n\x00-\x08\x0b-\x1f\x7f]*")


class _Parser:
    __slots__ = (
        "_aot_paths",
        "_aot_subpaths",
        "_current_section",
        "_dotted_paths",
        "_end",
        "_explicit_table_paths",
        "_implicit_table_paths",
        "_pos",
        "_src",
        "_value_depth",
        "_value_paths",
    )

    # Hard cap on nested array / inline-table depth. Well above any
    # realistic data (tomllib, for reference, accepts whatever Python's
    # recursion limit allows; we pick a friendlier ceiling so the user
    # gets a TOMLParseError instead of a RecursionError).
    _MAX_VALUE_DEPTH = 100

    def __init__(self, src: str) -> None:
        self._src = src
        self._end = len(src)
        self._pos = 0
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
    # Cursor helpers
    # ------------------------------------------------------------------

    def _peek(self, offset: int = 0) -> str:
        i = self._pos + offset
        if i >= self._end:
            return ""
        return self._src[i]

    def _starts_with(self, s: str) -> bool:
        return self._src.startswith(s, self._pos)

    def _eof(self) -> bool:
        return self._pos >= self._end

    def _advance(self, n: int = 1) -> str:
        s = self._src[self._pos : self._pos + n]
        self._pos += n
        return s

    def _line_col(self, pos: int) -> tuple[int, int]:
        line = 1
        last_nl = -1
        for i in range(pos):
            if self._src[i] == "\n":
                line += 1
                last_nl = i
        col = pos - last_nl
        return line, col

    def _error(self, message: str, *, at: int | None = None) -> TOMLParseError:
        offset = self._pos if at is None else at
        line, col = self._line_col(offset)
        return TOMLParseError(message, line=line, col=col, offset=offset)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def parse(self) -> DocumentNode:
        doc = DocumentNode()
        current = SectionNode(header=None)
        doc.sections.append(current)

        while not self._eof():
            leading = self._collect_trivia_block()
            if self._eof():
                # leading is purely trailing trivia for the document.
                doc.trailing_trivia.pieces.extend(leading.pieces)
                break

            ch = self._peek()
            if ch == "[":
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
    # Trivia
    # ------------------------------------------------------------------

    def _collect_trivia_block(self) -> Trivia:
        """Collect whitespace, blank lines and comment lines.

        Stops *before* the next non-trivia character on a line (i.e. the
        leading whitespace of that line **is** consumed as part of the
        block; the structural token starts at ``self._pos``).
        """
        trivia = Trivia()
        pieces = trivia.pieces
        src = self._src
        end = self._end
        pos = self._pos
        while pos < end:
            ch = src[pos]
            if ch == " " or ch == "\t":
                m = _RE_INLINE_WS.match(src, pos)
                # m is non-None because src[pos] matched the class.
                assert m is not None
                pieces.append(WhitespaceNode(m.group(0)))
                pos = m.end()
            elif ch == "#":
                self._pos = pos
                pieces.append(self._consume_comment())
                pos = self._pos
            elif ch == "\n":
                pos += 1
                pieces.append(NewlineNode("\n"))
            elif ch == "\r":
                if pos + 1 >= end or src[pos + 1] != "\n":
                    self._pos = pos
                    msg = "stray carriage return"
                    raise self._error(msg)
                pos += 2
                pieces.append(NewlineNode("\r\n"))
            else:
                break
        self._pos = pos
        return trivia

    def _consume_comment(self) -> CommentNode:
        # Comment starts at '#' and runs until newline or EOF. Control
        # chars (other than tab) are forbidden.
        src = self._src
        start = self._pos
        m = _RE_COMMENT_BODY.match(src, start + 1)
        # The pattern is unbounded above (*), so match always succeeds.
        assert m is not None
        end_pos = m.end()
        # If we stopped before EOF and not at newline, the next char is
        # an illegal control character.
        if end_pos < self._end:
            ch = src[end_pos]
            if ch != "\n" and ch != "\r":
                self._pos = end_pos
                cp = ord(ch)
                msg = f"invalid control character U+{cp:04X} in comment"
                raise self._error(msg)
        self._pos = end_pos
        return CommentNode(src[start:end_pos])

    def _consume_inline_ws(self) -> WhitespaceNode | None:
        """Whitespace (no newlines, no comments)."""
        m = _RE_INLINE_WS.match(self._src, self._pos)
        if m is None:
            return None
        self._pos = m.end()
        return WhitespaceNode(m.group(0))

    def _consume_array_trivia(self) -> Trivia:
        """Whitespace, newlines and comments allowed inside arrays."""
        trivia = Trivia()
        pieces = trivia.pieces
        src = self._src
        end = self._end
        pos = self._pos
        while pos < end:
            ch = src[pos]
            if ch == " " or ch == "\t":
                m = _RE_INLINE_WS.match(src, pos)
                assert m is not None
                pieces.append(WhitespaceNode(m.group(0)))
                pos = m.end()
            elif ch == "\n":
                pos += 1
                pieces.append(NewlineNode("\n"))
            elif ch == "\r" and pos + 1 < end and src[pos + 1] == "\n":
                pos += 2
                pieces.append(NewlineNode("\r\n"))
            elif ch == "#":
                self._pos = pos
                pieces.append(self._consume_comment())
                pos = self._pos
            else:
                break
        self._pos = pos
        return trivia

    def _consume_eol(
        self,
    ) -> tuple[WhitespaceNode | None, CommentNode | None, NewlineNode | None]:
        trailing = self._consume_inline_ws()
        comment: CommentNode | None = None
        if self._peek() == "#":
            comment = self._consume_comment()
        newline: NewlineNode | None = None
        if self._peek() == "\n":
            self._pos += 1
            newline = NewlineNode("\n")
        elif self._peek() == "\r" and self._peek(1) == "\n":
            self._pos += 2
            newline = NewlineNode("\r\n")
        elif not self._eof():
            msg = f"expected newline or end of file, got {self._peek()!r}"
            raise self._error(msg)
        return trailing, comment, newline

    # ------------------------------------------------------------------
    # Headers
    # ------------------------------------------------------------------

    def _parse_header(self, leading: Trivia) -> TableHeaderNode:
        kind: HeaderKind
        if self._starts_with("[["):
            self._advance(2)
            kind = "array"
        else:
            assert self._peek() == "["
            self._advance(1)
            kind = "table"

        inner_pre = self._consume_inline_ws()
        key = self._parse_key()
        inner_post = self._consume_inline_ws()

        if kind == "array":
            if not self._starts_with("]]"):
                msg = "expected ']]' to close array-of-tables header"
                raise self._error(msg)
            self._advance(2)
        else:
            if self._peek() != "]":
                msg = "expected ']' to close table header"
                raise self._error(msg)
            self._advance(1)

        trailing, comment, newline = self._consume_eol()

        path = key.path
        self._validate_header(path, kind, at=self._pos)
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
            leading=leading,
            kind=kind,
            inner_pre=inner_pre,
            key=key,
            inner_post=inner_post,
            trailing=trailing,
            trailing_comment=comment,
            newline=newline,
        )

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------

    def _parse_key(self) -> Key:
        parts: list[KeyPart] = [self._parse_key_part()]
        separators: list[str] = []
        src = self._src
        while True:
            save = self._pos
            m = _RE_INLINE_WS.match(src, save)
            ws_end = m.end() if m is not None else save
            if ws_end >= self._end or src[ws_end] != ".":
                self._pos = save
                break
            after_dot = ws_end + 1
            m2 = _RE_INLINE_WS.match(src, after_dot)
            sep_end = m2.end() if m2 is not None else after_dot
            self._pos = sep_end
            separators.append(src[save:sep_end])
            parts.append(self._parse_key_part())
        return Key(parts=parts, separators=separators)

    def _parse_key_part(self) -> KeyPart:
        ch = self._peek()
        if ch == '"':
            return self._parse_basic_key()
        if ch == "'":
            return self._parse_literal_key()
        m = _RE_BARE_KEY.match(self._src, self._pos)
        if m is not None:
            raw = m.group(0)
            self._pos = m.end()
            return KeyPart(raw=raw, value=raw, kind="bare")
        msg = f"expected key, got {ch!r}"
        raise self._error(msg)

    def _parse_basic_key(self) -> KeyPart:
        start = self._pos
        s = self._parse_basic_string(allow_multiline=False)
        raw = self._src[start : self._pos]
        return KeyPart(raw=raw, value=s.value, kind="basic")

    def _parse_literal_key(self) -> KeyPart:
        start = self._pos
        s = self._parse_literal_string(allow_multiline=False)
        raw = self._src[start : self._pos]
        return KeyPart(raw=raw, value=s.value, kind="literal")

    # ------------------------------------------------------------------
    # Key/value lines
    # ------------------------------------------------------------------

    def _parse_key_value(self, leading: Trivia) -> KeyValueNode:
        key = self._parse_key()
        pre_eq = self._consume_inline_ws()
        if self._peek() != "=":
            msg = f"expected '=' after key, got {self._peek()!r}"
            raise self._error(msg)
        self._advance(1)
        post_eq = self._consume_inline_ws()
        value = self._parse_value()
        trailing, comment, newline = self._consume_eol()
        return KeyValueNode(
            leading=leading,
            key=key,
            pre_eq=pre_eq,
            post_eq=post_eq,
            value=value,
            trailing=trailing,
            trailing_comment=comment,
            newline=newline,
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

    def _track(self, p: tuple[str, ...]) -> None:
        """Index ``p`` under its longest active-AoT-path ancestor, if any.

        Only paths registered here can be reset by ``_reset_scope_under``.
        Most paths in a typical document have no AoT ancestor and the
        prefix walk costs only a handful of hash lookups.
        """
        aot_paths = self._aot_paths
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
        full = section + kv.key.path
        at = self._pos
        # Final-path conflicts.
        if full in self._value_paths:
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
        for i in range(len(section) + 1, len(full)):
            sub = full[:i]
            if sub in self._value_paths:
                msg = f"key {'.'.join(sub)!r} already defined as a value"
                raise self._error(msg, at=at)
            if sub in self._explicit_table_paths:
                joined = ".".join(sub)
                msg = (
                    f"cannot extend explicitly-defined table {joined!r} via dotted keys"
                )
                raise self._error(msg, at=at)
            if sub in self._aot_paths:
                msg = f"cannot extend array-of-tables {'.'.join(sub)!r} via dotted keys"
                raise self._error(msg, at=at)
            self._dotted_paths.add(sub)
            self._track(sub)
        self._value_paths.add(full)
        self._track(full)
        # Inline-table values: validate within-table dups and register their
        # nested key paths so cross-section headers/keys see the conflicts.
        if isinstance(kv.value, InlineTableNode):
            self._validate_inline_table(kv.value, abs_prefix=full, at=at)
        elif isinstance(kv.value, ArrayNode):
            for item in kv.value.items:
                if isinstance(item.value, InlineTableNode):
                    self._validate_inline_table(item.value, abs_prefix=None, at=at)

    def _validate_inline_table(
        self,
        table: InlineTableNode,
        *,
        abs_prefix: tuple[str, ...] | None,
        at: int,
    ) -> None:
        """Reject duplicate / conflicting keys inside one inline table.

        If ``abs_prefix`` is given, also register the inline table's keys
        under that prefix in the document-wide tracking sets so later
        section headers or dotted keys can detect conflicts with paths
        owned by this inline table.
        """
        local_values: set[tuple[str, ...]] = set()
        local_prefixes: set[tuple[str, ...]] = set()
        for entry in table.entries:
            path = entry.key.path
            if path in local_values:
                msg = f"duplicate key {'.'.join(path)!r} in inline table"
                raise self._error(msg, at=at)
            if path in local_prefixes:
                msg = (
                    f"key {'.'.join(path)!r} in inline table conflicts with "
                    "an existing dotted-key prefix"
                )
                raise self._error(msg, at=at)
            for i in range(1, len(path)):
                sub = path[:i]
                if sub in local_values:
                    msg = (
                        f"inline-table key {'.'.join(sub)!r} already defined as a value"
                    )
                    raise self._error(msg, at=at)
                local_prefixes.add(sub)
            local_values.add(path)
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
                self._validate_inline_table(entry.value, abs_prefix=sub_abs, at=at)
            elif isinstance(entry.value, ArrayNode):
                for item in entry.value.items:
                    if isinstance(item.value, InlineTableNode):
                        self._validate_inline_table(
                            item.value,
                            abs_prefix=None,
                            at=at,
                        )

    # ------------------------------------------------------------------
    # Values
    # ------------------------------------------------------------------

    def _parse_value(self) -> ValueNode:
        ch = self._peek()
        if ch == '"':
            return self._parse_string('"')
        if ch == "'":
            return self._parse_string("'")
        if ch == "[":
            return self._parse_nested_value(self._parse_array)
        if ch == "{":
            return self._parse_nested_value(self._parse_inline_table)
        if ch in ("t", "f"):
            return self._parse_bool()
        # Everything else: a number or date/time. They share a leading
        # ambiguous prefix; sniff a window then dispatch.
        return self._parse_number_or_datetime()

    def _parse_nested_value(self, parser: Callable[[], ValueNode]) -> ValueNode:
        if self._value_depth >= self._MAX_VALUE_DEPTH:
            msg = f"value nesting exceeds maximum depth ({self._MAX_VALUE_DEPTH})"
            raise self._error(msg)
        self._value_depth += 1
        try:
            return parser()
        finally:
            self._value_depth -= 1

    # --- strings ------------------------------------------------------

    def _parse_string(self, quote: str) -> StringNode:
        start = self._pos
        if quote == '"':
            multiline = self._starts_with('"""')
            node = self._parse_basic_string(allow_multiline=multiline)
        else:
            multiline = self._starts_with("'''")
            node = self._parse_literal_string(allow_multiline=multiline)
        node.raw = self._src[start : self._pos]
        return node

    def _parse_basic_string(self, *, allow_multiline: bool) -> StringNode:
        if allow_multiline and self._starts_with('"""'):
            return self._parse_ml_basic_string()
        if self._peek() != '"':
            msg = "expected '\"' to start basic string"
            raise self._error(msg)
        self._pos += 1
        src = self._src
        end = self._end
        out: list[str] = []
        while True:
            m = _RE_BASIC_STR_BODY.match(src, self._pos)
            if m is not None:
                out.append(m.group(0))
                self._pos = m.end()
            if self._pos >= end:
                msg = "unterminated basic string"
                raise self._error(msg)
            ch = src[self._pos]
            if ch == '"':
                self._pos += 1
                return StringNode(raw="", value="".join(out), style="basic")
            if ch == "\\":
                out.append(self._parse_escape())
                continue
            if ch == "\n" or ch == "\r":
                msg = "newline in basic string"
                raise self._error(msg)
            cp = ord(ch)
            if cp == 0x7F:
                msg = "invalid control character U+007F in string"
                raise self._error(msg)
            msg = f"invalid control character U+{cp:04X} in string"
            raise self._error(msg)

    def _parse_ml_basic_string(self) -> StringNode:
        assert self._starts_with('"""')
        self._advance(3)
        # A newline immediately after the opening delimiter is trimmed.
        if self._peek() == "\n":
            self._advance(1)
        elif self._peek() == "\r" and self._peek(1) == "\n":
            self._advance(2)
        out: list[str] = []
        while True:
            if self._eof():
                msg = "unterminated multi-line basic string"
                raise self._error(msg)
            m = _RE_ML_BASIC_BODY.match(self._src, self._pos)
            if m is not None:
                out.append(m.group(0))
                self._pos = m.end()
                if self._eof():
                    continue
            if self._starts_with('"""'):
                # Up to two extra trailing quotes are allowed inside.
                self._advance(3)
                extras = 0
                while extras < 2 and self._peek() == '"':
                    out.append('"')
                    self._advance(1)
                    extras += 1
                return StringNode(raw="", value="".join(out), style="ml-basic")
            ch = self._peek()
            if ch == '"':
                # Single or double quote (not the closing triple) — emit and
                # continue. The body regex stops at any quote.
                out.append('"')
                self._pos += 1
                continue
            if ch == "\\":
                # Line-ending backslash: trim trailing ws+newline+leading-ws.
                if self._peek(1) in ("\n", " ", "\t", "\r"):
                    save = self._pos
                    self._pos += 1
                    # Skip trailing inline ws on this line.
                    while self._peek() in (" ", "\t"):
                        self._pos += 1
                    if self._peek() == "\n" or (
                        self._peek() == "\r" and self._peek(1) == "\n"
                    ):
                        # Eat one or more whitespace lines.
                        while True:
                            if self._peek() == "\n":
                                self._pos += 1
                            elif self._peek() == "\r" and self._peek(1) == "\n":
                                self._pos += 2
                            else:
                                break
                            while self._peek() in (" ", "\t"):
                                self._pos += 1
                        continue
                    # Not actually a line-ending backslash; rewind and
                    # treat as a normal escape.
                    self._pos = save
                out.append(self._parse_escape())
                continue
            if ch == "\r":
                if self._peek(1) != "\n":
                    msg = "stray carriage return in string"
                    raise self._error(msg)
                out.append("\r\n")
                self._pos += 2
                continue
            if ch == "\n":
                out.append("\n")
                self._pos += 1
                continue
            cp = ord(ch)
            if cp <= 0x1F and cp != 0x09:
                msg = f"invalid control character U+{cp:04X} in string"
                raise self._error(msg)
            if cp == 0x7F:
                msg = "invalid control character U+007F in string"
                raise self._error(msg)
            out.append(ch)
            self._pos += 1

    def _parse_literal_string(self, *, allow_multiline: bool) -> StringNode:
        if allow_multiline and self._starts_with("'''"):
            return self._parse_ml_literal_string()
        if self._peek() != "'":
            msg = 'expected "\'" to start literal string'
            raise self._error(msg)
        self._pos += 1
        src = self._src
        end = self._end
        start = self._pos
        m = _RE_LITERAL_STR_BODY.match(src, start)
        if m is not None:
            self._pos = m.end()
        if self._pos >= end:
            msg = "unterminated literal string"
            raise self._error(msg)
        ch = src[self._pos]
        if ch == "'":
            value = src[start : self._pos]
            self._pos += 1
            return StringNode(raw="", value=value, style="literal")
        if ch == "\n" or ch == "\r":
            msg = "newline in literal string"
            raise self._error(msg)
        cp = ord(ch)
        if cp == 0x7F:
            msg = "invalid control character U+007F in string"
            raise self._error(msg)
        msg = f"invalid control character U+{cp:04X} in string"
        raise self._error(msg)

    def _parse_ml_literal_string(self) -> StringNode:
        assert self._starts_with("'''")
        self._advance(3)
        if self._peek() == "\n":
            self._advance(1)
        elif self._peek() == "\r" and self._peek(1) == "\n":
            self._advance(2)
        out: list[str] = []
        while True:
            if self._eof():
                msg = "unterminated multi-line literal string"
                raise self._error(msg)
            m = _RE_ML_LITERAL_BODY.match(self._src, self._pos)
            if m is not None:
                out.append(m.group(0))
                self._pos = m.end()
                if self._eof():
                    continue
            if self._starts_with("'''"):
                self._advance(3)
                extras = 0
                while extras < 2 and self._peek() == "'":
                    out.append("'")
                    self._advance(1)
                    extras += 1
                return StringNode(raw="", value="".join(out), style="ml-literal")
            ch = self._peek()
            if ch == "'":
                # Single quote, not the closing triple — emit and continue.
                out.append("'")
                self._pos += 1
                continue
            if ch == "\n":
                out.append("\n")
                self._pos += 1
                continue
            if ch == "\r":
                if self._peek(1) != "\n":
                    msg = "stray carriage return in string"
                    raise self._error(msg)
                out.append("\r\n")
                self._pos += 2
                continue
            cp = ord(ch)
            if cp == 0x7F:
                msg = "invalid control character U+007F in string"
                raise self._error(msg)
            msg = f"invalid control character U+{cp:04X} in string"
            raise self._error(msg)

    def _parse_escape(self) -> str:
        assert self._peek() == "\\"
        self._advance(1)
        ch = self._peek()
        self._advance(1)
        simple: dict[str, str] = {
            "b": "\b",
            "t": "\t",
            "n": "\n",
            "f": "\f",
            "r": "\r",
            "e": "\x1b",  # TOML 1.1: ESC
            '"': '"',
            "\\": "\\",
        }
        if ch in simple:
            return simple[ch]
        if ch == "x":  # TOML 1.1: 2-digit hex escape (U+0000..U+00FF)
            return self._parse_unicode_escape(2)
        if ch == "u":
            return self._parse_unicode_escape(4)
        if ch == "U":
            return self._parse_unicode_escape(8)
        msg = f"invalid escape sequence: \\{ch}"
        raise self._error(msg, at=self._pos - 2)

    def _parse_unicode_escape(self, n: int) -> str:
        if self._pos + n > len(self._src):
            msg = f"truncated unicode escape; expected {n} hex digits"
            raise self._error(msg)
        hex_str = self._src[self._pos : self._pos + n]
        for c in hex_str:
            if c not in _HEX_DIGITS:
                msg = f"invalid hex digit {c!r} in unicode escape"
                raise self._error(msg)
        self._pos += n
        cp = int(hex_str, 16)
        if cp > 0x10FFFF or 0xD800 <= cp <= 0xDFFF:
            msg = f"invalid unicode scalar U+{cp:04X}"
            raise self._error(msg)
        return chr(cp)

    # --- bool ---------------------------------------------------------

    def _parse_bool(self) -> BoolNode:
        if self._starts_with("true"):
            self._advance(4)
            return BoolNode(raw="true", value=True)
        if self._starts_with("false"):
            self._advance(5)
            return BoolNode(raw="false", value=False)
        msg = "expected boolean"
        raise self._error(msg)

    # --- numbers and date-times --------------------------------------

    def _parse_number_or_datetime(self) -> ValueNode:
        # Find the end of the token (something that ends a value).
        start = self._pos
        end = self._scan_value_end(start)
        token = self._src[start:end]
        if not token:
            msg = f"expected value, got {self._peek()!r}"
            raise self._error(msg)

        # Special floats.
        if token in ("inf", "+inf"):
            self._pos = end
            return FloatNode(raw=token, value=float("inf"))
        if token == "-inf":  # noqa: S105 - "token" here is a lexical token, not a credential
            self._pos = end
            return FloatNode(raw=token, value=float("-inf"))
        if token in ("nan", "+nan", "-nan"):
            self._pos = end
            return FloatNode(raw=token, value=float("nan"))

        # Date/time: ISO-8601 forms always contain '-' (date) or ':' (time)
        # in a position that disambiguates from an integer with underscores.
        if self._looks_like_datetime(token):
            self._pos = end
            return self._parse_datetime_token(token, at=start)

        # Number: integer (with possible 0x/0o/0b prefix) or float.
        if self._looks_like_float(token):
            self._pos = end
            return self._parse_float_token(token, at=start)

        self._pos = end
        return self._parse_integer_token(token, at=start)

    def _scan_value_end(self, start: int) -> int:
        """Return the offset of the first char that ends the current value.

        Stops at whitespace, newline, ``,``, ``]``, ``}``, ``#``, EOF.
        """
        i = start
        n = len(self._src)
        while i < n:
            c = self._src[i]
            if c in (" ", "\t", "\n", "\r", ",", "]", "}", "#"):
                break
            i += 1
        return i

    def _looks_like_datetime(self, token: str) -> bool:
        # Date: "YYYY-MM-DD"; Local time: "HH:MM:SS"; Datetime contains both.
        if len(token) >= 5 and token[4] == "-" and token[:4].isdigit():
            return True
        return bool(len(token) >= 3 and token[2] == ":" and token[:2].isdigit())

    def _looks_like_float(self, token: str) -> bool:
        body = token.lstrip("+-")
        if body.startswith(("0x", "0o", "0b")):
            return False
        return any(c in body for c in (".", "e", "E"))

    def _parse_integer_token(self, token: str, *, at: int) -> IntegerNode:
        style: IntStyle
        body = token
        if body.startswith(("0x", "0o", "0b")):
            prefix = body[:2]
            digits = body[2:]
            if not digits or digits.startswith("_") or digits.endswith("_"):
                msg = f"invalid integer {token!r}"
                raise self._error(msg, at=at)
            allowed = {"0x": _HEX_DIGITS, "0o": _OCT_DIGITS, "0b": _BIN_DIGITS}[prefix]
            for c in digits:
                if c == "_":
                    continue
                if c not in allowed:
                    msg = f"invalid digit {c!r} in {token!r}"
                    raise self._error(msg, at=at)
            if "__" in digits:
                msg = f"consecutive underscores in {token!r}"
                raise self._error(msg, at=at)
            base = {"0x": 16, "0o": 8, "0b": 2}[prefix]
            value = int(digits.replace("_", ""), base)
            style_map: dict[str, IntStyle] = {"0x": "hex", "0o": "oct", "0b": "bin"}
            style = style_map[prefix]
            return IntegerNode(raw=token, value=value, style=style)
        # Decimal int with optional sign and underscores.
        sign = ""
        if body and body[0] in "+-":
            sign = body[0]
            body = body[1:]
        if not body:
            msg = f"invalid integer {token!r}"
            raise self._error(msg, at=at)
        if body.startswith("_") or body.endswith("_"):
            msg = f"invalid integer {token!r}"
            raise self._error(msg, at=at)
        if "__" in body:
            msg = f"consecutive underscores in {token!r}"
            raise self._error(msg, at=at)
        # Leading zero rule: no leading zeros except for the value 0 itself.
        digits_only = body.replace("_", "")
        if not digits_only.isdigit():
            msg = f"invalid integer {token!r}"
            raise self._error(msg, at=at)
        if len(digits_only) > 1 and digits_only.startswith("0"):
            msg = f"leading zeros are not allowed in {token!r}"
            raise self._error(msg, at=at)
        value = int(sign + digits_only)
        return IntegerNode(raw=token, value=value, style="dec")

    def _parse_float_token(self, token: str, *, at: int) -> FloatNode:
        body = token
        sign = ""
        if body and body[0] in "+-":
            sign = body[0]
            body = body[1:]
        if "__" in body:
            msg = f"consecutive underscores in {token!r}"
            raise self._error(msg, at=at)
        if body.startswith("_") or body.endswith("_") or "._" in body or "_." in body:
            msg = f"misplaced underscore in {token!r}"
            raise self._error(msg, at=at)
        if "_e" in body or "e_" in body or "_E" in body or "E_" in body:
            msg = f"misplaced underscore in {token!r}"
            raise self._error(msg, at=at)

        # Validate structure manually; ``float`` accepts forms TOML doesn't.
        norm = body.replace("_", "")
        # Split off exponent.
        exp_pos = -1
        for i, c in enumerate(norm):
            if c in ("e", "E"):
                exp_pos = i
                break
        if exp_pos != -1:
            mantissa = norm[:exp_pos]
            exponent = norm[exp_pos + 1 :]
            if not exponent or (exponent[0] in "+-" and len(exponent) == 1):
                msg = f"invalid float exponent in {token!r}"
                raise self._error(msg, at=at)
            if exponent[0] in "+-":
                exponent = exponent[1:]
            if not exponent.isdigit():
                msg = f"invalid float exponent in {token!r}"
                raise self._error(msg, at=at)
        else:
            mantissa = norm

        if "." in mantissa:
            int_part, _, frac_part = mantissa.partition(".")
            if not int_part or not frac_part:
                msg = f"invalid float {token!r}"
                raise self._error(msg, at=at)
            if not int_part.isdigit() or not frac_part.isdigit():
                msg = f"invalid float {token!r}"
                raise self._error(msg, at=at)
            if len(int_part) > 1 and int_part.startswith("0"):
                msg = f"leading zeros not allowed in float {token!r}"
                raise self._error(msg, at=at)
        else:
            if not mantissa.isdigit():
                msg = f"invalid float {token!r}"
                raise self._error(msg, at=at)
            if len(mantissa) > 1 and mantissa.startswith("0"):
                msg = f"leading zeros not allowed in float {token!r}"
                raise self._error(msg, at=at)
            if exp_pos == -1:
                # No '.' and no 'e': not a float.
                msg = f"invalid float {token!r}"
                raise self._error(msg, at=at)

        value = float(sign + norm)
        return FloatNode(raw=token, value=value)

    def _parse_datetime_token(self, token: str, *, at: int) -> DateTimeNode:
        # If the token looks like a date with a separator and the next
        # character is a space and what follows looks like a time, fold
        # them into one local-datetime token (TOML allows space as the
        # date/time separator).
        if (
            len(token) == 10
            and self._peek() == " "
            and self._pos + 1 < len(self._src)
            and self._src[self._pos + 1].isdigit()
            and self._pos + 3 < len(self._src)
            and self._src[self._pos + 3] == ":"
        ):
            self._pos += 1  # consume the space
            extra_end = self._scan_value_end(self._pos)
            extra = self._src[self._pos : extra_end]
            self._pos = extra_end
            full = token + " " + extra
            return self._parse_datetime_text(full, at=at, raw=full)
        return self._parse_datetime_text(token, at=at, raw=token)

    def _parse_datetime_text(
        self,
        text: str,
        *,
        at: int,
        raw: str,
    ) -> DateTimeNode:
        # Local time?
        if len(text) >= 3 and text[2] == ":":
            try:
                value = self._parse_time_text(text)
            except ValueError as exc:
                msg = f"invalid time {text!r}: {exc}"
                raise self._error(msg, at=at) from exc
            return DateTimeNode(raw=raw, value=value, kind="local-time")

        # Date or datetime.
        if len(text) < 10 or text[4] != "-" or text[7] != "-":
            msg = f"invalid date/datetime {text!r}"
            raise self._error(msg, at=at)
        date_part = text[:10]
        try:
            year = int(date_part[:4])
            month = int(date_part[5:7])
            day = int(date_part[8:10])
            d = date(year, month, day)
        except ValueError as exc:
            msg = f"invalid date {date_part!r}: {exc}"
            raise self._error(msg, at=at) from exc

        rest = text[10:]
        if not rest:
            return DateTimeNode(raw=raw, value=d, kind="local-date")
        if rest[0] not in ("T", "t", " "):
            msg = f"expected date/time separator, got {rest[0]!r}"
            raise self._error(msg, at=at)
        time_part = rest[1:]
        # Split off optional offset.
        offset_pos = -1
        for i, c in enumerate(time_part):
            if c in ("Z", "z", "+", "-") and i >= 1:
                offset_pos = i
                break
        if offset_pos == -1:
            try:
                t = self._parse_time_text(time_part)
            except ValueError as exc:
                msg = f"invalid time {time_part!r}: {exc}"
                raise self._error(msg, at=at) from exc
            return DateTimeNode(
                raw=raw,
                value=datetime.combine(d, t),
                kind="local-datetime",
            )
        try:
            t = self._parse_time_text(time_part[:offset_pos])
            tz = self._parse_offset(time_part[offset_pos:])
        except ValueError as exc:
            msg = f"invalid datetime {text!r}: {exc}"
            raise self._error(msg, at=at) from exc
        dt = datetime.combine(d, t).replace(tzinfo=tz)
        return DateTimeNode(raw=raw, value=dt, kind="offset-datetime")

    def _parse_time_text(self, text: str) -> time:
        # TOML 1.1: seconds are optional; "HH:MM" defaults to ":00".
        if len(text) < 5 or text[2] != ":":
            msg = f"bad time format: {text!r}"
            raise ValueError(msg)
        hh = int(text[:2])
        mm = int(text[3:5])
        rest = text[5:]
        if not rest:
            return time(hh, mm, 0, 0)
        if rest[0] != ":":
            msg = f"bad time format: {text!r}"
            raise ValueError(msg)
        if len(rest) < 3:
            msg = f"bad seconds in {text!r}"
            raise ValueError(msg)
        ss = int(rest[1:3])
        rest = rest[3:]
        usec = 0
        if rest:
            if rest[0] != ".":
                msg = f"bad fractional seconds in {text!r}"
                raise ValueError(msg)
            frac = rest[1:]
            if not frac or not frac.isdigit():
                msg = f"bad fractional seconds in {text!r}"
                raise ValueError(msg)
            # Truncate to 6 digits (microsecond precision).
            digits = (frac + "000000")[:6]
            usec = int(digits)
        return time(hh, mm, ss, usec)

    def _parse_offset(self, text: str) -> timezone:
        if text in ("Z", "z"):
            return timezone.utc
        if len(text) != 6 or text[0] not in "+-" or text[3] != ":":
            msg = f"bad timezone offset: {text!r}"
            raise ValueError(msg)
        sign = 1 if text[0] == "+" else -1
        hh = int(text[1:3])
        mm = int(text[4:6])
        if hh > 23 or mm > 59:
            msg = f"timezone offset out of range: {text!r}"
            raise ValueError(msg)
        delta = timedelta(hours=hh, minutes=mm) * sign
        return timezone(delta)

    # --- arrays -------------------------------------------------------

    def _parse_array(self) -> ArrayNode:
        assert self._peek() == "["
        self._advance(1)
        node = ArrayNode()
        leading = self._consume_array_trivia()
        if self._peek() == "]":
            node.final_trivia = leading
            self._advance(1)
            return node
        while True:
            value = self._parse_value()
            trailing = self._consume_array_trivia()
            has_comma = False
            post_comma = Trivia()
            if self._peek() == ",":
                self._advance(1)
                has_comma = True
                post_comma = self._consume_array_trivia()
            elif self._peek() != "]":
                msg = f"expected ',' or ']' in array, got {self._peek()!r}"
                raise self._error(msg)
            node.items.append(
                ArrayItem(
                    leading=leading,
                    value=value,
                    trailing=trailing,
                    has_comma=has_comma,
                    post_comma_trivia=post_comma,
                ),
            )
            if not has_comma:
                # We're at ']'.
                self._advance(1)
                return node
            leading = Trivia()  # next item's leading is empty; trivia
            # already attached as post_comma of previous item.
            if self._peek() == "]":
                # Trailing comma followed by closer.
                self._advance(1)
                return node
            # Next iteration starts a new value.

    # --- inline tables ------------------------------------------------

    def _parse_inline_table(self) -> InlineTableNode:
        assert self._peek() == "{"
        self._advance(1)
        node = InlineTableNode()
        # TOML 1.1: newlines and trailing comma are allowed inside an
        # inline table, with the same trivia rules as arrays.
        leading = self._consume_array_trivia()
        if self._peek() == "}":
            node.final_trivia = leading
            self._advance(1)
            return node
        # Track inline-table-local known keys.
        seen: set[tuple[str, ...]] = set()
        while True:
            key = self._parse_key()
            if key.path in seen:
                msg = f"duplicate key {'.'.join(key.path)!r} in inline table"
                raise self._error(msg)
            for i in range(1, len(key.path) + 1):
                seen.add(key.path[:i])
            pre_eq = self._consume_inline_ws()
            if self._peek() != "=":
                msg = f"expected '=' in inline table, got {self._peek()!r}"
                raise self._error(msg)
            self._advance(1)
            post_eq = self._consume_inline_ws()
            value = self._parse_value()
            trailing = self._consume_array_trivia()
            has_comma = False
            post_comma = Trivia()
            if self._peek() == ",":
                self._advance(1)
                has_comma = True
                post_comma = self._consume_array_trivia()
            elif self._peek() != "}":
                msg = f"expected ',' or '}}' in inline table, got {self._peek()!r}"
                raise self._error(msg)
            node.entries.append(
                InlineTableEntry(
                    leading=leading,
                    key=key,
                    pre_eq=pre_eq,
                    post_eq=post_eq,
                    value=value,
                    trailing=trailing,
                    has_comma=has_comma,
                    post_comma_trivia=post_comma,
                ),
            )
            if not has_comma:
                # We're at '}'.
                self._advance(1)
                return node
            leading = Trivia()
            if self._peek() == "}":
                # Trailing comma followed by the closer (TOML 1.1).
                self._advance(1)
                return node


# Re-export convenience.
def parse(src: str) -> DocumentNode:
    return _Parser(src).parse()


__all__ = ["parse"]
