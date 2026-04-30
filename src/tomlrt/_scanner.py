"""Cursor + scanner used by `tomlrt._parser`.

The scanner owns the `(src, end, pos)` triple — it is the single
cursor authority while the parser drives. Methods that recognise a
syntactic construct return either a fully-formed CST node (for the
messy, allocation-heavy bits like strings and trivia blocks) or a
small tuple/`str` (for bare value tokens that the parser still
needs to dispatch on).

This module currently covers cursor primitives, trivia / comment
scanners, and string scanners. Key and bare-value scanners migrate
in subsequent steps.

String scanning is, by design, *semantic* — escape sequences are
decoded, surrogate code points are rejected, the leading newline
after the opening triple-quote delimiter is trimmed, and the
multi-line trailing-quote allowance is enforced. The returned
`StringNode` carries both the raw lexeme (filled by `scan_string`)
and the decoded value.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, Final

from tomlrt._errors import TOMLParseError
from tomlrt._nodes import (
    BoolNode,
    CommentNode,
    DateTimeNode,
    FloatNode,
    IntegerNode,
    KeyPart,
    NewlineNode,
    StringNode,
    Trivia,
    WhitespaceNode,
)

if TYPE_CHECKING:
    from tomlrt._nodes import IntStyle, ValueNode

# Comment body: anything except newline + control chars (tab is OK).
_RE_COMMENT_BODY: Final = re.compile(r"[^\r\n\x00-\x08\x0b-\x1f\x7f]*")

# Body of a basic string: any run of chars that are NOT a quote,
# backslash, newline, or control char (control = U+0000-U+001F or
# U+007F, except tab which we *do* allow).
_RE_BASIC_STR_BODY: Final = re.compile(r'[^"\\\n\r\x00-\x08\x0b-\x1f\x7f]+')
# Body of a literal string: anything except quote, newline, control
# char (tab and newline-not-allowed handled by the caller).
_RE_LITERAL_STR_BODY: Final = re.compile(r"[^'\n\r\x00-\x08\x0b-\x1f\x7f]+")
# Body of a multi-line basic string fragment: stops at " or \\ or \r or
# \n or a control char. \n and \r\n are valid in ML strings, so the
# caller handles them; we stop at \r so the caller can verify it's
# followed by \n and emit a normalized pair.
_RE_ML_BASIC_BODY: Final = re.compile(r'[^"\\\r\n\x00-\x08\x0b-\x1f\x7f]+')
_RE_ML_LITERAL_BODY: Final = re.compile(r"[^'\r\n\x00-\x08\x0b-\x1f\x7f]+")
# Bare key: ASCII alphanum + underscore + dash. (TOML 1.1 broadens
# this; if/when tomlrt opts in, widen the pattern here.)
_RE_BARE_KEY: Final = re.compile(r"[A-Za-z0-9_\-]+")

_HEX_DIGITS: Final[frozenset[str]] = frozenset("0123456789abcdefABCDEF")
_OCT_DIGITS: Final[frozenset[str]] = frozenset("01234567")
_BIN_DIGITS: Final[frozenset[str]] = frozenset("01")

# First character that ends a bare-value token (whitespace, newline,
# array/table close, comma, comment).
_RE_VALUE_END: Final = re.compile(r"[ \t\n\r,\]}#]")

# Simple backslash-escape map, shared across every string parse so we
# don't rebuild the dict on each escape character.
_SIMPLE_ESCAPES: Final[dict[str, str]] = {
    "b": "\b",
    "t": "\t",
    "n": "\n",
    "f": "\f",
    "r": "\r",
    "e": "\x1b",  # TOML 1.1: ESC
    '"': '"',
    "\\": "\\",
}


class _Scanner:
    __slots__ = ("end", "pos", "src")

    def __init__(self, src: str) -> None:
        self.src = src
        self.end = len(src)
        self.pos = 0

    # ------------------------------------------------------------------
    # Cursor primitives
    # ------------------------------------------------------------------

    def peek(self, offset: int = 0) -> str:
        """Return the character `offset` chars ahead of the cursor.

        Returns the empty string at or past EOF; never raises.
        """
        i = self.pos + offset
        if i >= self.end:
            return ""
        return self.src[i]

    def starts_with(self, s: str) -> bool:
        """Return True iff `s` matches the source from the cursor."""
        return self.src.startswith(s, self.pos)

    def eof(self) -> bool:
        return self.pos >= self.end

    def advance(self, n: int = 1) -> str:
        """Move the cursor forward `n` chars; return the consumed slice.

        Does *not* validate that `n` characters remain — the caller is
        responsible (this is fine because every existing call site
        first peeks for the characters it expects).
        """
        s = self.src[self.pos : self.pos + n]
        self.pos += n
        return s

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def line_col(self, pos: int) -> tuple[int, int]:
        """Return the 1-based (line, column) for source offset `pos`."""
        line = 1
        last_nl = -1
        for i in range(pos):
            if self.src[i] == "\n":
                line += 1
                last_nl = i
        col = pos - last_nl
        return line, col

    def error(self, message: str, *, at: int | None = None) -> TOMLParseError:
        """Build a `TOMLParseError` pointing at `at` (default: cursor)."""
        offset = self.pos if at is None else at
        line, col = self.line_col(offset)
        return TOMLParseError(message, line=line, col=col, offset=offset)

    # ------------------------------------------------------------------
    # Trivia / comment scanners
    # ------------------------------------------------------------------

    def scan_comment(self) -> CommentNode:
        """Consume a comment from `#` to (but not including) the newline.

        The cursor must be on `#`. Raises if the comment body contains
        a control character other than tab.
        """
        src = self.src
        start = self.pos
        m = _RE_COMMENT_BODY.match(src, start + 1)
        assert m is not None  # pattern is unbounded above (*).
        end_pos = m.end()
        if end_pos < self.end:
            ch = src[end_pos]
            if ch != "\n" and ch != "\r":
                self.pos = end_pos
                cp = ord(ch)
                msg = f"invalid control character U+{cp:04X} in comment"
                raise self.error(msg)
        self.pos = end_pos
        return CommentNode(src[start:end_pos])

    def scan_doc_trivia(self) -> Trivia:
        """Consume a document-scope trivia block.

        Whitespace, blank lines and full-line comments. Stops *before*
        the next non-trivia character on a line — so the structural
        token (or EOF) follows immediately at the cursor.
        """
        trivia = Trivia()
        pieces = trivia.pieces
        src = self.src
        end = self.end
        pos = self.pos
        while pos < end:
            ch = src[pos]
            if ch == " " or ch == "\t":
                ws_start = pos
                pos += 1
                while pos < end:
                    c = src[pos]
                    if c != " " and c != "\t":
                        break
                    pos += 1
                pieces.append(WhitespaceNode(src[ws_start:pos]))
            elif ch == "#":
                self.pos = pos
                pieces.append(self.scan_comment())
                pos = self.pos
            elif ch == "\n":
                pos += 1
                pieces.append(NewlineNode("\n"))
            elif ch == "\r":
                if pos + 1 >= end or src[pos + 1] != "\n":
                    self.pos = pos
                    msg = "stray carriage return"
                    raise self.error(msg)
                pos += 2
                pieces.append(NewlineNode("\r\n"))
            else:
                break
        self.pos = pos
        return trivia

    def scan_inline_ws(self) -> WhitespaceNode | None:
        """Consume one run of inline whitespace; no newlines or comments.

        Returns `None` (and leaves the cursor untouched) if the next
        character is not space or tab.
        """
        src = self.src
        end = self.end
        pos = self.pos
        if pos >= end:
            return None
        ch = src[pos]
        if ch != " " and ch != "\t":
            return None
        start = pos
        pos += 1
        while pos < end:
            c = src[pos]
            if c != " " and c != "\t":
                break
            pos += 1
        self.pos = pos
        return WhitespaceNode(src[start:pos])

    def scan_array_trivia(self) -> Trivia:
        """Consume trivia inside an array (or TOML 1.1 inline table).

        Whitespace, newlines and comments are all permitted. Stops
        before the next structural character.
        """
        trivia = Trivia()
        pieces = trivia.pieces
        src = self.src
        end = self.end
        pos = self.pos
        while pos < end:
            ch = src[pos]
            if ch == " " or ch == "\t":
                ws_start = pos
                pos += 1
                while pos < end:
                    c = src[pos]
                    if c != " " and c != "\t":
                        break
                    pos += 1
                pieces.append(WhitespaceNode(src[ws_start:pos]))
            elif ch == "\n":
                pos += 1
                pieces.append(NewlineNode("\n"))
            elif ch == "\r" and pos + 1 < end and src[pos + 1] == "\n":
                pos += 2
                pieces.append(NewlineNode("\r\n"))
            elif ch == "#":
                self.pos = pos
                pieces.append(self.scan_comment())
                pos = self.pos
            else:
                break
        self.pos = pos
        return trivia

    def scan_eol(
        self,
    ) -> tuple[WhitespaceNode | None, CommentNode | None, NewlineNode | None]:
        """Consume optional trailing-ws + comment + newline (or EOF).

        Raises if a non-newline, non-comment, non-EOF character is
        found after the optional whitespace.
        """
        trailing = self.scan_inline_ws()
        comment: CommentNode | None = None
        src = self.src
        end = self.end
        pos = self.pos
        ch = src[pos] if pos < end else ""
        if ch == "#":
            comment = self.scan_comment()
            pos = self.pos
            ch = src[pos] if pos < end else ""
        newline: NewlineNode | None = None
        if ch == "\n":
            self.pos = pos + 1
            newline = NewlineNode("\n")
        elif ch == "\r" and pos + 1 < end and src[pos + 1] == "\n":
            self.pos = pos + 2
            newline = NewlineNode("\r\n")
        elif pos < end:
            msg = f"expected newline or end of file, got {ch!r}"
            raise self.error(msg)
        return trailing, comment, newline

    # ------------------------------------------------------------------
    # Strings
    # ------------------------------------------------------------------

    def scan_string(self, quote: str) -> StringNode:
        """Scan a string starting at the cursor; populate `raw`.

        `quote` is the opening quote character (double or single).
        The returned `StringNode` carries both the verbatim source
        slice (for round-tripping) and the decoded value.
        """
        start = self.pos
        if quote == '"':
            multiline = self.starts_with('"""')
            node = self.scan_basic_string(allow_multiline=multiline)
        else:
            multiline = self.starts_with("'''")
            node = self.scan_literal_string(allow_multiline=multiline)
        node.raw = self.src[start : self.pos]
        return node

    def scan_basic_string(self, *, allow_multiline: bool) -> StringNode:
        """Scan a basic string. Decodes escapes; never sets `raw`.

        Callers that need round-trip-faithful raw text use
        `scan_string` instead, which wraps this and fills `raw`.
        """
        if allow_multiline and self.starts_with('"""'):
            return self._scan_ml_basic_string()
        if self.peek() != '"':
            msg = "expected '\"' to start basic string"
            raise self.error(msg)
        self.pos += 1
        src = self.src
        end = self.end
        out: list[str] = []
        while True:
            m = _RE_BASIC_STR_BODY.match(src, self.pos)
            if m is not None:
                out.append(m.group(0))
                self.pos = m.end()
            if self.pos >= end:
                msg = "unterminated basic string"
                raise self.error(msg)
            ch = src[self.pos]
            if ch == '"':
                self.pos += 1
                return StringNode(raw="", value="".join(out), style="basic")
            if ch == "\\":
                out.append(self._scan_escape())
                continue
            if ch == "\n" or ch == "\r":
                msg = "newline in basic string"
                raise self.error(msg)
            cp = ord(ch)
            if cp == 0x7F:
                msg = "invalid control character U+007F in string"
                raise self.error(msg)
            msg = f"invalid control character U+{cp:04X} in string"
            raise self.error(msg)

    def _scan_ml_basic_string(self) -> StringNode:
        assert self.starts_with('"""')
        self.pos += 3
        # A newline immediately after the opening delimiter is trimmed.
        if self.peek() == "\n":
            self.pos += 1
        elif self.peek() == "\r" and self.peek(1) == "\n":
            self.pos += 2
        out: list[str] = []
        while True:
            if self.eof():
                msg = "unterminated multi-line basic string"
                raise self.error(msg)
            m = _RE_ML_BASIC_BODY.match(self.src, self.pos)
            if m is not None:
                out.append(m.group(0))
                self.pos = m.end()
                if self.eof():
                    continue
            if self.starts_with('"""'):
                # Up to two extra trailing quotes are allowed inside.
                self.pos += 3
                extras = 0
                while extras < 2 and self.peek() == '"':
                    out.append('"')
                    self.pos += 1
                    extras += 1
                return StringNode(raw="", value="".join(out), style="ml-basic")
            ch = self.peek()
            if ch == '"':
                # Single or double quote (not the closing triple) — emit and
                # continue. The body regex stops at any quote.
                out.append('"')
                self.pos += 1
                continue
            if ch == "\\":
                # Line-ending backslash: trim trailing ws+newline+leading-ws.
                if self.peek(1) in ("\n", " ", "\t", "\r"):
                    save = self.pos
                    self.pos += 1
                    # Skip trailing inline ws on this line.
                    while self.peek() in (" ", "\t"):
                        self.pos += 1
                    if self.peek() == "\n" or (
                        self.peek() == "\r" and self.peek(1) == "\n"
                    ):
                        # Eat one or more whitespace lines.
                        while True:
                            if self.peek() == "\n":
                                self.pos += 1
                            elif self.peek() == "\r" and self.peek(1) == "\n":
                                self.pos += 2
                            else:
                                break
                            while self.peek() in (" ", "\t"):
                                self.pos += 1
                        continue
                    # Not actually a line-ending backslash; rewind and
                    # treat as a normal escape.
                    self.pos = save
                out.append(self._scan_escape())
                continue
            if ch == "\r":
                if self.peek(1) != "\n":
                    msg = "stray carriage return in string"
                    raise self.error(msg)
                out.append("\r\n")
                self.pos += 2
                continue
            if ch == "\n":
                out.append("\n")
                self.pos += 1
                continue
            cp = ord(ch)
            if cp <= 0x1F and cp != 0x09:
                msg = f"invalid control character U+{cp:04X} in string"
                raise self.error(msg)
            if cp == 0x7F:
                msg = "invalid control character U+007F in string"
                raise self.error(msg)
            out.append(ch)
            self.pos += 1

    def scan_literal_string(self, *, allow_multiline: bool) -> StringNode:
        """Scan a literal string. No escapes; never sets `raw`."""
        if allow_multiline and self.starts_with("'''"):
            return self._scan_ml_literal_string()
        if self.peek() != "'":
            msg = 'expected "\'" to start literal string'
            raise self.error(msg)
        self.pos += 1
        src = self.src
        end = self.end
        start = self.pos
        m = _RE_LITERAL_STR_BODY.match(src, start)
        if m is not None:
            self.pos = m.end()
        if self.pos >= end:
            msg = "unterminated literal string"
            raise self.error(msg)
        ch = src[self.pos]
        if ch == "'":
            value = src[start : self.pos]
            self.pos += 1
            return StringNode("", value, "literal")
        if ch == "\n" or ch == "\r":
            msg = "newline in literal string"
            raise self.error(msg)
        cp = ord(ch)
        if cp == 0x7F:
            msg = "invalid control character U+007F in string"
            raise self.error(msg)
        msg = f"invalid control character U+{cp:04X} in string"
        raise self.error(msg)

    def _scan_ml_literal_string(self) -> StringNode:
        assert self.starts_with("'''")
        self.pos += 3
        if self.peek() == "\n":
            self.pos += 1
        elif self.peek() == "\r" and self.peek(1) == "\n":
            self.pos += 2
        out: list[str] = []
        while True:
            if self.eof():
                msg = "unterminated multi-line literal string"
                raise self.error(msg)
            m = _RE_ML_LITERAL_BODY.match(self.src, self.pos)
            if m is not None:
                out.append(m.group(0))
                self.pos = m.end()
                if self.eof():
                    continue
            if self.starts_with("'''"):
                self.pos += 3
                extras = 0
                while extras < 2 and self.peek() == "'":
                    out.append("'")
                    self.pos += 1
                    extras += 1
                return StringNode(raw="", value="".join(out), style="ml-literal")
            ch = self.peek()
            if ch == "'":
                # Single quote, not the closing triple — emit and continue.
                out.append("'")
                self.pos += 1
                continue
            if ch == "\n":
                out.append("\n")
                self.pos += 1
                continue
            if ch == "\r":
                if self.peek(1) != "\n":
                    msg = "stray carriage return in string"
                    raise self.error(msg)
                out.append("\r\n")
                self.pos += 2
                continue
            cp = ord(ch)
            if cp == 0x7F:
                msg = "invalid control character U+007F in string"
                raise self.error(msg)
            msg = f"invalid control character U+{cp:04X} in string"
            raise self.error(msg)

    def _scan_escape(self) -> str:
        assert self.peek() == "\\"
        self.pos += 1
        ch = self.peek()
        self.pos += 1
        escaped = _SIMPLE_ESCAPES.get(ch)
        if escaped is not None:
            return escaped
        if ch == "x":  # TOML 1.1: 2-digit hex escape (U+0000..U+00FF)
            return self._scan_unicode_escape(2)
        if ch == "u":
            return self._scan_unicode_escape(4)
        if ch == "U":
            return self._scan_unicode_escape(8)
        msg = f"invalid escape sequence: \\{ch}"
        raise self.error(msg, at=self.pos - 2)

    def _scan_unicode_escape(self, n: int) -> str:
        if self.pos + n > self.end:
            msg = f"truncated unicode escape; expected {n} hex digits"
            raise self.error(msg)
        hex_str = self.src[self.pos : self.pos + n]
        for c in hex_str:
            if c not in _HEX_DIGITS:
                msg = f"invalid hex digit {c!r} in unicode escape"
                raise self.error(msg)
        self.pos += n
        cp = int(hex_str, 16)
        if cp > 0x10FFFF or 0xD800 <= cp <= 0xDFFF:
            msg = f"invalid unicode scalar U+{cp:04X}"
            raise self.error(msg)
        return chr(cp)

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------

    def scan_key_part(self) -> KeyPart:
        """Scan one key part: bare, basic-quoted, or literal-quoted."""
        src = self.src
        pos = self.pos
        ch = src[pos] if pos < self.end else ""
        if ch == '"':
            start = pos
            s = self.scan_basic_string(allow_multiline=False)
            return KeyPart(src[start : self.pos], s.value, "basic")
        if ch == "'":
            start = pos
            s = self.scan_literal_string(allow_multiline=False)
            return KeyPart(src[start : self.pos], s.value, "literal")
        m = _RE_BARE_KEY.match(src, pos)
        if m is not None:
            end_pos = m.end()
            raw = src[pos:end_pos]
            self.pos = end_pos
            return KeyPart(raw, raw, "bare")
        msg = f"expected key, got {ch!r}"
        raise self.error(msg)

    def scan_key_separator(self) -> str | None:
        """Scan an optional dotted-key separator: ``ws "." ws``.

        Returns the separator's exact lexeme (which may include
        leading and trailing whitespace) and advances the cursor
        past it. Returns `None` and leaves the cursor put if the
        next non-whitespace character is not a dot — that is a
        regular `pre_eq` whitespace situation, owned by the caller.
        """
        src = self.src
        end = self.end
        save = self.pos
        ws_end = save
        while ws_end < end:
            c = src[ws_end]
            if c != " " and c != "\t":
                break
            ws_end += 1
        if ws_end >= end or src[ws_end] != ".":
            return None
        sep_end = ws_end + 1
        while sep_end < end:
            c = src[sep_end]
            if c != " " and c != "\t":
                break
            sep_end += 1
        self.pos = sep_end
        return src[save:sep_end]

    # ------------------------------------------------------------------
    # Bare value tokens: bool, special-float keywords, integer, float,
    # date / time / datetime. The parser dispatches strings, arrays and
    # inline tables itself; everything else funnels through
    # ``scan_value_atom``.
    # ------------------------------------------------------------------

    def scan_value_atom(self) -> ValueNode:
        """Scan a non-container, non-string value at the cursor.

        Recognises bools, special floats (``inf`` / ``nan``, with
        optional sign), integers, floats, and date/time/datetime
        literals. Bool and special-float keywords are matched on the
        whole bare token, so ``trueish`` / ``infinity`` reliably
        error rather than silently parsing as ``true`` / ``inf``
        followed by garbage.
        """
        start = self.pos
        end = self._scan_value_end(start)
        token = self.src[start:end]
        if not token:
            msg = f"expected value, got {self.peek()!r}"
            raise self.error(msg)

        # Whole-token keyword classification.
        if token == "true":  # noqa: S105
            self.pos = end
            return BoolNode("true", value=True)
        if token == "false":  # noqa: S105
            self.pos = end
            return BoolNode("false", value=False)
        if token in ("inf", "+inf"):
            self.pos = end
            return FloatNode(raw=token, value=float("inf"))
        if token == "-inf":  # noqa: S105
            self.pos = end
            return FloatNode(raw=token, value=float("-inf"))
        if token in ("nan", "+nan", "-nan"):
            self.pos = end
            return FloatNode(raw=token, value=float("nan"))

        # Date/time literals always carry a fixed punctuation char in
        # a known position. Try them before numbers so e.g. ``1979-…``
        # is not mistaken for an integer.
        if self._looks_like_datetime(token):
            self.pos = end
            return self._parse_datetime_token(token, at=start)

        if self._looks_like_float(token):
            self.pos = end
            return self._parse_float_token(token, at=start)

        self.pos = end
        return self._parse_integer_token(token, at=start)

    def _scan_value_end(self, start: int) -> int:
        """Return the offset of the first char that ends a bare value.

        Stops at whitespace, newline, ``,``, ``]``, ``}``, ``#``, EOF.
        """
        m = _RE_VALUE_END.search(self.src, start)
        return m.start() if m is not None else len(self.src)

    @staticmethod
    def _looks_like_datetime(token: str) -> bool:
        # Date: ``YYYY-MM-DD``; local time: ``HH:MM:SS``; datetime
        # contains both and is detected via the date head.
        if len(token) >= 5 and token[4] == "-" and token[:4].isdigit():
            return True
        return bool(len(token) >= 3 and token[2] == ":" and token[:2].isdigit())

    @staticmethod
    def _looks_like_float(token: str) -> bool:
        # A decimal float must contain ``.``, ``e`` or ``E``;
        # hex/oct/bin integers never do. A leading sign is fine to
        # keep since none of the marker characters are signs.
        body = token[1:] if token[:1] in "+-" else token
        if body.startswith(("0x", "0o", "0b")):
            return False
        return "." in body or "e" in body or "E" in body

    def _parse_integer_token(self, token: str, *, at: int) -> IntegerNode:
        body = token
        if body.startswith(("0x", "0o", "0b")):
            prefix = body[:2]
            digits = body[2:]
            if not digits or digits.startswith("_") or digits.endswith("_"):
                msg = f"invalid integer {token!r}"
                raise self.error(msg, at=at)
            allowed = {"0x": _HEX_DIGITS, "0o": _OCT_DIGITS, "0b": _BIN_DIGITS}[prefix]
            for c in digits:
                if c == "_":
                    continue
                if c not in allowed:
                    msg = f"invalid digit {c!r} in {token!r}"
                    raise self.error(msg, at=at)
            if "__" in digits:
                msg = f"consecutive underscores in {token!r}"
                raise self.error(msg, at=at)
            base = {"0x": 16, "0o": 8, "0b": 2}[prefix]
            value = int(digits.replace("_", ""), base)
            style_map: dict[str, IntStyle] = {"0x": "hex", "0o": "oct", "0b": "bin"}
            return IntegerNode(token, value, style_map[prefix])

        sign = ""
        if body and body[0] in "+-":
            sign = body[0]
            body = body[1:]
        if not body:
            msg = f"invalid integer {token!r}"
            raise self.error(msg, at=at)
        if body.startswith("_") or body.endswith("_"):
            msg = f"invalid integer {token!r}"
            raise self.error(msg, at=at)
        if "__" in body:
            msg = f"consecutive underscores in {token!r}"
            raise self.error(msg, at=at)
        digits_only = body.replace("_", "")
        if not digits_only.isdigit():
            msg = f"invalid integer {token!r}"
            raise self.error(msg, at=at)
        if len(digits_only) > 1 and digits_only.startswith("0"):
            msg = f"leading zeros are not allowed in {token!r}"
            raise self.error(msg, at=at)
        value = int(sign + digits_only)
        return IntegerNode(token, value, "dec")

    def _parse_float_token(self, token: str, *, at: int) -> FloatNode:
        body = token
        sign = ""
        if body and body[0] in "+-":
            sign = body[0]
            body = body[1:]
        if "__" in body:
            msg = f"consecutive underscores in {token!r}"
            raise self.error(msg, at=at)
        if body.startswith("_") or body.endswith("_") or "._" in body or "_." in body:
            msg = f"misplaced underscore in {token!r}"
            raise self.error(msg, at=at)
        if "_e" in body or "e_" in body or "_E" in body or "E_" in body:
            msg = f"misplaced underscore in {token!r}"
            raise self.error(msg, at=at)

        # Validate structure manually; ``float`` accepts forms TOML doesn't.
        norm = body.replace("_", "")
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
                raise self.error(msg, at=at)
            if exponent[0] in "+-":
                exponent = exponent[1:]
            if not exponent.isdigit():
                msg = f"invalid float exponent in {token!r}"
                raise self.error(msg, at=at)
        else:
            mantissa = norm

        if "." in mantissa:
            int_part, _, frac_part = mantissa.partition(".")
            if not int_part or not frac_part:
                msg = f"invalid float {token!r}"
                raise self.error(msg, at=at)
            if not int_part.isdigit() or not frac_part.isdigit():
                msg = f"invalid float {token!r}"
                raise self.error(msg, at=at)
            if len(int_part) > 1 and int_part.startswith("0"):
                msg = f"leading zeros not allowed in float {token!r}"
                raise self.error(msg, at=at)
        else:
            if not mantissa.isdigit():
                msg = f"invalid float {token!r}"
                raise self.error(msg, at=at)
            if len(mantissa) > 1 and mantissa.startswith("0"):
                msg = f"leading zeros not allowed in float {token!r}"
                raise self.error(msg, at=at)
            if exp_pos == -1:
                # No '.' and no 'e': not a float.
                msg = f"invalid float {token!r}"
                raise self.error(msg, at=at)

        value = float(sign + norm)
        return FloatNode(token, value)

    def _parse_datetime_token(self, token: str, *, at: int) -> DateTimeNode:
        # TOML allows a single space as the date/time separator. If we
        # just scanned a 10-char date and the next chars look like
        # ``" HH:..."``, fold them into one local-datetime token.
        src = self.src
        pos = self.pos
        if (
            len(token) == 10
            and pos < len(src)
            and src[pos] == " "
            and pos + 3 < len(src)
            and src[pos + 1].isdigit()
            and src[pos + 3] == ":"
        ):
            pos += 1
            extra_end = self._scan_value_end(pos)
            extra = src[pos:extra_end]
            self.pos = extra_end
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
                raise self.error(msg, at=at) from exc
            return DateTimeNode(raw, value, "local-time")

        if len(text) < 10 or text[4] != "-" or text[7] != "-":
            msg = f"invalid date/datetime {text!r}"
            raise self.error(msg, at=at)
        date_part = text[:10]
        try:
            year = int(date_part[:4])
            month = int(date_part[5:7])
            day = int(date_part[8:10])
            d = date(year, month, day)
        except ValueError as exc:
            msg = f"invalid date {date_part!r}: {exc}"
            raise self.error(msg, at=at) from exc

        rest = text[10:]
        if not rest:
            return DateTimeNode(raw, d, "local-date")
        if rest[0] not in ("T", "t", " "):
            msg = f"expected date/time separator, got {rest[0]!r}"
            raise self.error(msg, at=at)
        time_part = rest[1:]
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
                raise self.error(msg, at=at) from exc
            return DateTimeNode(raw, datetime.combine(d, t), "local-datetime")
        try:
            t = self._parse_time_text(time_part[:offset_pos])
            tz = self._parse_offset(time_part[offset_pos:])
        except ValueError as exc:
            msg = f"invalid datetime {text!r}: {exc}"
            raise self.error(msg, at=at) from exc
        dt = datetime.combine(d, t).replace(tzinfo=tz)
        return DateTimeNode(raw, dt, "offset-datetime")

    @staticmethod
    def _parse_time_text(text: str) -> time:
        # TOML 1.1: seconds are optional; ``HH:MM`` defaults to ``:00``.
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
            digits = (frac + "000000")[:6]
            usec = int(digits)
        return time(hh, mm, ss, usec)

    @staticmethod
    def _parse_offset(text: str) -> timezone:
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


__all__ = ["_Scanner"]
