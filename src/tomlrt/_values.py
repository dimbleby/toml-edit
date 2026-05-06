"""Value types — the inline-value layer.

Every TOML value (scalar, inline array, inline table) is one of
these. They are pure data: no document/slot stream awareness.

Each scalar carries its source ``lexeme`` together with the
decoded Python value so that re-emission is byte-exact.

`ArrayValue` and `InlineTableValue` carry their full internal
layout (every separator, comment, and whitespace run) for the
same reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from tomlrt._trivia import Trivia

if TYPE_CHECKING:
    from datetime import date, datetime, time


StringStyle = Literal["basic", "literal", "ml-basic", "ml-literal"]
IntStyle = Literal["dec", "hex", "oct", "bin"]
DateLikeKind = Literal[
    "offset-datetime",
    "local-datetime",
    "local-date",
    "local-time",
]


@dataclass(slots=True, eq=False)
class StringValue:
    lexeme: str  # including quotes
    value: str
    style: StringStyle

    def render(self) -> str:
        return self.lexeme


@dataclass(slots=True, eq=False)
class IntegerValue:
    lexeme: str
    value: int
    style: IntStyle

    def render(self) -> str:
        return self.lexeme


@dataclass(slots=True, eq=False)
class FloatValue:
    lexeme: str
    value: float

    def render(self) -> str:
        return self.lexeme


@dataclass(slots=True, eq=False)
class BoolValue:
    lexeme: str  # "true" or "false"
    value: bool

    def render(self) -> str:
        return self.lexeme


@dataclass(slots=True, eq=False)
class DateTimeValue:
    lexeme: str
    value: datetime | date | time
    kind: DateLikeKind

    def render(self) -> str:
        return self.lexeme


# ---------------------------------------------------------------------------
# Inline arrays
# ---------------------------------------------------------------------------


@dataclass(slots=True, eq=False)
class ArrayItem:
    """One slot inside an inline array.

    Layout: ``leading value trailing [comma post_comma_trivia]``.
    """

    leading: Trivia
    value: Value
    trailing: Trivia
    has_comma: bool
    post_comma_trivia: Trivia

    def render(self) -> str:
        out = f"{self.leading.render()}{self.value.render()}{self.trailing.render()}"
        if self.has_comma:
            out += f",{self.post_comma_trivia.render()}"
        return out


@dataclass(slots=True, eq=False)
class ArrayValue:
    """Inline array literal (``[ ... ]``).

    Trivia ownership (canonical model):
      - ``header_trivia`` owns the gap immediately after ``[`` and
        before the first item — bracket pad / leading newline / indent
        / interior comments above item 0.
      - ``items[0].leading`` is always empty.
      - ``items[k].leading`` (k >= 1) owns the entire physical gap
        before items[k]: structural newline + indent + above-item
        comment block + indent before the value.
      - ``items[k].post_comma_trivia`` carries only the row-attached
        EOL section: same-line whitespace + EOL comment + the
        terminating newline of that comment row.  Empty if no EOL
        comment.
      - ``final_trivia`` owns the gap before ``]`` (bracket pad /
        trailing newline). For an empty array this is the only place
        interior trivia can live.
    """

    items: list[ArrayItem] = field(default_factory=list)
    header_trivia: Trivia = field(default_factory=Trivia)
    final_trivia: Trivia = field(default_factory=Trivia)

    def render(self) -> str:
        body = "".join([item.render() for item in self.items])
        return (
            f"[{self.header_trivia.render()}{body}"
            f"{self.final_trivia.render()}]"
        )


# ---------------------------------------------------------------------------
# Inline tables
# ---------------------------------------------------------------------------


@dataclass(slots=True, eq=False)
class KeyPart:
    """A single dotted-key component."""

    raw: str  # source representation including any surrounding quotes
    value: str  # the decoded key string
    kind: Literal["bare", "basic", "literal"]

    def render(self) -> str:
        return self.raw


def quote_basic_key(s: str) -> str:
    """Encode ``s`` as a basic-quoted TOML key (escaping where required)."""
    out = ['"']
    for ch in s:
        c = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif c < 0x20 or c == 0x7F:
            out.append(f"\\u{c:04X}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


_RE_BARE_KEY_FULL = re.compile(r"\A[A-Za-z0-9_\-]+\Z")


def make_keypart(name: str) -> KeyPart:
    """Build a ``KeyPart`` for ``name``, choosing bare vs basic-quoted."""
    if _RE_BARE_KEY_FULL.match(name):
        return KeyPart(raw=name, value=name, kind="bare")
    return KeyPart(raw=quote_basic_key(name), value=name, kind="basic")


def make_keyparts(path: tuple[str, ...]) -> list[KeyPart]:
    """Build a list of ``KeyPart``s for each segment of ``path``."""
    return [make_keypart(p) for p in path]


@dataclass(slots=True, eq=False)
class InlineTableEntry:
    """One ``key = value`` slot inside an inline table."""

    leading: Trivia
    key_parts: list[KeyPart]
    key_seps: list[str]  # length = len(key_parts) - 1
    pre_eq: str
    post_eq: str
    value: Value
    trailing: Trivia
    has_comma: bool
    post_comma_trivia: Trivia

    def render_key(self) -> str:
        if len(self.key_parts) == 1:
            return self.key_parts[0].render()
        out: list[str] = []
        seps = self.key_seps
        for i, part in enumerate(self.key_parts):
            if i:
                out.append(seps[i - 1])
            out.append(part.render())
        return "".join(out)

    def render(self) -> str:
        out = (
            f"{self.leading.render()}{self.render_key()}"
            f"{self.pre_eq}={self.post_eq}"
            f"{self.value.render()}{self.trailing.render()}"
        )
        if self.has_comma:
            out += f",{self.post_comma_trivia.render()}"
        return out


@dataclass(slots=True, eq=False)
class InlineTableValue:
    """Inline table literal (``{ ... }``).

    Trivia ownership matches :class:`ArrayValue` — see that docstring
    for the canonical model.
    """

    entries: list[InlineTableEntry] = field(default_factory=list)
    header_trivia: Trivia = field(default_factory=Trivia)
    final_trivia: Trivia = field(default_factory=Trivia)

    def render(self) -> str:
        body = "".join([e.render() for e in self.entries])
        return (
            f"{{{self.header_trivia.render()}{body}"
            f"{self.final_trivia.render()}}}"
        )


Value = (
    StringValue
    | IntegerValue
    | FloatValue
    | BoolValue
    | DateTimeValue
    | ArrayValue
    | InlineTableValue
)


__all__ = [
    "ArrayItem",
    "ArrayValue",
    "BoolValue",
    "DateLikeKind",
    "DateTimeValue",
    "FloatValue",
    "InlineTableEntry",
    "InlineTableValue",
    "IntStyle",
    "IntegerValue",
    "KeyPart",
    "StringStyle",
    "StringValue",
    "Value",
]
