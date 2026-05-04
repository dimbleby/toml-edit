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
    """Inline array literal (``[ ... ]``)."""

    items: list[ArrayItem] = field(default_factory=list)
    final_trivia: Trivia = field(default_factory=Trivia)
    """Trivia after the last item (or comma) and before the closing ``]``."""

    def render(self) -> str:
        body = "".join([item.render() for item in self.items])
        return f"[{body}{self.final_trivia.render()}]"


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
    entries: list[InlineTableEntry] = field(default_factory=list)
    final_trivia: Trivia = field(default_factory=Trivia)

    def render(self) -> str:
        body = "".join([e.render() for e in self.entries])
        return f"{{{body}{self.final_trivia.render()}}}"


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
