"""tomlrt: a format-preserving TOML reader and writer."""

from __future__ import annotations

from tomlrt._document import (
    AoT,
    Array,
    Document,
    Table,
    TomlInput,
)
from tomlrt._errors import TOMLError, TOMLParseError
from tomlrt._public import document, dump, dumps, load, loads, parse

__all__ = [
    "AoT",
    "Array",
    "Document",
    "TOMLError",
    "TOMLParseError",
    "Table",
    "TomlInput",
    "document",
    "dump",
    "dumps",
    "load",
    "loads",
    "parse",
]
