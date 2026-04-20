"""tomlrt: a format-preserving TOML parser and writer."""

from __future__ import annotations

from tomlrt._document import AoT, Array, Document, SectionSpec, Table
from tomlrt._errors import TOMLError, TOMLParseError
from tomlrt._public import document, dump, dumps, load, loads, parse

__all__ = [
    "AoT",
    "Array",
    "Document",
    "SectionSpec",
    "TOMLError",
    "TOMLParseError",
    "Table",
    "document",
    "dump",
    "dumps",
    "load",
    "loads",
    "parse",
]

__version__ = "0.1.0"
