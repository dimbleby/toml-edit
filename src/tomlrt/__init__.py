"""tomlrt: a format-preserving TOML parser and writer."""

from __future__ import annotations

from tomlrt._document import AoT, Array, Document, Table
from tomlrt._errors import TOMLEditError, TOMLParseError
from tomlrt._public import dump, dumps, load, loads, parse

__all__ = [
    "AoT",
    "Array",
    "Document",
    "TOMLEditError",
    "TOMLParseError",
    "Table",
    "dump",
    "dumps",
    "load",
    "loads",
    "parse",
]

__version__ = "0.1.0"
