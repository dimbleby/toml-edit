"""toml-edit: a fast, ergonomic, format-preserving TOML parser and writer."""

from __future__ import annotations

from tomle._document import AoT, Array, Document, Table
from tomle._errors import TOMLEditError, TOMLParseError
from tomle._public import dump, dumps, load, loads, parse

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
