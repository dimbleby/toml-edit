"""tomlrt: a format-preserving TOML reader and writer."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from tomlrt._container import AoT, Array, Document, Table, TomlInput
from tomlrt._errors import TOMLError, TOMLParseError
from tomlrt._public import dump, dumps, load, loads

if TYPE_CHECKING:
    from collections.abc import Mapping


def document(data: Mapping[str, Any] | None = None) -> Document:
    """Deprecated alias for ``tomlrt.Document(data)``."""
    warnings.warn(
        "tomlrt.document() is deprecated; use tomlrt.Document() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return Document(data)


def parse(src: str) -> Document:
    """Deprecated alias for ``tomlrt.loads()``."""
    warnings.warn(
        "tomlrt.parse() is deprecated; use tomlrt.loads() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return loads(src)


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
