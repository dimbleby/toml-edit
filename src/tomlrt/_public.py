"""Public top-level API for tomlrt."""

from __future__ import annotations

import warnings
from typing import IO, TYPE_CHECKING, Any

from tomlrt._document import Document
from tomlrt._parser import _Parser

if TYPE_CHECKING:
    from collections.abc import Mapping


def document(data: Mapping[str, Any] | None = None) -> Document:
    """Deprecated alias for [`Document`][tomlrt.Document].

    Use ``Document(data)`` instead. This wrapper is retained for
    backwards compatibility and will be removed in a future release.
    """
    warnings.warn(
        "tomlrt.document() is deprecated; use tomlrt.Document() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return Document(data)


def loads(text: str) -> Document:
    """Parse a TOML document string into a [`Document`][tomlrt.Document]."""
    parser = _Parser(text)
    cst = parser.parse()
    return Document._from_node(cst, newline=parser.detected_newline())  # noqa: SLF001


def parse(text: str) -> Document:
    """Alias for [`loads`][tomlrt.parse]."""
    return loads(text)


def load(fp: IO[bytes]) -> Document:
    """Parse a TOML document from a *binary* file-like object.

    The file must be opened in binary mode (``open(path, "wb")``).
    """
    data = fp.read()
    if not isinstance(data, (bytes, bytearray)):
        msg = (  # type: ignore[unreachable]
            "tomlrt.load expects a binary file (open with mode='rb'); "
            f"got a text stream returning {type(data).__name__}"
        )
        raise TypeError(msg)
    return loads(bytes(data).decode("utf-8"))


def dumps(doc: Document) -> str:
    """Serialize a [`Document`][tomlrt.Document] back to a TOML string."""
    return doc.render()


def dump(doc: Document, fp: IO[bytes]) -> None:
    """Serialize a [`Document`][tomlrt.Document] and write it to a *binary* stream.

    The file must be opened in binary mode (``open(path, "wb")``).
    """
    fp.write(dumps(doc).encode("utf-8"))


__all__ = ["document", "dump", "dumps", "load", "loads", "parse"]
