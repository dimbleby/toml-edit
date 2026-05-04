"""Public top-level API for tomlrt."""

from __future__ import annotations

from typing import IO

from tomlrt._container import Document
from tomlrt._parser import _Parser


def loads(text: str) -> Document:
    """Parse a TOML document string into a [`Document`][tomlrt.Document]."""
    parser = _Parser(text)
    result = parser.parse()
    return Document._from_parse(result.slots, result.trailing, result.newline)  # noqa: SLF001


def load(fp: IO[bytes]) -> Document:
    """Parse a TOML document from a *binary* file-like object."""
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
    """Serialize a [`Document`][tomlrt.Document] and write to a *binary* stream."""
    fp.write(dumps(doc).encode("utf-8"))


__all__ = ["dump", "dumps", "load", "loads"]
