"""Public top-level API for toml-edit."""

from __future__ import annotations

from typing import IO

from tomle._document import Document
from tomle._parser import parse as _parse_to_cst


def parse(text: str) -> Document:
    """Parse a TOML document string into a :class:`Document`."""
    cst = _parse_to_cst(text)
    return Document(cst)


def loads(text: str) -> Document:
    """Alias for :func:`parse`, mirroring the stdlib ``tomllib`` API."""
    return parse(text)


def load(fp: IO[str] | IO[bytes]) -> Document:
    """Parse a TOML document from a text or binary file-like object."""
    data = fp.read()
    text = data.decode("utf-8") if isinstance(data, bytes) else data
    return parse(text)


def dumps(doc: Document) -> str:
    """Serialize a :class:`Document` back to a TOML string."""
    return doc.render()


def dump(doc: Document, fp: IO[str]) -> None:
    """Serialize a :class:`Document` and write it to a text file-like object."""
    fp.write(dumps(doc))


__all__ = ["dump", "dumps", "load", "loads", "parse"]
