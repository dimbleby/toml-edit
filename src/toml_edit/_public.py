"""Public top-level API for toml-edit."""

from __future__ import annotations

from typing import IO

from toml_edit._document import Document
from toml_edit._parser import parse as _parse_to_cst


def parse(text: str) -> Document:
    """Parse a TOML document string into a :class:`Document`."""
    cst = _parse_to_cst(text)
    return Document(cst)


def loads(text: str) -> Document:
    """Alias for :func:`parse`, mirroring the stdlib ``tomllib`` API."""
    return parse(text)


def load(fp: IO[bytes]) -> Document:
    """Parse a TOML document from a *binary* file-like object.

    The file must be opened in binary mode (``open(path, "rb")``).
    Mirroring :mod:`tomllib`, we refuse text-mode files because:

    * TOML is defined to be UTF-8 -- decoding it ourselves removes the
      risk of a locale-dependent text-mode default (e.g. ``cp1252`` on
      Windows) silently mangling non-ASCII strings.
    * Text mode performs newline translation (``\\r\\n`` -> ``\\n``),
      which would destroy this library's format-preservation guarantee
      for files originating on Windows.

    Raises :class:`TypeError` if ``fp`` looks like a text-mode stream.
    """
    data = fp.read()
    if not isinstance(data, (bytes, bytearray)):
        msg = (  # type: ignore[unreachable]
            "toml_edit.load expects a binary file (open with mode='rb'); "
            f"got a text stream returning {type(data).__name__}"
        )
        raise TypeError(msg)
    return parse(bytes(data).decode("utf-8"))


def dumps(doc: Document) -> str:
    """Serialize a :class:`Document` back to a TOML string."""
    return doc.render()


def dump(doc: Document, fp: IO[bytes]) -> None:
    """Serialize a :class:`Document` and write it to a *binary* stream.

    The file must be opened in binary mode (``open(path, "wb")``).
    Symmetric with :func:`load`: writing UTF-8 bytes ourselves avoids
    locale-dependent encoding and newline translation, so a
    ``load`` -> ``dump`` round-trip is byte-for-byte stable.
    """
    fp.write(dumps(doc).encode("utf-8"))


__all__ = ["dump", "dumps", "load", "loads", "parse"]
