"""Public top-level API for tomlrt."""

from __future__ import annotations

from collections.abc import Mapping
from typing import IO

from tomlrt._document import Document, Table
from tomlrt._nodes import DocumentNode
from tomlrt._parser import parse as _parse_to_cst


def _populate(table: Table, data: Mapping[str, object]) -> None:
    """Recursively pour ``data`` into ``table`` using section/AoT shapes
    for nested mappings and lists-of-mappings, scalars for leaves."""
    from tomlrt._document import AoT  # noqa: PLC0415

    for key, value in data.items():
        if isinstance(value, Mapping):
            sub = table.install(key, Table.section())
            _populate(sub, value)
        elif (
            isinstance(value, list)
            and value
            and all(isinstance(item, Mapping) for item in value)
        ):
            aot = table.install(key, AoT())
            for entry in value:
                assert isinstance(entry, Mapping)
                _populate(aot.add(), entry)
        else:
            table[key] = value


def document(data: Mapping[str, object] | None = None) -> Document:
    """Return a fresh :class:`Document`, optionally populated from ``data``.

    Without arguments, returns an empty document — equivalent to
    ``parse("")`` but more discoverable when the intent is "build a
    TOML file from scratch".

    With a mapping, recursively populates the document so that:

    * nested mappings become standard ``[section]`` blocks (not
      inline tables);
    * lists of mappings become ``[[array.of.tables]]`` blocks;
    * everything else is set with ordinary key-value assignment
      (so leaf lists become inline arrays, leaf dicts can't appear).

    Existing :class:`Table` / :class:`AoT` / :class:`Array` views are
    deep-cloned, so the returned document shares no mutable state
    with ``data``.
    """
    doc = Document(DocumentNode())
    if data is not None:
        _populate(doc, data)
    return doc


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
            "tomlrt.load expects a binary file (open with mode='rb'); "
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


__all__ = ["document", "dump", "dumps", "load", "loads", "parse"]
