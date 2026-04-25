"""Public top-level API for tomlrt."""

from __future__ import annotations

from collections.abc import Mapping
from typing import IO, Any

from tomlrt._document import Document, Table
from tomlrt._nodes import DocumentNode
from tomlrt._parser import parse as _parse_to_cst


def _populate(table: Table, data: Mapping[str, Any]) -> None:
    """Recursively pour ``data`` into ``table``.

    Uses section/AoT shapes for nested mappings and lists-of-mappings,
    scalars for leaves.
    """
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


def document(data: Mapping[str, Any] | None = None) -> Document:
    """Return a fresh [`Document`][tomlrt.Document], optionally populated from ``data``.

    Without arguments, returns an empty document — equivalent to
    ``parse("")`` but more discoverable when the intent is "build a
    TOML file from scratch".

    With a mapping, recursively populates the document so that:

    * nested mappings become standard ``[section]`` blocks (not
      inline tables);
    * lists of mappings become ``[[array.of.tables]]`` blocks;
    * everything else is set with ordinary key-value assignment
      (so leaf lists become inline arrays, leaf dicts can't appear).

    Existing [`Table`][tomlrt.Table] / [`AoT`][tomlrt.AoT] /
    [`Array`][tomlrt.Array] views are
    deep-cloned, so the returned document shares no mutable state
    with ``data``.
    """
    doc = Document(DocumentNode())
    if data is not None:
        _populate(doc, data)
    return doc


def parse(text: str) -> Document:
    """Parse a TOML document string into a [`Document`][tomlrt.Document]."""
    cst = _parse_to_cst(text)
    return Document(cst)


def loads(text: str) -> Document:
    """Alias for [`parse`][tomlrt.parse], mirroring the stdlib ``tomllib`` API."""
    return parse(text)


def load(fp: IO[bytes]) -> Document:
    """Parse a TOML document from a *binary* file-like object.

    The file must be opened in binary mode (``open(path, "rb")``); text
    mode would perform locale-dependent decoding and newline translation,
    breaking the byte-exact round-trip guarantee. Raises `TypeError`
    for a text stream.
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
    """Serialize a [`Document`][tomlrt.Document] back to a TOML string."""
    return doc.render()


def dump(doc: Document, fp: IO[bytes]) -> None:
    """Serialize a [`Document`][tomlrt.Document] and write it to a *binary* stream.

    The file must be opened in binary mode (``open(path, "wb")``).
    """
    fp.write(dumps(doc).encode("utf-8"))


__all__ = ["document", "dump", "dumps", "load", "loads", "parse"]
