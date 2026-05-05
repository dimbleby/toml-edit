"""Comment side-channel views over slot trivia.

Two primary views per `Container`:

- `Container.comments` — `MutableMapping[str, str]` over the *EOL*
  comment of the direct-KV slot for each key.  Reading returns the
  decoded comment text (the leading ``#`` and one optional space
  are stripped); writing accepts the same shape and re-encodes.
- `Container.leading_comments` — `MutableMapping[str, tuple[str, ...]]`
  over the *leading* comment block on the direct-KV slot for each
  key.  Empty tuple means "no comments above this key".

Implementation notes:

- Only direct-KV slots are exposed.  Dotted-implicit shared slots
  (e.g. ``b.c = 1`` under ``[a]`` viewed from ``a``) are not
  addressable through the container's comment view; the slot's
  trivia is instead reachable through the dotted parent's view.
- Inline-table containers do not expose a comment view (TOML
  forbids comments inside an inline table).
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, MutableMapping
from typing import TYPE_CHECKING

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tomlrt._errors import TOMLError
from tomlrt._slots import StructuralHeaderSlot
from tomlrt._trivia import (
    CommentNode,
    NewlineNode,
    WhitespaceNode,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tomlrt._container import Container, Document
    from tomlrt._slots import KVSlot, Slot
    from tomlrt._trivia import (
        Trivia,
        TriviaPiece,
    )


def _validate_comment_text(text: str) -> None:
    """Reject a comment value that would not round-trip via the parser."""
    if "\n" in text or "\r" in text:
        msg = "comment must be single-line"
        raise TOMLError(msg)
    for ch in text:
        cp = ord(ch)
        # TOML comments allow TAB plus any printable Unicode; reject
        # other ASCII control chars and DEL.
        if cp == 0x09:
            continue
        if cp < 0x20 or cp == 0x7F:
            msg = f"comment may not contain control character U+{cp:04X}"
            raise TOMLError(msg)


def _validate_comment_str(value: object, name: str) -> str:
    """Type-check ``value`` is a str and validate its content; return it."""
    if not isinstance(value, str):
        msg = f"{name} must be str, got {type(value).__name__}"
        raise TypeError(msg)
    _validate_comment_text(value)
    return value


def _decode_comment(raw: str) -> str:
    """Strip the leading ``#`` and one optional space from a raw comment."""
    if raw.startswith("#"):
        rest = raw[1:]
        if rest.startswith(" "):
            return rest[1:]
        return rest
    return raw


def _encode_comment(text: str) -> str:
    """Encode a logical comment into a raw ``# ...`` form."""
    if text == "":
        return "#"
    return f"# {text}"


def _direct_kv_slot(c: Container, key: str) -> KVSlot | None:
    """Return the primary direct-KV slot for ``key`` in ``c``, or None."""
    from tomlrt._slots import KVSlot  # noqa: PLC0415

    refs = c._index.get(key)  # noqa: SLF001
    if not refs:
        return None
    for ref in refs:
        slot = ref.slot
        if (
            isinstance(slot, KVSlot)
            and slot.host_path == c._path  # noqa: SLF001
            and len(slot.key_parts) == 1
            and slot.key_parts[0].value == key
        ):
            return slot
    return None


class EolCommentView(MutableMapping[str, str]):
    __slots__ = ("_c",)

    def __init__(self, container: Container) -> None:
        self._c = container

    def _slot(self, key: str) -> KVSlot | None:
        return _direct_kv_slot(self._c, key)

    @override
    def __repr__(self) -> str:
        return repr(dict(self))

    @override
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        slot = self._slot(key)
        return slot is not None and slot.eol.comment is not None

    @override
    def __getitem__(self, key: str) -> str:
        slot = self._slot(key)
        if slot is None or slot.eol.comment is None:
            raise KeyError(key)
        return _decode_comment(slot.eol.comment.text)

    @override
    def __setitem__(self, key: str, value: str) -> None:
        slot = self._slot(key)
        if slot is None:
            msg = f"key {key!r} not in container"
            raise KeyError(msg)
        _validate_comment_str(value, "comment")
        # Ensure trailing whitespace separator before the new comment.
        if slot.eol.trailing_ws is None:
            slot.eol.trailing_ws = WhitespaceNode(" ")
        elif slot.eol.trailing_ws.text == "":
            slot.eol.trailing_ws.text = " "
        slot.eol.comment = CommentNode(_encode_comment(value))
        if slot.eol.newline is None:
            from tomlrt._container import Document  # noqa: PLC0415

            lr = self._c._layout_root  # noqa: SLF001
            nl = lr._newline if isinstance(lr, Document) else "\n"  # noqa: SLF001
            slot.eol.newline = NewlineNode(nl)

    @override
    def __delitem__(self, key: str) -> None:
        slot = self._slot(key)
        if slot is None or slot.eol.comment is None:
            raise KeyError(key)
        slot.eol.comment = None
        # Also drop the gap-whitespace that preceded the comment so we
        # don't leave a dangling tail like `key = 1   \n`.
        if slot.eol.trailing_ws is not None:
            slot.eol.trailing_ws = None

    @override
    def __iter__(self) -> Iterator[str]:
        seen: set[str] = set()
        for ref in self._c._refs:  # noqa: SLF001
            k = ref.local_key
            if k is None or k in seen:
                continue
            slot = self._slot(k)
            if slot is not None and slot.eol.comment is not None:
                seen.add(k)
                yield k

    @override
    def __len__(self) -> int:
        return sum(1 for _ in self)


def _split_leading_into_lines(leading: Trivia) -> list[list[TriviaPiece]]:
    """Group trivia pieces into logical "lines" terminated by Newline.

    Returns a list of lists; each inner list is the pieces that
    appeared on a single source line up to and including the
    terminating newline (if any).
    """
    lines: list[list[TriviaPiece]] = []
    cur: list[TriviaPiece] = []
    for p in leading.pieces:
        cur.append(p)
        if isinstance(p, NewlineNode):
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    return lines


def _line_is_blank(line: list[TriviaPiece]) -> bool:
    return not any(isinstance(p, CommentNode) for p in line) and any(
        isinstance(p, NewlineNode) for p in line
    )


def _line_is_comment(line: list[TriviaPiece]) -> bool:
    return any(isinstance(p, CommentNode) for p in line)


def _split_attached_block(
    leading: Trivia,
) -> tuple[list[list[TriviaPiece]], list[list[TriviaPiece]], list[TriviaPiece]]:
    """Split the leading into (above_blank, attached_comment_lines, slot_indent).

    The "attached" group is the contiguous run of comment lines
    immediately preceding the slot — i.e. with no blank line
    between the run and the slot itself.  Everything before the
    last blank-separator-or-start is "above_blank" (preamble +
    archived blocks).  ``slot_indent`` is any trailing whitespace-
    only, newline-less "indent" line (the slot's own column
    offset) — preserved separately so callers can reapply it
    when rebuilding the leading.
    """
    lines = _split_leading_into_lines(leading)
    indent: list[TriviaPiece] = []
    if lines and not any(isinstance(p, NewlineNode) for p in lines[-1]):
        last = lines[-1]
        # Only treat as indent if it has no comment.
        if not any(isinstance(p, CommentNode) for p in last):
            indent = last
            lines = lines[:-1]
    i = len(lines)
    while i > 0 and _line_is_comment(lines[i - 1]):
        i -= 1
    above = lines[:i]
    attached = lines[i:]
    return above, attached, indent


def _extract_leading_comments(leading: Trivia) -> tuple[str, ...]:
    """Return only the *attached* run of comment-bearing lines.

    The "attached" run is the contiguous block immediately above
    the slot — comments separated by a blank line are considered
    preamble or archived blocks and are excluded.
    """
    _above, attached, _indent = _split_attached_block(leading)
    out: list[str] = []
    for line in attached:
        for p in line:
            if isinstance(p, CommentNode):
                out.append(_decode_comment(p.text))
                break
    return tuple(out)


def _slot_has_attached_comments(slot: Slot) -> bool:
    leading = slot.leading
    _above, attached, _indent = _split_attached_block(leading)
    return any(_line_is_comment(line) for line in attached)


class LeadingCommentView(MutableMapping[str, tuple[str, ...]]):
    __slots__ = ("_c",)

    def __init__(self, container: Container) -> None:
        self._c = container

    def _slot(self, key: str) -> KVSlot | None:
        return _direct_kv_slot(self._c, key)

    @override
    def __repr__(self) -> str:
        return repr(dict(self))

    @override
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        slot = self._slot(key)
        if slot is None:
            return False
        return _slot_has_attached_comments(slot)

    @override
    def __getitem__(self, key: str) -> tuple[str, ...]:
        slot = self._slot(key)
        if slot is None or not _slot_has_attached_comments(slot):
            raise KeyError(key)
        return _extract_leading_comments(slot.leading)

    @override
    def __setitem__(self, key: str, value: tuple[str, ...]) -> None:
        slot = self._slot(key)
        if slot is None:
            msg = f"key {key!r} not in container"
            raise KeyError(msg)
        comments = _validate_comment_seq(value, "leading_comments")
        from tomlrt._container import Document  # noqa: PLC0415

        lr = self._c._layout_root  # noqa: SLF001
        nl = lr._newline if isinstance(lr, Document) else "\n"  # noqa: SLF001
        # Replace only the *attached* comment block; preserve any
        # preamble / archived blocks above any blank-line separator,
        # and re-apply the slot's own indent before each new comment
        # line and (implicitly) before the slot itself.
        above, _attached, indent = _split_attached_block(slot.leading)
        kept: list[TriviaPiece] = []
        for line in above:
            kept.extend(line)
        new_pieces: list[TriviaPiece] = []
        for c in comments:
            new_pieces.extend(indent)
            new_pieces.append(CommentNode(_encode_comment(c)))
            new_pieces.append(NewlineNode(nl))
        new_pieces.extend(indent)
        slot.leading.pieces = [*kept, *new_pieces]

    @override
    def __delitem__(self, key: str) -> None:
        slot = self._slot(key)
        if slot is None:
            raise KeyError(key)
        if not _slot_has_attached_comments(slot):
            raise KeyError(key)
        above, _attached, indent = _split_attached_block(slot.leading)
        kept: list[TriviaPiece] = []
        for line in above:
            kept.extend(line)
        kept.extend(indent)
        slot.leading.pieces = kept

    @override
    def __iter__(self) -> Iterator[str]:
        seen: set[str] = set()
        for ref in self._c._refs:  # noqa: SLF001
            k = ref.local_key
            if k is None or k in seen:
                continue
            slot = self._slot(k)
            if slot is None:
                continue
            if _slot_has_attached_comments(slot):
                seen.add(k)
                yield k

    @override
    def __len__(self) -> int:
        return sum(1 for _ in self)


def _header_slot(c: Container) -> StructuralHeaderSlot | None:
    """Return the StructuralHeaderSlot for a section container, or raise.

    Raises TOMLError on inline tables (no header to attach a comment to).
    Returns None if this container has no own header (document root /
    purely-implicit container).
    """
    if c._inline:  # noqa: SLF001
        msg = "header comment API is not available on inline tables"
        raise TOMLError(msg)
    hr = c._header_ref  # noqa: SLF001
    if hr is None:
        return None
    slot = hr.slot
    assert isinstance(slot, StructuralHeaderSlot)
    return slot


def _header_comment_get(c: Container) -> str | None:
    h = _header_slot(c)
    if h is None:
        msg = "container has no header to attach a comment to"
        raise TOMLError(msg)
    eol = h.eol
    if eol.comment is None:
        return None
    return _decode_comment(eol.comment.text)


def _header_comment_set(c: Container, value: str | None) -> None:
    h = _header_slot(c)
    if h is None:
        msg = "container has no header to attach a comment to"
        raise TOMLError(msg)
    eol = h.eol
    if value is None:
        if eol.comment is not None:
            eol.comment = None
            if (
                eol.trailing_ws is not None
                and eol.trailing_ws.text.strip(
                    " \t",
                )
                == ""
            ):
                # Drop the gap whitespace that preceded the comment.
                eol.trailing_ws = None
        return
    _validate_comment_str(value, "header_comment")
    if eol.trailing_ws is None:
        eol.trailing_ws = WhitespaceNode(" ")
    elif eol.trailing_ws.text == "":
        eol.trailing_ws.text = " "
    eol.comment = CommentNode(_encode_comment(value))
    if eol.newline is None:
        from tomlrt._container import Document  # noqa: PLC0415

        lr = c._layout_root  # noqa: SLF001
        nl = lr._newline if isinstance(lr, Document) else "\n"  # noqa: SLF001
        eol.newline = NewlineNode(nl)


def _header_leading_get(c: Container) -> tuple[str, ...]:
    h = _header_slot(c)
    if h is None:
        msg = "container has no header to attach leading comments to"
        raise TOMLError(msg)
    leading = h.leading
    return _extract_leading_comments(leading)


def _header_leading_set(c: Container, value: tuple[str, ...]) -> None:
    h = _header_slot(c)
    if h is None:
        msg = "container has no header to attach leading comments to"
        raise TOMLError(msg)
    comments = _validate_comment_seq(value, "header_leading_comments")
    from tomlrt._container import Document  # noqa: PLC0415

    lr = c._layout_root  # noqa: SLF001
    nl = lr._newline if isinstance(lr, Document) else "\n"  # noqa: SLF001
    leading = h.leading
    above, _attached, indent = _split_attached_block(leading)
    kept: list[TriviaPiece] = []
    for line in above:
        kept.extend(line)
    new_pieces: list[TriviaPiece] = []
    for cm in comments:
        new_pieces.extend(indent)
        new_pieces.append(CommentNode(_encode_comment(cm)))
        new_pieces.append(NewlineNode(nl))
    new_pieces.extend(indent)
    leading.pieces = [*kept, *new_pieces]


def _validate_comment_seq(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        msg = f"{name} must be an iterable of comment strings"
        raise TypeError(msg)
    if not isinstance(value, Iterable):
        msg = f"{name} must be an iterable of comment strings"
        raise TypeError(msg)
    out: list[str] = []
    for c in value:
        if not isinstance(c, str):
            msg = f"{name} entries must be strings"
            raise TypeError(msg)
        if "\n" in c or "\r" in c:
            msg = "preamble lines must not contain a line terminator"
            raise TOMLError(msg)
        _validate_comment_text(c)
        out.append(c)
    return tuple(out)


def _trivia_attached_split(
    t: Trivia,
) -> tuple[list[list[TriviaPiece]], list[list[TriviaPiece]], list[TriviaPiece]]:
    """Split arbitrary trivia into (above_blank, attached, indent)."""
    return _split_attached_block(t)


def _doc_preamble_get(doc: Document) -> tuple[str, ...]:
    head = doc._head  # noqa: SLF001
    if head is None:
        # Empty doc: read from _trailing.
        trailing = doc._trailing  # noqa: SLF001
        out: list[str] = []
        for line in _split_leading_into_lines(trailing):
            for p in line:
                if isinstance(p, CommentNode):
                    out.append(_decode_comment(p.text))
                    break
        return tuple(out)
    leading = head.leading
    above, _attached, _indent = _split_attached_block(leading)
    # Preamble = comment lines in the "above" block (separated by blank).
    out = []
    for line in above:
        for p in line:
            if isinstance(p, CommentNode):
                out.append(_decode_comment(p.text))
                break
    return tuple(out)


def _doc_preamble_set(doc: Document, value: tuple[str, ...]) -> None:
    comments = _validate_comment_seq(value, "preamble")
    nl = doc._newline  # noqa: SLF001
    head = doc._head  # noqa: SLF001
    if head is None:
        # Empty doc: write into _trailing.
        if not comments:
            doc._trailing.pieces = []  # noqa: SLF001
            return
        new_pieces: list[TriviaPiece] = []
        for c in comments:
            new_pieces.append(CommentNode(_encode_comment(c)))
            new_pieces.append(NewlineNode(nl))
        doc._trailing.pieces = new_pieces  # noqa: SLF001
        return
    leading = head.leading
    _above, attached, indent = _split_attached_block(leading)
    if not comments:
        # Drop preamble; keep only the attached block (and indent).
        kept: list[TriviaPiece] = []
        for line in attached:
            kept.extend(line)
        kept.extend(indent)
        leading.pieces = kept
        return
    new_pieces = []
    for c in comments:
        new_pieces.append(CommentNode(_encode_comment(c)))
        new_pieces.append(NewlineNode(nl))
    # Add a blank-line separator between preamble and attached/key.
    new_pieces.append(NewlineNode(nl))
    kept = []
    for line in attached:
        kept.extend(line)
    kept.extend(indent)
    leading.pieces = [*new_pieces, *kept]


def _doc_epilogue_get(doc: Document) -> tuple[str, ...]:
    head = doc._head  # noqa: SLF001
    trailing = doc._trailing  # noqa: SLF001
    if head is None:
        # Empty doc: there is no epilogue separate from preamble; both
        # routes read _trailing, but tests expect epilogue == () on an
        # empty doc.
        return ()
    out: list[str] = []
    for line in _split_leading_into_lines(trailing):
        for p in line:
            if isinstance(p, CommentNode):
                out.append(_decode_comment(p.text))
                break
    return tuple(out)


def _doc_epilogue_set(doc: Document, value: tuple[str, ...]) -> None:
    comments = _validate_comment_seq(value, "epilogue")
    head = doc._head  # noqa: SLF001
    if head is None and comments:
        msg = "cannot set epilogue: document has no structural content"
        raise TOMLError(msg)
    nl = doc._newline  # noqa: SLF001
    new_pieces: list[TriviaPiece] = []
    for c in comments:
        new_pieces.append(CommentNode(_encode_comment(c)))
        new_pieces.append(NewlineNode(nl))
    doc._trailing.pieces = new_pieces  # noqa: SLF001


__all__ = ["EolCommentView", "LeadingCommentView"]
