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
from collections.abc import MutableMapping
from typing import TYPE_CHECKING

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tomlrt._trivia import (
    CommentNode,
    NewlineNode,
    WhitespaceNode,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tomlrt._container import Container
    from tomlrt._slots import KVSlot
    from tomlrt._trivia import (
        Trivia,
    )


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
        v: object = value
        if not isinstance(v, str):
            msg = f"comment must be str, got {type(v).__name__}"
            raise TypeError(msg)
        if "\n" in value or "\r" in value:
            from tomlrt._errors import TOMLError  # noqa: PLC0415

            msg = "EOL comment must be single-line"
            raise TOMLError(msg)
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


def _split_leading_into_lines(leading: Trivia) -> list[list[object]]:
    """Group trivia pieces into logical "lines" terminated by Newline.

    Returns a list of lists; each inner list is the pieces that
    appeared on a single source line up to and including the
    terminating newline (if any).
    """
    lines: list[list[object]] = []
    cur: list[object] = []
    for p in leading.pieces:
        cur.append(p)
        if isinstance(p, NewlineNode):
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    return lines


def _extract_leading_comments(leading: Trivia) -> tuple[str, ...]:
    """Return the run of comment-bearing lines from ``leading``."""
    out: list[str] = []
    for line in _split_leading_into_lines(leading):
        for p in line:
            if isinstance(p, CommentNode):
                out.append(_decode_comment(p.text))
                break
    return tuple(out)


class LeadingCommentView(MutableMapping[str, tuple[str, ...]]):
    __slots__ = ("_c",)

    def __init__(self, container: Container) -> None:
        self._c = container

    def _slot(self, key: str) -> KVSlot | None:
        return _direct_kv_slot(self._c, key)

    @override
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        slot = self._slot(key)
        if slot is None:
            return False
        return any(isinstance(p, CommentNode) for p in slot.leading.pieces)

    @override
    def __getitem__(self, key: str) -> tuple[str, ...]:
        slot = self._slot(key)
        if slot is None or not any(
            isinstance(p, CommentNode) for p in slot.leading.pieces
        ):
            raise KeyError(key)
        return _extract_leading_comments(slot.leading)

    @override
    def __setitem__(self, key: str, value: tuple[str, ...]) -> None:
        slot = self._slot(key)
        if slot is None:
            msg = f"key {key!r} not in container"
            raise KeyError(msg)
        v: object = value
        if isinstance(v, str):
            msg = "leading comments must be a sequence of strings"
            raise TypeError(msg)
        comments = tuple(value)
        for c in comments:
            cv: object = c
            if not isinstance(cv, str):
                msg = "leading comments must be strings"
                raise TypeError(msg)
            if "\n" in c or "\r" in c:
                from tomlrt._errors import TOMLError  # noqa: PLC0415

                msg = "leading comment lines must be single-line"
                raise TOMLError(msg)
        from tomlrt._container import Document  # noqa: PLC0415

        lr = self._c._layout_root  # noqa: SLF001
        nl = lr._newline if isinstance(lr, Document) else "\n"  # noqa: SLF001
        # Replace the current leading comment block while preserving
        # any structural blank lines that preceded it.  Strategy:
        # walk existing pieces; drop any line that contained a comment
        # (the entire line, including its newline); keep any line that
        # was purely whitespace/blank.  Then prepend the new comment
        # lines as their own line each.
        kept: list[object] = []
        for line in _split_leading_into_lines(slot.leading):
            if any(isinstance(p, CommentNode) for p in line):
                continue
            kept.extend(line)
        new_pieces: list[object] = []
        for c in comments:
            new_pieces.append(CommentNode(_encode_comment(c)))
            new_pieces.append(NewlineNode(nl))
        # Place new comments before the kept (structural) trivia, so
        # that any pre-existing blank-line separator remains *above*
        # the new comments — matching the typical "blank line then
        # comment block then key" layout.
        slot.leading.pieces = [*kept, *new_pieces]  # type: ignore[list-item]

    @override
    def __delitem__(self, key: str) -> None:
        slot = self._slot(key)
        if slot is None:
            raise KeyError(key)
        if not any(isinstance(p, CommentNode) for p in slot.leading.pieces):
            raise KeyError(key)
        kept: list[object] = []
        for line in _split_leading_into_lines(slot.leading):
            if any(isinstance(p, CommentNode) for p in line):
                continue
            kept.extend(line)
        slot.leading.pieces = kept  # type: ignore[assignment]

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
            if any(isinstance(p, CommentNode) for p in slot.leading.pieces):
                seen.add(k)
                yield k

    @override
    def __len__(self) -> int:
        return sum(1 for _ in self)


__all__ = ["EolCommentView", "LeadingCommentView"]
