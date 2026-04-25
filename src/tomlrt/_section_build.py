"""Section construction, cloning, and insertion helpers.

Builds new ``[name]`` / ``[[name]]`` blocks from scratch, deep-clones
existing sections (rebasing their key prefix), and splices blocks back
into a :class:`DocumentNode`. The cloning paths reach into the logical
view layer (``AoT`` / ``_StdTable`` privates) by design — they are the
inverse of the view layer's "give me the CST that backs this view".
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from tomlrt._errors import TOMLError
from tomlrt._nodes import (
    Key,
    KeyValueNode,
    NewlineNode,
    SectionNode,
    TableHeaderNode,
    Trivia,
    WhitespaceNode,
)
from tomlrt._synthesise import make_key_part
from tomlrt._trivia import _prepend_blank_line

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from tomlrt._document import AoT, _StdTable
    from tomlrt._nodes import (
        CommentNode,
        DocumentNode,
        HeaderKind,
        InlineTableEntry,
        InlineTableNode,
    )


def _new_section(
    path: tuple[str, ...],
    *,
    kind: HeaderKind = "table",
    leading: Trivia | None = None,
    trailing: WhitespaceNode | None = None,
    trailing_comment: CommentNode | None = None,
) -> SectionNode:
    """Build an empty ``[path]`` (or ``[[path]]``) section.

    Trivia defaults to empty; pass ``leading`` / ``trailing`` /
    ``trailing_comment`` to carry over comment material from a node
    that the new section is replacing (used by the promotion paths).
    """
    parts = [make_key_part(p) for p in path]
    seps = ["."] * (len(parts) - 1)
    header = TableHeaderNode(
        leading=leading if leading is not None else Trivia(),
        kind=kind,
        inner_pre=None,
        key=Key(parts=parts, separators=seps),
        inner_post=None,
        trailing=trailing,
        trailing_comment=trailing_comment,
        newline=NewlineNode("\n"),
    )
    return SectionNode(header=header, entries=[])


def _kv_from_inline_entry(entry: InlineTableEntry, *, deep: bool) -> KeyValueNode:
    """Wrap an inline-table entry as a standalone ``key = value`` KV.

    Used by the promotion paths to convert inline-table contents into
    section entries. ``deep=True`` deep-clones the key and value (used
    when the source might still be reachable from elsewhere).
    """
    return KeyValueNode(
        leading=Trivia(),
        key=deepcopy(entry.key) if deep else entry.key,
        pre_eq=WhitespaceNode(" "),
        post_eq=WhitespaceNode(" "),
        value=deepcopy(entry.value) if deep else entry.value,
        trailing=None,
        trailing_comment=None,
        newline=NewlineNode("\n"),
    )


def _build_promoted_section(
    path: tuple[str, ...],
    inline: InlineTableNode,
    source_kv: KeyValueNode,
) -> SectionNode:
    """Build a ``[path]`` section containing ``inline``'s entries.

    Comments that lived above the original inline-table KV (its
    ``leading`` trivia) and any inline EOL comment are carried over to
    the new header so authoring intent is preserved.
    """
    section = _new_section(
        path,
        leading=Trivia(list(source_kv.leading.pieces)),
        trailing=source_kv.trailing,
        trailing_comment=source_kv.trailing_comment,
    )
    section.entries = [_kv_from_inline_entry(e, deep=False) for e in inline.entries]
    return section


def _build_promoted_aot_section(
    path: tuple[str, ...],
    inline: InlineTableNode,
) -> SectionNode:
    """Build a ``[[path]]`` section containing ``inline``'s entries.

    Used by :meth:`Table.promote_array` to convert each element of an
    inline array of inline tables into its own AoT entry.
    """
    section = _new_section(path, kind="array")
    section.entries = [_kv_from_inline_entry(e, deep=True) for e in inline.entries]
    return section


def _iter_rebased(
    sections: Iterable[SectionNode],
    src_path: tuple[str, ...],
    full_path: tuple[str, ...],
) -> Iterable[tuple[SectionNode, tuple[str, ...]]]:
    """Yield ``(sec, new_path)`` for every section whose header path is
    rooted at ``src_path``, with ``new_path`` rebased onto ``full_path``.

    Skips implicit (header-less) sections and sections whose key falls
    outside ``src_path``. The clone-and-rebase and rebase-in-place
    helpers consume this in one shape so they cannot drift apart.
    """
    splen = len(src_path)
    for sec in sections:
        hdr = sec.header
        if hdr is None or len(hdr.key.path) < splen or hdr.key.path[:splen] != src_path:
            continue
        yield sec, (*full_path, *hdr.key.path[splen:])


def _clone_sections_rebased(
    sections: Iterable[SectionNode],
    src_path: tuple[str, ...],
    full_path: tuple[str, ...],
) -> list[SectionNode]:
    """Deep-clone every header section under ``src_path``, rebasing the prefix.

    Implicit pre-header sections (``header is None``) and headers
    whose path doesn't extend ``src_path`` are skipped. Other sections
    are deep-cloned with their key rebased: relative depth below
    ``src_path`` is preserved, so ``[a.sub.deep]`` cloned at
    ``("t",)`` becomes ``[t.sub.deep]``. Shared rebase primitive
    behind :func:`_clone_aot_sections` and :func:`_clone_table_sections`.
    """
    out: list[SectionNode] = []
    for sec, new_path in _iter_rebased(sections, src_path, full_path):
        cloned = deepcopy(sec)
        assert cloned.header is not None
        cloned.header.key = _make_dotted_key(new_path)
        out.append(cloned)
    return out


def _clone_aot_sections(
    value: AoT,
    full_path: tuple[str, ...],
) -> list[SectionNode]:
    """Deep-clone every CST section that contributes to ``value``, rebased.

    Each AoT entry contributes its ``[[path]]`` header *and* any
    sub-sections in its owned range (e.g. ``[path.sub]`` /
    ``[[path.nested]]``). All of them are deep-cloned and have their
    header paths rewritten so the ``len(value._path)``-prefix is
    replaced by ``full_path``. Surrounding placement (insert index,
    blank-line policy, dict-storage sync) is the caller's job.
    """
    sections = _aot_owned_sections(value)
    return _clone_sections_rebased(sections, value._path, full_path)  # noqa: SLF001


def _rebase_aot_sections_inplace(
    value: AoT,
    full_path: tuple[str, ...],
) -> list[SectionNode]:
    """Like :func:`_clone_aot_sections` but rebases in place, no clone.

    Used when live-attaching an unattached AoT: the orphan section
    nodes themselves migrate into the destination document, so we
    rewrite their header paths in place rather than producing
    independent copies. Only sections rooted at ``value._path`` are
    returned (matching the clone path's filter), so the two helpers
    are interchangeable shapes.
    """
    sections = _aot_owned_sections(value)
    out: list[SectionNode] = []
    for sec, new_path in _iter_rebased(sections, value._path, full_path):  # noqa: SLF001
        assert sec.header is not None
        sec.header.key = _make_dotted_key(new_path)
        out.append(sec)
    return out


def _aot_owned_sections(value: AoT) -> list[SectionNode]:
    """All section nodes contributing to ``value``, in document order."""
    doc_node = value._doc_node  # noqa: SLF001
    blocks = (doc_node.aot_entry_block(header) for header in value._own_sections())  # noqa: SLF001
    return [sec for block in blocks for sec in block]


def _new_host_section(path: tuple[str, ...]) -> SectionNode:
    """Synthesize an empty ``[path]`` section, ready to host dotted KVs."""
    return SectionNode(
        header=TableHeaderNode(
            leading=Trivia(),
            kind="table",
            inner_pre=None,
            key=_make_dotted_key(path),
            inner_post=None,
            trailing=None,
            trailing_comment=None,
            newline=NewlineNode("\n"),
        ),
        entries=[],
    )


def _clone_table_sections(
    value: _StdTable,
    full_path: tuple[str, ...],
    *,
    head_kind: HeaderKind = "table",
) -> list[SectionNode]:
    """Deep-clone every CST section that contributes to ``value``, rebased.

    The returned sections are independent of ``value._doc_node`` and have
    their headers rewritten so the ``len(value._path)``-prefix is replaced
    by ``full_path``. Surrounding placement is the caller's responsibility.

    Two sources of contributing CST data are handled: sections at or below
    ``value._path`` (cloned and re-keyed) and dotted KVs in ancestor
    sections that extend into ``value._path`` (cloned into a host section
    at ``full_path``, synthesised on demand). For a ``Document`` source
    (``value._path == ()``) the implicit pre-header section's entries are
    treated as ancestor extras, since :meth:`_compute_extras` returns
    ``None`` in that case.

    The leading section's header kind is forced to ``head_kind``: pass
    ``"table"`` (the default) to install as a ``[full_path]`` standard
    section, or ``"array"`` to install as a ``[[full_path]]`` AoT entry.
    Without this normalisation an AoT-entry source cloned into a
    standard-table slot would keep its ``[[..]]`` header (and vice
    versa).
    """
    src_path = value._path  # noqa: SLF001
    fplen = len(full_path)
    doc = value._doc_node  # noqa: SLF001

    src_scope = value._scope()  # noqa: SLF001
    src_sections = src_scope if src_scope is not None else doc.sections
    new_secs = _clone_sections_rebased(src_sections, src_path, full_path)
    head = next(
        (
            s
            for s in new_secs
            if s.header is not None and len(s.header.key.path) == fplen
        ),
        None,
    )

    if len(src_path) == 0:
        extras = [
            (kv.key.path, kv)
            for sec in doc.sections
            if sec.header is None
            for kv in sec.entries
        ]
    else:
        extras = value._compute_extras() or []  # noqa: SLF001

    if extras:
        if head is None:
            head = _new_host_section(full_path)
            new_secs.insert(0, head)
        for rel_path, kv in extras:
            cloned_kv = deepcopy(kv)
            cloned_kv.key = _make_dotted_key(rel_path)
            head.entries.append(cloned_kv)

    if head is not None and head.header is not None:
        head.header.kind = head_kind

    return new_secs


def _apply_prior_leading(
    new_secs: Sequence[SectionNode],
    prior_leading: Trivia | None,
) -> None:
    """Transplant a prior section's header trivia onto a replacement block.

    ``_prepare_section_slot`` snapshots the leading trivia of the
    section being replaced (the comments / blank lines that sat above
    its ``[name]`` / ``[[name]]`` line) before purging. When a new
    block of sections is about to be spliced into the same slot, we
    move that trivia onto the first new section's header so the
    replacement preserves the visual context of the original.

    A no-op when there was no prior section, when the new block is
    empty, or when the new block starts with an entries-only section
    (no header to attach the trivia to).
    """
    if prior_leading is None or not new_secs:
        return
    first = new_secs[0]
    if first.header is None:
        return
    first.header.leading = prior_leading


def _insert_section_block(
    doc_node: DocumentNode,
    insert_at: int,
    new_secs: Sequence[SectionNode],
    *,
    separate_within: bool = True,
    add_blank: bool = True,
    bump_next: bool = False,
) -> None:
    """Splice a freshly-built block of ``[ ... ]`` / ``[[ ... ]]`` sections.

    With ``add_blank=True`` (the default) a blank line is inserted
    before the block whenever ``sections[:insert_at]`` already holds
    rendered content, and (when ``separate_within`` is True) between
    consecutive entries within ``new_secs``. Pass ``add_blank=False``
    to render the block packed against its neighbours -- used by AoT
    callers that mirror the user's existing inter-sibling style and
    decide it is "no blank lines".

    Pass ``bump_next=True`` to also blank-separate the section that
    will follow the new block. AoT entry inserts at non-end positions
    need this so two ``[[..]]`` headers don't render glued together;
    the default ``False`` preserves the trailing neighbour's authored
    leading -- right for sub-section installs and section replaces,
    where the user's prior layout established the gap and the new
    block has no domain-level invariant about sibling separation.

    Pass ``separate_within=False`` when each section in ``new_secs``
    already carries its own inter-header trivia (e.g. cloned sections
    from another document).
    """
    sections = doc_node.sections
    if add_blank:
        preceding_has_content = any(
            s.header is not None or s.entries for s in sections[:insert_at]
        )
        for i, ns in enumerate(new_secs):
            if (i == 0 and preceding_has_content) or (i > 0 and separate_within):
                assert ns.header is not None
                _prepend_blank_line(ns.header.leading)
        if bump_next and insert_at < len(sections):
            next_hdr = sections[insert_at].header
            if next_hdr is not None:
                _prepend_blank_line(next_hdr.leading)
    if new_secs and new_secs[0].header is not None:
        doc_node.adopt_preamble_into(new_secs[0].header.leading)
    sections[insert_at:insert_at] = new_secs


def _parse_key_path(path: str | tuple[str, ...]) -> tuple[str, ...]:
    """Split a dotted key path string into its bare segments.

    A string is split on ``.`` (``"a.b.c"`` → ``("a", "b", "c")``).
    A tuple is taken verbatim: use this form to express a single
    segment containing a literal dot (e.g. ``("foo.bar",)`` to name a
    table whose only key is the quoted name ``"foo.bar"``).
    """
    if isinstance(path, tuple):
        if not path:
            msg = "key path must not be empty"
            raise TOMLError(msg)
        for p in path:
            if not isinstance(p, str):
                msg = (  # type: ignore[unreachable]
                    f"key path {path!r} segment must be str, not {type(p).__name__}"
                )
                raise TypeError(msg)
        if any(p == "" for p in path):
            msg = f"key path {path!r} contains an empty segment"
            raise TOMLError(msg)
        return path
    if not isinstance(path, str):
        msg = (  # type: ignore[unreachable]
            f"key path must be str or tuple of str, not {type(path).__name__}"
        )
        raise TypeError(msg)
    if not path:
        msg = "key path must not be empty"
        raise TOMLError(msg)
    parts = path.split(".")
    if any(not p for p in parts):
        msg = f"key path {path!r} contains an empty segment"
        raise TOMLError(msg)
    return tuple(parts)


def _section_insert_index(
    sections: list[SectionNode],
    full_path: tuple[str, ...],
) -> int:
    """Choose where to splice a new section with ``full_path``.

    Prefers placement immediately after the last section that shares
    ``full_path``'s parent prefix (so siblings group together); falls
    back to the end of the document.
    """
    parent = full_path[:-1]
    # Top-level sections always go to the end: every existing section
    # trivially shares the empty parent prefix, so the last-sibling
    # search would just yield ``len(sections) - 1``. Skip it.
    if not parent:
        return len(sections)
    last_sibling = -1
    for i, sec in enumerate(sections):
        hdr = sec.header
        if hdr is None:
            continue
        hpath = hdr.key.path
        if len(hpath) >= len(parent) and hpath[: len(parent)] == parent:
            last_sibling = i
    if last_sibling < 0:
        return len(sections)
    return last_sibling + 1


def _make_dotted_key(path: tuple[str, ...]) -> Key:
    parts = [make_key_part(p) for p in path]
    seps = ["."] * (len(parts) - 1)
    return Key(parts=parts, separators=seps)
