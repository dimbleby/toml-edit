"""Hypothesis round-trip tests: dumps(parse(s)) == s for many shapes."""

from __future__ import annotations

import string

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import toml_edit

# ---------------------------------------------------------------------------
# Strategies for safe TOML fragments (we generate values whose canonical
# rendering by the parser is byte-stable, so we can assert exact round-trip).
# ---------------------------------------------------------------------------

_BARE_KEY = st.from_regex(r"\A[A-Za-z][A-Za-z0-9_-]{0,15}\Z")

_BASIC_STR_CHARS = st.text(
    alphabet=st.characters(
        min_codepoint=0x20,
        max_codepoint=0x7E,
        blacklist_characters='"\\',
    ),
    max_size=20,
)


def _quoted(s: str) -> str:
    return '"' + s + '"'


_STRINGS = _BASIC_STR_CHARS.map(_quoted)
_INTS = st.integers(min_value=-(2**31), max_value=2**31 - 1).map(str)
_BOOLS = st.sampled_from(["true", "false"])
_FLOATS = st.sampled_from(["0.0", "1.5", "-3.25", "1e10", "-2.5e-3", "inf", "-inf"])
_SCALARS = st.one_of(_STRINGS, _INTS, _BOOLS, _FLOATS)


@st.composite
def _kv_lines(draw: st.DrawFn, keys: list[str] | None = None) -> str:
    if keys is None:
        keys = draw(st.lists(_BARE_KEY, max_size=4, unique=True))
    out: list[str] = []
    for k in keys:
        v = draw(_SCALARS)
        out.append(f"{k} = {v}")
    return "\n".join(out) + ("\n" if out else "")


@st.composite
def _array_value(draw: st.DrawFn) -> str:
    elems = draw(st.lists(_SCALARS, max_size=5))
    if not elems:
        return "[]"
    return "[ " + ", ".join(elems) + " ]"


@st.composite
def _section(draw: st.DrawFn) -> str:
    parts = draw(
        st.lists(_BARE_KEY, min_size=1, max_size=3, unique=True),
    )
    header = "[" + ".".join(parts) + "]\n"
    body = draw(_kv_lines())
    return header + body


@st.composite
def _document(draw: st.DrawFn) -> str:
    # Reserve a pool of names; partition them between pre-section keys
    # and section first-name components so a key like ``a = 1`` never
    # collides with a later ``[a]`` (which is invalid TOML).
    pool = draw(st.lists(_BARE_KEY, min_size=0, max_size=8, unique=True))
    cut = draw(st.integers(min_value=0, max_value=len(pool)))
    pre_keys = pool[:cut]
    section_roots = pool[cut:]

    pre = draw(_kv_lines(keys=pre_keys))
    # Build unique section paths from the available roots; each root
    # gets a unique single- or two-part path.
    sec_paths: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for root in section_roots:
        depth = draw(st.integers(min_value=1, max_value=2))
        if depth == 1:
            path: tuple[str, ...] = (root,)
        else:
            sub = draw(_BARE_KEY)
            path = (root, sub)
        if path in seen:
            continue
        seen.add(path)
        sec_paths.append(path)

    parts: list[str] = []
    if pre:
        parts.append(pre)
    for path in sec_paths:
        header = "[" + ".".join(path) + "]\n"
        # KVs inside a section must not collide with the section's own
        # first-part name reserved at root level.
        body_keys = draw(
            st.lists(
                _BARE_KEY.filter(lambda k: k not in section_roots),
                max_size=4,
                unique=True,
            ),
        )
        body = draw(_kv_lines(keys=body_keys))
        parts.append(header + body)
    return "".join(parts) or "\n"


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(src=_document())
def test_roundtrip_exact(src: str) -> None:
    """Parsing then dumping must reproduce the original source byte-for-byte."""
    doc = toml_edit.parse(src)
    assert toml_edit.dumps(doc) == src


@settings(max_examples=100, deadline=None)
@given(src=_document())
def test_roundtrip_idempotent(src: str) -> None:
    """A second parse/dump pass produces the same output as the first."""
    doc1 = toml_edit.parse(src)
    out1 = toml_edit.dumps(doc1)
    doc2 = toml_edit.parse(out1)
    out2 = toml_edit.dumps(doc2)
    assert out1 == out2


# Specific edge-case corpora that aren't easily generated.
_EDGE_CASES = [
    "",
    "\n",
    "key = 1",  # no trailing newline
    "key = 1\n",
    "# only a comment\n",
    "a = 1\n\n[b]\nc = 2\n",
    "x = [1, 2, 3]\n",
    "x = [\n  1,\n  2,\n]\n",
    "obj = { a = 1, b = 2 }\n",
    "[[items]]\nname = 'x'\n[[items]]\nname = 'y'\n",
    "[a.b.c]\nv = 1\n",
    "a.b.c = 1\n",
    f"strs = {list(string.ascii_letters[:5])}\n".replace("'", '"'),
]


@given(src=st.sampled_from(_EDGE_CASES))
@settings(max_examples=len(_EDGE_CASES), database=None)
def test_edge_cases_roundtrip(src: str) -> None:
    assert toml_edit.dumps(toml_edit.parse(src)) == src
