"""Hypothesis round-trip tests: dumps(parse(s)) == s for many shapes."""

from __future__ import annotations

import math
import string
import sys
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import tomlrt
from _toml_str import td

pytestmark = pytest.mark.slow

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def _deep_equal(a: object, b: object) -> bool:
    """Structural equality that treats NaN as equal to itself."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_deep_equal(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b) and type(a) is type(b)


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
    doc = tomlrt.parse(src)
    assert tomlrt.dumps(doc) == src


# Specific edge-case corpora that aren't easily generated.
_EDGE_CASES = [
    "",
    "\n",
    "key = 1",  # no trailing newline
    "key = 1\n",
    "# only a comment\n",
    td("""
        a = 1

        [b]
        c = 2
        """),
    "x = [1, 2, 3]\n",
    td("""
        x = [
          1,
          2,
        ]
        """),
    "obj = { a = 1, b = 2 }\n",
    td("""
        [[items]]
        name = 'x'
        [[items]]
        name = 'y'
        """),
    "[a.b.c]\nv = 1\n",
    "a.b.c = 1\n",
    f"strs = {list(string.ascii_letters[:5])}\n".replace("'", '"'),
]


@given(src=st.sampled_from(_EDGE_CASES))
@settings(max_examples=len(_EDGE_CASES), database=None)
def test_edge_cases_roundtrip(src: str) -> None:
    assert tomlrt.dumps(tomlrt.parse(src)) == src


# ---------------------------------------------------------------------------
# Comment-view round-trip: writing back what we read must be a no-op, and
# the rendered comment must read back as the value we wrote. These caught
# the "user already supplied #" branch, the empty-string-as-delete shortcut,
# and the rstrip in the marker-stripper.
# ---------------------------------------------------------------------------

_COMMENT_TEXT = st.text(
    alphabet=st.characters(
        blacklist_categories=["Cs"],
        blacklist_characters="\n\r"
        + "".join(chr(c) for c in range(0x20) if c != 0x09)
        + "\x7f",
    ),
    max_size=30,
)


@given(text=_COMMENT_TEXT)
@settings(max_examples=200, database=None)
def test_eol_comment_roundtrip(text: str) -> None:
    doc = tomlrt.parse("a = 1\n")
    doc.comments["a"] = text
    out = tomlrt.dumps(doc)
    assert tomlrt.parse(out).comments["a"] == text


@given(lines=st.lists(_COMMENT_TEXT, min_size=1, max_size=4))
@settings(max_examples=200, database=None)
def test_leading_comment_roundtrip(lines: list[str]) -> None:
    doc = tomlrt.parse("a = 1\n")
    doc.leading_comments["a"] = tuple(lines)
    out = tomlrt.dumps(doc)
    assert tomlrt.parse(out).leading_comments["a"] == tuple(lines)


@given(text=_COMMENT_TEXT)
@settings(max_examples=200, database=None)
def test_array_eol_comment_roundtrip(text: str) -> None:
    doc = tomlrt.parse("a = [1, 2]\n")
    doc.array("a").comments[0] = text
    out = tomlrt.dumps(doc)
    assert tomlrt.parse(out).array("a").comments[0] == text


@given(text=_COMMENT_TEXT)
@settings(max_examples=200, database=None)
def test_header_comment_roundtrip(text: str) -> None:
    doc = tomlrt.parse("[s]\nx = 1\n")
    doc.table("s").header_comment = text
    out = tomlrt.dumps(doc)
    assert tomlrt.parse(out).table("s").header_comment == text


@given(lines=st.lists(_COMMENT_TEXT, min_size=1, max_size=4))
@settings(max_examples=200, database=None)
def test_header_leading_comment_roundtrip(lines: list[str]) -> None:
    doc = tomlrt.parse("[s]\nx = 1\n")
    doc.table("s").header_leading_comments = tuple(lines)
    out = tomlrt.dumps(doc)
    assert tomlrt.parse(out).table("s").header_leading_comments == tuple(lines)


@given(lines=st.lists(_COMMENT_TEXT, min_size=1, max_size=4))
@settings(max_examples=200, database=None)
def test_array_leading_comment_roundtrip(lines: list[str]) -> None:
    doc = tomlrt.parse("a = [1, 2]\n")
    doc.array("a").leading_comments[1] = tuple(lines)
    out = tomlrt.dumps(doc)
    assert tomlrt.parse(out).array("a").leading_comments[1] == tuple(lines)


@given(lines=st.lists(_COMMENT_TEXT, min_size=1, max_size=4))
@settings(max_examples=200, database=None)
def test_preamble_roundtrip(lines: list[str]) -> None:
    doc = tomlrt.parse("a = 1\n")
    doc.preamble = tuple(lines)
    out = tomlrt.dumps(doc)
    assert tomlrt.parse(out).preamble == tuple(lines)


@given(lines=st.lists(_COMMENT_TEXT, min_size=1, max_size=4))
@settings(max_examples=200, database=None)
def test_epilogue_roundtrip(lines: list[str]) -> None:
    doc = tomlrt.parse("a = 1\n")
    doc.epilogue = tuple(lines)
    out = tomlrt.dumps(doc)
    assert tomlrt.parse(out).epilogue == tuple(lines)


@given(text=_COMMENT_TEXT.filter(bool))
@settings(max_examples=200, database=None)
def test_eol_comment_set_then_clear(text: str) -> None:
    """Setting then deleting an EOL comment must restore a comment-free dump."""
    base = "a = 1\n"
    doc = tomlrt.parse(base)
    doc.comments["a"] = text
    del doc.comments["a"]
    assert tomlrt.dumps(doc) == base


# ---------------------------------------------------------------------------
# Semantic cross-check: anything our parser accepts must agree with stdlib
# tomllib on the decoded data. Catches "bytes round-trip but model is wrong"
# bugs that the exact-round-trip property alone cannot see.
# ---------------------------------------------------------------------------


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(src=_document())
def test_semantic_match_tomllib(src: str) -> None:
    ours = tomlrt.parse(src).to_dict()
    theirs = tomllib.loads(src)
    assert _deep_equal(ours, theirs)


@given(src=st.sampled_from(_EDGE_CASES))
@settings(max_examples=len(_EDGE_CASES), database=None)
def test_edge_cases_match_tomllib(src: str) -> None:
    ours = tomlrt.parse(src).to_dict()
    theirs = tomllib.loads(src)
    assert _deep_equal(ours, theirs)


# ---------------------------------------------------------------------------
# CRLF preservation: the round-trip invariant explicitly covers line endings.
# Take a generated source and randomly map each '\n' to '\n' or '\r\n'.
# ---------------------------------------------------------------------------


@st.composite
def _crlf_variant(draw: st.DrawFn) -> str:
    src = draw(_document())
    n = src.count("\n")
    flips = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    out: list[str] = []
    i = 0
    for ch in src:
        if ch == "\n":
            out.append("\r\n" if flips[i] else "\n")
            i += 1
        else:
            out.append(ch)
    return "".join(out)


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(src=_crlf_variant())
def test_crlf_roundtrip_exact(src: str) -> None:
    assert tomlrt.dumps(tomlrt.parse(src)) == src


# ---------------------------------------------------------------------------
# Synthesis round-trip: build a Document from a plain Python tree, dump it,
# parse it back, and compare the recovered data to the original.
# ---------------------------------------------------------------------------

_PY_SCALARS: st.SearchStrategy[Any] = st.one_of(
    _BASIC_STR_CHARS,
    st.integers(min_value=-(2**31), max_value=2**31 - 1),
    st.booleans(),
    st.floats(allow_nan=False, allow_infinity=False, width=64),
)


def _py_dict(max_depth: int) -> st.SearchStrategy[dict[str, Any]]:
    if max_depth <= 0:
        values: st.SearchStrategy[Any] = _PY_SCALARS
    else:
        values = st.one_of(
            _PY_SCALARS,
            st.lists(_PY_SCALARS, max_size=4),
            _py_dict(max_depth - 1),
        )
    return st.dictionaries(_BARE_KEY, values, max_size=4)


@settings(max_examples=200, deadline=None)
@given(data=_py_dict(max_depth=2))
def test_synthesise_roundtrip(data: dict[str, Any]) -> None:
    doc = tomlrt.document(data)
    out = tomlrt.dumps(doc)
    recovered = tomlrt.parse(out).to_dict()
    assert _deep_equal(recovered, data)


# ---------------------------------------------------------------------------
# Mutation round-trip: assigning new scalar values to existing top-level
# scalar slots must reflect both in to_dict() and after a dump/parse cycle.
# ---------------------------------------------------------------------------


def _python_scalar(literal: str) -> Any:
    """Decode a TOML scalar literal (as our generator emits it) to Python."""
    return tomllib.loads(f"_x = {literal}")["_x"]


@st.composite
def _document_with_overrides(draw: st.DrawFn) -> tuple[str, dict[str, Any]]:
    src = draw(_document())
    parsed = tomllib.loads(src)
    top_keys = [k for k, v in parsed.items() if not isinstance(v, (dict, list))]
    overrides: dict[str, Any] = {}
    for k in top_keys:
        if draw(st.booleans()):
            overrides[k] = _python_scalar(draw(_SCALARS))
    return src, overrides


@settings(max_examples=200, deadline=None)
@given(case=_document_with_overrides())
def test_scalar_mutation_roundtrip(case: tuple[str, dict[str, Any]]) -> None:
    src, overrides = case
    doc = tomlrt.parse(src)
    expected = tomllib.loads(src)
    for k, v in overrides.items():
        doc[k] = v
        expected[k] = v
    assert _deep_equal(doc.to_dict(), expected)
    assert _deep_equal(tomlrt.parse(tomlrt.dumps(doc)).to_dict(), expected)
