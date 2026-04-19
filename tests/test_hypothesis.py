"""Hypothesis round-trip tests: dumps(parse(s)) == s for many shapes."""

from __future__ import annotations

import string

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import tomle

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
def _kv_lines(draw: st.DrawFn) -> str:
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
    pre = draw(_kv_lines())
    sec_paths = draw(
        st.lists(
            st.lists(_BARE_KEY, min_size=1, max_size=2, unique=True).map(tuple),
            max_size=3,
            unique=True,
        ),
    )
    parts: list[str] = []
    if pre:
        parts.append(pre)
    for path in sec_paths:
        header = "[" + ".".join(path) + "]\n"
        body = draw(_kv_lines())
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
    doc = tomle.parse(src)
    assert tomle.dumps(doc) == src


@settings(max_examples=100, deadline=None)
@given(src=_document())
def test_roundtrip_idempotent(src: str) -> None:
    """A second parse/dump pass produces the same output as the first."""
    doc1 = tomle.parse(src)
    out1 = tomle.dumps(doc1)
    doc2 = tomle.parse(out1)
    out2 = tomle.dumps(doc2)
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
    assert tomle.dumps(tomle.parse(src)) == src
