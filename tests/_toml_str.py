"""Helper for writing TOML fixtures inline as multi-line strings.

Tests use lots of small TOML documents as input and as expected output.
Written as ordinary string literals these turn into walls of escaped
newlines like ``"[a]\\nx = 1\\n[a.sub]\\ny = 2\\n"``.

``td`` lets the same fixture be written as an indented triple-quoted
literal::

    src = td('''
        [a]
        x = 1
        [a.sub]
        y = 2
    ''')

A single leading newline (the one immediately after the opening
``\"\"\"``) is stripped so the first content line starts at column 0,
then :func:`textwrap.dedent` removes the common leading whitespace.
The result is byte-identical to ``"[a]\\nx = 1\\n[a.sub]\\ny = 2\\n"``,
which matters because everything in this project is round-tripped
byte-for-byte.
"""

from __future__ import annotations

from textwrap import dedent


def td(src: str) -> str:
    """Dedent ``src`` and strip a single leading newline."""
    return dedent(src).removeprefix("\n")
