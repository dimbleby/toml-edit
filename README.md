# tomlrt

[![PyPI](https://img.shields.io/pypi/v/tomlrt.svg)](https://pypi.org/project/tomlrt/)
[![Python versions](https://img.shields.io/pypi/pyversions/tomlrt.svg)](https://pypi.org/project/tomlrt/)
[![License](https://img.shields.io/pypi/l/tomlrt.svg)](https://github.com/dimbleby/tomlrt/blob/main/LICENSE)
[![CI](https://github.com/dimbleby/tomlrt/actions/workflows/ci.yml/badge.svg)](https://github.com/dimbleby/tomlrt/actions/workflows/ci.yml)

A format-preserving TOML parser and writer for Python.

## Usage

```python
import tomlrt

# Files must be opened in binary mode.
with open("pyproject.toml", "rb") as f:
    doc = tomlrt.load(f)

doc["project"]["version"] = "0.2.0"
doc["project"]["dependencies"].append("requests>=2")

print(tomlrt.dumps(doc))   # comments and layout are preserved
```

### Structural assignment

A plain `dict` value installs as an inline table; a plain `list` installs as an
inline array.
To pick a different shape, assign a flavoured value:

```python
from tomlrt import AoT, Array, Table

doc["tool"] = Table.section({"version": 1})      # [tool] section
doc["xy"]   = Table.inline({"x": 1, "y": 2})     # xy = { x = 1, y = 2 }
doc["pkgs"] = AoT([{"a": 1}, {"b": 2}])          # [[pkgs]] … [[pkgs]]
doc["tags"] = Array(["a", "b"], multiline=True)  # multi-line array
```

### Live vs snapshot containers

Assigning a fresh `Table.inline(...)`, `Array(...)`, or `AoT(...)` *attaches it
live*: the user's reference becomes the live view at the destination, and
later mutations through that reference show up in the document.

```python
xs = Array([1, 2])
doc["xs"] = xs
xs.append(3)             # doc["xs"] is now [1, 2, 3]
assert doc["xs"] is xs
```

Plain `dict` / `list` values are *snapshot* on assignment — mutating the
original after assignment does *not* affect the document. Reach for
`Table.inline` or `Array` when you want live semantics.

A container that is already attached somewhere is deep-cloned on assignment,
so two slots never share the same underlying CST. This applies to
intra-document (`doc["b"] = doc["a"]`) and cross-document
(`d2["x"] = d1["x"]`) cases alike.

Use `doc.install(path, value)` for dotted-path placement.
Plain `doc["a.b"] = …` always treats `"a.b"` as a _single literal key_, so
`install` is the way to descend through `a` into `b`.
Pass a tuple when one of the segments itself contains a literal `.`:

```python
doc.install("tool.poetry.version", "0.1.0")            # [tool.poetry] version = "0.1.0"
doc.install(("tool", "weird.key"), 1)                  # [tool] "weird.key" = 1
```

### Comment API

```python
doc = tomlrt.loads("""
[server]
host = "localhost"  # default
port = 8080
""")

server = doc.table("server")
server.comments["port"] = "override with $PORT"
server.comments["host"] = None         # clear

print(tomlrt.dumps(doc))
# [server]
# host = "localhost"
# port = 8080 # override with $PORT
```
