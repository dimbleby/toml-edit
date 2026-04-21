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

A plain `dict` value installs as an inline table; a plain `list`
installs as an inline array. To pick a different shape, assign a
flavoured value:

```python
from tomlrt import AoT, Array, Table

doc["tool"] = Table.section({"version": 1})      # [tool] section
doc["xy"]   = {"x": 1, "y": 2}                   # xy = { x = 1, y = 2 }
doc["pkgs"] = AoT([{"a": 1}, {"b": 2])           # [[pkgs]] … [[pkgs]]
doc["tags"] = Array(["a", "b"], multiline=True)  # multi-line array
```

Use `doc.install(path, value)` for dotted-path placement, or pass a
tuple to escape keys that contain a literal `.`:

```python
doc.install("tool.poetry", Table.section({"name": "x"}))  # [tool.poetry]
doc.install(("weird.key",), 1)                            # "weird.key" = 1
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
