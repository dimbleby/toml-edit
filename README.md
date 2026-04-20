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

`Table` is a real `dict[str, Any]` subclass and `Array` is a real
`list[Any]` subclass, so `isinstance(t, dict)`, `**t`, `json.dumps(t)`
and any other API typed against `dict` / `list` accept tomlrt
containers directly — no snapshot dance.

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
