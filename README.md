# tomlrt

[![PyPI](https://img.shields.io/pypi/v/tomlrt.svg)](https://pypi.org/project/tomlrt/)
[![Python versions](https://img.shields.io/pypi/pyversions/tomlrt.svg)](https://pypi.org/project/tomlrt/)
[![License](https://img.shields.io/pypi/l/tomlrt.svg)](https://github.com/dimbleby/tomlrt/blob/main/LICENSE)
[![CI](https://github.com/dimbleby/tomlrt/actions/workflows/ci.yml/badge.svg)](https://github.com/dimbleby/tomlrt/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-zensical-informational)](https://dimbleby.github.io/tomlrt/)

A format-preserving TOML reader and writer for Python.

Parse a document, edit it, dump it, and the bytes you didn't touch round-trip exactly — comments, whitespace, string style, and number formatting all intact.

```python
import tomlrt

with open("pyproject.toml", "rb") as f:
    doc = tomlrt.load(f)

doc["project"]["version"] = "0.2.0"
doc["project"]["dependencies"].append("requests>=2")

print(tomlrt.dumps(doc))   # comments and layout are preserved
```

Build a document from scratch:

```python
import tomlrt

doc = tomlrt.document({"project": {"name": "demo", "version": "0.1.0"}})
print(tomlrt.dumps(doc))
```

## Documentation

Full docs at <https://dimbleby.github.io/tomlrt/>:

- [Quickstart](https://dimbleby.github.io/tomlrt/quickstart/)
- [Building documents](https://dimbleby.github.io/tomlrt/building/)
- [Editing documents](https://dimbleby.github.io/tomlrt/editing/)
- [Typed access](https://dimbleby.github.io/tomlrt/access/)
- [Comments](https://dimbleby.github.io/tomlrt/comments/)
- [API reference](https://dimbleby.github.io/tomlrt/api/)

See also the [changelog](CHANGELOG.md).
