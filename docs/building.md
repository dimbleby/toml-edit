# Building documents

## Empty document

```python
import tomlrt

doc = tomlrt.document()
doc["title"] = "example"
print(tomlrt.dumps(doc))
# title = "example"
```

`tomlrt.document()` is the discoverable way to build a TOML file from scratch.
It is equivalent to `tomlrt.parse("")` but signals intent.

## From a plain `dict`

Pass a mapping to populate the document recursively:

```python
data = {
    "project": {
        "name": "demo",
        "version": "0.1.0",
        "dependencies": ["requests>=2"],
    },
    "tool": {"ruff": {"line-length": 88}},
}

doc = tomlrt.document(data)
print(tomlrt.dumps(doc))
```

```toml
[project]
name = "demo"
version = "0.1.0"
dependencies = ["requests>=2"]

[tool.ruff]
line-length = 88
```

Nested mappings become `[section]` blocks (not inline tables); lists of mappings become `[[array.of.tables]]` blocks; everything else is an ordinary key-value assignment.

## Dotted-path placement

`doc["a.b"] = 1` always treats `"a.b"` as a *single literal key*.
Use `install` to descend through dotted segments, or `ensure_table` when you just want the intermediate table created on demand:

```python
doc.install("tool.poetry.version", "0.1.0")  # [tool.poetry] version = "..."
doc.install(("tool", "weird.key"), 1)        # [tool] "weird.key" = 1

ruff = doc.ensure_table("tool.ruff")          # creates [tool.ruff] if absent
ruff["line-length"] = 88
```

`install` replaces whatever is at the path; `ensure_table` is idempotent and only creates missing tables.
