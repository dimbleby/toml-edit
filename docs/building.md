# Building documents

## From scratch

```python
import tomlrt

doc = tomlrt.document()
doc["title"] = "example"
print(tomlrt.dumps(doc))
# title = "example"
```

## From a `dict`

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

## Dotted paths

`doc["a.b"] = 1` always treats `"a.b"` as a _single literal key_.
Use `install` to descend through dotted segments, or `ensure_table` when you just want the intermediate table created on demand:

```python
doc.install("tool.poetry.version", "0.1.0")  # [tool.poetry] version = "..."
doc.install(("tool", "weird.key"), 1)        # [tool] "weird.key" = 1

ruff = doc.ensure_table("tool.ruff")         # creates [tool.ruff] if absent
ruff["line-length"] = 88
```

`install` replaces whatever is at the path; `ensure_table` is idempotent and only creates missing tables.
