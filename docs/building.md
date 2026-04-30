# Building documents

## From scratch

```python
from tomlrt import Document, dumps

doc = Document()
doc["title"] = "example"
print(dumps(doc))
# title = "example"
```

## From a `dict`

Pass a mapping to populate the document recursively:

```python
from tomlrt import Document, dumps

data = {
    "project": {
        "name": "demo",
        "version": "0.1.0",
        "dependencies": ["requests>=2"],
    },
    "tool": {"ruff": {"line-length": 88}},
}

doc = Document(data)
print(dumps(doc))
```

```toml
[project]
name = "demo"
version = "0.1.0"
dependencies = ["requests>=2"]

[tool.ruff]
line-length = 88
```

Nested mappings become `[section]` blocks (not inline tables); lists of mappings
become `[[array.of.tables]]` blocks; everything else is an ordinary key-value
assignment.

## Dotted paths

`doc["a.b"] = 1` always treats `"a.b"` as a _single literal key_.
To descend into nested tables, pass a dotted string (split on `.`) or a tuple of
literal segments to `install` or `ensure_table`:

```python
doc.install("tool.poetry.version", "0.1.0")  # [tool.poetry] version = "..."
doc.install(("tool", "weird.key"), 1)        # [tool] "weird.key" = 1

ruff = doc.ensure_table("tool.ruff")         # creates [tool.ruff] if absent
ruff["line-length"] = 88
```

Both create any missing intermediate tables on the way down.
They differ at the leaf:

- `install(path, value)` writes `value` at `path`, replacing whatever was there.
  It returns the freshly-installed live view (or the leaf value).
- `ensure_table(path)` is idempotent: it returns the existing table at `path` if
  there is one, or creates an empty one if not.
  It never overwrites an existing value.
