# toml-edit

A fast, ergonomic, format-preserving TOML parser and writer for Python.

> **Status:** alpha. API may change.

## Why?

The Python ecosystem already has:

- `tomllib` / `tomli` — read-only, fast, but discards comments/formatting.
- `tomlkit` — preserves formatting, but is slow and its wrapper-object API
  surprises callers who expect plain dict semantics.

`toml-edit` aims for the best of both:

- **Format-preserving** round-trips (whitespace, comments, string style,
  number formatting).
- **Transparent dict-like API** — `doc["pkg"]["name"]` returns a plain
  `str`; `doc["deps"]` returns a real `list`; nested tables are still
  navigable.
- **Pure Python**, fully type-annotated (`mypy --strict`), no native
  build step.
- **TOML 1.0.0 + 1.1.0** compliant.

## Install

```bash
pip install toml-edit
```

## Usage

```python
import tomle

with open("pyproject.toml") as f:
    doc = tomle.load(f)

doc["project"]["version"] = "0.2.0"
doc["project"]["dependencies"].append("requests>=2")

print(tomle.dumps(doc))   # comments and layout are preserved
```

## License

MIT
