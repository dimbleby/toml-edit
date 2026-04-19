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
- **TOML 1.0.0** today; **1.1.0** support is on the roadmap.

## Performance

`toml-edit` is a hand-written recursive-descent parser; on a typical
`pyproject.toml` it parses **~8× faster than `tomlkit`** and round-trips
**~7× faster** while still preserving every byte of formatting:

```
Workload                              toml_edit (us)   tomlkit (us)    speedup
------------------------------------------------------------------------
pyproject.toml      parse                 1041.3         8456.6      8.12x
pyproject.toml      parse+dump            1275.4         8598.1      6.74x
```

Run `python benches/bench_vs_tomlkit.py` to reproduce on your machine.

## Install

```bash
pip install toml-edit
```

## Usage

```python
import toml_edit

with open("pyproject.toml") as f:
    doc = toml_edit.load(f)

doc["project"]["version"] = "0.2.0"
doc["project"]["dependencies"].append("requests>=2")

print(toml_edit.dumps(doc))   # comments and layout are preserved
```

## Status

Implemented:

- TOML 1.0.0 parser (every value type, dotted keys, AoT, inline tables, …)
- Byte-exact round-trip writer
- Dict-like read API on `Document` / `Table`
- Mutation API: replace/insert/delete scalars; full `list` mutator set on
  `Array`; insert/replace/delete on inline tables
- Strict `mypy` and `ruff ALL` clean
- Hypothesis-based round-trip tests

Roadmap:

- Comment manipulation API
- Creating new `[sub.tables]` / `[[arrays.of.tables]]` via assignment
- TOML 1.1.0 deltas (unicode bare keys, trailing commas in inline tables, …)
- The official `toml-test` compliance suite

## License

MIT
