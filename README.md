# toml-edit

A fast, ergonomic, format-preserving TOML parser and writer for Python.

> **Status:** beta (0.1.x). The public API is stable in shape but may see
> minor refinements before 1.0.

## Why?

The standard library's `tomllib` (and its backport `tomli`) is fast but
read-only — round-tripping through it discards comments, whitespace,
string style, and number formatting. That makes it unsuitable for any
tool that wants to *edit* a TOML file in place: linters, formatters,
package managers bumping a dependency, codemods, and so on.

`toml-edit` fills that gap:

- **Format-preserving** round-trips (whitespace, comments, string style,
  number formatting) — byte-exact for unmodified input.
- **Transparent dict-like API** — `doc["pkg"]["name"]` returns a plain
  `str`; `doc["deps"]` returns a real `list`; nested tables are still
  navigable.
- **Comment API** — read, write, and clear EOL and leading comments on
  keys, headers, and array elements without parsing the source by hand.
- **Pure Python**, fully type-annotated (`mypy --strict`, `py.typed`),
  no native build step, zero runtime dependencies.
- **TOML 1.0.0** and **TOML 1.1.0** supported.

## Install

```bash
pip install toml-edit
```

## Usage

```python
import toml_edit

# Files must be opened in binary mode (TOML is UTF-8; binary mode also
# preserves line endings byte-for-byte across platforms).
with open("pyproject.toml", "rb") as f:
    doc = toml_edit.load(f)

doc["project"]["version"] = "0.2.0"
doc["project"]["dependencies"].append("requests>=2")

print(toml_edit.dumps(doc))   # comments and layout are preserved
```

### Comment API

```python
doc = toml_edit.loads("""
[server]
host = "localhost"  # default
port = 8080
""")

server = doc.table("server")
server.comments["port"] = "override with $PORT"
server.comments["host"] = None         # clear

print(toml_edit.dumps(doc))
# [server]
# host = "localhost"
# port = 8080 # override with $PORT
```

`Table.comments`, `Table.leading_comments`, `Array.comments`, and
`Array.leading_comments` all behave as `MutableMapping[str, str | None]`
or the array-indexed equivalent, so editors and round-trip tools can
treat comments as ordinary structured data.

## Status

Implemented:

- TOML 1.0.0 and 1.1.0 parser (every value type, dotted keys, AoT,
  inline tables, unicode bare keys, trailing commas, multiline inline
  tables, `\xHH` / `\e` escapes, optional seconds in time/datetime).
- Byte-exact round-trip writer.
- Dict-like read API on `Document` / `Table`; real-list semantics on
  `Array`.
- Mutation API: replace/insert/delete scalars; full `list` mutator set
  on `Array`; insert/replace/delete on inline tables; create new
  `[sub.tables]` and `[[arrays.of.tables]]` via assignment.
- Comment manipulation API for keys, headers, and array elements.
- Typed accessors (`Table.array(k)`, `Table.table(k)`, `Table.aot(k)`,
  `Array.array(i)`, `Array.table(i)`) so callers don't need `cast()`.
- Strict `mypy --strict` and `ruff ALL` clean.
- Hypothesis-based round-trip tests; the official `toml-test`
  compliance suite.

## License

MIT
