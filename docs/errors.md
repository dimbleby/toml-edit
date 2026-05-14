# Errors

tomlrt raises a small exception hierarchy.

## `TOMLError`

Base class for everything tomlrt raises.

```python
try:
    doc.table("project").promote_inline("authors")
except tomlrt.TOMLError as exc:
    log.warning("could not promote: %s", exc)
```

## `TOMLParseError`

Raised by `loads` / `load` when the input isn't valid TOML.
Carries useful position information:

```python
try:
    tomlrt.loads("a = ?")
except tomlrt.TOMLParseError as exc:
    print(exc.line, exc.col, exc.offset)
```

| Attribute | Meaning                             |
| --------- | ----------------------------------- |
| `line`    | 1-based line number                 |
| `col`     | 1-based column number               |
| `offset`  | 0-based byte offset into the source |
