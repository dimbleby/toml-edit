# Errors

tomlrt raises a small, focused exception hierarchy.

## `TOMLError`

Base class for everything tomlrt raises.
Catch this when you want to treat any tomlrt-originated failure uniformly:

```python
try:
    doc.table("project").promote_inline("authors")
except tomlrt.TOMLError as exc:
    log.warning("could not promote: %s", exc)
```

## `TOMLParseError`

Raised by `loads` / `parse` / `load` when the input isn't valid TOML.
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

`TOMLParseError` is a subclass of `TOMLError`, so catching the base class is
enough when you don't care to distinguish parse failures from edit-time errors.
