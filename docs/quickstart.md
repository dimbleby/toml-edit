# Quickstart

## Parse, edit, dump

```python
import tomlrt

with open("pyproject.toml", "rb") as f:
    doc = tomlrt.load(f)

doc["project"]["version"] = "0.2.0"
doc["project"]["dependencies"].append("requests>=2")

with open("pyproject.toml", "wb") as f:
    tomlrt.dump(doc, f)
```

`tomlrt.parse` and `tomlrt.loads` are equivalent.

## Binary mode is required

`load` and `dump` require **binary** file objects (`open(path, "rb")` or `"wb"`).
Text mode would perform locale-dependent decoding and platform newline translation, which would break the byte-exact round-trip guarantee.
A text stream raises `TypeError`.

For string round-trips, use `parse` / `dumps`:

```python
doc = tomlrt.parse(text)
text_again = tomlrt.dumps(doc)
assert text == text_again        # if you didn't mutate
```

## Reading values

A `Document` behaves like a `dict`; nested tables are `Table` (also a `dict` subclass), inline arrays are `Array` (a `list` subclass), and arrays-of-tables are `AoT` (a `list` of `Table`).
Plain reads with `doc["key"]` work as you'd expect — see [Typed access](access.md) when you want `mypy`-friendly traversal.

## Writing values

Plain Python values do the right thing on assignment:

| Assigning           | Becomes                                       |
| ------------------- | --------------------------------------------- |
| `str`/`int`/`bool`/`float`/`datetime` | a TOML scalar                |
| `dict`              | an inline table (snapshot)                    |
| `list`              | an inline array (snapshot)                    |
| `Table.section({})` | a live `[section]` block                      |
| `Table.inline({})`  | a live inline table                           |
| `AoT([...])`        | `[[array.of.tables]]` blocks                  |
| `Array([...], multiline=True)` | a multi-line inline array          |

For the difference between snapshot and live containers, see [Editing documents](editing.md#live-vs-snapshot).
