# Editing documents

## Structural assignment

A plain `dict` value installs as an inline table; a plain `list` installs as an inline array.
To pick a different shape, assign a flavoured value:

```python
from tomlrt import AoT, Array, Table

doc["tool"] = Table.section({"version": 1})      # [tool] section
doc["xy"]   = Table.inline({"x": 1, "y": 2})     # xy = { x = 1, y = 2 }
doc["pkgs"] = AoT([{"a": 1}, {"b": 2}])          # [[pkgs]] … [[pkgs]]
doc["tags"] = Array(["a", "b"], multiline=True)  # multi-line array
```

## Live vs snapshot

Assigning a fresh `Table.section(...)`, `Table.inline(...)`, `Array(...)`, or `AoT(...)` _attaches it live_: your reference becomes the live view at the destination, and later mutations through that reference show up in the document.

```python
xs = Array([1, 2])
doc["xs"] = xs
xs.append(3)             # doc["xs"] is now [1, 2, 3]
assert doc["xs"] is xs

t = Table.section()
doc["a"] = t
t["x"] = 1               # doc["a"] is now {"x": 1}
assert doc["a"] is t
```

Plain `dict` / `list` values are _snapshot_ on assignment — mutating the original after assignment does _not_ affect the document.
Reach for `Table.section`, `Table.inline`, or `Array` when you want live semantics.

A container that is already attached somewhere is deep-cloned on assignment, so two slots never share state.
This applies whether the source and destination are in the same document (`doc["b"] = doc["a"]`) or different ones (`d2["x"] = d1["x"]`).

## Growing an array-of-tables

`AoT.add()` appends a fresh entry and returns the new `Table` view, so you can keep mutating it:

```python
pkgs = doc.aot("packages")
entry = pkgs.add({"name": "foo"})
entry["version"] = "1.0"
```

## Inline-array layout

`Array.multiline` flips between single- and multi-line layout in place.
For multi-line layout with a custom indent, call `set_multiline`:

```python
arr = doc.array("tags")
arr.set_multiline(multiline=True, indent="  ")
```

Collapsing a multi-line array to single-line is rejected if any item carries a comment — clear those first via `arr.comments` / `arr.leading_comments` (see [Comments](comments.md)).

## Promoting inline → section

If a value started life as an inline table or inline array of inline tables, you can promote it in place:

```python
doc["tool"] = {"ruff": {"line-length": 88}}    # inline tables
doc.table("tool").promote_inline("ruff")       # → [tool.ruff]

doc["pkgs"] = [{"a": 1}, {"b": 2}]             # inline array of tables
doc.promote_array("pkgs")                      # → [[pkgs]] … [[pkgs]]
```

Promotion is rejected if it would lose inner comments; clear them first.
