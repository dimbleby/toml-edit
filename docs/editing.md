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

Assigning a fresh `Table.inline(...)`, `Array(...)`, or `AoT(...)` *attaches it live*: the user's reference becomes the live view at the destination, and later mutations through that reference show up in the document.

```python
xs = Array([1, 2])
doc["xs"] = xs
xs.append(3)             # doc["xs"] is now [1, 2, 3]
assert doc["xs"] is xs
```

Plain `dict` / `list` values are *snapshot* on assignment — mutating the original after assignment does *not* affect the document.
Reach for `Table.inline` or `Array` when you want live semantics.

A container that is already attached somewhere is deep-cloned on assignment, so two slots never share the same underlying CST.
This applies to intra-document (`doc["b"] = doc["a"]`) and cross-document (`d2["x"] = d1["x"]`) cases alike.

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
