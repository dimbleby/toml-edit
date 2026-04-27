# Typed access

Plain `doc["key"]` returns `Any`.
The typed accessors give you a shape-checked `Table`, `Array`, or `AoT` directly.

## Strict accessors

Each raises `KeyError` if the key is missing and `TypeError` if the value at the key has the wrong shape:

```python
project = doc.table("project")            # -> Table
deps    = project.array("dependencies")   # -> Array
pkgs    = doc.aot("packages")             # -> AoT
```

The same accessors exist on `Array` for indexed lookup:

```python
first_pkg = pkgs.table(0)             # -> Table
nested    = some_array.array(2)       # -> Array
```

## Lenient accessors

`get_table` / `get_array` / `get_aot` mirror `dict.get`: return the value when the shape matches, otherwise the `default` (default `None`).
They raise `TypeError` only if the key exists but has the _wrong_ shape:

```python
ruff = doc.get_table("tool", default={}).get("ruff")
```

## Back to plain Python

`Table.to_dict()` and `Array.to_list()` (and `AoT.to_list()`) return deep copies that share no state with the document — useful for handing data to consumers that don't know about tomlrt:

```python
plain = doc.to_dict()
import json; print(json.dumps(plain))
```
