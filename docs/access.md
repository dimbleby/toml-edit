# Typed access

`Document` and `Table` are `dict` subclasses, so `doc["key"]` returns `Any` — fine at runtime, awkward under `mypy --strict`.
The typed accessors avoid `cast()` while preserving the live-view semantics.

## Required accessors

Each raises `TOMLError` if the value at the key is missing or has the wrong shape:

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

## Optional accessors

`get_table` / `get_array` / `get_aot` mirror `dict.get`: return the value when the shape matches, otherwise the `default` (default `None`).
They raise `TOMLError` only if the key exists but has the _wrong_ shape:

```python
ruff = doc.get_table("tool", default={}).get("ruff")
```

## Back to plain Python

`Table.to_dict()` and `Array.to_list()` (and `AoT.to_list()`) return deep copies that share no state with the document — useful for handing data to consumers that don't know about tomlrt:

```python
plain = doc.to_dict()
import json; print(json.dumps(plain))
```
