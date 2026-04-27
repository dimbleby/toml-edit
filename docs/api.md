# API reference

The complete public surface, generated from the docstrings in the source.

## Stability

Anything imported from the top-level `tomlrt` namespace is part of the public, semver-stable API:

| Symbol                                  | Kind      |
| --------------------------------------- | --------- |
| `loads`, `parse`, `load`                | function  |
| `dumps`, `dump`                         | function  |
| `document`                              | function  |
| `Document`, `Table`, `Array`, `AoT`     | class     |
| `TomlInput`                             | type alias|
| `TOMLError`, `TOMLParseError`           | exception |

Anything not re-exported from `tomlrt/__init__.py` (modules prefixed with `_`, internal helpers) may change without notice and should not be imported by user code.

## Top-level functions

::: tomlrt.loads
::: tomlrt.parse
::: tomlrt.load
::: tomlrt.dumps
::: tomlrt.dump
::: tomlrt.document

## Containers

::: tomlrt.Document
::: tomlrt.Table
::: tomlrt.Array
::: tomlrt.AoT

## Type aliases

::: tomlrt.TomlInput

## Errors

::: tomlrt.TOMLError
::: tomlrt.TOMLParseError
