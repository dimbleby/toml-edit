# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Array.set_multiline(*, multiline, indent="    ")` and the
  read/write `Array.multiline` property toggle an inline array
  between single-line and multi-line layout.
- `Table.set_aot(key, entries=())` creates an array-of-tables at
  ``key`` (overwriting any existing value) and returns the live view,
  so users can build `[[ ... ]]` sections without going through the
  inline-array path.
- `Table.set_table(key, value=())` creates a standard-table section
  at ``key``, replacing any existing value. Accepts dotted paths
  (e.g. `"tool.poetry"`); intermediate tables are kept implicit so
  no empty `[tool]` super-table headers are emitted.
- `Table.ensure_table(key)` returns the table at ``key``, creating
  an empty section if absent. Accepts dotted paths and walks through
  implicit super-tables.
- `Table.set_array(key, items=(), *, multiline=False, indent="    ")`
  creates an inline array at ``key`` (replacing any existing value),
  optionally laid out one item per line. Accepts dotted paths so a
  multiline array deep in the tree can be created in a single call.
- `Table.set_aot` now accepts dotted paths, mirroring `set_table`.
- `Table.table`, `Table.array` and `Table.aot` typed accessors now
  accept dotted paths for navigation through nested structures.
- `Table.promote_array(key)` converts an existing inline array of
  inline tables into an array-of-tables, mirroring the existing
  `Table.promote_inline` for tables.

### Fixed

- An empty array whose source contains a newline inside the brackets
  (`a = [\n]`) now round-trips and accepts subsequent `append` calls
  while preserving its multi-line shape.
- `Table.set_aot` and `Table.promote_array` now lay their `[[ ... ]]`
  blocks out with blank-line separators between entries, and with a
  blank line between the block and any preceding content.

## [0.1.0] - 2026-04-20

Initial release.

[Unreleased]: https://github.com/dimbleby/tomlrt/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dimbleby/tomlrt/releases/tag/v0.1.0
