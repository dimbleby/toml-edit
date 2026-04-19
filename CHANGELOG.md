# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - Initial release

### Added

- Hand-written recursive-descent parser supporting **TOML 1.0.0** and
  **TOML 1.1.0** (unicode bare keys, trailing commas in inline tables,
  multiline inline tables, `\xHH` and `\e` string escapes, optional
  seconds in time/datetime values).
- Format-preserving writer with **byte-exact round-trip** for unmodified
  input.
- Dict-like read API on `Document` and `Table`; real `list` semantics on
  `Array`; nested tables and arrays-of-tables are transparently
  navigable.
- Mutation API:
  - replace / insert / delete on scalar keys
  - full `list` mutator set on `Array`
  - insert / replace / delete on inline tables
  - create new `[sub.tables]` and `[[arrays.of.tables]]` via assignment
- **Comment manipulation API** for keys, headers, and array elements
  via `Table.comments`, `Table.leading_comments`, `Table.header_comment`,
  `Table.header_leading_comments`, `Array.comments`, and
  `Array.leading_comments`.
- Typed accessors `Table.array(key)`, `Table.table(key)`,
  `Table.aot(key)`, `Array.array(i)`, `Array.table(i)` so callers don't
  need `cast()`.
- Top-level API: `parse`, `loads`, `load`, `dumps`, `dump`. `load` and
  `dump` require binary file objects (matching `tomllib`); this also
  guarantees newline preservation across platforms.
- Cross-section conflict detection in the parser (duplicate keys,
  redefined tables, AoT-vs-table collisions).
- Friendly error messages with line and column numbers via
  `TOMLParseError`.
- Pure Python, fully type-annotated (`mypy --strict` clean), `py.typed`
  marker, zero runtime dependencies, supports CPython and PyPy
  3.10–3.13.
- Hypothesis-based round-trip tests; full coverage of the official
  `toml-test` compliance suite.

[Unreleased]: https://github.com/dch/toml-edit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dch/toml-edit/releases/tag/v0.1.0
