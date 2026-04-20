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
- `Table.promote_array(key)` converts an existing inline array of
  inline tables into an array-of-tables, mirroring the existing
  `Table.promote_inline` for tables.

### Fixed

- An empty array whose source contains a newline inside the brackets
  (`a = [\n]`) now round-trips and accepts subsequent `append` calls
  while preserving its multi-line shape.

## [0.1.0] - 2026-04-20

Initial release.

[Unreleased]: https://github.com/dimbleby/tomlrt/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dimbleby/tomlrt/releases/tag/v0.1.0
