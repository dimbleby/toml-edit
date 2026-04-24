# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- CST node dataclasses now declare `eq=False`, so `==` / `in` / `list.index`
  / `list.remove` / set membership all fall back to identity comparison.
  This is what every internal call site already wanted; the previous
  default of structural equality was a recurring footgun behind several
  past fixes (commits `3dbdb4c`, `227d1bc`, `bdb7ea2`). Two distinct
  CST nodes that happen to render the same are no longer ever conflated.
  As a consequence the bespoke identity-keyed helpers (`_index_of`,
  `remove_sections_by_id`, `remove_entry_by_id`) collapse back into
  ordinary `list.index` / `list.remove` and a thin chokepoint that pairs
  the structural change with `normalise_top_blank()`. There is no
  user-visible behavioural change — the public `Array.remove` and
  `AoT.remove` accept Python values and still match by value.

### Fixed

- Replacing a section in place via `doc[k] = Table.section({...})` (or the
  equivalent `Table.aot([...])` / `AoT(...)` assignment) no longer drops
  the leading comment block that sat above the original `[k]` header.
  Since `5527097` the slot was reused, but the prior header's leading
  trivia was discarded — comments above the section vanished silently.
  `_prepare_section_slot` now snapshots that trivia before purging and
  the install paths transplant it onto the first new section's header.

- Installing a detached `AoT` no longer drops per-entry formatting such as
  multi-line arrays.
- Installing a sub-section under one AoT entry no longer silently deletes a
  sibling entry's same-named sub-section.
- Cross-document assignment of a section-backed `Table` (e.g.
  `dest[k] = src[k]`) now deep-clones the source's CST, so comments and
  formatting survive and any nested array-of-tables is emitted as `[[..]]`
  instead of crashing the inline-table synthesiser.
- Self-overlapping assignment such as `doc[k] = doc[k]["child"]` now lifts
  the child to a `[k]` block, instead of either crashing (when the child
  contains an array-of-tables) or silently flattening to an inline table.
- Sequential cross-document section copies no longer produce doubled blank
  lines between sections.
- Deleting the first section in a document no longer leaves a stray blank
  line at the top of the rendered output.
- Removing the first entry of an AoT (`del aot[0]`, `aot.pop(0)`,
  `aot.remove(...)`, slice deletion) no longer leaves a stray blank line at
  the top of the rendered output. The section-removal helper now lives on
  `DocumentNode`, so identity-keyed removal and top-of-file normalisation
  are wired together at a single chokepoint instead of being open-coded
  per call site.
- Deleting the first top-level key (`del doc[k]` / `doc.pop(k)`) no longer
  leaves a stray blank line at the top of the rendered output. Same shape
  as the AoT-deletion bug; the entry-removal path now goes through a
  matching `DocumentNode.remove_entry` chokepoint that runs the
  top-of-file normalisation afterwards.
- `aot *= n` on a single-entry AoT no longer glues the duplicated
  ``[[t]]`` headers together: with no second entry to sample as an
  inter-entry separator, the repeat path now falls back to a
  blank-line separator (canonical TOML style) rather than empty trivia.
  The shared separator trivia is also deep-copied per repetition so
  later mutations on one duplicate don't bleed into the others.
- `Document.install(path, ...)` now rejects up-front, with a clear
  `TOMLError`, when ``path`` would have to thread through an
  array-of-tables. Previously this raised a bare `AssertionError`
  *after* partially mutating the document, leaving it inconsistent.
- Assigning `Table.section({})` and then a child section (e.g.
  `doc[k] = Table.section({}); doc[k][c] = ...`) no longer leaves an
  empty `[k]` header above the child.
- Assigning a sub-section into a non-last AoT entry (e.g.
  `aot[0]["x"] = Table.section({...})`) now lands inside that entry's
  range instead of being appended after every later entry, which
  previously caused silent re-attribution on round-trip.
- Deleting a key whose value is an in-cache table view that has no
  remaining CST footprint (e.g. after emptying its only descendant)
  no longer spuriously raises `KeyError`.
- Assigning the same `Table.section(...)` sub-key on multiple AoT entries
  (e.g. `aot[i]["source"] = Table.section({...})` in a loop) now keeps
  each entry's values separate instead of leaking writes into the first
  matching `[aot.k]` section and corrupting earlier entries.
- Installing a sub-section under an AoT entry whose `[[..]]` header is
  byte-identical to a sibling's no longer splices the new block into
  the wrong entry's range. The insert-index lookup now uses identity
  rather than equality.
- `doc[k] |= other` (and any other `doc[k] = doc[k]` self-assignment)
  no longer detaches and re-clones the existing block, which had the
  side effect of moving `[k]` to the end of its siblings and dropping
  surrounding blank-line trivia.
- Replacing a section with `Table.section({...})` (or any other
  flavoured-section install at a key that already names one) now
  reuses the existing section's slot among its siblings, instead of
  appending the new block at the end of the parent's range. Applies
  inside AoT entries too.

## [0.4.0] - 2026-04-23

### Fixed

- A broad sweep of correctness fixes in the mutation API, covering every flavour
  of structural change: assignment into and through array-of-tables,
  append/insert/pop on multi-line arrays with comments, attached-AoT
  installation, comment-trivia preservation across promotion and shifts, CRLF
  line-ending preservation, and copy/deepcopy of `Array` and `AoT`
  subviews.
  Several silent corruptions (CST and dict-side state diverging after a
  mutation) are gone, and a number of error messages are now more specific about
  which value was rejected and why.

## [0.3.0] - 2026-04-21

### Changed

- **Structural assignment is now driven by the value, not the method name.**
  The parallel `set_table` / `set_aot` / `set_array` methods have been removed
  in favour of a single assignment path:

  ```python
  doc[k] = Table.section({...})          # [k] standard section
  doc[k] = {...}                         # k = { ... } inline table
  doc[k] = AoT([{...}, {...}])           # [[k]] array of tables
  doc[k] = Array([...], multiline=True)  # multi-line array value
  ```

  `Table.section` is a classmethod factory returning the public tag type
  :class:`SectionSpec`.
  :class:`AoT` and :class:`Array` can now be constructed standalone and then
  assigned.

- **New `Table.install(path, value)`** accepts either a dotted `str` path or a
  `tuple[str, ...]` of literal segments.
  Tuples provide an escape for keys that legitimately contain a `.`::

        doc.install(("foo.bar",), 1)   # "foo.bar" = 1  (single segment)
        doc.install("foo.bar", 1)      # [foo]\nbar = 1 (dotted path)

  `ensure_table` also accepts both forms.

- `__setitem__` no longer splits `str` keys on `.`; a plain `str` is always
  treated as a single literal segment, matching the standard `dict` contract.
  Use `install()` for dotted-path placement.

### Removed

- `Table.set_table`, `Table.set_aot`, `Table.set_array`.
  Use the value-driven equivalents above, or `Table.install` for dotted paths /
  tuple keys.

### Fixed

- `AoT.insert(0, …)` now adds a blank-line separator between the newly inserted
  `[[..]]` entry and the existing one that follows it (matching sibling spacing,
  defaulting to blank-separated).
  The policy previously only looked at _preceding_ content, so inserting before
  existing entries glued two `[[..]]` headers together.
- The dict-style view of a parsed :class:`Document` no longer goes stale
  relative to :func:`dumps` after structural mutations.
  Assigning over an array-of-tables, deleting then re-binding a key, and `pop()`
  followed by re-assignment all kept showing the pre-mutation value while the
  rendered TOML reflected the new state.
  The cached per-table section scope that drove this has been replaced with
  on-demand derivation from the surrounding AoT entry (when there is one), so dict
  reads and `dumps` output are always consistent.
- Mutations on a sub-table reached via a dotted key from an ancestor section now
  work correctly.
  Given `poetry.name = "x"` written inside `[tool]`,
  `doc["tool"]["poetry"].pop("name")` and `doc["tool"]["poetry"]["name"] = "y"`
  previously raised `KeyError` or duplicated the key in a new section; both now
  edit the original entry in place.
- Setting :attr:`Document.preamble` on an empty document and then adding content
  now renders the preamble at the top of the file.
  It was previously parked in the document's trailing trivia and emitted _after_
  the new content (so `dumps` produced `x = 1\n# c\n` instead of `# c\n\nx =
1\n`); the comment also became invisible to the getter once content arrived.
  Migration now happens at the insertion site for any of `doc[k] = …`,
  :meth:`Table.install`, :meth:`AoT.insert`, or AoT assignment.
- :meth:`Table.promote_array` now carries the source inline-table KV's leading
  comments / blank lines onto the first new `[[..]]` header, and any trailing
  EOL comment onto the last new entry.
  The trivia was previously discarded outright, so promoting an inline array
  silently dropped any authoring comments around it.
- Import of `assert_never` no longer breaks on Python 3.10.
  The symbol is now sourced from `typing_extensions` on interpreters older than
  3.11, mirroring the existing `override` import.

## [0.2.0] - 2026-04-20

### Changed

- **`Table` is now a real `dict` subclass.** `isinstance(t, dict)` returns
  `True`, `**table` unpacking works, and any third-party API typed against
  `dict[str, Any]` / `isinstance(x, dict)` now accepts a `Table` directly.
  Reads go through `dict`'s native `__iter__` / `__getitem__` / `__len__` /
  `__contains__`; the CST is still the single source of truth for _layout_
  (whitespace, comments, key order, table-shape choices) and is kept in lock-step
  with the dict storage on every mutation.
  Held references behave like ordinary Python dict references: `del doc['foo']`
  orphans the held `Table` (data preserved, mutations no longer reach the
  document) and re-binding the path installs a fresh `Table` rather than
  re-attaching the old one.
  Identity is stable: `doc['foo'] is doc['foo']` and the same goes for nested
  children.
- `Table.pop` now returns the actual stored value (an orphaned `Table` / `AoT` /
  `Array` for container values) rather than a deep plain-Python snapshot.
  Use `Table.to_dict()` / `Array.to_list()` first if you need a snapshot.
- Detached tables and AoTs are now isolated from the original document.
  Structural mutations on a held container after its parent removed it
  (`set_table`, `set_aot`, `promote_inline`, `promote_array`, `AoT.add`,
  `AoT.append`, `AoT.insert` …) no longer leak back into the document by
  re-creating the removed sections.
- `AoT.pop` now returns the live entry object that was at the given index (then
  orphans it), mirroring `Table.pop` and preserving identity with whatever the
  caller previously read out of the AoT.
- `Table` now subclasses `MutableMapping[str, Any]` (was `MutableMapping[str,
TomlValue]`), and `Table.__getitem__` returns `Any` (was the strict `Scalar |
Array | AoT | Table` union).
  Symmetrically, `Array` now subclasses `list[Any]` and `Array.__getitem__` /
  `Array.pop` return `Any`.
  This matches what `tomllib.loads` returns (`dict[str, Any]`) and what `tomlkit`
  does, and lets chained subscripts like `doc["tool"]["poetry"]["name"]`
  type-check without `cast`.
  Consumers typed against `MutableMapping[str, Any]` or `list[Any]` (which is most
  of the ecosystem) now compose with `Table` / `Array` directly.
  The strict return type is still available through the `.table()` / `.array()` /
  `.aot()` accessors and their `get_*` counterparts when you want it.
- `Array.append` / `extend` / `insert` / `__setitem__` now type their input
  parameter as `object` instead of the narrower `TomlValue` alias, matching
  `Table.__setitem__` and the underlying `value_to_node` converter.
  At runtime they always accepted arbitrary Python values (plain `dict` -> inline
  table, plain `list` -> inline array); the annotations were lying.
- Synthesised inline arrays no longer carry padding spaces inside the brackets.
  `[1, 2, 3]` instead of `[ 1, 2, 3 ]`, and `[1]` instead of `[ 1 ]`.
  Inter-element spaces are unchanged.
  Inline tables (`{ a = 1, b = 2 }`) still keep their conventional inner spacing.
  Parsed arrays round-trip with their original spacing.
- Modest parse speedup: cache `Key.path` so the dotted-key tuple is built once
  per key, and pass the parent's already-scoped section list through to child
  `_StdTable` constructors so each child's initial population walks only its own
  subtree instead of the whole document.

### Added

- `Table.get_table(key, default=None)`, `Table.get_array(...)`,
  `Table.get_aot(...)` and the analogous `Array.get_table(index, ...)` /
  `Array.get_array(index, ...)` are typed-but-optional accessors.
  They mirror the strict `.table()` / `.array()` / `.aot()` accessors but return
  `default` (or `None`) when the key/index is missing, rather than raising.
  A wrong-type entry still raises :class:`TypeError`: missing is "no answer",
  wrong shape is a bug.
  Overloads preserve the type of a user-supplied default.
- `Table.to_dict()` / `Array.to_list()` / `AoT.to_list()` return a deep,
  plain-Python copy of the view, walking nested tomlrt views into real `dict` /
  `list` containers.
  Intended for the interop boundary with consumers that expect actual `dict`
  objects (`fastjsonschema`, `pydantic`, JSON encoders, code that does
  `isinstance(x, dict)`).
  Scalars are returned as-is; the result shares no mutable state with the
  document.
- `AoT.add(entry={})` appends `entry` and returns the new :class:`Table` view,
  sparing users the `aot.append(...); aot[-1]` two-step when they need a handle
  to the freshly-added entry for further population.
- `tomlrt.document(data=None)` returns a fresh :class:`Document`, optionally
  populated from a mapping.
  Without arguments, equivalent to `tomlrt.parse("")` but more discoverable for
  the "build a TOML file from scratch" use case.
  With a mapping, recursively walks the data: nested mappings become `[section]`
  blocks, lists of mappings become `[[array.of.tables]]` blocks, and leaf values
  use ordinary key-value assignment.
  The resulting document shares no mutable state with the input.
- `Array.set_multiline(*, multiline, indent="    ")` and the read/write
  `Array.multiline` property toggle an inline array between single-line and
  multi-line layout.
- `Table.set_aot(key, entries=())` creates an array-of-tables at `key`
  (overwriting any existing value) and returns the live view, so users can build
  `[[ ...
]]` sections without going through the inline-array path.
- `Table.set_table(key, value=())` creates a standard-table section at `key`,
  replacing any existing value.
  Accepts dotted paths (e.g.
  `"tool.poetry"`); intermediate tables are kept implicit so no empty `[tool]`
  super-table headers are emitted.
- `Table.ensure_table(key)` returns the table at `key`, creating an empty
  section if absent.
  Accepts dotted paths and walks through implicit super-tables.
- `Table.set_array(key, items=(), *, multiline=False, indent="    ")` creates an
  inline array at `key` (replacing any existing value), optionally laid out one
  item per line.
  Accepts dotted paths so a multiline array deep in the tree can be created in a
  single call.
- `Document.preamble` and `Document.epilogue` properties expose the comment
  block at the top and bottom of the document.
  They are blank-line-separated from any structural content (and from any
  "attached" leading comment of the first key), so writing one will not clobber
  the other or any per-key comment block.
- `Table.set_aot` now accepts dotted paths, mirroring `set_table`.
- `Table.table`, `Table.array` and `Table.aot` typed accessors now accept dotted
  paths for navigation through nested structures.
- `Table.promote_array(key)` converts an existing inline array of inline tables
  into an array-of-tables, mirroring the existing `Table.promote_inline` for
  tables.

### Fixed

- An empty array whose source contains a newline inside the brackets (`a =
[\n]`) now round-trips and accepts subsequent `append` calls while preserving
  its multi-line shape.
- `Table.set_aot` and `Table.promote_array` now lay their `[[ ...
]]` blocks out with blank-line separators between entries, and with a blank line
  between the block and any preceding content.
- Programmatically appending to an `AoT` (or appending the second entry into a
  freshly-built one) now blank-line-separates the new `[[ ...
]]` header from whatever precedes it in the document, matching round-trip output
  of equivalent parsed input.
  Previously, fresh AoTs and AoTs whose new entries followed an unrelated
  sub-section were rendered with the headers visually glued together.
  When existing entries clearly establish a no-blank-line style (≥ 2 sibling gaps
  to learn from), that style is still respected.

## [0.1.0] - 2026-04-20

Initial release.

[Unreleased]: https://github.com/dimbleby/tomlrt/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dimbleby/tomlrt/releases/tag/v0.1.0
