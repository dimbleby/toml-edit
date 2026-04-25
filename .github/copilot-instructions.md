# Instructions for AI coding agents

This file is read by GitHub Copilot, Copilot CLI, and similar agents
when they work in this repository. Humans are welcome to read it too —
it doubles as a high-signal contributor guide.

## What this project is

`tomlrt` is a **pure-Python, format-preserving** TOML parser and
writer. The non-negotiable invariant is:

> Parsing a document and dumping it again, with no mutations in
> between, must return the **exact same bytes** — including comments,
> whitespace, string style (literal vs basic, single vs multiline),
> number formatting, and line endings (LF vs CRLF).

If a change you are about to make could break that, stop and rethink.

## Toolchain

- **`uv`** is the only supported package/dependency manager. Do not
  introduce `pip`, `poetry`, `pipenv`, `tox`, `nox`, `setuptools`, or
  `requirements.txt`.
- Build backend is **`hatchling`**.
- Supported Python versions: **3.10 – 3.14**.

## Common commands

```bash
uv sync                          # install dev deps
uv run pytest -q                 # run the test suite (~10s)
uv run pytest --cov              # tests + branch coverage
uv run mypy                      # strict type-check src/ and tests/
uv run ruff check .              # lint
uv run ruff format .             # apply formatting
uv run ruff format --check .     # CI-style format check
```

All four checks (`pytest`, `mypy`, `ruff check`, `ruff format --check`)
must pass before any commit. CI runs the same set on Python 3.10–3.14.

## Coding standards

- **`mypy --strict`** clean — no `# type: ignore` without a specific
  error code, and ideally no ignores at all. Prefer fixing the type.
- **`ruff` with `select = ["ALL"]`** clean — see `[tool.ruff.lint.ignore]`
  in `pyproject.toml` for the curated exceptions. Do not add new
  per-line `# noqa` without a strong reason.
- **`ruff format`** is the source of truth for formatting. Run it
  before committing.
- **No runtime dependencies beyond conditional stdlib backports.**
  The only declared runtime dep is `typing_extensions` on
  Python < 3.12 (for `Self` / `override`), behind a `python_version`
  marker. Don't add others. `dependency-groups.dev` and
  `dependency-groups.docs` may grow, but only with care.
- **No `cast()` in user-facing code paths.** Tests should not need
  `cast()` either; the typed accessors `Table.array(k)`,
  `Table.table(k)`, `Table.aot(k)`, `Array.array(i)`, `Array.table(i)`
  exist precisely to avoid this.
- **`from __future__ import annotations`** at the top of every module
  (enforced by ruff's isort `required-imports`).
- Do not add comments that merely restate the code. Comment intent and
  invariants, not mechanics.

## Architecture (in `src/tomlrt/`)

The codebase is small and deliberately layered. Read in this order:

- **`_nodes.py`** — the **concrete syntax tree (CST)**: dataclasses
  that hold every byte of the original source (including trivia,
  comments, and the literal lexeme of every value). Mutate these only
  through helpers that maintain the round-trip invariant.
- **`_parser.py`** — hand-written recursive-descent parser, TOML 1.0 +
  1.1, that produces the CST. Performance-sensitive: prefer bulk
  `str` scans over per-character loops.
- **`_synthesise.py`** — converts plain Python values (`str`, `int`,
  `bool`, `datetime`, `list`, `dict`, …) into newly synthesised CST
  nodes when the user assigns into the document.
- **`_trivia.py`** — pure helpers over `Trivia` and `TriviaPiece`
  sequences: comment formatting, EOL-comment handling, scanning and
  rewriting leading/trailing-comment blocks, line-ending detection.
  Depends only on `_nodes` and `_errors`.
- **`_separator.py`** — comma-separator style sampling and
  re-application for inline arrays and inline tables
  (`_SeparatorStyle`, snapshot/restore of per-item leading comments).
  Depends on `_trivia` only.
- **`_section_build.py`** — section construction, deep-clone-with-
  rebase, and splice helpers used when assigning whole tables / AoTs
  into a document. Reaches into `_document` privates by design (it is
  the inverse of "give me the CST that backs this view").
- **`_comment_views.py`** — the `MutableMapping` / `MutableSequence`
  views returned by `Table.comments` / `.leading_comments` and the
  `Array` equivalents. `_PresenceFilteredView` is an `abc.ABC` with
  `@abstractmethod` hooks; the four concrete subclasses are imported
  by `_document.py` and constructed from the relevant properties.
- **`_document.py`** — the **logical view layer**: `Document`, `Table`,
  `Array`, `AoT` wrappers that present a dict/list-shaped API while
  delegating all mutation to the CST. The Comment API and typed
  accessors live here. This is by far the largest file.
- **`_public.py`** — the top-level `parse` / `loads` / `load` /
  `dumps` / `dump` functions. `load` and `dump` require **binary**
  file objects (`IO[bytes]`); text mode would silently translate
  newlines on Windows and break round-tripping.
- **`_errors.py`** — public exception hierarchy.
- **`__init__.py`** — re-exports the public API; keep `__all__`
  alphabetised.

When in doubt: a change that touches only one of these layers is
usually right; a change that has to touch all of them is usually wrong.

## Tests

- `tests/test_basic.py`, `test_spacing.py`, `test_edit_golden.py` —
  parser and writer regressions, including byte-exact round-trip
  fixtures.
- `tests/test_comments.py` — the comment manipulation API.
- `tests/test_compliance.py` — the official **`toml-test`** suite
  (vendored under `vendor/`). Do not edit fixtures there to make
  failures pass.
- `tests/test_dict_semantics.py` — pins the user-visible behaviours
  that come from `Table` actually being a `dict` subclass
  (`isinstance`, ``**t`` unpacking, identity stability of lookups).
- `tests/test_toml11.py` — TOML 1.1-specific coverage.
- `tests/test_hypothesis.py` — property-based round-trip tests. If you
  break round-tripping, this will usually catch it; add new strategies
  here when you add a new construct.
- `tests/test_mutation.py` — the dict/list mutation API.
- `tests/test_live_attach.py` — live-attach semantics for
  `Table.inline`, `Array`, and `AoT` when assigned into a document.
- `tests/test_synthesise_and_io.py` — value synthesis and binary I/O.

When adding behaviour, add a focused unit test in the relevant file
**and** consider whether the property tests should grow.

## Documentation

User-facing prose docs live under `docs/` and are published as a
MkDocs site at <https://dimbleby.github.io/tomlrt/>. The dependency
group is `docs`:

```bash
uv run --group docs mkdocs serve              # preview locally
uv run --group docs mkdocs build --strict     # what CI runs
```

The API reference page (`docs/api.md`) is generated from docstrings
via `mkdocstrings`, so docstring changes flow through automatically.
The task-oriented pages (`quickstart.md`, `building.md`, `editing.md`,
`access.md`, `comments.md`, `errors.md`) are hand-written — update
them when you add, rename, or change behaviour of any public API.

## Things to avoid

- Adding an unconditional runtime dependency.
- Reaching into `_nodes` from user-facing code instead of going through
  `_document`.
- "Fixing" formatting differences in the writer's output without
  adding a round-trip test that proves it.
- Touching `vendor/` (it is third-party, vendored verbatim).
- Editing `uv.lock` by hand — let `uv` regenerate it.
- Bumping action versions in `.github/workflows/*.yml` to a tag instead
  of a 40-char commit SHA. The workflows are **`zizmor` clean** and
  must stay that way (`uv tool run zizmor .`).

## Commit conventions

- Subject line: imperative mood, ≤ ~70 chars, no trailing period.
- Body: wrap around 72 chars; explain *why* not *what*. Bullets are
  fine.
- One logical change per commit. Keep mechanical reformat passes
  separate from substantive changes.
- Append a `Co-authored-by` trailer when an AI agent did the work, e.g.
  `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`.
