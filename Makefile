# Convenience targets for tomlrt. The source of truth is still the
# commands documented in .github/copilot-instructions.md and the CI
# workflows; this Makefile just gathers them in one place.

UV ?= uv

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Common targets:"
	@echo "  make test         # run the test suite (excludes slow/fuzz)"
	@echo "  make fuzz         # run the slow property-based suite"
	@echo "  make coverage     # tests + branch coverage"
	@echo "  make lint         # ruff check + mypy --strict"
	@echo "  make docs         # build the MkDocs site (strict)"
	@echo "  make docs-serve   # preview the docs locally"
	@echo "  make clean        # remove caches and build artefacts"

.PHONY: test
test:
	pytest -q

.PHONY: fuzz
fuzz:
	pytest -q -m slow

.PHONY: coverage
coverage:
	pytest --cov

.PHONY: lint
lint: ruff mypy

.PHONY: ruff
ruff:
	ruff check .

.PHONY: mypy
mypy:
	mypy

.PHONY: docs
docs:
	$(UV) run --group docs mkdocs build --strict

.PHONY: docs-serve
docs-serve:
	$(UV) run --group docs mkdocs serve

.PHONY: clean
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov site dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
