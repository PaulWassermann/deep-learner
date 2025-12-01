.PHONY: publish
publish:
	@echo Not implemented

.PHONY: build
build:
	@echo Not implemented

.PHONY: check
check:
	@uv run pre-commit run --all-files

.PHONY: format
format:
	@uv run ruff format

.PHONY: install
install:
	@uv sync --no-dev

.PHONY: install-dev
install-dev:
	@uv sync

.PHONY: quality-check
quality-check:
	@uv run ruff check examples src tests

# TODO: add a report file for coverage
.PHONY: test
test:
	@uv run pytest tests

.PHONY: coverage
coverage:
	@echo Not implemented

.PHONY: type-check
type-check:
	@uv run ty check
