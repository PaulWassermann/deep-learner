# .PHONY: publish
# build:
# 	@poetry publish --build

.PHONY: check
check:
	@uv run pre-commit run --all-files

.PHONY: format
format:
	@uv run black .

.PHONY: install
install:
	@uv sync --no-dev
.PHONY: install-dev
install-dev:
	@uv sync

.PHONY: quality-check
quality-check:
	@uv run pylint src

.PHONY: sort-imports
sort-imports:
	@uv run isort src

# TODO: add a report file for coverage
.PHONY: test
test:
	@uv run pytest tests

.PHONY: coverage
coverage:
	@echo Not implemented

.PHONY: type-check
type-check:
	@uv run mypy src
