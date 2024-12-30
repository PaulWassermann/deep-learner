.PHONY: check
check:
	@poetry run pre-commit run --all-files

.PHONY: format
format:
	@poetry run black .

.PHONY: install
install:
	@poetry install --without dev

.PHONY: install-dev
install-dev:
	@poetry install

.PHONY: quality-check
quality-check:
	@poetry run pylint src

.PHONY: sort-imports
sort-imports:
	@poetry run isort src

# TODO: add a report file for coverage
.PHONY: test
test:
	@poetry run pytest tests

.PHONY: coverage
coverage:
	@echo Not implemented

.PHONY: type-check
type-check:
	@poetry run mypy src
