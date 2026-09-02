.PHONY: install format lint typecheck test coverage quality clean

install:
	uv sync

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src

test:
	uv run pytest

coverage:
	uv run pytest --cov=rees46 --cov-report=term-missing

quality:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src
	uv run pytest

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
