.PHONY: install install-all format lint typecheck test coverage quality verify smoke experiments experiments-full api freeze clean

install:
	uv sync

install-all:
	uv sync --all-extras

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

verify:
	uv run python scripts/verify_source.py

quality: verify
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src
	uv run pytest

smoke:
	uv run rees46 run-experiments --profile dev --no-sequence --no-mlflow

experiments:
	uv run rees46 run-experiments --profile dev

experiments-full:
	uv run rees46 run-experiments --profile full

api:
	uv run uvicorn rees46.serving.api:app --host 0.0.0.0 --port 8000

freeze:
	uv run rees46 freeze-results --profile full

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage mlruns
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
