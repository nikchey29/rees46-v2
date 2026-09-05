#!/usr/bin/env python3
"""Fast source-only integrity checks that do not require the REES46 dataset."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    "pyproject.toml",
    "README.md",
    "Makefile",
    "configs/base.yaml",
    "configs/modeling.yaml",
    "src/rees46/cli.py",
    "src/rees46/experiments/runner.py",
    "src/rees46/recommendations/covisitation/model.py",
    "src/rees46/recommendations/collaborative/model.py",
    "src/rees46/recommendations/ranking/model.py",
    "src/rees46/recommendations/sequential.py",
    "src/rees46/serving/api.py",
    "sql/ddl/001_serving_schema.sql",
    "Dockerfile",
    "docker-compose.yml",
    ".github/workflows/ci.yml",
)


def main() -> int:
    missing = [path for path in REQUIRED if not (ROOT / path).exists()]
    if missing:
        print("FAIL required files")
        print("\n".join(missing))
        return 1
    print("PASS required files")

    failures: list[str] = []
    python_files = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py"))
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            failures.append(f"{path}: {exc}")
    if failures:
        print("FAIL Python syntax")
        print("\n".join(failures))
        return 1
    print(f"PASS Python syntax ({len(python_files)} files)")

    data_foundation = ROOT / "results" / "data_foundation.json"
    if data_foundation.exists():
        payload = json.loads(data_foundation.read_text(encoding="utf-8"))
        assert payload["bronze_rows"] == 411_709_736
        assert payload["silver_rows"] == 410_325_314
        assert payload["exact_duplicates_removed"] == 1_384_422
        print("PASS measured data-foundation record")

    print("PASS source package is internally consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
