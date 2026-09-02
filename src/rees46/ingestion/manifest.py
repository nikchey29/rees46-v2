"""Raw dataset provenance and integrity manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rees46.runtime.config import AppConfig

EXPECTED_FILES = (
    "2019-Oct.csv.gz",
    "2019-Nov.csv.gz",
    "2019-Dec.csv.gz",
    "2020-Jan.csv.gz",
    "2020-Feb.csv.gz",
    "2020-Mar.csv.gz",
    "2020-Apr.csv.gz",
)

SOURCE_BASE_URL = "https://data.rees46.com/datasets/marketplace"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Calculate a SHA-256 digest without loading a file into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def build_manifest(config: AppConfig) -> dict[str, Any]:
    """Build the immutable raw-data provenance manifest."""

    files: list[dict[str, Any]] = []

    for name in EXPECTED_FILES:
        path = config.paths.raw / name

        if not path.exists():
            raise FileNotFoundError(f"Missing REES46 raw archive: {path}")

        files.append(
            {
                "name": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "source": f"{SOURCE_BASE_URL}/{name}",
            }
        )

    return {
        "dataset": "REES46 multi-category marketplace behaviour",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "file_count": len(files),
        "files": files,
    }


def write_manifest(config: AppConfig) -> Path:
    """Write raw dataset provenance metadata."""

    manifest = build_manifest(config)
    target = config.paths.raw / "manifest.json"

    target.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return target
