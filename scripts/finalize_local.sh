#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-full}"

uv lock
uv sync --all-extras
make quality
uv run rees46 finalize --profile "${PROFILE}"
make quality
uv run rees46 freeze-results --profile "${PROFILE}"

echo
echo "REES46 V2 local finalization completed."
echo "Review results/latest and reports/metrics before committing."
git status --short
