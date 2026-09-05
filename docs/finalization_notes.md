# Finalization notes

This source completion intentionally does not contain raw REES46 files, generated Parquet layers, trained model binaries, MLflow storage, or invented recommendation metrics.

The local machine already produced the verified data-foundation measurements stored in `results/data_foundation.json`. The remaining Gold/EDA/model benchmark should be executed against that local data with `./scripts/finalize_local.sh full`. Afterward, `results/latest/` contains the small Git-trackable evidence to commit.

The first local command must be `uv lock` because the completed source adds final modeling/serving dependencies beyond the earlier lockfile state.
