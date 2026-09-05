# Measured results

`data_foundation.json` records measurements already produced by the real local REES46 pipeline.

`results/latest/` is populated only by `rees46 freeze-results` after the user runs the remaining Gold/EDA/model experiments. This intentionally keeps large Parquet datasets, model binaries, MLflow storage, and credentials out of Git while allowing small JSON evidence to be versioned.
