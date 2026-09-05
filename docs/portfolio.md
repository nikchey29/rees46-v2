# Portfolio / resume evidence

## Safe resume bullets now

- Engineered a production-style recommendation data platform over **411.7M real e-commerce events**, using DuckDB, typed Python, ZSTD Parquet and Bronze/Silver/Gold layers; automated schema/data-quality checks and reconciled **410.3M canonical events** after removing **1.38M exact duplicates**.
- Redesigned a failed 410M-row monolithic Gold aggregation into restartable month-partitioned facts after diagnosing a **30.6 GiB DuckDB spill/OOM**, improving scalability while preserving chronological train/validation/test boundaries.
- Implemented an offline recommendation stack spanning popularity, category popularity, session co-visitation, latent-factor collaborative filtering, hybrid reciprocal-rank fusion, gradient-boosted purchase ranking and a TensorFlow sequential next-item experiment, evaluated with Top-K ranking metrics.
- Built typed configuration, structured logging, CLI orchestration, FastAPI inference with cold-start fallback, MLflow tracking hooks, PostgreSQL publication, Docker Compose and GitHub Actions source CI.

## After the final benchmark

Add only the measured winner and metrics from `results/latest/model_benchmark.json`; do not invent or round metrics before the run completes.
