# REES46 V2 — Behavioral Recommendation Platform

A production-style recommendation engineering project built on the public REES46 multi-category marketplace behavior dataset. The repository treats recommendation as an end-to-end systems problem: data provenance, large-scale event processing, quality gates, leakage-safe evaluation, retrieval, ranking, sequential modeling, experiment tracking, serving, and reproducible operations.

<!-- recruiter-summary -->
[![CI](https://github.com/nikchey29/rees46-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/nikchey29/rees46-v2/actions/workflows/ci.yml) ![Python 3.11](https://img.shields.io/badge/Python-3.11-blue) [![Release](https://img.shields.io/badge/release-v0.1.0-blue)](https://github.com/nikchey29/rees46-v2/releases/tag/v0.1.0)

**411.7M Bronze events** · **410.3M canonical events** · **15.6M users** · **386K products** · **8.97M sessions**

**Production pipeline:** `Bronze → Silver → monthly Gold → retrieval → purchase ranking → Top-K evaluation → MLflow → FastAPI`

**Held-out April 2020 test vs. popularity baseline:** **+24.7% Recall@10** · **+10.6% Recall@20** · **+45.6% MRR@20** · **+23.3% NDCG@20**

## Verified scale

The local pipeline processed **411,709,736** Bronze events from October 2019 through April 2020. Exact deduplication removed **1,384,422** events (**0.336%**) and reconciled to **410,325,314** Silver events. Critical Bronze checks reported zero critical nulls, invalid event types, invalid IDs, negative prices, and wrong-month timestamps.

The October partitioned-Gold canary produced **23,307,630 user-item interactions**, **166,794 item rows**, **3,022,290 user rows**, **7,763,898 user-category affinities**, and **5,971,465 session sequences**.

Measured data-engineering evidence is tracked in [`results/data_foundation.json`](results/data_foundation.json). Final recommendation metrics are intentionally generated only by a real local run and frozen afterward; they are not fabricated in the repository.

## Architecture

```mermaid
flowchart LR
    RAW[REES46 CSV.GZ] --> B[Bronze Parquet]
    B --> V[Quality gates]
    V --> S[Silver canonical events]
    S --> G[Monthly Gold facts]
    G --> P[Popularity]
    G --> CV[Co-visitation]
    G --> CF[Collaborative SVD]
    P --> H[Hybrid retrieval]
    CV --> H
    CF --> H
    H --> R[Purchase ranker]
    G --> Q[TensorFlow sequence model]
    R --> E[Top-K evaluation]
    Q --> E
    E --> M[MLflow / JSON results]
    M --> API[FastAPI]
    M --> PG[(PostgreSQL)]
```

See [`docs/architecture.md`](docs/architecture.md) for the detailed design and [`docs/adr/001-partitioned-gold.md`](docs/adr/001-partitioned-gold.md) for the large-scale aggregation decision.

## Modeling protocol

The evaluation protocol is chronological and leakage-safe:

- **Train:** October 2019–February 2020
- **Validation:** March 2020
- **Test:** April 2020

The stack includes:

- global popularity;
- category/context popularity;
- session co-visitation;
- truncated-SVD collaborative filtering;
- reciprocal-rank-fusion hybrid retrieval;
- gradient-boosted purchase-oriented re-ranking;
- TensorFlow GRU next-item recommendation;
- Precision@K, Recall@K, HitRate@K, NDCG@K, MAP@K and MRR@K;
- MLflow experiment tracking.

## Repository structure

```text
configs/                 typed runtime + modeling profiles
docs/                    architecture, methodology, runbook, ADRs
results/                 small Git-trackable measured evidence
scripts/                 download, verification, local finalization
sql/ddl/                 PostgreSQL serving/metadata schema
src/rees46/
  ingestion/             raw manifest + Bronze ingestion
  validation/            Bronze quality gates
  preprocessing/         Silver canonicalization
  features/              Gold facts + temporal contract
  analytics/             behavioral EDA
  recommendations/       retrieval, ranking, sequence models
  evaluation/            Top-K metrics
  experiments/           leakage-safe datasets, runner, MLflow, freezing
  serving/               FastAPI inference
  warehouse/             PostgreSQL publication
  runtime/               typed config, logging, pipeline contract
tests/                    source/unit/API/metric contracts
```

Large datasets, Parquet files, trained models, MLflow storage and credentials are deliberately excluded from Git.

## Quick start

Python 3.11 and `uv` are required.

```bash
uv lock
uv sync --all-extras
make quality
```

If the local Bronze/Silver data already exists, finish everything from Gold onward with:

```bash
./scripts/finalize_local.sh full
```

The wrapper runs the complete source quality gate, builds any missing Gold layer, runs EDA, trains/evaluates the model stack, freezes small result JSONs, and re-runs the quality gate.

For a fast development smoke experiment:

```bash
uv run rees46 run-experiments --profile dev --no-sequence --no-mlflow
```

## CLI

```bash
uv run rees46 --help
uv run rees46 info --profile full
uv run rees46 pipeline --profile full
uv run rees46 build-data --profile full
uv run rees46 build-silver --profile full
uv run rees46 build-gold --profile full
uv run rees46 eda --profile full
uv run rees46 run-experiments --profile full
uv run rees46 freeze-results --profile full
```

## Serving

After an experiment creates `models/recommendation_bundle.joblib`:

```bash
make api
```

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model
curl 'http://localhost:8000/recommend/123456789?k=20'
```

Known users receive hybrid/ranked recommendations. Unknown users fall back explicitly to global popularity.

## PostgreSQL + Docker

Start PostgreSQL:

```bash
docker compose up -d postgres
```

Publish model metadata and item facts:

```bash
export REES46_POSTGRES_DSN='postgresql://rees46:rees46@localhost:5432/rees46'
uv run rees46 publish-postgres --profile full --dsn "$REES46_POSTGRES_DSN"
```

Run the API container after the model bundle exists:

```bash
docker compose up --build api
```

## Quality and reproducibility

```bash
make verify
make quality
```

The quality gate runs source verification, Ruff formatting/linting, strict mypy, and pytest. GitHub Actions runs the same source-only checks without downloading the 400M-event dataset or training models.

## Scalability lesson

The first global Gold aggregation over roughly 410M Silver rows exhausted DuckDB temporary storage after a ~30.6 GiB spill. The system was redesigned around monthly Gold partitions instead of increasing memory/disk limits and keeping a fragile monolithic query. This makes the pipeline restartable, bounded in working-set size, and naturally aligned with chronological evaluation.


## Documentation

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/methodology.md`](docs/methodology.md)
- [`docs/results.md`](docs/results.md)
- [`docs/runbook.md`](docs/runbook.md)
- [`docs/portfolio.md`](docs/portfolio.md)
- [`docs/adr/001-partitioned-gold.md`](docs/adr/001-partitioned-gold.md)

## Results

REES46 V2 was evaluated using a strict chronological split to prevent future-data leakage:

- **Training:** October 2019 - February 2020
- **Validation:** March 2020
- **Test:** April 2020

The production-style pipeline processed the seven-month REES46 multi-category behavioral dataset:

| Data statistic | Value |
| --- | ---: |
| Bronze events | 411,709,736 |
| Canonical Silver events | 410,325,314 |
| Exact duplicates removed | 1,384,422 |
| Users | 15,639,803 |
| Products | 386,299 |
| Categories | 1,325 |
| Sessions | 8,969,359 |

Data-quality validation found zero critical nulls, invalid event types, invalid IDs, negative prices, or wrong-month events in the validated Bronze partitions.

### Offline recommendation benchmark

The final benchmark used 5,000 validation users and 5,000 held-out test users.

| Model | Precision@5 | Recall@5 | Recall@10 | Recall@20 | HitRate@20 | MRR@20 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularity | 0.01084 | 0.03319 | 0.05685 | 0.08891 | 0.1250 | 0.02443 | 0.03593 |
| Final purchase ranker | **0.01180** | **0.03588** | **0.07087** | **0.09835** | **0.1372** | **0.03556** | **0.04429** |

Against the popularity baseline on the untouched April 2020 test set, the final purchase-oriented ranker improved:

- **Recall@10 by 24.7%**
- **Recall@20 by 10.6%**
- **MRR@20 by 45.6%**
- **NDCG@20 by 23.3%**

The experiments also evaluated category-popularity, co-visitation, collaborative-filtering, hybrid-retrieval, ranking, and TensorFlow sequential approaches. The sequential model was retained as an experiment rather than presented as the winner: additional model complexity did not outperform the final ranking pipeline on the held-out benchmark.

### Production validation

The final recommendation bundle is exposed through a FastAPI service.

Local API smoke testing verified `/health` returned status `ok` with the model loaded, and `/model` identified the deployed model as `hybrid+ranker` with a candidate pool size of 100.

Experiment metadata is tracked locally with MLflow backed by SQLite. Large raw datasets, derived Parquet datasets, MLflow state, and trained model binaries are intentionally excluded from Git.

## Limitations

This project evaluates recommendation quality offline on historical implicit-feedback data. The results therefore demonstrate performance on the defined chronological holdout protocol, not online user engagement or business impact.

The large-scale data pipeline processes the full seven-month REES46 dataset, while recommendation benchmarking uses bounded user/session samples so multiple models can be compared reproducibly on local hardware. Reported recommendation metrics should therefore be interpreted as benchmark results rather than metrics over every user in the full dataset.

The system does not currently include an online A/B-testing layer, real-time feature streaming, or continuous model retraining. New or sparse-history users also have less behavioral signal available for personalization and may rely more heavily on fallback popularity behavior.

The TensorFlow sequential recommender was implemented and evaluated as an experiment, but it did not outperform the final purchase-oriented ranking pipeline on the held-out benchmark. It is retained as experimental evidence rather than presented as the production winner.

Large raw files, generated Parquet datasets, local MLflow state, and trained model binaries are intentionally excluded from Git because of their size. The repository instead versions source code, configuration, tests, documentation, and compact measured-result artifacts.
