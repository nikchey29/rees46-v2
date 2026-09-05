# Architecture

```mermaid
flowchart TD
    A[REES46 CSV.GZ] --> B[Raw manifest + SHA-256]
    B --> C[DuckDB typed ingestion]
    C --> D[Bronze monthly Parquet]
    D --> E[Data-quality gates]
    E --> F[Silver exact dedup + normalization]
    F --> G[Month-partitioned Gold]
    G --> G1[User-item interactions]
    G --> G2[Item statistics]
    G --> G3[User statistics]
    G --> G4[Category affinity]
    G --> G5[Session sequences]
    G1 --> H[Oct-Feb training sample]
    G5 --> H
    H --> P[Popularity]
    H --> CP[Category popularity]
    H --> CV[Co-visitation]
    H --> CF[Truncated-SVD collaborative]
    P --> HY[Reciprocal-rank fusion]
    CP --> HY
    CV --> HY
    CF --> HY
    HY --> R[Gradient-boosted purchase ranker]
    G5 --> S[TensorFlow GRU sequence experiment]
    R --> EVAL[March validation / April final holdout]
    S --> EVAL
    EVAL --> M[Metrics + MLflow]
    M --> API[FastAPI serving]
    M --> PG[(PostgreSQL metadata/facts)]
```

## Boundaries

Bronze is a typed faithful copy. Silver performs exact deduplication and normalization. Gold contains reusable behavioral facts. Modeling never reads March/April while fitting train-time models. March is the ranking/tuning surface and April is the future holdout.
