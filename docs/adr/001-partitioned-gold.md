# ADR 001: Bounded, resumable Gold aggregation

## Context

The first global Gold query attempted to group roughly 410 million Silver rows in one operation. DuckDB spilled about 30.6 GiB to its temporary directory and raised `OutOfMemoryException` when the temporary-disk limit was exhausted.

Gold was then redesigned to aggregate one month at a time. That bounded ordinary grouped facts successfully, but the ordered `list()` aggregates used to build session sequences still exhausted memory on November 2019. This exposed a different engine constraint: DuckDB can spill many blocking operators to disk, but `list()` aggregate state cannot currently be offloaded during a large aggregation.

## Decision

Gold remains month-partitioned. Within each month:

1. user-item, item, user, and category-affinity facts are built directly from the monthly Silver partition;
2. session events are streamed once into 64 deterministic hash buckets on `user_session`;
3. ordered session lists are aggregated one bucket at a time with a single DuckDB thread;
4. the completed bucket outputs are streamed into the monthly session-sequence Parquet file;
5. temporary bucket data is deleted automatically;
6. completed atomic Gold outputs are reused on restart, making interrupted finalization resumable.

Hash partitioning preserves session correctness because every row with the same `user_session` maps to the same bucket. The final union does not require another aggregation.

## Consequences

- bounded working-set size for the non-spillable ordered list aggregation;
- one source scan to create bucket partitions instead of repeatedly rescanning Silver;
- natural restartability at both month and table boundaries;
- easier audit and reconciliation;
- train/validation/test isolation remains visible in storage layout;
- downstream multi-month modeling operates on compact Gold facts rather than raw events.
