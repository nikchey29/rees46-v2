# ADR 001: Month-partitioned Gold aggregation

## Context

The first global Gold query attempted to group roughly 410 million Silver rows in one operation. DuckDB spilled about 30.6 GiB to its temporary directory and raised `OutOfMemoryException` when the temporary-disk limit was exhausted.

## Decision

Gold is built one month at a time. Each monthly Silver partition creates monthly user-item, item, user, category-affinity and session-sequence facts. Modeling selects only the months permitted by the temporal split.

## Consequences

- bounded working-set size;
- natural restartability at month boundaries;
- easier audit/reconciliation;
- train/validation/test isolation is visible in storage layout;
- downstream multi-month aggregation happens on compact facts rather than raw events.
