# Results

## Verified data engineering results

The local pipeline processed **411,709,736** Bronze events across October 2019-April 2020. It removed **1,384,422** exact duplicates (**0.336%**) and reconciled to **410,325,314** Silver events.

All audited critical Bronze checks were zero for critical nulls, invalid event types, invalid IDs, negative prices and wrong-month timestamps.

The October partitioned-Gold canary produced:

| Gold dataset | Rows |
|---|---:|
| user-item interactions | 23,307,630 |
| item statistics | 166,794 |
| user statistics | 3,022,290 |
| category affinity | 7,763,898 |
| session sequences | 5,971,465 |

## Recommendation benchmark

Run `./scripts/finalize_local.sh full`. The measured benchmark is then copied to `results/latest/model_benchmark.json` by `rees46 freeze-results`.

No model metric is pre-filled here before execution.
