# Methodology

## Data protocol

The project uses a chronological month holdout rather than a random split:

- train: October 2019 through February 2020;
- validation: March 2020;
- test: April 2020.

This prevents future purchases from leaking into candidate generation, collaborative factors, popularity signals, ranking labels, or sequential training.

## Recommendation stack

1. Global popularity establishes the non-personalized floor.
2. Category popularity provides a contextual cold-start/personalization baseline.
3. Session co-visitation learns item-to-item behavioral proximity.
4. Truncated-SVD collaborative filtering provides latent user-item personalization.
5. Reciprocal-rank fusion creates a calibration-free hybrid candidate set.
6. A gradient-boosted purchase classifier re-ranks validation-labeled candidates.
7. A TensorFlow GRU is evaluated separately as a sequential next-item experiment.

## Metrics

Every user-level recommender is compared with Precision@K, Recall@K, HitRate@K, NDCG@K, MAP@K and MRR@K. The sequence experiment uses the same Top-K metric family on held-out next items.

The repository never invents final metrics. `results/latest/` is populated only after a real local run.
