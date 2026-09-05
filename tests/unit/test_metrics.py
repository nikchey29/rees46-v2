from rees46.evaluation.metrics import (
    average_precision_at_k,
    evaluate_recommendations,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_single_user_metrics() -> None:
    recommended = [1, 2, 3, 4]
    relevant = {2, 4}

    assert precision_at_k(recommended, relevant, 2) == 0.5
    assert recall_at_k(recommended, relevant, 2) == 0.5
    assert hit_rate_at_k(recommended, relevant, 2) == 1.0
    assert reciprocal_rank_at_k(recommended, relevant, 4) == 0.5
    assert average_precision_at_k(recommended, relevant, 4) > 0.0
    assert 0.0 < ndcg_at_k(recommended, relevant, 4) <= 1.0


def test_batch_evaluation() -> None:
    metrics = evaluate_recommendations(
        recommendations={1: [10, 20], 2: [30, 40]},
        ground_truth={1: {20}, 2: {99}},
        k_values=[1, 2],
    )

    assert metrics["users_evaluated"] == 2.0
    assert metrics["recall@2"] == 0.5
