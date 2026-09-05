import pandas as pd

from rees46.recommendations.ranking.model import PurchaseRanker


def test_ranker_fits_and_scores() -> None:
    frame = pd.DataFrame(
        {
            "product_id": [1, 2, 3, 4, 5, 6],
            "pop_rr": [1.0, 0.5, 0.3, 0.2, 0.1, 0.05],
            "covis_rr": [0.0, 0.2, 1.0, 0.0, 0.5, 0.0],
            "label": [1, 0, 1, 0, 1, 0],
        }
    )
    model = PurchaseRanker.fit(frame, ["pop_rr", "covis_rr"])
    scores = model.predict_scores(frame)
    assert len(scores) == len(frame)
    assert all(0.0 <= float(score) <= 1.0 for score in scores)
