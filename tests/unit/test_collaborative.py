import pandas as pd

from rees46.recommendations.collaborative.model import CollaborativeModel


def test_collaborative_model_recommends_for_known_user() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "product_id": [10, 20, 10, 30, 20, 30],
            "interaction_strength": [5.0, 1.0, 4.0, 2.0, 4.0, 3.0],
        }
    )
    model = CollaborativeModel.fit(interactions, components=2)
    recommendations = model.recommend(1, 2, exclude={10, 20})
    assert recommendations == [30]


def test_collaborative_unknown_user_is_empty() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "product_id": [10, 20, 10, 20],
            "interaction_strength": [1.0, 2.0, 2.0, 1.0],
        }
    )
    model = CollaborativeModel.fit(interactions, components=1)
    assert model.recommend(999, 5) == []
