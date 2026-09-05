import pandas as pd

from rees46.recommendations.popularity.model import CategoryPopularityModel, PopularityModel


def _interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": [1, 2, 3, 4],
            "category_id": [10, 10, 20, 20],
            "views": [10, 3, 2, 1],
            "carts": [1, 4, 1, 0],
            "purchases": [0, 2, 0, 0],
            "interaction_strength": [13.0, 25.0, 5.0, 1.0],
        }
    )


def test_global_popularity() -> None:
    model = PopularityModel.fit(_interactions())
    assert model.recommend(2) == [2, 1]
    assert model.recommend(2, exclude={2}) == [1, 3]


def test_category_popularity() -> None:
    model = CategoryPopularityModel.fit(_interactions())
    assert model.recommend(20, 2)[0] == 3
