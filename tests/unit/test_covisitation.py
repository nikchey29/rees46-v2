from rees46.recommendations.covisitation.model import CoVisitationModel


def test_covisitation_learns_neighbors() -> None:
    model = CoVisitationModel.fit([[1, 2, 3], [1, 2], [2, 4]], neighbors_per_item=3)
    assert 2 in dict(model.neighbors[1])
    assert model.recommend([1], 2)[0] == 2


def test_covisitation_respects_exclusions() -> None:
    model = CoVisitationModel.fit([[1, 2, 3]])
    assert 2 not in model.recommend([1], 3, exclude={2})
