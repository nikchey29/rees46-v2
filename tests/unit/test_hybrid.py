from rees46.recommendations.candidates.hybrid import candidate_features, reciprocal_rank_fusion


def test_rrf_promotes_cross_source_item() -> None:
    fused = reciprocal_rank_fusion(
        {"popularity": [1, 2, 3], "collaborative": [3, 4, 5]},
        k=3,
    )
    assert fused[0] == 3


def test_candidate_features_capture_presence_and_rank() -> None:
    features = candidate_features({"a": [10, 20]}, [10, 30])
    assert features[10]["a_present"] == 1.0
    assert features[10]["a_rr"] == 1.0
    assert features[30]["a_present"] == 0.0
