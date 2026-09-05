from rees46.recommendations.sequential import build_sequence_examples


def test_sequence_example_builder_is_chronological() -> None:
    x, y, mapping, vocabulary = build_sequence_examples(
        sessions=[[10, 20, 30], [10, 40]],
        vocab_size=10,
        max_len=3,
    )
    assert x.shape[1] == 3
    assert len(x) == len(y)
    assert 10 in mapping
    assert set(vocabulary) == {10, 20, 30, 40}
