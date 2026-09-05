from datetime import datetime

from rees46.features.split import (
    TEST,
    TRAIN,
    VALIDATION,
    validate_split_contract,
)


def test_split_contract_is_valid() -> None:
    validate_split_contract()


def test_train_does_not_contain_validation_timestamp() -> None:
    timestamp = datetime(2020, 3, 15)

    assert TRAIN.contains(timestamp) is False
    assert VALIDATION.contains(timestamp) is True


def test_test_window_is_future_holdout() -> None:
    timestamp = datetime(2020, 4, 15)

    assert TRAIN.contains(timestamp) is False
    assert VALIDATION.contains(timestamp) is False
    assert TEST.contains(timestamp) is True
