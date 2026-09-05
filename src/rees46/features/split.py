"""Canonical chronological train/validation/test contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TimeWindow:
    """One half-open temporal interval."""

    start: datetime
    end: datetime

    def contains(self, timestamp: datetime) -> bool:
        """Return whether a timestamp lies inside this window."""
        return self.start <= timestamp < self.end


TRAIN = TimeWindow(
    start=datetime(2019, 10, 1),
    end=datetime(2020, 3, 1),
)

VALIDATION = TimeWindow(
    start=datetime(2020, 3, 1),
    end=datetime(2020, 4, 1),
)

TEST = TimeWindow(
    start=datetime(2020, 4, 1),
    end=datetime(2020, 5, 1),
)


def validate_split_contract() -> None:
    """Ensure chronological split boundaries never overlap."""

    if TRAIN.end > VALIDATION.start:
        raise ValueError("Train overlaps validation.")

    if VALIDATION.end > TEST.start:
        raise ValueError("Validation overlaps test.")

    if not (TRAIN.start < TRAIN.end <= VALIDATION.start < VALIDATION.end <= TEST.start < TEST.end):
        raise ValueError("Temporal split ordering is invalid.")
