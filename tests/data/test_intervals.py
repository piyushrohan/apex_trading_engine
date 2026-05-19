import pytest

from src.data.intervals import interval_to_milliseconds, interval_to_timedelta


@pytest.mark.unit
def test_interval_to_timedelta_minutes():
    assert interval_to_timedelta("3m").total_seconds() == 180


@pytest.mark.unit
def test_interval_to_milliseconds():
    assert interval_to_milliseconds("3m") == 180_000
