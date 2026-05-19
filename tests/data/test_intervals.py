import pytest

from src.data.intervals import interval_to_milliseconds, interval_to_timedelta


@pytest.mark.unit
def test_interval_to_timedelta_minutes():
    assert interval_to_timedelta("3m").total_seconds() == 180


@pytest.mark.unit
def test_interval_to_milliseconds():
    assert interval_to_milliseconds("3m") == 180_000


@pytest.mark.unit
def test_interval_to_timedelta_supports_all_units_and_rejects_invalid():
    assert interval_to_timedelta("2h").total_seconds() == 7200
    assert interval_to_timedelta("1d").days == 1
    assert interval_to_timedelta("2w").days == 14

    with pytest.raises(ValueError, match="Unsupported interval"):
        interval_to_timedelta("bad")
