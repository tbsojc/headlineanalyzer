# tests/test_date_range_spec.py
from datetime import datetime, timedelta, timezone
import pytest

from app.api.date_range_spec import (
    validate_range, bucket_for_range, TO_IS_EXCLUSIVE,
    SMALL_RANGE_CUTOFF_DAYS, MAX_RANGE_DAYS
)

def _utc(y, m, d, H=0, M=0):
    return datetime(y, m, d, H, M, tzinfo=timezone.utc)

def test_validate_ok():
    f = _utc(2025, 1, 1)
    t = f + timedelta(days=1)
    validate_range(f, t)  # should not raise

def test_validate_exclusive():
    f = _utc(2025, 1, 1)
    t = f
    with pytest.raises(ValueError):
        validate_range(f, t)
    assert TO_IS_EXCLUSIVE is True

def test_validate_future_disallowed():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError):
        validate_range(now + timedelta(minutes=1), now + timedelta(hours=1))

def test_validate_max_span():
    f = _utc(2025, 1, 1)
    t = f + timedelta(days=MAX_RANGE_DAYS + 1)
    with pytest.raises(ValueError):
        validate_range(f, t)

def test_bucket_cutoff():
    f = _utc(2025, 1, 1)
    t_small = f + timedelta(days=SMALL_RANGE_CUTOFF_DAYS)
    t_large = f + timedelta(days=SMALL_RANGE_CUTOFF_DAYS, hours=1)
    assert bucket_for_range(f, t_small) == "hour"
    assert bucket_for_range(f, t_large) == "day"
