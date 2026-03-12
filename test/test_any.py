import datetime as dt

from src.utils.datetime_utils import SHANGHAI_TZ, UTC, coerce_any_to_utc_datetime, ensure_utc, utc_isoformat


def test_ensure_utc_converts_naive_datetime_from_shanghai():
    naive = dt.datetime(2026, 3, 10, 20, 0, 0)
    converted = ensure_utc(naive)
    assert converted.tzinfo == UTC and converted.hour == 12


def test_coerce_any_to_utc_datetime_parses_iso_z_string():
    converted = coerce_any_to_utc_datetime("2026-03-10T12:00:00Z")
    assert converted.tzinfo == UTC and converted.hour == 12


def test_utc_isoformat_ends_with_z():
    value = dt.datetime(2026, 3, 10, 12, 0, 0, tzinfo=UTC)
    assert utc_isoformat(value).endswith("Z")


def test_shanghai_timezone_constant_is_available():
    assert SHANGHAI_TZ is not None