"""Constants must define the full v2 (asset, timeframe) cross-product cleanly."""
from v2.data.constants import (
    Asset,
    Timeframe,
    INTERVAL_MS,
    KLINE_COLUMNS,
    KLINE_DTYPES,
    DEFAULT_HISTORY_DAYS,
)


def test_assets_are_btc_eth_sol():
    assert {a.value for a in Asset} == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


def test_timeframes_are_1m_and_5m():
    assert {t.value for t in Timeframe} == {"1m", "5m"}


def test_interval_ms_matches_timeframes():
    assert INTERVAL_MS[Timeframe.M1] == 60_000
    assert INTERVAL_MS[Timeframe.M5] == 300_000


def test_kline_columns_canonical_order():
    assert KLINE_COLUMNS == (
        "open_time", "open", "high", "low", "close", "volume", "close_time", "regime",
    )


def test_kline_dtypes_match_columns():
    assert set(KLINE_DTYPES.keys()) == set(KLINE_COLUMNS)
    assert KLINE_DTYPES["open_time"] == "int64"
    assert KLINE_DTYPES["close"] == "float64"
    assert KLINE_DTYPES["regime"] == "int8"


def test_default_history_is_four_years():
    assert DEFAULT_HISTORY_DAYS == 4 * 365
