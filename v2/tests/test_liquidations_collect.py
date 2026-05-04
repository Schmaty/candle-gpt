"""Live liquidation collector parsing + restart-safe append (no real socket here)."""
from pathlib import Path

import pandas as pd
import pytest

from v2.data.liquidations.rollup import EVENT_COLUMNS, EVENT_DTYPES
from v2.data.liquidations.collect import (
    parse_force_order_event,
    daily_parquet_path,
    append_events,
)


# Sample forceOrder@arr payload from Binance docs.
SAMPLE_PAYLOAD = {
    "e": "forceOrder",
    "E": 1700000000123,
    "o": {
        "s": "BTCUSDT",
        "S": "SELL",         # SELL = a long position got liquidated → side="long"
        "o": "LIMIT",
        "f": "IOC",
        "q": "0.5",          # qty
        "p": "30000.5",      # price (limit)
        "ap": "30001.2",     # average filled price
        "X": "FILLED",
        "l": "0.5",
        "z": "0.5",
        "T": 1700000000456,
    },
}


def test_parse_force_order_event_maps_sell_to_long():
    ev = parse_force_order_event(SAMPLE_PAYLOAD)
    assert ev["side"] == "long"
    assert ev["event_time"] == 1700000000456
    assert ev["price"] == pytest.approx(30001.2)  # uses average price, not limit
    assert ev["qty"] == pytest.approx(0.5)
    assert ev["notional"] == pytest.approx(0.5 * 30001.2)


def test_parse_force_order_event_maps_buy_to_short():
    payload = {**SAMPLE_PAYLOAD, "o": {**SAMPLE_PAYLOAD["o"], "S": "BUY"}}
    ev = parse_force_order_event(payload)
    assert ev["side"] == "short"


def test_parse_rejects_non_btc_symbol():
    payload = {**SAMPLE_PAYLOAD, "o": {**SAMPLE_PAYLOAD["o"], "s": "ETHUSDT"}}
    with pytest.raises(ValueError, match="symbol"):
        parse_force_order_event(payload, expected_symbol="BTCUSDT")


def test_daily_parquet_path_rolls_at_utc_midnight(tmp_path: Path):
    p = daily_parquet_path(tmp_path, event_time_ms=1700006400000)
    # 2023-11-15 00:00:00 UTC → "2023-11-15.parquet"
    assert p.name == "2023-11-15.parquet"


def test_append_events_creates_then_appends(tmp_path: Path):
    p = tmp_path / "2023-11-15.parquet"
    e1 = parse_force_order_event(SAMPLE_PAYLOAD)
    append_events(p, [e1])
    df1 = pd.read_parquet(p)
    assert len(df1) == 1
    assert list(df1.columns) == list(EVENT_COLUMNS)

    e2 = parse_force_order_event(
        {**SAMPLE_PAYLOAD, "o": {**SAMPLE_PAYLOAD["o"], "S": "BUY"}}
    )
    append_events(p, [e2])
    df2 = pd.read_parquet(p)
    assert len(df2) == 2  # restart-safe: existing rows preserved
