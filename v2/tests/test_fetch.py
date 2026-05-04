"""Fetcher pagination, dedup, and stop-condition logic, with HTTP mocked."""
from pathlib import Path

import pandas as pd
import pytest
import responses

from v2.data.constants import Asset, Timeframe
from v2.data.fetch import (
    BINANCE_KLINES_URL,
    chunk_to_rows,
    fetch_to_parquet,
)


def _kline(open_ms: int, interval_ms: int = 60_000) -> list:
    """Build one Binance-shaped kline row."""
    return [
        open_ms,
        "1.0", "1.2", "0.9", "1.1", "10.0",
        open_ms + interval_ms - 1,
        "11.0", 5, "5.0", "5.5", "0",
    ]


def test_chunk_to_rows_keeps_only_canonical_columns():
    rows = [_kline(0), _kline(60_000)]
    df = chunk_to_rows(rows)
    assert list(df.columns) == [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
    ]
    assert df["open_time"].tolist() == [0, 60_000]
    assert df["close"].tolist() == [1.1, 1.1]
    # chunk_to_rows returns the 7-col base before regime is added by fetch_to_parquet


@responses.activate
def test_fetch_walks_backward_until_target_start(tmp_path: Path):
    # Three pages, each with 2 candles, working backward from now.
    # Page 1 (newest): open_times 240_000, 300_000
    # Page 2:          open_times 120_000, 180_000
    # Page 3 (oldest): open_times 0, 60_000     <-- target_start_ms = 0, stop here
    page1 = [_kline(240_000), _kline(300_000)]
    page2 = [_kline(120_000), _kline(180_000)]
    page3 = [_kline(0), _kline(60_000)]
    responses.add(responses.GET, BINANCE_KLINES_URL, json=page1, status=200)
    responses.add(responses.GET, BINANCE_KLINES_URL, json=page2, status=200)
    responses.add(responses.GET, BINANCE_KLINES_URL, json=page3, status=200)

    out_path = tmp_path / "btcusdt_1m.parquet"
    df = fetch_to_parquet(
        asset=Asset.BTC,
        timeframe=Timeframe.M1,
        end_ms=300_000 + 60_000,  # "now" is just after the newest bar
        target_start_ms=0,
        out_path=out_path,
        sleep_s=0,
    )
    assert df["open_time"].tolist() == [0, 60_000, 120_000, 180_000, 240_000, 300_000]
    assert out_path.exists()
    # Schema-version sanity: fetcher writes the untagged sentinel.
    out_df = pd.read_parquet(out_path)
    assert "regime" in out_df.columns
    assert (out_df["regime"] == -1).all()
    assert out_df["regime"].dtype == "int8"


@responses.activate
def test_fetch_stops_on_empty_chunk(tmp_path: Path):
    responses.add(responses.GET, BINANCE_KLINES_URL, json=[_kline(60_000)], status=200)
    responses.add(responses.GET, BINANCE_KLINES_URL, json=[], status=200)
    out_path = tmp_path / "btcusdt_1m.parquet"
    df = fetch_to_parquet(
        asset=Asset.BTC,
        timeframe=Timeframe.M1,
        end_ms=120_000,
        target_start_ms=0,
        out_path=out_path,
        sleep_s=0,
    )
    assert df["open_time"].tolist() == [60_000]


@responses.activate
def test_fetch_dedupes_repeated_open_times(tmp_path: Path):
    # Binance occasionally returns overlapping pages; fetcher must dedupe.
    page1 = [_kline(120_000), _kline(180_000)]
    page2 = [_kline(60_000), _kline(120_000)]  # 120_000 repeated
    page3 = [_kline(0), _kline(60_000)]        # 60_000 repeated
    responses.add(responses.GET, BINANCE_KLINES_URL, json=page1, status=200)
    responses.add(responses.GET, BINANCE_KLINES_URL, json=page2, status=200)
    responses.add(responses.GET, BINANCE_KLINES_URL, json=page3, status=200)
    out_path = tmp_path / "btcusdt_1m.parquet"
    df = fetch_to_parquet(
        asset=Asset.BTC,
        timeframe=Timeframe.M1,
        end_ms=240_000,
        target_start_ms=0,
        out_path=out_path,
        sleep_s=0,
    )
    assert df["open_time"].tolist() == [0, 60_000, 120_000, 180_000]
