"""Funding-rate fetcher: pagination, schema, dedup, stop conditions."""
from pathlib import Path

import pandas as pd
import pytest
import responses

from v2.data.constants import Asset, FUNDING_COLUMNS
from v2.data.funding import (
    BINANCE_FUNDING_URL,
    chunk_to_funding_rows,
    fetch_funding_to_parquet,
)


def _payload_row(funding_time_ms: int, rate: str = "0.00010000",
                 mark_price: str = "30000.00") -> dict:
    return {
        "symbol": "BTCUSDT",
        "fundingTime": funding_time_ms,
        "fundingRate": rate,
        "markPrice": mark_price,
    }


def test_chunk_to_funding_rows_keeps_canonical_columns():
    rows = [_payload_row(0), _payload_row(8 * 3_600_000)]
    df = chunk_to_funding_rows(rows)
    assert list(df.columns) == list(FUNDING_COLUMNS)
    assert df["funding_time"].tolist() == [0, 8 * 3_600_000]
    assert df["funding_rate"].iloc[0] == pytest.approx(0.0001)
    assert df["mark_price"].iloc[0] == pytest.approx(30000.0)


def test_chunk_to_funding_rows_handles_missing_mark_price():
    """Some early historical rows lack markPrice — fill NaN, do not drop."""
    rows = [{"symbol": "BTCUSDT", "fundingTime": 0, "fundingRate": "0.0001"}]
    df = chunk_to_funding_rows(rows)
    assert pd.isna(df["mark_price"].iloc[0])


@responses.activate
def test_fetch_walks_forward_until_end_ms(tmp_path: Path):
    eight_h = 8 * 3_600_000
    page1 = [_payload_row(0), _payload_row(eight_h)]
    page2 = [_payload_row(2 * eight_h), _payload_row(3 * eight_h)]
    page3 = [_payload_row(4 * eight_h)]
    responses.add(responses.GET, BINANCE_FUNDING_URL, json=page1, status=200)
    responses.add(responses.GET, BINANCE_FUNDING_URL, json=page2, status=200)
    responses.add(responses.GET, BINANCE_FUNDING_URL, json=page3, status=200)

    out_path = tmp_path / "funding_btcusdt.parquet"
    df = fetch_funding_to_parquet(
        asset=Asset.BTC,
        target_start_ms=0,
        end_ms=4 * eight_h + 1,
        out_path=out_path,
        sleep_s=0,
    )
    assert df["funding_time"].tolist() == [0, eight_h, 2 * eight_h, 3 * eight_h, 4 * eight_h]
    assert out_path.exists()


@responses.activate
def test_fetch_stops_on_empty_chunk(tmp_path: Path):
    eight_h = 8 * 3_600_000
    responses.add(responses.GET, BINANCE_FUNDING_URL, json=[_payload_row(0)], status=200)
    responses.add(responses.GET, BINANCE_FUNDING_URL, json=[], status=200)
    out_path = tmp_path / "funding_btcusdt.parquet"
    df = fetch_funding_to_parquet(
        asset=Asset.BTC,
        target_start_ms=0,
        end_ms=10 * eight_h,
        out_path=out_path,
        sleep_s=0,
    )
    assert df["funding_time"].tolist() == [0]


@responses.activate
def test_fetch_dedupes_overlap(tmp_path: Path):
    eight_h = 8 * 3_600_000
    page1 = [_payload_row(0), _payload_row(eight_h)]
    page2 = [_payload_row(eight_h), _payload_row(2 * eight_h)]
    page3 = []
    responses.add(responses.GET, BINANCE_FUNDING_URL, json=page1, status=200)
    responses.add(responses.GET, BINANCE_FUNDING_URL, json=page2, status=200)
    responses.add(responses.GET, BINANCE_FUNDING_URL, json=page3, status=200)
    out_path = tmp_path / "funding_btcusdt.parquet"
    df = fetch_funding_to_parquet(
        asset=Asset.BTC,
        target_start_ms=0,
        end_ms=10 * eight_h,
        out_path=out_path,
        sleep_s=0,
    )
    assert df["funding_time"].tolist() == [0, eight_h, 2 * eight_h]
