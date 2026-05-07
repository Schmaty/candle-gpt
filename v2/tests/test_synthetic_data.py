from pathlib import Path

import numpy as np

from v2.data.synthetic import generate_synthetic_market, write_synthetic_raw_dir
from v2.data.store import read_klines, read_funding, read_liq_bucketed


def test_generate_synthetic_market_schema_and_basic_sanity():
    klines, funding, liq = generate_synthetic_market(n_minutes=2000, seed=123)
    assert len(klines) == 2000
    assert len(liq) == 2000
    assert len(funding) >= 4
    assert set(klines["regime"].unique()).issubset({0, 1, 2})
    assert (klines["high"] >= klines[["open", "close"]].max(axis=1)).all()
    assert (klines["low"] <= klines[["open", "close"]].min(axis=1)).all()
    assert (klines["close"] > 0).all()
    assert (klines["volume"] >= 0).all()
    assert (liq["count"] >= 0).all()
    assert np.isfinite(klines[["open", "high", "low", "close", "volume"]].to_numpy()).all()


def test_write_synthetic_raw_dir_roundtrips(tmp_path: Path):
    raw_dir = tmp_path / "synthetic_raw"
    write_synthetic_raw_dir(raw_dir, n_minutes=1500, seed=5)
    klines = read_klines(raw_dir / "btcusdt_1m.parquet")
    funding = read_funding(raw_dir / "funding_btcusdt.parquet")
    liq = read_liq_bucketed(raw_dir / "liq_btcusdt_per_minute.parquet")
    assert len(klines) == 1500
    assert len(liq) == 1500
    assert len(funding) > 0
