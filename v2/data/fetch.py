"""Binance public kline fetcher.

Walks backward in time, MAX_LIMIT bars per request, until either:
  - target_start_ms reached, or
  - Binance returns an empty page (start of available history).

All writes go through `store.write_klines`, which enforces the canonical schema.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from v2.data.constants import (
    Asset,
    Timeframe,
    INTERVAL_MS,
    KLINE_COLUMNS,
    KLINE_DTYPES,
    DEFAULT_HISTORY_DAYS,
)
from v2.data.store import parquet_path, write_klines
from v2.data.validate import dedupe_open_time

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000  # Binance hard cap per request

# Binance returns 12 fields per kline; we keep 7.
_RAW_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def chunk_to_rows(chunk: list[list]) -> pd.DataFrame:
    """Convert a raw Binance kline payload into a canonical-schema DataFrame."""
    df = pd.DataFrame(chunk, columns=_RAW_COLUMNS)
    df = df[list(KLINE_COLUMNS)].copy()
    for col, dtype in KLINE_DTYPES.items():
        df[col] = df[col].astype(dtype)
    return df


def _fetch_chunk(symbol: str, interval: str, end_ms: int, limit: int = MAX_LIMIT) -> list[list]:
    params = {"symbol": symbol, "interval": interval, "endTime": end_ms, "limit": limit}
    r = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_to_parquet(
    asset: Asset,
    timeframe: Timeframe,
    end_ms: int,
    target_start_ms: int,
    out_path: Path,
    sleep_s: float = 0.05,
) -> pd.DataFrame:
    """Fetch klines from `target_start_ms` to `end_ms` and write to `out_path`."""
    all_rows: list[pd.DataFrame] = []
    seen: set[int] = set()
    cursor_end = end_ms
    while True:
        chunk = _fetch_chunk(asset.value, timeframe.value, cursor_end)
        if not chunk:
            break
        df_chunk = chunk_to_rows(chunk)
        new_mask = ~df_chunk["open_time"].isin(seen)
        if not new_mask.any():
            break
        new_df = df_chunk[new_mask]
        seen.update(int(t) for t in new_df["open_time"].tolist())
        all_rows.append(new_df)
        oldest_open = int(new_df["open_time"].min())
        if oldest_open <= target_start_ms:
            break
        cursor_end = oldest_open - 1
        if sleep_s > 0:
            time.sleep(sleep_s)

    if not all_rows:
        raise RuntimeError(
            f"No data returned for {asset.value} {timeframe.value} in window "
            f"[{target_start_ms}, {end_ms}]"
        )
    df = pd.concat(all_rows, ignore_index=True)
    df = dedupe_open_time(df)
    write_klines(df, out_path)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Binance klines for a (asset, timeframe) pair.")
    ap.add_argument("--asset", required=True, choices=[a.value for a in Asset])
    ap.add_argument("--timeframe", required=True, choices=[t.value for t in Timeframe])
    ap.add_argument("--days", type=int, default=DEFAULT_HISTORY_DAYS,
                    help=f"history depth in days (default: {DEFAULT_HISTORY_DAYS})")
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).parent / "raw",
                    help="output dir (parquet path = root / <symbol>_<interval>.parquet)")
    args = ap.parse_args()

    asset = Asset(args.asset)
    timeframe = Timeframe(args.timeframe)
    out_path = parquet_path(args.root, asset, timeframe)

    now_ms = int(time.time() * 1000)
    target_start_ms = now_ms - args.days * 24 * 60 * 60 * 1000

    print(f"Fetching {asset.value} {timeframe.value} for {args.days} days → {out_path}")
    df = fetch_to_parquet(
        asset=asset,
        timeframe=timeframe,
        end_ms=now_ms,
        target_start_ms=target_start_ms,
        out_path=out_path,
    )
    print(f"Done. {len(df)} bars saved.")


if __name__ == "__main__":
    main()
