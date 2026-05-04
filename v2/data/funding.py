"""Binance perpetual funding-rate fetcher.

Primary source: Binance public data (data.binance.vision) monthly + daily zip files.
Fallback (geo-restricted regions): GET /fapi/v1/fundingRate paginated API.

Stores to parquet via store.write_funding (which embeds schema metadata).
"""
from __future__ import annotations
import argparse
import io
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from v2.data.constants import (
    Asset,
    FUNDING_COLUMNS,
    FUNDING_DTYPES,
)
from v2.data.store import funding_parquet_path, write_funding


BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
BINANCE_PUBLIC_DATA_BASE = "https://data.binance.vision/data/futures/um"
MAX_LIMIT = 1000  # Binance hard cap


def chunk_to_funding_rows(rows: list[dict]) -> pd.DataFrame:
    """Convert a /fapi/v1/fundingRate response into the canonical funding schema."""
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=FUNDING_DTYPES[c]) for c in FUNDING_COLUMNS})
    df = pd.DataFrame({
        "funding_time": pd.array([int(r["fundingTime"]) for r in rows], dtype="int64"),
        "funding_rate": [float(r["fundingRate"]) for r in rows],
        "mark_price":   [float(r["markPrice"]) if r.get("markPrice") is not None else float("nan")
                         for r in rows],
    })
    return df


def _fetch_chunk(symbol: str, start_ms: int, limit: int = MAX_LIMIT) -> list[dict]:
    params = {"symbol": symbol, "startTime": start_ms, "limit": limit}
    r = requests.get(BINANCE_FUNDING_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _download_public_zip(url: str) -> pd.DataFrame | None:
    """Download a monthly/daily funding rate zip and return parsed DataFrame or None."""
    r = requests.get(url, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            df = pd.read_csv(f)
    # Public data columns: calc_time, funding_interval_hours, last_funding_rate
    df = df.rename(columns={"calc_time": "funding_time", "last_funding_rate": "funding_rate"})
    df["funding_time"] = df["funding_time"].astype("int64")
    df["funding_rate"] = df["funding_rate"].astype("float64")
    df["mark_price"] = float("nan")  # not provided in public data
    return df[list(FUNDING_COLUMNS)]


def fetch_funding_from_public_data(
    asset: Asset,
    target_start_ms: int,
    end_ms: int,
    out_path: Path,
) -> pd.DataFrame:
    """Download monthly + daily funding-rate CSVs from data.binance.vision."""
    symbol = asset.value
    start_dt = datetime.fromtimestamp(target_start_ms / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)

    frames: list[pd.DataFrame] = []

    # Monthly files from start year-month through end year-month
    year, month = start_dt.year, start_dt.month
    while (year, month) <= (end_dt.year, end_dt.month):
        ym = f"{year:04d}-{month:02d}"
        url = f"{BINANCE_PUBLIC_DATA_BASE}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip"
        df = _download_public_zip(url)
        if df is not None and not df.empty:
            frames.append(df)
        # advance month
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

    # Daily files for current month (monthly archives lag ~1 month)
    today = datetime.now(tz=timezone.utc)
    d = datetime(end_dt.year, end_dt.month, 1, tzinfo=timezone.utc)
    while d.date() < today.date():
        ds = d.strftime("%Y-%m-%d")
        url = f"{BINANCE_PUBLIC_DATA_BASE}/daily/fundingRate/{symbol}/{symbol}-fundingRate-{ds}.zip"
        df = _download_public_zip(url)
        if df is not None and not df.empty:
            frames.append(df)
        d = datetime(d.year, d.month + 1 if d.month < 12 else 1,
                     1 if d.month < 12 else 1,
                     tzinfo=timezone.utc) if d.day == 1 else \
            datetime(d.year, d.month, d.day + 1, tzinfo=timezone.utc)

    if not frames:
        raise RuntimeError(f"No public funding data found for {symbol}")

    df_all = (
        pd.concat(frames, ignore_index=True)
          .drop_duplicates(subset=["funding_time"], keep="first")
          .sort_values("funding_time", kind="mergesort")
          .reset_index(drop=True)
    )
    # Filter to requested window
    df_all = df_all[
        (df_all["funding_time"] >= target_start_ms) &
        (df_all["funding_time"] < end_ms)
    ].reset_index(drop=True)

    for col, dtype in FUNDING_DTYPES.items():
        if col != "mark_price":  # mark_price is NaN float64, already correct
            df_all[col] = df_all[col].astype(dtype)

    write_funding(df_all, out_path)
    return df_all


def fetch_funding_to_parquet(
    asset: Asset,
    target_start_ms: int,
    end_ms: int,
    out_path: Path,
    sleep_s: float = 0.05,
) -> pd.DataFrame:
    all_chunks: list[pd.DataFrame] = []
    seen: set[int] = set()
    cursor = target_start_ms
    while cursor < end_ms:
        rows = _fetch_chunk(asset.value, cursor)
        if not rows:
            break
        df_chunk = chunk_to_funding_rows(rows)
        new_mask = ~df_chunk["funding_time"].isin(seen)
        if not new_mask.any():
            break
        new_df = df_chunk[new_mask & (df_chunk["funding_time"] < end_ms)]
        if new_df.empty:
            break
        seen.update(int(t) for t in new_df["funding_time"].tolist())
        all_chunks.append(new_df)
        latest = int(new_df["funding_time"].max())
        cursor = latest + 1
        if sleep_s > 0:
            time.sleep(sleep_s)

    if not all_chunks:
        raise RuntimeError(
            f"No funding data returned for {asset.value} in window "
            f"[{target_start_ms}, {end_ms}]"
        )
    df = (
        pd.concat(all_chunks, ignore_index=True)
          .sort_values("funding_time", kind="mergesort")
          .drop_duplicates(subset=["funding_time"], keep="first")
          .reset_index(drop=True)
    )
    write_funding(df, out_path)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Binance perpetual funding rates.")
    ap.add_argument("--asset", required=True, choices=[a.value for a in Asset])
    ap.add_argument("--years", type=float, default=6.5,
                    help="history depth in years (default: 6.5)")
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).parent / "raw",
                    help="output dir (parquet path = root / funding_<symbol>.parquet)")
    ap.add_argument("--use-api", action="store_true",
                    help="Use fapi REST API instead of public data (may be geo-blocked)")
    args = ap.parse_args()

    asset = Asset(args.asset)
    out_path = funding_parquet_path(args.root, asset)
    now_ms = int(time.time() * 1000)
    target_start_ms = now_ms - int(args.years * 365 * 24 * 60 * 60 * 1000)
    print(f"Fetching funding for {asset.value} ({args.years} years) → {out_path}")

    if args.use_api:
        df = fetch_funding_to_parquet(
            asset=asset,
            target_start_ms=target_start_ms,
            end_ms=now_ms,
            out_path=out_path,
        )
    else:
        df = fetch_funding_from_public_data(
            asset=asset,
            target_start_ms=target_start_ms,
            end_ms=now_ms,
            out_path=out_path,
        )
    print(f"Done. {len(df)} funding events saved.")


if __name__ == "__main__":
    main()
