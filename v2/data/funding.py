"""Binance perpetual funding-rate fetcher.

GET /fapi/v1/fundingRate — paginated, max 1000 rows per request, walks forward
in time. Stores to parquet via store.write_funding (which embeds schema metadata).
"""
from __future__ import annotations
import argparse
import time
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
    args = ap.parse_args()

    asset = Asset(args.asset)
    out_path = funding_parquet_path(args.root, asset)
    now_ms = int(time.time() * 1000)
    target_start_ms = now_ms - int(args.years * 365 * 24 * 60 * 60 * 1000)
    print(f"Fetching funding for {asset.value} ({args.years} years) → {out_path}")
    df = fetch_funding_to_parquet(
        asset=asset,
        target_start_ms=target_start_ms,
        end_ms=now_ms,
        out_path=out_path,
    )
    print(f"Done. {len(df)} funding events saved.")


if __name__ == "__main__":
    main()
