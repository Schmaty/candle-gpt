"""Tardis.dev liquidations backfill stub.

We don't have a Tardis subscription as of plan-write time. This module exposes:

  - write_empty_backfill(out_path): writes a valid parquet with zero rows but
    the canonical LIQ_BUCKETED_COLUMNS + schema metadata. The dataloader can
    join against this today and downstream code (regime, model) is written
    against real column names. When the real fill drops in later, no schema
    migration is required.

  - fetch_tardis_backfill(...): raises NotImplementedError. Wire this up once
    a Tardis API key is available; signature is committed.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from v2.data.constants import LIQ_BUCKETED_COLUMNS, LIQ_BUCKETED_DTYPES
from v2.data.store import write_liq_bucketed


def write_empty_backfill(out_path: Path) -> None:
    """Write a zero-row parquet matching LIQ_BUCKETED_COLUMNS, with metadata."""
    df = pd.DataFrame({
        c: pd.Series(dtype=LIQ_BUCKETED_DTYPES[c]) for c in LIQ_BUCKETED_COLUMNS
    })
    write_liq_bucketed(df, out_path)


def fetch_tardis_backfill(
    *,
    start_ms: int,
    end_ms: int,
    out_path: Path,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Pull `liquidations` channel from Tardis (binance-futures BTCUSDT), roll up, write.

    Pending Tardis subscription. When implemented:
      1. Request `liquidations` channel for `binance-futures BTCUSDT` over the window.
      2. Map each event {timestamp, side="sell"|"buy", price, amount} →
         {event_time, side="long" if "sell" else "short", price, qty, notional}.
      3. Pass to rollup.roll_to_per_minute.
      4. Persist via store.write_liq_bucketed(out_path).
    """
    raise NotImplementedError(
        "Tardis backfill not yet wired up. Use write_empty_backfill() to scaffold an "
        "empty parquet against the canonical schema; the dataloader and regime tagger "
        "tolerate empty liq parquets (zero counts everywhere)."
    )
