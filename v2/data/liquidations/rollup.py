"""Per-event → per-minute liquidation rollup.

Per-event schema (input from live collector / Tardis):
    event_time : int64 (ms epoch)
    side       : str ("long" | "short")  long = long position liquidated (market sell)
    price      : float64
    qty        : float64
    notional   : float64 (price * qty)

Per-minute schema (output, joined by dataloader): see LIQ_BUCKETED_COLUMNS.
"""
from __future__ import annotations
import pandas as pd

from v2.data.constants import LIQ_BUCKETED_COLUMNS, LIQ_BUCKETED_DTYPES


EVENT_COLUMNS: tuple[str, ...] = ("event_time", "side", "price", "qty", "notional")
EVENT_DTYPES: dict[str, str] = {
    "event_time": "int64",
    "side":       "object",
    "price":      "float64",
    "qty":        "float64",
    "notional":   "float64",
}

_MINUTE_MS = 60_000
_VALID_SIDES = frozenset({"long", "short"})


def _empty_bucketed() -> pd.DataFrame:
    return pd.DataFrame({
        c: pd.Series(dtype=LIQ_BUCKETED_DTYPES[c]) for c in LIQ_BUCKETED_COLUMNS
    })


def roll_to_per_minute(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-event rows into per-minute buckets matching LIQ_BUCKETED_COLUMNS."""
    if events.empty:
        return _empty_bucketed()

    bad = ~events["side"].isin(_VALID_SIDES)
    if bad.any():
        offenders = events.loc[bad, "side"].unique().tolist()
        raise ValueError(f"unknown side(s): {offenders}; expected one of {_VALID_SIDES}")

    df = events.copy()
    df["bucket_time"] = (df["event_time"] // _MINUTE_MS) * _MINUTE_MS
    df["is_long"] = df["side"] == "long"
    df["is_short"] = df["side"] == "short"
    df["long_notional"] = df["notional"].where(df["is_long"], 0.0)
    df["short_notional"] = df["notional"].where(df["is_short"], 0.0)

    grouped = df.groupby("bucket_time", sort=True).agg(
        count=("event_time", "count"),
        sum_notional=("notional", "sum"),
        max_single=("notional", "max"),
        long_liq_count=("is_long", "sum"),
        long_liq_notional=("long_notional", "sum"),
        short_liq_count=("is_short", "sum"),
        short_liq_notional=("short_notional", "sum"),
    ).reset_index()

    # Coerce to the canonical dtypes; groupby may produce int64 for sums of bool, etc.
    for col, dtype in LIQ_BUCKETED_DTYPES.items():
        grouped[col] = grouped[col].astype(dtype)
    return grouped[list(LIQ_BUCKETED_COLUMNS)]
