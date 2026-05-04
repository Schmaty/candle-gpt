"""CLI: compute regime labels and write them back into the kline parquet, in place.

Usage:
    python -m v2.data.tag_regimes --asset BTC --timeframe 1m

Requires the funding and liq-bucketed parquets to exist at the canonical paths.
The liq parquet may be empty (Tardis-stub state) — that's fine; high_vol_squeeze
just won't trigger via liq spikes until a real fill drops in.

Atomicity: write_klines (in store.py) writes to <path>.tmp then rename()s. If the
process dies mid-write, the original parquet is intact.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from v2.data.constants import Asset, Timeframe
from v2.data.regime import compute_regimes
from v2.data.store import (
    parquet_path, funding_parquet_path, liq_bucketed_parquet_path,
    read_klines, read_funding, read_liq_bucketed, write_klines,
)


def tag_kline_parquet(*, asset: Asset, timeframe: Timeframe, root: Path) -> int:
    """Read the three parquets, compute regime, write klines back. Returns row count."""
    kp = parquet_path(root, asset, timeframe)
    fp = funding_parquet_path(root, asset)
    lp = liq_bucketed_parquet_path(root, asset)

    klines = read_klines(kp)
    funding = read_funding(fp)
    liq = read_liq_bucketed(lp)

    regimes = compute_regimes(klines, funding, liq)
    klines = klines.copy()
    klines["regime"] = regimes.astype("int8").to_numpy()
    write_klines(klines, kp)  # write_klines does atomic temp+rename internally
    return len(klines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Tag regime labels into a kline parquet in place.")
    ap.add_argument("--asset", required=True, choices=[a.value for a in Asset])
    ap.add_argument("--timeframe", required=True, choices=[t.value for t in Timeframe])
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).parent / "raw",
                    help="parquet root dir (default: v2/data/raw)")
    args = ap.parse_args()
    n = tag_kline_parquet(
        asset=Asset(args.asset),
        timeframe=Timeframe(args.timeframe),
        root=args.root,
    )
    print(f"Tagged {n:,} bars in {parquet_path(args.root, Asset(args.asset), Timeframe(args.timeframe))}")


if __name__ == "__main__":
    main()
