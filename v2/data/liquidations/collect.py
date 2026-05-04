"""Live liquidation collector for Binance USDT-M futures.

Subscribes to the `forceOrder@arr` stream (all symbols), filters for the
target symbol (BTCUSDT by default), parses events into the per-event schema,
and appends to a dated parquet at `<root>/YYYY-MM-DD.parquet` (UTC).

Restart-safe: appending re-reads the existing dated file (if any) and rewrites
atomically. The script is meant to run as a long-lived background process;
this plan does NOT start it. Wire up via systemd / launchd separately.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import signal
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from v2.data.liquidations.rollup import EVENT_COLUMNS, EVENT_DTYPES


WS_URL = "wss://fstream.binance.com/ws/!forceOrder@arr"


def parse_force_order_event(payload: dict, expected_symbol: str = "BTCUSDT") -> dict:
    """Map a Binance forceOrder payload to the per-event schema."""
    o = payload["o"]
    if o["s"] != expected_symbol:
        raise ValueError(f"unexpected symbol: {o['s']!r}, want {expected_symbol!r}")
    side = "long" if o["S"] == "SELL" else "short" if o["S"] == "BUY" else None
    if side is None:
        raise ValueError(f"unexpected order side: {o['S']!r}")
    qty = float(o["q"])
    price = float(o["ap"])  # use average filled price, not the limit `p`
    return {
        "event_time": int(o["T"]),
        "side": side,
        "price": price,
        "qty": qty,
        "notional": price * qty,
    }


def daily_parquet_path(root: Path, event_time_ms: int) -> Path:
    dt = datetime.fromtimestamp(event_time_ms / 1000, tz=timezone.utc)
    return root / f"{dt.strftime('%Y-%m-%d')}.parquet"


def append_events(path: Path, events: list[dict]) -> None:
    if not events:
        return
    new_df = pd.DataFrame(events, columns=list(EVENT_COLUMNS))
    for col, dtype in EVENT_DTYPES.items():
        new_df[col] = new_df[col].astype(dtype)
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, new_df], ignore_index=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = new_df
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


async def _run(root: Path, expected_symbol: str) -> None:  # pragma: no cover - I/O
    import websockets  # local import keeps test imports fast

    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=20) as ws:
                buffer: list[dict] = []
                last_flush_ms = 0
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("e") != "forceOrder":
                        continue
                    if msg["o"]["s"] != expected_symbol:
                        continue
                    try:
                        ev = parse_force_order_event(msg, expected_symbol=expected_symbol)
                    except ValueError:
                        continue
                    buffer.append(ev)
                    # Flush every ~5 seconds OR when buffer grows.
                    if len(buffer) >= 16 or ev["event_time"] - last_flush_ms > 5000:
                        path = daily_parquet_path(root, ev["event_time"])
                        append_events(path, buffer)
                        buffer.clear()
                        last_flush_ms = ev["event_time"]
        except Exception as e:
            print(f"[collect] reconnecting after error: {e}")
            await asyncio.sleep(5)


def main() -> None:
    ap = argparse.ArgumentParser(description="Live Binance forceOrder collector.")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).resolve().parents[2] / "data" / "raw" / "liquidations")
    args = ap.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    print(f"[collect] subscribing to {WS_URL} for {args.symbol}; writing to {args.root}")
    loop = asyncio.new_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, loop.stop)
    try:
        loop.run_until_complete(_run(args.root, args.symbol))
    finally:
        loop.close()


if __name__ == "__main__":
    main()
