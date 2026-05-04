"""V2InferenceModel: wraps CandleGPTv2 for live single-step prediction."""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import requests
import time

from v2.model.model import CandleGPTv2
from v2.model.config import ModelConfig
from v2.model.tokenizer import ReturnTokenizerV2
from v2.data.dataset import KlineWindowDataset


BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "1d": 86_400_000,
}


class V2InferenceModel:
    def __init__(self) -> None:
        self.model: Optional[CandleGPTv2] = None
        self.tokenizer: Optional[ReturnTokenizerV2] = None
        self.device: str = "cpu"
        self.run_id: Optional[str] = None
        self.ckpt_step: Optional[int] = None
        self._last_mtime: float = 0.0

    def _select_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load(self, ckpt_path: Path, tokenizer_path: Path) -> bool:
        try:
            mtime = ckpt_path.stat().st_mtime
            if self.model is not None and mtime == self._last_mtime:
                return True
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg = ModelConfig.from_dict(ckpt["model_config"])
            device = self._select_device()
            model = CandleGPTv2(cfg).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            self.model = model
            self.tokenizer = ReturnTokenizerV2.load(tokenizer_path)
            self.device = device
            self.run_id = ckpt.get("run_id")
            self.ckpt_step = ckpt.get("step")
            self._last_mtime = mtime
            return True
        except Exception as e:
            print(f"[inference] load failed: {e}")
            return False

    def _fetch_recent_binance(self, limit: int = 520, interval: str = "1m") -> list[dict]:
        try:
            end_ms = int(time.time() * 1000)
            params = {"symbol": "BTCUSDT", "interval": interval, "limit": limit, "endTime": end_ms}
            r = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
            r.raise_for_status()
            out = []
            for row in r.json():
                out.append({
                    "time": int(row[0]) // 1000,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "open_time_ms": int(row[0]),
                })
            return sorted(out, key=lambda x: x["time"])
        except Exception as e:
            print(f"[inference] Binance fetch failed ({type(e).__name__}): {e}")
            return []

    @torch.no_grad()
    def predict_live(self, limit: int = 300, interval: str = "1m") -> dict:
        # We need at least block_size bars of history to feed the model.
        block_size = self.model.cfg.block_size if self.model is not None else 520
        fetch_limit = max(limit, block_size + 10)
        chart_raw = self._fetch_recent_binance(limit=fetch_limit, interval=interval)
        chart_candles = chart_raw[-limit:] if chart_raw else []
        if self.model is None or self.tokenizer is None:
            return {"candles": chart_candles, "prediction": None, "interval": interval}
        if not chart_raw:
            return {"candles": chart_candles, "prediction": None, "interval": interval}
        # Run the model on whatever interval the user chose. The model was
        # trained on 1m bars but the architecture is timeframe-agnostic; we
        # rely on the assumption that local return patterns transfer to
        # higher TFs. Quality may degrade — UI labels the prediction with
        # the actual interval so the user knows what they're looking at.
        window_candles = chart_raw[-block_size:]
        if len(window_candles) < 10:
            return {"candles": chart_candles, "prediction": None}
        try:
            import pandas as pd
            from v2.data.dataset import FEATURE_COLUMNS_WITH_JOIN
            from v2.features.engineer import compute_features
            n = len(window_candles)
            df_kline = pd.DataFrame({
                "open_time":  [int(c["open_time_ms"]) for c in window_candles],
                "open":       [float(c["open"]) for c in window_candles],
                "high":       [float(c["high"]) for c in window_candles],
                "low":        [float(c["low"]) for c in window_candles],
                "close":      [float(c["close"]) for c in window_candles],
                "volume":     [float(c["volume"]) for c in window_candles],
                "close_time": [int(c["open_time_ms"]) + 59_999 for c in window_candles],
                "regime":     pd.array([-1] * n, dtype="int8"),
            })
            df_kline["funding_rate"] = 0.0001
            df_kline["mark_price"] = df_kline["close"]
            df_kline["minutes_until_funding"] = 240.0
            for col in ["liq_count", "liq_sum_notional", "liq_max_single",
                        "long_liq_count", "long_liq_notional",
                        "short_liq_count", "short_liq_notional"]:
                df_kline[col] = 0.0
            df_kline = df_kline[list(FEATURE_COLUMNS_WITH_JOIN)]
            feats_df = compute_features(df_kline)
            feats = torch.from_numpy(feats_df.to_numpy(dtype=np.float32)).unsqueeze(0)
            feats = feats.to(self.device)
            logits = self.model(feats)
            probs = F.softmax(logits[0, -1, :], dim=-1).cpu().numpy()
            top5_idx = np.argsort(probs)[::-1][:5]
            top5_rets = self.tokenizer.decode(top5_idx).tolist()
            top5_probs = probs[top5_idx].tolist()
            # Bin centers for the full distribution (decoded return per bin).
            bin_centers = self.tokenizer.decode(np.arange(self.tokenizer.n_bins)).tolist()
            bin_centers_arr = np.asarray(bin_centers)
            # Directional summary: flat = |return| < 0.01% (1 bps).
            FLAT_EPS = 1e-4
            mask_up = bin_centers_arr > FLAT_EPS
            mask_down = bin_centers_arr < -FLAT_EPS
            mask_flat = ~(mask_up | mask_down)
            p_up = float(probs[mask_up].sum())
            p_down = float(probs[mask_down].sum())
            p_flat = float(probs[mask_flat].sum())
            expected_ret = float((probs * bin_centers_arr).sum())
            # Shannon entropy in bits and as a fraction of max (for "confidence").
            entropy_bits = float(-(probs * np.log2(np.clip(probs, 1e-12, 1.0))).sum())
            max_entropy_bits = float(np.log2(self.tokenizer.n_bins))
            confidence = max(0.0, 1.0 - entropy_bits / max_entropy_bits)  # 0..1
            last_close = float(window_candles[-1]["close"])
            expected_close = last_close * float(np.exp(expected_ret))

            # 30-step autoregressive rollout — one model forward pass per future
            # bar. We use the rollout to:
            #   (a) draw a 30-bar predicted path on the chart, and
            #   (b) score the BUY/HOLD/SELL decision off the *cumulative* 30-bar
            #       z-score rather than a single-step lean (since per-step
            #       expected returns are tiny on this model).
            interval_ms = INTERVAL_MS.get(interval, 60_000)
            last_open_ms = int(window_candles[-1]["open_time_ms"])
            HORIZON_BARS = 30
            predicted_path = []
            cumulative_log_ret = 0.0
            cumulative_variance = 0.0
            try:
                synth_volume = float(np.mean([float(c["volume"]) for c in window_candles[-10:]]))
                running_df = df_kline.copy()
                running_close = last_close
                step_probs = probs
                step_expected_ret = expected_ret
                step_variance = float((step_probs * (bin_centers_arr - step_expected_ret) ** 2).sum())
                for i in range(HORIZON_BARS):
                    if i > 0:
                        # Build a synthetic bar at the previous step's predicted close
                        # and append, then re-run features + model.
                        prev_open_ms = int(running_df.iloc[-1]["open_time"]) + interval_ms
                        synth_open = float(running_df.iloc[-1]["close"])
                        synth_close = running_close
                        synth_high = max(synth_open, synth_close)
                        synth_low = min(synth_open, synth_close)
                        synth_row = pd.DataFrame([{
                            "open_time": prev_open_ms,
                            "open": synth_open, "high": synth_high, "low": synth_low,
                            "close": synth_close, "volume": synth_volume,
                            "close_time": prev_open_ms + interval_ms - 1,
                            "regime": pd.array([-1], dtype="int8")[0],
                            "funding_rate": 0.0001,
                            "mark_price": synth_close,
                            "minutes_until_funding": 240.0,
                            "liq_count": 0.0, "liq_sum_notional": 0.0, "liq_max_single": 0.0,
                            "long_liq_count": 0.0, "long_liq_notional": 0.0,
                            "short_liq_count": 0.0, "short_liq_notional": 0.0,
                        }])[list(FEATURE_COLUMNS_WITH_JOIN)]
                        running_df = pd.concat([running_df, synth_row], ignore_index=True).iloc[-block_size:]
                        running_feats_df = compute_features(running_df)
                        running_feats = torch.from_numpy(running_feats_df.to_numpy(dtype=np.float32)).unsqueeze(0).to(self.device)
                        running_logits = self.model(running_feats)
                        step_probs = F.softmax(running_logits[0, -1, :], dim=-1).cpu().numpy()
                        step_expected_ret = float((step_probs * bin_centers_arr).sum())
                        step_variance = float((step_probs * (bin_centers_arr - step_expected_ret) ** 2).sum())

                    cumulative_log_ret += step_expected_ret
                    cumulative_variance += step_variance
                    new_close = running_close * float(np.exp(step_expected_ret))
                    new_open_ms = last_open_ms + (i + 1) * interval_ms
                    predicted_path.append({
                        "time": new_open_ms // 1000,
                        "close": new_close,
                        "ret_bps": step_expected_ret * 1e4,
                        "cumulative_ret_bps": cumulative_log_ret * 1e4,
                    })
                    running_close = new_close
            except Exception as e:
                print(f"[inference] {HORIZON_BARS}-step rollout failed at i={len(predicted_path)}: {e}")
                # Truncate the path to whatever finished and continue.

            cumulative_std = float(np.sqrt(max(cumulative_variance, 1e-24)))
            cumulative_z = cumulative_log_ret / cumulative_std
            cumulative_close = last_close * float(np.exp(cumulative_log_ret))
            return {
                "candles": chart_candles,
                "interval": interval,
                "prediction": {
                    "probs": probs.tolist(),
                    "top5_rets": top5_rets,
                    "top5_probs": top5_probs,
                    "bin_centers": bin_centers,
                    "p_up": p_up,
                    "p_down": p_down,
                    "p_flat": p_flat,
                    "flat_eps": FLAT_EPS,
                    "expected_ret": expected_ret,
                    "expected_close": expected_close,
                    "last_close": last_close,
                    "entropy_bits": entropy_bits,
                    "max_entropy_bits": max_entropy_bits,
                    "confidence": confidence,
                    "predicted_path": predicted_path,
                    "horizon_bars": HORIZON_BARS,
                    "horizon_cumulative_ret": cumulative_log_ret,
                    "horizon_cumulative_close": cumulative_close,
                    "horizon_cumulative_std": cumulative_std,
                    "horizon_cumulative_z": cumulative_z,
                },
            }
        except Exception as e:
            print(f"[inference] predict_live failed: {e}")
            return {"candles": chart_candles, "prediction": None, "interval": interval}
