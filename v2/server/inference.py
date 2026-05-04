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


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


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

    def _fetch_recent_binance(self, limit: int = 520) -> list[dict]:
        end_ms = int(time.time() * 1000)
        params = {"symbol": "BTCUSDT", "interval": "1m", "limit": limit, "endTime": end_ms}
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

    @torch.no_grad()
    def predict_live(self, limit: int = 300) -> dict:
        candles = self._fetch_recent_binance(limit=max(limit, 520))
        if not candles:
            return {"candles": [], "prediction": None}
        chart_candles = candles[-limit:]
        if self.model is None or self.tokenizer is None:
            return {"candles": chart_candles, "prediction": None}
        block_size = self.model.cfg.block_size
        window_candles = candles[-block_size:]
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
            return {
                "candles": chart_candles,
                "prediction": {
                    "probs": probs.tolist(),
                    "top5_rets": top5_rets,
                    "top5_probs": top5_probs,
                },
            }
        except Exception as e:
            print(f"[inference] predict_live failed: {e}")
            return {"candles": chart_candles, "prediction": None}
