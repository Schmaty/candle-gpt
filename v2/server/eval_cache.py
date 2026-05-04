"""EvalCache: loads REPORT.md metrics and computes equity curve at server startup."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import numpy as np


class EvalCache:
    def __init__(self) -> None:
        self.history: list[dict] = []
        self.calibration: dict = {"buckets": [], "ece": 0.0}
        self.regimes: list[dict] = []
        self.equity: dict = {"equity": [], "sharpe": 0.0, "max_dd": 0.0}
        self.loaded = False

    def load_from_report(self, report_path: Path, metrics_json_path: Optional[Path] = None) -> None:
        if metrics_json_path and metrics_json_path.exists():
            with open(metrics_json_path) as f:
                metrics = json.load(f)
        elif report_path.exists():
            metrics = self._parse_report_md(report_path)
        else:
            return
        self.calibration = {
            "buckets": metrics.get("calibration_buckets", []),
            "ece": metrics.get("ece", 0.0),
        }
        per_regime = metrics.get("per_regime_accuracy", {})
        regime_names = {"-1": "Untagged", "0": "Sideways", "1": "Trending", "2": "Volatile"}
        self.regimes = [
            {"id": int(r_id), "name": regime_names.get(r_id, f"Regime {r_id}"),
             "accuracy": stats["accuracy"], "n": stats["n"]}
            for r_id, stats in sorted(per_regime.items())
        ]
        self._build_history(metrics)
        self.loaded = True

    def _build_history(self, metrics: dict) -> None:
        samples = metrics.get("sample_predictions", [])
        self.history = []
        flat_idx = 0
        for win in samples:
            for pos_idx, (pred_ret, true_ret) in enumerate(
                zip(win.get("pred_rets", []), win.get("true_rets", []))
            ):
                self.history.append({
                    "idx": flat_idx,
                    "pred_ret": pred_ret,
                    "true_ret": true_ret,
                    "correct": win["pred_ids"][pos_idx] == win["true_ids"][pos_idx],
                    "confidence": 0.0,
                    "regime": 0,
                })
                flat_idx += 1
        equity = [{"idx": 0, "cumret": 0.0, "position": 0}]
        cumret = 0.0
        thresh = 0.0002
        for item in self.history:
            pos = 1 if item["pred_ret"] > thresh else (-1 if item["pred_ret"] < -thresh else 0)
            pnl = pos * item["true_ret"]
            cumret += pnl
            equity.append({"idx": item["idx"] + 1, "cumret": cumret, "position": pos})
        if len(equity) > 1:
            rets = [e["cumret"] - equity[i]["cumret"] for i, e in enumerate(equity[1:])]
            sr = (np.mean(rets) / (np.std(rets) + 1e-8)) * np.sqrt(525_600)
            peak = 0.0
            dd = 0.0
            for e in equity:
                peak = max(peak, e["cumret"])
                dd = min(dd, e["cumret"] - peak)
            self.equity = {"equity": equity, "sharpe": float(sr), "max_dd": float(dd)}
        else:
            self.equity = {"equity": equity, "sharpe": 0.0, "max_dd": 0.0}

    def _parse_report_md(self, path: Path) -> dict:
        return {"calibration_buckets": [], "ece": 0.0,
                "per_regime_accuracy": {}, "sample_predictions": []}
