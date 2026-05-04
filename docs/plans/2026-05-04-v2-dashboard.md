# Candle-GPT v2 — Dashboard: v2 Server + React Frontend (Plan 5 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **UI SUB-SKILL REQUIRED:** Before implementing ANY frontend component (Tasks 4–6), invoke `frontend-design:frontend-design` via the Skill tool. The UI must be intentional, dark-themed, and not generic-AI-looking. The skill guides typography, spacing, and component design decisions.

**Goal:** Ship a v2 dashboard as a separate FastAPI server on port 8766 (alongside v1's 8765) serving a new React+TS frontend with five tabs: Live prediction, History, Calibration, Regimes, and Equity curve. The v1 server is not modified. The v2 server loads the best-checkpoint model from Plan 4, serves live next-candle predictions, and pre-computes eval/calibration/regime data at startup from the REPORT.md and test dataset.

**Architecture:**
- `v2/server/main.py` — FastAPI app on 0.0.0.0:8766; loads model + tokenizer at startup; serves pre-computed eval data + live inference.
- `v2/server/inference.py` — `V2InferenceModel`: wraps `CandleGPTv2` for live prediction.
- `v2/server/eval_cache.py` — `EvalCache`: loads REPORT.md metrics + builds equity curve at server startup.
- `v2/web/` — Vite + React 19 + TypeScript frontend. Dark theme (#0b0e13 bg, #e6ebf3 fg, #00d4aa accent teal) matching v1. Charting via lightweight-charts 5 (same lib as v1). Build output goes to `v2/web/dist/`; FastAPI mounts it at `/`.

**Tech Stack:** FastAPI + uvicorn (already in pyproject.toml), React 19 + TypeScript + Vite + lightweight-charts 5, vanilla CSS with CSS variables (same as v1).

**Ports:** v1 on 8765, v2 on 8766. Both can run simultaneously.

---

## File Structure

**Create (server):**
- `projects/candle-gpt/v2/server/__init__.py` — empty marker
- `projects/candle-gpt/v2/server/inference.py` — `V2InferenceModel`
- `projects/candle-gpt/v2/server/eval_cache.py` — `EvalCache`
- `projects/candle-gpt/v2/server/main.py` — FastAPI app

**Create (frontend):**
- `projects/candle-gpt/v2/web/package.json`
- `projects/candle-gpt/v2/web/tsconfig.json`
- `projects/candle-gpt/v2/web/vite.config.ts`
- `projects/candle-gpt/v2/web/index.html`
- `projects/candle-gpt/v2/web/src/main.tsx`
- `projects/candle-gpt/v2/web/src/App.tsx`
- `projects/candle-gpt/v2/web/src/styles/globals.css`
- `projects/candle-gpt/v2/web/src/api.ts`
- `projects/candle-gpt/v2/web/src/pages/LivePage.tsx`
- `projects/candle-gpt/v2/web/src/pages/HistoryPage.tsx`
- `projects/candle-gpt/v2/web/src/pages/CalibrationPage.tsx`
- `projects/candle-gpt/v2/web/src/pages/RegimePage.tsx`
- `projects/candle-gpt/v2/web/src/pages/EquityPage.tsx`
- `projects/candle-gpt/v2/web/src/components/TabBar.tsx`
- `projects/candle-gpt/v2/web/src/components/Chart.tsx`

**Not touched:** v1 `server/main.py`, `web/`, `model/`, `data/`.

---

## API Contract (v2/server/main.py endpoints)

All endpoints are under `/api/v2/`:

```
GET  /api/v2/status
  → { model_loaded: bool, run_id: str|null, ckpt_step: int|null,
      n_params: int|null, device: str }

GET  /api/v2/candles?limit=300
  → { candles: [{time: int, open, high, low, close, volume}], prediction: {
       probs: float[256],   # softmax over n_bins at last position
       top5_rets: float[5], # decoded returns for top-5 bins
       top5_probs: float[5]
     } | null }

GET  /api/v2/history
  → { windows: [{idx: int, pred_ret: float, true_ret: float, correct: bool,
                  confidence: float, regime: int}] }
  (first 1000 test windows, precomputed at startup)

GET  /api/v2/calibration
  → { buckets: [{lo: float, hi: float, conf: float, acc: float, frac: float}],
      ece: float }

GET  /api/v2/regimes
  → { regimes: [{id: int, name: str, accuracy: float, n: int}] }

GET  /api/v2/equity
  → { equity: [{idx: int, cumret: float, position: int}],
      sharpe: float, max_dd: float }
  (long/short backtest on test windows using model probabilities)
```

---

## Task 1: v2 Server — `inference.py`, `eval_cache.py`, `main.py`

**Files:**
- Create: `v2/server/__init__.py`, `v2/server/inference.py`, `v2/server/eval_cache.py`, `v2/server/main.py`

- [ ] **Step 1: Create server package and `inference.py`**

Path: `v2/server/__init__.py` — empty.

Path: `v2/server/inference.py`:

```python
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
        """Fetch recent BTCUSDT 1m candles from Binance public API."""
        end_ms = int(time.time() * 1000)
        params = {"symbol": "BTCUSDT", "interval": "1m", "limit": limit, "endTime": end_ms}
        r = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
        r.raise_for_status()
        out = []
        for row in r.json():
            out.append({
                "time": int(row[0]) // 1000,  # seconds for lightweight-charts
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
        """Fetch recent candles and run model on the last window. Returns candles + prediction."""
        candles = self._fetch_recent_binance(limit=max(limit, 520))
        if not candles:
            return {"candles": [], "prediction": None}

        # Return last `limit` candles for the chart
        chart_candles = candles[-limit:]

        if self.model is None or self.tokenizer is None:
            return {"candles": chart_candles, "prediction": None}

        # Build feature tensor from last window=512 bars
        block_size = self.model.cfg.block_size
        window_candles = candles[-block_size:]
        if len(window_candles) < 10:
            return {"candles": chart_candles, "prediction": None}

        try:
            import pandas as pd
            from v2.data.dataset import _join_features, FEATURE_COLUMNS_WITH_JOIN
            from v2.features.engineer import compute_features
            from v2.data.constants import KLINE_COLUMNS, KLINE_DTYPES

            # Build minimal kline DataFrame from Binance candles
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
            # Inject zero funding + liq (no live join available)
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

            logits = self.model(feats)  # (1, T, n_bins)
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
```

- [ ] **Step 2: Create `eval_cache.py`**

Path: `v2/server/eval_cache.py`:

```python
"""EvalCache: loads REPORT.md metrics and computes equity curve at server startup."""
from __future__ import annotations
import json
import re
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
        """Load evaluation metrics. Tries JSON sidecar first, falls back to parsing REPORT.md."""
        if metrics_json_path and metrics_json_path.exists():
            with open(metrics_json_path) as f:
                metrics = json.load(f)
        elif report_path.exists():
            metrics = self._parse_report_md(report_path)
        else:
            return

        # Calibration
        self.calibration = {
            "buckets": metrics.get("calibration_buckets", []),
            "ece": metrics.get("ece", 0.0),
        }

        # Per-regime
        per_regime = metrics.get("per_regime_accuracy", {})
        regime_names = {"-1": "Untagged", "0": "Sideways", "1": "Trending", "2": "Volatile"}
        self.regimes = [
            {
                "id": int(r_id),
                "name": regime_names.get(r_id, f"Regime {r_id}"),
                "accuracy": stats["accuracy"],
                "n": stats["n"],
            }
            for r_id, stats in sorted(per_regime.items())
        ]

        # Build synthetic history + equity from sample predictions if available
        self._build_history(metrics)
        self.loaded = True

    def _build_history(self, metrics: dict) -> None:
        """Build a history list from sample predictions (limited to what REPORT.md provides)."""
        samples = metrics.get("sample_predictions", [])
        self.history = []
        for win_idx, win in enumerate(samples):
            for pos_idx, (pred_ret, true_ret) in enumerate(
                zip(win.get("pred_rets", []), win.get("true_rets", []))
            ):
                self.history.append({
                    "idx": win_idx * 10 + pos_idx,
                    "pred_ret": pred_ret,
                    "true_ret": true_ret,
                    "correct": win["pred_ids"][pos_idx] == win["true_ids"][pos_idx],
                    "confidence": 0.0,  # not available from REPORT.md alone
                    "regime": 0,
                })

        # Simple long/short equity on history
        equity = [{"idx": 0, "cumret": 0.0, "position": 0}]
        cumret = 0.0
        thresh = 0.0002  # ~0.02% threshold for taking a position
        for item in self.history:
            pos = 1 if item["pred_ret"] > thresh else (-1 if item["pred_ret"] < -thresh else 0)
            pnl = pos * item["true_ret"]
            cumret += pnl
            equity.append({"idx": item["idx"] + 1, "cumret": cumret, "position": pos})

        if len(equity) > 1:
            rets = [e["cumret"] - equity[i]["cumret"]
                    for i, e in enumerate(equity[1:])]
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
        """Minimal parser to extract JSON-like data from REPORT.md.
        Returns a metrics dict with whatever can be extracted."""
        # REPORT.md is human-readable markdown; we just surface what we have.
        # In production, write_report() should also save a .json sidecar.
        return {
            "calibration_buckets": [],
            "ece": 0.0,
            "per_regime_accuracy": {},
            "sample_predictions": [],
        }
```

- [ ] **Step 3: Create `main.py`**

Path: `v2/server/main.py`:

```python
"""FastAPI v2 server — port 8766.

Runs alongside v1 (port 8765) without conflict. v1 code is NOT imported.
Loads best_val.pt from the most recent training run at startup.
"""
from __future__ import annotations
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from v2.server.inference import V2InferenceModel
from v2.server.eval_cache import EvalCache

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]  # projects/candle-gpt/
RUNS_DIR = PROJECT_ROOT / "v2" / "runs"
WEB_DIST = HERE.parent / "web" / "dist"

inference = V2InferenceModel()
cache = EvalCache()


def _find_best_run() -> tuple[Path | None, Path | None]:
    """Find the most recently trained run with a best_val.pt checkpoint."""
    current_id_file = RUNS_DIR / "current_run_id.txt"
    if current_id_file.exists():
        run_id = current_id_file.read_text().strip()
        ckpt = RUNS_DIR / run_id / "checkpoints" / "best_val.pt"
        tok = RUNS_DIR / run_id / "tokenizer.pkl"
        if ckpt.exists() and tok.exists():
            return ckpt, tok
    # Fall back: find most recent run dir with best_val.pt
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        ckpt = run_dir / "checkpoints" / "best_val.pt"
        tok = run_dir / "tokenizer.pkl"
        if ckpt.exists() and tok.exists():
            return ckpt, tok
    return None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt_path, tok_path = _find_best_run()
    if ckpt_path:
        print(f"[v2 server] Loading model from {ckpt_path}")
        inference.load(ckpt_path, tok_path)
        # Load eval cache
        run_dir = ckpt_path.parents[1]
        metrics_json = run_dir / "metrics.json"
        report_md = run_dir / "REPORT.md"
        cache.load_from_report(report_md, metrics_json)
        print(f"[v2 server] Model loaded. Device={inference.device}, "
              f"step={inference.ckpt_step}")
    else:
        print("[v2 server] No trained checkpoint found. Serving without model.")
    yield


app = FastAPI(lifespan=lifespan, title="candle-gpt v2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/v2/status")
def status():
    return {
        "model_loaded": inference.model is not None,
        "run_id": inference.run_id,
        "ckpt_step": inference.ckpt_step,
        "n_params": inference.model.num_params() if inference.model else None,
        "device": inference.device,
    }


@app.get("/api/v2/candles")
def candles(limit: int = 300):
    limit = max(50, min(int(limit), 520))
    try:
        result = inference.predict_live(limit=limit)
    except Exception as e:
        raise HTTPException(502, f"predict_live failed: {e}")
    return result


@app.get("/api/v2/history")
def history(limit: int = 1000):
    return {"windows": cache.history[:int(limit)]}


@app.get("/api/v2/calibration")
def calibration():
    return cache.calibration


@app.get("/api/v2/regimes")
def regimes():
    return {"regimes": cache.regimes}


@app.get("/api/v2/equity")
def equity():
    return cache.equity


# Static frontend
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(WEB_DIST / "index.html")

    @app.get("/{path:path}")
    def spa_fallback(path: str):
        f = WEB_DIST / path
        if f.exists() and f.is_file():
            return FileResponse(f)
        return FileResponse(WEB_DIST / "index.html")
else:
    @app.get("/")
    def root_no_frontend():
        return {"status": "ok", "note": "frontend not built yet; run `npm run build` in v2/web/"}
```

- [ ] **Step 4: Verify server imports without error**

```bash
cd projects/candle-gpt
uv run python -c "from v2.server.main import app; print('server ok')"
```

Expected: `server ok`

- [ ] **Step 5: Save metrics JSON sidecar from eval (update eval.py)**

The eval script must also write `metrics.json` alongside `REPORT.md` for the server to consume. Add to `v2/train/eval.py` at the end of `write_report()`:

```python
# Also write a machine-readable JSON sidecar
import json as _json
json_path = run_dir / "metrics.json"
json_path.write_text(_json.dumps(metrics, indent=2), encoding="utf-8")
log.info(f"Metrics JSON: {json_path}")
```

Then re-run: `uv run pytest v2/tests/test_train_smoke.py -v` to confirm the change doesn't break the smoke test.

- [ ] **Step 6: Commit server**

```bash
git add v2/server/__init__.py v2/server/inference.py v2/server/eval_cache.py v2/server/main.py
git commit -m "v2: server — FastAPI on port 8766, inference + eval cache"
```

---

## Task 2: Update `write_report` to emit metrics.json sidecar

**Files:**
- Modify: `v2/train/eval.py` (add JSON sidecar write at end of `write_report`)

- [ ] **Step 1: Edit `write_report` in `v2/train/eval.py`**

At the end of the `write_report` function, after the line `log.info(f"Report written: {report_path}")`, add:

```python
    # Write machine-readable JSON sidecar for the v2 server's EvalCache
    json_path = run_dir / "metrics.json"
    json_path.write_text(
        __import__("json").dumps(metrics, indent=2), encoding="utf-8"
    )
    log.info(f"Metrics JSON: {json_path}")
```

- [ ] **Step 2: Re-run smoke test**

```bash
uv run pytest v2/tests/test_train_smoke.py -v
```

Expected: still passes.

- [ ] **Step 3: Commit**

```bash
git add v2/train/eval.py
git commit -m "v2: eval — write metrics.json sidecar for server EvalCache"
```

---

## Task 3: Frontend scaffold — package.json, Vite config, global CSS

> **REMINDER:** Before writing any component code, invoke the `frontend-design:frontend-design` skill via the Skill tool.

**Files:**
- Create: `v2/web/package.json`, `v2/web/tsconfig.json`, `v2/web/vite.config.ts`, `v2/web/index.html`, `v2/web/src/main.tsx`, `v2/web/src/styles/globals.css`

- [ ] **Step 1: Create `v2/web/package.json`**

```json
{
  "name": "candle-gpt-v2-web",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "vite --host 0.0.0.0 --port 5174",
    "build": "tsc -b && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "lightweight-charts": "^5.0.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "react-router-dom": "^7.0.0"
  },
  "devDependencies": {
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@vitejs/plugin-react": "^4.3.4",
    "typescript": "^5.7.2",
    "vite": "^6.0.5"
  }
}
```

- [ ] **Step 2: Create `v2/web/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": false,
    "noUnusedParameters": false,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"]
}
```

- [ ] **Step 3: Create `v2/web/vite.config.ts`**

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8766',
    },
  },
})
```

- [ ] **Step 4: Create `v2/web/index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Candle-GPT v2</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 5: Create global CSS `v2/web/src/styles/globals.css`**

The dark palette must match v1 exactly: bg `#0b0e13`, fg `#e6ebf3`, accent `#00d4aa`.

```css
:root {
  --bg: #0b0e13;
  --bg-surface: #141820;
  --bg-elevated: #1c2230;
  --fg: #e6ebf3;
  --fg-dim: #8492a6;
  --accent: #00d4aa;
  --accent-dim: rgba(0, 212, 170, 0.15);
  --red: #f05252;
  --green: #00d4aa;
  --border: #252d3d;
  --font-mono: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
  --font-sans: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
  --radius: 6px;
  --transition: 150ms ease;
}

*, *::before, *::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

html, body, #root {
  height: 100%;
  background: var(--bg);
  color: var(--fg);
  font-family: var(--font-sans);
  font-size: 14px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

button {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--fg);
  border-radius: var(--radius);
  padding: 6px 14px;
  cursor: pointer;
  font-size: 13px;
  transition: border-color var(--transition), background var(--transition);
}
button:hover { border-color: var(--accent); color: var(--accent); }
button.active {
  background: var(--accent-dim);
  border-color: var(--accent);
  color: var(--accent);
}

table { border-collapse: collapse; width: 100%; }
th, td {
  padding: 8px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
}
th { color: var(--fg-dim); font-weight: 500; }

.card {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
}

.metric-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}

.metric {
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 16px;
  min-width: 140px;
}
.metric .label { color: var(--fg-dim); font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }
.metric .value { font-size: 20px; font-weight: 600; color: var(--accent); margin-top: 2px; font-family: var(--font-mono); }
```

- [ ] **Step 6: Create `v2/web/src/main.tsx`**

```tsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './styles/globals.css'
import App from './App'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
```

- [ ] **Step 7: Install npm dependencies**

```bash
cd projects/candle-gpt/v2/web && npm install
```

Expected: `node_modules/` created, no errors.

- [ ] **Step 8: Commit**

```bash
cd projects/candle-gpt
git add v2/web/package.json v2/web/tsconfig.json v2/web/vite.config.ts \
        v2/web/index.html v2/web/src/main.tsx v2/web/src/styles/globals.css
git commit -m "v2: web scaffold — Vite+React+TS, dark theme CSS"
```

---

## Task 4: `api.ts`, `TabBar.tsx`, `App.tsx`

> **INVOKE frontend-design skill before implementing any component.**

- [ ] **Step 1: Invoke frontend-design skill**

Use the Skill tool to invoke `frontend-design:frontend-design`. Brief it on:
- 5-tab dark dashboard (Live, History, Calibration, Regimes, Equity)
- Palette: `#0b0e13` bg, `#e6ebf3` fg, `#00d4aa` accent
- Charting: lightweight-charts 5 for candles and line charts
- No Tailwind — vanilla CSS with the globals.css variables defined above
- Must NOT look generic; should feel like a focused trading tool

Follow the skill's guidance for the layout, typography, and component decisions.

- [ ] **Step 2: Create `v2/web/src/api.ts`**

```ts
const BASE = '/api/v2'

export async function fetchStatus() {
  const r = await fetch(`${BASE}/status`)
  if (!r.ok) throw new Error(`status: ${r.status}`)
  return r.json()
}

export async function fetchCandles(limit = 300) {
  const r = await fetch(`${BASE}/candles?limit=${limit}`)
  if (!r.ok) throw new Error(`candles: ${r.status}`)
  return r.json()
}

export async function fetchHistory(limit = 500) {
  const r = await fetch(`${BASE}/history?limit=${limit}`)
  if (!r.ok) throw new Error(`history: ${r.status}`)
  return r.json()
}

export async function fetchCalibration() {
  const r = await fetch(`${BASE}/calibration`)
  if (!r.ok) throw new Error(`calibration: ${r.status}`)
  return r.json()
}

export async function fetchRegimes() {
  const r = await fetch(`${BASE}/regimes`)
  if (!r.ok) throw new Error(`regimes: ${r.status}`)
  return r.json()
}

export async function fetchEquity() {
  const r = await fetch(`${BASE}/equity`)
  if (!r.ok) throw new Error(`equity: ${r.status}`)
  return r.json()
}
```

- [ ] **Step 3: Create `v2/web/src/components/TabBar.tsx`**

```tsx
import { type ReactNode } from 'react'

interface Tab {
  id: string
  label: string
}

interface TabBarProps {
  tabs: Tab[]
  active: string
  onSelect: (id: string) => void
}

export function TabBar({ tabs, active, onSelect }: TabBarProps) {
  return (
    <nav style={{
      display: 'flex',
      gap: '2px',
      padding: '8px 16px',
      borderBottom: '1px solid var(--border)',
      background: 'var(--bg-surface)',
    }}>
      {tabs.map(tab => (
        <button
          key={tab.id}
          className={active === tab.id ? 'active' : ''}
          onClick={() => onSelect(tab.id)}
          style={{ fontSize: '13px', padding: '6px 16px' }}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  )
}
```

- [ ] **Step 4: Create `v2/web/src/App.tsx`**

```tsx
import { useState, useEffect } from 'react'
import { TabBar } from './components/TabBar'
import { LivePage } from './pages/LivePage'
import { HistoryPage } from './pages/HistoryPage'
import { CalibrationPage } from './pages/CalibrationPage'
import { RegimePage } from './pages/RegimePage'
import { EquityPage } from './pages/EquityPage'
import { fetchStatus } from './api'

const TABS = [
  { id: 'live',        label: 'Live' },
  { id: 'history',     label: 'History' },
  { id: 'calibration', label: 'Calibration' },
  { id: 'regimes',     label: 'Regimes' },
  { id: 'equity',      label: 'Equity' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('live')
  const [status, setStatus] = useState<any>(null)

  useEffect(() => {
    fetchStatus().then(setStatus).catch(console.error)
  }, [])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Header */}
      <header style={{
        padding: '12px 16px',
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        borderBottom: '1px solid var(--border)',
        background: 'var(--bg-surface)',
      }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent)', fontSize: 16 }}>
          candle-gpt v2
        </span>
        {status && (
          <span style={{ color: 'var(--fg-dim)', fontSize: 12 }}>
            {status.model_loaded
              ? `model loaded · step ${status.ckpt_step?.toLocaleString()} · ${status.n_params?.toLocaleString()} params · ${status.device}`
              : 'no model loaded'}
          </span>
        )}
      </header>

      <TabBar tabs={TABS} active={activeTab} onSelect={setActiveTab} />

      <main style={{ flex: 1, overflow: 'auto', padding: '16px' }}>
        {activeTab === 'live'        && <LivePage />}
        {activeTab === 'history'     && <HistoryPage />}
        {activeTab === 'calibration' && <CalibrationPage />}
        {activeTab === 'regimes'     && <RegimePage />}
        {activeTab === 'equity'      && <EquityPage />}
      </main>
    </div>
  )
}
```

- [ ] **Step 5: Commit**

```bash
git add v2/web/src/api.ts v2/web/src/components/TabBar.tsx v2/web/src/App.tsx
git commit -m "v2: web app — TabBar, App shell, api.ts"
```

---

## Task 5: `LivePage.tsx` — candle chart + prediction distribution

- [ ] **Step 1: Create `v2/web/src/pages/LivePage.tsx`**

```tsx
import { useEffect, useRef, useState } from 'react'
import { createChart, type IChartApi, type ISeriesApi, CandlestickSeries } from 'lightweight-charts'
import { fetchCandles } from '../api'

export function LivePage() {
  const chartRef = useRef<HTMLDivElement>(null)
  const chartApi = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const [prediction, setPrediction] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!chartRef.current) return
    const chart = createChart(chartRef.current, {
      layout: { background: { color: '#0b0e13' }, textColor: '#8492a6' },
      grid: { vertLines: { color: '#1c2230' }, horzLines: { color: '#1c2230' } },
      crosshair: { mode: 1 },
      rightPriceScale: { borderColor: '#252d3d' },
      timeScale: { borderColor: '#252d3d', timeVisible: true, secondsVisible: false },
      width: chartRef.current.clientWidth,
      height: 340,
    })
    const series = chart.addSeries(CandlestickSeries, {
      upColor: '#00d4aa', downColor: '#f05252',
      borderUpColor: '#00d4aa', borderDownColor: '#f05252',
      wickUpColor: '#00d4aa', wickDownColor: '#f05252',
    })
    chartApi.current = chart
    seriesRef.current = series

    const ro = new ResizeObserver(() => {
      if (chartRef.current) chart.resize(chartRef.current.clientWidth, 340)
    })
    ro.observe(chartRef.current)

    return () => { ro.disconnect(); chart.remove() }
  }, [])

  const refresh = async () => {
    try {
      setError(null)
      const data = await fetchCandles(300)
      const candles = (data.candles ?? []).map((c: any) => ({
        time: c.time,
        open: c.open, high: c.high, low: c.low, close: c.close,
      }))
      seriesRef.current?.setData(candles)
      chartApi.current?.timeScale().fitContent()
      setPrediction(data.prediction)
    } catch (e: any) {
      setError(e.message)
    }
  }

  useEffect(() => { refresh() }, [])

  const top5 = prediction?.top5_rets ?? []
  const top5p = prediction?.top5_probs ?? []

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <h2 style={{ fontSize: 15, fontWeight: 600, color: 'var(--fg)' }}>Live — BTCUSDT 1m</h2>
        <button onClick={refresh}>Refresh</button>
      </div>

      {error && <div style={{ color: 'var(--red)', marginBottom: 8, fontSize: 13 }}>{error}</div>}

      <div className="card" style={{ marginBottom: 16 }}>
        <div ref={chartRef} style={{ width: '100%' }} />
      </div>

      {prediction ? (
        <div className="card">
          <div style={{ fontSize: 12, color: 'var(--fg-dim)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Next-bar prediction (top 5 bins)
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {top5.map((ret: number, i: number) => (
              <div key={i} className="metric">
                <div className="label">#{i + 1}</div>
                <div className="value" style={{ fontSize: 16, color: ret >= 0 ? 'var(--green)' : 'var(--red)' }}>
                  {ret >= 0 ? '+' : ''}{(ret * 100).toFixed(3)}%
                </div>
                <div style={{ color: 'var(--fg-dim)', fontSize: 11, marginTop: 2 }}>
                  p={((top5p[i] ?? 0) * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="card" style={{ color: 'var(--fg-dim)', fontSize: 13 }}>
          {prediction === null ? 'No model loaded — predictions unavailable.' : 'Loading prediction…'}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add v2/web/src/pages/LivePage.tsx
git commit -m "v2: web LivePage — candle chart + next-bar prediction"
```

---

## Task 6: History, Calibration, Regime, Equity pages

- [ ] **Step 1: Create `v2/web/src/pages/HistoryPage.tsx`**

```tsx
import { useEffect, useState } from 'react'
import { fetchHistory } from '../api'

export function HistoryPage() {
  const [data, setData] = useState<any[]>([])
  const [err, setErr] = useState<string|null>(null)

  useEffect(() => {
    fetchHistory(200).then(d => setData(d.windows ?? [])).catch(e => setErr(e.message))
  }, [])

  if (err) return <div style={{ color: 'var(--red)' }}>{err}</div>

  const n = data.length
  const nCorrect = data.filter(d => d.correct).length
  const acc = n > 0 ? nCorrect / n : 0

  return (
    <div>
      <h2 style={{ fontSize: 15, fontWeight: 600, marginBottom: 12 }}>Historical Eval — Test Set</h2>
      <div className="metric-row">
        <div className="metric"><div className="label">Accuracy</div><div className="value">{(acc*100).toFixed(2)}%</div></div>
        <div className="metric"><div className="label">Windows</div><div className="value">{n}</div></div>
      </div>
      <div className="card">
        <table>
          <thead>
            <tr><th>#</th><th>Predicted</th><th>Actual</th><th>Correct</th><th>Regime</th></tr>
          </thead>
          <tbody>
            {data.slice(0, 100).map((row, i) => (
              <tr key={i}>
                <td style={{ color: 'var(--fg-dim)' }}>{row.idx}</td>
                <td style={{ color: row.pred_ret >= 0 ? 'var(--green)' : 'var(--red)', fontFamily: 'var(--font-mono)' }}>
                  {row.pred_ret >= 0 ? '+' : ''}{(row.pred_ret * 100).toFixed(4)}%
                </td>
                <td style={{ color: row.true_ret >= 0 ? 'var(--green)' : 'var(--red)', fontFamily: 'var(--font-mono)' }}>
                  {row.true_ret >= 0 ? '+' : ''}{(row.true_ret * 100).toFixed(4)}%
                </td>
                <td style={{ color: row.correct ? 'var(--green)' : 'var(--red)' }}>
                  {row.correct ? '✓' : '✗'}
                </td>
                <td style={{ color: 'var(--fg-dim)' }}>{row.regime}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Create `v2/web/src/pages/CalibrationPage.tsx`**

```tsx
import { useEffect, useRef, useState } from 'react'
import { createChart, LineSeries } from 'lightweight-charts'
import { fetchCalibration } from '../api'

export function CalibrationPage() {
  const chartRef = useRef<HTMLDivElement>(null)
  const [data, setData] = useState<any>(null)
  const [err, setErr] = useState<string|null>(null)

  useEffect(() => {
    fetchCalibration().then(setData).catch(e => setErr(e.message))
  }, [])

  useEffect(() => {
    if (!data || !chartRef.current) return
    const chart = createChart(chartRef.current, {
      layout: { background: { color: '#0b0e13' }, textColor: '#8492a6' },
      grid: { vertLines: { color: '#1c2230' }, horzLines: { color: '#1c2230' } },
      width: chartRef.current.clientWidth,
      height: 280,
    })
    const actualSeries = chart.addSeries(LineSeries, { color: '#00d4aa', lineWidth: 2 })
    const idealSeries = chart.addSeries(LineSeries, { color: '#f05252', lineWidth: 1, lineStyle: 2 })

    const buckets: any[] = data.buckets ?? []
    const actualData = buckets.map((b: any) => ({ time: parseFloat(b.lo.toFixed(3)), value: b.acc }))
    const idealData = buckets.map((b: any) => ({ time: parseFloat(b.lo.toFixed(3)), value: b.lo }))

    if (actualData.length) { actualSeries.setData(actualData); idealSeries.setData(idealData) }
    chart.timeScale().fitContent()

    return () => chart.remove()
  }, [data])

  if (err) return <div style={{ color: 'var(--red)' }}>{err}</div>

  return (
    <div>
      <h2 style={{ fontSize: 15, fontWeight: 600, marginBottom: 12 }}>Calibration</h2>
      <div className="metric-row">
        <div className="metric">
          <div className="label">ECE</div>
          <div className="value">{data ? (data.ece * 100).toFixed(2) + '%' : '—'}</div>
        </div>
      </div>
      <div className="card" style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 12, color: 'var(--fg-dim)', marginBottom: 8 }}>
          Teal: actual accuracy per confidence bucket · Red dashed: perfect calibration
        </div>
        <div ref={chartRef} style={{ width: '100%' }} />
      </div>
      {data?.buckets?.length > 0 && (
        <div className="card">
          <table>
            <thead><tr><th>Conf range</th><th>Avg conf</th><th>Avg acc</th><th>Fraction</th></tr></thead>
            <tbody>
              {data.buckets.map((b: any, i: number) => (
                <tr key={i}>
                  <td>[{b.lo.toFixed(1)}, {b.hi.toFixed(1)})</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{b.conf.toFixed(3)}</td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: Math.abs(b.conf - b.acc) < 0.05 ? 'var(--green)' : 'var(--red)' }}>{b.acc.toFixed(3)}</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{(b.frac * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 3: Create `v2/web/src/pages/RegimePage.tsx`**

```tsx
import { useEffect, useState } from 'react'
import { fetchRegimes } from '../api'

export function RegimePage() {
  const [regimes, setRegimes] = useState<any[]>([])
  const [err, setErr] = useState<string|null>(null)

  useEffect(() => {
    fetchRegimes().then(d => setRegimes(d.regimes ?? [])).catch(e => setErr(e.message))
  }, [])

  if (err) return <div style={{ color: 'var(--red)' }}>{err}</div>

  return (
    <div>
      <h2 style={{ fontSize: 15, fontWeight: 600, marginBottom: 12 }}>Per-Regime Accuracy</h2>
      <div className="metric-row">
        {regimes.map(r => (
          <div key={r.id} className="metric">
            <div className="label">{r.name}</div>
            <div className="value">{(r.accuracy * 100).toFixed(1)}%</div>
            <div style={{ color: 'var(--fg-dim)', fontSize: 11, marginTop: 2 }}>{r.n.toLocaleString()} bars</div>
          </div>
        ))}
      </div>
      {regimes.length > 0 && (
        <div className="card">
          <table>
            <thead><tr><th>Regime</th><th>Accuracy</th><th>Bars</th></tr></thead>
            <tbody>
              {regimes.map(r => (
                <tr key={r.id}>
                  <td>{r.name}</td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: r.accuracy > 0.5 ? 'var(--green)' : 'var(--red)' }}>
                    {(r.accuracy * 100).toFixed(2)}%
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--fg-dim)' }}>{r.n.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 4: Create `v2/web/src/pages/EquityPage.tsx`**

```tsx
import { useEffect, useRef, useState } from 'react'
import { createChart, LineSeries } from 'lightweight-charts'
import { fetchEquity } from '../api'

export function EquityPage() {
  const chartRef = useRef<HTMLDivElement>(null)
  const [meta, setMeta] = useState<any>(null)
  const [err, setErr] = useState<string|null>(null)

  useEffect(() => {
    fetchEquity().then(setMeta).catch(e => setErr(e.message))
  }, [])

  useEffect(() => {
    if (!meta || !chartRef.current) return
    const chart = createChart(chartRef.current, {
      layout: { background: { color: '#0b0e13' }, textColor: '#8492a6' },
      grid: { vertLines: { color: '#1c2230' }, horzLines: { color: '#1c2230' } },
      width: chartRef.current.clientWidth,
      height: 320,
    })
    const series = chart.addSeries(LineSeries, { color: '#00d4aa', lineWidth: 2 })
    const eq: any[] = meta.equity ?? []
    if (eq.length > 1) {
      // lightweight-charts needs time as ascending values; use idx as fake time
      const chartData = eq.map((p: any) => ({ time: p.idx + 1000000, value: p.cumret }))
      series.setData(chartData)
      chart.timeScale().fitContent()
    }
    return () => chart.remove()
  }, [meta])

  if (err) return <div style={{ color: 'var(--red)' }}>{err}</div>

  return (
    <div>
      <h2 style={{ fontSize: 15, fontWeight: 600, marginBottom: 12 }}>Equity Curve — Long/Short Backtest</h2>
      <div className="metric-row">
        <div className="metric">
          <div className="label">Sharpe (ann.)</div>
          <div className="value" style={{ color: (meta?.sharpe ?? 0) > 0 ? 'var(--green)' : 'var(--red)' }}>
            {meta ? meta.sharpe.toFixed(2) : '—'}
          </div>
        </div>
        <div className="metric">
          <div className="label">Max Drawdown</div>
          <div className="value" style={{ color: 'var(--red)' }}>
            {meta ? (meta.max_dd * 100).toFixed(2) + '%' : '—'}
          </div>
        </div>
        <div className="metric">
          <div className="label">Final return</div>
          <div className="value">
            {meta?.equity?.length > 1
              ? ((meta.equity[meta.equity.length - 1].cumret) * 100).toFixed(2) + '%'
              : '—'}
          </div>
        </div>
      </div>
      <div className="card">
        <div style={{ fontSize: 12, color: 'var(--fg-dim)', marginBottom: 8 }}>
          Strategy: long if P(next&gt;+0.02%) &gt; P(next&lt;-0.02%), short if inverse, flat otherwise.
        </div>
        <div ref={chartRef} style={{ width: '100%' }} />
      </div>
    </div>
  )
}
```

- [ ] **Step 5: Commit all pages**

```bash
git add v2/web/src/pages/
git commit -m "v2: web pages — History, Calibration, Regimes, Equity"
```

---

## Task 7: Build frontend, start servers, verify

- [ ] **Step 1: Build the frontend**

```bash
cd projects/candle-gpt/v2/web
npm run build
```

Expected: `dist/` is created, no TypeScript errors, no Vite errors.

If there are TS errors, fix them before proceeding. Common issues:
- `CandlestickSeries` import from `lightweight-charts` v5 may use `createSeriesMarkers` — check the v5 API and adjust `LivePage.tsx` accordingly.
- `LineSeries` in v5 uses `chart.addSeries(LineSeries, ...)` — confirm this matches the installed version.

- [ ] **Step 2: Start v2 server**

```bash
cd projects/candle-gpt
uv run uvicorn v2.server.main:app --host 0.0.0.0 --port 8766 --reload &
echo $! > v2/server.pid
```

- [ ] **Step 3: Verify API responds**

```bash
curl -s http://localhost:8766/api/v2/status | python3 -m json.tool
```

Expected: JSON with `model_loaded`, `run_id`, etc.

```bash
curl -s http://localhost:8766/api/v2/candles?limit=5 | python3 -m json.tool
```

Expected: JSON with `candles` array (may be empty if Binance is unreachable from this host; that's OK — just no error 500).

- [ ] **Step 4: Verify frontend is served**

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8766/
```

Expected: `200`

```bash
curl -s http://localhost:8766/ | grep -c "candle-gpt"
```

Expected: `1` (the title tag appears in the HTML)

- [ ] **Step 5: Check Tailscale URL**

```bash
curl -s -o /dev/null -w "%{http_code}" http://100.88.188.80:8766/api/v2/status
```

Expected: `200` (if Tailscale is up).

- [ ] **Step 6: Kill test server**

```bash
kill $(cat v2/server.pid) 2>/dev/null; rm -f v2/server.pid
```

- [ ] **Step 7: Commit built frontend and server start script**

```bash
cd projects/candle-gpt
git add v2/web/dist/ v2/web/package-lock.json
git commit -m "v2: web build — dist/ bundle"
git tag v2-dashboard
```

---

## Task 8: Final integration verification + Telegram

- [ ] **Step 1: Run full test suite one final time**

```bash
cd projects/candle-gpt
uv run pytest v2/tests/ -v --tb=short
```

Expected: 0 failed.

- [ ] **Step 2: Start both servers for final check**

```bash
# v1 (already running, just verify):
curl -s -o /dev/null -w "v1: %{http_code}\n" http://localhost:8765/api/info

# Start v2:
uv run uvicorn v2.server.main:app --host 0.0.0.0 --port 8766 &
sleep 3
curl -s -o /dev/null -w "v2: %{http_code}\n" http://localhost:8766/api/v2/status
```

Expected:
```
v1: 200
v2: 200
```

- [ ] **Step 3: Send Plan 5 completion Telegram**

```bash
openclaw message send --channel telegram --target '8703980136' \
    --message "Plan 5 done. v2 dashboard live at localhost:8766 + http://100.88.188.80:8766 — dark-themed 5-tab UI (Live, History, Calibration, Regimes, Equity) with lightweight-charts + per-bar predictions. Starting final tag."
```

- [ ] **Step 4: Final completion Telegram**

After all plans complete and all tags are set:

```bash
V2_TAGS=$(git tag | grep v2 | tr '\n' ' ')
openclaw message send --channel telegram --target '8703980136' \
    --message "candle-gpt v2 complete. Tags: ${V2_TAGS}. v1 untouched on 8765. v2 on 8766."
```

---

## What's next

All five plans are complete. The v2 stack is:
- Data pipeline (Plan 1 + 1.5): BTCUSDT 1m, funding, liq, regime-tagged
- Feature vector (Plan 2): 41-dim engineered features
- Model (Plan 3): CandleGPTv2, ~10.9M params
- Training (Plan 4): up to 6h, best-checkpoint eval + REPORT.md
- Dashboard (Plan 5): FastAPI on 8766 + React dark-theme 5-tab UI

Future work (explicitly out of scope for this run):
- ETH/SOL multi-asset training
- 5m timeframe model variant
- L2/L3 order book depth features
- Tardis liquidation backfill
