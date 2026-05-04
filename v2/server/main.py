"""FastAPI v2 server — port 8766."""
from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from v2.server.inference import V2InferenceModel
from v2.server.eval_cache import EvalCache
from v2.server.training_view import TrainingView
from v2.server.sweep import SweepService

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
RUNS_DIR = PROJECT_ROOT / "v2" / "runs"
WEB_DIST = HERE.parent / "web" / "dist"
RAW_DIR = PROJECT_ROOT / "v2" / "data" / "raw"

inference = V2InferenceModel()
cache = EvalCache()
training_view = TrainingView(RUNS_DIR)
# Sweep / backtest service — points at the most-recent run's checkpoint.
# It loads lazily on first request to avoid blocking server startup.
_BEST_RUN_DIR: Optional[Path] = None
sweep_service: Optional[SweepService] = None


def _find_best_run() -> tuple[Optional[Path], Optional[Path]]:
    current_id_file = PROJECT_ROOT / "v2" / "current_run_id.txt"
    if current_id_file.exists():
        run_id = current_id_file.read_text().strip()
        ckpt = RUNS_DIR / run_id / "checkpoints" / "best_val.pt"
        tok = RUNS_DIR / run_id / "tokenizer.pkl"
        if ckpt.exists() and tok.exists():
            return ckpt, tok
    for run_dir in sorted(RUNS_DIR.iterdir() if RUNS_DIR.exists() else [], reverse=True):
        if not run_dir.is_dir():
            continue
        ckpt = run_dir / "checkpoints" / "best_val.pt"
        tok = run_dir / "tokenizer.pkl"
        if ckpt.exists() and tok.exists():
            return ckpt, tok
    return None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _BEST_RUN_DIR, sweep_service
    ckpt_path, tok_path = _find_best_run()
    if ckpt_path:
        print(f"[v2 server] Loading model from {ckpt_path}")
        inference.load(ckpt_path, tok_path)
        run_dir = ckpt_path.parents[1]
        cache.load_from_report(run_dir / "REPORT.md", run_dir / "metrics.json")
        print(f"[v2 server] Model loaded. Device={inference.device}, step={inference.ckpt_step}")
        _BEST_RUN_DIR = run_dir
        sweep_service = SweepService(
            run_dir=run_dir,
            kline_path=RAW_DIR / "btcusdt_1m.parquet",
            funding_path=RAW_DIR / "funding_btcusdt.parquet",
            liq_path=RAW_DIR / "liq_btcusdt_per_minute.parquet",
        )
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


_ALLOWED_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"}


@app.get("/api/v2/candles")
def candles(limit: int = 300, interval: str = "1m"):
    limit = max(50, min(int(limit), 520))
    if interval not in _ALLOWED_INTERVALS:
        raise HTTPException(400, f"interval must be one of {sorted(_ALLOWED_INTERVALS)}")
    try:
        result = inference.predict_live(limit=limit, interval=interval)
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


@app.get("/api/v2/training/status")
def training_status():
    return training_view.read_status()


@app.get("/api/v2/training/events")
def training_events(after: Optional[float] = None, limit: int = 5000):
    return training_view.read_events(after_ts=after, limit=limit)


@app.get("/api/v2/system/stats")
def system_stats():
    from v2.server.system_stats import read_system_stats
    return read_system_stats()


@app.get("/api/v2/eval_history")
def eval_history(run_id: Optional[str] = None):
    """Read v2/runs/<run_id>/eval_history.jsonl. If run_id is omitted,
    we use the most-recent run that has a history file."""
    target_dir: Optional[Path] = None
    if run_id:
        target_dir = RUNS_DIR / run_id
    else:
        if not RUNS_DIR.exists():
            return {"available": False, "run_id": None, "entries": []}
        candidates = []
        for d in RUNS_DIR.iterdir():
            if d.is_dir() and (d / "eval_history.jsonl").exists():
                candidates.append(d)
        if candidates:
            target_dir = max(candidates, key=lambda p: p.stat().st_mtime)
    if target_dir is None or not target_dir.exists():
        return {"available": False, "run_id": run_id, "entries": []}
    history_path = target_dir / "eval_history.jsonl"
    entries = []
    if history_path.exists():
        with history_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(__import__("json").loads(line))
                except Exception:
                    continue
    return {"available": True, "run_id": target_dir.name, "entries": entries}


@app.get("/api/v2/calibration/sweep")
def calibration_sweep(
    temperatures: str = "0.5,0.8,1.0,1.5,2.0",
    horizons: str = "1,3,5,10,20,30",
    n_samples: int = 200,
):
    """Sweep over (temperature, horizon) pairs and report directional accuracy.

    Single forward pass per sampled window — temperatures rescale the SAME
    logits, horizons just change which actual cumulative return we compare
    sign against. Designed to be fast enough for interactive tuning."""
    if sweep_service is None:
        raise HTTPException(503, "sweep service unavailable — no trained run")
    try:
        T_list = [float(t) for t in temperatures.split(",") if t.strip()]
        H_list = [int(h) for h in horizons.split(",") if h.strip()]
    except ValueError as e:
        raise HTTPException(400, f"bad temperatures/horizons: {e}")
    if not T_list or not H_list:
        raise HTTPException(400, "need at least one temperature and one horizon")
    n_samples = max(20, min(int(n_samples), 1000))
    try:
        return sweep_service.sweep(temperatures=T_list, horizons=H_list, n_samples=n_samples)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"sweep failed: {e}")


@app.get("/api/v2/backtest")
def backtest(
    temperature: float = 1.0,
    horizon: int = 30,
    z_threshold: float = 0.3,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    fee_bps: float = 1.0,
):
    """Run a long/short backtest over a slice of the test set with the
    chosen settings. Returns equity curve + summary stats."""
    if sweep_service is None:
        raise HTTPException(503, "sweep service unavailable — no trained run")
    try:
        return sweep_service.backtest(
            temperature=temperature, horizon=horizon, z_threshold=z_threshold,
            start_frac=start_frac, end_frac=end_frac, fee_bps=fee_bps,
        )
    except Exception as e:
        raise HTTPException(500, f"backtest failed: {e}")


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
        return {"status": "ok", "note": "frontend not built yet"}
