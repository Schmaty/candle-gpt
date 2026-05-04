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

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
RUNS_DIR = PROJECT_ROOT / "v2" / "runs"
WEB_DIST = HERE.parent / "web" / "dist"

inference = V2InferenceModel()
cache = EvalCache()
training_view = TrainingView(RUNS_DIR)


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
    ckpt_path, tok_path = _find_best_run()
    if ckpt_path:
        print(f"[v2 server] Loading model from {ckpt_path}")
        inference.load(ckpt_path, tok_path)
        run_dir = ckpt_path.parents[1]
        cache.load_from_report(run_dir / "REPORT.md", run_dir / "metrics.json")
        print(f"[v2 server] Model loaded. Device={inference.device}, step={inference.ckpt_step}")
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
