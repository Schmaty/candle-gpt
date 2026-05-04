"""Live training progress emission for the dashboard."""
from __future__ import annotations
import json
import os
import platform
import socket
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch


@dataclass
class HardwareSpecs:
    hostname: str
    platform: str
    python: str
    torch: str
    device: str
    device_name: str
    cpu_count: int
    ram_gb: float


@dataclass
class ModelSpecs:
    n_params: int
    n_layers: int
    n_heads: int
    d_model: int
    n_bins: int
    window: int


@dataclass
class TrainingStatus:
    run_id: str
    state: str
    started_at_utc: str
    last_update_utc: str
    elapsed_s: float
    wall_clock_cap_s: float
    eta_s: float
    step: int
    max_steps: int
    progress_frac: float
    train_loss: Optional[float]
    val_loss: Optional[float]
    best_val_loss: Optional[float]
    lr: float
    throughput_tok_per_s: float
    grad_norm: Optional[float]
    last_checkpoint_step: Optional[int]
    last_eval_step: Optional[int]
    hardware: HardwareSpecs
    model: ModelSpecs


class ProgressEmitter:
    def __init__(self, run_dir: Path, hw: HardwareSpecs, model: ModelSpecs,
                 wall_clock_cap_s: float, max_steps: int,
                 min_interval_s: float = 5.0):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.run_dir / "status.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.hw = hw
        self.model = model
        self.wall_clock_cap_s = wall_clock_cap_s
        self.max_steps = max_steps
        self.min_interval_s = min_interval_s
        self.started_at = time.time()
        self.started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._last_status_write = 0.0
        self._best_val_loss: Optional[float] = None
        self._last_val_loss: Optional[float] = None  # sticky between val passes

    def _atomic_write_status(self, status: TrainingStatus) -> None:
        tmp = self.status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(status), indent=2))
        os.replace(tmp, self.status_path)

    def update(self, *, step: int, state: str = "training",
               train_loss: Optional[float] = None,
               val_loss: Optional[float] = None,
               lr: float = 0.0,
               throughput_tok_per_s: float = 0.0,
               grad_norm: Optional[float] = None,
               last_checkpoint_step: Optional[int] = None,
               last_eval_step: Optional[int] = None,
               force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_status_write) < self.min_interval_s:
            return
        elapsed = now - self.started_at
        if val_loss is not None:
            self._last_val_loss = val_loss
            if self._best_val_loss is None or val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
        step_eta = ((self.max_steps - step) * (elapsed / max(step, 1))) if step > 0 else self.wall_clock_cap_s
        wall_eta = max(self.wall_clock_cap_s - elapsed, 0.0)
        eta = min(step_eta, wall_eta)
        progress = max(step / max(self.max_steps, 1), elapsed / self.wall_clock_cap_s)
        progress = min(progress, 1.0)
        status = TrainingStatus(
            run_id=self.run_dir.name,
            state=state,
            started_at_utc=self.started_at_utc,
            last_update_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            elapsed_s=elapsed,
            wall_clock_cap_s=self.wall_clock_cap_s,
            eta_s=eta,
            step=step,
            max_steps=self.max_steps,
            progress_frac=progress,
            train_loss=train_loss,
            val_loss=val_loss if val_loss is not None else self._last_val_loss,
            best_val_loss=self._best_val_loss,
            lr=lr,
            throughput_tok_per_s=throughput_tok_per_s,
            grad_norm=grad_norm,
            last_checkpoint_step=last_checkpoint_step,
            last_eval_step=last_eval_step,
            hardware=self.hw,
            model=self.model,
        )
        self._atomic_write_status(status)
        self._last_status_write = now

    def event(self, kind: str, body: dict[str, Any]) -> None:
        rec = {"ts": time.time(), "kind": kind, **body}
        with self.events_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    @staticmethod
    def collect_hardware() -> HardwareSpecs:
        if torch.cuda.is_available():
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            device = "mps"
            device_name = "Apple Silicon (MPS)"
        else:
            device = "cpu"
            device_name = platform.processor() or "CPU"
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
        except Exception:
            ram_gb = 0.0
        return HardwareSpecs(
            hostname=socket.gethostname(),
            platform=platform.platform(),
            python=platform.python_version(),
            torch=torch.__version__,
            device=device,
            device_name=device_name,
            cpu_count=os.cpu_count() or 1,
            ram_gb=round(ram_gb, 1),
        )
