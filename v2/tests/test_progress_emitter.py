import json
import time
from pathlib import Path

import pytest

from v2.train.progress import (
    HardwareSpecs, ModelSpecs, ProgressEmitter, TrainingStatus,
)


def _hw():
    return HardwareSpecs(
        hostname="test", platform="test", python="3.11", torch="2.x",
        device="cpu", device_name="CPU", cpu_count=4, ram_gb=16.0,
    )


def _ms():
    return ModelSpecs(n_params=10_900_000, n_layers=8, n_heads=8,
                      d_model=384, n_bins=256, window=512)


def test_status_round_trip(tmp_path: Path):
    em = ProgressEmitter(tmp_path / "run1", _hw(), _ms(),
                         wall_clock_cap_s=3600, max_steps=1000, min_interval_s=0.0)
    em.update(step=42, train_loss=2.31, lr=3e-4, throughput_tok_per_s=12345.0)
    s = json.loads((tmp_path / "run1" / "status.json").read_text())
    assert s["step"] == 42
    assert s["train_loss"] == pytest.approx(2.31)
    assert 0 <= s["progress_frac"] <= 1.0
    assert s["eta_s"] >= 0


def test_events_jsonl_appends(tmp_path: Path):
    em = ProgressEmitter(tmp_path / "run2", _hw(), _ms(),
                         wall_clock_cap_s=3600, max_steps=1000, min_interval_s=0.0)
    em.event("step", {"step": 1, "loss": 5.0})
    em.event("step", {"step": 2, "loss": 4.5})
    em.event("val", {"step": 500, "val_loss": 3.8})
    lines = (tmp_path / "run2" / "events.jsonl").read_text().strip().split("\n")
    assert len(lines) == 3
    assert json.loads(lines[2])["kind"] == "val"


def test_min_interval_throttles(tmp_path: Path):
    em = ProgressEmitter(tmp_path / "run3", _hw(), _ms(),
                         wall_clock_cap_s=3600, max_steps=1000, min_interval_s=10.0)
    em.update(step=1, train_loss=2.0, lr=3e-4)
    first_mtime = (tmp_path / "run3" / "status.json").stat().st_mtime
    em.update(step=2, train_loss=1.9, lr=3e-4)  # throttled
    second_mtime = (tmp_path / "run3" / "status.json").stat().st_mtime
    assert first_mtime == second_mtime
    em.update(step=3, train_loss=1.8, lr=3e-4, force=True)
    third_mtime = (tmp_path / "run3" / "status.json").stat().st_mtime
    assert third_mtime > first_mtime


def test_best_val_tracked(tmp_path: Path):
    em = ProgressEmitter(tmp_path / "run4", _hw(), _ms(),
                         wall_clock_cap_s=3600, max_steps=1000, min_interval_s=0.0)
    em.update(step=100, val_loss=2.0, lr=3e-4)
    em.update(step=200, val_loss=2.5, lr=3e-4)
    em.update(step=300, val_loss=1.5, lr=3e-4)
    s = json.loads((tmp_path / "run4" / "status.json").read_text())
    assert s["best_val_loss"] == pytest.approx(1.5)
