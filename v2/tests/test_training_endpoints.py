import json
import time
from pathlib import Path

from v2.server.training_view import TrainingView


def _seed_run(runs_dir: Path, run_id: str, status: dict, events: list[dict]) -> Path:
    run = runs_dir / run_id
    run.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps(status))
    with (run / "events.jsonl").open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return run


def test_training_view_no_runs(tmp_path: Path):
    tv = TrainingView(tmp_path)
    s = tv.read_status()
    assert s["available"] is False
    assert tv.read_events()["events"] == []


def test_training_view_picks_latest(tmp_path: Path):
    _seed_run(tmp_path, "20260504_080000", {"step": 1}, [])
    time.sleep(0.05)
    _seed_run(tmp_path, "20260504_090000", {"step": 99}, [])
    s = TrainingView(tmp_path).read_status()
    assert s["available"] is True
    assert s["step"] == 99


def test_training_view_event_cursor(tmp_path: Path):
    _seed_run(tmp_path, "r", {"step": 0}, [
        {"ts": 1.0, "kind": "step", "loss": 5.0},
        {"ts": 2.0, "kind": "step", "loss": 4.5},
        {"ts": 3.0, "kind": "val",  "val_loss": 3.8},
    ])
    tv = TrainingView(tmp_path)
    page1 = tv.read_events(after_ts=None)
    assert [e["ts"] for e in page1["events"]] == [1.0, 2.0, 3.0]
    page2 = tv.read_events(after_ts=2.0)
    assert [e["ts"] for e in page2["events"]] == [3.0]
    assert page2["cursor"] == 3.0
