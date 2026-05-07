"""Read-only view over the most recent training run for the dashboard."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Optional


class TrainingView:
    def __init__(self, runs_dir: Path):
        self.runs_dir = Path(runs_dir)

    def _latest_run_dir(self) -> Optional[Path]:
        if not self.runs_dir.exists():
            return None
        # Prefer the run explicitly selected by the server/dashboard. This
        # avoids newer abandoned "starting" runs hiding the loaded model.
        current_id_file = self.runs_dir / "current_run_id.txt"
        if current_id_file.exists():
            run_id = current_id_file.read_text().strip()
            selected = self.runs_dir / run_id
            if selected.is_dir() and (selected / "status.json").exists():
                return selected
        # Fallback: sort by name (YYYYMMDD_HHMMSS format) — more reliable than mtime.
        candidates = sorted(
            (p for p in self.runs_dir.iterdir() if p.is_dir() and (p / "status.json").exists()),
            key=lambda p: p.name,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def read_status(self) -> dict[str, Any]:
        run = self._latest_run_dir()
        if run is None:
            return {"available": False, "run_id": None}
        try:
            data = json.loads((run / "status.json").read_text())
            data["available"] = True
            return data
        except (OSError, json.JSONDecodeError):
            return {"available": False, "run_id": run.name}

    def read_events(self, after_ts: Optional[float] = None,
                    limit: int = 5000) -> dict[str, Any]:
        run = self._latest_run_dir()
        if run is None:
            return {"events": [], "cursor": after_ts or 0.0}
        events_path = run / "events.jsonl"
        if not events_path.exists():
            return {"events": [], "cursor": after_ts or 0.0}
        out: list[dict[str, Any]] = []
        cursor = after_ts or 0.0
        with events_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = float(rec.get("ts", 0.0))
                if after_ts is not None and ts <= after_ts:
                    continue
                out.append(rec)
                if ts > cursor:
                    cursor = ts
                if len(out) >= limit:
                    break
        return {"events": out, "cursor": cursor}
