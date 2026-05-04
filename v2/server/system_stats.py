"""macOS system-stats reader for the Training tab.

CPU and RAM come from psutil (no special privileges).
GPU device utilization comes from `ioreg -rc IOAccelerator` (no sudo).
Temperature + fan need `powermetrics`, which requires root. We try
`sudo -n` (non-interactive) and degrade gracefully if it can't run.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Optional

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # noqa


_GPU_UTIL_RE = re.compile(r'"Device Utilization %"\s*=\s*(\d+)')
_FAN_RE = re.compile(r'(?:Fan)\s+(\d+)\s+rpm', re.IGNORECASE)


def _read_gpu_util() -> Optional[float]:
    """Return GPU device utilization % (0..100) or None if unavailable."""
    if not shutil.which("ioreg"):
        return None
    try:
        out = subprocess.check_output(
            ["ioreg", "-rc", "IOAccelerator", "-d", "1"],
            stderr=subprocess.DEVNULL,
            timeout=2,
            text=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    matches = [int(m.group(1)) for m in _GPU_UTIL_RE.finditer(out)]
    if not matches:
        return None
    # Max across IOAccelerator nodes — Apple Silicon reports under multiple keys.
    return float(max(matches))


def _read_cpu_temp_via_powermetrics(timeout: float = 2.0) -> dict[str, Any]:
    """Try `sudo -n powermetrics --samplers smc -i 200 -n 1`.

    Returns a dict with whatever we could parse. Keys all optional:
      cpu_temp_c, gpu_temp_c, fan_rpm, available, hint
    """
    if not shutil.which("powermetrics"):
        return {"available": False, "hint": "powermetrics not installed"}
    try:
        out = subprocess.check_output(
            ["sudo", "-n", "powermetrics", "--samplers", "smc,thermal",
             "-i", "200", "-n", "1", "-f", "plist"],
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.CalledProcessError:
        return {
            "available": False,
            "hint": "powermetrics requires sudo. Add a NOPASSWD entry in /etc/sudoers.d/powermetrics: '<user> ALL=(root) NOPASSWD: /usr/bin/powermetrics'",
        }
    except (subprocess.SubprocessError, OSError):
        return {"available": False, "hint": "powermetrics call failed"}
    # The plist output is verbose; parse the bits we care about defensively.
    text = out.decode("utf-8", errors="replace")
    cpu_temp = None
    gpu_temp = None
    fan_rpm = None
    for m in re.finditer(r"<key>(?P<k>[^<]+)</key>\s*<real>(?P<v>[-\d.]+)</real>", text):
        k = m.group("k").lower()
        v = float(m.group("v"))
        if "cpu_die" in k or k.endswith("cpu_die_temperature_c"):
            cpu_temp = v
        elif "gpu_die" in k or k.endswith("gpu_die_temperature_c"):
            gpu_temp = v
        elif k.endswith("fan_speed") or k == "fan_rpm":
            fan_rpm = v
    fan_match = _FAN_RE.search(text)
    if fan_match and fan_rpm is None:
        fan_rpm = float(fan_match.group(1))
    return {
        "available": True,
        "cpu_temp_c": cpu_temp,
        "gpu_temp_c": gpu_temp,
        "fan_rpm": fan_rpm,
    }


# Cheap thread-safe cache so repeated polls don't shell out every time.
_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {"ts": 0.0, "value": None}
_TTL_S = 1.5


def read_system_stats() -> dict[str, Any]:
    with _LOCK:
        now = time.time()
        if _CACHE["value"] is not None and now - _CACHE["ts"] < _TTL_S:
            return _CACHE["value"]

    cpu_percent: Optional[float] = None
    cpu_per_core: Optional[list[float]] = None
    cpu_count: Optional[int] = None
    ram_percent: Optional[float] = None
    ram_used_gb: Optional[float] = None
    ram_total_gb: Optional[float] = None
    if psutil is not None:
        try:
            # Non-blocking call — uses delta since last invocation. Caller polls
            # this endpoint at ~1Hz, which is plenty.
            cpu_percent = float(psutil.cpu_percent(interval=None))
            cpu_per_core = [float(x) for x in psutil.cpu_percent(interval=None, percpu=True)]
            cpu_count = psutil.cpu_count(logical=True)
            vm = psutil.virtual_memory()
            ram_percent = float(vm.percent)
            ram_used_gb = round(vm.used / (1024 ** 3), 2)
            ram_total_gb = round(vm.total / (1024 ** 3), 2)
        except Exception:
            pass

    gpu_util = _read_gpu_util()
    thermal = _read_cpu_temp_via_powermetrics()

    result = {
        "ts": time.time(),
        "cpu_percent": cpu_percent,
        "cpu_per_core": cpu_per_core,
        "cpu_count": cpu_count,
        "ram_percent": ram_percent,
        "ram_used_gb": ram_used_gb,
        "ram_total_gb": ram_total_gb,
        "gpu_util_percent": gpu_util,
        "thermal": thermal,
    }
    with _LOCK:
        _CACHE["ts"] = time.time()
        _CACHE["value"] = result
    return result
