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


def _read_thermal_via_powermetrics(timeout: float = 2.0) -> dict[str, Any]:
    """Run `sudo -n powermetrics --samplers thermal,gpu_power,cpu_power -i 200 -n 1`
    and parse thermal pressure level + GPU/CPU power.

    Apple Silicon (M-series) does NOT expose CPU/GPU die temperatures or fan
    RPM through powermetrics or public ioreg keys — those readings live behind
    SMC keys that only third-party tools using private APIs (iStat, smctemp,
    asitop) can read. We return what IS available and surface the rest as a
    null value with an explanatory hint.
    """
    if not shutil.which("powermetrics"):
        return {"available": False, "hint": "powermetrics not installed"}
    try:
        out = subprocess.check_output(
            ["sudo", "-n", "powermetrics",
             "--samplers", "thermal,gpu_power,cpu_power",
             "-i", "200", "-n", "1"],
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.CalledProcessError:
        return {
            "available": False,
            "hint": "powermetrics requires passwordless sudo. Run: echo \"$USER ALL=(root) NOPASSWD: /usr/bin/powermetrics\" | sudo tee /etc/sudoers.d/powermetrics && sudo chmod 440 /etc/sudoers.d/powermetrics",
        }
    except (subprocess.SubprocessError, OSError):
        return {"available": False, "hint": "powermetrics call failed"}
    text = out.decode("utf-8", errors="replace")

    # Thermal pressure level: Nominal / Fair / Serious / Critical
    pressure = None
    m = re.search(r"Current pressure level:\s*(\w+)", text)
    if m:
        pressure = m.group(1)

    # GPU power in mW
    gpu_power_mw = None
    m = re.search(r"GPU Power:\s*(\d+)\s*mW", text)
    if m:
        gpu_power_mw = int(m.group(1))

    # GPU active frequency
    gpu_freq_mhz = None
    m = re.search(r"GPU HW active frequency:\s*(\d+)\s*MHz", text)
    if m:
        gpu_freq_mhz = int(m.group(1))

    # CPU power totals (sum of E and P clusters when reported separately)
    cpu_power_mw = 0
    cpu_power_seen = False
    for m in re.finditer(r"(?:E|P)-Cluster Power:\s*(\d+)\s*mW", text):
        cpu_power_mw += int(m.group(1))
        cpu_power_seen = True
    if not cpu_power_seen:
        m = re.search(r"CPU Power:\s*(\d+)\s*mW", text)
        if m:
            cpu_power_mw = int(m.group(1))
            cpu_power_seen = True
    if not cpu_power_seen:
        cpu_power_mw = None  # type: ignore

    return {
        "available": True,
        "pressure_level": pressure,         # Nominal / Fair / Serious / Critical
        "gpu_power_mw": gpu_power_mw,
        "gpu_freq_mhz": gpu_freq_mhz,
        "cpu_power_mw": cpu_power_mw,
        # Apple Silicon doesn't expose these via powermetrics:
        "cpu_temp_c": None,
        "gpu_temp_c": None,
        "fan_rpm": None,
        "hint": "Apple Silicon doesn't expose CPU/GPU die temperatures or fan RPM through powermetrics. Pressure level + GPU/CPU power are the available proxies for thermal load.",
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
    thermal = _read_thermal_via_powermetrics()

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
