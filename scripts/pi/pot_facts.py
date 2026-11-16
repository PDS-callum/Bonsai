#!/usr/bin/env python3
"""Print a JSON snapshot of the pot's hardware, memory, storage and thermals.

Runs on the Pi with the standard library only, so it needs nothing installed.
Every field is read from the system; anything unavailable is reported as null
rather than guessed.

Usage: pot_facts.py [path-for-free-space]   (default: the home directory)
"""

from __future__ import annotations

import json
import os
import platform
import resource
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip().rstrip("\x00")
    except OSError:
        return None


def read_int(path: str) -> int | None:
    text = read_text(path)
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def run(*command: str) -> str | None:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def meminfo() -> dict[str, int]:
    """Return /proc/meminfo as a mapping of field name to bytes."""
    fields: dict[str, int] = {}
    for line in (read_text("/proc/meminfo") or "").splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            scale = 1024 if parts[-1] == "kB" else 1
            fields[name] = int(parts[0]) * scale
    return fields


def cpu_features() -> list[str]:
    for line in (read_text("/proc/cpuinfo") or "").splitlines():
        if line.startswith("Features"):
            return line.partition(":")[2].split()
    return []


def caches() -> dict[str, str]:
    """Cache sizes as lscpu reports them, e.g. {"L2": "2 MiB (4 instances)"}."""
    found: dict[str, str] = {}
    for line in (run("lscpu") or "").splitlines():
        name, _, value = line.partition(":")
        if name.strip().endswith("cache"):
            found[name.strip().removesuffix(" cache")] = value.strip()
    return found


def vcgencmd(*args: str) -> str | None:
    """Return the value part of a vcgencmd reply such as "temp=47.7'C"."""
    reply = run("vcgencmd", *args)
    return reply.partition("=")[2] if reply else None


def mlock_limit_bytes() -> int | None:
    soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    return None if soft == resource.RLIM_INFINITY else soft


def fan_rpm() -> int | None:
    for path in Path("/sys/devices/platform/cooling_fan").glob("hwmon/*/fan1_input"):
        return read_int(str(path))
    return None


def facts(storage_path: Path) -> dict[str, Any]:
    mem = meminfo()
    cpufreq = "/sys/devices/system/cpu/cpu0/cpufreq"
    temp_milli = read_int("/sys/class/thermal/thermal_zone0/temp")
    clock = vcgencmd("measure_clock", "arm")
    disk = shutil.disk_usage(storage_path)
    load = os.getloadavg()

    return {
        "machine": {
            "model": read_text("/proc/device-tree/model"),
            "kernel": platform.release(),
            "os": platform.freedesktop_os_release().get("PRETTY_NAME"),
        },
        "cpu": {
            "model": next(
                (
                    line.partition(":")[2].strip()
                    for line in (run("lscpu") or "").splitlines()
                    if line.startswith("Model name")
                ),
                None,
            ),
            "cores": os.cpu_count(),
            "features": cpu_features(),
            "min_khz": read_int(f"{cpufreq}/cpuinfo_min_freq"),
            "max_khz": read_int(f"{cpufreq}/cpuinfo_max_freq"),
            "governor": read_text(f"{cpufreq}/scaling_governor"),
            "caches": caches(),
        },
        "memory": {
            "total_bytes": mem.get("MemTotal"),
            "available_bytes": mem.get("MemAvailable"),
            "swap_total_bytes": mem.get("SwapTotal"),
            "swap_free_bytes": mem.get("SwapFree"),
            "mlock_limit_bytes": mlock_limit_bytes(),
        },
        "storage": {
            "path": str(storage_path),
            "total_bytes": disk.total,
            "free_bytes": disk.free,
        },
        "thermal": {
            "temp_c": temp_milli / 1000 if temp_milli is not None else None,
            "arm_clock_hz": int(clock) if clock and clock.isdigit() else None,
            "throttled": vcgencmd("get_throttled"),
            "fan_rpm": fan_rpm(),
        },
        "load": {"1m": load[0], "5m": load[1], "15m": load[2]},
    }


def main() -> None:
    storage_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home()
    json.dump(facts(storage_path), sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
