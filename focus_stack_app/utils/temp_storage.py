"""Prefer a detected SSD for Hugin's private, disposable TIFF workspace."""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import tempfile


@lru_cache(maxsize=1)
def windows_drive_media():
    if os.name != "nt":
        return {}
    # Read-only storage inventory. Unsupported/storage-space mappings remain
    # unknown; never infer SSD from a drive letter or a fast cached read.
    script = (
        "$physical = Get-PhysicalDisk; "
        "$rows = @(Get-Partition | Where-Object DriveLetter | ForEach-Object { "
        "$partition = $_; $disk = $partition | Get-Disk; "
        "$match = @($physical | Where-Object { [string]$_.DeviceId -eq [string]$disk.Number }); "
        "if ($match.Count -eq 1) { [pscustomobject]@{ "
        "drive = [string]$partition.DriveLetter; media = [string]$match[0].MediaType } } }); "
        "ConvertTo-Json -InputObject $rows -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=True,
        )
        rows = json.loads(result.stdout)
        if isinstance(rows, dict):
            rows = [rows]
        return {str(row["drive"]).upper(): str(row["media"]).upper() for row in rows}
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        return {}


def select_hugin_temp_root(cache_dir, *, explicit=None):
    cache_root = Path(cache_dir) / "temp"
    media = windows_drive_media()

    def kind(path):
        return media.get(Path(path).drive.rstrip(":").upper(), "UNKNOWN")

    if explicit:
        return Path(explicit).expanduser().resolve(), "explicit", kind(explicit)
    if kind(cache_root) == "SSD":
        return cache_root, "cache_on_ssd", "SSD"
    candidates = [Path(tempfile.gettempdir()) / "focus_stack_hugin",
                  Path(__file__).resolve().parents[2] / ".stack_hugin_temp"]
    for candidate in candidates:
        if kind(candidate) == "SSD":
            return candidate, "detected_ssd", "SSD"
    return cache_root, "no_detected_ssd_keep_cache", kind(cache_root)
