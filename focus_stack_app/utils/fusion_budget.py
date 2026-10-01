"""Apply a measured worker budget without inventing a high-pixel calibration."""
from __future__ import annotations

from dataclasses import dataclass
from .memory import memory_snapshot


@dataclass(frozen=True)
class FusionWorkerBudget:
    requested_workers: int
    effective_workers: int
    available_bytes: int
    reserved_bytes: int
    measured_worker_peak_bytes: int
    reason: str


def quality_worker_budget(requested, measured_peak=0, *, snapshot=None,
                          minimum_bytes=2 * 1024**3, minimum_fraction=0.10):
    if (not 1 <= int(requested) <= 6 or int(measured_peak) < 0
            or minimum_bytes < 0 or not 0 <= minimum_fraction <= 1):
        raise ValueError("fusion worker budget requires workers 1..6 and a nonnegative measured peak")
    current = snapshot or memory_snapshot()
    reserve = max(int(minimum_bytes), int(current.total_bytes * minimum_fraction))
    if not measured_peak:
        return FusionWorkerBudget(requested, requested, current.available_bytes, reserve, 0,
                                  "60MP worker calibration pending; configured concurrency retained")
    capacity = max(0, current.available_bytes - reserve) // int(measured_peak)
    effective = max(1, min(requested, capacity))
    return FusionWorkerBudget(requested, effective, current.available_bytes, reserve, int(measured_peak),
                              "measured 60MP worker peak; each fusion also waits for memory headroom")
