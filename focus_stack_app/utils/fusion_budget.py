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


DEFAULT_AUTO_MERGE_WORKER_BYTES = int(2.5 * 1024**3)
LOW_MEMORY_AUTO_MERGE_WORKER_BYTES = int(1.5 * 1024**3)


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


def resolve_merge_worker_budget(
    requested,
    measured_peak=0,
    *,
    auto_reduce=False,
    per_worker_bytes=DEFAULT_AUTO_MERGE_WORKER_BYTES,
    snapshot=None,
    minimum_bytes=2 * 1024**3,
    minimum_fraction=0.10,
):
    """Resolve effective merge concurrency, optionally scaling down by available RAM.

    When ``measured_peak`` is explicitly provided (calibrated streaming mode),
    the calibrated budget always takes precedence.  Otherwise, enabling
    ``auto_reduce`` scales ``requested`` down toward 1 based on ``per_worker_bytes``
    without imposing extra per-job MemoryGuard blocking on the final worker.
    """
    if int(measured_peak) > 0 or not auto_reduce:
        return quality_worker_budget(
            requested,
            measured_peak,
            snapshot=snapshot,
            minimum_bytes=minimum_bytes,
            minimum_fraction=minimum_fraction,
        )
    if (not 1 <= int(requested) <= 6 or int(per_worker_bytes) <= 0
            or minimum_bytes < 0 or not 0 <= minimum_fraction <= 1):
        raise ValueError("auto merge worker budget requires workers 1..6 and positive per_worker_bytes")
    current = snapshot or memory_snapshot()
    reserve = max(int(minimum_bytes), int(current.total_bytes * minimum_fraction))
    capacity = max(0, current.available_bytes - reserve) // int(per_worker_bytes)
    effective = max(1, min(int(requested), capacity))
    return FusionWorkerBudget(
        int(requested),
        effective,
        current.available_bytes,
        reserve,
        0,
        "auto memory-aware merge worker limit",
    )
