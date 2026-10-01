"""Focused tests for the optional measured fusion-worker memory budget."""

from __future__ import annotations

import logging
import threading
import time

import pytest

from focus_stack_app.config import AppConfig, RuntimeConfig
from focus_stack_app.pipeline import application_controller as controller_module
from focus_stack_app.pipeline.application_controller import ApplicationController, ApplicationOptions
from focus_stack_app.pipeline.coordinator import PipelineConfig, PipelineCoordinator
from focus_stack_app.pipeline.events import PipelineStage
from focus_stack_app.pipeline.memory_guard import MemoryGuard
from focus_stack_app.utils.fusion_budget import quality_worker_budget
from focus_stack_app.utils.memory import MemorySnapshot


GIB = 1024**3


def _snapshot(*, total_gib: int, available_gib: int) -> MemorySnapshot:
    total = total_gib * GIB
    available = available_gib * GIB
    return MemorySnapshot(total_bytes=total, available_bytes=available, used_bytes=total - available)


def test_unmeasured_budget_keeps_requested_workers_and_reports_pending_calibration():
    budget = quality_worker_budget(6, 0, snapshot=_snapshot(total_gib=16, available_gib=1))

    assert budget.requested_workers == 6
    assert budget.effective_workers == 6
    assert budget.measured_worker_peak_bytes == 0
    assert "calibration pending" in budget.reason


def test_measured_low_memory_budget_reduces_workers():
    budget = quality_worker_budget(6, 1 * GIB, snapshot=_snapshot(total_gib=16, available_gib=5))

    # reserve=2 GiB, leaving capacity for three 1 GiB workers.
    assert budget.effective_workers == 3
    assert budget.measured_worker_peak_bytes == 1 * GIB
    assert "measured 60MP worker peak" in budget.reason


def test_measured_budget_capacity_zero_still_keeps_one_worker():
    budget = quality_worker_budget(6, 8 * GIB, snapshot=_snapshot(total_gib=16, available_gib=2))

    assert budget.effective_workers == 1


def test_pipeline_config_default_three_and_upper_bound_six():
    assert PipelineConfig().merge_workers == 3
    assert PipelineConfig(merge_workers=6).merge_workers == 6
    with pytest.raises(ValueError):
        PipelineConfig(merge_workers=0)
    with pytest.raises(ValueError):
        PipelineConfig(merge_workers=7)


def test_memory_guard_extra_bytes_wait_is_cancellable():
    # Use small explicit units so the test does not depend on host memory.
    low = MemorySnapshot(total_bytes=1_000, available_bytes=100, used_bytes=900)
    guard = MemoryGuard(
        minimum_bytes=100,
        minimum_fraction=0.10,
        snapshot_fn=lambda: low,
        poll_interval=0.001,
        notice_interval=60,
    )
    assert guard.has_headroom(snapshot=low)
    assert not guard.has_headroom(snapshot=low, extra_bytes=1)

    cancel = threading.Event()
    result: list[bool] = []
    thread = threading.Thread(
        target=lambda: result.append(guard.wait(cancel, stage=PipelineStage.FUSION, extra_bytes=1)),
        daemon=True,
    )
    thread.start()
    time.sleep(0.02)
    cancel.set()
    thread.join(1)

    assert not thread.is_alive()
    assert result == [False]


def test_coordinator_places_extra_memory_on_merge_worker_and_logs_reason(caplog):
    calls: list[dict[str, object]] = []

    class RecordingGuard:
        def wait(self, _cancel_event, **kwargs):
            calls.append(dict(kwargs))
            return True

    measured = 123
    coordinator = PipelineCoordinator(
        config=PipelineConfig(queue_size=1, merge_workers=1, parallel=False),
        analyzer=lambda _group: {"needs_merge": True},
        merger=lambda job: {"status": "DONE", "group": job.group},
        memory_guard=RecordingGuard(),
        measured_worker_peak_bytes=measured,
    )
    caplog.set_level(logging.INFO, logger="focus_stack_app.pipeline.coordinator")

    summary = coordinator.run([{"id": 1}])

    assert summary.finished
    assert coordinator.effective_workers == 1
    assert coordinator._analysis_worker is not None
    assert not hasattr(coordinator._analysis_worker, "extra_memory_bytes")
    assert len(coordinator._merge_workers) == 1
    assert coordinator._merge_workers[0].extra_memory_bytes == measured
    analysis_calls = [call for call in calls if call.get("stage") == PipelineStage.ANALYSIS]
    fusion_calls = [call for call in calls if call.get("stage") == PipelineStage.FUSION]
    assert analysis_calls
    assert all(call.get("extra_bytes", 0) == 0 for call in analysis_calls)
    assert len(fusion_calls) == 1
    assert fusion_calls[0].get("extra_bytes") == measured
    assert "reason=measured 60MP worker peak" in caplog.text


@pytest.mark.parametrize(
    ("backend", "execution", "expected"),
    [
        ("quality", "streaming", 123),
        ("quality", "cached", 0),
        ("hugin_enfuse", "streaming", 0),
    ],
)
def test_controller_only_passes_calibration_to_quality_streaming(
    tmp_path, monkeypatch, backend, execution, expected
):
    captured: list[dict[str, object]] = []

    class StubCoordinator:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(controller_module, "PipelineCoordinator", StubCoordinator)
    config = AppConfig(
        runtime=RuntimeConfig(
            quality_execution=execution,
            quality_jpeg_decoder="opencv" if execution != "cached" else "pillow",
            quality_worker_peak_60mp_bytes=123,
        )
    )
    controller = ApplicationController(
        ApplicationOptions(
            source_dir=tmp_path / "source",
            output_dir=tmp_path / "output",
            fusion_backend=backend,
        ),
        config=config,
    )

    controller._build_coordinator(object(), object())

    assert len(captured) == 1
    assert captured[0]["measured_worker_peak_bytes"] == expected


def test_runtime_default_does_not_invent_60mp_calibration():
    assert RuntimeConfig().quality_worker_peak_60mp_bytes == 0
