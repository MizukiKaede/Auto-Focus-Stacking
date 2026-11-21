"""Tests for 4 GiB shared cache default, auto merge-worker scaling, cache UI controls, and low-memory mode."""

from __future__ import annotations

import os
import pytest

from focus_stack_app.config import LOW_MEMORY_PRESET, NORMAL_MEMORY_PRESET, RuntimeConfig
from focus_stack_app.fusion.aligned_cache import AlignedFrameCache, AlignedTIFFImageCache
from focus_stack_app.pipeline.application_controller import ApplicationController, ApplicationOptions
from focus_stack_app.pipeline.coordinator import PipelineConfig, PipelineCoordinator
from focus_stack_app.utils.fusion_budget import (
    DEFAULT_AUTO_MERGE_WORKER_BYTES,
    resolve_merge_worker_budget,
)
from focus_stack_app.utils.memory import MemorySnapshot


GIB = 1024**3
MIB = 1024**2


def _snap(*, total_gib: float, avail_gib: float) -> MemorySnapshot:
    total = int(total_gib * GIB)
    avail = int(avail_gib * GIB)
    return MemorySnapshot(total_bytes=total, available_bytes=avail, used_bytes=total - avail)


def test_runtime_defaults_use_4gib_shared_cache_and_2gib_group_cache():
    rt = RuntimeConfig()
    assert rt.fusion_cache_budget_bytes == 4 * GIB
    assert rt.opencv_aligned_cache_bytes == 2 * GIB
    assert rt.min_available_memory_bytes == 2 * GIB
    assert rt.min_available_memory_fraction == 0.10
    assert rt.max_hugin_workers == 3
    assert rt.auto_merge_workers is True
    assert rt.low_memory_mode is False


def test_auto_merge_worker_budget_scales_down_by_available_memory():
    # 32 GiB host, 20 GiB free, reserve = 3.2 GiB -> capacity 6 -> capped at requested 3
    high = resolve_merge_worker_budget(3, auto_reduce=True, snapshot=_snap(total_gib=32, avail_gib=20))
    assert high.effective_workers == 3

    # 16 GiB host, 8.5 GiB free, reserve = 2.0 GiB -> (6.5 // 2.5) = 2
    mid = resolve_merge_worker_budget(3, auto_reduce=True, snapshot=_snap(total_gib=16, avail_gib=8.5))
    assert mid.effective_workers == 2
    assert mid.measured_worker_peak_bytes == 0

    # 8 GiB host, 3.0 GiB free, reserve = 2.0 GiB -> capacity 0 -> floor 1
    tight = resolve_merge_worker_budget(3, auto_reduce=True, snapshot=_snap(total_gib=8, avail_gib=3.0))
    assert tight.effective_workers == 1

    # Disabled auto_reduce preserves requested concurrency even when RAM is tight
    fixed = resolve_merge_worker_budget(3, auto_reduce=False, snapshot=_snap(total_gib=8, avail_gib=3.0))
    assert fixed.effective_workers == 3


def test_low_memory_mode_applies_1gib_redline_and_conservative_cache_caps(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()
    ctl = ApplicationController(
        ApplicationOptions(
            source_dir=src,
            output_dir=out,
            low_memory_mode=True,
        )
    )
    assert ctl.config.runtime.low_memory_mode is True
    assert ctl.config.runtime.min_available_memory_bytes == 1 * GIB
    assert ctl.config.runtime.min_available_memory_fraction == 0.0
    assert ctl.config.runtime.fusion_cache_budget_bytes == LOW_MEMORY_PRESET["fusion_cache_budget_bytes"]
    assert ctl.config.runtime.opencv_aligned_cache_bytes == LOW_MEMORY_PRESET["opencv_aligned_cache_bytes"]
    assert ctl.config.analysis.selection_frame_cache_bytes == LOW_MEMORY_PRESET["selection_frame_cache_bytes"]
    assert ctl.config.runtime.aligned_tiff_cache_bytes == LOW_MEMORY_PRESET["aligned_tiff_cache_bytes"]

    # Explicit UI cache overrides still take precedence even in low-memory mode
    ctl_custom = ApplicationController(
        ApplicationOptions(
            source_dir=src,
            output_dir=out,
            low_memory_mode=True,
            fusion_cache_budget_bytes=int(1.5 * GIB),
            opencv_aligned_cache_bytes=768 * MIB,
        )
    )
    assert ctl_custom.config.runtime.min_available_memory_bytes == 1 * GIB
    assert ctl_custom.config.runtime.fusion_cache_budget_bytes == int(1.5 * GIB)
    assert ctl_custom.config.runtime.opencv_aligned_cache_bytes == 768 * MIB


def test_aligned_caches_respect_custom_1gib_redline():
    snap = _snap(total_gib=16, avail_gib=2.5)
    # Default 2 GiB reserve leaves 0.5 GiB spare -> 256 MiB cache limit
    normal_cache = AlignedFrameCache(lambda i: None, max_bytes=2 * GIB, snapshot_fn=lambda: snap)
    assert normal_cache.reserve_bytes == 2 * GIB
    assert normal_cache.max_bytes == 256 * MIB

    # Low-memory 1 GiB redline (fraction=0) leaves 1.5 GiB spare -> 768 MiB (clamped by 512 MiB cap)
    low_cache = AlignedFrameCache(
        lambda i: None,
        max_bytes=512 * MIB,
        snapshot_fn=lambda: snap,
        minimum_available_bytes=1 * GIB,
        minimum_available_fraction=0.0,
    )
    assert low_cache.reserve_bytes == 1 * GIB
    assert low_cache.max_bytes == 512 * MIB

    low_tiff = AlignedTIFFImageCache(
        max_bytes=256 * MIB,
        snapshot_fn=lambda: snap,
        minimum_available_bytes=1 * GIB,
        minimum_available_fraction=0.0,
    )
    assert low_tiff.reserve_bytes == 1 * GIB
    assert low_tiff.max_bytes == 256 * MIB


def test_coordinator_auto_merge_workers_without_extra_memory_guard_block():
    calls = []

    class Guard:
        minimum_bytes = 2 * GIB
        minimum_fraction = 0.10

        def wait(self, _cancel, **kwargs):
            calls.append(dict(kwargs))
            return True

    coord = PipelineCoordinator(
        config=PipelineConfig(
            queue_size=2,
            merge_workers=3,
            parallel=False,
            auto_merge_workers=True,
            auto_merge_worker_bytes=DEFAULT_AUTO_MERGE_WORKER_BYTES,
        ),
        analyzer=lambda _g: {"needs_merge": True},
        merger=lambda job: {"status": "DONE", "group": job.group},
        memory_guard=Guard(),
    )
    summary = coord.run([{"id": 1}])
    assert summary.finished
    # extra_memory_bytes must stay 0 so the final worker is never deadlocked by auto down-scaling
    assert all(w.extra_memory_bytes == 0 for w in coord._merge_workers)


def test_main_window_memory_controls_and_low_memory_preset_toggle(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from focus_stack_app.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow(restore_settings=False)
    try:
        # Normal defaults
        assert window.low_memory_check.isChecked() is False
        assert window.merge_workers_spin.value() == 3
        assert window.auto_merge_workers_check.isChecked() is True
        assert window.parallel_check.isChecked() is True
        assert window.focus_analysis_workers_spin.value() == 0
        assert window.global_cache_spin.value() == pytest.approx(4.0)
        assert window.group_cache_spin.value() == pytest.approx(2.0)
        assert "2.0 GiB" in window.memory_redline_label.text()

        # Enable low-memory mode -> switches to LOW_MEMORY_PRESET
        window.low_memory_check.setChecked(True)
        app.processEvents()
        assert window.merge_workers_spin.value() == 1
        assert window.auto_merge_workers_check.isChecked() is True
        assert window.parallel_check.isChecked() is False
        assert window.focus_analysis_workers_spin.value() == 1
        assert window.global_cache_spin.value() == pytest.approx(1.0)
        assert window.group_cache_spin.value() == pytest.approx(0.5)
        assert "1.0 GiB" in window.memory_redline_label.text()

        # Verify _settings() carries exact values to ApplicationController
        window.source_edit.setText(str(tmp_path))
        settings = window._settings()
        assert settings["low_memory_mode"] is True
        assert settings["auto_merge_workers"] is True
        assert settings["fusion_cache_budget_bytes"] == 1 * GIB
        assert settings["opencv_aligned_cache_bytes"] == 512 * MIB
        ctl = ApplicationController(settings)
        assert ctl.config.runtime.min_available_memory_bytes == 1 * GIB
        assert ctl.config.runtime.min_available_memory_fraction == 0.0
        assert ctl.config.runtime.fusion_cache_budget_bytes == 1 * GIB
        assert ctl.config.runtime.opencv_aligned_cache_bytes == 512 * MIB

        # Custom tweak on UI spinboxes propagates cleanly
        window.low_memory_check.setChecked(False)
        app.processEvents()
        assert window.merge_workers_spin.value() == 3
        assert window.global_cache_spin.value() == pytest.approx(4.0)
        assert window.group_cache_spin.value() == pytest.approx(2.0)

        window.global_cache_spin.setValue(6.0)
        window.group_cache_spin.setValue(2.5)
        window.auto_merge_workers_check.setChecked(False)
        settings2 = window._settings()
        ctl2 = ApplicationController(settings2)
        assert ctl2.config.runtime.fusion_cache_budget_bytes == 6 * GIB
        assert ctl2.config.runtime.opencv_aligned_cache_bytes == int(2.5 * GIB)
        assert ctl2.options.auto_merge_workers is False
        assert ctl2.config.runtime.min_available_memory_bytes == 2 * GIB

        # Reset defaults returns to NORMAL_MEMORY_PRESET
        window._reset_defaults()
        assert window.low_memory_check.isChecked() is False
        assert window.merge_workers_spin.value() == NORMAL_MEMORY_PRESET["merge_workers"]
        assert window.auto_merge_workers_check.isChecked() is True
        assert window.global_cache_spin.value() == pytest.approx(4.0)
        assert window.group_cache_spin.value() == pytest.approx(2.0)
    finally:
        window.close()
