import pytest

from focus_stack_app.config import AppConfig, RuntimeConfig
from focus_stack_app.pipeline.application_controller import ApplicationOptions
from focus_stack_app.utils.concurrency import (
    opencv_thread_budget,
    resolve_focus_analysis_budget,
)
from focus_stack_app.utils.memory import MemorySnapshot


def test_auto_budget_matches_twenty_thread_machine_and_reserves_merges():
    budget = resolve_focus_analysis_budget(
        0,
        image_count=26,
        merge_workers=2,
        parallel_pipeline=True,
        snapshot=MemorySnapshot(32 * 1024**3, 20 * 1024**3, 12 * 1024**3),
        logical_cpus=20,
        current_opencv_threads=20,
    )
    assert budget.workers == 4
    assert budget.opencv_threads == 4
    assert budget.cpu_budget == 16
    assert not budget.memory_limited


def test_manual_budget_is_an_upper_bound_and_low_memory_degrades_to_one():
    budget = resolve_focus_analysis_budget(
        8,
        image_count=3,
        merge_workers=1,
        snapshot=MemorySnapshot(32 * 1024**3, 2 * 1024**3, 30 * 1024**3),
        logical_cpus=20,
        current_opencv_threads=20,
    )
    assert budget.workers == 1
    assert budget.memory_limited
    assert budget.memory_worker_cap == 1


def test_focus_worker_options_validate_and_config_roundtrips(tmp_path):
    options = ApplicationOptions("in", "out", focus_analysis_workers=6)
    assert options.focus_analysis_workers == 6
    with pytest.raises(ValueError, match="focus_analysis_workers"):
        ApplicationOptions("in", "out", focus_analysis_workers=9)
    config = AppConfig(runtime=RuntimeConfig(focus_analysis_workers=7))
    path = tmp_path / "config.json"
    config.save(path)
    assert AppConfig.load(path).runtime.focus_analysis_workers == 7


def test_opencv_thread_budget_restores_previous_value():
    cv2 = pytest.importorskip("cv2")
    previous = cv2.getNumThreads()
    with opencv_thread_budget(2) as state:
        assert state == {"previous": previous, "effective": 2}
        assert cv2.getNumThreads() == 2
    assert cv2.getNumThreads() == previous
