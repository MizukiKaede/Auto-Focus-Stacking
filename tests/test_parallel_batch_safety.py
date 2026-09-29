import threading
import os
import pytest

from PIL import Image

from focus_stack_app.files.archiver import FileArchiver
from focus_stack_app.fusion.backends import FusionBackend, FusionResult
from focus_stack_app.pipeline.coordinator import PipelineConfig, PipelineCoordinator
from focus_stack_app.pipeline.memory_guard import MemoryGuard
from focus_stack_app.pipeline.merge_worker import StackMergeService
from focus_stack_app.utils.memory import MemorySnapshot


def test_two_workers_failure_does_not_archive_and_retry_succeeds(tmp_path):
    groups = []
    for gid in (1, 2):
        paths = [tmp_path / f"{gid}_{i}.jpg" for i in range(3)]
        for path in paths:
            Image.new("RGB", (16, 16), "gray").save(path)
        groups.append(dict(id=gid, images=paths))
    barrier = threading.Barrier(2)
    class Backend(FusionBackend):
        name = "test"
        retry = False
        def fuse(self, group, analysis, output_path, *args):
            if not self.retry:
                barrier.wait(timeout=5)
                if group["id"] == 2:
                    raise RuntimeError("not enough control points")
            Image.new("RGB", (16, 16), "gray").save(output_path)
            return FusionResult(output_path, self.name)
    backend = Backend()
    archive = tmp_path / "archive"
    merger = StackMergeService(tmp_path / "output", archiver=FileArchiver(archive, "copy"), fusion_backend=backend,
                               minimum_stack_group_size=3)
    def analyze(group):
        return dict(selected_paths=group["images"], needs_merge=True)
    def run(values):
        return PipelineCoordinator(config=PipelineConfig(merge_workers=2), analyzer=analyze, merger=merger).run(values)
    first = run(groups)
    assert first.errors == 1
    assert len(list(archive.glob("1_*.jpg"))) == 3
    assert not list(archive.glob("2_*.jpg"))
    assert all(path.exists() for group in groups for path in group["images"])
    backend.retry = True
    second = run([groups[1]])
    assert second.errors == 0
    assert len(list(archive.glob("2_*.jpg"))) == 3


def test_two_workers_cancel_under_memory_pressure():
    pressure_seen = threading.Event()
    guard = MemoryGuard(minimum_bytes=100, minimum_fraction=0,
                        snapshot_fn=lambda: MemorySnapshot(1000, 0, 1000),
                        poll_interval=.01, event_callback=lambda event: pressure_seen.set())
    calls = []
    coordinator = PipelineCoordinator(config=PipelineConfig(merge_workers=2),
                                      analyzer=lambda group: group, merger=lambda job: calls.append(job),
                                      memory_guard=guard)
    result = []
    thread = threading.Thread(target=lambda: result.append(coordinator.run([dict(id=1, images=["a", "b", "c"])])))
    thread.start()
    try:
        assert pressure_seen.wait(5)
    finally:
        coordinator.cancel()
        thread.join(5)
    assert not thread.is_alive()
    assert result[0].cancelled
    assert calls == []


@pytest.mark.skipif(not os.environ.get("FOCUS_STACK_ENFUSE"), reason="Enable real Hugin/Enfuse integration explicitly")
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("repeat", range(3))
def test_real_low_texture_alignment_failure_keeps_inputs(tmp_path, workers, repeat):
    """Three repetitions of unalignable low-texture inputs at both concurrencies."""
    groups = []
    for gid in (1, 2):
        paths = [tmp_path / f"{gid}_{i}.jpg" for i in range(3)]
        for index, path in enumerate(paths):
            Image.new("RGB", (128, 96), (50 + index * 50,) * 3).save(path)
        groups.append(dict(id=gid, images=paths))
    archive = tmp_path / "archive"
    merger = StackMergeService(tmp_path / "output", archiver=FileArchiver(archive, "move"),
                               minimum_stack_group_size=3)
    pipeline = PipelineCoordinator(
        config=PipelineConfig(merge_workers=workers), merger=merger,
        analyzer=lambda group: dict(selected_paths=group["images"], alignment_order=group["images"], needs_merge=True))
    summary = pipeline.run(groups)
    assert summary.errors == 2
    assert all(result.status == "FAILED_FUSION" for result in summary.results)
    assert all(path.exists() for group in groups for path in group["images"])
    assert not list(archive.glob("*.jpg"))
    assert not list((tmp_path / "output").glob("*_stack.jpg"))

