from __future__ import annotations

from collections import Counter
import threading

import numpy as np
import pytest
import cv2
from PIL import Image

from focus_stack_app.fusion.aligned_cache import AlignedFrameCache
from focus_stack_app.fusion import aligned_cache as aligned_cache_module
from focus_stack_app.fusion import backends as backends_module
from focus_stack_app.config import RuntimeConfig
from focus_stack_app.fusion.backends import QualityFusionBackend
from focus_stack_app.hugin.output_encoder import OutputConfig
from focus_stack_app.utils.memory import MemorySnapshot
from focus_stack_app.utils.shared_cache_budget import SharedCacheBudget


HIGH_MEMORY = MemorySnapshot(32 * 1024**3, 20 * 1024**3, 12 * 1024**3)


def test_shared_budget_is_atomic_and_rejects_accounting_underflow():
    budget = SharedCacheBudget(100)
    start = threading.Barrier(8)
    acquired: list[bool] = []
    lock = threading.Lock()

    def reserve():
        start.wait()
        result = budget.try_reserve(60)
        with lock:
            acquired.append(result)

    workers = [threading.Thread(target=reserve) for _ in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert sum(acquired) == 1
    assert budget.bytes_used == budget.peak_bytes == 60
    budget.release(60)
    assert budget.bytes_used == 0
    with pytest.raises(ValueError, match="exceeds reserved"):
        budget.release(1)


def test_shared_budget_can_tighten_while_active_and_grow_after_release():
    budget = SharedCacheBudget(120)
    assert budget.try_reserve(100)
    assert budget.configure_cap(50) == 50
    assert budget.configure_cap(90) == 50
    assert budget.requested_max_bytes == 90
    assert not budget.try_reserve(1)

    budget.release(100)
    assert budget.configure_cap(90) == 90


def test_aligned_frame_cache_rolls_back_shared_reservation_on_insert_error():
    class RejectInsert(dict):
        def __setitem__(self, _key, _value):
            raise MemoryError("simulated cache metadata allocation failure")

    budget = SharedCacheBudget(300)
    cache = AlignedFrameCache(
        lambda _index: np.zeros((10, 10, 3), np.uint8),
        max_bytes=300, snapshot_fn=lambda: HIGH_MEMORY, shared_budget=budget,
    )
    cache.frames = RejectInsert()

    with pytest.raises(MemoryError, match="simulated cache metadata"):
        cache.for_focus(0)

    assert cache.bytes_used == 0
    assert budget.bytes_used == 0


def test_tiff_cache_rolls_back_shared_reservation_on_insert_error(tmp_path, monkeypatch):
    class RejectInsert(dict):
        def __setitem__(self, _key, _value):
            raise MemoryError("simulated cache metadata allocation failure")

    path = tmp_path / "aligned.tif"
    path.write_bytes(b"mock TIFF")
    frame = np.zeros((10, 10, 3), np.uint8)
    monkeypatch.setattr(aligned_cache_module, "load_rgb", lambda _path: frame)
    budget = SharedCacheBudget(300)
    cache = aligned_cache_module.AlignedTIFFImageCache(
        max_bytes=300, snapshot_fn=lambda: HIGH_MEMORY, shared_budget=budget,
    )
    cache.frames = RejectInsert()

    with pytest.raises(MemoryError, match="simulated cache metadata"):
        cache.peek(path)

    assert cache.bytes_used == 0
    assert budget.bytes_used == 0


def test_fusion_backend_clears_cache_when_diagnostic_raises(monkeypatch):
    class Cache:
        stats = {"cache": "test"}

        def __init__(self):
            self.cleared = False

        def clear(self):
            self.cleared = True

    cache = Cache()

    def fail_diagnostic(*_args, **_kwargs):
        raise RuntimeError("simulated diagnostic sink failure")

    monkeypatch.setattr(backends_module, "diagnostic", fail_diagnostic)
    with pytest.raises(RuntimeError, match="simulated diagnostic sink"):
        backends_module._report_cache_then_clear(cache, "aligned_frame_cache")

    assert cache.cleared


def test_cached_focus_frame_is_reused_by_auxiliary_pass_without_redecode():
    calls: Counter[int] = Counter()

    def loader(index):
        calls[index] += 1
        return np.full((10, 10, 3), index, np.uint8)

    budget = SharedCacheBudget(600)
    cache = AlignedFrameCache(
        loader, max_bytes=600, snapshot_fn=lambda: HIGH_MEMORY,
        shared_budget=budget,
    )
    first = cache.for_focus(0)
    cache.for_focus(1)

    assert cache.for_focus(0) is first
    assert calls == {0: 1, 1: 1}
    assert cache.bytes_used == budget.bytes_used == 600
    assert cache.stats["focus_cache_reuse_hits"] == 1
    assert cache.stats["duplicate_loader_calls"] == 0

    cache.for_blend(0)
    assert budget.bytes_used == 300
    cache.clear()
    assert cache.bytes_used == budget.bytes_used == 0


def test_shared_budget_caps_multiple_cache_instances_and_releases_consumed_frames():
    budget = SharedCacheBudget(600)
    first = AlignedFrameCache(
        lambda index: np.full((10, 10, 3), index, np.uint8),
        max_bytes=600, snapshot_fn=lambda: HIGH_MEMORY, shared_budget=budget,
    )
    second = AlignedFrameCache(
        lambda index: np.full((10, 10, 3), index, np.uint8),
        max_bytes=600, snapshot_fn=lambda: HIGH_MEMORY, shared_budget=budget,
    )

    first.for_focus(0)
    second.for_focus(1)
    first.for_focus(2)  # Local room exists, but the process pool is full.
    assert first.bytes_used == second.bytes_used == 300
    assert budget.bytes_used == budget.max_bytes == 600
    assert first.stats["cache_rejections"] == 1

    first.for_blend(0)
    second.for_focus(2)
    assert budget.bytes_used == 600
    first.clear()
    second.clear()
    assert budget.bytes_used == 0


def test_tiff_preparation_peek_is_non_consuming_and_invalidates_changed_path(
    tmp_path, monkeypatch,
):
    path = tmp_path / "aligned.tif"
    path.write_bytes(b"first")
    shared = SharedCacheBudget(600)
    validation_calls = []
    consumer_calls = []

    def validation_load(_path, edges):
        validation_calls.append(_path)
        frame = np.full((10, 10, 3), 1, np.uint8)
        return frame, {edge: frame[:1, :1] for edge in edges}

    def consumer_load(_path):
        consumer_calls.append(_path)
        return np.full((10, 10, 3), 2, np.uint8)

    monkeypatch.setattr(aligned_cache_module, "load_rgb_with_previews", validation_load)
    monkeypatch.setattr(aligned_cache_module, "load_rgb", consumer_load)
    cache = aligned_cache_module.AlignedTIFFImageCache(
        max_bytes=600, snapshot_fn=lambda: HIGH_MEMORY, shared_budget=shared,
    )
    cache.validation_previews(path)
    first = next(iter(cache.frames.values()))
    assert cache.peek(path) is first
    assert cache.bytes_used == shared.bytes_used == 300

    path.write_bytes(b"a changed TIFF with a different size")
    second = cache.peek(path)
    assert int(second[0, 0, 0]) == 2
    assert second is cache.load(path)
    assert len(validation_calls) == 1
    assert len(consumer_calls) == 1
    assert cache.bytes_used == shared.bytes_used == 0
    assert cache.stats["preparation_peek_hits"] == 1
    assert cache.stats["preparation_peek_misses"] == 1
    assert cache.stats["cache_invalidations"] == 1
    assert cache.stats["cache_hits"] == 1
    assert cache.stats["cache_duplicate_path_decodes"] == 1
    cache.clear()
    assert shared.bytes_used == 0


def test_tiff_memory_pressure_releases_shared_reservation(tmp_path, monkeypatch):
    current = [HIGH_MEMORY]
    path = tmp_path / "aligned.tif"
    path.write_bytes(b"tiff")
    shared = SharedCacheBudget(300)
    frame = np.zeros((10, 10, 3), np.uint8)
    monkeypatch.setattr(
        aligned_cache_module, "load_rgb_with_previews",
        lambda _path, edges: (frame, {edge: frame[:1, :1] for edge in edges}),
    )
    monkeypatch.setattr(aligned_cache_module, "load_rgb", lambda _path: frame)
    cache = aligned_cache_module.AlignedTIFFImageCache(
        max_bytes=300, snapshot_fn=lambda: current[0], shared_budget=shared,
    )
    cache.validation_previews(path)
    assert shared.bytes_used == 300

    current[0] = MemorySnapshot(32 * 1024**3, 1 * 1024**3, 31 * 1024**3)
    cache.peek(path)
    assert cache.max_bytes == cache.bytes_used == shared.bytes_used == 0
    assert cache.frames == {}
    cache.clear()
    assert shared.bytes_used == 0


def test_zero_global_cache_cap_falls_back_and_memory_mode_has_no_cache_stats(
    tmp_path, monkeypatch,
):
    rng = np.random.default_rng(42)
    sharp = rng.integers(0, 256, (80, 160, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 3)
    left, right = sharp.copy(), blurred.copy()
    left[:, 80:] = blurred[:, 80:]
    right[:, 80:] = sharp[:, 80:]
    paths = [tmp_path / "left.png", tmp_path / "right.png"]
    for path, rgb in zip(paths, (left, right)):
        Image.fromarray(rgb).save(path)
    analysis = {
        "selected_paths": paths,
        "selected_indices": [0, 1],
        "preview_reference": paths[0],
        "preview_reference_index": 0,
        "preview_transforms": [np.eye(3).tolist()] * 2,
        "analysis_shapes": [[80, 160]] * 2,
        "reference_analysis_shape": [80, 160],
    }
    monkeypatch.setattr(aligned_cache_module, "memory_snapshot", lambda: HIGH_MEMORY)
    events = []
    monkeypatch.setattr(
        backends_module, "diagnostic",
        lambda name, **details: events.append((name, details)),
    )

    no_cache = QualityFusionBackend(runtime_config=RuntimeConfig(
        quality_execution="cached", quality_jpeg_decoder="pillow",
        quality_variant="legacy", fusion_cache_budget_bytes=0,
    ))
    cached_result = no_cache.fuse(
        {}, analysis, tmp_path / "no-cache.jpg", tmp_path / "no-cache-work",
        OutputConfig(), threading.Event(),
    )
    cache_stats = [details for name, details in events if name == "aligned_frame_cache"]
    assert len(cache_stats) == 1
    assert cache_stats[0]["shared_cache_limit_bytes"] == 0
    assert cache_stats[0]["cache_rejections"] >= 2

    events.clear()
    memory = QualityFusionBackend(runtime_config=RuntimeConfig(
        quality_execution="memory", quality_jpeg_decoder="opencv",
        quality_variant="legacy", fusion_cache_budget_bytes=0,
    ))
    memory_result = memory.fuse(
        {}, analysis, tmp_path / "memory.jpg", tmp_path / "memory-work",
        OutputConfig(), threading.Event(),
    )
    assert cached_result.output_path.read_bytes() == memory_result.output_path.read_bytes()
    assert not any(name == "aligned_frame_cache" for name, _details in events)
