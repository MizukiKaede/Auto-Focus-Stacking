from collections import Counter
import threading

import cv2
import numpy as np
import pytest
from PIL import Image

from focus_stack_app.config import AppConfig, RuntimeConfig
from focus_stack_app.fusion import aligned_cache, backends, focus_masks
from focus_stack_app.hugin.output_encoder import OutputConfig, OutputCollisionError
from focus_stack_app.pipeline.merge_worker import StackMergeService
from focus_stack_app.utils.memory import MemorySnapshot


HIGH_MEMORY = MemorySnapshot(32 * 1024**3, 20 * 1024**3, 12 * 1024**3)
LOW_MEMORY = MemorySnapshot(32 * 1024**3, 1024**3, 31 * 1024**3)


def test_partial_cache_avoids_sequential_scan_thrashing_and_releases_consumed_frames():
    calls = Counter()
    def loader(index):
        calls[index] += 1
        return np.full((10, 10, 3), index, np.uint8)
    cache = aligned_cache.AlignedFrameCache(loader, max_bytes=600, snapshot_fn=lambda: HIGH_MEMORY)
    for index in range(5):
        cache.for_focus(index)
        assert cache.bytes_used <= 600
    for index in range(5):
        np.testing.assert_array_equal(cache.for_blend(index), np.full((10, 10, 3), index, np.uint8))
    assert calls == {0: 1, 1: 1, 2: 2, 3: 2, 4: 2}
    assert cache.bytes_used == 0
    assert cache.frames == {}


def test_memory_pressure_releases_cache_and_recomputes_same_pixels():
    current = [HIGH_MEMORY]
    cache = aligned_cache.AlignedFrameCache(
        lambda i: np.full((10, 10, 3), i, np.uint8), max_bytes=1000,
        snapshot_fn=lambda: current[0],
    )
    expected = cache.for_focus(7)
    assert cache.bytes_used > 0
    current[0] = LOW_MEMORY
    np.testing.assert_array_equal(cache.for_blend(7), expected)
    assert cache.bytes_used == cache.max_bytes == 0


@pytest.mark.parametrize('snapshot', [LOW_MEMORY, MemorySnapshot(0, 0, 0), None])
def test_unavailable_or_low_memory_disables_cache(snapshot):
    def read_memory():
        if snapshot is None:
            raise OSError('unavailable')
        return snapshot
    cache = aligned_cache.AlignedFrameCache(lambda i: np.zeros((5, 5, 3), np.uint8), snapshot_fn=read_memory)
    cache.for_focus(0)
    assert cache.bytes_used == cache.max_bytes == 0


def make_analysis(tmp_path):
    rng = np.random.default_rng(42)
    sharp = rng.integers(0, 256, (80, 160, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 3)
    left, right = sharp.copy(), blurred.copy()
    left[:, 80:] = blurred[:, 80:]
    right[:, 80:] = sharp[:, 80:]
    paths = [tmp_path / 'left.png', tmp_path / 'right.png']
    for path, data in zip(paths, (left, right)):
        Image.fromarray(data).save(path)
    return {'selected_paths': paths, 'selected_indices': [0, 1],
            'preview_reference': paths[0], 'preview_reference_index': 0,
            'preview_transforms': [np.eye(3).tolist()] * 2,
            'analysis_shapes': [[80, 160]] * 2, 'reference_analysis_shape': [80, 160]}


def test_cached_partial_and_uncached_outputs_are_identical_without_previews_or_tiff(tmp_path, monkeypatch):
    analysis = make_analysis(tmp_path)
    monkeypatch.setattr(aligned_cache, 'memory_snapshot', lambda: HIGH_MEMORY)
    original_load = backends.load_rgb
    loads = []
    def load(path, long_edge=None):
        assert long_edge is None, 'known registration must not decode unused previews'
        loads.append(path)
        return original_load(path)
    monkeypatch.setattr(backends, 'load_rgb', load)
    payloads = []
    frame_bytes = 80 * 160 * 3
    for limit, expected_reads in ((0, 5), (frame_bytes, 3), (frame_bytes * 2, 2)):
        loads.clear()
        result = backends.QualityFusionBackend(aligned_cache_bytes=limit).fuse(
            {}, analysis, tmp_path / f'{limit}.jpg', tmp_path / f'work-{limit}', OutputConfig(), threading.Event(),
        )
        payloads.append(result.output_path.read_bytes())
        assert len(loads) == expected_reads
    assert payloads[0] == payloads[1] == payloads[2]
    assert not list(tmp_path.rglob('*.tif'))


@pytest.mark.parametrize('failure', ['cancel', 'error'])
def test_failed_or_cancelled_fusion_releases_retained_frames(tmp_path, monkeypatch, failure):
    analysis = make_analysis(tmp_path)
    monkeypatch.setattr(aligned_cache, 'memory_snapshot', lambda: HIGH_MEMORY)
    created = []
    original = aligned_cache.AlignedFrameCache
    def cache(*args, **kwargs):
        value = original(*args, **kwargs)
        created.append(value)
        return value
    monkeypatch.setattr(aligned_cache, 'AlignedFrameCache', cache)
    def stop(*args, **kwargs):
        assert created[0].bytes_used > 0
        if failure == 'cancel':
            kwargs['cancel_event'].set()
        raise RuntimeError(failure)
    monkeypatch.setattr(focus_masks, 'blend_focus_pyramid', stop)
    with pytest.raises(RuntimeError, match=failure):
        backends.QualityFusionBackend().fuse({}, analysis, tmp_path/'out.jpg', tmp_path/'work', OutputConfig(), threading.Event())
    assert not created[0].frames
    assert created[0].bytes_used == 0
    assert not (tmp_path/'out.jpg').exists()


def test_direct_fusion_cannot_overwrite_any_selected_source(tmp_path):
    analysis = make_analysis(tmp_path)
    target = analysis['selected_paths'][1]
    before = target.read_bytes()
    with pytest.raises(OutputCollisionError, match='input image'):
        backends.QualityFusionBackend().fuse({}, analysis, target, tmp_path/'work', OutputConfig(overwrite=True), threading.Event())
    assert target.read_bytes() == before


def test_runtime_cache_limit_reaches_quality_backend_and_roundtrips(tmp_path):
    config = AppConfig(runtime=RuntimeConfig(opencv_aligned_cache_bytes=0))
    config.save(tmp_path/'config.json')
    loaded = AppConfig.load(tmp_path/'config.json')
    service = StackMergeService(tmp_path/'out', fusion_backend='quality', runtime=loaded)
    assert service.backend.aligned_cache_bytes == 0
    with pytest.raises(ValueError, match='opencv_aligned_cache_bytes'):
        RuntimeConfig(opencv_aligned_cache_bytes=-1)

