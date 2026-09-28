from contextlib import nullcontext
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from focus_stack_app.core import alignment_order, registration
from focus_stack_app.core.registration_cache import PreprocessingCache, _current, preprocessing_cache


@pytest.fixture(autouse=True)
def available_memory(monkeypatch):
    from focus_stack_app.utils.memory import MemorySnapshot
    monkeypatch.setattr("focus_stack_app.utils.memory.memory_snapshot",
                        lambda: MemorySnapshot(32 * 1024**3, 16 * 1024**3, 16 * 1024**3))


@pytest.mark.parametrize("kind", ["texture", "blank", "unrelated"])
def test_graph_results_identical_with_and_without_cache(monkeypatch, kind):
    rng = np.random.default_rng(11)
    first = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
    if kind == "blank":
        first[:] = 127
    images = [first, np.roll(first, 3, axis=1), cv2.GaussianBlur(first, (3, 3), 0)]
    if kind == "unrelated":
        images[-1] = rng.integers(0, 256, first.shape, dtype=np.uint8)
    originals = [image.copy() for image in images]
    config = registration.RegistrationConfig(analysis_long_edge=100)
    cv2.setRNGSeed(7)
    with monkeypatch.context() as patch:
        patch.setattr(alignment_order, "preprocessing_cache", nullcontext)
        expected = alignment_order.analyze_pairwise_registration(images, config=config)
    cv2.setRNGSeed(7)
    actual = alignment_order.analyze_pairwise_registration(images, config=config)
    assert [asdict(e) for e in actual] == [asdict(e) for e in expected]
    for original, image in zip(originals, images):
        np.testing.assert_array_equal(original, image)
    assert _current.get() is None


def test_reuses_resize_gray_and_lazy_features(monkeypatch):
    calls = dict(resize=0, gray=0, features=0)
    for name, key in [("resize_for_analysis", "resize"), ("_gray_u8_uncached", "gray")]:
        original = getattr(registration, name)
        def counted(*args, _fn=original, _key=key, **kwargs):
            calls[_key] += 1
            return _fn(*args, **kwargs)
        monkeypatch.setattr(registration, name, counted)
    class Detector:
        def detectAndCompute(self, *args):
            calls["features"] += 1
            return [], None
    monkeypatch.setattr(cv2, "AKAZE_create", Detector, raising=False)
    images = [np.full((96, 128, 3), i, np.uint8) for i in range(4)]
    # Flat frames force the AKAZE fallback on all six edges.
    alignment_order.analyze_pairwise_registration(images)
    assert calls == dict(resize=4, gray=4, features=4)


def test_bound_and_exception_cleanup():
    cache = PreprocessingCache(100)
    for i in range(20):
        source = np.zeros(30, np.uint8)
        cache.get("copy", source, source.copy)
        assert cache.bytes_used <= 100
    with pytest.raises(RuntimeError), preprocessing_cache() as scoped:
        scoped.get("copy", source, source.copy)
        raise RuntimeError("cancel")
    assert scoped.bytes_used == 0
    assert not scoped.entries
    assert _current.get() is None


def test_contexts_are_isolated_between_threads():
    import threading
    barrier = threading.Barrier(2)
    def run(_):
        with preprocessing_cache() as cache:
            barrier.wait(timeout=5)
            assert _current.get() is cache
            return cache
    with ThreadPoolExecutor(2) as executor:
        caches = list(executor.map(run, range(2)))
    assert caches[0] is not caches[1]


def test_low_memory_disables_cache_without_changing_values(monkeypatch):
    from focus_stack_app.utils.memory import MemorySnapshot
    monkeypatch.setattr("focus_stack_app.utils.memory.memory_snapshot",
                        lambda: MemorySnapshot(32 * 1024**3, 1024**3, 31 * 1024**3))
    source = np.arange(100, dtype=np.uint8)
    with preprocessing_cache() as cache:
        assert cache.max_bytes == 0
        np.testing.assert_array_equal(cache.get("copy", source, source.copy), source)
        assert not cache.entries

