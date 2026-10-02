"""Regression coverage for partial aligned-frame-cache pressure recovery.

The snapshots and RGB data are synthetic; these tests do not read photographs
or depend on the machine's current available memory.
"""
from dataclasses import dataclass

import numpy as np

from focus_stack_app.fusion.aligned_cache import AlignedFrameCache
from focus_stack_app.utils.shared_cache_budget import SharedCacheBudget


GIB = 1024**3
TOTAL_BYTES = 40 * GIB
RESERVE_BYTES = 4 * GIB
SAFE_AVAILABLE_BYTES = RESERVE_BYTES + 4 * GIB


@dataclass(frozen=True)
class FakeMemorySnapshot:
    total_bytes: int
    available_bytes: int
    used_bytes: int


class SnapshotSequence:
    def __init__(self, available_values):
        self.available_values = list(available_values)
        self.calls = 0
        self.last = None

    def __call__(self):
        index = self.calls
        self.calls += 1
        value = (self.available_values[index]
                 if index < len(self.available_values)
                 else SAFE_AVAILABLE_BYTES)
        if isinstance(value, BaseException):
            raise value
        value = int(value)
        self.last = FakeMemorySnapshot(TOTAL_BYTES, value, TOTAL_BYTES - value)
        return self.last


def _rgb_frame(index):
    return np.arange(index * 12, index * 12 + 12, dtype=np.uint8).reshape(2, 2, 3)


def _filled_cache(after_fill_snapshots):
    # The constructor takes one snapshot; three focus reads take two each.
    # This creates a deterministic three-frame cache under a 64-byte limit.
    snapshots = SnapshotSequence(
        [SAFE_AVAILABLE_BYTES] * 7 + list(after_fill_snapshots))
    pool = SharedCacheBudget(1024)
    loader_calls = []

    def loader(index):
        loader_calls.append(index)
        return _rgb_frame(index)

    cache = AlignedFrameCache(loader, max_bytes=64, snapshot_fn=snapshots,
                              shared_budget=pool)
    assert cache.max_bytes == 64
    assert cache.reserve_bytes == RESERVE_BYTES
    originals = {}
    for index in range(3):
        originals[index] = cache.for_focus(index).copy()
    assert snapshots.calls == 7
    assert list(cache.frames) == [0, 1, 2]
    assert cache.bytes_used == pool.bytes_used == 36
    return cache, pool, snapshots, loader_calls, originals


def test_small_deficit_keeps_near_prefix_without_duplicate_decode():
    cache, pool, snapshots, loader_calls, originals = _filled_cache(
        [RESERVE_BYTES - 1, RESERVE_BYTES + 1])

    cache._check_memory()

    assert snapshots.calls == 9  # low trigger, then fresh recovered snapshot
    assert snapshots.last.available_bytes >= cache.reserve_bytes
    assert list(cache.frames) == [0, 1]
    assert cache.max_bytes == cache.bytes_used == 24
    assert pool.bytes_used == 24

    for index in (0, 1):
        assert np.array_equal(cache.for_focus(index), originals[index])

    # A later healthy snapshot must not restore the old 64-byte ceiling or
    # reinsert the evicted tail. The returned frame remains usable by callers.
    assert np.array_equal(cache.for_focus(3), _rgb_frame(3))
    assert list(cache.frames) == [0, 1]
    assert cache.max_bytes == 24
    assert cache.bytes_used == pool.bytes_used == 24

    assert np.array_equal(cache.for_blend(0), originals[0])
    assert np.array_equal(cache.for_blend(1), originals[1])
    assert cache.bytes_used == pool.bytes_used == 0
    assert loader_calls == [0, 1, 2, 3]


def test_multiple_tail_releases_wait_for_snapshot_recovery():
    cache, pool, snapshots, loader_calls, originals = _filled_cache(
        [RESERVE_BYTES - 20, RESERVE_BYTES - 10, RESERVE_BYTES + 1])

    cache._check_memory()

    # The first freed frame did not restore headroom. The cache only stops
    # after the next existing snapshot reports availability above reserve.
    assert snapshots.calls == 10
    assert snapshots.last.available_bytes >= cache.reserve_bytes
    assert list(cache.frames) == [0]
    assert cache.max_bytes == cache.bytes_used == pool.bytes_used == 12
    assert np.array_equal(cache.for_blend(0), originals[0])
    assert loader_calls == [0, 1, 2]
    assert pool.bytes_used == 0


def test_cache_is_fully_released_when_headroom_never_returns():
    cache, pool, snapshots, _loader_calls, _originals = _filled_cache(
        [RESERVE_BYTES - 30, RESERVE_BYTES - 20,
         RESERVE_BYTES - 10, RESERVE_BYTES - 1])

    cache._check_memory()

    assert snapshots.calls == 11  # one fresh check after each of 3 releases
    assert snapshots.last.available_bytes < cache.reserve_bytes
    assert cache.frames == {}
    assert cache.bytes_used == 0
    assert cache.max_bytes == 0
    assert pool.bytes_used == 0


def test_snapshot_recheck_exception_clears_remainder_and_shared_reservations():
    cache, pool, snapshots, _loader_calls, _originals = _filled_cache(
        [RESERVE_BYTES - 1, RuntimeError("synthetic memory snapshot failure")])

    cache._check_memory()

    # The newest frame is released before the recheck fails; the fail-closed
    # path then releases the two older reservations exactly once.
    assert snapshots.calls == 9
    assert cache.frames == {}
    assert cache.bytes_used == 0
    assert cache.max_bytes == 0
    assert pool.bytes_used == 0
