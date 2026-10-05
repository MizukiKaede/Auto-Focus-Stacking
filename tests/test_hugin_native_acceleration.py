"""Focused correctness checks for Hugin's optional native and parallel paths."""
from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
import threading
from types import SimpleNamespace
import logging

import cv2
import numpy as np
from PIL import Image
import pytest

import focus_stack_app
from focus_stack_app.config import AppConfig, RuntimeConfig
from focus_stack_app.fusion import fast_cpp
from focus_stack_app.fusion import quality_fusion
from focus_stack_app.hugin import alignment_refinement, enfuse, exterior_ownership, parallel


_CANDIDATE_SRC = Path(__file__).resolve().parents[1]
assert Path(focus_stack_app.__file__).resolve().is_relative_to(_CANDIDATE_SRC), (
    "This test must import the candidate source tree, not the primary checkout"
)


def _tensor_reference(xx, yy, xy, energy):
    coherence = np.sqrt((xx - yy) ** 2 + 4.0 * xy * xy) / np.maximum(xx + yy, 1e-8)
    return (energy > (6.0 / 255.0) ** 2) & (coherence < 0.6)


def test_hugin_optional_kernels_are_loaded_separately_and_keep_legacy_abi():
    assert fast_cpp._library.available, fast_cpp._library.reason
    assert fast_cpp._library.abi == 3
    for name in ("structure_tensor_texture", "hard_mask"):
        library = fast_cpp._hugin_libraries[name]
        assert library.available, {"kernel": name, "reason": library.reason}
        assert library.abi == 3


def test_frozen_abi3_library_still_loads_the_existing_exports():
    frozen = _CANDIDATE_SRC.parent.parent / "before" / "src" / "focus_stack_app" / "fusion" / "fast_core.dll"
    if not frozen.is_file():
        pytest.skip("frozen-source DLL compatibility check is only available in the validation checkout")
    from focus_stack_app.fusion.native_runtime import NativeLibrary

    library = NativeLibrary(
        frozen,
        fast_cpp._signatures,
        abi_name="fast_core_abi",
        expected_abi=3,
    )
    assert library.available, library.reason
    assert library.abi == 3
    assert set(library.functions) == set(fast_cpp._signatures)


def test_structure_tensor_texture_matches_the_old_float32_formula_exactly():
    rng = np.random.default_rng(961)
    xx = rng.uniform(0.0, 0.9, (53, 71)).astype(np.float32)
    yy = rng.uniform(0.0, 0.9, xx.shape).astype(np.float32)
    xy = rng.uniform(-0.6, 0.6, xx.shape).astype(np.float32)
    energy = rng.uniform(0.0, 0.004, xx.shape).astype(np.float32)
    expected = _tensor_reference(xx, yy, xy, energy)

    actual = fast_cpp.structure_tensor_texture(xx, yy, xy, energy)

    assert actual.dtype == np.bool_
    assert actual.shape == xx.shape
    np.testing.assert_array_equal(actual, expected)


def test_structure_tensor_texture_uses_strict_energy_and_coherence_thresholds():
    xx = np.array([[0.0, 1.0, 0.8, 0.8]], dtype=np.float32)
    yy = np.array([[0.0, 1.0, 0.2, 0.2]], dtype=np.float32)
    xy = np.zeros_like(xx)
    threshold = np.float32((6.0 / 255.0) ** 2)
    energy = np.array(
        [[threshold, np.nextafter(threshold, np.float32(np.inf)), 1.0, 1.0]],
        dtype=np.float32,
    )

    actual = fast_cpp.structure_tensor_texture(xx, yy, xy, energy)

    np.testing.assert_array_equal(actual, _tensor_reference(xx, yy, xy, energy))
    assert not actual[0, 0]  # Equality at the energy threshold is excluded.
    assert actual[0, 1]  # Zero coherence and energy just above the threshold.
    assert not actual[0, 2]  # Coherence exactly on the strict 0.6 boundary.


def test_structure_tensor_texture_rejects_noncontiguous_or_mismatched_inputs():
    xx = np.zeros((8, 12), dtype=np.float32)[:, ::2]
    contiguous = np.zeros(xx.shape, dtype=np.float32)
    with pytest.raises(ValueError, match="contiguous"):
        fast_cpp.structure_tensor_texture(xx, contiguous, contiguous, contiguous)
    with pytest.raises(ValueError, match="expected shape"):
        fast_cpp.structure_tensor_texture(
            np.zeros((4, 5), np.float32), np.zeros((5, 4), np.float32),
            np.zeros((4, 5), np.float32), np.zeros((4, 5), np.float32),
        )
    with pytest.raises(ValueError, match="dtype"):
        fast_cpp.structure_tensor_texture(
            np.zeros((4, 5), np.float64), np.zeros((4, 5), np.float32),
            np.zeros((4, 5), np.float32), np.zeros((4, 5), np.float32),
        )


def test_hard_mask_returns_exact_255_binary_plane():
    labels = np.array(
        [[0, 1, 2, 65535], [2, 2, 1, 0], [65535, 0, 1, 2]],
        dtype=np.uint16,
    )
    for index in (0, 1, 2, 65535, 19):
        mask = fast_cpp.hard_mask(labels, index)
        assert mask.dtype == np.uint8
        assert mask.shape == labels.shape
        np.testing.assert_array_equal(mask, np.where(labels == index, 255, 0).astype(np.uint8))
        assert set(np.unique(mask)).issubset({0, 255})


def test_hard_mask_rejects_invalid_stride_and_indices():
    labels = np.zeros((6, 12), dtype=np.uint16)[:, ::2]
    with pytest.raises(ValueError, match="contiguous"):
        fast_cpp.hard_mask(labels, 0)
    contiguous = np.zeros((3, 4), dtype=np.uint16)
    for index in (-1, 65536, 1.5):
        with pytest.raises(ValueError):
            fast_cpp.hard_mask(contiguous, index)


@pytest.mark.parametrize("kernel_name", ["structure_tensor_texture", "hard_mask"])
def test_optional_hugin_kernel_numpy_fallback_matches_native(kernel_name, monkeypatch):
    library = fast_cpp._hugin_libraries[kernel_name]
    assert library.available, library.reason
    if kernel_name == "structure_tensor_texture":
        rng = np.random.default_rng(1105)
        xx = rng.uniform(0.01, 0.8, (43, 67)).astype(np.float32)
        yy = rng.uniform(0.01, 0.8, xx.shape).astype(np.float32)
        xy = rng.uniform(-0.3, 0.3, xx.shape).astype(np.float32)
        energy = rng.uniform(0.0, 0.003, xx.shape).astype(np.float32)
        call = lambda: fast_cpp.structure_tensor_texture(xx, yy, xy, energy)
    else:
        labels = np.arange(43 * 67, dtype=np.uint16).reshape(43, 67) % 13
        call = lambda: fast_cpp.hard_mask(labels, 7)
    native = call()
    monkeypatch.setattr(library, "dll", None)
    fallback = call()
    np.testing.assert_array_equal(fallback, native)


def test_exterior_rim_ownership_routes_both_support_fields_through_fast_cpp(monkeypatch):
    calls = []

    def native_support(strength, seeds, radius):
        calls.append((strength.dtype, seeds.dtype, int(radius)))
        return np.zeros_like(strength, dtype=np.float32)

    monkeypatch.setattr(fast_cpp, "nearest_support", native_support)
    monkeypatch.setattr(
        quality_fusion,
        "_nearest_edge_support",
        lambda *_args, **_kwargs: pytest.fail("Python quality-fusion support was called"),
    )
    rgb = np.full((100, 100, 3), 220, dtype=np.uint8)
    rgb[18:82, 18:82] = (205, 70, 45)

    exterior_ownership.ExteriorRimOwnership(maximum_pixels=4096).observe(0, rgb)

    assert len(calls) == 2
    assert all(dtype == np.float32 for dtype, _, _ in calls)
    assert all(seed_dtype == np.uint8 for _, seed_dtype, _ in calls)
    assert calls[0][2] == calls[1][2]


def test_bounded_map_caps_weighted_inflight_jobs_and_propagates_contextvars(monkeypatch):
    context = ContextVar("hugin_validation_context", default="unset")
    condition = threading.Condition()
    release = threading.Event()
    active = 0
    peak_active = 0
    started = []
    pulled_threads = []
    results = []
    failure = []
    caller_thread = []
    diagnostics = []
    monkeypatch.setattr(parallel, "diagnostic", lambda name, **fields: diagnostics.append((name, fields)))

    def items():
        for value in range(9):
            pulled_threads.append(threading.get_ident())
            yield value

    def work(value):
        nonlocal active, peak_active
        with condition:
            active += 1
            peak_active = max(peak_active, active)
            started.append(value)
            condition.notify_all()
        try:
            assert release.wait(5), "test did not release the bounded workers"
            return value, context.get(), threading.get_ident()
        finally:
            with condition:
                active -= 1
                condition.notify_all()

    def consume():
        token = context.set("consumer-context")
        caller_thread.append(threading.get_ident())
        try:
            results.extend(parallel.bounded_map(
                work, items(), workers=4, cpu_cost=3, cpu_budget=6,
                working_bytes=1024,
            ))
        except BaseException as error:
            failure.append(error)
        finally:
            context.reset(token)

    consumer = threading.Thread(target=consume, name="bounded-map-caller")
    consumer.start()
    try:
        with condition:
            reached_two = condition.wait_for(lambda: active >= 2, timeout=5)
        assert reached_two, "two weighted jobs did not start within the CPU budget"
        assert len(started) == 2
        assert len(pulled_threads) == 2, "the input iterator was advanced beyond the bounded window"
        assert parallel._shared_budget.capacity <= 6
    finally:
        release.set()
        consumer.join(timeout=10)

    assert not consumer.is_alive()
    assert not failure
    assert [row[0] for row in results] == list(range(9))
    assert all(row[1] == "consumer-context" for row in results)
    assert peak_active == 2  # cost=3, budget=6, regardless of workers=4.
    assert all(thread_id == caller_thread[0] for thread_id in pulled_threads)
    assert all(row[2] != caller_thread[0] for row in results)
    assert parallel._shared_budget.used == 0
    assert parallel._shared_budget.bytes_used == 0
    execution = [fields for name, fields in diagnostics if name == "hugin_preparation_execution"]
    assert execution[-1]["peak_active_tasks"] == 2


@pytest.mark.parametrize(("cpu_cost", "expected_tasks"), [(4, 3), (3, 4), (1, 12)])
def test_bounded_map_enforces_one_process_wide_weighted_pool(monkeypatch, cpu_cost, expected_tasks):
    from types import SimpleNamespace

    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(
        parallel,
        "memory_snapshot",
        lambda: SimpleNamespace(total_bytes=64 * 1024**3, available_bytes=64 * 1024**3),
    )
    entered = threading.Condition()
    release = threading.Event()
    active_tasks = 0
    active_units = 0
    peak_tasks = 0
    peak_units = 0
    errors = []

    def run_one():
        nonlocal active_tasks, active_units, peak_tasks, peak_units

        def work(value):
            nonlocal active_tasks, active_units, peak_tasks, peak_units
            with entered:
                active_tasks += 1
                active_units += cpu_cost
                peak_tasks = max(peak_tasks, active_tasks)
                peak_units = max(peak_units, active_units)
                entered.notify_all()
            try:
                assert release.wait(8)
                return value
            finally:
                with entered:
                    active_tasks -= 1
                    active_units -= cpu_cost
                    entered.notify_all()

        try:
            assert list(parallel.bounded_map(
                work, range(expected_tasks), workers=12, cpu_cost=cpu_cost, cpu_budget=12,
                working_bytes=1024,
            )) == list(range(expected_tasks))
        except BaseException as error:
            errors.append(error)

    # Concurrent fusion groups share the same 12-unit process budget.
    threads = [threading.Thread(target=run_one) for _ in range(2)]
    for thread in threads:
        thread.start()
    try:
        with entered:
            assert entered.wait_for(lambda: active_tasks >= expected_tasks, timeout=6)
            assert active_tasks == expected_tasks
            assert active_units == 12
        assert parallel._shared_budget.capacity == 12
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert not errors
    assert peak_tasks == expected_tasks
    assert peak_units == 12
    assert parallel._shared_budget.used == 0
    assert parallel._shared_budget.bytes_used == 0


def test_bounded_map_cancellation_failure_and_budget_validation():
    advanced = []
    cancelled = threading.Event()
    cancelled.set()

    def items():
        advanced.append(True)
        yield 1

    with pytest.raises(InterruptedError):
        list(parallel.bounded_map(lambda value: value, items(), workers=2,
                                  cpu_budget=2, cancel_event=cancelled))
    assert advanced == []

    mid_cancel = threading.Event()

    def cancel_after_one(value):
        mid_cancel.set()
        return value

    with pytest.raises(InterruptedError):
        list(parallel.bounded_map(cancel_after_one, range(8), workers=1,
                                  cpu_budget=1, cancel_event=mid_cancel))
    assert parallel._shared_budget.used == 0
    assert parallel._shared_budget.bytes_used == 0

    def fail(value):
        if value == 2:
            raise LookupError("injected Hugin preparation failure")
        return value

    with pytest.raises(LookupError, match="injected Hugin"):
        list(parallel.bounded_map(fail, range(4), workers=1, cpu_budget=1))
    assert parallel._shared_budget.used == 0
    assert parallel._shared_budget.bytes_used == 0

    for valid in range(1, 13):
        assert parallel.validate_budget(valid) == valid
    for invalid in (0, 13, True, 3.0):
        with pytest.raises(ValueError):
            parallel.validate_budget(invalid)
    assert RuntimeConfig().hugin_parallel_cpu_budget == 12
    assert AppConfig.from_mapping({"runtime": {}}).runtime.hugin_parallel_cpu_budget == 12
    assert AppConfig.from_mapping({"runtime": {"max_hugin_workers": 3}}).runtime.hugin_parallel_cpu_budget == 12
    assert RuntimeConfig(hugin_parallel_cpu_budget=6).hugin_parallel_cpu_budget == 6
    assert parallel.validate_budget(6) == 6
    for invalid in (0, 13, True, 3.0):
        with pytest.raises(ValueError):
            RuntimeConfig(hugin_parallel_cpu_budget=invalid)


def test_bounded_map_clamps_large_budget_to_small_host_and_one_is_serial(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        parallel, "memory_snapshot",
        lambda: SimpleNamespace(total_bytes=32 * 1024**3, available_bytes=32 * 1024**3),
    )
    diagnostics = []
    monkeypatch.setattr(parallel, "diagnostic", lambda name, **fields: diagnostics.append((name, fields)))

    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 2)
    assert list(parallel.bounded_map(lambda value: value, range(5), workers=12, cpu_budget=12)) == list(range(5))
    small_host = [fields for name, fields in diagnostics if name == "hugin_preparation_budget"][-1]
    assert small_host["cpu_budget"] == 2
    assert small_host["workers"] == 2

    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 12)
    assert list(parallel.bounded_map(lambda value: value, range(3), workers=12, cpu_budget=1)) == [0, 1, 2]
    serial = [fields for name, fields in diagnostics if name == "hugin_preparation_budget"][-1]
    assert serial["cpu_budget"] == 1
    assert serial["workers"] == 1


def test_bounded_map_stops_pulling_after_failure_and_releases_weighted_budget(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        parallel,
        "memory_snapshot",
        lambda: SimpleNamespace(total_bytes=32 * 1024**3, available_bytes=16 * 1024**3),
    )
    diagnostics = []
    monkeypatch.setattr(parallel, "diagnostic", lambda name, **fields: diagnostics.append((name, fields)))
    started_failure = threading.Event()
    release_first = threading.Event()
    pulled = []
    errors = []

    def items():
        for item in range(10):
            pulled.append(item)
            yield item

    def work(item):
        if item == 0:
            assert release_first.wait(5)
            return item
        if item == 1:
            started_failure.set()
            raise LookupError("injected parallel worker failure")
        return item

    def consume():
        try:
            list(parallel.bounded_map(
                work, items(), workers=2, cpu_cost=2, cpu_budget=4, working_bytes=4096,
            ))
        except BaseException as exc:
            errors.append(exc)

    consumer = threading.Thread(target=consume, name="bounded-map-failure-caller")
    consumer.start()
    try:
        assert started_failure.wait(5)
        assert pulled == [0, 1], "no replacement job may be submitted after a worker failure"
    finally:
        release_first.set()
        consumer.join(timeout=10)

    assert not consumer.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], LookupError)
    assert pulled == [0, 1]
    assert parallel._shared_budget.used == 0
    assert parallel._shared_budget.bytes_used == 0
    execution = [fields for name, fields in diagnostics if name == "hugin_preparation_execution"]
    assert execution[-1]["peak_active_tasks"] == 2


def _gray_stack(frame_count=6):
    return [np.full((48, 64, 3), 20 * (index + 1), dtype=np.uint8)
            for index in range(frame_count)]


def _fake_ecc_from_intensity(*, rejected_index=None, calls=None):
    active = 0
    peak = 0
    thread_ids = set()
    lock = threading.Lock()

    def find_transform(_template, image, initial, _motion, _criteria):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            thread_ids.add(threading.get_ident())
        try:
            frame = int(round(float(np.mean(image)) * 255.0 / 20.0)) - 1
            if calls is not None:
                calls.append(frame)
            affine = np.eye(2, 3, dtype=np.float32)
            affine[0, 2] = np.float32((frame + 1) * 0.7)
            affine[1, 2] = np.float32(-(frame + 1) * 0.35)
            correlation = 0.97 if frame == rejected_index else 0.995
            return correlation, affine
        finally:
            with lock:
                active -= 1

    return find_transform, lambda: (peak, set(thread_ids))


@pytest.mark.parametrize("cpu_budget", [3, 6, 12])
def test_alignment_refiner_runs_ecc_serially_by_index_on_caller_thread(monkeypatch, cpu_budget):
    frames = _gray_stack(frame_count=7)
    main_thread = threading.get_ident()
    loader_threads = []
    monkeypatch.setattr(cv2, "getNumThreads", lambda: 3)

    def loader_for(calls):
        def load(index):
            calls.append(index)
            loader_threads.append(threading.get_ident())
            return frames[index]
        return load

    ecc_calls = []
    fake_ecc, stats = _fake_ecc_from_intensity(calls=ecc_calls)
    monkeypatch.setattr(alignment_refinement.cv2, "findTransformECC", fake_ecc)

    def fail_if_shared_budget_is_used(*_args, **_kwargs):
        raise AssertionError("per-stack ECC must not enter the shared preparation budget")

    monkeypatch.setattr(parallel, "bounded_map", fail_if_shared_budget_is_used)
    diagnostics = []
    monkeypatch.setattr(
        alignment_refinement, "diagnostic",
        lambda name, **fields: diagnostics.append((name, fields)),
    )
    loads = []
    refiner = alignment_refinement.HuginAlignmentRefiner(
        loader_for(loads), 2, cpu_budget=cpu_budget,
    )
    refiner.prepare(len(frames))

    expected = [i for i in range(len(frames)) if i != 2]
    assert loads == [2, *expected]
    assert loader_threads and set(loader_threads) == {main_thread}
    max_active, ecc_threads = stats()
    assert ecc_calls == expected
    assert max_active == 1
    assert ecc_threads == {main_thread}
    assert refiner.cpu_budget == cpu_budget
    assert refiner.prepared_count == len(frames)
    assert set(refiner.matrices) == set(range(len(frames)))
    execution = [fields for name, fields in diagnostics if name == "hugin_residual_execution"]
    assert execution == [{"execution": "serial_per_stack", "shared_preparation_budget": False,
                          "opencv_threads": 3}]


def test_alignment_refiner_rejection_resets_the_complete_stack_to_identity(monkeypatch):
    frames = _gray_stack()
    monkeypatch.setattr(cv2, "getNumThreads", lambda: 3)
    fake_ecc, _ = _fake_ecc_from_intensity(rejected_index=4)
    monkeypatch.setattr(alignment_refinement.cv2, "findTransformECC", fake_ecc)
    refiner = alignment_refinement.HuginAlignmentRefiner(
        lambda index: frames[index], 2, cpu_budget=3,
    )

    refiner.prepare(len(frames))

    identity = np.eye(3, dtype=np.float32)
    assert refiner.prepared_count == len(frames)
    assert set(refiner.matrices) == set(range(len(frames)))
    assert all(not event["accepted"] for event in refiner.events.values())
    assert refiner.events[4]["candidate_accepted"] is False
    for matrix in refiner.matrices.values():
        np.testing.assert_array_equal(matrix, identity)


def test_alignment_refiner_cancel_or_loader_failure_does_not_commit_partial_results(monkeypatch):
    frames = _gray_stack()
    monkeypatch.setattr(cv2, "getNumThreads", lambda: 3)
    fake_ecc, _ = _fake_ecc_from_intensity()
    monkeypatch.setattr(alignment_refinement.cv2, "findTransformECC", fake_ecc)

    calls = []
    cancelled = threading.Event()
    cancelled.set()
    refiner = alignment_refinement.HuginAlignmentRefiner(
        lambda index: calls.append(index) or frames[index], 2, cpu_budget=3,
    )
    with pytest.raises(InterruptedError):
        refiner.prepare(len(frames), cancel_event=cancelled)
    assert calls == [2]
    assert set(refiner.matrices) == {2}
    assert refiner.events == {}
    assert refiner.prepared_count is None

    def broken_loader(index):
        if index == 3:
            raise OSError("injected TIFF loader failure")
        return frames[index]

    broken = alignment_refinement.HuginAlignmentRefiner(broken_loader, 2, cpu_budget=3)
    with pytest.raises(OSError, match="injected TIFF"):
        broken.prepare(len(frames))
    assert set(broken.matrices) == {2}
    assert broken.events == {}
    assert broken.prepared_count is None


def test_enfuse_hard_masks_use_fast_masks_and_keep_deflate_tiff_pixels(monkeypatch, tmp_path):
    labels = np.array(
        [[0, 1, 2, 0], [2, 2, 1, 1], [0, 2, 1, 0]], dtype=np.uint16,
    )
    observed = {}
    real_map = parallel.bounded_map

    def record_map(fn, items, **kwargs):
        observed.update(kwargs)
        return real_map(fn, items, **kwargs)

    monkeypatch.setattr(parallel, "bounded_map", record_map)
    enfuse._write_hard_masks(labels, 3, tmp_path, cpu_budget=3)

    assert observed["workers"] == 3
    assert observed["cpu_budget"] == 3
    for index in range(3):
        path = tmp_path / f"hardmask-{index + 1}.tif"
        assert path.is_file()
        with Image.open(path) as image:
            assert image.tag_v2.get(259) == 8  # TIFF Adobe Deflate.
            actual = np.asarray(image).copy()
        np.testing.assert_array_equal(actual, np.where(labels == index, 255, 0).astype(np.uint8))


def test_enfuse_hard_masks_keep_two_digit_templates_for_fifty_frames(monkeypatch, tmp_path):
    labels = np.arange(6 * 8, dtype=np.uint16).reshape(6, 8) % 50
    enfuse._write_hard_masks(labels, 50, tmp_path, cpu_budget=1)

    assert sorted(path.name for path in tmp_path.glob("hardmask-*.tif")) == [
        f"hardmask-{index:02d}.tif" for index in range(1, 51)
    ]


def test_enfuse_hard_mask_cancel_and_failure_are_propagated(monkeypatch, tmp_path):
    labels = np.arange(16, dtype=np.uint16).reshape(4, 4) % 3
    original = fast_cpp.hard_mask
    event = threading.Event()

    def cancel_after_first(labels_plane, index):
        value = original(labels_plane, index)
        event.set()
        return value

    monkeypatch.setattr(fast_cpp, "hard_mask", cancel_after_first)
    cancel_work = tmp_path / "cancel"
    cancel_work.mkdir()
    with pytest.raises(enfuse.EnfuseError, match="cancelled while generating focus masks"):
        enfuse._write_hard_masks(labels, 3, cancel_work, cpu_budget=1,
                                  cancel_event=event)
    assert list(cancel_work.glob("*.tif")) == []

    def fail_on_second(_labels, index):
        if index == 1:
            raise RuntimeError("injected native hard-mask failure")
        return original(_labels, index)

    monkeypatch.setattr(fast_cpp, "hard_mask", fail_on_second)
    failure_work = tmp_path / "failure"
    failure_work.mkdir()
    with pytest.raises(RuntimeError, match="native hard-mask failure"):
        enfuse._write_hard_masks(labels, 3, failure_work, cpu_budget=1)
