"""Static behavioral checks for the isolated Hugin refined-frame cache candidate.

The candidate is loaded under a private package name so these tests exercise
the isolated implementation without changing or shadowing the application
package used by the rest of the test suite.
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def candidate_package():
    imported_package = importlib.import_module("focus_stack_app")
    package_root = Path(imported_package.__file__).resolve().parent
    package_name = f"_hugin_cache_candidate_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    try:
        yield SimpleNamespace(
            name=package_name,
            enfuse=importlib.import_module(f"{package_name}.hugin.enfuse"),
            process=importlib.import_module(f"{package_name}.hugin.process"),
            repair=importlib.import_module(f"{package_name}.hugin.focus_repair"),
            focus_masks=importlib.import_module(f"{package_name}.fusion.focus_masks"),
            quality_fusion=importlib.import_module(f"{package_name}.fusion.quality_fusion"),
            aligned_cache=importlib.import_module(f"{package_name}.fusion.aligned_cache"),
            shared_budget=importlib.import_module(f"{package_name}.utils.shared_cache_budget"),
        )
    finally:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)


def _make_harness(candidate_package, monkeypatch, tmp_path, *, repair_failure=None):
    package = candidate_package
    height, width, frame_count = 4, 5, 2
    images = [
        np.full((height, width, 3), 30 + index * 50, dtype=np.uint8)
        for index in range(frame_count)
    ]
    original_images = [image.copy() for image in images]
    records = {
        "prepare": [], "refined_load": [], "focus": [], "repair_observe": [],
        "repair_apply": [], "tone": [], "diagnostics": [], "commands": [],
    }
    caches = []
    budget = package.shared_budget.SharedCacheBudget(1024)

    class FakeRefiner:
        shape = (height, width)

        def __init__(self, consumer_loader):
            self.consumer_loader = consumer_loader
            self.prepared = True
            self.matrices = {index: np.eye(3, dtype=np.float32) for index in range(frame_count)}

        def load(self, index):
            assert self.prepared
            records["refined_load"].append(index)
            return self.consumer_loader(index)

    def prepare_refiner(consumer_loader, preparation_loader, reference_index, count,
                        *, cpu_budget=12, cancel_event=None):
        assert cpu_budget in (6, 12)
        records["prepare_cpu_budget"] = cpu_budget
        records["prepare_cancel_event"] = cancel_event
        assert preparation_loader is not None
        prep_order = [reference_index, *(i for i in range(count) if i != reference_index)]
        for index in prep_order:
            preparation_loader(index)
            records["prepare"].append(index)
        refiner = FakeRefiner(consumer_loader)
        records["refiner"] = refiner
        return refiner

    monkeypatch.setattr(package.enfuse, "_prepare_alignment_refiner", prepare_refiner)

    def build_labels(count, loader, **kwargs):
        assert count == frame_count
        observer = kwargs.get("frame_observer")
        for index in range(count):
            rgb = loader(index)
            assert rgb is images[index]
            records["focus"].append(index)
            if observer is not None:
                observer(index, rgb, np.zeros((height, width), np.uint8),
                         np.zeros((height, width), np.float32))
        return np.tile(np.arange(width, dtype=np.uint16) % count, (height, 1))

    monkeypatch.setattr(package.focus_masks, "build_focus_labels", build_labels)
    monkeypatch.setattr(
        package.quality_fusion,
        "stabilize_neutral_labels",
        lambda labels, _reference: labels,
    )

    class FakeRepair:
        def __init__(self, *_args, **_kwargs):
            self.boundary = object()
            self.tone = object()
            self.texture = None

        def observe(self, index, rgb, _gray, _score):
            assert rgb is images[index]
            records["repair_observe"].append(index)

        def apply(self, labels, *, load_aligned):
            for index in range(frame_count):
                assert load_aligned(index) is images[index]
                records["repair_apply"].append(index)
            return labels

        def corrected_inputs(self, paths, load_aligned, _work_dir, *, cancel_event=None,
                             force_rewrite=False, tone_tiff_compression="tiff_deflate"):
            assert force_rewrite is True
            assert cancel_event is not None or repair_failure != "cancel"
            records["tone_tiff_compression"] = tone_tiff_compression
            for index in range(frame_count):
                rgb = load_aligned(index)
                assert rgb is images[index]
                records["tone"].append(index)
                if repair_failure == "error" and index == 0:
                    raise RuntimeError("injected repair failure")
                if repair_failure == "cancel" and index == 0:
                    cancel_event.set()
                    raise package.enfuse.EnfuseError("Enfuse cancelled while correcting material tone")
            return tuple(paths)

    monkeypatch.setattr(package.repair, "HuginFocusRepair", FakeRepair)

    real_cache = package.aligned_cache.AlignedFrameCache

    def create_cache(loader, *, max_bytes, working_bytes, shared_budget):
        refiner = records.get("refiner")
        assert refiner is not None and refiner.prepared
        assert loader.__self__ is refiner
        cache = real_cache(
            loader,
            max_bytes=max_bytes,
            working_bytes=working_bytes,
            snapshot_fn=lambda: SimpleNamespace(
                total_bytes=16 * 1024**3,
                available_bytes=16 * 1024**3,
            ),
            shared_budget=shared_budget,
        )
        caches.append(cache)
        records["cache_config"] = {
            "max_bytes": max_bytes,
            "working_bytes": working_bytes,
            "shared_budget": shared_budget,
        }
        return cache

    monkeypatch.setattr(package.aligned_cache, "AlignedFrameCache", create_cache)
    monkeypatch.setattr(
        package.enfuse,
        "diagnostic",
        lambda name, **fields: records["diagnostics"].append((name, fields)),
    )

    input_paths = []
    for index in range(frame_count):
        path = tmp_path / f"frame-{index}.tif"
        path.write_bytes(b"input placeholder")
        input_paths.append(path)

    class FakeRunner:
        def run(self, command, **_kwargs):
            command = tuple(map(str, command))
            records["commands"].append(command)
            temporary = Path(command[command.index("-o") + 1])
            temporary.write_bytes(b"fake Enfuse output")
            return package.process.CommandResult(command, 0)

    enfuser = package.enfuse.Enfuser(
        "fake-enfuse",
        runner=FakeRunner(),
        config=package.enfuse.EnfuseConfig(focus_blend_levels=1),
    )
    return SimpleNamespace(
        enfuser=enfuser,
        paths=input_paths,
        images=images,
        original_images=original_images,
        records=records,
        caches=caches,
        budget=budget,
        height=height,
        width=width,
    )


def _fuse(harness, tmp_path, *, cancel_event=None, edge=True, tone=True,
          tone_tiff_compression="tiff_deflate", hugin_cpu_budget=None):
    kwargs = dict(
        work_dir=tmp_path / "work",
        cleanup_on_success=False,
        image_loader=lambda index: harness.images[index],
        preparation_image_loader=lambda index: harness.images[index],
        refined_frame_cache_bytes=1024,
        refined_frame_cache_shared_budget=harness.budget,
        tone_tiff_compression=tone_tiff_compression,
        focus_mask_mode="quality",
        focus_reference_index=1,
        focus_edge_ownership=edge,
        focus_surface_tone=tone,
        cancel_event=cancel_event,
    )
    if hugin_cpu_budget is not None:
        kwargs["hugin_parallel_cpu_budget"] = hugin_cpu_budget
    return harness.enfuser.fuse(harness.paths, tmp_path / "output.tif", **kwargs)


@pytest.mark.parametrize("compression", ["tiff_deflate", "raw"])
@pytest.mark.parametrize("configured_budget", [None, 6])
def test_refined_cache_starts_after_prepare_and_reuses_focus_repair_and_tone(
    candidate_package, monkeypatch, tmp_path, compression, configured_budget,
):
    harness = _make_harness(candidate_package, monkeypatch, tmp_path)

    result = _fuse(
        harness, tmp_path, tone_tiff_compression=compression,
        hugin_cpu_budget=configured_budget,
    )

    assert result.ok
    assert harness.records["prepare"] == [1, 0]
    assert harness.records["prepare_cpu_budget"] == (12 if configured_budget is None else configured_budget)
    assert harness.records["prepare_cancel_event"] is None
    assert harness.records["focus"] == [0, 1]
    assert harness.records["repair_observe"] == [0, 1]
    assert harness.records["repair_apply"] == [0, 1]
    assert harness.records["tone"] == [0, 1]
    assert harness.records["tone_tiff_compression"] == compression
    # Only the two focus loads reach the refined loader; the stabilization,
    # repair and tone consumers reuse those exact arrays in original order.
    assert harness.records["refined_load"] == [0, 1]
    assert harness.records["cache_config"]["working_bytes"] == harness.height * harness.width * 80
    assert harness.records["cache_config"]["shared_budget"] is harness.budget
    assert harness.budget.peak_bytes == sum(image.nbytes for image in harness.images)
    assert harness.budget.bytes_used == 0
    stats_events = [fields for name, fields in harness.records["diagnostics"]
                    if name == "hugin_refined_frame_cache"]
    assert len(stats_events) == 1
    stats = stats_events[0]
    assert stats["focus_cache_reuse_hits"] == 3
    assert stats["blend_cache_hits"] == 2
    assert stats["duplicate_loader_calls"] == 0
    assert stats["shared_cache_bytes_used"] == 0
    for image, original in zip(harness.images, harness.original_images):
        np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("failure", ["error", "cancel"])
def test_refined_cache_releases_shared_budget_on_repair_error_or_cancellation(
    candidate_package, monkeypatch, tmp_path, failure,
):
    harness = _make_harness(candidate_package, monkeypatch, tmp_path, repair_failure=failure)
    import threading

    cancel_event = threading.Event() if failure == "cancel" else None
    expected = RuntimeError if failure == "error" else candidate_package.enfuse.EnfuseError
    with pytest.raises(expected):
        _fuse(harness, tmp_path, cancel_event=cancel_event)

    assert harness.records["prepare_cpu_budget"] == 12
    assert harness.records["prepare_cancel_event"] is cancel_event
    assert len(harness.caches) == 1
    assert harness.caches[0].bytes_used == 0
    assert harness.budget.bytes_used == 0
    assert harness.budget.peak_bytes == sum(image.nbytes for image in harness.images)
    assert cancel_event is None or cancel_event.is_set()
    assert any(name == "hugin_refined_frame_cache" for name, _ in harness.records["diagnostics"])
    assert harness.records["commands"] == []


def test_no_repair_branch_does_not_construct_or_leak_refined_cache(
    candidate_package, monkeypatch, tmp_path,
):
    harness = _make_harness(candidate_package, monkeypatch, tmp_path)

    def forbidden_cache(*_args, **_kwargs):
        pytest.fail("the refined cache must not be created without a repair/refiner")

    monkeypatch.setattr(candidate_package.aligned_cache, "AlignedFrameCache", forbidden_cache)
    result = _fuse(harness, tmp_path, edge=False, tone=False)

    assert result.ok
    assert harness.records["prepare"] == []
    assert harness.records["refined_load"] == []
    assert harness.caches == []
    assert harness.budget.bytes_used == 0
    assert not any(name == "hugin_refined_frame_cache" for name, _ in harness.records["diagnostics"])


def test_focus_cache_rolls_back_shared_reservation_if_frame_insertion_fails(candidate_package):
    cache_type = candidate_package.aligned_cache.AlignedFrameCache
    budget = candidate_package.shared_budget.SharedCacheBudget(64)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    cache = cache_type(
        lambda _index: image,
        max_bytes=64,
        working_bytes=0,
        snapshot_fn=lambda: SimpleNamespace(total_bytes=16 * 1024**3, available_bytes=16 * 1024**3),
        shared_budget=budget,
    )

    class FailingDict(dict):
        def __setitem__(self, _key, _value):
            raise MemoryError("injected frame-map insertion failure")

    cache.frames = FailingDict()
    with pytest.raises(MemoryError, match="injected frame-map insertion failure"):
        cache.for_focus(0)

    assert budget.bytes_used == 0
    assert cache.bytes_used == 0
    assert cache.cache_insertions == 0


@pytest.mark.parametrize("compression", ["raw", "tiff_deflate"])
def test_corrected_tone_tiff_compression_roundtrips_rgb_without_pixel_changes(
    candidate_package, tmp_path, compression,
):
    rng = np.random.default_rng(2451)
    rgb = rng.integers(0, 256, size=(23, 31, 3), dtype=np.uint8)
    source = tmp_path / "aligned-input.tif"
    source.write_bytes(b"input placeholder")
    repair = candidate_package.repair.HuginFocusRepair(
        0, edge_ownership=False, surface_tone=False,
    )

    written = repair.corrected_inputs(
        [source], lambda _index: rgb, tmp_path / "repair-work",
        force_rewrite=True, tone_tiff_compression=compression,
    )

    with Image.open(written[0]) as encoded:
        decoded = np.asarray(encoded.convert("RGB"))
    np.testing.assert_array_equal(decoded, rgb)


def test_runtime_defaults_to_raw_with_deflate_as_an_explicit_rollback(candidate_package):
    runtime_config = importlib.import_module(f"{candidate_package.name}.config").RuntimeConfig

    assert runtime_config().hugin_tone_tiff_compression == "raw"
    assert runtime_config().aligned_tiff_cache_bytes == 1024**3
    assert runtime_config(hugin_tone_tiff_compression="tiff_deflate").hugin_tone_tiff_compression == "tiff_deflate"
    with pytest.raises(ValueError, match="hugin_tone_tiff_compression"):
        runtime_config(hugin_tone_tiff_compression="jpeg")
