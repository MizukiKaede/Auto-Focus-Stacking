import cv2
import numpy as np
import pytest
from PIL import Image
import threading
import time

from focus_stack_app.core.group_analyzer import GroupAnalyzer, GroupAnalyzerConfig
from focus_stack_app.core.registration import RegistrationConfig
from focus_stack_app.core.types import SceneGroup
from focus_stack_app.storage.database import Database
from focus_stack_app.storage.models import ImageRecord
from focus_stack_app.utils.image_io import load_rgb


def _selection_fixture():
    rng = np.random.default_rng(17)
    truth = rng.integers(0, 256, (180, 260, 3), dtype=np.uint8)
    left = truth.copy()
    right = truth.copy()
    left[:, 130:] = cv2.GaussianBlur(left[:, 130:], (0, 0), 2.0)
    right[:, :130] = cv2.GaussianBlur(right[:, :130], (0, 0), 2.0)
    return {"a": left, "b": truth, "c": right}


def test_selection_pixel_cache_is_exact_and_reduces_decodes_to_n():
    from focus_stack_app.core.whole_frame_selection import select_whole_frame

    pixels = _selection_fixture()
    uncached_calls = []
    cached_calls = []
    uncached = select_whole_frame(
        pixels,
        loader=lambda path, edge: (uncached_calls.append(path), pixels[path])[1],
        frame_cache_bytes=0,
    )
    cached = select_whole_frame(
        pixels,
        loader=lambda path, edge: (cached_calls.append(path), pixels[path])[1],
        frame_cache_bytes=128 * 1024**2,
    )

    for key in (
        "selected_indices", "reference_index", "matrices", "correlations",
        "errors", "preview_shape", "qualities", "coverage", "gains",
    ):
        assert cached[key] == uncached[key]
    assert len(uncached_calls) == 2 * len(pixels) + 1
    assert len(cached_calls) == len(pixels)
    assert cached["decode_count"] == len(pixels)
    assert cached["frame_cache_enabled"] is True


def test_selection_pixel_cache_budget_falls_back_without_partial_reuse():
    from focus_stack_app.core.whole_frame_selection import select_whole_frame

    pixels = _selection_fixture()
    calls = []
    result = select_whole_frame(
        pixels,
        loader=lambda path, edge: (calls.append(path), pixels[path])[1],
        frame_cache_bytes=1,
    )

    assert len(calls) == 2 * len(pixels) + 1
    assert result["frame_cache_enabled"] is False
    assert result["frame_cache_hits"] == 0


def test_selection_cancellation_does_not_decode_or_publish_a_plan():
    from focus_stack_app.core.whole_frame_selection import select_whole_frame

    event = threading.Event()
    event.set()
    calls = []
    with pytest.raises(RuntimeError, match="cancelled"):
        select_whole_frame(
            ["a", "b"], cancel_event=event,
            loader=lambda path, edge: calls.append(path),
        )
    assert calls == []


def test_parallel_selection_is_bounded_and_matches_serial_results():
    from focus_stack_app.core.whole_frame_selection import select_whole_frame

    pixels = _selection_fixture()
    lock = threading.Lock()
    active = 0
    peak = 0

    def loader(path, edge):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.02)
            return pixels[path]
        finally:
            with lock:
                active -= 1

    serial = select_whole_frame(
        pixels, loader=lambda path, edge: pixels[path], workers=1,
        frame_cache_bytes=128 * 1024**2,
    )
    parallel = select_whole_frame(
        pixels, loader=loader, workers=2, requested_workers=0,
        frame_cache_bytes=128 * 1024**2,
    )

    assert 1 < peak <= 2
    assert parallel["analysis_workers_requested"] == 0
    assert parallel["analysis_workers_effective"] == 2
    for key in ("selected_indices", "reference_index", "errors", "preview_shape", "gains"):
        assert parallel[key] == serial[key]
    np.testing.assert_allclose(parallel["matrices"], serial["matrices"], rtol=0, atol=0)
    np.testing.assert_allclose(parallel["correlations"], serial["correlations"], rtol=0, atol=0)
    np.testing.assert_allclose(parallel["qualities"], serial["qualities"], rtol=0, atol=0)
    assert parallel["coverage"] == serial["coverage"]


def test_parallel_selection_midflight_cancel_stops_new_submissions():
    from focus_stack_app.core.whole_frame_selection import select_whole_frame

    event = threading.Event()
    calls = []
    pixels = np.zeros((24, 32, 3), np.uint8)

    def loader(path, edge):
        calls.append(path)
        event.set()
        return pixels

    with pytest.raises(RuntimeError, match="cancelled"):
        select_whole_frame(
            [str(index) for index in range(10)],
            loader=loader,
            workers=3,
            cancel_event=event,
            frame_cache_bytes=0,
        )
    assert 1 <= len(calls) <= 3


def test_selection_refuses_manually_mixed_subject_poses():
    from focus_stack_app.core.whole_frame_selection import select_whole_frame
    front = np.full((480, 640, 3), 220, np.uint8)
    front[120:360, 300:330] = (170, 30, 30)
    back = np.full_like(front, 220)
    back[120:360, 330:360] = (170, 30, 30)
    with pytest.raises(ValueError, match="主体发生转面"):
        select_whole_frame(['front', 'back'], loader=lambda path, edge: front if path == 'front' else back)


def test_group_analyzer_preserves_established_scene_selection(tmp_path):
    truth = np.full((240, 360, 3), 235, np.uint8)
    truth[60:180, 40:320] = (170, 25, 35)
    cv2.putText(truth, "FOCUS", (52, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(truth, "STACK", (190, 157), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
    blurred = cv2.GaussianBlur(truth, (0, 0), 2.5)
    left, right = truth.copy(), truth.copy()
    left[:, 180:] = blurred[:, 180:]
    right[:, :180] = blurred[:, :180]
    paths = [tmp_path / "left.jpg", tmp_path / "right.jpg"]
    for path, pixels in zip(paths, (left, right)):
        Image.fromarray(pixels).save(path, quality=100, subsampling=0)
    records = [ImageRecord.from_path(path, sequence_index=index) for index, path in enumerate(paths)]

    result = GroupAnalyzer(
        GroupAnalyzerConfig(minimum_stack_group_size=2),
    ).analyze(SceneGroup(1, records))

    assert result["selected_indices"] == [0, 1]
    assert result["needs_merge"] is True
    assert result["merge_status"] == "READY_FOR_MERGE"
    assert result["pairwise_analysis_summary"]["strategy"] == "established_whole_frame_ecc"


def test_group_analyzer_skips_focus_analysis_below_minimum_group_size():
    records = [
        ImageRecord.from_path(f"C:/missing/frame_{index}.jpg", sequence_index=index)
        for index in range(2)
    ]
    calls = []

    def loader(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("too-small groups must not decode image pixels")

    result = GroupAnalyzer(
        GroupAnalyzerConfig(minimum_stack_group_size=3),
        loader=loader,
    ).analyze_group(SceneGroup(7, records))

    assert calls == []
    assert result["merge_status"] == "NO_MERGE_TOO_SMALL"
    assert result["needs_merge"] is False
    assert result["selected_indices"] == []
    assert result["analysis_skipped"] is True
    assert result["analysis_skip_reason"] == "NO_MERGE_TOO_SMALL"
    assert result["selection_decode_count"] == 0
    assert result["analysis_workers_effective"] == 0
    assert result["quality_scan_seconds"] == 0.0
    assert result["registration_seconds"] == 0.0


def test_selection_plan_cache_rerun_uses_zero_image_decodes_and_file_change_invalidates(tmp_path):
    pixels = _selection_fixture()
    paths = [tmp_path / f"{name}.jpg" for name in pixels]
    for path, image in zip(paths, pixels.values()):
        Image.fromarray(image).save(path, quality=100, subsampling=0)
    records = [ImageRecord.from_path(path, sequence_index=index) for index, path in enumerate(paths)]
    group = SceneGroup(1, records)
    database = Database(tmp_path / "plans.sqlite")
    first_calls = []

    def first_loader(item, max_long_edge=None):
        first_calls.append((item.path, max_long_edge))
        return load_rgb(item.path, max_long_edge)

    first = GroupAnalyzer(plan_cache=database, loader=first_loader).analyze_group(group)
    second = GroupAnalyzer(
        plan_cache=database,
        focus_analysis_workers=2,
        focus_analysis_workers_requested=2,
        loader=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache hit decoded pixels")),
    ).analyze_group(group)

    assert first_calls
    assert first["selection_plan_cache_hit"] is False
    assert second["selection_plan_cache_hit"] is True
    assert second["selection_decode_count"] == 0
    assert second["analysis_workers_requested"] == 2
    assert second["analysis_workers_effective"] == 0
    for key in (
        "selected_indices", "quality_scores", "preview_transforms",
        "alignment_order_indices", "coverage",
    ):
        assert second[key] == first[key]

    config_calls = []

    def config_loader(item, max_long_edge=None):
        config_calls.append((item.path, max_long_edge))
        return load_rgb(item.path, max_long_edge)

    config_changed = GroupAnalyzer(
        plan_cache=database,
        loader=config_loader,
        registration_config=RegistrationConfig(analysis_long_edge=800),
    ).analyze_group(group)
    assert config_changed["selection_plan_cache_hit"] is False
    assert config_calls

    paths[0].write_bytes(paths[0].read_bytes())
    changed_calls = []

    def changed_loader(item, max_long_edge=None):
        changed_calls.append((item.path, max_long_edge))
        return load_rgb(item.path, max_long_edge)

    changed = GroupAnalyzer(plan_cache=database, loader=changed_loader).analyze_group(group)
    assert changed["selection_plan_cache_hit"] is False
    assert changed_calls
    database.close()

