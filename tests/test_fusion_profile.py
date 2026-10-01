from contextvars import copy_context
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from focus_stack_app.fusion.backends import FusionBackend, FusionResult
from focus_stack_app.fusion.exposure_gain import ExposureGain
from focus_stack_app.fusion.registration_diagnostics import RegistrationResiduals
from focus_stack_app.pipeline.analysis_worker import AnalysisJob
from focus_stack_app.pipeline.merge_worker import StackMergeService
from focus_stack_app.utils.performance import FusionProfile, stage


def _flat_rgb(value, shape=(80, 80)):
    return np.full((*shape, 3), value, dtype=np.uint8)


def _identity():
    return np.eye(3, dtype=np.float32)


def test_fusion_profile_persists_threaded_prefetch_events(tmp_path):
    def prefetch(worker):
        with stage("prefetch", worker=worker):
            return worker * 2

    with FusionProfile(tmp_path, "quality"):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for worker in range(2):
                context = copy_context()
                futures.append(executor.submit(context.run, prefetch, worker))
            assert [future.result() for future in futures] == [0, 2]

    report = json.loads((tmp_path / "fusion_profile.json").read_text(encoding="utf-8"))
    assert report["outcome"] == "ok"
    events = [event for event in report["events"] if event["stage"] == "prefetch"]
    assert {event["worker"] for event in events} == {0, 1}
    assert all(event["seconds"] >= 0 for event in events)


def test_fusion_profile_persists_error_event_and_outcome(tmp_path):
    with pytest.raises(RuntimeError, match="profile failure"):
        with FusionProfile(tmp_path, "quality"):
            with stage("fusion"):
                raise RuntimeError("profile failure")

    report = json.loads((tmp_path / "fusion_profile.json").read_text(encoding="utf-8"))
    assert report["outcome"] == "error"
    assert report["error"] == "profile failure"
    assert any(event["stage"] == "fusion" and event["outcome"] == "error"
               for event in report["events"])


def test_registration_residual_diagnostic_does_not_change_pixels():
    yy, xx = np.indices((64, 64))
    checker = ((xx + yy) % 2 * 255).astype(np.uint8)
    reference = np.dstack([checker, np.roll(checker, 1, axis=0), checker])
    aligned = reference.copy()
    reference_before = reference.copy()
    aligned_before = aligned.copy()

    residuals = RegistrationResiduals(reference, roi_size=32, maximum_rois=2)
    residuals.measure(aligned, "frame-1", np.eye(3), reference.shape[:2])

    assert np.array_equal(reference, reference_before)
    assert np.array_equal(aligned, aligned_before)


def test_exposure_gain_records_and_reuses_one_scalar_gain():
    reference = _flat_rgb(110)
    source = _flat_rgb(100)
    matrix = _identity()
    model = ExposureGain(reference, matrix, reference.shape[:2], 0)
    calls = []
    original_estimate = model.estimate

    def tracked_estimate(index, value, transform):
        calls.append(index)
        return original_estimate(index, value, transform)

    model.estimate = tracked_estimate
    model.apply(1, source, matrix)
    gain = model.gains[1]
    second_source = _flat_rgb(80)
    corrected = model.apply(1, second_source, matrix)

    assert gain == pytest.approx(1.1)
    assert calls == [1]
    expected = np.rint(second_source.astype(np.float32) * gain).clip(0, 255).astype(np.uint8)
    np.testing.assert_array_equal(corrected, expected)


def test_exposure_gain_excludes_boundary_reflection_from_estimate():
    reference = _flat_rgb(110)
    clean = _flat_rgb(100)
    reflected = clean.copy()
    reflected[:4, :] = 255
    reflected[-4:, :] = 255
    reflected[:, :4] = 255
    reflected[:, -4:] = 255
    matrix = _identity()

    clean_model = ExposureGain(reference, matrix, reference.shape[:2], 0)
    reflected_model = ExposureGain(reference, matrix, reference.shape[:2], 0)
    clean_gain = clean_model.estimate(1, clean, matrix)
    reflected_gain = reflected_model.estimate(1, reflected, matrix)

    assert reflected_gain == pytest.approx(clean_gain)
    assert reflected_gain == pytest.approx(1.1)


@pytest.mark.parametrize(
    ("source_value", "reference_value", "expected"),
    [(20, 220, 1.1), (220, 20, 0.9)],
)
def test_exposure_gain_clamps_extreme_ratios(source_value, reference_value, expected):
    reference = _flat_rgb(reference_value)
    source = _flat_rgb(source_value)
    model = ExposureGain(reference, _identity(), reference.shape[:2], 0)

    gain = model.estimate(1, source, _identity())

    assert gain == pytest.approx(expected)


def test_exposure_gain_headroom_preserves_saturation_without_new_255():
    source = _flat_rgb(220)
    reference = _flat_rgb(242)
    source[20, 20] = 240
    reference[20, 20] = 249
    source[30, 30] = 255
    reference[30, 30] = 255
    matrix = _identity()
    model = ExposureGain(reference, matrix, reference.shape[:2], 0)

    gain = model.estimate(1, source, matrix)
    corrected = model.apply(1, source, matrix)

    assert gain == pytest.approx(254.49 / 240.0)
    assert corrected[20, 20, 0] == 254
    assert corrected[30, 30, 0] == 255
    assert int(corrected[source < 255].max()) <= 254


def test_exposure_gain_uses_one_rgb_scalar_with_rounding_error_only():
    source = _flat_rgb([101, 137, 199])
    reference = np.rint(source.astype(np.float32) * 1.05).clip(0, 255).astype(np.uint8)
    matrix = _identity()
    model = ExposureGain(reference, matrix, reference.shape[:2], 0)

    gain = model.estimate(1, source, matrix)
    corrected = model.apply(1, source, matrix)
    expected = np.rint(source.astype(np.float32) * gain).clip(0, 255).astype(np.uint8)

    np.testing.assert_array_equal(corrected, expected)
    assert np.max(np.abs(corrected.astype(np.float32) - source.astype(np.float32) * gain)) <= 0.5


def test_exposure_gain_known_normal_subject_is_exact_rgb():
    source = _flat_rgb([31, 127, 231])
    matrix = _identity()
    model = ExposureGain(source, matrix, source.shape[:2], 0)

    corrected = model.apply(1, source, matrix)

    assert model.gains[1] == pytest.approx(1.0)
    assert corrected is source
    np.testing.assert_array_equal(corrected, source)


class _ProfileWritingBackend(FusionBackend):
    name = "profile-test"

    def __init__(self, *, keep=False):
        self.keep = keep

    def fuse(self, group, analysis, output_path, work_dir, output_config, cancel_event):
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "fusion_profile.json").write_text(
            '{"schema_version": 1, "backend": "profile-test", "outcome": "ok"}',
            encoding="utf-8",
        )
        (work_dir / "aligned.tif").write_bytes(b"temporary aligned tiff")
        if self.keep:
            (work_dir / ".keep").touch()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fused output")
        return FusionResult(output_path, self.name, (work_dir / "aligned.tif",))


def _service_cleanup_job(tmp_path, *, keep=False):
    source_dir = tmp_path / "photo"
    output_dir = tmp_path / "output"
    cache_dir = tmp_path / "cache"
    source_dir.mkdir()
    sources = [source_dir / f"IMG{index}.jpg" for index in range(4)]
    for source in sources:
        source.write_bytes(b"source")
    group = {"id": 17, "images": [{"path": source} for source in sources]}
    analysis = {
        "all_images": group["images"],
        "selected_paths": sources,
        "first_original_image": group["images"][0],
        "needs_merge": True,
    }
    service = StackMergeService(
        output_dir,
        archive_enabled=False,
        cache_dir=cache_dir,
        fusion_backend=_ProfileWritingBackend(keep=keep),
    )
    result = service.process(AnalysisJob(group=group, analysis=analysis))
    return result, cache_dir


def test_service_cleanup_persists_profile_and_removes_tiffs(tmp_path):
    result, cache_dir = _service_cleanup_job(tmp_path)

    assert result.status == "DONE"
    assert result.work_dir is not None and not result.work_dir.exists()
    profiles = list((cache_dir / "logs" / "fusion_profiles").glob("*.json"))
    assert len(profiles) == 1
    assert json.loads(profiles[0].read_text(encoding="utf-8"))["outcome"] == "ok"
    assert not list(cache_dir.rglob("*.tif"))


def test_service_cleanup_preserves_keep_work_directory(tmp_path):
    result, cache_dir = _service_cleanup_job(tmp_path, keep=True)

    assert result.status == "DONE"
    assert result.work_dir is not None and result.work_dir.is_dir()
    assert (result.work_dir / ".keep").is_file()
    assert (result.work_dir / "aligned.tif").is_file()
    profiles = list((cache_dir / "logs" / "fusion_profiles").glob("*.json"))
    assert len(profiles) == 1
