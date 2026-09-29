from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import numpy as np
import pytest

from focus_stack_app.fusion.backends import HuginEnfuseBackend, OpenCVFusionBackend
from focus_stack_app.hugin.align import AlignImageStack
from focus_stack_app.hugin.align import AlignmentError
from focus_stack_app.hugin.process import CommandResult
from focus_stack_app.hugin.output_encoder import OutputConfig


@pytest.mark.parametrize('background', [25, 220])
@pytest.mark.parametrize('fault', ['shift', 'scale', 'duplicate', 'missing'])
def test_colour_alignment_rejects_bad_geometry(tmp_path, background, fault):
    import cv2
    pixels = np.full((480, 640, 3), background, np.uint8)
    cv2.fillPoly(pixels, [np.array([[130, 180], [490, 290], [480, 320], [120, 210]])], (190, 30, 30))
    if fault == 'shift':
        bad = cv2.warpAffine(pixels, np.float32([[1, 0, 12], [0, 1, 0]]), (640, 480), borderValue=(background,) * 3)
    elif fault == 'scale':
        bad = cv2.warpAffine(pixels, cv2.getRotationMatrix2D((320, 240), 0, 1.2), (640, 480), borderValue=(background,) * 3)
    elif fault == 'duplicate':
        bad = pixels.copy()
        bad[70:110, 130:490] = (190, 30, 30)
    else:
        bad = np.full_like(pixels, background)
    paths = [tmp_path / 'a.tif', tmp_path / 'b.tif']
    for path, frame in zip(paths, (pixels, bad)):
        Image.fromarray(frame).save(path)
    with pytest.raises(AlignmentError, match='主体对齐检查未通过'):
        HuginEnfuseBackend()._validate_alignment(SimpleNamespace(input_paths=paths, aligned_paths=paths), paths[0])


def test_colour_alignment_accepts_defocus_and_neutral_background_changes(tmp_path):
    import cv2
    pixels = np.full((480, 640, 3), 25, np.uint8)
    pixels[160:300, 140:500] = (190, 30, 30)
    blurred = cv2.GaussianBlur(pixels, (0, 0), 3)
    blurred[350:430, 30:600] = 160
    paths = [tmp_path / 'a.tif', tmp_path / 'b.tif']
    for path, frame in zip(paths, (pixels, blurred)):
        Image.fromarray(frame).save(path)
    assert HuginEnfuseBackend()._validate_alignment(SimpleNamespace(input_paths=paths, aligned_paths=paths), paths[0]) == (1.0, [])


def test_invalid_colour_alignment_never_reaches_fusion(tmp_path):
    import threading
    class Runner:
        def __init__(self):
            self.calls = 0
        def run(self, command, **kwargs):
            self.calls += 1
            prefix = command[command.index('-a') + 1]
            for index, x in enumerate((140, 180)):
                pixels = np.full((480, 640, 3), 25, np.uint8)
                pixels[120:360, x:x + 45] = (190, 30, 30)
                Image.fromarray(pixels).save(f'{prefix}{index:04d}.tif')
            return CommandResult(tuple(command), 0)
    class NoFusion:
        def fuse(self, *args, **kwargs):
            pytest.fail('Misaligned images must never reach fusion')
    paths = [tmp_path / 'a.jpg', tmp_path / 'b.jpg']
    for path in paths:
        Image.new('RGB', (640, 480), 'black').save(path)
    runner = Runner()
    backend = HuginEnfuseBackend(AlignImageStack('align.exe', runner=runner), NoFusion())
    output = tmp_path / 'output.jpg'
    with pytest.raises(AlignmentError, match='主体对齐检查未通过'):
        backend.fuse({}, {'selected_paths': paths}, output, tmp_path / 'work', OutputConfig(), threading.Event())
    assert runner.calls == 4
    assert not output.exists()
    # Failed validation used to retain four full TIFF sets. The latest
    # complete attempt remains available for diagnosis, plus the command log.
    for level in (1, 2, 3):
        assert not list((tmp_path / 'work' / f'alignment_level_{level}').glob('*.tif'))
    assert len(list((tmp_path / 'work/alignment_level_4').glob('*.tif'))) == 2


def test_alignment_validation_rejects_subject_ghost_despite_matching_dimensions(tmp_path):
    paths = [tmp_path / 'first.tif', tmp_path / 'shifted.tif']
    for path, x in zip(paths, (300, 330)):
        pixels = np.full((480, 640, 3), 220, np.uint8)
        pixels[120:360, x:x + 30] = (170, 30, 30)
        Image.fromarray(pixels).save(path)
    alignment = SimpleNamespace(input_paths=paths, aligned_paths=paths)
    backend = HuginEnfuseBackend()
    with pytest.raises(AlignmentError, match='主体对齐检查未通过'):
        backend._validate_alignment(alignment, paths[0])
    alignment.aligned_paths = [paths[0], paths[0]]
    assert backend._validate_alignment(alignment, paths[0]) == (1.0, [])


def test_hugin_backend_retries_in_isolated_dirs_and_preserves_order(tmp_path):
    anchor = tmp_path / "anchor.jpg"
    other = tmp_path / "other.jpg"
    Image.new("RGB", (100, 100), "red").save(anchor)
    Image.new("RGB", (100, 100), "blue").save(other)

    class Runner:
        def __init__(self):
            self.calls = []

        def run(self, command, **kwargs):
            self.calls.append((list(command), Path(kwargs["cwd"])))
            if len(self.calls) == 1:
                return CommandResult(tuple(command), 1, stderr="not enough control points")
            prefix = Path(command[command.index("-a") + 1])
            for index in range(2):
                Image.new("RGB", (80, 80), "green").save(Path(f"{prefix}{index:04d}.tif"))
            return CommandResult(tuple(command), 0)

    class Enfuser:
        def fuse(self, paths, output_path, **kwargs):
            Image.new("RGB", (80, 80), "green").save(output_path)
            return SimpleNamespace(output_path=Path(output_path))

    runner = Runner()
    backend = HuginEnfuseBackend(AlignImageStack("align.exe", runner=runner), Enfuser())
    result = backend.fuse(
        {},
        {"alignment_order": [str(other), str(anchor)], "selected_paths": [str(anchor), str(other)],
         "first_original_path": str(anchor)},
        tmp_path / "result.jpg", tmp_path / "work", OutputConfig(),
        __import__("threading").Event(),
    )

    assert result.alignment_level == 2
    assert result.fallback_used is False  # no backend fallback occurred
    assert runner.calls[0][1] != runner.calls[1][1]
    assert "--corr=0.8" not in runner.calls[0][0]
    assert "--corr=0.8" in runner.calls[1][0]
    assert "--use-given-order" in runner.calls[1][0]
    assert "--align-to-first" not in runner.calls[1][0]
    assert runner.calls[1][0][-2:] == [str(other), str(anchor)]
    assert any(item.startswith("CROP_RATIO_WARNING") for item in result.diagnostics)


def test_experimental_opencv_backend_uses_generic_analysis_data(tmp_path):
    import cv2
    import numpy as np
    import threading

    truth = np.full((120, 180, 3), 230, np.uint8)
    cv2.putText(truth, "STACK", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 40, 180), 2)
    blurred = cv2.GaussianBlur(truth, (0, 0), 3)
    near, far = truth.copy(), truth.copy()
    near[:, 90:] = blurred[:, 90:]
    far[:, :90] = blurred[:, :90]
    paths = [tmp_path / "near.jpg", tmp_path / "far.jpg"]
    for path, pixels in zip(paths, (near, far)):
        Image.fromarray(pixels).save(path, quality=100)
    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    result = OpenCVFusionBackend().fuse(
        {},
        {"selected_paths": list(map(str, paths)), "selected_indices": [0, 1],
         "preview_reference": str(paths[0]), "preview_reference_index": 0,
         "preview_transforms": [identity, identity], "analysis_shapes": [[120, 180], [120, 180]],
         "reference_analysis_shape": [120, 180]},
        tmp_path / "stack.jpg", tmp_path / "work-opencv", OutputConfig(), threading.Event(),
    )
    assert result.actual_backend == "opencv"
    assert result.output_path.is_file()


def test_opencv_recovered_metadata_preserves_full_frame(tmp_path):
    """Large recovered inputs must not magnify the upper-left background."""
    import threading
    import numpy as np

    pixels = np.full((900, 1200, 3), 225, np.uint8)
    pixels[250:750, 400:900] = [190, 35, 25]
    pixels[350:650:20, 450:850] = [15, 20, 25]
    paths = [tmp_path / "a.png", tmp_path / "b.png"]
    for path in paths:
        Image.fromarray(pixels).save(path)
    result = OpenCVFusionBackend().fuse(
        {}, {"selected_paths": list(map(str, paths)), "selected_indices": [1, 3],
             "preview_reference": str(paths[1])},
        tmp_path / "recovered.jpg", tmp_path / "work", OutputConfig(), threading.Event(),
    )
    actual = np.asarray(Image.open(result.output_path)).astype(float)
    assert actual.shape == pixels.shape
    assert np.abs(actual - pixels).mean() < 2
    assert "PREVIEW_REGISTRATION_REBUILT" in result.diagnostics


def test_opencv_known_preview_transform_preserves_full_frame(tmp_path):
    import threading
    import cv2
    import numpy as np

    pixels = np.full((900, 1200, 3), 225, np.uint8)
    pixels[250:750, 400:900] = [190, 35, 25]
    shifted = cv2.warpAffine(pixels, np.float32([[1, 0, 30], [0, 1, 0]]),
                             (1200, 900), borderMode=cv2.BORDER_REFLECT_101)
    paths = [tmp_path / "a.png", tmp_path / "b.png"]
    for path, value in zip(paths, (pixels, shifted)):
        Image.fromarray(value).save(path)
    result = OpenCVFusionBackend().fuse(
        {}, {"selected_paths": list(map(str, paths)), "selected_indices": [0, 1],
             "preview_reference": str(paths[0]), "reference_analysis_shape": [300, 400],
             "analysis_shapes": [[300, 400], [300, 400]],
             "preview_transforms": [np.eye(3).tolist(), [[1, 0, -10], [0, 1, 0], [0, 0, 1]]]},
        tmp_path / "aligned.jpg", tmp_path / "work", OutputConfig(), threading.Event(),
    )
    actual = np.asarray(Image.open(result.output_path)).astype(float)
    assert actual.shape == pixels.shape
    assert np.abs(actual - pixels).mean() < 2
    assert "PREVIEW_REGISTRATION_REBUILT" not in result.diagnostics


def test_superseded_tiff_release_respects_keep_marker_and_outside_paths(tmp_path):
    work = tmp_path / 'work'
    attempt = work / 'alignment_level_1'
    attempt.mkdir(parents=True)
    inside = attempt / 'aligned.tif'
    outside = tmp_path / 'original.tif'
    for path in (inside, outside):
        Image.new('RGB', (16, 16)).save(path)
    alignment = SimpleNamespace(work_dir=attempt, aligned_paths=[inside, outside])
    backend = HuginEnfuseBackend()
    marker = work / '.keep'
    marker.touch()
    backend._release_superseded_tiffs(alignment, work)
    assert inside.exists() and outside.exists()
    marker.unlink()
    backend._release_superseded_tiffs(alignment, work)
    assert not inside.exists() and outside.exists()


def test_validation_rgb_cache_is_released_as_focus_pass_consumes_frames(tmp_path):
    from focus_stack_app.fusion.aligned_cache import AlignedTIFFImageCache
    cache = AlignedTIFFImageCache(max_bytes=1024 ** 2, snapshot_fn=lambda: SimpleNamespace(
        total_bytes=32 * 1024 ** 3, available_bytes=16 * 1024 ** 3))
    path = tmp_path / 'frame.tif'
    Image.new('RGB', (32, 24), (180, 35, 35)).save(path)
    cache.validation_previews(path)
    assert cache.bytes_used == 32 * 24 * 3
    first = cache.load(path)
    assert cache.bytes_used == 0 and not cache.frames
    np.testing.assert_array_equal(cache.load(path), first)


@pytest.mark.parametrize('shift', [0, 12])
def test_alignment_distinguishes_defocus_outline_changes_from_real_translation(tmp_path, shift):
    import cv2
    from focus_stack_app.fusion.alignment_quality import colour_subject_mask, colour_subject_mismatch
    sharp = np.full((480, 640, 3), 220, np.uint8)
    sharp[60:420, 250:330] = (150, 70, 70)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 8)
    if shift:
        blurred = cv2.warpAffine(blurred, np.float32([[1, 0, shift], [0, 1, 0]]),
                                 (640, 480), borderValue=(220, 220, 220))
    assert colour_subject_mismatch(colour_subject_mask(sharp), colour_subject_mask(blurred))
    paths = [tmp_path / 'sharp.tif', tmp_path / 'blurred.tif']
    for path, pixels in zip(paths, (sharp, blurred)):
        Image.fromarray(pixels).save(path)
    alignment = SimpleNamespace(input_paths=paths, aligned_paths=paths)
    backend = HuginEnfuseBackend()
    if shift:
        with pytest.raises(AlignmentError, match='主体对齐检查未通过'):
            backend._validate_alignment(alignment, paths[0])
    else:
        crop, notes = backend._validate_alignment(alignment, paths[0])
        assert crop == 1.0
        assert 'DEFOCUS_GEOMETRY_CONFIRMED:1' in notes


def test_retry_uses_recorded_capture_order_before_lens_distortion(tmp_path):
    import threading
    paths = [tmp_path / name for name in ('DSC9999.jpg', 'DSC0001.jpg', 'DSC0003.jpg')]
    for path in paths:
        Image.new('RGB', (100, 100), 'white').save(path)
    commands = []
    class Runner:
        def run(self, command, **kwargs):
            commands.append(command)
            if len(commands) == 1:
                return CommandResult(tuple(command), 1, stderr='not enough control points')
            prefix = command[command.index('-a') + 1]
            for index in range(3):
                Image.new('RGB', (100, 100), 'white').save(f'{prefix}{index:04d}.tif')
            return CommandResult(tuple(command), 0)
    class Fusion:
        def fuse(self, paths, output_path, **kwargs):
            Image.new('RGB', (100, 100), 'white').save(output_path)
            return SimpleNamespace(output_path=Path(output_path))
    backend = HuginEnfuseBackend(AlignImageStack('align.exe', runner=Runner()), Fusion())
    tour = [paths[0], paths[2], paths[1]]
    result = backend.fuse({}, {'selected_paths': paths, 'alignment_order': tour,
                             'capture_order': paths}, tmp_path / 'result.jpg',
                          tmp_path / 'work', OutputConfig(), threading.Event())
    assert commands[0][-3:] == list(map(str, tour))
    assert commands[1][-3:] == list(map(str, paths))
    assert '--corr=0.8' not in commands[1] and '-d' not in commands[1]
    assert result.actual_hugin_input_order == tuple(paths)
    assert 'ALIGNMENT_CAPTURE_ORDER_RETRY' in result.diagnostics


@pytest.mark.parametrize('count, capture_kind', [(20, 'complete'), (21, 'complete'),
                                               (55, 'complete'), (21, 'missing'),
                                               (21, 'incomplete'), (21, 'duplicate')])
def test_large_stacks_prefer_complete_capture_order(tmp_path, count, capture_kind):
    import threading
    # Numbering wraps; recorded chronology, not a filename sort, is authoritative.
    paths = [tmp_path / f'DSC{(9999 + index) % 10000:04d}.jpg' for index in range(count)]
    for path in paths:
        Image.new('RGB', (32, 32), 'white').save(path)
    tour = paths[::2] + paths[1::2]
    capture = [tmp_path / 'unselected.jpg', *paths]
    if capture_kind == 'missing':
        capture = []
    elif capture_kind == 'incomplete':
        capture = paths[:-1]
    elif capture_kind == 'duplicate':
        capture = paths[:-1] + [paths[0]]
    calls = []
    class Aligner:
        def align(self, inputs, **kwargs):
            calls.append(list(inputs))
            return SimpleNamespace(input_paths=tuple(inputs), aligned_paths=tuple(inputs))
    class Fusion:
        def fuse(self, inputs, output_path, **kwargs):
            Image.new('RGB', (32, 32), 'white').save(output_path)
            return SimpleNamespace(output_path=Path(output_path))
    result = HuginEnfuseBackend(Aligner(), Fusion()).fuse(
        {}, {'selected_paths': paths, 'alignment_order': tour, 'capture_order': capture},
        tmp_path / 'result.jpg', tmp_path / 'work', OutputConfig(), threading.Event())
    primary = count > 20 and capture_kind == 'complete'
    expected = paths if primary else tour
    assert calls == [expected]
    assert result.actual_hugin_input_order == tuple(expected)
    assert ('ALIGNMENT_CAPTURE_ORDER_PRIMARY' in result.diagnostics) == primary

