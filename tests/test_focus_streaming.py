from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from focus_stack_app.fusion.backends import HuginEnfuseBackend
from focus_stack_app.fusion.focus_masks import build_focus_labels
from focus_stack_app.hugin.align import AlignImageStack
from focus_stack_app.hugin.output_encoder import OutputConfig
from focus_stack_app.hugin.process import CommandResult


def _focus_frames():
    rng = np.random.default_rng(23)
    frames = []
    for base, x in zip((80, 120, 160), (8, 40, 72)):
        frame = np.full((96, 128, 3), base, dtype=np.uint8)
        frame[32:64, x:x + 32] = rng.integers(20, 240, (32, 32, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


def test_focus_labels_streaming_prefetch_and_reorder_are_equivalent():
    frames = _focus_frames()
    baseline = build_focus_labels(
        len(frames), lambda index: frames[index], prefetch=False,
    )
    prefetched = build_focus_labels(
        len(frames), lambda index: frames[index], prefetch=True,
    )
    reordered = build_focus_labels(
        len(frames), lambda index: frames[index], frame_order=[2, 0, 1], prefetch=True,
    )

    np.testing.assert_array_equal(prefetched, baseline)
    np.testing.assert_array_equal(reordered, baseline)


def test_focus_frame_observer_receives_each_frame_without_retaining_fullmaps():
    frames = _focus_frames()
    before = [frame.copy() for frame in frames]
    observed = []

    def observer(index, rgb, gray, score):
        observed.append((index, rgb.shape, gray.shape, score.shape,
                         float(gray.mean()), float(score.mean())))

    labels = build_focus_labels(
        len(frames), lambda index: frames[index], prefetch=False,
        stabilize_background=False, frame_observer=observer,
    )

    assert labels.shape == frames[0].shape[:2]
    assert [row[0] for row in observed] == [0, 1, 2]
    assert all(row[1] == frames[0].shape for row in observed)
    assert all(row[2] == row[3] == frames[0].shape[:2] for row in observed)
    assert all(np.isscalar(value) for row in observed for value in row[4:])
    for actual, expected in zip(frames, before):
        np.testing.assert_array_equal(actual, expected)


def test_focus_label_cancel_stops_streaming_decode():
    frames = _focus_frames()
    cancel = threading.Event()
    decoded = []

    def loader(index):
        decoded.append(index)
        cancel.set()
        return frames[index]

    with pytest.raises(RuntimeError, match="cancelled"):
        build_focus_labels(len(frames), loader, cancel_event=cancel, prefetch=False,
                          stabilize_background=False)
    assert decoded == [0]


@pytest.mark.parametrize("keep", [False, True])
def test_hugin_retry_releases_first_tiffs_before_second_align_except_keep(tmp_path, keep):
    anchor = tmp_path / "anchor.jpg"
    other = tmp_path / "other.jpg"
    Image.new("RGB", (100, 100), "red").save(anchor)
    Image.new("RGB", (100, 100), "blue").save(other)
    work = tmp_path / "work"
    if keep:
        work.mkdir()
        (work / ".keep").touch()
    first_paths = []
    observed_before_second = []

    class Runner:
        def __init__(self):
            self.calls = 0

        def run(self, command, **kwargs):
            self.calls += 1
            current_work = Path(kwargs["cwd"])
            prefix = Path(command[command.index("-a") + 1])
            if self.calls == 2:
                observed_before_second.append({
                    "first_tiffs_exist": all(path.exists() for path in first_paths),
                    "sources_exist": anchor.is_file() and other.is_file(),
                })
            outputs = []
            for index in range(2):
                path = Path(f"{prefix}{index:04d}.tif")
                Image.new("RGB", (80, 80), "green").save(path)
                outputs.append(path)
            if self.calls == 1:
                first_paths.extend(outputs)
                return CommandResult(tuple(command), 1, stderr="not enough control points")
            return CommandResult(tuple(command), 0)

    class Enfuser:
        def fuse(self, paths, output_path, **kwargs):
            output = Path(output_path)
            Image.new("RGB", (80, 80), "green").save(output)
            return SimpleNamespace(output_path=output)

    runner = Runner()
    backend = HuginEnfuseBackend(AlignImageStack("align.exe", runner=runner), Enfuser())
    result = backend.fuse(
        {}, {"alignment_order": [other, anchor], "selected_paths": [anchor, other],
             "first_original_path": anchor},
        tmp_path / "result.jpg", work, OutputConfig(), threading.Event(),
    )

    assert result.alignment_level == 2
    assert observed_before_second == [{"first_tiffs_exist": keep, "sources_exist": True}]
    assert anchor.is_file() and other.is_file()
