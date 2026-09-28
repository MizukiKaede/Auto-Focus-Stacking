from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from focus_stack_app.core.group_detector import StackGroupDetector, StackGroupDetectorConfig
from focus_stack_app.core.manual_groups import group_with_overrides


def detect(pixels, seconds, config=None):
    start = datetime(2026, 1, 1)
    records = [SimpleNamespace(path=Path(f"arbitrary-{i}.jpg"),
                              capture_time=(start + timedelta(seconds=t)).isoformat())
               for i, t in enumerate(seconds)]
    images = {record.path: image for record, image in zip(records, pixels)}
    return list(StackGroupDetector(config=config, loader=lambda path, edge: images[path]).iter_groups(records))


def white_subject(*, shifted=0, marked=True):
    scene = np.full((480, 640, 3), 220, np.uint8)
    scene[0:320, 550:] = 35  # A border-connected backdrop is not the subject.
    scene[125:360, 300 + shifted:326 + shifted] = (170, 30, 30)
    scene[245:360, 300 + shifted:326 + shifted] = 35
    if marked:
        for y in range(145, 220, 12):
            scene[y:y + 5, 307 + shifted:319 + shifted] = 235
    return scene


def test_small_subject_turn_splits_despite_unchanged_background_and_short_gap():
    front, back = white_subject(), white_subject(shifted=20, marked=False)
    assert [len(g.items) for g in detect([front, front, back, back], [0, 1, 15, 16])] == [2, 2]


def test_reflective_tip_disconnect_and_focus_blur_do_not_split():
    scene = white_subject()
    scene[90:125, 305:321] = 146
    other = scene.copy()
    other[90:125, 305:321] = 143
    soft = cv2.GaussianBlur(scene, (0, 0), 1.2)
    assert [len(g.items) for g in detect([scene, other, soft], [0, 1, 2])] == [3]


def test_configurable_pause_changes_boundary_and_keeps_default_20():
    scene = np.full((120, 160, 3), 100, np.uint8)
    changed = np.full_like(scene, 103)  # Change between pause and composition cutoffs.
    assert StackGroupDetectorConfig().pause_seconds == 20
    assert [len(g.items) for g in detect([scene, changed], [0, 15])] == [2]
    config = StackGroupDetectorConfig(pause_seconds=10)
    assert [len(g.items) for g in detect([scene, changed], [0, 15], config)] == [1, 1]
    config = StackGroupDetectorConfig(pause_seconds=120)
    assert [len(g.items) for g in detect([scene, scene], [0, 90], config)] == [2]


def test_focus_changes_do_not_split_one_stack():
    scene = np.full((120, 160, 3), (125, 85, 55), np.uint8)
    cv2.putText(scene, "TOOLS", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (230, 20, 25), 2)
    blurred = cv2.GaussianBlur(scene, (0, 0), 1)
    assert [len(group.items) for group in detect([scene, blurred, scene], [0, 1, 34])] == [3]


def test_composition_and_long_pause_split_groups():
    scene = np.full((120, 160, 3), 100, np.uint8)
    changed = scene.copy()
    changed[:80, :100] = 180
    assert [len(group.items) for group in detect([scene, scene, changed], [0, 1, 2])] == [2, 1]
    assert [len(group.items) for group in detect([scene] * 4, [0, 1, 90, 91])] == [2, 2]


def test_manual_groups_override_automatic_detection(tmp_path):
    records = [SimpleNamespace(path=tmp_path / name) for name in ("one.JPG", "two.JPG", "three.JPG")]
    (tmp_path / "stack_groups.json").write_text('{"groups": [["one.JPG", "three.JPG"]]}', encoding="utf-8")
    automatic_calls = []

    def automatic(values):
        automatic_calls.append(list(values))
        return [SimpleNamespace(items=list(values), group_id=0, start_index=0, end_index=0)] if values else []

    groups = group_with_overrides(records, tmp_path / "stack_groups.json", automatic)
    assert automatic_calls == [[records[1]]]
    assert any(group.items == [records[0], records[2]] for group in groups)

